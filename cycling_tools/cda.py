"""CdA estimation from power-meter rides (dynamic power balance, robust fit, diagnostics).

Pure functions and dataclasses; no streamlit.

Model (per sample)::

    P*eff = Crr*m*g*cos(th)*v + m*g*sin(th)*v + 0.5*rho*CdA*v_air*|v_air|*v + (m+I)*v*a
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter

from .fit_io import Ride
from .geo import angle_diff_deg, bearing_deg, smooth_heading_deg
from .physics import G, WHEEL_INERTIA_KG, air_density, standard_pressure_pa
from .weather import Weather, WeatherError, constant_weather, fetch_weather

REASONS = [
    "gap",
    "low_speed",
    "coasting",
    "braking",
    "cornering",
    "steep_grade",
    "high_accel",
    "low_airspeed",
]
REASON_LABELS = {
    "gap": "Gaps / missing data",
    "low_speed": "Low speed / stopped",
    "coasting": "Coasting / low power",
    "braking": "Braking",
    "cornering": "Cornering",
    "steep_grade": "Steep grade",
    "high_accel": "High acceleration",
    "low_airspeed": "Low airspeed",
}


@dataclass
class MaskConfig:
    min_speed_ms: float = 4.0
    min_power_w: float = 30.0
    min_cadence_rpm: float = 10.0
    braking_aero_w: float = 25.0        # implied aero power below -this => braking
    braking_margin_ms2: float = 0.15    # decel beyond max physical coast decel by this => braking
    braking_cda_hi: float = 0.6         # CdA used to compute the max physical coasting decel
    max_heading_rate_dps: float = 4.0
    max_lateral_accel: float = 1.2
    max_grade: float = 0.08
    max_accel_ms2: float = 0.5
    min_airspeed_ms: float = 3.0
    dilate_s: dict = field(
        default_factory=lambda: {
            "gap": 3.0, "low_speed": 3.0, "coasting": 5.0, "braking": 5.0,
            "cornering": 5.0, "steep_grade": 3.0, "high_accel": 2.0, "low_airspeed": 2.0,
        }
    )


@dataclass
class CdAConfig:
    mass_kg: float = 100.0
    crr: float = 0.0031
    fit_crr: bool = False
    drivetrain_eff: float = 0.97
    wheel_inertia_kg: float = WHEEL_INERTIA_KG
    # wind: "weather" (needs Weather, scaled by wind_scale), "manual", "none"
    wind_mode: str = "weather"
    wind_scale: float = 0.7
    manual_wind_ms: float = 0.0
    manual_wind_from_deg: float = 0.0
    rho_override: float | None = None
    speed_smooth_s: int = 5
    power_smooth_s: int = 5
    alt_smooth_s: int = 21
    heading_span_s: int = 4
    heading_smooth_s: int = 5
    rolling_window_s: int = 120
    rolling_min_frac: float = 0.25
    huber_k: float = 1.345
    bootstrap_n: int = 200
    block_s: int = 30
    seed: int = 0
    min_lap_valid_s: int = 60
    masks: MaskConfig = field(default_factory=MaskConfig)


@dataclass
class CdAResult:
    cda: float
    cda_ci: tuple[float, float]
    crr: float
    cda_fixed_crr: float
    joint: dict | None
    n_valid: int
    n_selected: int
    resid_rms_w: float
    series: pd.DataFrame
    mask_report: pd.DataFrame
    laps: pd.DataFrame
    wind_split: pd.DataFrame
    speed_split: pd.DataFrame
    energy: dict
    energy_selection: dict
    cfg: CdAConfig
    meta: dict
    warnings: list[str] = field(default_factory=list)
    wind_scan: pd.DataFrame | None = None

    @property
    def valid_pct(self) -> float:
        return 100.0 * self.n_valid / max(self.n_selected, 1)


# ----------------------------------------------------------------------------- helpers

def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.asarray(a, dtype=float)
    return pd.Series(a, dtype=float).rolling(int(win), center=True, min_periods=1).mean().to_numpy()


def _dilate(mask: np.ndarray, seconds: float) -> np.ndarray:
    n = int(round(seconds))
    if n <= 0 or not mask.any():
        return mask
    return binary_dilation(mask, structure=np.ones(2 * n + 1, dtype=bool))


def _fill(a: np.ndarray) -> np.ndarray:
    s = pd.Series(a, dtype=float)
    return s.interpolate(limit_area="inside").ffill().bfill().to_numpy()


def _huber(X: np.ndarray, y: np.ndarray, k: float = 1.345, iters: int = 40):
    """Huber IRLS regression. Returns (beta, weights, scale)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n < max(p, 3):
        return np.full(p, np.nan), np.ones(n), np.nan
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    w = np.ones(n)
    s = np.nan
    for _ in range(iters):
        r = y - X @ beta
        s = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-6)
        u = np.abs(r) / (k * s)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
        if p == 1:
            nb = np.array([np.sum(w * X[:, 0] * y) / np.sum(w * X[:, 0] ** 2)])
        else:
            sw = np.sqrt(w)
            nb = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.allclose(nb, beta, rtol=1e-7, atol=1e-10):
            beta = nb
            break
        beta = nb
    return beta, w, s


def try_fetch_weather(ride: Ride, laps: Sequence[int] | None = None) -> tuple[Weather | None, str | None]:
    """Fetch Open-Meteo weather for a ride; returns (weather, warning)."""
    df = ride.df
    if laps:
        sub = df[df["lap"].isin(list(laps))]
        df = sub if len(sub) else df
    g = df.dropna(subset=["lat", "lon"])
    if g.empty:
        return None, "No GPS in this ride, so weather cannot be looked up."
    try:
        w = fetch_weather(
            g["lat"].median(), g["lon"].median(),
            df["timestamp"].iloc[0] - pd.Timedelta(hours=1), df["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        )
        return w, None
    except (WeatherError, Exception) as exc:  # network etc.
        return None, f"Weather unavailable ({exc}); falling back to no wind and FIT temperature."


# ----------------------------------------------------------------------------- preparation

def _prepare(ride: Ride, cfg: CdAConfig, weather: Weather | None, laps) -> tuple[pd.DataFrame, list[str], dict]:
    df = ride.df
    warn: list[str] = []
    meta: dict = {}
    n = len(df)
    if df["power"].notna().sum() < 30:
        raise ValueError("This ride has no usable power data; CdA estimation needs a power meter.")
    if df["speed"].notna().sum() < 30:
        raise ValueError("This ride has no usable speed data.")

    ts = df["timestamp"]
    v_raw = df["speed"].to_numpy(float)
    v = _smooth(v_raw, cfg.speed_smooth_s)
    p = _smooth(df["power"].to_numpy(float), cfg.power_smooth_s)
    a = np.gradient(v)

    # gradient from altitude
    alt_raw = df["alt"].to_numpy(float)
    if np.isfinite(alt_raw).sum() > 30:
        alt = _fill(alt_raw)
        win = int(cfg.alt_smooth_s) | 1
        win = min(win, (len(alt) // 2) * 2 - 1) if len(alt) > 5 else 5
        climb = savgol_filter(alt, max(win, 5), 2, deriv=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_t = np.where(v > 1.0, climb / np.where(v > 1.0, v, 1.0), 0.0)
        sin_t = np.clip(np.nan_to_num(sin_t), -0.3, 0.3)
        has_alt = True
    else:
        alt = np.zeros(n)
        sin_t = np.zeros(n)
        has_alt = False
        warn.append("No altitude data: grade assumed zero (CdA will be biased on hilly rides).")
    cos_t = np.sqrt(1 - sin_t ** 2)
    grade = sin_t / cos_t

    # heading
    has_gps = df["lat"].notna().sum() > 30
    if has_gps:
        lat = _fill(df["lat"].to_numpy(float))
        lon = _fill(df["lon"].to_numpy(float))
        h = max(1, int(cfg.heading_span_s) // 2)
        i0 = np.clip(np.arange(n) - h, 0, n - 1)
        i1 = np.clip(np.arange(n) + h, 0, n - 1)
        hd = bearing_deg(lat[i0], lon[i0], lat[i1], lon[i1])
        hd = smooth_heading_deg(hd, cfg.heading_smooth_s)
        j0 = np.clip(np.arange(n) - 1, 0, n - 1)
        j1 = np.clip(np.arange(n) + 1, 0, n - 1)
        hrate = angle_diff_deg(hd[j1], hd[j0]) / np.maximum(j1 - j0, 1)
        lat_acc = np.abs(np.nan_to_num(v) * np.radians(hrate))
    else:
        hd = np.full(n, np.nan)
        hrate = np.zeros(n)
        lat_acc = np.zeros(n)
        warn.append("No GPS: heading unavailable, so wind is ignored and cornering is not masked.")

    # weather / wind
    mode = cfg.wind_mode
    wx = None
    wind_src = "none"
    if mode in ("weather", "auto"):
        if weather is None:
            warn.append("No weather supplied: wind ignored; air density from FIT temperature + ISA pressure.")
            mode = "none"
        elif not has_gps:
            mode = "none"
        else:
            wx = weather.at(ts)
            wind_src = weather.source
    if mode == "manual":
        if not has_gps:
            mode = "none"
        else:
            wind_src = "manual"
    hw_raw = np.zeros(n)
    if mode in ("weather", "auto") and wx is not None:
        spd = wx["wind_ms"].to_numpy(float)
        frm = wx["wind_dir"].to_numpy(float)
        hw_raw = spd * np.cos(np.radians(frm - hd))
        scale = cfg.wind_scale
    elif mode == "manual":
        hw_raw = cfg.manual_wind_ms * np.cos(np.radians(cfg.manual_wind_from_deg - hd))
        scale = 1.0
    else:
        scale = 0.0
    hw_raw = np.nan_to_num(hw_raw)
    hw = hw_raw * scale
    meta["wind_source"] = wind_src
    meta["wind_scale_used"] = scale

    # air density
    if cfg.rho_override is not None:
        rho = np.full(n, float(cfg.rho_override))
        meta["rho_source"] = "manual"
    elif weather is not None:
        w2 = wx if wx is not None else weather.at(ts)
        rho = air_density(w2["temp_c"].to_numpy(float), w2["pressure_pa"].to_numpy(float), w2["rh"].to_numpy(float))
        meta["rho_source"] = weather.source
    else:
        temp = df["temp"].to_numpy(float)
        temp = _fill(temp) if np.isfinite(temp).any() else np.full(n, 15.0)
        pres = standard_pressure_pa(alt if has_alt else 0.0) * np.ones(n)
        rho = air_density(temp, pres, 0.5)
        meta["rho_source"] = "FIT temperature + ISA pressure"
    rho = np.asarray(rho, float)
    meta["rho_mean"] = float(np.nanmean(rho))

    m = cfg.mass_kg
    m_eff = m + cfg.wheel_inertia_kg
    p_wheel = p * cfg.drivetrain_eff
    v_air = v + hw
    x = 0.5 * rho * v_air * np.abs(v_air) * v
    r_term = m * G * cos_t * v
    y0 = p_wheel - m * G * sin_t * v - m_eff * v * a
    y = y0 - cfg.crr * r_term
    with np.errstate(invalid="ignore", divide="ignore"):
        implied_aero = y
        implied_cda = np.where(np.abs(x) > 1.0, y / x, np.nan)

    sel = np.ones(n, bool) if not laps else df["lap"].isin(list(laps)).to_numpy()

    out = pd.DataFrame(
        {
            "t_s": df["t_s"].to_numpy(float), "timestamp": ts.to_numpy(), "lap": df["lap"].to_numpy(),
            "dist_km": df["dist"].to_numpy(float) / 1000.0,
            "lat": df["lat"].to_numpy(float), "lon": df["lon"].to_numpy(float),
            "speed": v, "power": p, "p_wheel": p_wheel, "cadence": df["cadence"].to_numpy(float),
            "alt": alt, "sin_t": sin_t, "grade": grade, "accel": a, "heading": hd,
            "heading_rate": hrate, "lat_accel": lat_acc,
            "hw_raw": hw_raw, "headwind": hw, "v_air": v_air, "rho": rho,
            "x": x, "r": r_term, "y0": y0, "y": y, "implied_aero": implied_aero, "implied_cda": implied_cda,
            "selected": sel,
        }
    )
    out["gap_flag"] = df["gap"].to_numpy(bool) if "gap" in df else False
    return out, warn, meta


def compute_masks(s: pd.DataFrame, cfg: CdAConfig) -> dict[str, np.ndarray]:
    mc = cfg.masks
    m_eff = cfg.mass_kg + cfg.wheel_inertia_kg
    v = s["speed"].to_numpy(float)
    raw: dict[str, np.ndarray] = {}
    fin = np.isfinite(v) & np.isfinite(s["power"].to_numpy(float)) & np.isfinite(s["x"].to_numpy(float)) \
        & np.isfinite(s["y"].to_numpy(float)) & np.isfinite(s["accel"].to_numpy(float))
    raw["gap"] = ~fin | s["gap_flag"].to_numpy(bool)
    with np.errstate(invalid="ignore"):
        raw["low_speed"] = ~(v >= mc.min_speed_ms)
        low_p = ~(s["power"].to_numpy(float) >= mc.min_power_w)
        cad = s["cadence"].to_numpy(float)
        low_c = (np.isfinite(cad) & (cad < mc.min_cadence_rpm)) if np.isfinite(cad).sum() > 30 else np.zeros(len(s), bool)
        raw["coasting"] = low_p | low_c
        # braking: implied aero strongly negative, or deceleration beyond physical coasting limit
        v_air = s["v_air"].to_numpy(float)
        f_aero_hi = 0.5 * s["rho"].to_numpy(float) * mc.braking_cda_hi * v_air * np.abs(v_air)
        f_other = cfg.crr * cfg.mass_kg * G * np.sqrt(1 - s["sin_t"].to_numpy(float) ** 2) \
            + cfg.mass_kg * G * s["sin_t"].to_numpy(float)
        a_min = -(f_aero_hi + f_other) / m_eff
        a = s["accel"].to_numpy(float)
        raw["braking"] = (s["implied_aero"].to_numpy(float) < -mc.braking_aero_w) | (a < a_min - mc.braking_margin_ms2)
        raw["cornering"] = (np.abs(s["heading_rate"].to_numpy(float)) > mc.max_heading_rate_dps) | (
            s["lat_accel"].to_numpy(float) > mc.max_lateral_accel)
        raw["steep_grade"] = np.abs(s["grade"].to_numpy(float)) > mc.max_grade
        raw["high_accel"] = np.abs(a) > mc.max_accel_ms2
        raw["low_airspeed"] = ~(np.abs(v_air) >= mc.min_airspeed_ms)
    return {k: _dilate(np.asarray(raw[k], bool), mc.dilate_s.get(k, 3.0)) for k in REASONS}


def _mask_report(masks: dict[str, np.ndarray], sel: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    n = max(int(sel.sum()), 1)
    taken = np.zeros(len(sel), bool)
    rows = []
    for k in REASONS:
        mk = masks[k] & sel
        uniq = mk & ~taken
        taken |= mk
        rows.append({"reason": k, "label": REASON_LABELS[k], "seconds": int(mk.sum()),
                     "pct": 100.0 * mk.sum() / n, "unique_seconds": int(uniq.sum())})
    rows.append({"reason": "valid", "label": "Valid for fit", "seconds": int((valid & sel).sum()),
                 "pct": 100.0 * (valid & sel).sum() / n, "unique_seconds": int((valid & sel).sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- fitting

def _solve(s: pd.DataFrame, idx: np.ndarray, cfg: CdAConfig, joint: bool = False):
    x = s["x"].to_numpy()[idx]
    if joint:
        X = np.column_stack([x, s["r"].to_numpy()[idx]])
        return _huber(X, s["y0"].to_numpy()[idx], cfg.huber_k)
    return _huber(x, s["y0"].to_numpy()[idx] - cfg.crr * s["r"].to_numpy()[idx], cfg.huber_k)


def _bootstrap(s: pd.DataFrame, fit_idx: np.ndarray, cfg: CdAConfig, joint: bool, crr_used: float):
    if cfg.bootstrap_n <= 0 or len(fit_idx) < 2 * cfg.block_s:
        return (np.nan, np.nan)
    rng = np.random.default_rng(cfg.seed)
    blocks = fit_idx // cfg.block_s
    ub = np.unique(blocks)
    groups = [fit_idx[blocks == b] for b in ub]
    x_all = s["x"].to_numpy()
    y_all = s["y0"].to_numpy() - crr_used * s["r"].to_numpy()
    r_all = s["r"].to_numpy()
    est = []
    for _ in range(cfg.bootstrap_n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if joint:
            beta, _, _ = _huber(np.column_stack([x_all[idx], r_all[idx]]), s["y0"].to_numpy()[idx], cfg.huber_k, iters=15)
        else:
            beta, _, _ = _huber(x_all[idx], y_all[idx], cfg.huber_k, iters=15)
        est.append(beta[0])
    est = np.asarray(est)
    return (float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5)))


def _fit_subset(s: pd.DataFrame, mask: np.ndarray, cfg: CdAConfig, crr: float) -> tuple[float, int]:
    idx = np.flatnonzero(mask)
    if len(idx) < 30:
        return np.nan, len(idx)
    beta, _, _ = _huber(s["x"].to_numpy()[idx], s["y0"].to_numpy()[idx] - crr * s["r"].to_numpy()[idx], cfg.huber_k)
    return float(beta[0]), len(idx)


def _energy(s: pd.DataFrame, mask: np.ndarray, cda: float, crr: float, cfg: CdAConfig) -> dict:
    mk = mask & np.isfinite(s["x"].to_numpy()) & np.isfinite(s["y0"].to_numpy())
    if not mk.any():
        return {}
    m_eff = cfg.mass_kg + cfg.wheel_inertia_kg
    v = s["speed"].to_numpy()[mk]
    aero = float(np.mean(cda * s["x"].to_numpy()[mk]))
    roll = float(np.mean(crr * s["r"].to_numpy()[mk]))
    climb = float(np.mean(cfg.mass_kg * G * s["sin_t"].to_numpy()[mk] * v))
    accel = float(np.mean(m_eff * v * s["accel"].to_numpy()[mk]))
    model = aero + roll + climb + accel
    meas = float(np.mean(s["p_wheel"].to_numpy()[mk]))
    return {"aero": aero, "rolling": roll, "climb": climb, "accel": accel, "modelled": model,
            "measured_wheel": meas, "unexplained": meas - model, "n": int(mk.sum())}


def _rolling_cda(s: pd.DataFrame, w: np.ndarray, valid: np.ndarray, crr: float, cfg: CdAConfig) -> np.ndarray:
    x = np.where(valid, s["x"].to_numpy(), 0.0)
    y = np.where(valid, s["y0"].to_numpy() - crr * s["r"].to_numpy(), 0.0)
    ww = np.where(valid, w, 0.0)
    win = int(cfg.rolling_window_s)
    roll = lambda a: pd.Series(a).rolling(win, center=True, min_periods=1).sum().to_numpy()
    num, den, cnt = roll(ww * x * y), roll(ww * x * x), roll(valid.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt >= cfg.rolling_min_frac * win, num / den, np.nan)
    return out


def wind_scale_scan(result: CdAResult, scales: Sequence[float] | None = None) -> pd.DataFrame:
    """Scan wind scale; report CdA, fit loss and head-vs-tail CdA difference (should be ~0 at the right scale)."""
    cfg = result.cfg
    s = result.series
    scales = np.linspace(0.0, 1.5, 16) if scales is None else np.asarray(scales, float)
    mask = (s["valid"] & s["selected"]).to_numpy()
    v = s["speed"].to_numpy()
    crr = result.crr
    rows = []
    for sc in scales:
        v_air = v + sc * s["hw_raw"].to_numpy()
        x = 0.5 * s["rho"].to_numpy() * v_air * np.abs(v_air) * v
        y = s["y0"].to_numpy() - crr * s["r"].to_numpy()
        idx = np.flatnonzero(mask & np.isfinite(x))
        beta, w, sc_res = _huber(x[idx], y[idx], cfg.huber_k)
        r = y[idx] - beta[0] * x[idx]
        loss = float(np.sum(np.where(np.abs(r) <= cfg.huber_k * sc_res, 0.5 * r ** 2,
                                     cfg.huber_k * sc_res * (np.abs(r) - 0.5 * cfg.huber_k * sc_res))) / len(idx))
        hw = sc * s["hw_raw"].to_numpy()[idx]
        out = {"wind_scale": float(sc), "cda": float(beta[0]), "rms_w": float(np.sqrt(np.mean(r ** 2))), "loss": loss}
        for nm, sub in (("cda_head", hw > 0.5), ("cda_tail", hw < -0.5)):
            if sub.sum() >= 30:
                b, _, _ = _huber(x[idx][sub], y[idx][sub], cfg.huber_k)
                out[nm] = float(b[0])
            else:
                out[nm] = np.nan
        out["head_minus_tail"] = out["cda_head"] - out["cda_tail"]
        rows.append(out)
    return pd.DataFrame(rows)


def best_wind_scale(scan: pd.DataFrame) -> dict:
    """Best scale by minimum loss and by head/tail balance (zero crossing)."""
    res = {"by_loss": float(scan.loc[scan["loss"].idxmin(), "wind_scale"]), "by_balance": np.nan}
    d = scan.dropna(subset=["head_minus_tail"])
    if len(d) >= 2:
        sc, df_ = d["wind_scale"].to_numpy(), d["head_minus_tail"].to_numpy()
        for i in range(len(sc) - 1):
            if df_[i] == 0 or df_[i] * df_[i + 1] < 0:
                res["by_balance"] = float(sc[i] - df_[i] * (sc[i + 1] - sc[i]) / (df_[i + 1] - df_[i]))
                break
    return res


# ----------------------------------------------------------------------------- top level

def analyse_ride(ride: Ride, cfg: CdAConfig | None = None, weather: Weather | None = None,
                 laps: Sequence[int] | None = None, scan_wind: bool = False) -> CdAResult:
    cfg = cfg or CdAConfig()
    s, warn, meta = _prepare(ride, cfg, weather, laps)
    masks = compute_masks(s, cfg)
    any_mask = np.zeros(len(s), bool)
    for k in REASONS:
        s["mask_" + k] = masks[k]
        any_mask |= masks[k]
    valid = ~any_mask
    s["valid"] = valid
    sel = s["selected"].to_numpy()
    report = _mask_report(masks, sel, valid)

    fit_idx = np.flatnonzero(valid & sel)
    n_sel = int(sel.sum())
    if len(fit_idx) < 60:
        raise ValueError(
            f"Only {len(fit_idx)} valid seconds after masking ({n_sel} selected); loosen the mask thresholds "
            "or choose different laps."
        )
    beta_f, w_f, _ = _solve(s, fit_idx, cfg)
    cda_fixed = float(beta_f[0])
    joint = None
    cda, crr = cda_fixed, cfg.crr
    beta_j, w_j, sc_j = _solve(s, fit_idx, cfg, joint=True)
    X = np.column_stack([s["x"].to_numpy()[fit_idx], s["r"].to_numpy()[fit_idx]])
    Xn = X / np.linalg.norm(X, axis=0)
    cond = float(np.linalg.cond(Xn))
    corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    joint = {"cda": float(beta_j[0]), "crr": float(beta_j[1]), "cond": cond, "corr": corr,
             "ill_conditioned": bool(cond > 20 or corr > 0.95 or not (0.001 <= beta_j[1] <= 0.012))}
    if cfg.fit_crr:
        if joint["ill_conditioned"]:
            warn.append(
                f"Joint CdA/Crr fit is poorly conditioned (cond={cond:.0f}, corr={corr:.2f}, "
                f"Crr={beta_j[1]:.4f}); CdA and Crr trade off. Prefer a fixed Crr unless speed varies a lot."
            )
        cda, crr = float(beta_j[0]), float(beta_j[1])
        w_use = w_j
    else:
        w_use = w_f
    w_full = np.zeros(len(s))
    w_full[fit_idx] = w_use
    resid = s["y0"].to_numpy()[fit_idx] - crr * s["r"].to_numpy()[fit_idx] - cda * s["x"].to_numpy()[fit_idx]
    rms = float(np.sqrt(np.mean(resid ** 2)))
    ci = _bootstrap(s, fit_idx, cfg, cfg.fit_crr, crr)

    s["robust_w"] = w_full
    s["fit_resid"] = np.where(valid, s["y0"] - crr * s["r"] - cda * s["x"], np.nan)
    s["rolling_cda"] = _rolling_cda(s, w_full, valid & sel, crr, cfg)

    fm = valid & sel
    hw = s["headwind"].to_numpy()
    rows = []
    for nm, sub in (("headwind (> +0.5 m/s)", hw > 0.5), ("neutral", np.abs(hw) <= 0.5), ("tailwind (< -0.5 m/s)", hw < -0.5)):
        c, nn = _fit_subset(s, fm & sub, cfg, crr)
        rows.append({"condition": nm, "cda": c, "seconds": nn, "mean_headwind": float(np.mean(hw[fm & sub])) if nn else np.nan})
    wind_split = pd.DataFrame(rows)
    vv = s["speed"].to_numpy()
    spd_rows = []
    if len(fit_idx) >= 100:
        edges = np.unique(np.quantile(vv[fm], np.linspace(0, 1, 6)))
        for lo, hi in zip(edges[:-1], edges[1:]):
            sub = fm & (vv >= lo) & ((vv < hi) | (hi == edges[-1]) & (vv <= hi))
            c, nn = _fit_subset(s, sub, cfg, crr)
            spd_rows.append({"speed_lo_kmh": lo * 3.6, "speed_hi_kmh": hi * 3.6, "cda": c, "seconds": nn})
    speed_split = pd.DataFrame(spd_rows, columns=["speed_lo_kmh", "speed_hi_kmh", "cda", "seconds"])

    # per-lap
    lap_rows = []
    for lp in sorted(pd.unique(s["lap"])):
        lm = (s["lap"] == lp).to_numpy()
        c, nn = _fit_subset(s, lm & valid, cfg, crr) if (lm & valid).sum() >= cfg.min_lap_valid_s else (np.nan, int((lm & valid).sum()))
        lap_rows.append({
            "lap": int(lp), "selected": bool(sel[lm].any()), "seconds": int(lm.sum()),
            "distance_km": float(np.nanmax(s["dist_km"][lm]) - np.nanmin(s["dist_km"][lm])) if lm.any() else np.nan,
            "avg_power": float(np.nanmean(ride.df["power"].to_numpy()[lm])),
            "avg_speed_kmh": float(np.nanmean(s["speed"][lm]) * 3.6),
            "valid_s": nn if np.isfinite(c) else int((lm & valid).sum()),
            "valid_pct": 100.0 * (lm & valid).sum() / max(lm.sum(), 1),
            "cda": c,
        })
    laps_df = pd.DataFrame(lap_rows)

    if meta.get("wind_source") == "none" and cfg.wind_mode != "none":
        pass
    res = CdAResult(
        cda=cda, cda_ci=ci, crr=crr, cda_fixed_crr=cda_fixed, joint=joint, n_valid=len(fit_idx), n_selected=n_sel,
        resid_rms_w=rms, series=s, mask_report=report, laps=laps_df, wind_split=wind_split, speed_split=speed_split,
        energy=_energy(s, fm, cda, crr, cfg), energy_selection=_energy(s, sel & np.isfinite(s["x"].to_numpy()), cda, crr, cfg),
        cfg=cfg, meta=meta, warnings=warn,
    )
    if scan_wind and meta.get("wind_source") not in ("none", None):
        res.wind_scan = wind_scale_scan(res)
    return res


# ----------------------------------------------------------------------------- virtual elevation

def virtual_elevation(series: pd.DataFrame, cda: float, crr: float, cfg: CdAConfig,
                      start: int | None = None, end: int | None = None) -> pd.DataFrame:
    """Chung virtual elevation over rows [start, end) of an analysis ``series``.

    Returns DataFrame with ``dist_km``, ``ve`` (starting at the actual altitude) and ``alt`` (actual).
    """
    sub = series.iloc[start:end]
    m = cfg.mass_kg
    m_eff = m + cfg.wheel_inertia_kg
    v = sub["speed"].to_numpy(float)
    f_aero_v = 0.5 * sub["rho"].to_numpy(float) * cda * sub["v_air"].to_numpy(float) * np.abs(sub["v_air"].to_numpy(float)) * v
    dh = (sub["p_wheel"].to_numpy(float) - crr * m * G * v - f_aero_v - m_eff * v * sub["accel"].to_numpy(float)) / (m * G)
    dh = np.nan_to_num(dh)  # dt = 1 s
    alt = sub["alt"].to_numpy(float)
    ve = alt[0] + np.cumsum(dh)
    return pd.DataFrame({"t_s": sub["t_s"].to_numpy(), "dist_km": sub["dist_km"].to_numpy(), "ve": ve, "alt": alt}, index=sub.index)


def fit_cda_ve(series: pd.DataFrame, crr: float, cfg: CdAConfig, start: int | None = None,
               end: int | None = None, method: str = "lsq") -> float:
    """CdA for which the virtual elevation best matches actual altitude ('lsq' shape fit or 'endpoint' match)."""
    f = lambda c: virtual_elevation(series, c, crr, cfg, start, end)
    a0, a1 = f(0.0), f(1.0)
    alt = a0["alt"].to_numpy()
    A, B = a0["ve"].to_numpy(), a1["ve"].to_numpy() - a0["ve"].to_numpy()  # VE = A + c*B (linear in c)
    if method == "endpoint":
        d = (alt[-1] - A[-1]) / (B[-1] if abs(B[-1]) > 1e-9 else np.nan)
        return float(d)
    At, Bt, Lt = A - A.mean(), B - B.mean(), alt - alt.mean()
    return float(np.sum(Bt * (Lt - At)) / np.sum(Bt ** 2))
