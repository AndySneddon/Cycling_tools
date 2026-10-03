"""CdA estimation from power-meter rides (dynamic power balance, robust fit, diagnostics).

Pure functions and dataclasses; no streamlit.

Model (per sample)::

    P*eff = Crr*m*g*cos(th)*v + m*g*sin(th)*v + 0.5*rho*CdA*v_air*|v_air|*v + (m+I)*v*a
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from concurrent.futures import ThreadPoolExecutor
import os

from .fit_io import Ride
from .geo import angle_diff_deg, bearing_deg, smooth_heading_deg
from .physics import G, WHEEL_INERTIA_KG, air_density, standard_pressure_pa
from .weather import Weather, WeatherError, fetch_weather

WIND_SCAN_SCALES = np.round(np.arange(0.0, 1.2001, 0.05), 2)
FLAT_ALT_RANGE_M = 40.0      # selected-lap altitude range (p1-p99) below this counts as flat
ALT_SMOOTH_FLAT_S = 41
ALT_SMOOTH_HILLY_S = 21
CRR_STEP = 0.001             # +/- Crr used for the sensitivity / systematic range
EFF_REL = 0.015              # +/- relative drivetrain-efficiency uncertainty
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
    # Crr is only valid as a PAIR with the CdA it was fitted with (the race planner uses both from the rider profile),
    # so the default is not changed here: the page reads it from the profile. Typical values: smooth tarmac
    # 0.003-0.004, real roads 0.004-0.006; CdA moves by about -0.010 per +0.001 Crr (see CdAResult.crr_sensitivity).
    crr: float = 0.0031
    # joint CdA+Crr fit WITH a Gaussian prior on Crr (an unconstrained fit returns absurd Crr of 0.008-0.030)
    fit_crr: bool = False
    crr_prior_mean: float = 0.0040
    crr_prior_sd: float = 0.0008
    drivetrain_eff: float = 0.97
    wheel_inertia_kg: float = WHEEL_INERTIA_KG
    # wind: "weather" (needs Weather, scaled by wind_scale), "manual", "none"
    # wind_scale: a number (manual override) or "auto" = the scale (0..1.2) minimising the Huber loss, see auto_wind_scale
    wind_mode: str = "weather"
    wind_scale: float | str = "auto"
    manual_wind_ms: float = 0.0
    manual_wind_from_deg: float = 0.0
    rho_override: float | None = None
    speed_smooth_s: int = 5
    power_smooth_s: int = 5
    alt_smooth_s: int | str = "auto"   # "auto": >= 41 s on flat selections (altitude range < FLAT_ALT_RANGE_M), else 21 s
    heading_span_s: int = 4
    heading_smooth_s: int = 5
    rolling_window_s: int = 120
    rolling_min_frac: float = 0.25
    huber_k: float = 1.345
    bootstrap_n: int = 200
    block_s: int = 600   # bootstrap block length (s): short blocks understate the CI (autocorrelated errors)
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
    wind_auto: dict | None = None        # wind-scale optimum, thirds stability, head/tail gap (weather mode only)
    crr_sensitivity: float = float("nan")  # CdA change for +0.001 Crr
    sys_range: dict | None = None        # systematic range components and combined (quadrature) half-width
    confidence: dict | None = None       # {"level": High/Medium/Low, "items": [...]}

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


try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

if njit is not None:
    @njit(cache=True, nogil=True)
    def _huber1_nb(x, y, k, iters):
        """Single-regressor Huber IRLS through the origin: same algorithm as ``_huber`` for p == 1."""
        n = x.shape[0]
        sxy = 0.0
        sxx = 0.0
        for i in range(n):
            sxy += x[i] * y[i]
            sxx += x[i] * x[i]
        beta = sxy / sxx
        w = np.ones(n)
        s = np.nan
        r = np.empty(n)
        for _ in range(iters):
            for i in range(n):
                r[i] = y[i] - x[i] * beta
            med = np.median(r)
            ar = np.abs(r - med)
            s = max(1.4826 * np.median(ar), 1e-6)
            num = 0.0
            den = 0.0
            for i in range(n):
                u = abs(r[i]) / (k * s)
                wi = 1.0 if u <= 1.0 else 1.0 / max(u, 1e-12)
                w[i] = wi
                num += wi * x[i] * y[i]
                den += wi * x[i] * x[i]
            nb = num / den
            done = abs(nb - beta) <= 1e-10 + 1e-7 * abs(beta)
            beta = nb
            if done:
                break
        return beta, w, s


def _huber(X: np.ndarray, y: np.ndarray, k: float = 1.345, iters: int = 40):
    """Huber IRLS regression. Returns (beta, weights, scale)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n < max(p, 3):
        return np.full(p, np.nan), np.ones(n), np.nan
    if p == 1 and njit is not None:
        b, w, s = _huber1_nb(np.ascontiguousarray(X[:, 0]), np.ascontiguousarray(y, dtype=float), float(k), int(iters))
        return np.array([b]), w, s
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

def _prepare(ride: Ride, cfg: CdAConfig, weather: Weather | None, laps,
             scale_override: float | None = None) -> tuple[pd.DataFrame, list[str], dict]:
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

    sel = np.ones(n, bool) if not laps else df["lap"].isin(list(laps)).to_numpy()

    # gradient from altitude
    alt_raw = df["alt"].to_numpy(float)
    if np.isfinite(alt_raw).sum() > 30:
        alt = _fill(alt_raw)
        if isinstance(cfg.alt_smooth_s, str):  # "auto": flatter selections need more smoothing (barometer noise)
            a_sel = alt[sel] if sel.any() else alt
            flat = (np.nanpercentile(a_sel, 99) - np.nanpercentile(a_sel, 1)) < FLAT_ALT_RANGE_M
            alt_win = ALT_SMOOTH_FLAT_S if flat else ALT_SMOOTH_HILLY_S
        else:
            alt_win = int(cfg.alt_smooth_s)
        meta["alt_smooth_used"] = alt_win
        win = int(alt_win) | 1
        win = min(win, (len(alt) // 2) * 2 - 1) if len(alt) > 5 else 5
        from scipy.signal import savgol_filter  # lazy: scipy.signal drags in scipy.stats (~0.5 s of import time)

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
        ws = cfg.wind_scale
        scale = float(scale_override) if scale_override is not None else (0.5 if isinstance(ws, str) else float(ws))
    elif mode == "manual":
        hw_raw = cfg.manual_wind_ms * np.cos(np.radians(cfg.manual_wind_from_deg - hd))
        scale = 1.0
    else:
        scale = 0.0
    hw_raw = np.nan_to_num(hw_raw)
    hw = hw_raw * scale
    meta["wind_source"] = wind_src
    meta["wind_mode_used"] = mode
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


def _autocorr_tau(resid: np.ndarray, max_lag: int = 120) -> float:
    """Integrated autocorrelation time of a residual series (NaN = missing): n samples carry about n/tau of
    independent information. Used to deflate the data weight against the Crr prior (1 Hz data are strongly correlated)."""
    e = np.asarray(resid, float)
    ok = np.isfinite(e)
    e = np.where(ok, e - np.nanmean(e), 0.0)
    den = float(np.sum(e * e))
    if den <= 0:
        return 1.0
    tau = 1.0
    for k in range(1, min(max_lag, len(e) - 1)):
        both = ok[:-k] & ok[k:]
        rho = float(np.sum(e[:-k] * e[k:] * both)) / den * (len(e) / max(both.sum(), 1))
        if rho <= 0:
            break
        tau += 2 * rho
    return float(np.clip(tau, 1.0, 300.0))


def _joint_prior(x, r, y0, k: float, mu: float, sd: float, tau: float, iters: int = 40):
    """Huber-robust joint CdA + Crr fit with a Gaussian prior Crr ~ N(mu, sd) (MAP estimate).

    Minimises  sum_i huber(res_i) / (tau * s^2)  +  (crr - mu)^2 / sd^2  with s the robust residual scale and tau
    the residual autocorrelation time (so the 1 Hz samples are not treated as independent). Returns
    (beta=[cda, crr], weights, s, cov 2x2).
    """
    x = np.asarray(x, float)
    r = np.asarray(r, float)
    y = np.asarray(y0, float)
    n = len(x)
    if n < 10:
        return np.array([np.nan, np.nan]), np.ones(n), np.nan, np.full((2, 2), np.nan)
    beta = np.array([np.sum(x * (y - mu * r)) / np.sum(x * x), mu])
    w = np.ones(n)
    s = np.nan
    N = np.eye(2)
    for _ in range(iters):
        res = y - beta[0] * x - beta[1] * r
        s = max(1.4826 * np.median(np.abs(res - np.median(res))), 1e-6)
        u = np.abs(res) / (k * s)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
        f = 1.0 / (tau * s * s)
        wx = w * x
        N = f * np.array([[np.sum(wx * x), np.sum(wx * r)], [np.sum(wx * r), np.sum(w * r * r)]])
        N[1, 1] += 1.0 / sd ** 2
        b = np.array([f * np.sum(wx * y), f * np.sum(w * r * y) + mu / sd ** 2])
        nb = np.linalg.solve(N, b)
        done = np.allclose(nb, beta, rtol=1e-7, atol=1e-10)
        beta = nb
        if done:
            break
    return beta, w, s, np.linalg.inv(N)


if njit is not None:
    @njit(cache=True, nogil=True)
    def _boot_one(x_all, y_all, flat, starts, pick, k, iters):
        n = 0
        for g in pick:
            n += starts[g + 1] - starts[g]
        xb = np.empty(n)
        yb = np.empty(n)
        j = 0
        for g in pick:
            for q in range(starts[g], starts[g + 1]):
                i = flat[q]
                xb[j] = x_all[i]
                yb[j] = y_all[i]
                j += 1
        return _huber1_nb(xb, yb, k, iters)[0]


def _bootstrap(s: pd.DataFrame, fit_idx: np.ndarray, cfg: CdAConfig, joint: bool, crr_used: float, tau: float = 1.0):
    """Moving-block bootstrap 95% CI of CdA (block length cfg.block_s, shortened for short selections so that
    there are at least ~6 blocks). ``joint`` re-fits CdA+Crr with the Crr prior instead of fixing Crr."""
    if cfg.bootstrap_n <= 0 or len(fit_idx) < 60:
        return (np.nan, np.nan)
    span = int(fit_idx[-1] - fit_idx[0] + 1)
    block = int(max(30, min(cfg.block_s, span // 6)))
    rng = np.random.default_rng(cfg.seed)
    blocks = fit_idx // block
    ub = np.unique(blocks)
    groups = [fit_idx[blocks == b] for b in ub]
    if len(groups) < 3:
        return (np.nan, np.nan)
    x_all = s["x"].to_numpy()
    y_all = s["y0"].to_numpy() - crr_used * s["r"].to_numpy()
    r_all = s["r"].to_numpy()
    y0_all = s["y0"].to_numpy()
    picks = [rng.integers(0, len(groups), len(groups)) for _ in range(cfg.bootstrap_n)]
    if not joint and njit is not None:
        flat = np.concatenate(groups).astype(np.int64)
        starts = np.concatenate([[0], np.cumsum([len(g) for g in groups])]).astype(np.int64)
        xa, ya = np.ascontiguousarray(x_all, dtype=float), np.ascontiguousarray(y_all, dtype=float)
        one = lambda pk: _boot_one(xa, ya, flat, starts, pk.astype(np.int64), float(cfg.huber_k), 15)
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:  # kernel releases the GIL
            est = np.asarray(list(ex.map(one, picks)))
        return (float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5)))
    est = []
    for pick in picks:
        idx = np.concatenate([groups[i] for i in pick])
        if joint:
            beta, _, _, _ = _joint_prior(x_all[idx], r_all[idx], y0_all[idx], cfg.huber_k, cfg.crr_prior_mean,
                                         cfg.crr_prior_sd, tau, iters=15)
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


def _scaled_series(s: pd.DataFrame, sc: float, cfg: CdAConfig) -> pd.DataFrame:
    """Copy of a prepared series with the wind re-applied at scale ``sc`` (headwind, airspeed, aero regressor and
    the columns derived from them), so masks and fits can be recomputed for that scale."""
    t = s.copy(deep=False)
    hw = s["hw_raw"].to_numpy(float) * sc
    v = s["speed"].to_numpy(float)
    va = v + hw
    x = 0.5 * s["rho"].to_numpy(float) * va * np.abs(va) * v
    y = s["y0"].to_numpy(float) - cfg.crr * s["r"].to_numpy(float)
    t["headwind"], t["v_air"], t["x"], t["y"], t["implied_aero"] = hw, va, x, y, y
    with np.errstate(invalid="ignore", divide="ignore"):
        t["implied_cda"] = np.where(np.abs(x) > 1.0, y / x, np.nan)
    return t


def _valid_for(t: pd.DataFrame, cfg: CdAConfig) -> np.ndarray:
    anym = np.zeros(len(t), bool)
    for m in compute_masks(t, cfg).values():
        anym |= m
    return ~anym


def _parabolic_min(xs: np.ndarray, ys: np.ndarray) -> float:
    """Location of the minimum of ys(xs): grid argmin refined by a parabola through its neighbours."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if not np.isfinite(ys).any():
        return float("nan")
    i = int(np.nanargmin(ys))
    if 0 < i < len(xs) - 1 and np.isfinite(ys[i - 1]) and np.isfinite(ys[i + 1]):
        d = ys[i - 1] - 2 * ys[i] + ys[i + 1]
        if d > 0:
            off = 0.5 * (ys[i - 1] - ys[i + 1]) / d
            return float(xs[i] + np.clip(off, -1, 1) * (xs[i + 1] - xs[i]))
    return float(xs[i])


def _huber_loss(r: np.ndarray, delta: float) -> float:
    a = np.abs(r)
    return float(np.mean(np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))))


def _scan_core(s: pd.DataFrame, cfg: CdAConfig, crr: float, scales: Sequence[float]) -> tuple[pd.DataFrame, dict]:
    """Scan the wind scale. Every scale gets its OWN validity mask (the low-airspeed / braking masks depend on the
    wind) and its own scaled head/tail split (``cda``, ``loss``, ``cda_head``, ``cda_tail`` columns). The optimum
    is chosen from ``loss_common``: the Huber loss on the samples valid at EVERY scale with one fixed Huber
    threshold, so all scales are compared on identical data. Per-third optima measure its stability.
    """
    scales = np.asarray(scales, float)
    sel = s["selected"].to_numpy(bool)
    y = s["y0"].to_numpy(float) - crr * s["r"].to_numpy(float)
    hw_raw = s["hw_raw"].to_numpy(float)
    xs, valids = [], []
    for sc in scales:
        t = _scaled_series(s, sc, cfg)
        x = t["x"].to_numpy(float)
        xs.append(x)
        valids.append(_valid_for(t, cfg) & sel & np.isfinite(x) & np.isfinite(y))
    cidx = np.flatnonzero(np.logical_and.reduce(valids))
    k = cfg.huber_k
    delta = np.nan
    if len(cidx) >= 30:  # one fixed Huber threshold for all scales: from the median robust residual scale
        delta = k * float(np.median([_huber(x[cidx], y[cidx], k)[2] for x in xs]))
    thirds = np.array_split(cidx, 3) if len(cidx) >= 90 else []
    rows = []
    loss_c = []
    loss_t = [[] for _ in thirds]
    for sc, x, valid in zip(scales, xs, valids):
        idx = np.flatnonzero(valid)
        out = {"wind_scale": float(sc), "n_valid": int(len(idx))}
        if len(idx) >= 30:
            b, _, s_own = _huber(x[idx], y[idx], k)
            r = y[idx] - b[0] * x[idx]
            hw = sc * hw_raw[idx]
            out.update(cda=float(b[0]), rms_w=float(np.sqrt(np.mean(r ** 2))), loss=_huber_loss(r, k * s_own))
            for nm, sub in (("cda_head", hw > 0.5), ("cda_tail", hw < -0.5)):
                out[nm] = float(_huber(x[idx][sub], y[idx][sub], k)[0][0]) if sub.sum() >= 30 else np.nan
        else:
            out.update(cda=np.nan, rms_w=np.nan, loss=np.nan, cda_head=np.nan, cda_tail=np.nan)
        out["head_minus_tail"] = out["cda_head"] - out["cda_tail"]
        if len(cidx) >= 30:
            b, _, _ = _huber(x[cidx], y[cidx], k)
            loss_c.append(_huber_loss(y[cidx] - b[0] * x[cidx], delta))
            for j, th in enumerate(thirds):
                bt, _, _ = _huber(x[th], y[th], k)
                loss_t[j].append(_huber_loss(y[th] - bt[0] * x[th], delta))
        else:
            loss_c.append(np.nan)
        out["loss_common"] = loss_c[-1]
        rows.append(out)
    table = pd.DataFrame(rows)
    opt = _parabolic_min(scales, np.asarray(loss_c))
    t_opt = [_parabolic_min(scales, np.asarray(l)) for l in loss_t]
    info = {"scale": opt, "thirds": t_opt, "common_n": int(len(cidx)),
            "at_edge": bool(np.isfinite(opt) and (opt <= scales[0] + 1e-9 or opt >= scales[-1] - 1e-9)),
            "third_range": (float(np.nanmax(t_opt) - np.nanmin(t_opt)) if t_opt and np.isfinite(t_opt).any() else np.nan)}
    return table, info


def wind_scale_scan(result: CdAResult, scales: Sequence[float] | None = None) -> pd.DataFrame:
    """Scan wind scale; report CdA, fit loss and head-vs-tail CdA difference (should be ~0 at the right scale).

    Each scale uses its own validity mask and its own scaled head/tail split.
    """
    scales = WIND_SCAN_SCALES if scales is None else np.asarray(scales, float)
    table, _ = _scan_core(result.series, result.cfg, result.crr, scales)
    return table


def best_wind_scale(scan: pd.DataFrame) -> dict:
    """Best scale by minimum loss (common-sample loss when available) and by head/tail balance (zero crossing)."""
    col = "loss_common" if "loss_common" in scan and scan["loss_common"].notna().any() else "loss"
    res = {"by_loss": _parabolic_min(scan["wind_scale"].to_numpy(float), scan[col].to_numpy(float)), "by_balance": np.nan}
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
    scan_tab, scan_info = None, None
    if meta["wind_mode_used"] == "weather":
        # scan 0..1.2 (own mask per scale); in "auto" mode adopt the loss-minimising scale
        scan_tab, scan_info = _scan_core(s, cfg, cfg.crr, WIND_SCAN_SCALES)
        if isinstance(cfg.wind_scale, str):
            sc = scan_info["scale"]
            if not np.isfinite(sc):
                sc = 0.5
                warn.append("Automatic wind scale failed (too little valid data); using 0.5.")
            s = _scaled_series(s, float(sc), cfg)
            meta["wind_scale_used"] = float(sc)
            if scan_info["at_edge"]:
                warn.append(f"Automatic wind scale ended at the edge of the scan ({sc:.2f}); the wind data may not match this ride.")
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
    cda, crr = cda_fixed, cfg.crr
    xv, rv, y0v = s["x"].to_numpy(), s["r"].to_numpy(), s["y0"].to_numpy()
    # residual autocorrelation time of the fixed-Crr fit: 1 Hz samples are far from independent
    e_full = np.full(len(s), np.nan)
    e_full[fit_idx] = y0v[fit_idx] - cfg.crr * rv[fit_idx] - cda_fixed * xv[fit_idx]
    tau = _autocorr_tau(e_full)
    beta_j, w_j, sc_j, cov_j = _joint_prior(xv[fit_idx], rv[fit_idx], y0v[fit_idx], cfg.huber_k,
                                            cfg.crr_prior_mean, cfg.crr_prior_sd, tau)
    X = np.column_stack([xv[fit_idx], rv[fit_idx]])
    cond = float(np.linalg.cond(X / np.linalg.norm(X, axis=0)))
    corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    crr_sd = float(np.sqrt(cov_j[1, 1]))
    joint = {"cda": float(beta_j[0]), "crr": float(beta_j[1]), "cda_sd": float(np.sqrt(cov_j[0, 0])), "crr_sd": crr_sd,
             "prior_mean": cfg.crr_prior_mean, "prior_sd": cfg.crr_prior_sd, "tau": tau, "cond": cond, "corr": corr,
             # data barely informs Crr: the estimate is mostly the prior (CdA and Crr are strongly correlated)
             "ill_conditioned": bool(cond > 20 or corr > 0.95 or crr_sd > 0.7 * cfg.crr_prior_sd)}
    if cfg.fit_crr:
        if joint["ill_conditioned"]:
            warn.append(
                f"Joint CdA/Crr fit is prior-dominated (cond={cond:.0f}, corr={corr:.2f}, posterior Crr sd "
                f"{crr_sd:.4f} vs prior sd {cfg.crr_prior_sd:.4f}): the data barely inform Crr, so CdA and Crr trade off. "
                "Prefer a fixed Crr unless speed varies a lot."
            )
        cda, crr = float(beta_j[0]), float(beta_j[1])
        w_use = w_j
    else:
        w_use = w_f
    w_full = np.zeros(len(s))
    w_full[fit_idx] = w_use
    resid = s["y0"].to_numpy()[fit_idx] - crr * s["r"].to_numpy()[fit_idx] - cda * s["x"].to_numpy()[fit_idx]
    rms = float(np.sqrt(np.mean(resid ** 2)))
    ci = _bootstrap(s, fit_idx, cfg, cfg.fit_crr, crr, tau)

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

    gap = (wind_split["cda"].iloc[0] - wind_split["cda"].iloc[2]) if len(wind_split) == 3 else np.nan
    meta["head_minus_tail"] = float(gap)
    meta["tau_s"] = tau
    meta["ci_block_s"] = int(max(30, min(cfg.block_s, int(fit_idx[-1] - fit_idx[0] + 1) // 6)))
    # ---- sensitivities and systematic range (CdA moves with Crr, wind scale and drivetrain efficiency)
    fixed = lambda c, dy=0.0: float(_huber(xv[fit_idx], y0v[fit_idx] + dy - c * rv[fit_idx], cfg.huber_k)[0][0])
    d_up, d_dn = fixed(crr + CRR_STEP) - fixed(crr), fixed(crr - CRR_STEP) - fixed(crr)
    crr_half = 0.5 * abs(d_up - d_dn)
    pw = s["power"].to_numpy()[fit_idx]
    base_y = y0v[fit_idx] - crr * rv[fit_idx]
    cda_e = [float(_huber(xv[fit_idx], base_y + pw * (cfg.drivetrain_eff * f - cfg.drivetrain_eff), cfg.huber_k)[0][0])
             for f in (1 - EFF_REL, 1 + EFF_REL)]
    eff_half = 0.5 * abs(cda_e[1] - cda_e[0])
    wind_half, wind_list = 0.0, []
    if scan_tab is not None and scan_info["thirds"]:
        grid, cda_grid = scan_tab["wind_scale"].to_numpy(), scan_tab["cda"].to_numpy()
        okg = np.isfinite(cda_grid)
        if okg.sum() >= 2:
            ref = float(np.interp(meta["wind_scale_used"], grid[okg], cda_grid[okg]))
            wind_list = [float(np.interp(t, grid[okg], cda_grid[okg])) - ref for t in scan_info["thirds"] if np.isfinite(t)]
            wind_half = max([abs(d) for d in wind_list], default=0.0)
    half = float(np.sqrt(crr_half ** 2 + eff_half ** 2 + wind_half ** 2))
    sys_range = {"crr_half": float(crr_half), "wind_half": float(wind_half), "eff_half": float(eff_half), "half": half,
                 "lo": cda - half, "hi": cda + half, "crr_step": CRR_STEP, "eff_rel": EFF_REL,
                 "note": "Components added in quadrature. Excludes power-meter bias (a 2% error is ~2% in CdA), mass and wind-model error."}
    wind_auto = None
    if scan_info is not None:
        wind_auto = {"auto": isinstance(cfg.wind_scale, str), "used": float(meta["wind_scale_used"]),
                     "optimum": float(scan_info["scale"]), "thirds": [float(t) for t in scan_info["thirds"]],
                     "third_range": float(scan_info["third_range"]), "common_n": scan_info["common_n"],
                     "head_minus_tail": float(gap), "at_edge": scan_info["at_edge"]}
    res = CdAResult(
        cda=cda, cda_ci=ci, crr=crr, cda_fixed_crr=cda_fixed, joint=joint, n_valid=len(fit_idx), n_selected=n_sel,
        resid_rms_w=rms, series=s, mask_report=report, laps=laps_df, wind_split=wind_split, speed_split=speed_split,
        energy=_energy(s, fm, cda, crr, cfg), energy_selection=_energy(s, sel & np.isfinite(s["x"].to_numpy()), cda, crr, cfg),
        cfg=cfg, meta=meta, warnings=warn, wind_scan=scan_tab, wind_auto=wind_auto, crr_sensitivity=float(d_up),
        sys_range=sys_range,
    )
    res.confidence = assess_confidence(res)
    return res


def assess_confidence(res: CdAResult) -> dict:
    """High / Medium / Low from: head-minus-tail CdA gap at the chosen wind scale, stability of the wind-scale
    optimum across thirds of the ride, % valid data and the systematic range. Returns {"level", "items"}."""
    items = []

    def add(name, value, text, grade):  # grade: 2 good, 1 ok, 0 bad, None = not applicable
        items.append({"name": name, "value": value, "text": text, "grade": grade})

    wa, sr = res.wind_auto, res.sys_range or {}
    if wa is not None:
        g = abs(wa["head_minus_tail"])
        add("Head/tail gap", g, f"{wa['head_minus_tail']:+.3f} m² at wind scale {wa['used']:.2f}",
            None if not np.isfinite(g) else 2 if g <= 0.015 else 1 if g <= 0.03 else 0)
        tr, wh = wa["third_range"], sr.get("wind_half", 0.0)
        # an unstable optimum only matters if CdA is sensitive to the wind scale
        grade = None if not np.isfinite(tr) else 2 if (tr <= 0.2 or wh <= 0.004) else 1 if tr <= 0.4 else 0
        add("Wind-scale stability", tr, f"optimum varies by {tr:.2f} across thirds of the ride (CdA effect ±{wh:.3f})", grade)
    else:
        add("Wind", None, "no weather wind applied; head/tail check not available", None)
    vp = res.valid_pct
    add("Valid data", vp, f"{vp:.0f}% of the selection", 2 if vp >= 60 else 1 if vp >= 35 else 0)
    half = sr.get("half", np.nan)
    add("Systematic range", half, f"±{half:.3f} m² (Crr ±{CRR_STEP:.3f}, wind scale, efficiency ±{EFF_REL * 100:.1f}%)",
        None if not np.isfinite(half) else 2 if half <= 0.012 else 1 if half <= 0.025 else 0)
    g = [i["grade"] for i in items if i["grade"] is not None]
    bad, ok = g.count(0), g.count(1)
    level = "Low" if bad >= 2 else "Medium" if (bad == 1 or ok >= 3) else "High"
    return {"level": level, "items": items}


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
    alt = sub["alt"].to_numpy(float)
    # Rows the CdA fit excludes (coasting, braking, corners, stops, gaps...) or that cannot be modelled carry no
    # information about CdA: there the virtual elevation follows the actual altitude change instead of integrating
    # a physically wrong power balance, and it does so identically for every CdA (so VE stays linear in CdA).
    bad = ~np.isfinite(dh)
    if "valid" in sub:
        bad |= ~sub["valid"].to_numpy(bool)
    dalt = np.nan_to_num(np.diff(alt, prepend=alt[0]))
    dh = np.where(bad, dalt, dh)  # dt = 1 s
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
