"""Forward dynamic race simulation: course + rider + power plan + weather -> speed / time / splits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .course import Course
from .physics import WHEEL_INERTIA_KG, air_density, headwind_component, normalised_power
from .profile import RiderProfile
from .weather import Weather

try:  # numba is optional: fall back to plain python (slower but identical results)
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

DEFAULT_RHO = float(air_density(15.0, 101325.0, 0.5))
V_MIN = 0.3      # m/s: stall floor
V_MAX = 45.0
N_SUB = 16       # sub-steps used when a whole segment cannot be solved (see _kernel)


# --------------------------------------------------------------------- config
@dataclass
class SimConfig:
    coast_speed_mps: float = 19.4   # above this the rider stops pedalling (~70 km/h)
    start_speed_mps: float = 0.0    # speed at the start line
    wheel_inertia_kg: float = WHEEL_INERTIA_KG


@dataclass
class Environment:
    """Weather for a simulation. ``weather=None`` means still air at 15 C / sea level.

    Time-varying (forecast / archive) weather is sampled at (start_time + elapsed time) so wind follows the
    race clock. ``wind_scale`` converts 10 m weather-model wind to wind at the rider.
    """
    weather: Weather | None = None
    start_time: pd.Timestamp | None = None  # UTC (naive) race start
    wind_scale: float = 1.0

    @property
    def time_dependent(self) -> bool:
        return self.weather is not None and self.weather.source != "manual" and self.start_time is not None

    def arrays(self, course: Course, t_seg_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-segment headwind (m/s) and air density (kg/m^3) at elapsed race time ``t_seg_s``."""
        n = course.n_seg
        if self.weather is None:
            return np.zeros(n), np.full(n, DEFAULT_RHO)
        t0 = pd.Timestamp(self.start_time) if self.start_time is not None else pd.Timestamp("2000-01-02")
        # same arithmetic as ``weather.at(t0 + to_timedelta(t, "s"))`` (ns rounding, floor to whole seconds) but
        # without building a DatetimeIndex / DataFrame per call (that was ~85% of the cost of one simulation)
        ns = np.int64(t0.value) + np.rint(np.asarray(t_seg_s, float) * 1e9).astype(np.int64)
        w = self.weather.at_seconds((ns // 1_000_000_000).astype(float))
        hw = headwind_component(w["wind_ms"] * self.wind_scale, w["wind_dir"], course.heading)
        rho = air_density(w["temp_c"], w["pressure_pa"], w["rh"])
        return np.asarray(hw, float), np.asarray(rho, float)

    def describe(self) -> str:
        if self.weather is None:
            return "still air"
        return self.weather.source


# --------------------------------------------------------------------- kernel
@njit(cache=True, nogil=True)
def _resid(v1, v0, ds, fres, hw, k_aero, p, m_eff):
    vavg = 0.5 * (v0 + v1)
    t = ds / vavg
    va = vavg + hw
    return 0.5 * m_eff * (v1 * v1 - v0 * v0) - p * t + fres * ds + k_aero * va * abs(va) * ds


@njit(cache=True, nogil=True)
def _newton(v0, ds, fres, hw, k_aero, p, m_eff, a, b, flim):
    """Root of the segment energy residual on [a, b] (F(a) < 0 < F(b) = ``flim``): safeguarded Newton with the
    analytic derivative and a bisection fallback inside the bracket."""
    c = min(max(v0, a), b)
    for _ in range(80):
        fc = _resid(c, v0, ds, fres, hw, k_aero, p, m_eff)
        if abs(fc) < 1e-8 or (b - a) < 1e-11:
            break
        if fc > 0.0:
            b = c
        else:
            a = c
        vavg = 0.5 * (v0 + c)
        dfc = m_eff * c + 0.5 * p * ds / (vavg * vavg) + k_aero * abs(vavg + hw) * ds
        cn = c - fc / dfc
        if not (a < cn < b):
            cn = 0.5 * (a + b)
        c = cn
    return c


@njit(cache=True, nogil=True)
def _kernel(ds, grade, vcap, headwind, rho, p_set, v_start, mass, m_eff, cda, crr, eff, v_coast):
    """Per-segment energy balance with trapezoidal average speed (constant crank power over each segment).

    The root is found with a safeguarded Newton iteration (~3 residual evaluations per segment). When no root
    exists even at the 0.3 m/s floor the single-step trapezoid rule is not valid (it assumes the *average* speed
    sets the work done, which fails for a rider decelerating towards a slow steady-state speed on a steep
    ramp), so the segment is sub-stepped (``N_SUB`` pieces); only a sub-step that still has no root is a genuine
    stall (crawl at 0.3 m/s)."""
    n = ds.shape[0]
    v_node = np.empty(n + 1)
    seg_t = np.empty(n)
    p_app = np.empty(n)
    v_node[0] = v_start
    v0 = v_start
    for i in range(n):
        th = math.atan(grade[i])
        fres = mass * 9.80665 * (crr * math.cos(th) + math.sin(th))
        k_aero = 0.5 * cda * rho[i]
        coasting = v0 > v_coast
        p = 0.0 if coasting else p_set[i] * eff
        vl = min(vcap[i], 45.0)
        if vl < 0.3:
            vl = 0.3
        flim = _resid(vl, v0, ds[i], fres, headwind[i], k_aero, p, m_eff)
        if flim <= 0.0:
            v1 = vl
            t = ds[i] / (0.5 * (v0 + v1))
            tseg = t
            need = (0.5 * m_eff * (v1 * v1 - v0 * v0) + fres * ds[i]
                    + k_aero * (0.5 * (v0 + v1) + headwind[i]) * abs(0.5 * (v0 + v1) + headwind[i]) * ds[i]) / t
            wheel = min(max(need, 0.0), p)
            p_app[i] = wheel / eff
        else:
            a = 0.3
            fa = _resid(a, v0, ds[i], fres, headwind[i], k_aero, p, m_eff)
            if fa >= 0.0:
                dsub = ds[i] / N_SUB
                vs = v0
                tseg = 0.0
                for _s in range(N_SUB):
                    fa2 = _resid(a, vs, dsub, fres, headwind[i], k_aero, p, m_eff)
                    if fa2 >= 0.0:
                        v1s = a  # genuine stall: crawl at the floor speed
                    else:
                        fl2 = _resid(vl, vs, dsub, fres, headwind[i], k_aero, p, m_eff)
                        if fl2 <= 0.0:
                            v1s = vl
                        else:
                            v1s = _newton(vs, dsub, fres, headwind[i], k_aero, p, m_eff, a, vl, fl2)
                    tseg += dsub / (0.5 * (vs + v1s))
                    vs = v1s
                v1 = vs
            else:
                v1 = _newton(v0, ds[i], fres, headwind[i], k_aero, p, m_eff, a, vl, flim)
                tseg = ds[i] / (0.5 * (v0 + v1))
            p_app[i] = p / eff
        v_node[i + 1] = v1
        seg_t[i] = tseg
        v0 = v1
    return v_node, seg_t, p_app


@njit(cache=True, nogil=True)
def _p1hz_numba(seg_t, p_app):
    """Exact 1 s bin means of piecewise-constant segment power (via cumulative energy), whole seconds only."""
    n = seg_t.shape[0]
    t_node = np.empty(n + 1)
    e_node = np.empty(n + 1)
    t_node[0] = 0.0
    e_node[0] = 0.0
    for i in range(n):
        t_node[i + 1] = t_node[i] + seg_t[i]
        e_node[i + 1] = e_node[i] + p_app[i] * seg_t[i]
    total = t_node[n]
    ng = int(math.floor(total)) + 1
    if ng < 2:
        out = np.empty(1)
        out[0] = e_node[n] / max(total, 1e-9)
        return out
    m = ng - 1
    p1 = np.empty(m)
    j = 0
    prev = 0.0  # energy at grid point 0 (t = 0)
    for g in range(1, ng):
        x = float(g)
        while j < n - 1 and t_node[j + 1] <= x:
            j += 1
        if x >= t_node[n]:
            e = e_node[n]
        else:
            slope = (e_node[j + 1] - e_node[j]) / (t_node[j + 1] - t_node[j])
            e = slope * (x - t_node[j]) + e_node[j]
        p1[g - 1] = e - prev
        prev = e
    return p1


@njit(cache=True, nogil=True)
def _np_from_p1(p1, win):
    """Coggan-style normalised power of a 1 Hz series for a ``win`` second rolling window."""
    m = p1.shape[0]
    if m < win:
        return p1.sum() / m
    acc = 0.0
    cnt = m - win + 1
    s = 0.0
    for k in range(win):
        s += p1[k]
    for i in range(cnt):
        if i > 0:
            s += p1[i + win - 1] - p1[i - 1]
        r = s / win
        r2 = r * r
        acc += r2 * r2
    return (acc / cnt) ** 0.25


@njit(cache=True, nogil=True)
def _np_numba(seg_t, p_app):
    """Normalised power of piecewise-constant segment power; same maths as ``np_from_segments`` (exact 1 s bin
    means via cumulative energy, then 30 s rolling 4th-power mean) without the temporary arrays."""
    return _np_from_p1(_p1hz_numba(seg_t, p_app), 30)


@njit(cache=True, nogil=True)
def _rolling_numba(p1, win, starts):
    """``win`` second rolling means starting at the (clipped) sample indices ``starts``. The output length is
    always ``len(starts)``; a series shorter than the window gives its overall mean everywhere."""
    m = p1.shape[0]
    out = np.empty(starts.shape[0])
    if m <= win:
        mean = p1.sum() / m
        for k in range(starts.shape[0]):
            out[k] = mean
        return out
    nw = m - win + 1
    cs = np.empty(m + 1)
    cs[0] = 0.0
    for i in range(m):
        cs[i + 1] = cs[i] + p1[i]
    for k in range(starts.shape[0]):
        j = min(starts[k], nw - 1)
        out[k] = (cs[j + win] - cs[j]) / win
    return out


def window_starts(n_samples: int, win_s: int, stride: int = 1) -> np.ndarray:
    """Start indices of rolling windows of ``win_s`` seconds in an ``n_samples`` series, every ``stride`` s
    (always including the last window)."""
    nw = max(1, n_samples - int(win_s) + 1)
    st = np.arange(0, nw, max(1, int(stride)), dtype=np.int64)
    if st[-1] != nw - 1:
        st = np.append(st, nw - 1)
    return st


def rolling_mean(p1, win_s: int) -> np.ndarray:
    """Every ``win_s`` second rolling mean of a 1 Hz series (the overall mean if the series is shorter)."""
    p1 = np.asarray(p1, float)
    return _rolling_numba(p1, int(win_s), window_starts(len(p1), win_s))


def rolling_peak(p1, win_s: int) -> float:
    """Best ``win_s`` second average power (the overall mean if the ride is shorter than the window)."""
    return float(rolling_mean(p1, win_s).max())


def np_window(p1, win_s: int) -> float:
    """Normalised power with a ``win_s`` second rolling window (30 = Coggan NP)."""
    return float(_np_from_p1(np.asarray(p1, float), int(win_s)))


# --------------------------------------------------------------------- result
def np_from_segments(seg_t: np.ndarray, p_app: np.ndarray) -> float:
    """Normalised power from piecewise-constant segment power, using exact 1 s bin means (smooth in inputs)."""
    return normalised_power(power_1hz_from(seg_t, p_app))


def power_1hz_from(seg_t: np.ndarray, p_app: np.ndarray) -> np.ndarray:
    t_node = np.concatenate([[0.0], np.cumsum(seg_t)])
    e_node = np.concatenate([[0.0], np.cumsum(p_app * seg_t)])
    grid = np.arange(0.0, math.floor(t_node[-1]) + 1.0)
    if len(grid) < 2:
        return np.array([float(np.sum(p_app * seg_t) / max(t_node[-1], 1e-9))])
    return np.diff(np.interp(grid, t_node, e_node))


@dataclass
class SimResult:
    course: Course
    rider: RiderProfile
    env: Environment
    v_node: np.ndarray
    seg_time: np.ndarray
    power_set: np.ndarray       # planned crank power per segment
    power_applied: np.ndarray   # crank power actually pedalled (0 when coasting / capped surplus)
    headwind: np.ndarray
    rho: np.ndarray
    label: str = ""
    _np: float | None = field(default=None, repr=False)

    @property
    def t_node(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.seg_time)])

    @property
    def total_time_s(self) -> float:
        return float(self.seg_time.sum())

    @property
    def v_seg(self) -> np.ndarray:
        return self.course.ds / self.seg_time

    @property
    def avg_speed_kmh(self) -> float:
        return self.course.length_m / self.total_time_s * 3.6

    @property
    def avg_power(self) -> float:
        return float(np.sum(self.power_applied * self.seg_time) / self.total_time_s)

    @property
    def power_1hz(self) -> np.ndarray:
        return power_1hz_from(self.seg_time, self.power_applied)

    @property
    def np_w(self) -> float:
        if self._np is None:
            self._np = normalised_power(self.power_1hz)
        return self._np

    @property
    def vi(self) -> float:
        return self.np_w / max(self.avg_power, 1e-9)

    @property
    def work_kj(self) -> float:
        return float(np.sum(self.power_applied * self.seg_time) / 1000.0)

    def power_stats(self) -> dict:
        """Average power, 30 s NP and the effort metrics a coach checks: peak 1 / 5 / 20 min power and 120 s NP."""
        p1 = self.power_1hz
        return {
            "avg_power_w": self.avg_power, "np_w": self.np_w, "np120_w": np_window(p1, 120),
            "peak_1min_w": rolling_peak(p1, 60), "peak_5min_w": rolling_peak(p1, 300),
            "peak_20min_w": rolling_peak(p1, 1200),
        }

    def summary(self) -> dict:
        return {
            "time_s": self.total_time_s,
            "time": fmt_time(self.total_time_s),
            "avg_speed_kmh": self.avg_speed_kmh,
            "avg_power_w": self.avg_power,
            "np_w": self.np_w,
            "vi": self.vi,
            "work_kj": self.work_kj,
        }

    def to_frame(self) -> pd.DataFrame:
        c = self.course
        return pd.DataFrame({
            "dist_m": c.seg_mid_m, "elev_m": 0.5 * (c.elev_m[:-1] + c.elev_m[1:]), "grade": c.grade,
            "speed_kmh": self.v_seg * 3.6, "power_set": self.power_set, "power_applied": self.power_applied,
            "headwind_ms": self.headwind, "t_s": self.t_node[:-1] + 0.5 * self.seg_time,
        })


def fmt_time(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ------------------------------------------------------------------ simulate
def _run(course: Course, rider: RiderProfile, power: np.ndarray, hw: np.ndarray, rho: np.ndarray,
         cfg: SimConfig):
    m_eff = rider.mass_kg + cfg.wheel_inertia_kg
    return _kernel(
        course.ds, course.grade, course.vcap, hw, rho, power, float(cfg.start_speed_mps),
        float(rider.mass_kg), float(m_eff), float(rider.cda), float(rider.crr),
        float(rider.drivetrain_eff), float(cfg.coast_speed_mps),
    )


def _simulate_arrays(course: Course, rider: RiderProfile, p: np.ndarray, env: Environment, cfg: SimConfig,
                     frozen_wind: tuple | None, guess_wind: tuple | None = None):
    """Run the kernel (two passes when the weather is time-dependent). Returns (v_node, seg_t, p_app, hw, rho)."""
    if frozen_wind is not None:
        hw, rho = frozen_wind
        v_node, seg_t, p_app = _run(course, rider, p, hw, rho, cfg)
    else:
        hw, rho = guess_wind if guess_wind is not None else env.arrays(course, np.cumsum(course.ds) / 10.0)
        v_node, seg_t, p_app = _run(course, rider, p, hw, rho, cfg)
        if env.time_dependent:
            t_mid = np.cumsum(seg_t) - 0.5 * seg_t
            hw, rho = env.arrays(course, t_mid)
            v_node, seg_t, p_app = _run(course, rider, p, hw, rho, cfg)
    return v_node, seg_t, p_app, hw, rho


def simulate(course: Course, rider: RiderProfile, power, env: Environment | None = None,
             config: SimConfig | None = None, label: str = "", frozen_wind: tuple | None = None) -> SimResult:
    """Simulate riding ``course``. ``power`` is a scalar or per-segment array of crank watts.

    Weather is applied in two passes so the wind follows elapsed time. ``frozen_wind=(hw, rho)`` skips the
    weather lookup (used inside optimisation loops).
    """
    env = env or Environment()
    cfg = config or SimConfig()
    p = np.full(course.n_seg, float(power)) if np.ndim(power) == 0 else np.asarray(power, dtype=float)
    v_node, seg_t, p_app, hw, rho = _simulate_arrays(course, rider, p, env, cfg, frozen_wind)
    return SimResult(course, rider, env, v_node, seg_t, p, p_app, hw, rho, label)


def simulate_even(course: Course, rider: RiderProfile, np_target: float, env: Environment | None = None,
                  config: SimConfig | None = None, frozen_wind: tuple | None = None,
                  p_hint: float | None = None) -> SimResult:
    """Constant pedalling power chosen so that the simulated normalised power equals ``np_target``.

    ``p_hint`` (a nearby solution, e.g. from a neighbouring NP) tightens the starting bracket.
    """
    env = env or Environment()
    cfg = config or SimConfig()
    guess = env.arrays(course, np.cumsum(course.ds) / 10.0) if frozen_wind is None else None  # independent of p

    def f(p):
        _, seg_t, p_app, _, _ = _simulate_arrays(course, rider, np.full(course.n_seg, float(p)), env, cfg,
                                                 frozen_wind, guess)
        return _np_numba(seg_t, p_app) - np_target

    if p_hint is not None and p_hint > 0:
        lo, hi = 0.97 * p_hint, 1.03 * p_hint
    else:
        lo, hi = 0.4 * np_target, 1.6 * np_target
    while f(lo) > 0 and lo > 1.0:
        lo *= 0.5
    while f(hi) < 0 and hi < 5000:
        hi *= 1.5
    p = brentq(f, lo, hi, xtol=1e-4, rtol=1e-9)
    p_arr = np.full(course.n_seg, float(p))
    v_node, seg_t, p_app, hw, rho = _simulate_arrays(course, rider, p_arr, env, cfg, frozen_wind, guess)
    return SimResult(course, rider, env, v_node, seg_t, p_arr, p_app, hw, rho, "even")


# -------------------------------------------------------------------- splits
def splits_table(result: SimResult, every_m: float = 1000.0) -> pd.DataFrame:
    """Per-split time, speed, power, climb and headwind (every_m of horizontal course distance)."""
    c = result.course
    edges = np.arange(0.0, c.length_m, every_m)
    edges = np.append(edges, c.length_m)
    if edges[-1] - edges[-2] < 1.0:
        edges = edges[:-1]
    t_node = result.t_node
    t_at = np.interp(edges, c.dist_m, t_node)
    e_at = np.interp(edges, c.dist_m, c.elev_m)
    cum_energy = np.concatenate([[0.0], np.cumsum(result.power_applied * result.seg_time)])
    cum_hw = np.concatenate([[0.0], np.cumsum(result.headwind * c.ds)])
    en = np.interp(edges, c.dist_m, cum_energy)
    hw = np.interp(edges, c.dist_m, cum_hw)
    tag = every_m / 1000.0
    rows = []
    for i in range(len(edges) - 1):
        dt = t_at[i + 1] - t_at[i]
        dd = edges[i + 1] - edges[i]
        seg_set = (c.dist_m[:-1] >= edges[i]) & (c.dist_m[:-1] < edges[i + 1])
        rows.append({
            "split": f"{edges[i] / 1000:.1f}-{edges[i + 1] / 1000:.1f} km",
            "end_km": edges[i + 1] / 1000.0,
            "split_time": fmt_time(dt),
            "split_s": dt,
            "cum_time": fmt_time(t_at[i + 1]),
            "speed_kmh": dd / dt * 3.6,
            "power_w": (en[i + 1] - en[i]) / dt,
            "set_power_w": float(np.average(result.power_set[seg_set], weights=c.ds[seg_set])) if seg_set.any() else np.nan,
            "elev_gain_m": float(np.clip(np.diff(c.elev_m)[seg_set], 0, None).sum()),
            "net_elev_m": float(e_at[i + 1] - e_at[i]),
            "headwind_ms": (hw[i + 1] - hw[i]) / dd,
        })
    df = pd.DataFrame(rows)
    df.attrs["every_km"] = tag
    return df


# ------------------------------------------------------------ NP sensitivity
def time_vs_np(course: Course, rider: RiderProfile, np_values, env: Environment | None = None,
               config: SimConfig | None = None) -> pd.DataFrame:
    """Predicted time for even pacing at each normalised power (the 'sliding scale')."""
    env = env or Environment()
    rows = []
    hint = None
    prev_np = None
    for npv in np_values:
        npv = float(npv)
        # constant-power solutions scale ~linearly with NP: warm-start the bracket from the previous point
        h = None if hint is None else hint * npv / prev_np
        r = simulate_even(course, rider, npv, env, config, p_hint=h)
        hint, prev_np = float(r.power_set[0]), npv
        rows.append({"np_w": npv, "time_s": r.total_time_s, "avg_speed_kmh": r.avg_speed_kmh,
                     "avg_power_w": r.avg_power})
    df = pd.DataFrame(rows)
    df["time"] = df["time_s"].map(fmt_time)
    # seconds gained per extra 5 W (positive = faster); central differences where possible
    dt_dw = np.gradient(df["time_s"].to_numpy(), df["np_w"].to_numpy()) if len(df) > 1 else np.array([np.nan])
    df["s_saved_per_5w"] = -5.0 * dt_dw
    return df


def what_if(course: Course, rider: RiderProfile, np_target: float, env: Environment | None = None,
            config: SimConfig | None = None, crr_options=(0.0025, 0.0035, 0.0045)) -> pd.DataFrame:
    """Time saved (positive = faster) for equipment / power changes, all at even pacing."""
    env = env or Environment()
    base_res = simulate_even(course, rider, np_target, env, config)
    base = base_res.total_time_s
    p0 = float(base_res.power_set[0])
    scenarios = [
        ("CdA -0.010 m^2", replace(rider, cda=rider.cda - 0.01), np_target),
        ("CdA -0.005 m^2", replace(rider, cda=rider.cda - 0.005), np_target),
        ("Mass -1 kg", replace(rider, mass_kg=rider.mass_kg - 1.0), np_target),
        ("Mass -5 kg", replace(rider, mass_kg=rider.mass_kg - 5.0), np_target),
        ("+5 W NP", rider, np_target + 5.0),
        ("+10 W NP", rider, np_target + 10.0),
    ]
    for crr in crr_options:
        if abs(crr - rider.crr) > 1e-6:
            scenarios.append((f"Crr {crr:.4f}", replace(rider, crr=crr), np_target))
    rows = [{"scenario": "baseline", "time_s": base, "time": fmt_time(base), "saved_s": 0.0}]
    for label, rd, npv in scenarios:
        # every scenario is a small perturbation of the baseline: start from its constant power
        t = simulate_even(course, rd, npv, env, config, p_hint=p0 * npv / np_target).total_time_s
        rows.append({"scenario": label, "time_s": t, "time": fmt_time(t), "saved_s": base - t})
    return pd.DataFrame(rows)


# -------------------------------------------------------------- gearing glue
def target_cadence(power_w, rider: RiderProfile) -> np.ndarray:
    """Preferred cadence for a given power from the rider profile (cadence_flat at 200 W)."""
    return rider.cadence_flat + rider.cadence_per_100w * (np.asarray(power_w, float) - 200.0) / 100.0


def chainring_recommendation(result: SimResult, chainrings=range(48, 65), cassette=None,
                             weights: dict | None = None) -> pd.DataFrame:
    """Rank chainrings for this course using cycling_tools.gearing. Samples are weighted by seconds
    spent pedalling (coasting / braking excluded)."""
    from . import gearing

    rider = result.rider
    cassette = list(cassette) if cassette is not None else list(rider.cassette)
    pedalling = result.power_applied > 1.0
    speed = result.v_seg[pedalling]
    tgt = target_cadence(result.power_applied[pedalling], rider)
    df = gearing.evaluate_from_cadence_model(
        speed, tgt, result.seg_time[pedalling], list(chainrings), cassette,
        circumference_m=rider.tyre_circumference_m, weights=weights,
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def setup_recommendation(result: SimResult, setups, cassette=None, weights: dict | None = None) -> pd.DataFrame:
    """Rank 1x / 2x drivetrain setups (``gearing.Setup``) for this course, weighted by pedalling seconds.

    The 2x aero penalty (front derailleur) is an assumption inside ``gearing.DEFAULT_SETUP_WEIGHTS``.
    """
    from . import gearing

    rider = result.rider
    cassette = list(cassette) if cassette is not None else list(rider.cassette)
    pedalling = result.power_applied > 1.0
    tgt = target_cadence(result.power_applied[pedalling], rider)
    df = gearing.evaluate_setups_from_cadence_model(
        result.v_seg[pedalling], tgt, result.seg_time[pedalling], list(setups), cassette,
        circumference_m=rider.tyre_circumference_m, weights=weights,
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def gear_usage(result: SimResult, chainrings, cassette=None) -> pd.DataFrame:
    """Percent of pedalling time in each sprocket (columns) for each candidate chainring (rows), at target cadence."""
    from . import gearing

    rider = result.rider
    cassette = sorted(cassette if cassette is not None else rider.cassette)
    pedalling = result.power_applied > 1.0
    speed = result.v_seg[pedalling]
    tgt = target_cadence(result.power_applied[pedalling], rider)
    w = result.seg_time[pedalling]
    rows = {}
    for ring in chainrings:
        idx, _ = gearing.pick_sprocket_for_cadence(speed, ring, cassette, tgt, rider.tyre_circumference_m)
        rows[int(ring)] = np.bincount(idx, weights=w, minlength=len(cassette)) * 100.0 / w.sum()
    df = pd.DataFrame(rows, index=cassette).T
    df.index.name = "chainring"
    return df
