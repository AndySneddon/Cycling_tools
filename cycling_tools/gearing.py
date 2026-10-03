"""Best-chainring analysis for a 1x drivetrain.

Core idea: at every pedalling sample we know speed and cadence, hence the gear
ratio actually ridden (wheel revs per crank rev).  For a candidate chainring the
sprocket that would deliver that ratio is ``chainring / ratio``; we snap it to
the nearest real sprocket (log scale) and ask how much time would be spent in the
middle of the cassette (good: room to shift both ways) versus the ends (bad) or
outside the cassette altogether (spin-out / can't climb).

ASSUMPTION (important): the analysis assumes the rider keeps the same speed *and*
cadence when changing ring, i.e. cadence preference is unchanged.  In reality a
too-small ring may be pedalled faster, a too-big one ground slower; the
``cadence_err_*`` columns show how far the nearest real gear is from the
cadence that was actually ridden.

Score = mid4_pct - w_ends * ends_pct - w_oor * out_of_range_pct
(defaults in ``DEFAULT_WEIGHTS``; override via the ``weights`` dict).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:
    from numba import njit as _njit
except ImportError:  # pragma: no cover
    _njit = None

CASSETTES: dict[str, list[int]] = {
    "11-28 (11sp)": [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 28],
    "Ultegra 11-30 (12sp)": [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30],
    "Ultegra 11-32 (12sp)": [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 32],
    "Ultegra 11-34 (12sp)": [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 34],
    "Dura-Ace 11-28 (12sp)": [11, 12, 13, 14, 15, 16, 17, 19, 21, 23, 25, 28],
    "SRAM Force 10-33 (12sp)": [10, 11, 12, 13, 14, 15, 17, 19, 21, 24, 28, 33],
    "SRAM Force 10-36 (12sp)": [10, 11, 12, 13, 15, 17, 19, 21, 24, 28, 32, 36],
    "SRAM Rival 10-30 (12sp)": [10, 11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30],
    "Campagnolo 11-29 (12sp)": [11, 12, 13, 14, 15, 17, 19, 21, 23, 25, 27, 29],
    "Shimano 105 11-34 (12sp)": [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 34],
}

DEFAULT_CIRCUMFERENCE_M = 2.13

DEFAULT_WEIGHTS = {"w_ends": 0.25, "w_oor": 2.0}

MIN_SPEED_MPS = 1.0


def parse_cassette(text: str) -> list[int]:
    """'11,12,13' / '11 12 13' / '11-12-13' -> sorted list of ints (smallest first)."""
    nums = [int(n) for n in re.findall(r"\d+", str(text))]
    if len(nums) < 2:
        raise ValueError("A cassette needs at least two sprockets, e.g. '11,12,13,14,15,17,19,21,24,27,30'.")
    return sorted(nums)


# ----------------------------------------------------------------- basic maths
def gear_ratio_from(speed_mps, cadence_rpm, circumference_m=DEFAULT_CIRCUMFERENCE_M) -> np.ndarray:
    """Wheel revolutions per crank revolution (NaN where cadence <= 0)."""
    s = np.asarray(speed_mps, dtype=float)
    c = np.asarray(cadence_rpm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = s * 60.0 / (c * circumference_m)
    return np.where(c > 0, r, np.nan)


def cadence_for(speed_mps, chainring, sprocket, circumference_m=DEFAULT_CIRCUMFERENCE_M) -> np.ndarray:
    """Cadence (rpm) needed to hold ``speed_mps`` in chainring/sprocket."""
    s = np.asarray(speed_mps, dtype=float)
    ratio = np.asarray(chainring, dtype=float) / np.asarray(sprocket, dtype=float)
    return s * 60.0 / (circumference_m * ratio)


def pick_sprocket_for_cadence(speed_mps, chainring, cassette, target_cadence,
                              circumference_m=DEFAULT_CIRCUMFERENCE_M):
    """Sprocket whose cadence is closest (log scale) to target.

    Returns (sprocket_idx, resulting_cadence); idx indexes into ``cassette`` as given.
    """
    cas = np.asarray(cassette, dtype=float)
    s = np.atleast_1d(np.asarray(speed_mps, dtype=float))
    tgt = np.asarray(target_cadence, dtype=float)
    scalar = np.ndim(speed_mps) == 0 and tgt.ndim == 0
    cad = cadence_for(s[:, None], chainring, cas[None, :], circumference_m)  # (n, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(np.log(cad / np.reshape(tgt, (-1, 1)) if tgt.ndim else cad / tgt))
    err = np.where(np.isfinite(err), err, np.inf)
    idx = np.argmin(err, axis=1)
    res = cad[np.arange(len(s)), idx]
    return (idx[0], res[0]) if scalar else (idx, res)


# ------------------------------------------------------------------ internals
def _wq(x, w, q):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    x, w = x[m], w[m]
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    return float(x[min(np.searchsorted(cw, q), len(x) - 1)])


def _wmean(x, w):
    m = np.isfinite(x)
    return float(np.average(x[m], weights=w[m])) if m.any() and w[m].sum() > 0 else np.nan


def _classify(needed_sprocket: np.ndarray, cassette: list[int]):
    """Snap continuous needed sprocket to a real one. Returns (idx, out_of_range mask)."""
    cas = np.asarray(sorted(cassette), dtype=float)
    lc = np.log(cas)
    ln = np.log(np.where(needed_sprocket > 0, needed_sprocket, np.nan))
    idx = np.abs(ln[:, None] - lc[None, :]).argmin(axis=1)
    lo = lc[0] - 0.5 * (lc[1] - lc[0])
    hi = lc[-1] + 0.5 * (lc[-1] - lc[-2])
    oor = (ln < lo) | (ln > hi)
    return idx, oor


def _band_sets(n: int):
    mid2 = {n // 2 - 1, n // 2} if n % 2 == 0 else {n // 2, n // 2 + 1} if n > 3 else {n // 2}
    mid4 = ({n // 2 - 2, n // 2 - 1, n // 2, n // 2 + 1} if n % 2 == 0
            else {n // 2 - 1, n // 2, n // 2 + 1} | {n // 2 + 2})
    mid4 = {i for i in mid4 if 0 <= i < n}
    mid2 = {i for i in mid2 if 0 <= i < n}
    ends = {i for i in (0, 1, n - 2, n - 1) if 0 <= i < n}
    return mid2, mid4, ends


def _band_weights(n: int):
    """Per-sprocket weights (mid2, mid4, ends) with the middle bands centred on the cassette.

    Even n: whole sprockets, as in :func:`_band_sets`. Odd n has no symmetric 2 or 4 whole sprockets, so the
    centre sprocket counts fully and its outer neighbours count half (mid2 = 1 + 2x0.5, mid4 = 3 + 2x0.5):
    the same total width, centred on sprocket (n-1)/2 instead of drifting half a sprocket to the big end.
    """
    mid2, mid4 = np.zeros(n), np.zeros(n)
    c = n // 2
    if n % 2 == 0:
        mid2[[c - 1, c]] = 1.0
        mid4[max(c - 2, 0):c + 2] = 1.0
    else:
        mid2[c] = 1.0
        mid4[max(c - 1, 0):c + 2] = 1.0
        for arr, k in ((mid2, c + 1), (mid2, c - 1), (mid4, c + 2), (mid4, c - 2)):
            if 0 <= k < n:
                arr[k] = 0.5
    ends = np.zeros(n)
    ends[[i for i in (0, 1, n - 2, n - 1) if 0 <= i < n]] = 1.0
    return mid2, mid4, ends


def _prep(speed, w):
    s = np.asarray(speed, float)
    w = np.ones_like(s) if w is None else np.asarray(w, float)
    return s, w


def _row(ring, idx, oor, w, cas_n, err, weights):
    tot = w.sum()
    b2, b4, be = _band_weights(cas_n)
    inr = ~oor

    def pct(sel):
        return 100.0 * w[sel].sum() / tot if tot > 0 else np.nan

    def pctw(wt):  # weighted band membership
        return 100.0 * (w * wt[idx] * inr).sum() / tot if tot > 0 else np.nan

    row = dict(
        chainring=int(ring), mid2_pct=pctw(b2), mid4_pct=pctw(b4), ends_pct=pctw(be),
        ends_only_pct=pctw(be * (b4 == 0)),
        out_of_range_pct=pct(oor),
        cadence_err_mean=_wmean(np.abs(err), w), cadence_err_p90=_wq(np.abs(err), w, 0.9),
    )
    row["score"] = row["mid4_pct"] - weights["w_ends"] * row["ends_pct"] - weights["w_oor"] * row["out_of_range_pct"]
    return row


def _weights(weights):
    out = dict(DEFAULT_WEIGHTS)
    out.update(weights or {})
    return out


def _clean(speed, cadence, weights_s, segments=None):
    """Drop unusable samples. With ``segments`` (one id per input sample) also returns the surviving ids."""
    s = np.asarray(speed, float)
    c = np.asarray(cadence, float)
    w = np.ones_like(s) if weights_s is None else np.asarray(weights_s, float)
    ok = np.isfinite(s) & np.isfinite(c) & np.isfinite(w) & (s > 0) & (c > 0)
    if segments is None:
        return s[ok], c[ok], w[ok]
    return s[ok], c[ok], w[ok], np.asarray(segments)[ok]


# ------------------------------------------------------------------ evaluation
def evaluate_chainrings(speed_mps, cadence_rpm, weights_s, chainrings, cassette,
                        circumference_m=DEFAULT_CIRCUMFERENCE_M, weights=None) -> pd.DataFrame:
    """Rank chainrings from observed speed/cadence (see module docstring for assumptions)."""
    wts = _weights(weights)
    cas = sorted(cassette)
    s, c, w = _clean(speed_mps, cadence_rpm, weights_s)
    ratio = gear_ratio_from(s, c, circumference_m)
    rows = []
    for ring in chainrings:
        need = ring / ratio
        idx, oor = _classify(need, cas)
        achieved = cadence_for(s, ring, np.asarray(cas, float)[idx], circumference_m)
        rows.append(_row(ring, idx, oor, w, len(cas), achieved - c, wts))
    return pd.DataFrame(rows)


def evaluate_from_cadence_model(speed_mps, target_cadence, weights_s, chainrings, cassette,
                                circumference_m=DEFAULT_CIRCUMFERENCE_M, weights=None) -> pd.DataFrame:
    """As :func:`evaluate_chainrings`, but target cadence is the rider's preferred cadence per sample."""
    wts = _weights(weights)
    cas = sorted(cassette)
    s = np.atleast_1d(np.asarray(speed_mps, float))
    t = np.broadcast_to(np.asarray(target_cadence, float), s.shape)
    s, t, w = _clean(s, t, weights_s if weights_s is not None else np.ones_like(s))
    rows = []
    for ring in chainrings:
        need = ring / gear_ratio_from(s, t, circumference_m)
        _, oor = _classify(need, cas)
        idx, achieved = pick_sprocket_for_cadence(s, ring, cas, t, circumference_m)
        rows.append(_row(ring, np.asarray(idx), oor, w, len(cas), achieved - t, wts))
    return pd.DataFrame(rows)


def usage_matrix(speed, cadence, weights_s, chainrings, cassette,
                 circ=DEFAULT_CIRCUMFERENCE_M) -> pd.DataFrame:
    """% of time per chainring (rows) x sprocket (columns). Out-of-range time is in the 'beyond' column(s)."""
    cas = sorted(cassette)
    s, c, w = _clean(speed, cadence, weights_s)
    ratio = gear_ratio_from(s, c, circ)
    tot = w.sum()
    out = {}
    for ring in chainrings:
        idx, oor = _classify(ring / ratio, cas)
        hist = np.bincount(idx[~oor], weights=w[~oor], minlength=len(cas)) * 100.0 / tot
        out[int(ring)] = hist
    df = pd.DataFrame(out, index=cas).T
    df.index.name = "chainring"
    df.columns.name = "sprocket"
    return df


# ------------------------------------------------------------- data preparation
def _grade(df: pd.DataFrame, half_window_s: int = 10) -> np.ndarray:
    alt = df["alt"].rolling(9, center=True, min_periods=1).median().to_numpy()
    dist = df["dist"].to_numpy()
    n = len(df)
    i0 = np.clip(np.arange(n) - half_window_s, 0, n - 1)
    i1 = np.clip(np.arange(n) + half_window_s, 0, n - 1)
    dd = dist[i1] - dist[i0]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (alt[i1] - alt[i0]) / dd * 100.0
    g = np.where(dd > 20, g, np.nan)
    return np.clip(g, -25, 25)


MIN_POWER_COVERAGE = 0.05  # a lap with less than this share of valid power readings counts as "no power"
SEGMENT_GAP_S = 5          # samples further apart than this (s) belong to different ride segments
_SAMPLE_COLS = ["speed", "cadence", "power", "grade", "weight_s", "ride", "seg"]


def lap_power_coverage(ride, *, min_cadence=40) -> pd.DataFrame:
    """Per lap: pedalling candidate samples and the % of them that have a power reading (lap, n, power_pct)."""
    d = ride.df
    cand = (d["speed"] >= MIN_SPEED_MPS) & (d["cadence"] > min_cadence) & ~d["gap"]
    x = pd.DataFrame({"lap": d["lap"][cand], "has": d["power"][cand].notna()})
    out = x.groupby("lap")["has"].agg(n="size", power_pct=lambda v: 100.0 * v.mean()).reset_index()
    return out


def prepare_gearing_samples(rides, *, min_power=0, min_cadence=40, laps=None,
                            terrain="all", power_range=None, require_power=True) -> pd.DataFrame:
    """Pedalling samples from one or more rides -> DataFrame(speed, cadence, power, grade, weight_s, ride, seg).

    laps: {ride.name: [lap numbers]} to restrict a ride to those laps (missing key = whole ride).
    terrain: 'all' | 'climb' (grade > 2%) | 'flat' (-2..2%) | 'descent' (< -2%).
    power_range: (lo, hi) watts inclusive (None bound allowed).
    require_power: samples without a power reading are dropped (a NaN never passes ``min_power`` even when 0, so
        this used to happen silently). Selected laps that have no power at all (e.g. the swim/run legs of a
        multisport file) are listed in ``df.attrs['no_power_laps']`` as (ride, lap, n_samples) so the caller can
        say so. With ``require_power=False`` power-less samples are kept (power filters then ignore them).
    seg: id of the contiguous ride segment (breaks at ride changes and at gaps > SEGMENT_GAP_S in the original
        1 Hz record), used to count front shifts only within segments.
    """
    parts, no_power = [], []
    for ri, r in enumerate(rides):
        d = r.df
        g = _grade(d) if "alt" in d and d["alt"].notna().any() else np.full(len(d), np.nan)
        pw = d["power"]
        okp = (pw >= min_power) if require_power else ((pw >= min_power) | pw.isna())
        m = (d["speed"] >= MIN_SPEED_MPS) & (d["cadence"] > min_cadence) & okp & ~d["gap"]
        sel_laps = None
        if laps and r.name in laps and laps[r.name] is not None:
            sel_laps = list(laps[r.name])
            m &= d["lap"].isin(sel_laps)
        cov = lap_power_coverage(r, min_cadence=min_cadence)
        for _, row in cov[cov["power_pct"] < 100 * MIN_POWER_COVERAGE].iterrows():
            if sel_laps is None or int(row["lap"]) in sel_laps:
                no_power.append((r.name, int(row["lap"]), int(row["n"])))
        out = pd.DataFrame({"speed": d["speed"], "cadence": d["cadence"], "power": pw,
                            "grade": g, "weight_s": 1.0, "ride": r.name,
                            "pos": np.arange(len(d)) + ri * 10**9})[m.to_numpy()]
        parts.append(out)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=_SAMPLE_COLS[:-1] + ["pos"])
    df = df.dropna(subset=["speed", "cadence"])
    if terrain == "climb":
        df = df[df["grade"] > 2]
    elif terrain == "flat":
        df = df[df["grade"].between(-2, 2)]
    elif terrain == "descent":
        df = df[df["grade"] < -2]
    elif terrain != "all":
        raise ValueError("terrain must be 'all', 'climb', 'flat' or 'descent'")
    if power_range is not None:
        lo, hi = power_range
        nan_ok = df["power"].isna() & (not require_power)
        if lo is not None:
            df = df[(df["power"] >= lo) | nan_ok]
            nan_ok = df["power"].isna() & (not require_power)
        if hi is not None:
            df = df[(df["power"] <= hi) | nan_ok]
    pos = df["pos"].to_numpy()
    seg = np.r_[0, np.cumsum(np.diff(pos) > SEGMENT_GAP_S)] if len(pos) else np.zeros(0, int)
    df = df.assign(seg=seg).drop(columns="pos").reset_index(drop=True)
    df.attrs["no_power_laps"] = [] if not require_power else no_power
    return df


if _njit is not None:
    @_njit(cache=True, nogil=True)
    def _theil_sen_nb(x, y):
        """Median of all pairwise slopes with x_i > x_j (what scipy.stats.theilslopes uses), via quickselect."""
        n = x.shape[0]
        m = 0
        for i in range(n):
            for j in range(n):
                if x[i] - x[j] > 0:
                    m += 1
        sl = np.empty(m)
        k = 0
        for i in range(n):
            for j in range(n):
                dx = x[i] - x[j]
                if dx > 0:
                    sl[k] = (y[i] - y[j]) / dx
                    k += 1
        return np.median(sl) if m else np.nan


def _theil_sen(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of ``scipy.stats.theilslopes(y, x)`` ('separate' intercept) without sorting all slopes
    or computing the confidence interval, which this app never uses."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if _njit is None:
        from scipy.stats import theilslopes
        r = theilslopes(y, x)
        return float(r[0]), float(r[1])
    slope = float(_theil_sen_nb(np.ascontiguousarray(x), np.ascontiguousarray(y)))
    return slope, float(np.median(y) - slope * np.median(x))


def fit_cadence_model(df: pd.DataFrame) -> dict:
    """Robust (Theil-Sen) linear fit: cadence = cadence_flat + cadence_per_100w * (power - ref_power) / 100.

    ``ref_power`` is the median pedalling power of the samples (cadence_flat is the cadence at that power).
    Slope is clamped near zero when data are too few/narrow.
    """
    d = df.dropna(subset=["power", "cadence"])
    d = d[d["power"] > 0]
    if len(d) < 30:
        return dict(cadence_flat=float(d["cadence"].median()) if len(d) else 88.0,
                    cadence_per_100w=0.0, ref_power=float(d["power"].median()) if len(d) else 200.0, n=len(d))
    d = d.sample(min(len(d), 4000), random_state=0)
    ref = float(df["power"][df["power"] > 0].median())
    slope, icpt = _theil_sen(d["cadence"].to_numpy(), d["power"].to_numpy() - ref)
    return dict(cadence_flat=float(icpt), cadence_per_100w=float(slope * 100.0), ref_power=ref, n=int(len(d)))


# =============================================================================
# 1x / 2x setups
# =============================================================================
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class Setup:
    """A drivetrain front end: one ring (1x) or two (2x). Ring order is preserved; ring_idx indexes it."""

    chainrings: tuple
    name: str = ""

    def __post_init__(self):
        rings = tuple(int(r) for r in self.chainrings)
        if len(rings) not in (1, 2) or any(r < 20 for r in rings):
            raise ValueError("A setup has one or two chainrings, e.g. (58,) or (56, 42).")
        object.__setattr__(self, "chainrings", rings)
        if not self.name:
            object.__setattr__(self, "name", "/".join(str(r) for r in rings))

    @property
    def is_2x(self) -> bool:
        return len(self.chainrings) == 2

    @property
    def label(self) -> str:
        return self.name

    @property
    def big_idx(self) -> int:
        return int(np.argmax(self.chainrings))

    @property
    def small_idx(self) -> int:
        return int(np.argmin(self.chainrings))


def parse_setup(text: str) -> Setup:
    """'58' -> 1x58; '56/42' (or '56-42', '56 42') -> 2x."""
    nums = [int(n) for n in re.findall(r"\d+", str(text))]
    return Setup(tuple(nums))


DEFAULT_1X_RINGS = list(range(50, 67, 2))
DEFAULT_2X_PAIRS = ["56/42", "54/40", "55/42", "54/42", "52/36", "53/39", "58/44", "56/44", "55/40", "52/38"]


def default_setups(mode: str = "both") -> list[Setup]:
    """Candidate setups. mode: '1x' | '2x' | 'both'."""
    out: list[Setup] = []
    if mode in ("1x", "both"):
        out += [Setup((r,)) for r in DEFAULT_1X_RINGS]
    if mode in ("2x", "both"):
        out += [parse_setup(p) for p in DEFAULT_2X_PAIRS]
    return out


DEFAULT_SETUP_WEIGHTS = {
    **DEFAULT_WEIGHTS,
    "w_cross": 0.25,          # score points per % of time cross-chained
    "w_shift": 0.05,          # score points per front shift per hour
    "w_aero": 3.0,            # score points per watt of aero penalty
    "aero_delta_cda": 0.002,  # ASSUMPTION: m^2 CdA cost of a front derailleur + 2nd ring (set to 0 to ignore)
    "rho": 1.2,               # air density for the aero penalty (kg/m^3)
    "ref_speed_mps": None,    # None -> use the time-weighted mean of v^3 of the samples
    "w_cad": 0.0,             # score points per rpm of mean cadence error (off by default)
    "n_cross": 3,             # outer sprockets treated as cross-chained on the 'wrong' ring
    "cross_margin": 0.04,     # log-cadence cost of a cross-chained gear when choosing (~half a gear step)
    "small_ring_bias": 0.05,  # log-cadence cost of using the small ring (~ a sprocket step, so the choice does not flicker)
    "min_dwell_s": 10.0,      # front shifts shorter than this are treated as flicker
}


def _setup_weights(weights):
    out = dict(DEFAULT_SETUP_WEIGHTS)
    out.update(weights or {})
    return out


def cross_chain_mask(setup: Setup, n_sprockets: int, n_cross: int = 3) -> np.ndarray:
    """bool (n_rings, n_sprockets): True where the combo is cross-chained (always False for 1x)."""
    m = np.zeros((len(setup.chainrings), n_sprockets), bool)
    if setup.is_2x:
        nc = min(int(n_cross), n_sprockets // 2)
        if nc > 0:
            m[setup.big_idx, n_sprockets - nc:] = True
            m[setup.small_idx, :nc] = True
    return m


def _assign_gears(speed, target, setup: Setup, cassette, circ, wts) -> dict:
    cas = np.asarray(sorted(cassette), float)
    K = len(cas)
    s = np.atleast_1d(np.asarray(speed, float))
    t = np.broadcast_to(np.asarray(target, float), s.shape)
    rings = np.asarray(setup.chainrings, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cad = s[:, None, None] * 60.0 / (circ * (rings[None, :, None] / cas[None, None, :]))
        cost = np.abs(np.log(cad / t[:, None, None]))
    cost = np.where(np.isfinite(cost), cost, np.inf)
    mask = cross_chain_mask(setup, K, wts["n_cross"])
    if setup.is_2x:
        cost = cost + mask[None] * wts["cross_margin"]
        cost[:, setup.small_idx, :] += wts["small_ring_bias"]
    flat = cost.reshape(len(s), -1).argmin(axis=1) if len(s) else np.zeros(0, int)
    ri, si = np.divmod(flat, K)
    achieved = cad[np.arange(len(s)), ri, si]
    # out of range: needed ratio beyond the full hardest/easiest gear by > half a gear step
    lc = np.log(cas)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(s * 60.0 / (t * circ))
    hard = np.log(rings.max() / cas[0]) + 0.5 * (lc[1] - lc[0])
    easy = np.log(rings.min() / cas[-1]) - 0.5 * (lc[-1] - lc[-2])
    oor = (lr > hard) | (lr < easy)
    return dict(ring_idx=ri, sprocket_idx=si, cadence=achieved, crossed=mask[ri, si], oor=oor)


def pick_gear_for_setup(speed, setup, cassette, target_cadence, circ=DEFAULT_CIRCUMFERENCE_M, weights=None):
    """(ring_idx, sprocket_idx, cadence) for the gear closest (log) to target cadence, avoiding cross-chaining.

    ring_idx indexes ``setup.chainrings``; sprocket_idx indexes the *ascending-sorted* cassette. 1x reduces to
    :func:`pick_sprocket_for_cadence`. Scalars in -> scalars out.
    """
    if not isinstance(setup, Setup):
        setup = parse_setup(setup)
    scalar = np.ndim(speed) == 0 and np.ndim(target_cadence) == 0
    a = _assign_gears(speed, target_cadence, setup, cassette, circ, _setup_weights(weights))
    if scalar:
        return int(a["ring_idx"][0]), int(a["sprocket_idx"][0]), float(a["cadence"][0])
    return a["ring_idx"], a["sprocket_idx"], a["cadence"]


def _count_front_shifts_py(ring_idx, weights_s, min_dwell_s):
    """Reference implementation (O(runs^2)); kept for the no-numba fallback and for the tests."""
    ring_idx = np.asarray(ring_idx)
    if len(ring_idx) == 0:
        return 0
    brk = np.flatnonzero(np.diff(ring_idx) != 0) + 1
    starts = np.r_[0, brk]
    vals = ring_idx[starts].tolist()
    durs = np.add.reduceat(np.asarray(weights_s, float), starts).tolist()
    runs = list(zip(vals, durs))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (v, d) in enumerate(runs):
            if d < min_dwell_s:
                j = i - 1 if i > 0 else i + 1
                runs[j] = (runs[j][0], runs[j][1] + d)
                del runs[i]
                merged = [runs[0]]
                for v2, d2 in runs[1:]:
                    if v2 == merged[-1][0]:
                        merged[-1] = (v2, merged[-1][1] + d2)
                    else:
                        merged.append((v2, d2))
                runs = merged
                changed = True
                break
    return max(len(runs) - 1, 0)


if _njit is not None:
    @_njit(cache=True, nogil=True)
    def _shifts_nb(vals, durs, min_dwell):
        """Same merging rule as the reference, on a linked list with a moving scan pointer (O(runs))."""
        n = vals.shape[0]
        prv = np.arange(-1, n - 1)
        nxt = np.arange(1, n + 1)
        nxt[n - 1] = -1
        d = durs.copy()
        alive = n
        head = 0
        cur = 0
        while alive > 1 and cur != -1:
            if d[cur] < min_dwell:
                j = prv[cur] if prv[cur] != -1 else nxt[cur]
                d[j] += d[cur]
                p, q = prv[cur], nxt[cur]
                if p != -1:
                    nxt[p] = q
                else:
                    head = q
                if q != -1:
                    prv[q] = p
                alive -= 1
                if p != -1 and q != -1 and vals[p] == vals[q]:  # neighbours now touch and share a ring: merge
                    d[p] += d[q]
                    r = nxt[q]
                    nxt[p] = r
                    if r != -1:
                        prv[r] = p
                    alive -= 1
                cur = p if p != -1 else head
            else:
                cur = nxt[cur]
        return alive - 1


def count_front_shifts(ring_idx: np.ndarray, weights_s: np.ndarray, min_dwell_s: float = 10.0) -> int:
    """Ring changes along a time-ordered sequence; runs shorter than min_dwell_s are absorbed (flicker)."""
    ring_idx = np.asarray(ring_idx)
    if len(ring_idx) == 0:
        return 0
    if _njit is None:
        return _count_front_shifts_py(ring_idx, weights_s, min_dwell_s)
    brk = np.flatnonzero(np.diff(ring_idx) != 0) + 1
    starts = np.r_[0, brk]
    vals = ring_idx[starts].astype(np.int64)
    durs = np.add.reduceat(np.asarray(weights_s, float), starts)
    return int(_shifts_nb(vals, durs, float(min_dwell_s)))


def count_front_shifts_segments(ring_idx, weights_s, segments, min_dwell_s: float = 10.0) -> int:
    """Sum of :func:`count_front_shifts` over contiguous segments (no shift is counted across a segment break)."""
    ring_idx = np.asarray(ring_idx)
    if segments is None or len(ring_idx) == 0:
        return count_front_shifts(ring_idx, weights_s, min_dwell_s)
    seg = np.asarray(segments)
    edges = np.r_[0, np.flatnonzero(np.diff(seg) != 0) + 1, len(seg)]
    w = np.asarray(weights_s, float)
    return sum(count_front_shifts(ring_idx[a:b], w[a:b], min_dwell_s) for a, b in zip(edges[:-1], edges[1:]))


def _evaluate_setups(s, t, w, setups, cassette, circ, weights, ref_cadence, segments=None) -> pd.DataFrame:
    wts = _setup_weights(weights)
    cas = sorted(cassette)
    K = len(cas)
    b2, b4, be = _band_weights(K)
    tot = w.sum()
    hours = tot / 3600.0
    rows = []
    for st in setups:
        if not isinstance(st, Setup):
            st = parse_setup(st)
        a = _assign_gears(s, t, st, cas, circ, wts)
        inr = ~a["oor"]
        sp = a["sprocket_idx"]

        def pct(sel):
            return 100.0 * w[sel].sum() / tot if tot > 0 else np.nan

        def pctw(wt, sel):  # weighted band membership
            return 100.0 * (w * wt[sp] * sel).sum() / tot if tot > 0 else np.nan

        err = np.abs(a["cadence"] - t)
        rmax, rmin = max(st.chainrings), min(st.chainrings)
        top_ratio, bot_ratio = rmax / cas[0], rmin / cas[-1]
        if st.is_2x:
            v3 = (ref := wts["ref_speed_mps"]) ** 3 if wts["ref_speed_mps"] else (
                np.average(s ** 3, weights=w) if tot > 0 else 0.0)
            aero_w = 0.5 * wts["rho"] * wts["aero_delta_cda"] * v3
            shifts = (count_front_shifts_segments(a["ring_idx"], w, segments, wts["min_dwell_s"]) / hours
                      if hours > 0 else 0.0)
        else:
            aero_w, shifts = 0.0, 0.0
        row = dict(
            setup=st.label, n_rings=len(st.chainrings), big_ring=rmax,
            mid2_pct=pctw(b2, inr & ~a["crossed"]),
            mid4_pct=pctw(b4, inr & ~a["crossed"]),
            ends_pct=pctw(be, inr),
            ends_only_pct=pctw(be * (b4 == 0), inr & ~a["crossed"]),  # disjoint from cross-chained, for the bars
            out_of_range_pct=pct(a["oor"]),
            cross_chain_pct=pct(a["crossed"] & inr),
            cadence_err_mean=_wmean(err, w), cadence_err_p90=_wq(err, w, 0.9),
            front_shifts_per_hour=shifts, aero_penalty_w=aero_w,
            top_speed_kmh=top_ratio * ref_cadence * circ / 60.0 * 3.6,
            bottom_speed_kmh=bot_ratio * ref_cadence * circ / 60.0 * 3.6,
            gear_range_pct=100.0 * (top_ratio / bot_ratio - 1.0),
        )
        row["score"] = (row["mid4_pct"] - wts["w_ends"] * row["ends_pct"] - wts["w_oor"] * row["out_of_range_pct"]
                        - wts["w_cross"] * row["cross_chain_pct"] - wts["w_shift"] * shifts
                        - wts["w_aero"] * aero_w - wts["w_cad"] * row["cadence_err_mean"])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_setups(speed_mps, cadence_rpm, weights_s, setups, cassette,
                    circumference_m=DEFAULT_CIRCUMFERENCE_M, weights=None, ref_cadence=90.0,
                    segments=None) -> pd.DataFrame:
    """One row per setup (1x or 2x), from observed speed/cadence (time-ordered samples).

    For 1x the shared columns/score equal :func:`evaluate_chainrings`. For 2x the gear for each sample is the
    (ring, sprocket) with cadence closest to the observed cadence, cross-chained combos penalised, big ring
    preferred when equivalent. Assumes the rider keeps their cadence when changing setup. The 2x aero penalty
    (``aero_delta_cda`` m^2, default 0.002) is an ASSUMPTION; set weights={'aero_delta_cda': 0} to ignore it.
    ``weights`` keys: see DEFAULT_SETUP_WEIGHTS. ``segments``: optional contiguous-segment id per input sample
    (``prepare_gearing_samples`` 'seg' column); front shifts are counted within segments only.
    """
    s, c, w, sg = _clean(speed_mps, cadence_rpm, weights_s,
                         np.zeros(len(np.atleast_1d(speed_mps)), int) if segments is None else segments)
    return _evaluate_setups(s, c, w, setups, cassette, circumference_m, weights, ref_cadence, sg)


def evaluate_setups_from_cadence_model(speed_mps, target_cadence, weights_s, setups, cassette,
                                       circumference_m=DEFAULT_CIRCUMFERENCE_M, weights=None,
                                       ref_cadence=90.0) -> pd.DataFrame:
    """As :func:`evaluate_setups` but the cadence is the rider's preferred cadence per sample (course simulation)."""
    s = np.atleast_1d(np.asarray(speed_mps, float))
    t = np.broadcast_to(np.asarray(target_cadence, float), s.shape)
    s, t, w = _clean(s, t, weights_s if weights_s is not None else np.ones_like(s))
    return _evaluate_setups(s, t, w, setups, cassette, circumference_m, weights, ref_cadence)


def usage_matrix_setup(speed, cadence, weights_s, setup, cassette, circ=DEFAULT_CIRCUMFERENCE_M,
                       weights=None) -> pd.DataFrame:
    """% time per (ring x sprocket) combo for one setup; rows are ring teeth in setup order."""
    wts = _setup_weights(weights)
    cas = sorted(cassette)
    s, c, w = _clean(speed, cadence, weights_s)
    a = _assign_gears(s, c, setup, cas, circ, wts)
    ok = ~a["oor"]
    R = len(setup.chainrings)
    m = np.zeros((R, len(cas)))
    np.add.at(m, (a["ring_idx"][ok], a["sprocket_idx"][ok]), w[ok])
    df = pd.DataFrame(m * 100.0 / w.sum(), index=list(setup.chainrings), columns=cas)
    df.index.name = "chainring"
    df.columns.name = "sprocket"
    return df


def gear_sequence(speed, cadence, weights_s, setup, cassette, circ=DEFAULT_CIRCUMFERENCE_M,
                  weights=None) -> pd.DataFrame:
    """Per-sample chosen gear (t_s cumulative, ring, sprocket, crossed, oor) for time-strip plots."""
    wts = _setup_weights(weights)
    s, c, w = _clean(speed, cadence, weights_s)
    a = _assign_gears(s, c, setup, sorted(cassette), circ, wts)
    return pd.DataFrame({"t_s": np.cumsum(w), "ring": np.asarray(setup.chainrings)[a["ring_idx"]],
                         "sprocket": np.asarray(sorted(cassette))[a["sprocket_idx"]],
                         "crossed": a["crossed"], "oor": a["oor"]})


def recommend(ranked: pd.DataFrame, tol: float = 1.0, edge_group_min: int = 3) -> dict:
    """Plateau-aware pick from a score-sorted ``evaluate_setups`` table (best first).

    Setups of the winner's kind (1x or 2x) scoring within ``tol`` points of the best are a statistical tie; the
    pick is the smallest big ring among them (ties by smallest small ring). ``edge`` is 'low'/'high' when the tied
    set touches the smallest/largest candidate of that kind (1x only; needs >= ``edge_group_min`` candidates),
    i.e. the true optimum may lie outside the range searched.
    Returns dict(best, pick, near (list of setup labels, small to large), tol, edge, edge_ring).
    """
    best = ranked.iloc[0]
    grp = ranked[ranked["n_rings"] == best["n_rings"]]
    near = grp[grp["score"] >= best["score"] - tol].copy()
    near["_small"] = [min(parse_setup(s).chainrings) for s in near["setup"]]
    near = near.sort_values(["big_ring", "_small"])
    pick = near.iloc[0]
    edge, edge_ring = None, None
    if best["n_rings"] == 1 and len(grp) >= edge_group_min:
        lo, hi = grp["big_ring"].min(), grp["big_ring"].max()
        if near["big_ring"].max() == hi:
            edge, edge_ring = "high", int(hi)
        elif near["big_ring"].min() == lo:
            edge, edge_ring = "low", int(lo)
    return dict(best=best, pick=pick, near=list(near["setup"]), tol=tol, edge=edge, edge_ring=edge_ring)
