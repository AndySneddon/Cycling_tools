"""Pacing optimiser: minimise race time at a fixed normalised power using block power targets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os

from scipy.optimize import brentq, minimize
from scipy.optimize._numdiff import approx_derivative

from .course import Course
from .profile import RiderProfile
from .simulate import (Environment, SimConfig, SimResult, _np_from_p1, _p1hz_numba, _rolling_numba, _run, fmt_time,
                       np_window, simulate, simulate_even, splits_table, window_starts)

# Rolling-window power caps (window in seconds -> % of the NP target). Pure NP only penalises variability on a
# ~30 s scale, so on its own the optimiser happily asks for 1-min surges at 120 %+ of NP that no rider can hold.
DEFAULT_CAPS = {60: 110.0, 300: 106.0}


@dataclass
class OptResult:
    course: Course
    np_target: float
    even: SimResult
    optimised: SimResult
    block_edges_m: np.ndarray
    block_power: np.ndarray       # crank W per block
    bound_pct: float
    smooth: float
    success: bool                 # False: could not beat even pacing, so ``optimised`` is the even plan
    message: str
    n_evals: int = 0
    converged: bool = True        # solver reached a stationary point (not just the iteration limit)
    n_iter: int = 0
    power_caps: dict = field(default_factory=dict)   # window s -> % of NP
    cap_excess_w: dict = field(default_factory=dict)  # window s -> W above the cap in the final plan (<= 0 ok)
    block_m: float = 1000.0
    saved_equal_np120_s: float = 0.0   # time saved if the plan is scaled to the same 120 s NP as even pacing

    @property
    def time_saved_s(self) -> float:
        return self.even.total_time_s - self.optimised.total_time_s

    @property
    def power_seg(self) -> np.ndarray:
        return self.optimised.power_set

    def block_table(self) -> pd.DataFrame:
        c = self.course
        mids = 0.5 * (self.block_edges_m[:-1] + self.block_edges_m[1:])
        grade = np.interp(mids, c.seg_mid_m, c.grade)
        hw = np.interp(mids, c.seg_mid_m, self.optimised.headwind)
        return pd.DataFrame({
            "start_km": self.block_edges_m[:-1] / 1000.0, "end_km": self.block_edges_m[1:] / 1000.0,
            "target_power_w": self.block_power, "pct_of_np": 100.0 * self.block_power / self.np_target,
            "grade_pct": grade * 100.0, "headwind_ms": hw,
        })

    def km_table(self, every_m: float = 1000.0) -> pd.DataFrame:
        """Per-km power targets with predicted splits (even vs optimised) - the race-day cheat sheet."""
        opt = splits_table(self.optimised, every_m)
        ev = splits_table(self.even, every_m)
        out = pd.DataFrame({
            "split": opt["split"], "end_km": opt["end_km"],
            "target_power_w": opt["power_w"].round(0),
            "pct_of_np": (100.0 * opt["power_w"] / self.np_target).round(0),
            "even_power_w": ev["power_w"].round(0),
            "pred_speed_kmh": opt["speed_kmh"].round(1),
            "pred_split": opt["split_time"], "pred_cum_time": opt["cum_time"],
            "grade_net_m": opt["net_elev_m"].round(1), "headwind_ms": opt["headwind_ms"].round(1),
        })
        return out

    def to_csv(self, every_m: float = 1000.0) -> str:
        return self.km_table(every_m).to_csv(index=False)

    @property
    def even_stats(self) -> dict:
        return self.even.power_stats()

    @property
    def opt_stats(self) -> dict:
        return self.optimised.power_stats()

    def stats_table(self) -> pd.DataFrame:
        """Even vs optimised: average power, NP, 120 s NP and peak 1 / 5 / 20 min power (W)."""
        e, o = self.even_stats, self.opt_stats
        rows = {"Average power": "avg_power_w", "NP (30 s)": "np_w", "NP (120 s)": "np120_w",
                "Peak 1 min": "peak_1min_w", "Peak 5 min": "peak_5min_w", "Peak 20 min": "peak_20min_w"}
        return pd.DataFrame({"metric": list(rows), "even_w": [e[k] for k in rows.values()],
                             "optimised_w": [o[k] for k in rows.values()],
                             "diff_w": [o[k] - e[k] for k in rows.values()]})

    @property
    def avg_power_deficit_pct(self) -> float:
        """How much lower the optimised plan's average power is than even pacing's, % (positive = lower)."""
        e = self.even_stats["avg_power_w"]
        return 100.0 * (e - self.opt_stats["avg_power_w"]) / max(e, 1e-9)

    @property
    def cautions(self) -> list[str]:
        out = []
        if self.avg_power_deficit_pct > 1.0:
            e, o = self.even_stats["avg_power_w"], self.opt_stats["avg_power_w"]
            out.append(
                f"Average power is {self.avg_power_deficit_pct:.1f}% lower than even pacing ({o:.0f} vs {e:.0f} W) at "
                "the same NP: the plan trades harder surges for easier stretches, which NP flatters. Judge the gain "
                f"by the equal-effort figure ({self.saved_equal_np120_s:.0f} s saved at the same 120 s NP).")
        n120 = self.opt_stats["np120_w"] / max(self.even_stats["np120_w"], 1e-9) - 1.0
        if n120 > 0.01:
            out.append(f"The plan's 120 s NP is {100 * n120:.1f}% above even pacing's, i.e. it is physiologically "
                       f"harder than the same-NP even ride; the equal-effort saving is {self.saved_equal_np120_s:.0f} s.")
        if not self.converged:
            out.append("The solver stopped before fully converging; the plan is feasible but may be slightly "
                       "sub-optimal.")
        bad = {w: x for w, x in self.cap_excess_w.items() if x > 0.5}
        if bad:
            out.append("Rolling-power caps were exceeded by "
                       + ", ".join(f"{x:.1f} W ({w // 60} min)" for w, x in bad.items()) + ".")
        return out

    def summary(self) -> dict:
        return {
            "even_time": fmt_time(self.even.total_time_s), "opt_time": fmt_time(self.optimised.total_time_s),
            "saved_s": self.time_saved_s, "even_np": self.even.np_w, "opt_np": self.optimised.np_w,
            "opt_vi": self.optimised.vi, "opt_avg_power": self.optimised.avg_power,
            "even_avg_power": self.even.avg_power, "success": self.success,
            "converged": self.converged, "saved_equal_np120_s": self.saved_equal_np120_s,
        }




def _block_index(course: Course, block_m: float) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = max(1, int(round(course.length_m / block_m)))
    edges = np.linspace(0.0, course.length_m, n_blocks + 1)
    idx = np.clip(np.searchsorted(edges, course.seg_mid_m, side="right") - 1, 0, n_blocks - 1)
    return edges, idx


def _cap_stride(win_s: int) -> int:
    """Sampling stride (s) of a rolling-window constraint: 20 s for 1 min windows, 60 s for 5 min and longer."""
    return int(min(60, max(5, win_s / 3 if win_s <= 60 else win_s / 5)))


def _cap_excess_w(p1: np.ndarray, caps: dict, np_target: float) -> dict:
    """Peak rolling mean minus the cap, in W, at full 1 s resolution (<= 0 means the cap holds)."""
    return {int(w): float(_rolling_numba(p1, int(w), window_starts(len(p1), w)).max() - c / 100.0 * np_target)
            for w, c in caps.items()}


def optimise_pacing(
    course: Course,
    rider: RiderProfile,
    np_target: float,
    env: Environment | None = None,
    config: SimConfig | None = None,
    *,
    block_m: float = 1000.0,
    bound_pct: float = 15.0,
    smooth: float = 0.0,
    max_iter: int = 150,
    even: SimResult | None = None,
    workers: int = min(8, os.cpu_count() or 1),
    power_caps: dict | None = DEFAULT_CAPS,
    warm_start: bool = True,
    start_power: np.ndarray | None = None,
) -> OptResult:
    """Choose a power per block minimising time subject to simulated NP == ``np_target``.

    * Bounds are +/- ``bound_pct`` % of NP.
    * ``power_caps`` maps a rolling-window length in seconds to a cap in % of NP (default 1 min <= 110 %,
      5 min <= 106 %; ``None`` or ``{}`` switches the caps off, i.e. NP is the only effort limit).
    * ``smooth`` penalises squared changes between adjacent blocks, per km of course so it means the same at every
      block length (units: fractional NP; try 0.5-5).
    * ``warm_start`` solves finer blocks (< 1 km) first on a coarser grid and starts from that solution.
    * ``start_power`` (W per block) overrides the starting point (used to check the optimum is start-independent).

    Wind is frozen to the even-pacing time map during the search; the final plan is re-simulated with the full
    time-following weather and rescaled (never beyond the bounds) to hit NP exactly, and the caps are re-checked
    at 1 s resolution and tightened/re-solved if the finite sampling missed a peak.
    """
    env = env or Environment()
    cfg = config or SimConfig()
    caps = {int(w): float(c) for w, c in (power_caps or {}).items()}
    if even is None:
        even = simulate_even(course, rider, np_target, env, cfg)
    even.label = "even"
    edges, idx = _block_index(course, block_m)
    nb = len(edges) - 1
    lo, hi = max(0.0, 1.0 - bound_pct / 100.0), 1.0 + bound_pct / 100.0
    p_even = float(even.power_set[0])
    t_ref = even.total_time_s
    x0 = np.full(nb, p_even / np_target)
    # per-km, block-length independent smoothness weight (1.0 at 1 km blocks)
    pen_w = smooth * 1000.0 / (course.length_m / nb)
    n_evals = [0]
    n_iter = [0]

    if start_power is not None:
        x_start = np.clip(np.asarray(start_power, float) / np_target, lo, hi)
        if x_start.shape != (nb,):
            raise ValueError(f"start_power needs {nb} values (one per block), got {x_start.shape}")
    elif warm_start and block_m < 999.0 and nb > 4:
        coarse_m = min(1000.0, 2.0 * block_m)
        coarse = optimise_pacing(course, rider, np_target, env, cfg, block_m=coarse_m, bound_pct=bound_pct,
                                 smooth=smooth, max_iter=max_iter, even=even, workers=workers, power_caps=caps,
                                 warm_start=True)
        n_evals[0] += coarse.n_evals
        n_iter[0] += coarse.n_iter
        mids = 0.5 * (edges[:-1] + edges[1:])
        ci = np.clip(np.searchsorted(coarse.block_edges_m, mids, side="right") - 1, 0, len(coarse.block_power) - 1)
        x_start = np.clip(coarse.block_power[ci] / np_target, lo, hi)
    else:
        x_start = x0

    pool = ThreadPoolExecutor(max_workers=max(1, min(nb, workers))) if workers > 1 else None
    pmap = pool.map if pool is not None else map
    bnds = (np.full(nb, lo), np.full(nb, hi))

    def solve(xs, caps_eff, frozen, iters):
        """SLSQP on the frozen-wind problem. Returns (x, converged, status message)."""
        hw_f, rho_f = frozen
        cap_items = sorted(caps_eff.items())
        # fixed sample positions (from the even-pacing duration) so every simulation returns the same vector length
        n_even = len(even.power_1hz)
        starts = {w: window_starts(n_even, w, _cap_stride(w)) for w, _ in cap_items}
        memo: dict[bytes, np.ndarray] = {}

        # one simulation gives time, NP and every rolling-window sample: shared by objective, NP constraint and
        # cap constraints (finite differences revisit the same perturbed points) and run on worker threads
        # (the numba kernel releases the GIL)
        def sim_vec(x):
            key = np.ascontiguousarray(x, dtype=float).tobytes()
            hit = memo.get(key)
            if hit is None:
                n_evals[0] += 1
                _, seg_t, p_app = _run(course, rider, (x * np_target)[idx], hw_f, rho_f, cfg)
                p1 = _p1hz_numba(seg_t, p_app)
                parts = [np.array([float(seg_t.sum()) / t_ref, float(_np_from_p1(p1, 30)) / np_target - 1.0])]
                for w, c in cap_items:
                    parts.append(c / 100.0 - _rolling_numba(p1, w, starts[w]) / np_target)
                hit = np.concatenate(parts)
                if len(memo) > 4 * (nb + 2):
                    memo.clear()
                memo[key] = hit
            return hit

        jac_cache: list = [None, None]

        def jac_all(x):
            key = np.ascontiguousarray(x, dtype=float).tobytes()
            if jac_cache[0] != key:
                # identical stencil to SLSQP's built-in finite differences (2-point, abs step eps, bound aware)
                jac_cache[1] = approx_derivative(sim_vec, np.clip(x, lo, hi), method="2-point", abs_step=2e-3,
                                                 bounds=bnds, workers=pmap)
                jac_cache[0] = key
            return jac_cache[1]

        def objective(x):
            pen = pen_w * float(np.sum(np.diff(x) ** 2)) if pen_w > 0 else 0.0
            return float(sim_vec(x)[0]) + pen

        def obj_jac(x):
            g = np.array(jac_all(x)[0], float)
            if pen_w > 0:
                d = np.diff(x)
                g[:-1] -= 2.0 * pen_w * d
                g[1:] += 2.0 * pen_w * d
            return g

        cons = [{"type": "eq", "fun": lambda x: float(sim_vec(x)[1]), "jac": lambda x: jac_all(x)[1]}]
        if cap_items:
            cons.append({"type": "ineq", "fun": lambda x: sim_vec(x)[2:], "jac": lambda x: jac_all(x)[2:]})

        hist: list[float] = []
        state = {"stopped": False}

        def callback(xk):
            n_iter[0] += 1
            v = sim_vec(np.clip(xk, lo, hi))
            state["x"] = np.clip(xk, lo, hi)
            hist.append(objective(state["x"]))
            feas = abs(v[1]) < 1e-6 and (len(v) <= 2 or v[2:].min() > -1e-6)
            # stationary: no measurable gain (< 1e-9 of the race time) over the last 10 iterations while feasible
            if feas and len(hist) > 10 and hist[-11] - hist[-1] < 1e-9:
                state["stopped"] = True
                raise StopIteration

        try:
            res = minimize(objective, xs, jac=obj_jac, method="SLSQP", bounds=[(lo, hi)] * nb, constraints=cons,
                           callback=callback, options={"maxiter": iters, "ftol": 1e-9, "eps": 2e-3})
            xr, converged, message = np.clip(res.x, lo, hi), bool(res.success), str(res.message)
            if not converged and res.status == 9 and len(hist) > 10 and hist[-11] - hist[-1] < 2e-6:
                converged = True  # iteration limit hit, but the objective had already stopped moving
                message = "Converged (objective stationary)"
        except StopIteration:  # raised by our callback: scipy does not catch it for SLSQP
            xr, converged, message = state["x"], True, "Converged (objective stationary)"
        return xr, converged, message

    def plan(x, k):
        return (np.clip(x * k, lo, hi) * np_target)[idx]

    def rescale(x):
        """Scalar k (applied before clipping, so the bounds always hold) giving NP == target in the full weather."""
        f = lambda k: simulate(course, rider, plan(x, k), env, cfg).np_w - np_target
        try:
            return brentq(f, 0.5, 2.0, xtol=1e-8)
        except ValueError:
            return 1.0

    try:
        x, converged, message = solve(x_start, caps, (even.headwind, even.rho), max_iter)
        k = rescale(x)
        final = simulate(course, rider, plan(x, k), env, cfg, label="optimised")
        excess = _cap_excess_w(final.power_1hz, caps, np_target)
        # the caps were sampled every few seconds and the final NP rescale / weather re-pass moves things a little:
        # tighten any cap that is still exceeded and re-solve (warm) so the plan really satisfies them
        eff = dict(caps)
        for _ in range(4):
            worst = {w: e for w, e in excess.items() if e > 1e-3}
            if not worst:
                break
            for w, e in worst.items():
                eff[w] -= 100.0 * (e / np_target + 2e-4)
            x, conv2, message = solve(np.clip(x * k, lo, hi), eff, (final.headwind, final.rho), 60)
            converged = converged and conv2
            k = rescale(x)
            final = simulate(course, rider, plan(x, k), env, cfg, label="optimised")
            excess = _cap_excess_w(final.power_1hz, caps, np_target)
    finally:
        if pool is not None:
            pool.shutdown()

    success = bool(final.total_time_s <= even.total_time_s)
    if not success:  # fall back to even pacing rather than recommend something slower
        final = even
        x = x0
        k = 1.0
        excess = _cap_excess_w(even.power_1hz, caps, np_target)
    block_power = np.clip(x * k, lo, hi) * np_target if success else x0 * np_target

    saved_eq = 0.0
    if success:
        saved_eq = _saved_equal_np120(course, rider, cfg, even, final, block_power, idx)
    return OptResult(
        course=course, np_target=np_target, even=even, optimised=final, block_edges_m=edges,
        block_power=block_power, bound_pct=bound_pct, smooth=smooth, success=success,
        message=message, n_evals=n_evals[0], converged=converged, n_iter=n_iter[0], power_caps=caps,
        cap_excess_w=excess, block_m=block_m, saved_equal_np120_s=saved_eq,
    )


def _saved_equal_np120(course, rider, cfg, even: SimResult, final: SimResult, block_power, idx) -> float:
    """Seconds saved vs even pacing if the optimised plan is scaled (all blocks together) until its 120 s NP
    equals even pacing's: the fair comparison when the plan is spikier than a same-NP even ride."""
    target = np_window(even.power_1hz, 120)
    hw, rho = final.headwind, final.rho
    pw = block_power[idx]

    def run(k):
        _, seg_t, p_app = _run(course, rider, pw * k, hw, rho, cfg)
        return seg_t, float(_np_from_p1(_p1hz_numba(seg_t, p_app), 120))

    try:
        k = brentq(lambda k: run(k)[1] - target, 0.6, 1.4, xtol=1e-7)
    except ValueError:
        return even.total_time_s - final.total_time_s
    return even.total_time_s - float(run(k)[0].sum())
