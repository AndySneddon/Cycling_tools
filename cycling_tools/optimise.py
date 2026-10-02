"""Pacing optimiser: minimise race time at a fixed normalised power using block power targets."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from .course import Course
from .profile import RiderProfile
from .simulate import (Environment, SimConfig, SimResult, fmt_time, simulate, simulate_even, splits_table)


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
    success: bool
    message: str
    n_evals: int = 0

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

    def summary(self) -> dict:
        return {
            "even_time": fmt_time(self.even.total_time_s), "opt_time": fmt_time(self.optimised.total_time_s),
            "saved_s": self.time_saved_s, "even_np": self.even.np_w, "opt_np": self.optimised.np_w,
            "opt_vi": self.optimised.vi, "opt_avg_power": self.optimised.avg_power,
            "even_avg_power": self.even.avg_power, "success": self.success,
        }


def _block_index(course: Course, block_m: float) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = max(1, int(round(course.length_m / block_m)))
    edges = np.linspace(0.0, course.length_m, n_blocks + 1)
    idx = np.clip(np.searchsorted(edges, course.seg_mid_m, side="right") - 1, 0, n_blocks - 1)
    return edges, idx


def optimise_pacing(
    course: Course,
    rider: RiderProfile,
    np_target: float,
    env: Environment | None = None,
    config: SimConfig | None = None,
    *,
    block_m: float = 1000.0,
    bound_pct: float = 25.0,
    smooth: float = 0.0,
    max_iter: int = 80,
    even: SimResult | None = None,
) -> OptResult:
    """Choose a power per block minimising time subject to simulated NP == ``np_target``.

    Bounds are +/- ``bound_pct`` % of NP. ``smooth`` penalises squared changes between adjacent blocks
    (in units of fractional NP; try 0.5-5). Wind is frozen to the even-pacing time map during the search,
    then the final plan is re-simulated with the full time-following weather and rescaled to hit NP exactly.
    """
    env = env or Environment()
    cfg = config or SimConfig()
    if even is None:
        even = simulate_even(course, rider, np_target, env, cfg)
    frozen = (even.headwind, even.rho)
    edges, idx = _block_index(course, block_m)
    nb = len(edges) - 1
    p_even = float(even.power_set[0])
    t_ref = even.total_time_s
    lo, hi = 1.0 - bound_pct / 100.0, 1.0 + bound_pct / 100.0
    # x = block power as a fraction of NP target
    x0 = np.full(nb, p_even / np_target)
    n_evals = [0]

    def sim_x(x):
        n_evals[0] += 1
        return simulate(course, rider, (x * np_target)[idx], env, cfg, frozen_wind=frozen)

    def objective(x):
        r = sim_x(x)
        pen = smooth * float(np.sum(np.diff(x) ** 2)) if smooth > 0 else 0.0
        return r.total_time_s / t_ref + pen

    def constraint(x):
        return sim_x(x).np_w / np_target - 1.0

    res = minimize(
        objective, x0, method="SLSQP", bounds=[(lo, hi)] * nb,
        constraints=[{"type": "eq", "fun": constraint}],
        options={"maxiter": max_iter, "ftol": 1e-9, "eps": 2e-3},
    )
    x = np.clip(res.x, lo, hi)
    # final: exact NP under full (time-following) weather via a scalar rescale
    plan = lambda k: (np.clip(x * k, 0.0, None) * np_target)[idx]

    def f(k):
        return simulate(course, rider, plan(k), env, cfg).np_w - np_target

    try:
        k = brentq(f, 0.8, 1.25, xtol=1e-9)
    except ValueError:
        k = 1.0
    final = simulate(course, rider, plan(k), env, cfg, label="optimised")
    even.label = "even"
    success = bool(final.total_time_s <= even.total_time_s)
    if not success:  # fall back to even pacing rather than recommend something slower
        final = even
        x = x0
        k = 1.0
    block_power = np.clip(x * k, 0.0, None) * np_target
    return OptResult(
        course=course, np_target=np_target, even=even, optimised=final, block_edges_m=edges,
        block_power=block_power, bound_pct=bound_pct, smooth=smooth, success=success,
        message=str(res.message), n_evals=n_evals[0],
    )
