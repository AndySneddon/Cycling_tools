"""Validation of the race planner: physics consistency, NP solver, optimiser and a real-ride back-test."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cycling_tools import physics  # noqa: E402
from cycling_tools.course import CourseSettings, build_course, course_from_ride, load_gpx  # noqa: E402
from cycling_tools.optimise import optimise_pacing  # noqa: E402
from cycling_tools.physics import normalised_power  # noqa: E402
from cycling_tools.profile import RiderProfile  # noqa: E402
from cycling_tools.simulate import (DEFAULT_RHO, Environment, SimConfig, simulate, simulate_even,  # noqa: E402
                                    splits_table, time_vs_np, what_if)
from cycling_tools.weather import constant_weather  # noqa: E402

FIT = ROOT / "Fit_files" / "Manchester_District_TTA_50_WU_CD.fit"
RIDER = RiderProfile(mass_kg=100.0, cda=0.25, crr=0.0031, drivetrain_eff=0.97)


def straight_course(length_km=20.0, grade=0.0, **kw):
    n = int(length_km * 100)
    lat = 53.0 + np.linspace(0, length_km / 111.2, n)
    lon = np.full(n, -2.0)
    elev = np.linspace(0, grade * length_km * 1000, n)
    return build_course(lat, lon, elev, settings=CourseSettings(**kw))


def hilly_out_and_back(wind_ms=0.0, hills=True):
    n = 2001
    lat = np.concatenate([np.linspace(53.0, 53.09, n), np.linspace(53.09, 53.0, n)])
    lon = np.full(2 * n, -2.0)
    d = np.linspace(0, 1, 2 * n)
    elev = 30 * np.sin(2 * np.pi * d * 5) if hills else np.zeros(2 * n)
    env = Environment(constant_weather(15.0, wind_ms, 0.0))  # wind from north: headwind on the way out
    return build_course(lat, lon, elev), env


# ------------------------------------------------------------------- course
def test_course_resampling_and_ascent():
    c = straight_course(10.0, grade=0.02)
    assert np.allclose(c.ds, c.ds[0], rtol=1e-6)
    assert abs(c.ds[0] - 10.0) < 0.5
    assert abs(c.ascent_m - 200.0) < 3.0
    assert abs(c.length_m - 10000) < 60
    assert np.allclose(c.grade[50:-50], 0.02, atol=1e-3)


def test_corner_cap_and_braking_feasibility():
    # square-ish route with sharp 90 degree turns
    pts = [(53.0, -2.0), (53.0, -1.99), (53.01, -1.99), (53.01, -2.0)]
    lat, lon = [], []
    for (a, b), (c_, d) in zip(pts[:-1], pts[1:]):
        lat += list(np.linspace(a, c_, 200, endpoint=False))
        lon += list(np.linspace(b, d, 200, endpoint=False))
    lat.append(pts[-1][0]); lon.append(pts[-1][1])
    c = build_course(lat, lon, np.zeros(len(lat)))
    assert c.vcap.min() < 12.0
    # braking pass: speed cap may only fall at <= max_brake_decel
    v2 = c.vcap ** 2
    assert np.all(v2[:-1] - v2[1:] <= 2 * c.settings.max_brake_decel * c.ds[:-1] + 1e-6)
    # simulation never exceeds the cap
    s = simulate(c, RIDER, 300.0)
    assert np.all(s.v_node[1:] <= c.vcap + 1e-6)


def test_gpx_roundtrip():
    c = load_gpx(ROOT / "courses" / "Manchester_TTA_50_sample.gpx")
    assert 79_000 < c.length_m < 82_000
    assert 300 < c.ascent_m < 800


# ---------------------------------------------------------- steady-state physics
@pytest.mark.parametrize("grade,hw", [(0.0, 0.0), (0.0, 4.0), (0.0, -3.0), (0.03, 0.0), (-0.02, 2.0)])
def test_steady_state_matches_physics(grade, hw):
    c = straight_course(30.0, grade=grade)
    env = Environment(constant_weather(15.0, abs(hw), 0.0 if hw >= 0 else 180.0, 1013.25, 0.5))
    # heading is ~0 deg (north); wind from north = headwind
    s = simulate(c, RIDER, 250.0, env)
    rho = float(physics.air_density(15.0, 101325.0, 0.5))
    expected = physics.steady_speed(250.0, grade, mass_kg=100, cda=0.25, crr=0.0031, rho=rho,
                                    headwind_mps=hw, drivetrain_eff=0.97)
    got = s.v_seg[len(s.v_seg) // 2: -50]
    assert abs(got.mean() - expected) < 2e-3 * expected, (got.mean(), expected)


def test_energy_conservation():
    n = 3000
    lat = 53.0 + np.linspace(0, 0.2, n)
    elev = 25 * np.sin(np.linspace(0, 8 * np.pi, n)) + 5 * np.linspace(0, 1, n)  # straight, hilly
    c = build_course(lat, np.full(n, -2.0), elev)
    env = Environment(constant_weather(15.0, 4.0, 0.0))
    s = simulate(c, RIDER, 240.0, env, SimConfig(coast_speed_mps=100.0))
    assert np.all(s.v_node[1:] < c.vcap - 1e-3), "test requires no cap to be active"
    m, g = RIDER.mass_kg, physics.G
    v = s.v_node
    vavg = 0.5 * (v[:-1] + v[1:])
    th = np.arctan(c.grade)
    resist = (m * g * (RIDER.crr * np.cos(th) + np.sin(th)) * c.ds
              + 0.5 * RIDER.cda * s.rho * (vavg + s.headwind) * np.abs(vavg + s.headwind) * c.ds)
    work_in = np.sum(s.power_applied * RIDER.drivetrain_eff * s.seg_time)
    m_eff = m + SimConfig().wheel_inertia_kg
    d_ke = 0.5 * m_eff * (v[-1] ** 2 - v[0] ** 2)
    assert abs(work_in - (np.sum(resist) + d_ke)) < 1e-6 * work_in


def test_time_monotonic_in_power_and_cda():
    c, env = hilly_out_and_back(3.0)
    t = [simulate(c, RIDER, p, env).total_time_s for p in (200, 250, 300)]
    assert t[0] > t[1] > t[2]
    slow = simulate(c, replace(RIDER, cda=0.30), 250, env).total_time_s
    assert slow > t[1]


# ------------------------------------------------------------------ NP solver
@pytest.mark.parametrize("target", [180.0, 250.0, 320.0])
def test_np_solve_accuracy(target):
    c, env = hilly_out_and_back(4.0)
    e = simulate_even(c, RIDER, target, env)
    assert abs(e.np_w - target) < 0.05
    # independent NP check from the 1 Hz series
    assert abs(normalised_power(e.power_1hz) - target) < 0.05
    # whole-second series reproduces the ride duration
    assert abs(len(e.power_1hz) - e.total_time_s) < 1.5


def test_time_vs_np_and_whatif():
    c, env = hilly_out_and_back(2.0)
    df = time_vs_np(c, RIDER, [200, 225, 250, 275, 300], env)
    assert df["time_s"].is_monotonic_decreasing
    assert (df["s_saved_per_5w"] > 0).all()
    wi = what_if(c, RIDER, 250.0, env)
    saved = dict(zip(wi["scenario"], wi["saved_s"]))
    assert saved["CdA -0.010 m^2"] > 0 and saved["Mass -1 kg"] > 0 and saved["+5 W NP"] > 0
    assert saved["CdA -0.010 m^2"] > saved["CdA -0.005 m^2"]


def test_splits_sum_to_total():
    c, env = hilly_out_and_back(2.0)
    s = simulate_even(c, RIDER, 250.0, env)
    sp = splits_table(s, 1000.0)
    assert abs(sp["split_s"].sum() - s.total_time_s) < 1e-6


# ------------------------------------------------------------------ optimiser
def test_optimiser_beats_even_hilly_windy():
    c, env = hilly_out_and_back(wind_ms=5.0)
    o = optimise_pacing(c, RIDER, 250.0, env, block_m=1000, bound_pct=25)
    assert o.success
    assert o.optimised.total_time_s <= o.even.total_time_s
    assert o.time_saved_s > 5.0, o.time_saved_s
    assert abs(o.optimised.np_w - 250.0) < 0.05
    # power rises on climbs relative to descents
    t = o.block_table()
    assert t.loc[t["grade_pct"] > 1.0, "target_power_w"].mean() > t.loc[t["grade_pct"] < -1.0, "target_power_w"].mean() + 20
    # bounds respected (+ small rescale slack)
    assert t["pct_of_np"].max() <= 126 and t["pct_of_np"].min() >= 74
    csv = o.to_csv()
    assert "target_power_w" in csv and csv.count("\n") >= 20


def test_optimiser_headwind_gets_more_power():
    c, env = hilly_out_and_back(wind_ms=6.0, hills=False)
    o = optimise_pacing(c, RIDER, 250.0, env, block_m=2000, bound_pct=25)
    assert o.optimised.total_time_s <= o.even.total_time_s
    t = o.block_table()
    out_leg = t[t["end_km"] <= 10.0]["target_power_w"].mean()    # headwind
    back_leg = t[t["start_km"] >= 10.0]["target_power_w"].mean()  # tailwind
    assert out_leg > back_leg


def test_optimiser_never_worse_than_even_flat():
    c = straight_course(10.0)
    o = optimise_pacing(c, RIDER, 250.0, Environment(), block_m=1000)
    assert o.optimised.total_time_s <= o.even.total_time_s + 1e-6
    assert abs(o.time_saved_s) < 1.0  # nothing to gain on a flat, windless course


# ------------------------------------------------------------------- back-test
def _weather_or_skip(lat, lon, t0):
    from cycling_tools.weather import WeatherError, fetch_weather
    try:
        return fetch_weather(lat, lon, t0, t0 + pd.Timedelta(hours=3))
    except WeatherError as exc:
        pytest.skip(f"weather unavailable: {exc}")


@pytest.mark.skipif(not FIT.exists(), reason="sample FIT missing")
def test_backtest_manchester_tt():
    from cycling_tools.fit_io import load_fit

    ride = load_fit(FIT)
    lap = ride.df[ride.df["lap"] == 2]
    actual_s = float(len(lap))
    actual_np = normalised_power(lap["power"].fillna(0).to_numpy())
    course = course_from_ride(ride, lap=2)
    t0 = lap["timestamp"].iloc[0]
    wx = _weather_or_skip(*course.centre, t0)
    cfg = SimConfig(start_speed_mps=float(lap["speed"].iloc[0]))

    still = simulate_even(course, RIDER, actual_np, Environment(), cfg)
    windy = simulate_even(course, RIDER, actual_np, Environment(wx, t0, 0.7), cfg)
    err_still = (still.total_time_s - actual_s) / actual_s * 100
    err_wind = (windy.total_time_s - actual_s) / actual_s * 100
    print(f"\nBACKTEST actual {actual_s:.0f} s ({lap['speed'].mean()*3.6:.1f} km/h) NP {actual_np:.0f} W")
    print(f"  still air   : {still.total_time_s:.0f} s  error {err_still:+.1f}%")
    print(f"  archive wind: {windy.total_time_s:.0f} s  error {err_wind:+.1f}%")
    # Loose bound: the model is physically sane (within ~12 %); the residual is diagnosed in the README.
    assert abs(err_still) < 12.0
    assert abs(err_wind) < 15.0
    assert abs(windy.np_w - actual_np) < 0.1


# ------------------------------------------------------------------ page smoke
def test_page_smoke():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "pages" / "3_Race_Planner.py"), default_timeout=120)
    at.run()
    assert not at.exception
    pick = [s for s in at.selectbox if s.label.startswith("...or pick")][0]
    pick.select("Manchester_TTA_50_sample.gpx")
    at.run()
    [r for r in at.radio if r.label == "Weather"][0].set_value("None (still air)")
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Time" for m in at.metric)
    [b for b in at.button if b.label == "Optimise pacing"][0].click()
    at.run()
    assert not at.exception, at.exception
    assert any(m.label == "Optimised" for m in at.metric)
