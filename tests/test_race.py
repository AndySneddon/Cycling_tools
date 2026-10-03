"""Validation of the race planner: physics consistency, NP solver, optimiser and a real-ride back-test."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cycling_tools import physics  # noqa: E402
from cycling_tools.course import (CourseSettings, _smooth_distance, build_course, course_from_ride,  # noqa: E402
                                  despike_altitude, load_gpx)
from cycling_tools.optimise import optimise_pacing  # noqa: E402
from cycling_tools.simulate import np_window, rolling_peak  # noqa: E402
from cycling_tools.physics import normalised_power  # noqa: E402
from cycling_tools.profile import RiderProfile  # noqa: E402
from cycling_tools.simulate import (Environment, SimConfig, simulate, simulate_even,  # noqa: E402
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


# --------------------------------------------------------- audit regressions
def sinusoid_hill(step_m, wavelength=400.0, amp=0.185, length=3000.0):
    s = np.linspace(0, length, int(length) + 1)
    g = amp * np.sin(2 * np.pi * s / wavelength)
    elev = np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(s))])
    return build_course(53 + s / 111195.0, np.full(len(s), -2.0), elev,
                        settings=CourseSettings(step_m=step_m, elev_smooth_m=0.0, max_grade=0.3))


def test_steep_ramp_is_not_declared_a_stall():
    # 18.5 % sinusoidal hill at 250 W: the rider carries speed into the ramp and decelerates towards its slow
    # steady-state speed. A single trapezoid step used to find "no root" and crawl at 0.3 m/s (250 s error at 50 m).
    cfg = SimConfig(start_speed_mps=8.0)
    ref = simulate(sinusoid_hill(5.0), RIDER, 250.0, Environment(), cfg).total_time_s
    for step in (20.0, 50.0):
        r = simulate(sinusoid_hill(step), RIDER, 250.0, Environment(), cfg)
        assert r.v_node[1:].min() > 0.5, "spurious stall"
        assert abs(r.total_time_s - ref) < 0.01 * ref, (step, r.total_time_s, ref)


def test_segment_time_is_trapezoid_in_every_branch():
    # capped / coasting / pedalling / sub-stepped segments must all report ds / mean(v0, v1)
    c, env = hilly_out_and_back(3.0)
    for p in (150.0, 400.0):
        s = simulate(c, RIDER, p, env)
        assert np.allclose(s.seg_time, c.ds / (0.5 * (s.v_node[:-1] + s.v_node[1:])), rtol=1e-9)
    s = simulate(sinusoid_hill(50.0), RIDER, 250.0, Environment(), SimConfig(start_speed_mps=8.0))
    assert abs(s.seg_time.sum() - sum(s.course.ds / s.v_seg)) < 1e-6


def test_despike_altitude_keeps_real_climbs():
    elev = np.linspace(0, 40, 200)
    spiked = elev.copy()
    spiked[[20, 21, 100]] += [25.0, -30.0, 18.0]
    out = despike_altitude(spiked)
    assert np.abs(out - elev).max() < 1.0
    assert np.allclose(despike_altitude(elev), elev)
    assert np.isnan(despike_altitude(np.r_[np.nan, elev])[0])


def _flat_ride(alt):
    from types import SimpleNamespace
    n = len(alt)
    t = pd.date_range("2026-05-01 08:00", periods=n, freq="1s")
    df = pd.DataFrame({"timestamp": t, "lat": 53.0 + np.arange(n) * 6.0 / 111195.0, "lon": -2.0, "alt": alt,
                       "lap": 1})
    return SimpleNamespace(df=df, name="synthetic")


def test_baro_warmup_transient_is_dropped():
    # Almere: altitude jumped -17 -> +4 m in the first 6 s (a fake 20 % wall) before settling at a flat ~-3 m
    alt = np.full(400, -3.0)
    alt[:6] = [-17, -17.4, -15, -12.8, -11, -8.6]
    alt[6:30] = np.linspace(4.4, -3.0, 24)
    ride = _flat_ride(alt)
    c = course_from_ride(ride)
    assert np.abs(c.grade).max() < 0.02
    raw = course_from_ride(ride, settings=CourseSettings(alt_warmup_s=0.0, alt_spike_window=0))
    assert np.abs(raw.grade).max() > 0.15  # the transient really did produce a wall without the fix


def test_smoothing_window_is_centred():
    # an even uniform_filter window shifts the profile by half a step per pass; ours must not move the centroid
    for window in (50.0, 60.0, 70.0):
        x = np.zeros(201)
        x[100] = 1.0
        y = _smooth_distance(x, 10.0, window)
        centroid = float(np.sum(np.arange(201) * y) / y.sum())
        assert abs(centroid - 100.0) < 1e-9, (window, centroid)


@pytest.mark.parametrize("step", [5.0, 7.0, 10.0, 15.0])
def test_curvature_window_is_arc_length(step):
    radius = 150.0
    ang = np.linspace(0.0, np.pi, 300)
    lat = 53.0 + radius * np.sin(ang) / 111195.0
    lon = -2.0 + radius * (1 - np.cos(ang)) / (111195.0 * np.cos(np.radians(53.0)))
    c = build_course(lat, lon, np.zeros_like(ang),
                     settings=CourseSettings(step_m=step, heading_smooth_m=step))
    mid = c.curvature[len(c.curvature) // 4: 3 * len(c.curvature) // 4]
    assert abs(np.median(mid) * radius - 1.0) < 0.05, (step, np.median(mid) * radius)


def _fake_weather_response(start, end):
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in idx], "temperature_2m": [10.0] * len(idx),
                       "relative_humidity_2m": [60.0] * len(idx), "surface_pressure": [1000.0] * len(idx),
                       "wind_speed_10m": [4.0] * len(idx), "wind_direction_10m": [200.0] * len(idx),
                       "wind_gusts_10m": [6.0] * len(idx)}}


def test_weather_cache_ttl_forecast_only(tmp_path, monkeypatch):
    import os
    import time
    from cycling_tools import weather as wmod

    monkeypatch.setattr(wmod, "CACHE_DIR", tmp_path)
    calls = []

    class Resp:
        status_code = 200

        def __init__(self, params):
            self.params = params

        def json(self):
            return _fake_weather_response(self.params["start_date"], self.params["end_date"])

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return Resp(params)

    monkeypatch.setattr(wmod.requests, "get", fake_get)
    p = {"a": 1}
    for url, ttl in ((wmod.FORECAST_URL, wmod.FORECAST_TTL_S), (wmod.ARCHIVE_URL, None)):
        calls.clear()
        wmod._request(url, {"start_date": "2020-01-01", "end_date": "2020-01-01", **p}, ttl_s=ttl)
        wmod._request(url, {"start_date": "2020-01-01", "end_date": "2020-01-01", **p}, ttl_s=ttl)
        assert len(calls) == 1, "second call should hit the cache"
        for f in tmp_path.glob("*.json"):  # age every cache file by 3 h
            os.utime(f, (time.time() - 3 * 3600, time.time() - 3 * 3600))
        wmod._request(url, {"start_date": "2020-01-01", "end_date": "2020-01-01", **p}, ttl_s=ttl)
        assert len(calls) == (2 if ttl else 1), (url, len(calls))


def test_weather_gaps_not_hidden():
    from cycling_tools.weather import WeatherError, _fill_gaps

    idx = pd.date_range("2026-01-01", periods=10, freq="h")
    df = pd.DataFrame({"temp_c": 10.0, "rh": 0.5, "pressure_pa": 1e5, "wind_ms": 5.0,
                       "wind_dir": [350.0] + [np.nan] * 2 + [10.0] * 7, "gust_ms": 7.0}, index=idx)
    df.loc[idx[1:3], "wind_ms"] = np.nan
    out = _fill_gaps(df)
    # 350 deg -> 10 deg wind interpolated as a vector passes through north (speed dips), not the long way round
    assert out["wind_ms"].iloc[1] < 5.0 and (out["wind_dir"].iloc[1] > 340 or out["wind_dir"].iloc[1] < 20)
    big = df.copy()
    big.loc[idx[2:8], ["wind_ms", "wind_dir"]] = np.nan
    with pytest.raises(WeatherError):
        _fill_gaps(big)
    lead = df.copy()
    lead[["wind_ms", "wind_dir"]] = [5.0, 10.0]
    lead.loc[idx[0], ["wind_ms", "wind_dir"]] = np.nan  # leading hole is trimmed, never back-filled
    out = _fill_gaps(lead)
    assert out.index[0] == idx[1] and not out.isna().any().any()


def test_fetch_weather_uses_utc_dates_and_horizon(monkeypatch):
    from cycling_tools import weather as wmod

    seen = {}
    monkeypatch.setattr(wmod, "_request", lambda url, params, ttl_s=None: seen.update(params=params, url=url)
                        or _fake_weather_response(params["start_date"], params["end_date"]))
    # 00:30 BST on 1 June is 23:30 UTC on 31 May
    wmod.fetch_weather(53.0, -2.0, pd.Timestamp("2025-06-01 00:30", tz="Europe/London"),
                       pd.Timestamp("2025-06-01 05:30", tz="Europe/London"))
    assert seen["params"]["start_date"] == "2025-05-31" and seen["url"] == wmod.ARCHIVE_URL
    far = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)
    with pytest.raises(wmod.WeatherError, match="16 days"):
        wmod.fetch_weather(53.0, -2.0, far, far + pd.Timedelta(hours=4))
    # a race that starts inside the horizon but ends beyond it is clamped rather than refused
    near = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=15, hours=1)
    near = near.replace(hour=22, minute=0)
    if near.date() > (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=15)).date():
        near = near - pd.Timedelta(days=1)
    wmod.fetch_weather(53.0, -2.0, near, near + pd.Timedelta(hours=8))


def test_weather_at_fast_path_matches_dataframe_path():
    from cycling_tools.weather import Weather

    idx = pd.date_range("2026-05-01", periods=8, freq="h")
    df = pd.DataFrame({"temp_c": np.linspace(10, 17, 8), "rh": 0.5, "pressure_pa": 1e5,
                       "wind_ms": np.linspace(3, 8, 8), "wind_dir": np.linspace(300, 420, 8) % 360, "gust_ms": 6.0},
                      index=idx)
    w = Weather(df, "Open-Meteo forecast")
    t = pd.date_range("2026-05-01 00:30", periods=40, freq="7min")
    a = w.at(t)
    b = w.at_seconds(t.values.astype("datetime64[s]").astype(float))
    for col in a.columns:
        assert np.allclose(a[col], b[col])


def test_simulate_even_hint_and_whatif_match_unhinted():
    c, env = hilly_out_and_back(3.0)
    base = simulate_even(c, RIDER, 250.0, env)
    hinted = simulate_even(c, RIDER, 250.0, env, p_hint=float(base.power_set[0]) * 1.02)
    assert abs(base.total_time_s - hinted.total_time_s) < 1e-3
    far = simulate_even(c, RIDER, 250.0, env, p_hint=50.0)  # a poor hint must still bracket correctly
    assert abs(base.total_time_s - far.total_time_s) < 0.05


# --------------------------------------------------------- optimiser realism
def _opt_case():
    c, env = hilly_out_and_back(wind_ms=5.0)
    return c, env


def test_optimiser_respects_bounds_and_np_exactly():
    c, env = _opt_case()
    for bound in (10, 15, 25):
        o = optimise_pacing(c, RIDER, 250.0, env, block_m=1000, bound_pct=bound)
        assert o.success
        pct = o.block_power / 250.0
        assert pct.max() <= 1 + bound / 100 + 1e-9 and pct.min() >= 1 - bound / 100 - 1e-9, (bound, pct.min(), pct.max())
        assert np.all(o.optimised.power_set <= 250.0 * (1 + bound / 100) + 1e-9)
        assert abs(o.optimised.np_w - 250.0) < 0.01


def test_optimiser_rolling_caps_hold():
    c, env = _opt_case()
    caps = {60: 108.0, 300: 106.0}
    o = optimise_pacing(c, RIDER, 250.0, env, block_m=1000, bound_pct=25, power_caps=caps)
    p1 = o.optimised.power_1hz
    assert rolling_peak(p1, 60) <= 1.08 * 250.0 + 0.1
    assert rolling_peak(p1, 300) <= 1.06 * 250.0 + 0.1
    assert all(v <= 0.1 for v in o.cap_excess_w.values())
    free = optimise_pacing(c, RIDER, 250.0, env, block_m=1000, bound_pct=25, power_caps=None)
    assert rolling_peak(free.optimised.power_1hz, 60) > 1.08 * 250.0 + 1.0, "caps should have been binding"
    assert free.time_saved_s >= o.time_saved_s - 0.3  # caps cost a little time, never gain it


def test_optimiser_multi_start_agreement_and_not_slower_than_even():
    c, env = _opt_case()
    rng = np.random.default_rng(3)
    nb = 20
    starts = [None, 250.0 * (1 + 0.1 * rng.uniform(-1, 1, nb)), 250.0 * np.linspace(0.9, 1.1, nb)]
    saved = []
    for sp in starts:
        o = optimise_pacing(c, RIDER, 250.0, env, block_m=1000, bound_pct=15, start_power=sp)
        assert o.optimised.total_time_s <= o.even.total_time_s + 1e-6
        saved.append(o.time_saved_s)
    assert max(saved) - min(saved) < 0.5, saved


def test_finer_blocks_do_not_lose_without_caps():
    c, env = _opt_case()
    coarse = optimise_pacing(c, RIDER, 250.0, env, block_m=2000, bound_pct=15, power_caps=None)
    fine = optimise_pacing(c, RIDER, 250.0, env, block_m=1000, bound_pct=15, power_caps=None)
    assert fine.time_saved_s >= coarse.time_saved_s - 0.3, (fine.time_saved_s, coarse.time_saved_s)


def test_smooth_penalty_means_the_same_at_every_block_length():
    c, env = _opt_case()
    # a strong penalty should flatten the plan at any block length (relative to the unpenalised one)
    for bm in (1000, 2000):
        free = optimise_pacing(c, RIDER, 250.0, env, block_m=bm, bound_pct=15, power_caps=None)
        sm = optimise_pacing(c, RIDER, 250.0, env, block_m=bm, bound_pct=15, power_caps=None, smooth=5.0)
        tv = lambda r: np.abs(np.diff(r.block_power)).sum() / 250.0
        assert tv(sm) < 0.7 * tv(free), (bm, tv(sm), tv(free))


def test_optimiser_reports_honest_stats_and_convergence():
    c, env = _opt_case()
    o = optimise_pacing(c, RIDER, 250.0, env, block_m=1000, bound_pct=15)
    st_ = o.stats_table().set_index("metric")
    assert {"Average power", "NP (30 s)", "NP (120 s)", "Peak 1 min", "Peak 5 min", "Peak 20 min"} <= set(st_.index)
    assert abs(st_.loc["NP (120 s)", "optimised_w"] - np_window(o.optimised.power_1hz, 120)) < 1e-6
    assert o.converged and "Iteration limit" not in o.message
    # a plan that rides fewer average watts than even pacing at the same NP must carry a caution
    import types
    fake = types.SimpleNamespace(**{k: getattr(o, k) for k in ("cap_excess_w", "converged", "saved_equal_np120_s")})
    low = {**o.opt_stats, "avg_power_w": o.even_stats["avg_power_w"] * 0.97}
    fake.even_stats, fake.opt_stats = o.even_stats, low
    fake.avg_power_deficit_pct = 3.0
    assert any("lower than even" in m for m in type(o).cautions.fget(fake))


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


def _planner(monkeypatch=None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "pages" / "3_Race_Planner.py"), default_timeout=180)
    at.run()
    [s for s in at.selectbox if s.label.startswith("...or pick")][0].select("Manchester_TTA_50_sample.gpx")
    at.run()
    return at


def test_planner_vi_labels_are_explicit():
    at = _planner()
    [r for r in at.radio if r.label == "Weather"][0].set_value("None (still air)")
    [n for n in at.number_input if n.label == "Target normalised power (W)"][0].set_value(218.0)
    [n for n in at.number_input if n.label.startswith("Expected race variability")][0].set_value(1.02)
    at.run()
    assert not at.exception, at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["Simulated average power (NP/VI)"] == f"{218.0 / 1.02:.0f} W"
    assert m["Your NP"] == "218 W" and m["Your VI"] == "1.02"
    assert "Normalised power (target)" not in m
    vi = [n for n in at.number_input if n.label.startswith("Expected race variability")][0]
    assert "course" in vi.help.lower() and "1.02" in vi.help and "1.00" in vi.help


def test_planner_requests_weather_window_covering_the_race(monkeypatch):
    import cycling_tools.weather as wmod

    asked = {}

    def fake_fetch(lat, lon, start, end):
        asked["start"], asked["end"] = pd.Timestamp(start), pd.Timestamp(end)
        idx = pd.date_range(pd.Timestamp(start).floor("h"), pd.Timestamp(end).ceil("h"), freq="h")
        df = pd.DataFrame({"temp_c": 12.0, "rh": 0.6, "pressure_pa": 1.0e5, "wind_ms": 3.0, "wind_dir": 200.0,
                           "gust_ms": 5.0}, index=idx)
        return wmod.Weather(df, "Open-Meteo forecast")

    import streamlit as st

    st.cache_data.clear()  # an earlier test may have cached a real fetch for the same arguments
    monkeypatch.setattr(wmod, "fetch_weather", fake_fetch)
    at = _planner()
    assert not at.exception, at.exception
    t = [x for x in at.metric if x.label == "Time"][0].value.split(":")
    secs = sum(int(v) * 60 ** i for i, v in enumerate(reversed(t)))
    assert asked["end"] - asked["start"] >= pd.Timedelta(seconds=secs) + pd.Timedelta(hours=1)
    assert not [w for w in at.warning if "weather data ends" in w.value]


def test_planner_optimise_shows_stats_and_caps():
    at = _planner()
    [r for r in at.radio if r.label == "Weather"][0].set_value("None (still air)")
    at.run()
    labels = [n.label for n in at.number_input]
    assert "1-minute power cap (% of NP)" in labels and "5-minute power cap (% of NP)" in labels
    bound = [s for s in at.slider if s.label.startswith("Power bounds")][0]
    assert bound.value == 15
    block = [s for s in at.select_slider if s.label == "Block length"][0]
    assert block.value == 1000
    [b for b in at.button if b.label == "Optimise pacing"][0].click()
    at.run()
    assert not at.exception, at.exception
    assert any(m.label.startswith("Saved at equal effort") for m in at.metric)
    assert any("0.5-1%" in i.value for i in at.info)
