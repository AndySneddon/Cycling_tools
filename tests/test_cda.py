"""Validation of the CdA estimator against a synthetic ride with known physics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cycling_tools.cda import (
    CdAConfig, analyse_ride, fit_cda_ve, virtual_elevation, wind_scale_scan, best_wind_scale,
)
from cycling_tools.fit_io import Ride
from cycling_tools.physics import G, WHEEL_INERTIA_KG, air_density
from cycling_tools.weather import constant_weather

TRUE_CDA = 0.250
TRUE_CRR = 0.0035
MASS = 85.0
EFF = 0.975
WIND_TRUE = 4.0      # m/s at rider
WIND_FROM = 90.0
TEMP = 15.0


def synth_ride(seed=1, n=3600, cda=TRUE_CDA, crr=TRUE_CRR, wind=WIND_TRUE, noise=True):
    rng = np.random.default_rng(seed)
    rho = float(air_density(TEMP, 101325.0, 0.6))
    m_eff = MASS + WHEEL_INERTIA_KG
    # heading: 6 legs with 90 degree turns taking 8 s each
    leg = 560
    heading = np.zeros(n)
    turn_starts = []
    h = 90.0
    t = 0
    k = 0
    while t < n:
        end = min(t + leg, n)
        heading[t:end] = h
        t = end
        if t < n:
            turn_starts.append(t)
            for i in range(8):
                if t + i < n:
                    heading[t + i] = h + 90.0 * (i + 1) / 8
            h = (h + 90.0) % 360
            t += 8
    heading = heading % 360
    # hills: gentle sinusoid altitude
    s_pos = 0.0
    v = 10.0
    alt = np.zeros(n)
    sp = np.zeros(n)
    pw = np.zeros(n)
    brake = np.zeros(n, bool)
    cad = np.zeros(n)
    base = 260.0 + 40 * np.sin(np.arange(n) / 150.0)
    coast = np.zeros(n, bool)
    for c0 in range(700, n, 900):
        coast[c0:c0 + 25] = True
    for ts in turn_starts:
        brake[max(ts - 6, 0):ts] = True
    alt_fn = lambda s: 15.0 * np.sin(s / 1500.0) + 8.0 * np.sin(s / 400.0)
    for i in range(n):
        P = 0.0 if coast[i] else base[i] + (rng.normal(0, 15) if noise else 0)
        pw[i] = max(P, 0)
        cad[i] = 0 if coast[i] else 90
        sp[i] = v
        alt[i] = alt_fn(s_pos)
        hw = wind * np.cos(np.radians(WIND_FROM - heading[i]))
        for _ in range(10):  # 0.1 s substeps
            ds = v * 0.1
            sin_t = (alt_fn(s_pos + 1.0) - alt_fn(s_pos)) / 1.0
            cos_t = np.sqrt(1 - sin_t ** 2)
            va = v + hw
            F = pw[i] * EFF / max(v, 1.0) - crr * MASS * G * cos_t - MASS * G * sin_t - 0.5 * rho * cda * va * abs(va)
            if brake[i]:
                F -= 90.0
            v = max(v + 0.1 * F / m_eff, 0.5)
            s_pos += ds
    # integrate position
    east = np.cumsum(sp * np.sin(np.radians(heading)))
    north = np.cumsum(sp * np.cos(np.radians(heading)))
    lat = 53.0 + north / 111195.0
    lon = -2.0 + east / (111195.0 * np.cos(np.radians(53.0)))
    t0 = pd.Timestamp("2026-06-13 12:00:00")
    df = pd.DataFrame({
        "t_s": np.arange(n, dtype=float),
        "timestamp": pd.date_range(t0, periods=n, freq="1s"),
        "lat": lat, "lon": lon,
        "speed": sp + (rng.normal(0, 0.03, n) if noise else 0),
        "power": pw, "cadence": cad,
        "alt": alt + (rng.normal(0, 0.15, n) if noise else 0),
        "dist": np.cumsum(sp), "temp": TEMP, "hr": 150.0, "lap": 1, "gap": False,
    })
    ride = Ride("synthetic", df, pd.DataFrame())
    return ride, brake, coast, heading


def _cfg(**kw):
    base = dict(mass_kg=MASS, crr=TRUE_CRR, drivetrain_eff=EFF, bootstrap_n=60)
    base.update(kw)
    return CdAConfig(**base)


def _weather():
    return constant_weather(temp_c=TEMP, wind_ms=WIND_TRUE / 0.7, wind_from_deg=WIND_FROM, pressure_hpa=1013.25, rh=0.6)


def test_recovers_cda_with_wind():
    ride, *_ = synth_ride()
    res = analyse_ride(ride, _cfg(wind_scale=0.7), weather=_weather())
    assert res.cda == pytest.approx(TRUE_CDA, rel=0.05)
    assert res.cda_ci[0] <= res.cda <= res.cda_ci[1]
    assert res.valid_pct > 30


def test_recovers_cda_manual_wind():
    ride, *_ = synth_ride(seed=2)
    res = analyse_ride(ride, _cfg(wind_mode="manual", manual_wind_ms=WIND_TRUE, manual_wind_from_deg=WIND_FROM))
    assert res.cda == pytest.approx(TRUE_CDA, rel=0.05)


def test_ignoring_wind_is_worse_than_modelling_it():
    ride, *_ = synth_ride(seed=3)
    with_wind = analyse_ride(ride, _cfg(), weather=_weather())
    no_wind = analyse_ride(ride, _cfg(wind_mode="none"))
    assert abs(with_wind.cda - TRUE_CDA) <= abs(no_wind.cda - TRUE_CDA) + 0.002


def test_masks_catch_braking_coasting_corners():
    ride, brake, coast, heading = synth_ride(seed=4)
    res = analyse_ride(ride, _cfg(), weather=_weather())
    s = res.series
    # injected braking seconds (those where speed is moving) are excluded from the valid mask
    assert (~s["valid"].to_numpy()[brake]).mean() > 0.95
    assert s["mask_braking"].to_numpy()[brake].mean() > 0.5
    assert s["mask_coasting"].to_numpy()[coast].mean() > 0.95
    turning = np.abs(np.diff(heading, prepend=heading[0])) > 5
    assert s["mask_cornering"].to_numpy()[turning].mean() > 0.8
    rep = res.mask_report.set_index("reason")
    assert rep.loc["braking", "seconds"] > 0 and rep.loc["coasting", "seconds"] > 0


def test_braking_would_bias_without_masks():
    ride, *_ = synth_ride(seed=5)
    cfg = _cfg()
    cfg.masks.braking_aero_w = 1e9
    cfg.masks.braking_margin_ms2 = 1e9
    cfg.masks.dilate_s = {k: 0.0 for k in cfg.masks.dilate_s}
    cfg.masks.max_heading_rate_dps = 1e9
    cfg.masks.max_lateral_accel = 1e9
    cfg.masks.max_accel_ms2 = 1e9
    cfg.masks.min_power_w = 0.0
    cfg.masks.min_cadence_rpm = 0.0
    biased = analyse_ride(ride, cfg, weather=_weather())
    clean = analyse_ride(ride, _cfg(), weather=_weather())
    assert abs(clean.cda - TRUE_CDA) < 0.012
    assert abs(clean.cda - TRUE_CDA) <= abs(biased.cda - TRUE_CDA) + 0.002


def test_wind_split_and_scan():
    ride, *_ = synth_ride(seed=6)
    res = analyse_ride(ride, _cfg(wind_scale=0.7), weather=_weather(), scan_wind=True)
    assert len(res.wind_split) == 3
    scan = res.wind_scan
    assert scan is not None and len(scan) > 5
    best = best_wind_scale(scan)
    assert 0.4 < best["by_loss"] < 1.0


def test_joint_crr_runs_and_flags():
    ride, *_ = synth_ride(seed=7)
    res = analyse_ride(ride, _cfg(fit_crr=True), weather=_weather())
    assert res.joint is not None and "cond" in res.joint
    assert 0.1 < res.cda < 0.5


def test_energy_breakdown_sane():
    ride, *_ = synth_ride(seed=8)
    res = analyse_ride(ride, _cfg(), weather=_weather())
    e = res.energy
    assert e["aero"] > e["rolling"] > 0
    assert abs(e["unexplained"]) < 0.1 * e["measured_wheel"]


def test_virtual_elevation_recovers_cda():
    ride, *_ = synth_ride(seed=9)
    cfg = _cfg(wind_mode="manual", manual_wind_ms=WIND_TRUE, manual_wind_from_deg=WIND_FROM)
    res = analyse_ride(ride, cfg)
    c = fit_cda_ve(res.series, TRUE_CRR, cfg, start=0, end=len(res.series))
    assert c == pytest.approx(TRUE_CDA, rel=0.15)
    ve = virtual_elevation(res.series, TRUE_CDA, TRUE_CRR, cfg)
    assert len(ve) == len(res.series)


def test_no_power_raises():
    ride, *_ = synth_ride(seed=10, n=300)
    ride.df["power"] = np.nan
    with pytest.raises(ValueError):
        analyse_ride(ride, _cfg())


def test_no_gps_degrades_gracefully():
    ride, *_ = synth_ride(seed=11)
    ride.df[["lat", "lon"]] = np.nan
    res = analyse_ride(ride, _cfg(), weather=_weather())
    assert any("GPS" in w for w in res.warnings)
    assert np.isfinite(res.cda)


def test_auto_wind_scale_finds_truth_and_reports_stability():
    ride, *_ = synth_ride(seed=12)
    res = analyse_ride(ride, _cfg(), weather=_weather())  # default wind_scale="auto"
    wa = res.wind_auto
    assert wa is not None and wa["auto"]
    assert wa["used"] == pytest.approx(0.7, abs=0.2)
    assert res.meta["wind_scale_used"] == pytest.approx(wa["used"])
    assert len(wa["thirds"]) == 3 and np.isfinite(wa["third_range"])
    assert abs(wa["head_minus_tail"]) < 0.05
    assert res.cda == pytest.approx(TRUE_CDA, rel=0.05)
    # a manual numeric scale is respected
    man = analyse_ride(ride, _cfg(wind_scale=0.3), weather=_weather())
    assert man.meta["wind_scale_used"] == 0.3 and not man.wind_auto["auto"]


def test_wind_scan_uses_per_scale_mask_and_split():
    from cycling_tools.cda import _scaled_series
    ride, *_ = synth_ride(seed=13)
    cfg = _cfg(wind_scale=0.7)
    cfg.masks.min_airspeed_ms = 8.5  # airspeed mask now depends strongly on the scale
    res = analyse_ride(ride, cfg, weather=_weather())
    scan = wind_scale_scan(res, [0.0, 0.7, 1.4])
    assert scan["n_valid"].nunique() > 1  # masks differ between scales (old code reused the configured-scale mask)
    row = scan.set_index("wind_scale").loc[0.7]
    assert row["n_valid"] == res.n_valid and row["cda"] == pytest.approx(res.cda, rel=1e-9)
    # the head/tail split follows the scaled wind: at scale 0 there is no wind, so no head/tail subsets
    assert np.isnan(scan.set_index("wind_scale").loc[0.0, "cda_head"])
    # re-applying the used scale reproduces the prepared aero terms
    t = _scaled_series(res.series, 0.7, cfg)
    np.testing.assert_allclose(t["x"], res.series["x"])
    np.testing.assert_allclose(t["v_air"], res.series["v_air"])


def test_crr_sensitivity_and_prior_joint_fit():
    ride, *_ = synth_ride(seed=14)
    res = analyse_ride(ride, _cfg(wind_scale=0.7), weather=_weather())
    assert -0.02 < res.crr_sensitivity < -0.005  # about -0.010 CdA per +0.001 Crr
    j = res.joint
    assert j["prior_mean"] == 0.0040 and j["prior_sd"] == 0.0008
    assert 0.0016 < j["crr"] < 0.0064  # within 3 prior sd; the unconstrained fit wandered to 0.008-0.030
    fit = analyse_ride(ride, _cfg(wind_scale=0.7, fit_crr=True), weather=_weather())
    assert fit.crr == pytest.approx(fit.joint["crr"]) and 0.0016 < fit.crr < 0.0064


def test_uncertainty_defaults_and_systematic_range():
    assert CdAConfig().block_s == 600 and CdAConfig().wind_scale == "auto" and CdAConfig().alt_smooth_s == "auto"
    ride, *_ = synth_ride(seed=15)
    res = analyse_ride(ride, _cfg(), weather=_weather())
    sr = res.sys_range
    assert sr["half"] >= max(sr["crr_half"], sr["wind_half"], sr["eff_half"]) - 1e-12
    assert sr["crr_half"] == pytest.approx(abs(res.crr_sensitivity), rel=0.2)
    assert sr["lo"] < res.cda < sr["hi"]
    assert res.meta["ci_block_s"] <= 600


def test_altitude_smoothing_auto_and_override():
    ride, *_ = synth_ride(seed=16)
    flat = Ride("flat", ride.df.assign(alt=ride.df["alt"] * 0.1), pd.DataFrame())
    hilly = Ride("hilly", ride.df.assign(alt=ride.df["alt"] * 5.0), pd.DataFrame())
    assert analyse_ride(flat, _cfg(), weather=_weather()).meta["alt_smooth_used"] >= 41
    assert analyse_ride(hilly, _cfg(), weather=_weather()).meta["alt_smooth_used"] == 21
    assert analyse_ride(flat, _cfg(alt_smooth_s=11), weather=_weather()).meta["alt_smooth_used"] == 11


def test_virtual_elevation_skips_masked_rows_consistently():
    ride, *_ = synth_ride(seed=17)
    cfg = _cfg(wind_mode="manual", manual_wind_ms=WIND_TRUE, manual_wind_from_deg=WIND_FROM)
    res = analyse_ride(ride, cfg)
    s = res.series.copy()
    base = virtual_elevation(s, TRUE_CDA, TRUE_CRR, cfg)
    # garbage in rows the fit excludes (coasting/braking/etc.) must not move the virtual elevation
    bad = np.flatnonzero(~s["valid"].to_numpy())[:40]
    assert len(bad) == 40
    s2 = s.copy()
    s2.loc[s2.index[bad], "p_wheel"] = 3000.0
    s2.loc[s2.index[bad], "accel"] = 2.0
    np.testing.assert_allclose(virtual_elevation(s2, TRUE_CDA, TRUE_CRR, cfg)["ve"], base["ve"])
    # NaN rows are treated the same way (follow the altitude) instead of silently contributing zero rise
    s3 = s.copy()
    s3.loc[s3.index[1000:1030], "p_wheel"] = np.nan
    s3.loc[s3.index[1000:1030], "valid"] = True
    ve3 = virtual_elevation(s3, TRUE_CDA, TRUE_CRR, cfg)["ve"].to_numpy()
    assert np.isfinite(ve3).all()
    # VE stays linear in CdA (fit_cda_ve relies on this) with masked rows present
    v0, v1, v2 = (virtual_elevation(s, c, TRUE_CRR, cfg)["ve"].to_numpy() for c in (0.1, 0.2, 0.3))
    np.testing.assert_allclose(v2 - v1, v1 - v0, atol=1e-6)


def test_confidence_badge():
    from dataclasses import replace

    from cycling_tools.cda import assess_confidence
    ride, *_ = synth_ride(seed=18)
    res = analyse_ride(ride, _cfg(), weather=_weather())
    assert res.confidence["level"] in ("High", "Medium", "Low")
    assert {i["name"] for i in res.confidence["items"]} >= {"Head/tail gap", "Valid data", "Systematic range"}
    worse = replace(res, wind_auto={**res.wind_auto, "head_minus_tail": 0.12}, sys_range={**res.sys_range, "half": 0.06})
    assert assess_confidence(worse)["level"] == "Low"


def test_real_manchester_tt_lap2():
    from pathlib import Path
    from cycling_tools.cda import try_fetch_weather
    from cycling_tools.fit_io import load_fit
    p = Path(__file__).resolve().parent.parent / "Fit_files" / "Manchester_District_TTA_50_WU_CD.fit"
    if not p.exists():
        pytest.skip("sample file missing")
    ride = load_fit(p)
    wx, warn = try_fetch_weather(ride, [2])
    if wx is None:
        pytest.skip(f"weather offline: {warn}")
    res = analyse_ride(ride, CdAConfig(mass_kg=85.0, crr=0.0031, drivetrain_eff=0.97), weather=wx, laps=[2], scan_wind=True)
    print(res.cda, res.cda_ci, res.meta, res.mask_report, res.wind_split, res.energy, sep="\n")
    assert 0.15 < res.cda < 0.40


def test_streamlit_page_smoke():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "pages" / "1_CdA_Estimator.py"),
                           default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric and at.metric[0].label.startswith("CdA")


def test_series_figure_with_many_mask_runs_is_fast():
    """Regression: shading thousands of excluded runs with add_vrect once took minutes."""
    import time

    import numpy as np
    import pandas as pd

    from cycling_tools import viz_cda
    from cycling_tools.cda import REASONS

    n = 6000
    rng = np.random.default_rng(0)
    cols = {"t_s": np.arange(n, dtype=float), "dist_km": np.arange(n) / 1000.0, "power": 250.0,
            "speed": 11.0, "headwind": 0.0, "rolling_cda": 0.22, "selected": True}
    for k in REASONS:
        cols["mask_" + k] = rng.random(n) < 0.3  # ~900 alternating runs per reason
    series = pd.DataFrame(cols)

    class Res:
        pass

    res = Res()
    res.series, res.cda = series, 0.22
    t0 = time.time()
    fig = viz_cda.series_figure(res, "time")
    assert time.time() - t0 < 5
    assert len(fig.layout.shapes) > 100


def test_map_figure_centre_is_on_route():
    import pandas as pd

    from cycling_tools import viz_cda

    n = 200
    series = pd.DataFrame({"lat": 53.2 + 0.0003 * pd.Series(range(n)), "lon": -2.3 + 0.0004 * pd.Series(range(n)),
                           "rolling_cda": 0.21, "speed": 10.0, "power": 200.0, "selected": True})

    class Res:
        pass

    res = Res()
    res.series = series
    fig = viz_cda.map_figure(res)
    assert abs(fig.layout.map.center.lat - 53.23) < 0.05
    assert 5 < fig.layout.map.zoom < 17
