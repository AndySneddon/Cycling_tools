"""CdA estimator page."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from cycling_tools import viz_cda as vz
from cycling_tools.cda import (
    REASON_LABELS, CdAConfig, MaskConfig, analyse_ride, best_wind_scale, fit_cda_ve, try_fetch_weather,
    wind_scale_scan,
)
from cycling_tools.fit_io import load_fit
from cycling_tools.profile import RiderProfile
from cycling_tools.weather import constant_weather

st.set_page_config(page_title="CdA Estimator", layout="wide")
st.title("CdA estimator")
st.caption("Full dynamic power balance (aero, rolling, climbing, acceleration) with weather-corrected wind and air "
           "density. Coasting, braking, cornering and stops are masked, then CdA is fitted robustly (Huber).")

FIT_DIR = ROOT / "Fit_files"
profile = RiderProfile.load()


@st.cache_data(show_spinner="Parsing FIT file...")
def _load(key: str, data: bytes | None):
    return load_fit(FIT_DIR / key) if data is None else load_fit(data, name=key)


@st.cache_data(show_spinner="Fetching weather...")
def _weather(key: str, _ride, laps: tuple):
    return try_fetch_weather(_ride, list(laps) or None)


@st.cache_data(show_spinner="Analysing ride...")
def _analyse(key: str, _ride, cfg_dict: dict, _weather_obj, wx_key: str, laps: tuple, scan: bool):
    cfg_dict = dict(cfg_dict)
    cfg_dict["masks"] = MaskConfig(**cfg_dict["masks"])
    return analyse_ride(_ride, CdAConfig(**cfg_dict), weather=_weather_obj, laps=list(laps) or None, scan_wind=scan)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Ride")
    avail = sorted(p.name for p in FIT_DIR.glob("*") if p.suffix.lower() == ".fit")
    up = st.file_uploader("Upload a FIT file", type=["fit"])
    chosen = st.selectbox("...or choose from Fit_files/", avail) if avail else None

if up is None and chosen is None:
    st.info("Upload a FIT file or put one in Fit_files/.")
    st.stop()
try:
    ride = _load(up.name, up.getvalue()) if up is not None else _load(chosen, None)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read FIT file: {e}")
    st.stop()

nlap = int(ride.df["lap"].max())
with st.sidebar:
    if nlap > 1:
        st.subheader("Laps to analyse")
        sel_laps = []
        for i in range(1, nlap + 1):
            row = ride.laps[ride.laps["lap"] == i]
            info = ""
            if len(row):
                r0 = row.iloc[0]
                info = f" ({r0['duration_s'] / 60:.0f} min, {r0['avg_power']:.0f} W)"
            default = i == int(ride.laps.sort_values("duration_s").iloc[-1]["lap"]) if len(ride.laps) else True
            if st.checkbox(f"Lap {i}{info}", value=default, key=f"lap_{ride.name}_{i}"):
                sel_laps.append(i)
        if not sel_laps:
            st.warning("Select at least one lap.")
            st.stop()
    else:
        sel_laps = []

    st.header("Rider & bike")
    mass = st.number_input("System mass (kg)", 40.0, 200.0, float(profile.mass_kg), 0.5)
    crr = st.number_input("Crr", 0.001, 0.015, float(profile.crr), 0.0001, format="%.4f")
    eff = st.number_input("Drivetrain efficiency", 0.90, 1.0, float(profile.drivetrain_eff), 0.005)
    fit_crr = st.checkbox("Fit Crr jointly (needs varied speed)", value=False)

    st.header("Weather / wind")
    src = st.radio("Source", ["Open-Meteo (auto)", "Manual", "None"], index=0)
    wind_scale = st.slider("Wind scale (10 m -> rider)", 0.0, 1.5, float(profile.wind_scale), 0.05,
                           disabled=src != "Open-Meteo (auto)")
    man_ws = man_wd = man_t = None
    if src == "Manual":
        man_ws = st.number_input("Wind speed at rider (m/s)", 0.0, 25.0, 0.0, 0.5)
        man_wd = st.number_input("Wind from (deg, 0 = N)", 0.0, 359.0, 0.0, 5.0)
        man_t = st.number_input("Air temperature (C)", -20.0, 45.0, 15.0, 0.5)
    do_scan = st.checkbox("Scan wind scale (diagnostic)", value=False, disabled=src == "None")

    with st.expander("Smoothing & fit"):
        sm_v = st.slider("Speed smoothing (s)", 1, 15, 5)
        sm_p = st.slider("Power smoothing (s)", 1, 15, 5)
        sm_a = st.slider("Altitude smoothing (s)", 5, 61, 21, 2)
        roll_w = st.slider("Rolling CdA window (s)", 30, 600, 120, 10)
        boots = st.slider("Bootstrap resamples", 0, 500, 150, 10)

    mc = MaskConfig()
    with st.expander("Mask thresholds"):
        mc.min_speed_ms = st.number_input("Min speed (m/s)", 0.0, 15.0, mc.min_speed_ms, 0.5)
        mc.min_power_w = st.number_input("Min power (W, below = coasting)", 0.0, 200.0, mc.min_power_w, 5.0)
        mc.min_cadence_rpm = st.number_input("Min cadence (rpm)", 0.0, 60.0, mc.min_cadence_rpm, 1.0)
        mc.braking_aero_w = st.number_input("Braking: implied aero below -X W", 0.0, 200.0, mc.braking_aero_w, 5.0)
        mc.braking_margin_ms2 = st.number_input("Braking: extra decel margin (m/s²)", 0.0, 1.0, mc.braking_margin_ms2, 0.05)
        mc.max_heading_rate_dps = st.number_input("Max heading rate (deg/s)", 0.5, 30.0, mc.max_heading_rate_dps, 0.5)
        mc.max_lateral_accel = st.number_input("Max lateral accel (m/s²)", 0.2, 5.0, mc.max_lateral_accel, 0.1)
        mc.max_grade = st.number_input("Max |grade|", 0.01, 0.4, mc.max_grade, 0.01)
        mc.max_accel_ms2 = st.number_input("Max |acceleration| (m/s²)", 0.1, 2.0, mc.max_accel_ms2, 0.05)
        mc.min_airspeed_ms = st.number_input("Min airspeed (m/s)", 0.0, 10.0, mc.min_airspeed_ms, 0.5)
        dil = st.slider("Dilate every mask by (s)", 0, 15, 3)
        if dil != 3:
            mc.dilate_s = {k: float(dil) for k in mc.dilate_s}

# ------------------------------------------------------------------ weather
warnings_ui: list[str] = []
weather = None
wind_mode = "none"
rho_override = None
if src == "Open-Meteo (auto)":
    if not ride.has_gps:
        warnings_ui.append("No GPS in this ride, so weather and wind cannot be applied.")
    else:
        weather, wmsg = _weather(ride.name, ride, tuple(sel_laps))
        if weather is None:
            warnings_ui.append(wmsg or "Weather unavailable.")
        else:
            wind_mode = "weather"
elif src == "Manual":
    wind_mode = "manual"
    weather = constant_weather(temp_c=man_t, wind_ms=0.0)

cfg_obj = CdAConfig(
    mass_kg=mass, crr=crr, fit_crr=fit_crr, drivetrain_eff=eff, wind_mode=wind_mode, wind_scale=wind_scale,
    manual_wind_ms=man_ws or 0.0, manual_wind_from_deg=man_wd or 0.0, speed_smooth_s=sm_v,
    power_smooth_s=sm_p, alt_smooth_s=sm_a, rolling_window_s=roll_w, bootstrap_n=boots, masks=mc,
)
cfg_dict = asdict(cfg_obj)
wx_key = (weather.source if weather is not None else "none") + (f"{man_t}" if src == "Manual" else "")
try:
    res = _analyse(ride.name, ride, cfg_dict, weather, wx_key, tuple(sel_laps), bool(do_scan and wind_mode != "none"))
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Analysis failed: {e}")
    st.stop()

for w in warnings_ui + res.warnings:
    st.warning(w)

# ------------------------------------------------------------------ headline
lo, hi = res.cda_ci
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("CdA (m²)", f"{res.cda:.3f}")
c2.metric("95% CI", f"{lo:.3f} - {hi:.3f}" if np.isfinite(lo) else "n/a")
c3.metric("Crr used", f"{res.crr:.4f}")
c4.metric("Valid data", f"{res.valid_pct:.0f}%", f"{res.n_valid} s")
c5.metric("Air density", f"{res.meta['rho_mean']:.3f} kg/m³")
st.caption(f"Wind: {res.meta['wind_source']} (scale {res.meta['wind_scale_used']:.2f}); density: {res.meta['rho_source']}; "
           f"residual RMS {res.resid_rms_w:.0f} W. The CI reflects sampling noise only, not systematic error "
           "(wind, mass, Crr, power-meter calibration).")
if res.joint and fit_crr is False:
    with st.expander("Joint CdA + Crr fit (diagnostic)"):
        j = res.joint
        st.write(f"CdA {j['cda']:.3f}, Crr {j['crr']:.4f}; condition number {j['cond']:.1f}, "
                 f"x/r correlation {j['corr']:.2f}. " + ("Poorly conditioned: do not trust." if j["ill_conditioned"] else "Reasonably conditioned."))

tabs = st.tabs(["Series", "Map", "Diagnostics", "Laps & energy", "Virtual elevation", "Masks"])
with tabs[0]:
    xa = st.radio("X axis", ["time", "distance"], horizontal=True)
    st.plotly_chart(vz.series_figure(res, xa, show_masks=st.checkbox("Shade excluded regions", True)),
                    width="stretch")
with tabs[1]:
    if ride.has_gps:
        st.plotly_chart(vz.map_figure(res), width="stretch")
    else:
        st.info("No GPS data.")
with tabs[2]:
    a, b = st.columns(2)
    a.subheader("Implied CdA vs speed")
    a.plotly_chart(vz.scatter_figure(res, "speed"), width="stretch")
    b.subheader("Implied CdA vs headwind")
    b.plotly_chart(vz.scatter_figure(res, "headwind"), width="stretch")
    st.subheader("CdA distribution")
    st.plotly_chart(vz.histogram_figure(res), width="stretch")
    a, b = st.columns(2)
    a.subheader("Headwind vs tailwind")
    a.dataframe(res.wind_split.round(3), hide_index=True)
    ws = res.wind_split.set_index("condition")["cda"]
    h, t = ws.iloc[0], ws.iloc[2]
    if np.isfinite(h) and np.isfinite(t):
        if t - h > 0.03:
            a.info("Tailwind CdA is higher than headwind CdA: the applied wind is probably too strong (lower the wind scale).")
        elif h - t > 0.03:
            a.info("Headwind CdA is higher than tailwind CdA: the applied wind is probably too weak (raise the wind scale).")
        else:
            a.success("Headwind and tailwind CdA agree: wind scale looks reasonable.")
    b.subheader("By speed bin")
    b.dataframe(res.speed_split.round(3), hide_index=True)
    if res.wind_scan is not None:
        st.subheader("Wind scale scan")
        st.plotly_chart(vz.wind_scan_figure(res.wind_scan), width="stretch")
        bw = best_wind_scale(res.wind_scan)
        st.write(f"Best scale by residual: **{bw['by_loss']:.2f}**; where head/tail CdA balance: "
                 f"**{bw['by_balance']:.2f}**" if np.isfinite(bw["by_balance"]) else f"Best scale by residual: **{bw['by_loss']:.2f}**")
with tabs[3]:
    st.subheader("Per lap")
    st.plotly_chart(vz.laps_figure(res), width="stretch")
    st.dataframe(res.laps.round(3), hide_index=True)
    st.subheader("Where the watts go")
    which = st.radio("Samples", ["Valid fit samples", "Whole selection"], horizontal=True)
    st.plotly_chart(vz.energy_figure(res, which == "Whole selection"), width="stretch")
    e = res.energy_selection if which == "Whole selection" else res.energy
    if e:
        tot = e["measured_wheel"]
        st.write(" | ".join(f"{k.title()}: {e[k]:.0f} W ({100 * e[k] / tot:.0f}%)" for k in ("aero", "rolling", "climb", "accel")))
with tabs[4]:
    st.caption("Virtual elevation (Chung method): the right CdA makes the VE profile match the true altitude. "
               "Works best on loops or repeated laps with the same start/end elevation.")
    n_sel = res.n_selected
    rng = st.slider("Segment (s into selection)", 0, max(n_sel - 1, 1), (0, max(n_sel - 1, 1)))
    cfg_ve = res.cfg
    s_sel = res.series[res.series["selected"]].reset_index(drop=True)
    key = f"ve_cda_{ride.name}"
    if key not in st.session_state:
        st.session_state[key] = float(np.clip(res.cda, 0.1, 0.6))
    if st.button("Fit CdA to altitude (least squares)"):
        st.session_state[key] = float(np.clip(fit_cda_ve(s_sel, res.crr, cfg_ve, rng[0], rng[1] + 1), 0.1, 0.6))
    cda_ve = st.slider("CdA (m²)", 0.10, 0.60, step=0.001, key=key, format="%.3f")
    st.plotly_chart(vz.ve_figure(res, cda_ve, res.crr, cfg_ve, rng[0], rng[1] + 1,
                                 "distance" if ride.has_gps or True else "time"), width="stretch")
with tabs[5]:
    st.dataframe(res.mask_report[["label", "seconds", "pct", "unique_seconds"]].round(1), hide_index=True)
    st.caption("Each reason counts every sample it flags (so rows overlap); 'unique' attributes each sample to the first reason only.")

# ------------------------------------------------------------------ save
st.divider()
if st.button("Save CdA to rider profile"):
    p = RiderProfile.load()
    p.cda = round(float(res.cda), 4)
    p.crr = round(float(res.crr), 5)
    p.wind_scale = float(wind_scale)
    p.mass_kg = float(mass)
    p.drivetrain_eff = float(eff)
    try:
        p.save()
        st.success(f"Saved CdA {p.cda:.3f}, Crr {p.crr:.4f}, wind scale {p.wind_scale:.2f}, mass {p.mass_kg:.1f} kg.")
    except OSError as e:
        st.error(f"Could not write profile: {e}")
