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

from cycling_tools.branding import PAGE_ICON, apply_branding, page_header

from cycling_tools import viz_cda as vz
from cycling_tools.cda import (
    CRR_STEP, EFF_REL, CdAConfig, MaskConfig, analyse_ride, best_wind_scale, fit_cda_ve, try_fetch_weather,
)
from cycling_tools.fit_io import load_fit
from cycling_tools.physics import normalised_power
from cycling_tools.profile import RiderProfile
from cycling_tools.weather import constant_weather

st.set_page_config(page_title="CdA Estimator", page_icon=PAGE_ICON, layout="wide")
apply_branding()
page_header("CdA Estimator",
            "Full dynamic power balance (aero, rolling, climbing, acceleration) with weather-corrected wind and air "
            "density. Coasting, braking, cornering and stops are masked, then CdA is fitted robustly.",
            eyebrow="Tool 1 · Aerodynamics")

FIT_DIR = ROOT / "Fit_files"
profile = RiderProfile.load()


@st.cache_data(show_spinner="Parsing FIT file...", max_entries=8)
def _load(key: str, data: bytes | None):
    return load_fit(FIT_DIR / key) if data is None else load_fit(data, name=key)


@st.cache_data(show_spinner="Fetching weather...", max_entries=16)
def _weather(key: str, _ride, laps: tuple):
    return try_fetch_weather(_ride, list(laps) or None)


@st.cache_data(show_spinner="Analysing ride...", max_entries=16)
def _analyse(key: str, _ride, cfg_dict: dict, _weather_obj, wx_key: str, laps: tuple):
    cfg_dict = dict(cfg_dict)
    cfg_dict["masks"] = MaskConfig(**cfg_dict["masks"])
    return analyse_ride(_ride, CdAConfig(**cfg_dict), weather=_weather_obj, laps=list(laps) or None)


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
    crr = st.number_input("Crr", 0.001, 0.015, float(profile.crr), 0.0001, format="%.4f",
                          help="Typical: smooth tarmac 0.003-0.004, real roads 0.004-0.006. CdA and Crr are only valid "
                               "as a PAIR: the race planner uses the profile Crr together with the profile CdA, so "
                               "use the Crr you want to race with. Each +0.001 Crr lowers the fitted CdA by about 0.010.")
    st.caption("Smooth tarmac 0.003-0.004, real roads 0.004-0.006. CdA is only valid with this Crr.")
    eff = st.number_input("Drivetrain efficiency", 0.90, 1.0, float(profile.drivetrain_eff), 0.005)
    fit_crr = st.checkbox("Joint CdA + Crr fit (Gaussian prior on Crr)", value=False,
                          help="Crr ~ N(0.0040, 0.0008). An unconstrained joint fit returns absurd Crr (0.008-0.030) "
                               "because CdA and Crr are almost collinear; treat this as a diagnostic.")

    st.header("Weather / wind")
    src = st.radio("Source", ["Open-Meteo (auto)", "Manual", "None"], index=0)
    auto_ws = st.checkbox("Automatic wind scale (minimise residual)", value=True, disabled=src != "Open-Meteo (auto)",
                          help="Scans 0 to 1.2 and picks the scale with the lowest Huber loss. Untick to set it by hand.")
    wind_scale = st.slider("Wind scale (10 m -> rider)", 0.0, 1.5, float(np.clip(profile.wind_scale, 0.0, 1.5)), 0.05,
                           disabled=src != "Open-Meteo (auto)" or auto_ws)
    man_ws = man_wd = man_t = None
    if src == "Manual":
        man_ws = st.number_input("Wind speed at rider (m/s)", 0.0, 25.0, 0.0, 0.5)
        man_wd = st.number_input("Wind from (deg, 0 = N)", 0.0, 359.0, 0.0, 5.0)
        man_t = st.number_input("Air temperature (C)", -20.0, 45.0, 15.0, 0.5)

    with st.expander("Smoothing & fit"):
        sm_v = st.slider("Speed smoothing (s)", 1, 15, 5)
        sm_p = st.slider("Power smoothing (s)", 1, 15, 5)
        auto_alt = st.checkbox("Automatic altitude smoothing", value=True,
                               help="41 s when the selected laps are flat (altitude range < 40 m), otherwise 21 s.")
        sm_a = st.slider("Altitude smoothing (s)", 5, 61, 21, 2, disabled=auto_alt)
        roll_w = st.slider("Rolling CdA window (s)", 30, 600, 120, 10)
        boots = st.slider("Bootstrap resamples", 0, 500, 150, 10)
        block = st.slider("Bootstrap block length (s)", 30, 900, 600, 30,
                          help="Longer blocks respect the autocorrelation of the errors; 30 s blocks understate the CI.")

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
    mass_kg=mass, crr=crr, fit_crr=fit_crr, drivetrain_eff=eff, wind_mode=wind_mode,
    wind_scale="auto" if auto_ws else float(wind_scale),
    manual_wind_ms=man_ws or 0.0, manual_wind_from_deg=man_wd or 0.0, speed_smooth_s=sm_v,
    power_smooth_s=sm_p, alt_smooth_s="auto" if auto_alt else int(sm_a), rolling_window_s=roll_w, bootstrap_n=boots,
    block_s=int(block), masks=mc,
)
cfg_dict = asdict(cfg_obj)
wx_key = (weather.source if weather is not None else "none") + (f"{man_t}" if src == "Manual" else "")
try:
    res = _analyse(ride.name, ride, cfg_dict, weather, wx_key, tuple(sel_laps))
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
sr = res.sys_range or {}
conf = res.confidence or {"level": "n/a", "items": []}
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("CdA (m²)", f"{res.cda:.3f}")
c2.metric("95% CI (sampling)", f"{lo:.3f}–{hi:.3f}" if np.isfinite(lo) else "n/a",
          help=f"Moving-block bootstrap, {res.meta.get('ci_block_s', '?')} s blocks. Sampling noise only.")
c3.metric("Systematic range", f"±{sr['half']:.3f}" if sr else "n/a",
          help=f"Crr ±{CRR_STEP:.3f}: ±{sr.get('crr_half', 0):.3f}; wind-scale optimum across thirds: "
               f"±{sr.get('wind_half', 0):.3f}; drivetrain efficiency ±{EFF_REL * 100:.1f}%: ±{sr.get('eff_half', 0):.3f}. "
               "Combined in quadrature. Power-meter bias is NOT included.")
c4.metric("Crr used", f"{res.crr:.4f}", f"{res.crr_sensitivity:+.3f} CdA per +0.001 Crr", delta_color="off")
c5.metric("Valid data", f"{res.valid_pct:.0f}%", f"{res.n_valid} s")
c6.metric("Confidence", conf["level"], help="From head/tail CdA gap, wind-scale stability, valid data and systematic range (see below).")
with st.expander(f"Why {conf['level']} confidence?"):
    for it in conf["items"]:
        mark = {2: "good", 1: "fair", 0: "poor", None: "n/a"}[it["grade"]]
        st.write(f"- **{it['name']}** ({mark}): {it['text']}")
    st.caption("The CI and systematic range do not include power-meter bias or mass error: a 2% power-meter error "
               "is about a 2% CdA error. CdA is only valid together with the Crr used.")

# Power summary for the selected laps (all samples, including coasting), for use in the Race Planner
sel = res.series[res.series["selected"]]
sel_power = sel["power"].dropna().to_numpy(dtype=float)
if sel_power.size >= 30:
    avg_p = float(sel_power.mean())
    np_p = normalised_power(sel_power)
    vi_p = np_p / avg_p if avg_p > 0 else float("nan")
    secs = int(sel_power.size)
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Normalised power", f"{np_p:.0f} W", help="30 s rolling 4th-power mean over the selected laps")
    p2.metric("Average power", f"{avg_p:.0f} W", help="Includes coasting (zeros), as in a race")
    p3.metric("Variability index", f"{vi_p:.3f}", help="NP / average power. Enter this as the race VI in the Race Planner")
    p4.metric("Duration", f"{secs // 3600}:{secs % 3600 // 60:02d}:{secs % 60:02d}")
    dist_km = float(sel["dist_km"].iloc[-1] - sel["dist_km"].iloc[0]) if "dist_km" in sel else float("nan")
    p5.metric("Average speed", f"{dist_km / (secs / 3600):.1f} km/h" if np.isfinite(dist_km) and secs else "n/a")
    st.caption("Use these in the Race Planner: set the target NP and expected VI to the values above.")
wa = res.wind_auto
wind_txt = (f"Wind: {res.meta['wind_source']} (scale {res.meta['wind_scale_used']:.2f}"
            + (", automatic" if wa and wa["auto"] else "") + ")")
if wa:
    wind_txt += (f"; head-minus-tail CdA {wa['head_minus_tail']:+.3f}; optimum across thirds "
                 + " / ".join(f"{t:.2f}" for t in wa["thirds"]))
st.caption(f"{wind_txt}; density: {res.meta['rho_source']} ({res.meta['rho_mean']:.3f} kg/m³); "
           f"altitude smoothing {res.meta.get('alt_smooth_used', '-')} s; residual RMS {res.resid_rms_w:.0f} W. "
           "The CI reflects sampling noise only; the systematic range adds Crr, wind scale and efficiency but not "
           "power-meter bias.")
if res.joint and fit_crr is False:
    with st.expander("Joint CdA + Crr fit with Crr prior (diagnostic)"):
        j = res.joint
        st.write(f"With a Gaussian prior Crr ~ N({j['prior_mean']:.4f}, {j['prior_sd']:.4f}): CdA {j['cda']:.3f}, "
                 f"Crr {j['crr']:.4f} ± {j['crr_sd']:.4f}; condition number {j['cond']:.1f}, x/r correlation "
                 f"{j['corr']:.2f}. " + ("The data barely inform Crr here (estimate is mostly the prior): do not "
                                           "trust it." if j["ill_conditioned"] else "The data do inform Crr."))
        st.caption("Rule of thumb: CdA falls by about 0.010 per +0.001 Crr. Use the fixed-Crr CdA above, with a Crr "
                   "you will also race with.")

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
        bw = best_wind_scale(res.wind_scan)
        st.plotly_chart(vz.wind_scan_figure(res.wind_scan, opt=bw["by_loss"]), width="stretch")
        st.write(f"Best scale by residual: **{bw['by_loss']:.2f}**; where head/tail CdA balance: "
                 f"**{bw['by_balance']:.2f}**" if np.isfinite(bw["by_balance"]) else f"Best scale by residual: **{bw['by_loss']:.2f}**")
        if wa:
            st.write(f"Optimum by thirds of the ride: {' / '.join(f'{t:.2f}' for t in wa['thirds'])} "
                     f"(spread {wa['third_range']:.2f}). Each scale in the scan uses its own validity mask.")
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
st.caption("CdA and Crr are saved together: the race planner uses both, and this CdA is only valid with this Crr.")
if st.button("Save CdA + Crr to rider profile"):
    p = RiderProfile.load()
    p.cda = round(float(res.cda), 4)
    p.crr = round(float(res.crr), 5)
    p.wind_scale = round(float(res.meta["wind_scale_used"]), 3) if res.meta.get("wind_mode_used") == "weather" else float(p.wind_scale)
    p.mass_kg = float(mass)
    p.drivetrain_eff = float(eff)
    try:
        p.save()
        st.success(f"Saved as a pair: CdA {p.cda:.3f} with Crr {p.crr:.4f} (wind scale {p.wind_scale:.2f}, "
                   f"mass {p.mass_kg:.1f} kg).")
    except OSError as e:
        st.error(f"Could not write profile: {e}")
