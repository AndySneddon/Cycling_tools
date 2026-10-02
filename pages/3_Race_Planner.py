"""Tool 3: race bike-split planner and pacing optimiser."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cycling_tools import viz_race
from cycling_tools.course import CourseSettings, load_gpx
from cycling_tools.optimise import optimise_pacing
from cycling_tools.profile import RiderProfile
from cycling_tools.simulate import (Environment, chainring_recommendation, setup_recommendation, simulate_even, splits_table,
                                    time_vs_np, what_if, fmt_time)
from cycling_tools.weather import WeatherError, constant_weather, fetch_weather

try:
    from cycling_tools.gearing import CASSETTES
except ImportError:  # pragma: no cover
    CASSETTES = {"Ultegra 11-30 (12sp)": [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30]}

st.set_page_config(page_title="Race Planner", page_icon="🏁", layout="wide")
st.title("🏁 Race Planner")
st.caption("Predict a bike split from a GPX course, a power target and the weather; then optimise pacing.")

COURSES_DIR = ROOT / "courses"


# ----------------------------------------------------------------- cached work
@st.cache_data(show_spinner="Loading course...")
def cached_course(data: bytes, name: str, settings: dict, fill_elev: bool):
    return load_gpx(data, name=name, settings=CourseSettings(**settings), fetch_missing_elevation=fill_elev)


@st.cache_data(show_spinner="Fetching weather...", ttl=3600)
def cached_weather(lat: float, lon: float, start_iso: str, end_iso: str):
    return fetch_weather(lat, lon, pd.Timestamp(start_iso), pd.Timestamp(end_iso))


@st.cache_data(show_spinner=False, max_entries=64)
def cached_even(_course, _rider, _env, key, np_target):
    return simulate_even(_course, _rider, np_target, _env)


@st.cache_data(show_spinner="Computing NP curve...", max_entries=8)
def cached_curve(_course, _rider, _env, key, lo, hi):
    return time_vs_np(_course, _rider, np.linspace(lo, hi, 11), _env)


@st.cache_data(show_spinner="Computing what-ifs...", max_entries=8)
def cached_whatif(_course, _rider, _env, key, np_target):
    return what_if(_course, _rider, np_target, _env)


@st.cache_data(show_spinner="Optimising pacing...", max_entries=8)
def cached_opt(_course, _rider, _env, key, np_target, block_m, bound_pct, smooth):
    return optimise_pacing(_course, _rider, np_target, _env, block_m=block_m, bound_pct=bound_pct, smooth=smooth)


# --------------------------------------------------------------------- sidebar
profile = RiderProfile.load()
with st.sidebar:
    st.header("Rider")
    rider = replace(
        profile,
        mass_kg=st.number_input("System mass (kg)", 40.0, 200.0, float(profile.mass_kg), 0.5),
        cda=st.number_input("CdA (m²)", 0.10, 0.60, float(profile.cda), 0.001, format="%.3f"),
        crr=st.number_input("Crr", 0.001, 0.015, float(profile.crr), 0.0001, format="%.4f"),
        drivetrain_eff=st.number_input("Drivetrain efficiency", 0.90, 1.0, float(profile.drivetrain_eff), 0.005),
        wind_scale=st.number_input("Wind scale (10 m → rider)", 0.2, 1.2, float(profile.wind_scale), 0.05),
        tyre_circumference_m=st.number_input("Wheel circumference (m)", 1.9, 2.3, float(profile.tyre_circumference_m),
                                             0.005, format="%.3f"),
        cadence_flat=st.number_input("Cadence at 200 W (rpm)", 50.0, 120.0, float(profile.cadence_flat), 1.0),
        cadence_per_100w=st.number_input("Cadence change per +100 W (rpm)", -20.0, 40.0,
                                         float(profile.cadence_per_100w), 0.5),
    )
    preset = st.selectbox("Cassette", ["(profile)"] + list(CASSETTES))
    if preset != "(profile)":
        rider.cassette = list(CASSETTES[preset])
    st.caption("Sprockets: " + ", ".join(map(str, rider.cassette)))
    if st.button("Save as rider profile"):
        rider.save()
        st.success("Profile saved.")

    st.header("Course model")
    with st.expander("Smoothing & limits"):
        cs = CourseSettings(
            elev_smooth_m=st.number_input("Elevation smoothing (m)", 0.0, 500.0, 60.0, 10.0),
            max_grade=st.number_input("Max grade clamp (%)", 5.0, 40.0, 20.0, 1.0) / 100.0,
            max_lat_accel=st.number_input("Max lateral accel (m/s²)", 1.0, 8.0, 3.5, 0.1),
            max_descent_speed=st.number_input("Max descent speed (km/h)", 40.0, 110.0, 79.0, 1.0) / 3.6,
        )
    fill_elev = st.checkbox("Look up elevation online if GPX has none", value=True)

# ---------------------------------------------------------------------- course
st.subheader("1. Course")
gpx_files = sorted(COURSES_DIR.glob("*.gpx")) if COURSES_DIR.exists() else []
c1, c2 = st.columns(2)
upload = c1.file_uploader("Upload a GPX file", type=["gpx"])
pick = c2.selectbox("...or pick from courses/", ["(none)"] + [p.name for p in gpx_files])

data, cname = None, None
if upload is not None:
    data, cname = upload.getvalue(), Path(upload.name).stem
elif pick != "(none)":
    data, cname = (COURSES_DIR / pick).read_bytes(), Path(pick).stem
if data is None:
    st.info("Upload a GPX file (or drop one into the `courses/` folder) to get started.")
    st.stop()

try:
    course = cached_course(data, cname, {k: v for k, v in cs.__dict__.items()}, fill_elev)
except Exception as exc:  # noqa: BLE001 - show any parse/elevation problem to the user
    st.error(f"Could not load course: {exc}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Distance", f"{course.length_m / 1000:.1f} km")
m2.metric("Ascent", f"{course.ascent_m:.0f} m")
m3.metric("Descent", f"{course.descent_m:.0f} m")
m4.metric("Corner-limited segments", f"{int((course.vcap < 0.9 * cs.max_descent_speed).sum())}")

# --------------------------------------------------------------------- inputs
st.subheader("2. Power and weather")
i1, i2 = st.columns([1, 2])
np_race = i1.number_input("Target normalised power (W)", 50.0, 600.0, 250.0, 1.0)
race_vi = i1.number_input(
    "Expected race variability index (VI)", 1.00, 1.15, 1.02, 0.01,
    help="Real races are not dead-even: you coast, brake and surge, so average power is NP / VI. "
         "Almere (flat, 178 km) was 1.024. Set 1.00 for a perfectly constant-power ride.")
# The simulator rides at constant power, so hold the average power a variable ride with this NP would have
np_target = np_race / race_vi
mode = i2.radio("Weather", ["Open-Meteo (forecast / archive)", "Manual wind", "None (still air)"], horizontal=True)

env = Environment()
env_key = ("none",)
if mode.startswith("Open-Meteo"):
    w1, w2, w3 = st.columns(3)
    race_date = w1.date_input("Race date", date.today() + timedelta(days=2))
    start_t = w2.time_input("Start time (local)", time(8, 0))
    tzname = w3.text_input("Timezone", "Europe/London")
    try:
        tz = ZoneInfo(tzname)
        start_utc = datetime.combine(race_date, start_t, tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        lat, lon = course.centre
        wx = cached_weather(round(lat, 3), round(lon, 3), start_utc.isoformat(),
                            (start_utc + timedelta(hours=6)).isoformat())
        env = Environment(wx, pd.Timestamp(start_utc), rider.wind_scale)
        env_key = ("om", round(lat, 3), round(lon, 3), start_utc.isoformat(), rider.wind_scale)
        h = wx.at([start_utc])
        st.caption(f"{wx.source}: at start {h['temp_c'].iloc[0]:.1f} °C, wind {h['wind_ms'].iloc[0]:.1f} m/s "
                   f"from {h['wind_dir'].iloc[0]:.0f}° (×{rider.wind_scale:.2f} at rider height)")
    except (WeatherError, Exception) as exc:  # noqa: BLE001
        st.warning(f"Weather unavailable ({exc}); using still air.")
elif mode.startswith("Manual"):
    w1, w2, w3, w4 = st.columns(4)
    ws = w1.number_input("Wind speed at rider (km/h)", 0.0, 80.0, 10.0, 1.0)
    wd = w2.number_input("Wind from (deg, 0 = N, 90 = E)", 0.0, 360.0, 180.0, 5.0)
    temp = w3.number_input("Temperature (°C)", -10.0, 45.0, 15.0, 1.0)
    pres = w4.number_input("Pressure (hPa)", 900.0, 1050.0, 1013.0, 1.0)
    env = Environment(constant_weather(temp, ws / 3.6, wd, pres), None, 1.0)
    env_key = ("manual", ws, wd, temp, pres)

ckey = (course.name, course.n_seg, round(course.length_m), tuple(sorted(cs.__dict__.items())))
rkey = (rider.mass_kg, rider.cda, rider.crr, rider.drivetrain_eff)
key = (ckey, rkey, env_key)

# --------------------------------------------------------------------- results
even = cached_even(course, rider, env, key, np_target)
st.subheader("3. Predicted split (even pacing)")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Time", fmt_time(even.total_time_s))
r2.metric("Average speed", f"{even.avg_speed_kmh:.1f} km/h")
r3.metric("Average power", f"{even.avg_power:.0f} W")
r4.metric("Normalised power (target)", f"{np_race:.0f} W")
r5.metric("Assumed VI", f"{race_vi:.2f}")

st.plotly_chart(viz_race.fig_elevation(course, even.headwind if env.weather is not None else None),
                use_container_width=True)
colour = st.radio("Colour map by", ["speed", "power", "grade"], horizontal=True)
st.plotly_chart(viz_race.fig_map(course, even, colour), use_container_width=True)

split_km = st.selectbox("Split length", [1, 5, 10], index=0, format_func=lambda k: f"{k} km")
sp = splits_table(even, split_km * 1000.0)
st.dataframe(sp[["split", "split_time", "cum_time", "speed_kmh", "power_w", "elev_gain_m", "net_elev_m", "headwind_ms"]]
             .round(1), use_container_width=True, hide_index=True)

# ------------------------------------------------------------- NP sensitivity
st.subheader("4. How much does NP matter?")
curve = cached_curve(course, rider, env, key, float(round(np_target * 0.75)), float(round(np_target * 1.25)))
sel = st.slider("Try a different NP (W)", float(curve["np_w"].min()), float(curve["np_w"].max()), float(np_target), 1.0)
alt = cached_even(course, rider, env, key, sel)
s1, s2, s3 = st.columns(3)
s1.metric("Time at selected NP", fmt_time(alt.total_time_s), f"{alt.total_time_s - even.total_time_s:+.0f} s vs target")
s2.metric("Average speed", f"{alt.avg_speed_kmh:.1f} km/h")
near = curve.iloc[(curve["np_w"] - np_target).abs().argmin()]
s3.metric("Seconds per +5 W", f"{near['s_saved_per_5w']:.0f} s")
st.plotly_chart(viz_race.fig_np_curve(curve, sel), use_container_width=True)

st.markdown("**What-if (even pacing, same NP unless stated)**")
wi = cached_whatif(course, rider, env, key, np_target)
st.dataframe(wi[["scenario", "time", "saved_s"]].rename(columns={"saved_s": "time saved (s)"}).round(1),
             use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- gearing
st.subheader("5. Chainring recommendation")
rings = list(range(46, 65))
rec = chainring_recommendation(even, rings, rider.cassette)
best = int(rec.iloc[0]["chainring"])
st.success(f"Best chainring for this course at {np_target:.0f} W NP: **{best}T** "
           f"(cassette {rider.cassette[0]}-{rider.cassette[-1]}).")
st.dataframe(rec.round(1), use_container_width=True, hide_index=True)
top = sorted(rec["chainring"].head(5).astype(int))
st.plotly_chart(viz_race.fig_gear_heatmap(even, top, rider.cassette), use_container_width=True)

st.markdown("**1x vs 2x for this course**")
st.caption("Compare the setups you actually race. The 2x score includes an assumed front-derailleur aero penalty "
           "(see `DEFAULT_SETUP_WEIGHTS` in gearing.py); set `aero_delta_cda` to 0 to ignore it.")
from cycling_tools.gearing import parse_setup
setup_text = st.text_input("Setups to compare (comma separated)", "58, 60, 56/42")
try:
    setups = [parse_setup(t.strip()) for t in setup_text.split(",") if t.strip()]
except ValueError as exc:
    setups = []
    st.warning(f"Could not read setups: {exc}")
if setups:
    srec = setup_recommendation(even, setups, rider.cassette)
    st.success(f"Best setup for this course at {np_target:.0f} W NP: **{srec.iloc[0]['setup']}**")
    cols = [c for c in ["setup", "mid4_pct", "ends_pct", "out_of_range_pct", "cross_chain_pct",
                        "cadence_err_mean", "front_shifts_per_hour", "aero_penalty_w", "score"] if c in srec.columns]
    st.dataframe(srec[cols].round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------- optimisation
st.subheader("6. Optimise pacing")
o1, o2, o3 = st.columns(3)
block_m = o1.select_slider("Block length", [500, 1000, 1500, 2000], 1000, format_func=lambda m: f"{m} m")
bound = o2.slider("Power bounds (± % of NP)", 5, 50, 25)
smooth = o3.slider("Smoothness penalty", 0.0, 5.0, 0.0, 0.5)
if st.button("Optimise pacing", type="primary"):
    st.session_state["opt"] = (key, np_target, cached_opt(course, rider, env, key, np_target, block_m, bound, smooth))

res = st.session_state.get("opt")
if res and res[0] == key and res[1] == np_target:
    opt = res[2]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Even", fmt_time(opt.even.total_time_s))
    a2.metric("Optimised", fmt_time(opt.optimised.total_time_s), f"-{opt.time_saved_s:.0f} s")
    a3.metric("Optimised NP / VI", f"{opt.optimised.np_w:.0f} W / {opt.optimised.vi:.3f}")
    a4.metric("Avg power (even → opt)", f"{opt.even.avg_power:.0f} → {opt.optimised.avg_power:.0f} W")
    if not opt.success:
        st.info("The optimiser could not beat even pacing here; showing even pacing.")
    st.plotly_chart(viz_race.fig_speed_power(opt.even, opt.optimised), use_container_width=True)
    st.plotly_chart(viz_race.fig_pacing_blocks(opt), use_container_width=True)
    km = opt.km_table()
    st.dataframe(km, use_container_width=True, hide_index=True)
    st.download_button("Download per-km power targets (CSV)", opt.to_csv(), file_name=f"{course.name}_pacing.csv",
                       mime="text/csv")
else:
    st.caption("Press the button to compute an optimised power plan at the same NP.")

st.caption("Model: forward energy balance per 10 m segment; corner/descent speed caps; weather wind scaled to rider "
           "height. See README for limitations.")
