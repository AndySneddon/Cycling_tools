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

from cycling_tools.branding import PAGE_ICON, apply_branding, page_header

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cycling_tools import race_profiles as rp
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

st.set_page_config(page_title="Race Planner", page_icon=PAGE_ICON, layout="wide")
apply_branding()
page_header("Race Planner",
            "Predict a bike split from a GPX course, a power target and the weather, choose the right chainring, "
            "then optimise your pacing.",
            eyebrow="Tool 3 · Race strategy")

COURSES_DIR = ROOT / "courses"

# ------------------------------------------------------------ saved race plans
# Loading a race bumps a token that is part of every widget key, so each input is rebuilt with the saved value as its
# default (and then stays freely editable). D() reads a saved value, falling back to the usual default.
T = st.session_state.get("rp_token", 0)
LOADED = st.session_state.get("rp_loaded", {})


def D(key, default):
    return LOADED.get(key, default)


def _load_race(slug: str) -> None:
    try:
        profile, gpx = rp.load_profile(slug)
    except rp.RaceProfileError as exc:
        st.session_state["rp_flash"] = ("error", str(exc))
        return
    st.session_state["rp_loaded"] = profile.settings
    st.session_state["rp_loaded_name"] = profile.name
    st.session_state["rp_notes"] = profile.notes
    st.session_state["rp_course"] = {"name": profile.course_name or profile.slug, "data": gpx} if gpx else None
    st.session_state["rp_token"] = st.session_state.get("rp_token", 0) + 1
    st.session_state.pop("opt", None)
    st.session_state["rp_flash"] = ("success", f"Loaded “{profile.name}”.")


def _delete_race(slug: str) -> None:
    rp.delete_profile(slug)
    if rp.slugify(st.session_state.get("rp_loaded_name", "")) == slug:
        for k in ("rp_loaded", "rp_loaded_name", "rp_notes", "rp_course"):
            st.session_state.pop(k, None)
        st.session_state["rp_token"] = st.session_state.get("rp_token", 0) + 1
    st.session_state["rp_flash"] = ("success", "Race plan deleted.")


flash = st.session_state.pop("rp_flash", None)
if flash:
    (st.success if flash[0] == "success" else st.error)(flash[1])
save_box = st.container()  # filled at the bottom, once every input value is known (the border is added there)


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
def cached_opt(_course, _rider, _env, key, np_target, block_m, bound_pct, smooth, caps):
    return optimise_pacing(_course, _rider, np_target, _env, block_m=block_m, bound_pct=bound_pct, smooth=smooth,
                           power_caps=dict(caps))


# --------------------------------------------------------------------- sidebar
profile = RiderProfile.load()
with st.sidebar:
    st.header("Saved races")
    saved = rp.list_profiles()
    if saved:
        by_slug = {p.slug: p for p in saved}
        pick_slug = st.selectbox(
            "Race plan", list(by_slug), key=f"rp_pick_{T}", format_func=lambda sl: by_slug[sl].name,
            index=list(by_slug).index(rp.slugify(st.session_state.get("rp_loaded_name", "")))
            if rp.slugify(st.session_state.get("rp_loaded_name", "")) in by_slug else 0)
        chosen = by_slug[pick_slug]
        meta = f"{chosen.course_name or 'no course'} · saved {chosen.saved_at[:10]}"
        if chosen.last_prediction.get("time"):
            meta += f" · predicted {chosen.last_prediction['time']}"
        st.caption(meta + (f"\n\n{chosen.notes}" if chosen.notes else ""))
        lc, dc = st.columns(2)
        lc.button("Load", on_click=_load_race, args=(pick_slug,), use_container_width=True)
        confirm = dc.checkbox("Confirm delete", key=f"rp_confirm_{pick_slug}")
        st.button("Delete", on_click=_delete_race, args=(pick_slug,), disabled=not confirm,
                  use_container_width=True)
    else:
        st.caption("No saved races yet. Build a plan, then save it from the box at the top of the page.")

    st.header("Rider")
    rider = replace(
        profile,
        mass_kg=st.number_input("System mass (kg)", 40.0, 200.0, float(D("mass_kg", profile.mass_kg)), 0.5,
                                key=f"mass_{T}"),
        cda=st.number_input("CdA (m²)", 0.10, 0.60, float(D("cda", profile.cda)), 0.001, format="%.3f",
                            key=f"cda_{T}"),
        crr=st.number_input("Crr", 0.001, 0.015, float(D("crr", profile.crr)), 0.0001, format="%.4f",
                            key=f"crr_{T}"),
        drivetrain_eff=st.number_input("Drivetrain efficiency", 0.90, 1.0,
                                       float(D("drivetrain_eff", profile.drivetrain_eff)), 0.005, key=f"eff_{T}"),
        wind_scale=st.number_input("Wind scale (10 m → rider)", 0.2, 1.2,
                                   float(D("wind_scale", profile.wind_scale)), 0.05, key=f"wind_{T}"),
        tyre_circumference_m=st.number_input("Wheel circumference (m)", 1.9, 2.3,
                                             float(D("tyre_circumference_m", profile.tyre_circumference_m)),
                                             0.005, format="%.3f", key=f"circ_{T}"),
        cadence_flat=st.number_input("Cadence at 200 W (rpm)", 50.0, 120.0,
                                     float(D("cadence_flat", profile.cadence_flat)), 1.0, key=f"cad_{T}"),
        cadence_per_100w=st.number_input("Cadence change per +100 W (rpm)", -20.0, 40.0,
                                         float(D("cadence_per_100w", profile.cadence_per_100w)), 0.5,
                                         key=f"cadslope_{T}"),
    )
    rider.cassette = list(D("cassette", profile.cassette))
    rider.chainring = int(D("chainring", profile.chainring))
    preset = st.selectbox("Cassette", ["(profile)"] + list(CASSETTES), key=f"cass_{T}")
    if preset != "(profile)":
        rider.cassette = list(CASSETTES[preset])
    st.caption("Sprockets: " + ", ".join(map(str, rider.cassette)))
    if st.button("Save as rider profile"):
        rider.save()
        st.success("Profile saved.")

    st.header("Course model")
    with st.expander("Smoothing & limits"):
        cs = CourseSettings(
            elev_smooth_m=st.number_input("Elevation smoothing (m)", 0.0, 500.0, float(D("elev_smooth_m", 60.0)), 10.0,
                                          key=f"elev_{T}"),
            max_grade=st.number_input("Max grade clamp (%)", 5.0, 40.0, float(D("max_grade_pct", 20.0)), 1.0,
                                      key=f"grade_{T}") / 100.0,
            max_lat_accel=st.number_input("Max lateral accel (m/s²)", 1.0, 8.0, float(D("max_lat_accel", 3.5)), 0.1,
                                          key=f"lat_{T}"),
            max_descent_speed=st.number_input("Max descent speed (km/h)", 40.0, 110.0,
                                              float(D("max_descent_kmh", 79.0)), 1.0, key=f"desc_{T}") / 3.6,
        )
    fill_elev = st.checkbox("Look up elevation online if GPX has none", value=bool(D("fill_elev", True)),
                            key=f"fill_{T}")

# ---------------------------------------------------------------------- course
st.subheader("1. Course")
gpx_files = sorted(COURSES_DIR.glob("*.gpx")) if COURSES_DIR.exists() else []
c1, c2 = st.columns(2)
upload = c1.file_uploader("Upload a GPX file", type=["gpx"], key=f"upload_{T}")
pick = c2.selectbox("...or pick from courses/", ["(none)"] + [p.name for p in gpx_files], key=f"pick_{T}")
saved_course = st.session_state.get("rp_course")

data, cname = None, None
if upload is not None:
    data, cname = upload.getvalue(), Path(upload.name).stem
elif pick != "(none)":
    data, cname = (COURSES_DIR / pick).read_bytes(), Path(pick).stem
elif saved_course:
    data, cname = saved_course["data"], saved_course["name"]
    st.caption(f"Using the course saved with “{st.session_state.get('rp_loaded_name', 'this race')}”. "
               "Upload or pick another file to replace it.")
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
ckey = (course.name, course.n_seg, round(course.length_m), tuple(sorted(cs.__dict__.items())))
rkey_even = (rider.mass_kg, rider.cda, rider.crr, rider.drivetrain_eff)
st.subheader("2. Power and weather")
i1, i2 = st.columns([1, 2])
np_race = i1.number_input("Target normalised power (W)", 50.0, 600.0, float(D("np_race", 250.0)), 1.0,
                          key=f"np_{T}")
race_vi = i1.number_input(
    "Expected race variability index (VI)", 1.00, 1.15, float(D("race_vi", 1.02)), 0.01, key=f"vi_{T}",
    help="Real races are not dead-even: you coast, brake and surge, so average power is NP / VI. VI depends on the "
         "course: roughly 1.02 for a flat, surgy course (Almere, 178 km, was 1.024), about 1.00 for a rolling TT "
         "ridden to power, higher on hilly or technical routes. Set 1.00 for a perfectly constant-power ride. "
         "The simulator rides at constant power, so it is given NP / VI (shown below as the simulated power).")
# The simulator rides at constant power, so hold the average power a variable ride with this NP would have
np_target = np_race / race_vi
i1.caption(f"Your NP {np_race:.0f} W / VI {race_vi:.2f} = **{np_target:.0f} W** simulated constant power.")
WX_MODES = ["Open-Meteo (forecast / archive)", "Manual wind", "None (still air)"]
mode = i2.radio("Weather", WX_MODES, horizontal=True, key=f"wxmode_{T}",
                index=WX_MODES.index(D("weather_mode", WX_MODES[0])) if D("weather_mode", WX_MODES[0]) in WX_MODES else 0)
# values of the inactive weather modes are still saved, so start from the saved/default ones
race_date = date.fromisoformat(D("race_date", (date.today() + timedelta(days=2)).isoformat()))
start_t = time.fromisoformat(D("start_time", "08:00"))
tzname = D("timezone", "Europe/London")
ws, wd, temp, pres = (float(D("wind_kmh", 10.0)), float(D("wind_from_deg", 180.0)), float(D("temp_c", 15.0)),
                      float(D("pressure_hpa", 1013.0)))

env = Environment()
env_key = ("none",)
if mode.startswith("Open-Meteo"):
    w1, w2, w3 = st.columns(3)
    race_date = w1.date_input("Race date", race_date, key=f"date_{T}")
    start_t = w2.time_input("Start time (local)", start_t, key=f"time_{T}")
    tzname = w3.text_input("Timezone", tzname, key=f"tz_{T}")
    try:
        tz = ZoneInfo(tzname)
        start_utc = datetime.combine(race_date, start_t, tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        lat, lon = course.centre
        # weather window = start + predicted duration (still-air estimate) + margin, so long races do not run on
        # a frozen last-hour wind
        est_s = cached_even(course, rider, Environment(), ("still", ckey, rkey_even), np_target).total_time_s
        end_utc = (pd.Timestamp(start_utc) + pd.Timedelta(seconds=1.3 * est_s + 3600.0)).ceil("h").to_pydatetime()
        wx = cached_weather(round(lat, 3), round(lon, 3), start_utc.isoformat(), end_utc.isoformat())
        env = Environment(wx, pd.Timestamp(start_utc), rider.wind_scale)
        env_key = ("om", round(lat, 3), round(lon, 3), start_utc.isoformat(), rider.wind_scale, wx.source,
                   str(wx.hourly.index[-1]))
        h = wx.at([start_utc])
        st.caption(f"{wx.source}: at start {h['temp_c'].iloc[0]:.1f} °C, wind {h['wind_ms'].iloc[0]:.1f} m/s "
                   f"from {h['wind_dir'].iloc[0]:.0f}° (×{rider.wind_scale:.2f} at rider height)")
    except WeatherError as exc:
        st.warning(f"Weather unavailable ({exc}) Using still air for this prediction; switch to **Manual wind** to "
                   "enter conditions yourself.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Weather unavailable ({exc}); using still air.")
elif mode.startswith("Manual"):
    w1, w2, w3, w4 = st.columns(4)
    ws = w1.number_input("Wind speed at rider (km/h)", 0.0, 80.0, ws, 1.0, key=f"ws_{T}")
    wd = w2.number_input("Wind from (deg, 0 = N, 90 = E)", 0.0, 360.0, wd, 5.0, key=f"wd_{T}")
    temp = w3.number_input("Temperature (°C)", -10.0, 45.0, temp, 1.0, key=f"temp_{T}")
    pres = w4.number_input("Pressure (hPa)", 900.0, 1050.0, pres, 1.0, key=f"pres_{T}")
    env = Environment(constant_weather(temp, ws / 3.6, wd, pres), None, 1.0)
    env_key = ("manual", ws, wd, temp, pres)

rkey = rkey_even
key = (ckey, rkey, env_key)

# --------------------------------------------------------------------- results
even = cached_even(course, rider, env, key, np_target)
if env.time_dependent and env.weather.hourly.index[-1] < env.start_time + pd.Timedelta(seconds=even.total_time_s):
    st.warning("The weather data ends before the predicted finish "
               f"({env.weather.hourly.index[-1]:%d %b %H:%M} UTC): the last available wind is held for the rest of "
               "the race (the forecast horizon is ~16 days).")
st.subheader("3. Predicted split (even pacing)")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Time", fmt_time(even.total_time_s))
r2.metric("Average speed", f"{even.avg_speed_kmh:.1f} km/h")
r3.metric("Simulated average power (NP/VI)", f"{np_target:.0f} W",
          help="The simulator pedals at this constant power. Ride average including coasting: "
               f"{even.avg_power:.0f} W.")
r4.metric("Your NP", f"{np_race:.0f} W")
r5.metric("Your VI", f"{race_vi:.2f}", help="Course dependent: ~1.02 flat/surgy, ~1.00 for a rolling TT.")

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
curve = curve.assign(np_w=curve["np_w"] * race_vi, s_saved_per_5w=curve["s_saved_per_5w"] / race_vi)  # to your NP
sel = st.slider("Try a different NP (W)", float(curve["np_w"].min()), float(curve["np_w"].max()), float(np_race), 1.0)
alt = cached_even(course, rider, env, key, sel / race_vi)
s1, s2, s3 = st.columns(3)
s1.metric("Time at selected NP", fmt_time(alt.total_time_s), f"{alt.total_time_s - even.total_time_s:+.0f} s vs target NP")
s2.metric("Average speed", f"{alt.avg_speed_kmh:.1f} km/h")
near = curve.iloc[(curve["np_w"] - np_race).abs().argmin()]
s3.metric("Seconds per +5 W of NP", f"{near['s_saved_per_5w']:.0f} s")
st.plotly_chart(viz_race.fig_np_curve(curve, sel), use_container_width=True)

st.markdown("**What-if (even pacing, same NP unless stated; power changes are on the simulated NP/VI scale)**")
wi = cached_whatif(course, rider, env, key, np_target)
st.dataframe(wi[["scenario", "time", "saved_s"]].rename(columns={"saved_s": "time saved (s)"}).round(1),
             use_container_width=True, hide_index=True)

# ------------------------------------------------------------------- gearing
st.subheader("5. Chainring recommendation")
rings = list(range(46, 65))
rec = chainring_recommendation(even, rings, rider.cassette)
best = int(rec.iloc[0]["chainring"])
st.success(f"Best chainring for this course at {np_race:.0f} W NP: **{best}T** "
           f"(cassette {rider.cassette[0]}-{rider.cassette[-1]}).")
st.dataframe(rec.round(1), use_container_width=True, hide_index=True)
top = sorted(rec["chainring"].head(5).astype(int))
st.plotly_chart(viz_race.fig_gear_heatmap(even, top, rider.cassette), use_container_width=True)

st.markdown("**1x vs 2x for this course**")
st.caption("Compare the setups you actually race. The 2x score includes an assumed front-derailleur aero penalty "
           "(see `DEFAULT_SETUP_WEIGHTS` in gearing.py); set `aero_delta_cda` to 0 to ignore it.")
from cycling_tools.gearing import parse_setup
setup_text = st.text_input("Setups to compare (comma separated)", D("setups", "58, 60, 56/42"), key=f"setups_{T}")
try:
    setups = [parse_setup(t.strip()) for t in setup_text.split(",") if t.strip()]
except ValueError as exc:
    setups = []
    st.warning(f"Could not read setups: {exc}")
if setups:
    srec = setup_recommendation(even, setups, rider.cassette)
    st.success(f"Best setup for this course at {np_race:.0f} W NP: **{srec.iloc[0]['setup']}**")
    cols = [c for c in ["setup", "mid4_pct", "ends_pct", "out_of_range_pct", "cross_chain_pct",
                        "cadence_err_mean", "front_shifts_per_hour", "aero_penalty_w", "score"] if c in srec.columns]
    st.dataframe(srec[cols].round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------- optimisation
st.subheader("6. Optimise pacing")
st.info("Be realistic: a well-chosen plan typically saves about 0.5-1% of race time over even pacing, and only "
        "if you can actually hold the power targets. Gains much larger than that usually come from surges that no "
        "rider can repeat, so judge the result by the peak powers and the equal-effort figure below.")
o1, o2, o3 = st.columns(3)
block_m = o1.select_slider("Block length", [500, 1000, 1500, 2000], int(D("block_m", 1000)), key=f"block_{T}",
                           format_func=lambda m: f"{m} m" + (" (aggressive)" if m < 1000 else ""),
                           help="Shorter blocks chase every roll in the road and ask for more surging; 500 m is "
                                "labelled aggressive. 1 km is the recommended default.")
bound = o2.slider("Power bounds (± % of NP)", 5, 50, int(D("bound_pct", 15)), key=f"bound_{T}")
smooth = o3.slider("Smoothness penalty", 0.0, 5.0, float(D("smooth", 0.0)), 0.5, key=f"smooth_{T}",
                   help="Penalises changes between adjacent blocks; normalised per km so it means the same at every "
                        "block length.")
use_caps = st.checkbox("Cap rolling power (keeps the plan rideable)", value=bool(D("use_caps", True)),
                       key=f"caps_{T}",
                       help="The NP constraint alone only limits variability over ~30 s. These caps (% of the "
                            "simulated NP) stop the optimiser asking for long hard surges.")
cc1, cc2 = st.columns(2)
cap1 = cc1.number_input("1-minute power cap (% of NP)", 100.0, 150.0, float(D("cap1", 110.0)), 1.0,
                        disabled=not use_caps, key=f"cap1_{T}")
cap5 = cc2.number_input("5-minute power cap (% of NP)", 100.0, 140.0, float(D("cap5", 106.0)), 1.0,
                        disabled=not use_caps, key=f"cap5_{T}")
caps = ((60, float(cap1)), (300, float(cap5))) if use_caps else ()
if st.button("Optimise pacing", type="primary"):
    st.session_state["opt"] = (key, np_target,
                               cached_opt(course, rider, env, key, np_target, block_m, bound, smooth, caps))

res = st.session_state.get("opt")
if res and res[0] == key and res[1] == np_target:
    opt = res[2]
    race_s = opt.even.total_time_s
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Even", fmt_time(race_s))
    a2.metric("Optimised", fmt_time(opt.optimised.total_time_s),
              f"-{opt.time_saved_s:.0f} s ({100 * opt.time_saved_s / race_s:.2f}% of race time)")
    a3.metric("Saved at equal effort (same 120 s NP)", f"{opt.saved_equal_np120_s:.0f} s",
              help="The optimised plan scaled up or down until its 120 s normalised power matches even pacing's, "
                   "so surges are not rewarded with a lower overall effort.")
    a4.metric("Optimised NP / VI", f"{opt.optimised.np_w:.0f} W / {opt.optimised.vi:.3f}")
    if not opt.success:
        st.info("The optimiser could not beat even pacing here; showing even pacing.")
    elif opt.converged:
        st.caption("Solver converged.")
    for msg in opt.cautions:
        st.warning(msg)
    stt = opt.stats_table().rename(columns={"metric": "Power metric", "even_w": "Even (W)",
                                            "optimised_w": "Optimised (W)", "diff_w": "Difference (W)"})
    st.dataframe(stt.round(1), use_container_width=True, hide_index=True)
    st.plotly_chart(viz_race.fig_speed_power(opt.even, opt.optimised), use_container_width=True)
    st.plotly_chart(viz_race.fig_rolling_power(opt), use_container_width=True)
    st.plotly_chart(viz_race.fig_pacing_blocks(opt), use_container_width=True)
    km = opt.km_table()
    st.dataframe(km, use_container_width=True, hide_index=True)
    st.download_button("Download per-km power targets (CSV)", opt.to_csv(), file_name=f"{course.name}_pacing.csv",
                       mime="text/csv")
else:
    st.caption("Press the button to compute an optimised power plan at the same NP.")

st.caption("Model: forward energy balance per 10 m segment; corner/descent speed caps; weather wind scaled to rider "
           "height. See README for limitations.")


# ------------------------------------------------------------------ save race
with save_box, st.container(border=True):
    st.markdown("#### Save this race plan")
    st.caption("Stores the course, rider values, power target, weather and optimiser settings under a name you choose, "
               "so you can come back to it later.")
    n1, n2 = st.columns([2, 3])
    race_name = n1.text_input("Race name", value=st.session_state.get("rp_loaded_name", ""), key=f"rp_name_{T}",
                              placeholder="e.g. Challenge Almere 2027", max_chars=rp.MAX_NAME_LEN)
    race_notes = n2.text_input("Notes (optional)", value=st.session_state.get("rp_notes", ""), key=f"rp_notes_{T}",
                               placeholder="e.g. flat, expect a headwind on the way back")
    slug = rp.slugify(race_name)
    will_update = bool(slug) and rp.exists(slug)
    if st.button(f"Update “{race_name.strip()}”" if will_update else "Save race plan", type="primary",
                 disabled=not slug, key=f"rp_save_{T}"):
        settings = {
            "mass_kg": rider.mass_kg, "cda": rider.cda, "crr": rider.crr, "drivetrain_eff": rider.drivetrain_eff,
            "wind_scale": rider.wind_scale, "tyre_circumference_m": rider.tyre_circumference_m,
            "cadence_flat": rider.cadence_flat, "cadence_per_100w": rider.cadence_per_100w,
            "cassette": list(rider.cassette), "chainring": int(rider.chainring),
            "elev_smooth_m": cs.elev_smooth_m, "max_grade_pct": cs.max_grade * 100.0,
            "max_lat_accel": cs.max_lat_accel, "max_descent_kmh": cs.max_descent_speed * 3.6, "fill_elev": fill_elev,
            "np_race": np_race, "race_vi": race_vi, "weather_mode": mode,
            "race_date": race_date.isoformat(), "start_time": start_t.strftime("%H:%M"), "timezone": tzname,
            "wind_kmh": ws, "wind_from_deg": wd, "temp_c": temp, "pressure_hpa": pres,
            "setups": setup_text, "block_m": int(block_m), "bound_pct": int(bound), "smooth": float(smooth),
            "use_caps": bool(use_caps), "cap1": float(cap1), "cap5": float(cap5),
        }
        try:
            saved_profile = rp.save_profile(
                race_name, settings, gpx_bytes=data, course_name=course.name, notes=race_notes,
                last_prediction={"time": fmt_time(even.total_time_s), "time_s": round(even.total_time_s),
                                 "avg_speed_kmh": round(even.avg_speed_kmh, 2), "np": np_race, "vi": race_vi})
        except rp.RaceProfileError as exc:
            st.error(str(exc))
        else:
            st.session_state["rp_loaded_name"] = saved_profile.name
            st.session_state["rp_notes"] = saved_profile.notes
            st.session_state["rp_flash"] = ("success", f"{'Updated' if will_update else 'Saved'} “{saved_profile.name}”.")
            st.rerun()
