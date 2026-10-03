"""Home page: rider profile shared by the CdA, gearing and race-planner tools."""

from __future__ import annotations

import streamlit as st

from cycling_tools.branding import PAGE_ICON, apply_branding, page_header

from cycling_tools.profile import PROFILE_PATH, RiderProfile

try:
    from cycling_tools.gearing import CASSETTES
except ImportError:  # gearing module not available yet
    CASSETTES = {"11-30 (12sp)": [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30]}

st.set_page_config(page_title="Sneddon in Sport", page_icon=PAGE_ICON, layout="wide")
apply_branding()
page_header(
    "Know your bike. Plan your race.",
    "Learn what your riding says about you, then use it to predict splits, pick the right gearing and pace the bike "
    "leg of your next triathlon. Work left to right; the first two tools feed the third through your rider profile.",
    eyebrow="Sneddon in Sport · Triathlon tools",
)

cards = [
    ("1 · CdA Estimator", "Upload a ride and get your CdA, corrected for wind, air density, braking, coasting and corners.",
     "pages/1_CdA_Estimator.py", "Open CdA Estimator"),
    ("2 · Gearing", "Compare 1x and 2x setups and find the chainring that keeps you in the middle of the cassette.",
     "pages/2_Gearing.py", "Open Gearing"),
    ("3 · Race Planner", "Load a GPX, set a power target and get a predicted time, a chainring and an optimised pacing plan.",
     "pages/3_Race_Planner.py", "Open Race Planner"),
]
for col, (title, text, page, label) in zip(st.columns(3), cards):
    with col.container(border=True):
        st.markdown(f"#### {title}")
        st.write(text)
        st.page_link(page, label=label, icon=":material/arrow_forward:")

profile = RiderProfile.load()
st.subheader("Your rider profile")
st.caption("Shared by all three tools. CdA and Crr only make sense as a pair: use the Crr the CdA was measured with.")
c1, c2, c3 = st.columns(3)
with c1:
    profile.mass_kg = st.number_input("System mass (kg, rider+bike+kit)", 40.0, 200.0, float(profile.mass_kg), 0.5)
    profile.cda = st.number_input("CdA (m²)", 0.10, 0.60, float(profile.cda), 0.001, format="%.3f")
    profile.crr = st.number_input("Crr", 0.001, 0.015, float(profile.crr), 0.0001, format="%.4f")
with c2:
    profile.drivetrain_eff = st.number_input("Drivetrain efficiency", 0.90, 1.0, float(profile.drivetrain_eff), 0.005)
    profile.wind_scale = st.number_input("Wind scale (10 m wind → rider height)", 0.2, 1.2, float(profile.wind_scale), 0.05)
    profile.tyre_circumference_m = st.number_input("Wheel circumference (m)", 1.9, 2.3, float(profile.tyre_circumference_m), 0.005, format="%.3f")
with c3:
    preset = st.selectbox("Cassette preset", ["(keep current)"] + list(CASSETTES))
    if preset != "(keep current)":
        profile.cassette = list(CASSETTES[preset])
    txt = st.text_input("Cassette sprockets", ", ".join(map(str, profile.cassette)))
    try:
        profile.cassette = sorted(int(x) for x in txt.replace(",", " ").split())
    except ValueError:
        st.warning("Cassette must be whole numbers separated by commas.")
    profile.chainring = int(st.number_input("Chainring (T)", 30, 70, int(profile.chainring), 1))
    profile.cadence_flat = st.number_input("Preferred cadence (rpm)", 50.0, 120.0, float(profile.cadence_flat), 1.0)

if st.button("Save profile", type="primary"):
    profile.save()
    st.success(f"Saved to {PROFILE_PATH.name}")
