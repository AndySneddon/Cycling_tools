"""Home page: rider profile shared by the CdA, gearing and race-planner tools."""

from __future__ import annotations

import streamlit as st

from cycling_tools.branding import apply_branding

from cycling_tools.profile import PROFILE_PATH, RiderProfile

try:
    from cycling_tools.gearing import CASSETTES
except ImportError:  # gearing module not available yet
    CASSETTES = {"11-30 (12sp)": [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30]}

st.set_page_config(page_title="Cycling Tools", page_icon="🚴", layout="wide")
apply_branding(show_header_logo=True)
st.title("🚴 Cycling Tools")
st.write(
    "Learn about yourself on the bike, then plan races with it. Work left to right:\n\n"
    "1. **CdA Estimator** – upload a ride, get your CdA (weather-corrected, braking/coasting/corners removed).\n"
    "2. **Gearing** – find the chainring that keeps your riding in the middle of the cassette.\n"
    "3. **Race Planner** – load a GPX, set a normalised power, get a predicted time, chainring and an optimised pacing plan.\n\n"
    "Tools 1 and 2 can save their results into the profile below, which Tool 3 uses."
)

profile = RiderProfile.load()
st.subheader("Rider profile")
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
