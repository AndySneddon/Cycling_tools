# Cycling Tools

Three analysis tools for a time-trial / triathlon rider, sharing one rider profile (`rider_profile.json`):

1. **CdA Estimator** (`pages/1_*`) - estimate CdA from a ride, weather-corrected.
2. **Gearing** (`pages/2_Gearing.py`) - which chainring keeps you in the middle of your cassette.
3. **Race Planner** (`pages/3_Race_Planner.py`) - bike-split prediction and pacing optimiser for a GPX course.

## Install and run

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
.venv/bin/python -m pytest tests/test_race.py      # validation, includes a real-ride back-test (needs internet)
```

Weather comes from Open-Meteo (free, no key). Responses are cached in `.cache/`.

## Race Planner

Load a GPX (upload, or drop files in `courses/`), set your rider parameters in the sidebar and a target normalised
power (NP), and choose weather (Open-Meteo forecast/archive for race date and start time, manual wind, or none).
You get predicted time, average speed, VI, splits per 1/5/10 km, an elevation profile shaded by gradient with a
headwind overlay, a route map, a finish-time-vs-NP curve with slider (and seconds per 5 W), a what-if table
(CdA, mass, Crr, watts), a chainring recommendation using the gearing tool, and an *Optimise pacing* button that
produces per-block power targets (CSV download) at the same NP.

Code layout: `cycling_tools/course.py` (route -> 10 m profile, grade, heading, corner caps), `simulate.py`
(numba forward simulation, NP solver, splits, what-if, gearing glue), `optimise.py` (pacing optimiser),
`viz_race.py` (Plotly figures).

### Model
- Course resampled to 10 m; elevation smoothed over a distance window (default 60 m, two passes), grade clamped
  to +/-20 %. Wind and gradient use these segment values.
- Each segment solves the energy balance exactly (kinetic energy included, rolling resistance, gravity, aero with
  air speed = ground speed + headwind, drivetrain efficiency, wheel inertia).
- Speed caps: corners (lateral acceleration 3.5 m/s^2 from smoothed GPS curvature, with a backward braking pass at
  3 m/s^2) and a maximum descent speed (79 km/h). Above 70 km/h the rider coasts. When capped, only the power
  needed to hold the cap is counted as applied.
- Weather: hourly Open-Meteo wind (10 m) x `wind_scale` (default 0.7), resolved against the local heading, sampled
  at the race clock (two passes so wind follows elapsed time). Air density from temperature, pressure, humidity.
- NP is computed from exact 1 s bin means of the simulated power (30 s rolling, 4th-power mean); "even pacing"
  solves a constant crank power giving NP == target.
- Optimiser: block power (500 m - 2 km) minimising time subject to NP == target (SLSQP seeded from even
  pacing, +/- % bounds, optional smoothness penalty), then rescaled to hit NP exactly; falls back to even pacing if
  it cannot do better.

### Limitations
- Weather is a ~10 km reanalysis/forecast grid at 10 m height at one point (the course centre). Local shelter,
  hedges, gusts and wind direction errors are not captured; the back-test shows this can dominate the error.
- Yaw-dependent CdA, drafting, standing starts, traffic, surface changes, and rider fatigue/physiology (anaerobic
  reserve, W') are not modelled. The optimiser minimises time for a given NP, which is a proxy: it does not know
  how hard surges actually feel; use bounds and smoothness to keep plans rideable.
- GPS elevation and heading noise affect grade and corner caps. Smoothing parameters are in the sidebar.
  Speeds are along the horizontal course distance (road length differs by under 0.5 % at typical grades).
- Predictions are only as good as the inputs (CdA, Crr, mass, power-meter accuracy).
