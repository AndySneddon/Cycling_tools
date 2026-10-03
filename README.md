# Sneddon in Sport: Cycling Tools

Triathlon bike-leg tools that share one rider profile:

| Tool | Page | What it does |
|---|---|---|
| **1. CdA Estimator** | `pages/1_CdA_Estimator.py` | Estimates CdA from a FIT ride using a full power balance, weather-corrected wind and air density, with braking, coasting and cornering removed. Reports a confidence interval, a systematic range and a confidence badge. |
| **2. Gearing** | `pages/2_Gearing.py` | Finds the 1x chainring or 1x/2x setup that keeps your riding in the middle of your cassette, from one or more FIT files. |
| **3. Race Planner** | `pages/3_Race_Planner.py` | Predicts a bike split from a GPX course, target NP and weather; recommends a chainring or setup; shows time vs NP; optimises pacing. |

Tools 1 and 2 can save their results into the rider profile (`rider_profile.json`, kept local), which Tool 3 uses.

## Install and run

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Run the tests (about 30 s; a few use the network for weather and skip if it is unavailable):

```bash
.venv/bin/python -m pytest tests -q
```

## Using it

1. Put your `.fit` files in `Fit_files/` (or upload them in the page) and GPX courses in `courses/`.
2. **CdA Estimator**: pick a ride and the laps to analyse, set mass and Crr, and leave the wind scale on *automatic*.
   Save the CdA **and** Crr to your profile together: the CdA is only valid with the Crr it was measured with.
3. **Gearing**: choose rides, cassette and 1x/2x candidates. Save your preferred setup and cadence model.
4. **Race Planner**: load a GPX, enter your target NP and an expected variability index (VI), choose the weather
   source, and read off the predicted time, chainring and pacing plan.

Weather comes from [Open-Meteo](https://open-meteo.com) (free, no key): the reanalysis archive for past dates and the
forecast for the next ~16 days. Responses are cached in `.cache/` (forecasts expire after 2 h).

## How the models work

### CdA Estimator (`cycling_tools/cda.py`)
- Per-sample power balance: aero + rolling resistance + gravity + acceleration (including wheel inertia), with
  drivetrain efficiency.
- Headwind is resolved from GPS heading and the weather wind; the 10 m reanalysis wind is scaled to rider height by a
  *wind scale* that is **fitted per ride** by minimising the robust fit loss (typically 0.2–0.45, not the 0.7 a
  textbook power law would suggest). The head-minus-tail CdA gap is shown as a diagnostic.
- Samples are masked for coasting, braking, cornering, stops, steep grade, hard acceleration and low airspeed
  (each dilated by a few seconds), then CdA is fitted with a Huber robust regression.
- Uncertainty: block-bootstrap confidence interval (10-minute blocks) plus a **systematic range** from Crr,
  wind scale and drivetrain efficiency. Power-meter bias is not included.
- Crr dominates the systematic error: each +0.001 of Crr lowers CdA by about 0.010.
- Virtual elevation (Chung method) and a per-lap, per-speed and head/tail breakdown are available as checks.

### Gearing (`cycling_tools/gearing.py`)
- Assumes you keep your observed speed and cadence preference when you change setup, and maps each pedalling sample
  to the nearest real sprocket (1x) or ring/sprocket combination (2x, with cross-chain avoidance).
- Scores time in the middle of the cassette, time at the ends, out-of-range time, cadence error, cross-chaining and
  front-shift count. The 2x score includes an **assumed** front-derailleur aero penalty (default 0.002 m² CdA); it is
  an assumption you can change or set to zero.
- Near-ties are reported: when several rings score within a point the smallest is recommended, and a winner at the
  edge of the range is flagged.

### Race Planner (`cycling_tools/course.py`, `simulate.py`, `optimise.py`)
- The course is resampled to 10 m; elevation is despiked and smoothed over a distance window, and grade is clamped.
- Each segment solves the energy balance exactly (kinetic energy, rolling, gravity, aero with airspeed = ground speed
  + headwind, drivetrain efficiency, wheel inertia). The simulator is numba-compiled.
- Speed caps: corners (lateral acceleration from smoothed GPS curvature, with a backward braking pass) and a maximum
  descent speed. The rider coasts above ~70 km/h.
- Weather is sampled along the race clock. Air density comes from temperature, pressure and humidity.
- "Even pacing" solves a constant power that gives the target NP. Real races are not dead-even, so the planner takes an
  **expected VI** and simulates at NP ÷ VI (about 1.02 for a flat, surgy course; about 1.00 for a rolling TT).
- Pacing optimiser: block powers minimise time subject to NP, with ±% bounds and rolling-power caps (1 min ≤ 110 % and
  5 min ≤ 106 % of NP by default). It reports peak 1/5/20-minute power, average power and a 120 s NP so the saving is
  not overstated. Realistic gains are about 0.5–1 % of race time.

### Validation
Back-tested against real races using the rider's own measured CdA: predictions for a 50-mile TT and a 178 km
Ironman-distance bike leg land within about 1 % of the actual time. Each result is one race and used that race's own
CdA, so it shows the model is consistent, not that every future prediction will be that close.

## Project layout

```
app.py                 navigation entry point
home.py                home page and rider profile
pages/                 the three tool pages
cycling_tools/         shared engine
  physics.py geo.py    air density, wind geometry, NP, bearings
  fit_io.py fitfast.py FIT loading (fast, checksum-validated decoder with fitparse fallback)
  weather.py           Open-Meteo client with caching
  cda.py viz_cda.py    Tool 1
  gearing.py viz_gearing.py    Tool 2
  course.py simulate.py optimise.py viz_race.py    Tool 3
  profile.py branding.py       rider profile, logo and styling
assets/                logo files
courses/               GPX courses
tests/                 pytest suite
strava/                optional Strava activity downloader (see below)
Fit_files/             FIT rides and the legacy parser Fit_file_parser.py
```

`Fit_files/Fit_file_parser.py` is the original standalone script; the `cycling_tools` package supersedes it.

## Strava downloader

`strava/strava_api.py` reads credentials from environment variables, never from source:

```bash
export STRAVA_CLIENT_ID=... STRAVA_CLIENT_SECRET=... STRAVA_REFRESH_TOKEN=...
```

## Limitations

- Weather is a coarse grid at 10 m height at one point; local shelter, gusts and direction errors are not captured and
  can dominate the error.
- Yaw-dependent CdA, drafting, traffic, surface changes and fatigue are not modelled. CdA differs by bike and position,
  so measure it on the bike you will race.
- The optimiser minimises time at a given NP, which is a proxy; it does not know how hard surges feel. Use the bounds
  and caps to keep plans rideable. A 4th-power NP target is not appropriate for a very long bike leg.
- GPS elevation and heading noise affect grade and corner caps.
- Predictions are only as good as the inputs (CdA, Crr, mass, power-meter accuracy).
