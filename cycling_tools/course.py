"""Course model: GPX / FIT route -> fixed-step profile with grade, heading and speed caps."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from .geo import bearing_deg, haversine_m, smooth_heading_deg

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"


@dataclass
class CourseSettings:
    step_m: float = 10.0            # resampling step
    elev_smooth_m: float = 60.0     # distance window for elevation smoothing (uniform, applied twice)
    max_grade: float = 0.20         # grade clamp (+/-, rise/run)
    heading_smooth_m: float = 40.0  # window for smoothing bearing
    curvature_window_m: float = 30.0  # baseline over which heading change is measured
    max_lat_accel: float = 3.5      # m/s^2 lateral acceleration limit in corners
    max_brake_decel: float = 3.0    # m/s^2 comfortable braking, used for the backward feasibility pass
    min_corner_speed: float = 3.0   # m/s floor on corner caps (tight U-turns / roundabouts)
    max_descent_speed: float = 22.0  # m/s (~80 km/h) cap on descents


@dataclass
class Course:
    """A route resampled to (approximately) uniform spacing. Node arrays have n+1 entries,
    segment arrays (between consecutive nodes) have n."""

    name: str
    dist_m: np.ndarray        # node distance
    elev_m: np.ndarray        # node (smoothed) elevation
    lat: np.ndarray
    lon: np.ndarray
    ds: np.ndarray            # segment length
    grade: np.ndarray         # segment grade (rise/run), clamped
    heading: np.ndarray       # segment bearing (deg, smoothed)
    curvature: np.ndarray     # segment curvature (1/m)
    vcap: np.ndarray          # segment max speed (m/s) from corners, descents and braking feasibility
    settings: CourseSettings = field(default_factory=CourseSettings)
    start_time: pd.Timestamp | None = None  # UTC naive, if known (FIT-derived)

    # ------------------------------------------------------------------ props
    @property
    def n_seg(self) -> int:
        return len(self.ds)

    @property
    def length_m(self) -> float:
        return float(self.dist_m[-1])

    @property
    def seg_mid_m(self) -> np.ndarray:
        return 0.5 * (self.dist_m[:-1] + self.dist_m[1:])

    @property
    def ascent_m(self) -> float:
        d = np.diff(self.elev_m)
        return float(d[d > 0].sum())

    @property
    def descent_m(self) -> float:
        d = np.diff(self.elev_m)
        return float(-d[d < 0].sum())

    @property
    def centre(self) -> tuple[float, float]:
        return float(np.mean(self.lat)), float(np.mean(self.lon))

    def summary(self) -> dict:
        return {
            "name": self.name,
            "distance_km": self.length_m / 1000.0,
            "ascent_m": self.ascent_m,
            "descent_m": self.descent_m,
            "min_elev_m": float(self.elev_m.min()),
            "max_elev_m": float(self.elev_m.max()),
            "n_corner_limited": int((self.vcap < 0.9 * self.settings.max_descent_speed).sum()),
        }


# ---------------------------------------------------------------------- build
def _smooth_distance(values: np.ndarray, step_m: float, window_m: float, passes: int = 2) -> np.ndarray:
    size = int(round(window_m / step_m))
    if size < 2:
        return values.copy()
    out = values
    for _ in range(passes):
        out = uniform_filter1d(out, size=size, mode="nearest")
    return out


def build_course(
    lat,
    lon,
    elev=None,
    *,
    name: str = "course",
    settings: CourseSettings | None = None,
    start_time=None,
    elevation_fetcher=None,
) -> Course:
    """Turn raw lat/lon(/elevation) points into a resampled :class:`Course`.

    ``elevation_fetcher(lat, lon) -> elev`` is called when ``elev`` is None / mostly missing.
    """
    cfg = settings or CourseSettings()
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[ok], lon[ok]
    elev_arr = None if elev is None else np.asarray(elev, dtype=float)[ok]
    if len(lat) < 2:
        raise ValueError("Need at least two GPS points to build a course.")

    step = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    keep = np.concatenate([[True], step > 0.5])  # drop stationary duplicates
    lat, lon = lat[keep], lon[keep]
    if elev_arr is not None:
        elev_arr = elev_arr[keep]
    raw_d = np.concatenate([[0.0], np.cumsum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    if raw_d[-1] < 200:
        raise ValueError("Course is shorter than 200 m.")

    if elev_arr is None or np.isfinite(elev_arr).mean() < 0.5:
        if elevation_fetcher is None:
            raise ValueError("Route has no elevation data (enable elevation lookup).")
        elev_arr = elevation_fetcher(lat, lon)
    elev_arr = np.asarray(elev_arr, dtype=float)
    good = np.isfinite(elev_arr)
    elev_arr = np.interp(raw_d, raw_d[good], elev_arr[good])

    n = max(2, int(round(raw_d[-1] / cfg.step_m)))
    d = np.linspace(0.0, raw_d[-1], n + 1)
    la = np.interp(d, raw_d, lat)
    lo = np.interp(d, raw_d, lon)
    el = np.interp(d, raw_d, elev_arr)
    step_actual = float(d[1] - d[0])

    el = _smooth_distance(el, step_actual, cfg.elev_smooth_m)
    ds = np.diff(d)
    grade = np.clip(np.diff(el) / ds, -cfg.max_grade, cfg.max_grade)

    head = bearing_deg(la[:-1], lo[:-1], la[1:], lo[1:])
    head = smooth_heading_deg(head, int(round(cfg.heading_smooth_m / step_actual)))
    curvature = _curvature(head, ds, step_actual, cfg)

    vcap = corner_speed_caps(curvature, ds, grade, cfg)

    return Course(
        name=name, dist_m=d, elev_m=el, lat=la, lon=lo, ds=ds, grade=grade,
        heading=head, curvature=curvature, vcap=vcap, settings=cfg,
        start_time=None if start_time is None else pd.Timestamp(start_time),
    )


def _curvature(heading_deg: np.ndarray, ds: np.ndarray, step_m: float, cfg: CourseSettings) -> np.ndarray:
    """|d heading / d s| (rad/m) measured over ``curvature_window_m``."""
    h = np.unwrap(np.radians(heading_deg))
    k = max(1, int(round(cfg.curvature_window_m / step_m / 2)))
    n = len(h)
    i0 = np.clip(np.arange(n) - k, 0, n - 1)
    i1 = np.clip(np.arange(n) + k, 0, n - 1)
    dist = np.maximum((i1 - i0) * step_m, step_m)
    return np.abs(h[i1] - h[i0]) / dist


def corner_speed_caps(curvature: np.ndarray, ds: np.ndarray, grade: np.ndarray, cfg: CourseSettings) -> np.ndarray:
    """Per-segment speed cap: lateral-accel limit, descent limit and a backward braking pass so
    the caps are reachable with deceleration <= ``max_brake_decel``."""
    with np.errstate(divide="ignore"):
        v = np.sqrt(cfg.max_lat_accel / np.maximum(curvature, 1e-9))
    v = np.clip(v, cfg.min_corner_speed, cfg.max_descent_speed)
    return _braking_pass(v, ds, cfg.max_brake_decel)


def _braking_pass(v: np.ndarray, ds: np.ndarray, a_brake: float) -> np.ndarray:
    v = v.copy()
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_brake * ds[i]))
    return v


def with_settings(course: Course, **changes) -> Course:
    """Rebuild a course's caps with different corner / descent settings (geometry unchanged)."""
    cfg = CourseSettings(**{**course.settings.__dict__, **changes})
    vcap = corner_speed_caps(course.curvature, course.ds, course.grade, cfg)
    return Course(**{**course.__dict__, "settings": cfg, "vcap": vcap})


# --------------------------------------------------------------------- loaders
def fetch_elevations(lat, lon, *, spacing_m: float = 60.0, timeout: float = 20.0) -> np.ndarray:
    """Open-Meteo elevation lookup (batches of 100). Queried on a thinned set of points, then interpolated."""
    import requests

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    d = np.concatenate([[0.0], np.cumsum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    nq = max(2, int(d[-1] / spacing_m) + 1)
    dq = np.linspace(0, d[-1], nq)
    qlat, qlon = np.interp(dq, d, lat), np.interp(dq, d, lon)
    out = []
    for i in range(0, nq, 100):
        la, lo = qlat[i:i + 100], qlon[i:i + 100]
        resp = requests.get(
            ELEVATION_URL,
            params={"latitude": ",".join(f"{x:.5f}" for x in la), "longitude": ",".join(f"{x:.5f}" for x in lo)},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Elevation API returned {resp.status_code}: {resp.text[:200]}")
        out.extend(resp.json()["elevation"])
    return np.interp(d, dq, np.asarray(out, dtype=float))


def load_gpx(source, *, name: str | None = None, settings: CourseSettings | None = None,
             fetch_missing_elevation: bool = False) -> Course:
    """Load a GPX file (path, bytes or file-like) as a Course. Uses the first track (else first route)."""
    import gpxpy

    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8", errors="replace")
        name = name or Path(source).stem
    else:
        raw = source if isinstance(source, (bytes, bytearray)) else source.read()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
        name = name or "uploaded route"
    gpx = gpxpy.parse(text)
    pts = [p for t in gpx.tracks for s in t.segments for p in s.points]
    if not pts:
        pts = [p for r in gpx.routes for p in r.points]
    if len(pts) < 2:
        raise ValueError("GPX contains no track or route points.")
    if gpx.tracks and gpx.tracks[0].name:
        name = name if name != "uploaded route" else gpx.tracks[0].name
    lat = np.array([p.latitude for p in pts])
    lon = np.array([p.longitude for p in pts])
    elev = np.array([np.nan if p.elevation is None else p.elevation for p in pts], dtype=float)
    return build_course(
        lat, lon, elev, name=name, settings=settings,
        elevation_fetcher=fetch_elevations if fetch_missing_elevation else None,
        start_time=pts[0].time.replace(tzinfo=None) if pts[0].time else None,
    )


def course_from_ride(ride, lap: int | None = None, *, settings: CourseSettings | None = None,
                     name: str | None = None) -> Course:
    """Build a Course from the GPS trace of a FIT ride (optionally restricted to one lap)."""
    df = ride.df
    if lap is not None:
        df = df[df["lap"] == lap]
    df = df.dropna(subset=["lat", "lon"])
    if len(df) < 10:
        raise ValueError("Not enough GPS points in that ride/lap.")
    nm = name or (f"{ride.name} (lap {lap})" if lap is not None else ride.name)
    return build_course(
        df["lat"].to_numpy(), df["lon"].to_numpy(), df["alt"].to_numpy() if "alt" in df else None,
        name=nm, settings=settings, start_time=df["timestamp"].iloc[0],
    )
