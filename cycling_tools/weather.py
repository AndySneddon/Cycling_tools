"""Weather lookup via Open-Meteo (free, no API key): archive for past dates, forecast for upcoming."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "weather"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 5  # archive data trails real time slightly; use forecast API for newer dates
FORECAST_HORIZON_DAYS = 15  # the forecast API reaches ~16 days ahead (today + 15)
FORECAST_TTL_S = 2 * 3600   # forecasts are revised every few hours; the archive never changes
MAX_GAP_HOURS = 3           # interpolate at most this many consecutive missing hours; larger gaps are an error


class WeatherError(RuntimeError):
    pass


@dataclass
class Weather:
    """Hourly weather series (UTC) with vector-aware interpolation."""

    hourly: pd.DataFrame  # index: UTC DatetimeIndex; cols: temp_c, rh, pressure_pa, wind_ms, wind_dir, gust_ms
    source: str

    def _prep(self):
        """Per-object cache of the arrays used by every lookup (hourly x-axis and wind u/v components)."""
        c = self.__dict__.get("_cache")
        if c is None:
            h = self.hourly
            x = h.index.values.astype("datetime64[s]").astype(float)
            theta = np.radians(h["wind_dir"].to_numpy(dtype=float))
            spd = h["wind_ms"].to_numpy(dtype=float)
            c = {
                "x": x,
                **{col: h[col].to_numpy(dtype=float) for col in ("temp_c", "rh", "pressure_pa", "gust_ms")},
                "east": spd * np.sin(theta), "north": spd * np.cos(theta),
            }
            self.__dict__["_cache"] = c
        return c

    def at_seconds(self, xi: np.ndarray) -> dict:
        """Interpolate to UTC times given as float epoch-seconds (whole seconds). Returns a dict of arrays."""
        c = self._prep()
        x = c["x"]
        out = {col: np.interp(xi, x, c[col]) for col in ("temp_c", "rh", "pressure_pa", "gust_ms")}
        east = np.interp(xi, x, c["east"])
        north = np.interp(xi, x, c["north"])
        out["wind_ms"] = np.hypot(east, north)
        out["wind_dir"] = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
        return out

    def at(self, times) -> pd.DataFrame:
        """Interpolate to arbitrary UTC timestamps. Wind is interpolated as u/v vectors."""
        times = pd.DatetimeIndex(pd.to_datetime(times))
        xi = times.values.astype("datetime64[s]").astype(float)
        return pd.DataFrame(self.at_seconds(xi), index=times)


def constant_weather(
    temp_c: float = 15.0,
    wind_ms: float = 0.0,
    wind_from_deg: float = 0.0,
    pressure_hpa: float = 1013.25,
    rh: float = 0.6,
) -> Weather:
    """A flat 'weather' object for manual overrides."""
    idx = pd.date_range("2000-01-01", "2100-01-01", periods=2, tz=None)
    df = pd.DataFrame(
        {
            "temp_c": temp_c,
            "rh": rh,
            "pressure_pa": pressure_hpa * 100.0,
            "wind_ms": wind_ms,
            "wind_dir": wind_from_deg,
            "gust_ms": wind_ms,
        },
        index=idx,
    )
    return Weather(df, "manual")


def _request(url: str, params: dict, ttl_s: float | None = None) -> dict:
    """GET with a disk cache. ``ttl_s=None`` caches forever (archive); a number expires the entry (forecast).
    If the network fails but an expired entry exists it is used rather than failing the whole page."""
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    stale = None
    if cache_file.exists():
        try:
            fresh = ttl_s is None or (time.time() - cache_file.stat().st_mtime) < ttl_s
            data = json.loads(cache_file.read_text())
            if fresh:
                return data
            stale = data
        except (OSError, ValueError):
            pass
    try:
        resp = requests.get(url, params=params, timeout=20)
    except requests.RequestException as exc:
        if stale is not None:
            return stale
        raise WeatherError(f"Could not reach Open-Meteo: {exc}") from exc
    if resp.status_code != 200:
        if stale is not None:
            return stale
        raise WeatherError(f"Open-Meteo returned {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
    except OSError:
        pass
    return data


def _utc_naive(ts) -> pd.Timestamp:
    """Timestamp -> naive UTC (tz-aware values are converted, naive ones are taken to be UTC already)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate short gaps only (<= MAX_GAP_HOURS), wind as u/v vectors, never extrapolate. Rows at the very
    start / end that are still incomplete are trimmed; any remaining interior gap is an error."""
    df = df.apply(pd.to_numeric, errors="coerce")
    theta = np.radians(df["wind_dir"])
    u = (df["wind_ms"] * np.sin(theta)).interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
    v = (df["wind_ms"] * np.cos(theta)).interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
    for col in ("temp_c", "rh", "pressure_pa", "gust_ms"):
        df[col] = df[col].interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
    df["wind_ms"] = np.hypot(u, v)
    df["wind_dir"] = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
    df.loc[u.isna() | v.isna(), ["wind_ms", "wind_dir"]] = np.nan
    valid = ~df.isna().any(axis=1)
    if valid.sum() < 2:
        raise WeatherError("Open-Meteo returned incomplete data for that period.")
    first, last = valid.idxmax(), valid[::-1].idxmax()
    df = df.loc[first:last]
    if df.isna().any().any():
        raise WeatherError(f"Open-Meteo data has gaps longer than {MAX_GAP_HOURS} h in that period.")
    return df


def fetch_weather(lat: float, lon: float, start, end) -> Weather:
    """
    Fetch hourly weather covering [start, end] (datetimes or dates; tz-aware values are converted to UTC,
    naive values are taken as UTC).

    Uses the reanalysis archive (cached forever) for dates older than a few days and the forecast API (cached
    for 2 h) for recent / upcoming dates. Races starting more than ~16 days ahead raise ``WeatherError``; a
    race that merely ends after the horizon gets the data that exists (check ``hourly.index[-1]``).
    """
    start_ts = _utc_naive(start)
    end_ts = _utc_naive(end)
    start_d = start_ts.date()
    end_d = end_ts.date()
    today = datetime.now(timezone.utc).date()
    horizon = today + timedelta(days=FORECAST_HORIZON_DAYS)
    if start_d > horizon:
        raise WeatherError(
            f"Forecasts only extend ~16 days ahead (to {horizon.isoformat()}). For later races, enter wind and "
            "temperature manually."
        )
    end_d = min(end_d, horizon)

    params = {
        "latitude": round(float(lat), 3),
        "longitude": round(float(lon), 3),
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    if end_d >= today - timedelta(days=ARCHIVE_LAG_DAYS):
        data = _request(FORECAST_URL, params, ttl_s=FORECAST_TTL_S)
        source = "Open-Meteo forecast"
    else:
        data = _request(ARCHIVE_URL, params)
        source = "Open-Meteo archive (reanalysis)"

    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        raise WeatherError("Open-Meteo returned no hourly data.")
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df = df.rename(
        columns={
            "temperature_2m": "temp_c",
            "relative_humidity_2m": "rh",
            "surface_pressure": "pressure_pa",
            "wind_speed_10m": "wind_ms",
            "wind_direction_10m": "wind_dir",
            "wind_gusts_10m": "gust_ms",
        }
    )
    df["rh"] = df["rh"] / 100.0
    df["pressure_pa"] = df["pressure_pa"] * 100.0
    return Weather(_fill_gaps(df), source)
