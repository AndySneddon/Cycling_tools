"""Weather lookup via Open-Meteo (free, no API key): archive for past dates, forecast for upcoming."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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


class WeatherError(RuntimeError):
    pass


@dataclass
class Weather:
    """Hourly weather series (UTC) with vector-aware interpolation."""

    hourly: pd.DataFrame  # index: UTC DatetimeIndex; cols: temp_c, rh, pressure_pa, wind_ms, wind_dir, gust_ms
    source: str

    def at(self, times) -> pd.DataFrame:
        """Interpolate to arbitrary UTC timestamps. Wind is interpolated as u/v vectors."""
        times = pd.DatetimeIndex(pd.to_datetime(times))
        x = self.hourly.index.values.astype("datetime64[s]").astype(float)
        xi = times.values.astype("datetime64[s]").astype(float)
        out = {}
        for col in ("temp_c", "rh", "pressure_pa", "gust_ms"):
            out[col] = np.interp(xi, x, self.hourly[col].to_numpy(dtype=float))
        theta = np.radians(self.hourly["wind_dir"].to_numpy(dtype=float))
        spd = self.hourly["wind_ms"].to_numpy(dtype=float)
        # vector pointing to where the wind comes FROM
        east = np.interp(xi, x, spd * np.sin(theta))
        north = np.interp(xi, x, spd * np.cos(theta))
        out["wind_ms"] = np.hypot(east, north)
        out["wind_dir"] = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
        return pd.DataFrame(out, index=times)


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


def _request(url: str, params: dict) -> dict:
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    try:
        resp = requests.get(url, params=params, timeout=20)
    except requests.RequestException as exc:
        raise WeatherError(f"Could not reach Open-Meteo: {exc}") from exc
    if resp.status_code != 200:
        raise WeatherError(f"Open-Meteo returned {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
    except OSError:
        pass
    return data


def fetch_weather(lat: float, lon: float, start, end) -> Weather:
    """
    Fetch hourly weather covering [start, end] (UTC datetimes or dates).

    Uses the reanalysis archive for dates older than a few days and the forecast API for
    recent / upcoming dates (up to ~16 days ahead).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_d = start_ts.date()
    end_d = end_ts.date()
    today = datetime.now(timezone.utc).date()

    params = {
        "latitude": round(float(lat), 3),
        "longitude": round(float(lon), 3),
        "start_date": start_d.isoformat(),
        "end_date": (end_d + timedelta(days=0)).isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    if end_d > today + timedelta(days=15):
        raise WeatherError(
            "Forecasts only extend ~16 days ahead. For later races, enter wind/temperature manually."
        )
    if end_d >= today - timedelta(days=ARCHIVE_LAG_DAYS):
        data = _request(FORECAST_URL, params)
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
    df = df.interpolate(limit_direction="both")
    if df.isna().any().any():
        raise WeatherError("Open-Meteo returned incomplete data for that period.")
    return Weather(df, source)
