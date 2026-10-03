"""Load FIT files into a clean, regular 1 Hz DataFrame."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fitparse import FitFile

from . import fitfast
from .geo import path_distance_m, semicircles_to_degrees

MAX_INTERP_GAP_S = 3  # fill dropouts up to this long; longer gaps stay NaN


@dataclass
class Ride:
    name: str
    df: pd.DataFrame  # 1 Hz: t_s, timestamp, lat, lon, speed, power, cadence, alt, dist, temp, hr, lap, gap
    laps: pd.DataFrame

    @property
    def has_gps(self) -> bool:
        return self.df["lat"].notna().any()

    @property
    def start_time(self) -> pd.Timestamp:
        return self.df["timestamp"].iloc[0]

    @property
    def end_time(self) -> pd.Timestamp:
        return self.df["timestamp"].iloc[-1]


def _messages(fit: FitFile, kind: str) -> pd.DataFrame:
    rows = [{f.name: f.value for f in m} for m in fit.get_messages(kind)]
    return pd.DataFrame.from_records(rows)


def _coalesce(df: pd.DataFrame, *names: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for n in names:
        if n in df.columns:
            out = out.fillna(pd.to_numeric(df[n], errors="coerce"))
    return out


def load_fit(source: str | Path | bytes | io.BytesIO, name: str | None = None) -> Ride:
    """Parse a FIT file (path, bytes or file-like) into a :class:`Ride`."""
    if isinstance(source, (str, Path)):
        name = name or Path(source).name
        data = Path(source).read_bytes()
    else:
        data = source if isinstance(source, (bytes, bytearray)) else source.read()
        name = name or "upload.fit"

    try:  # fast numpy decoder for the two message types we need (~100x faster than fitparse)
        raw, laps_raw = fitfast.read_messages(bytes(data))
        if raw.empty or "timestamp" not in raw.columns:
            raise fitfast.FitDecodeError("no records")
    except Exception:  # unusual layout (chained file, compressed timestamps...): use the reference parser
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = FitFile(io.BytesIO(data))
            raw = _messages(fit, "record")
            laps_raw = _messages(fit, "lap")

    if raw.empty or "timestamp" not in raw.columns:
        raise ValueError(f"{name}: no record messages found in FIT file.")

    raw = raw.dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")
    ts = pd.to_datetime(raw["timestamp"])

    tidy = pd.DataFrame(
        {
            "speed": _coalesce(raw, "enhanced_speed", "speed"),
            "power": _coalesce(raw, "power"),
            "cadence": _coalesce(raw, "cadence"),
            "alt": _coalesce(raw, "enhanced_altitude", "altitude"),
            "dist": _coalesce(raw, "distance"),
            "temp": _coalesce(raw, "temperature"),
            "hr": _coalesce(raw, "heart_rate"),
            "lat": semicircles_to_degrees(_coalesce(raw, "position_lat")),
            "lon": semicircles_to_degrees(_coalesce(raw, "position_long")),
        }
    )
    tidy.index = ts.dt.floor("s").to_numpy()

    full_index = pd.date_range(tidy.index[0], tidy.index[-1], freq="1s")
    tidy = tidy[~tidy.index.duplicated(keep="first")].reindex(full_index)
    gap = tidy["speed"].isna() & tidy["power"].isna() & tidy["lat"].isna()
    # label runs of missing data; only short ones get interpolated
    run_id = (gap != gap.shift()).cumsum()
    run_len = gap.groupby(run_id).transform("sum")
    fillable = gap & (run_len <= MAX_INTERP_GAP_S)
    tidy = tidy.interpolate(limit=MAX_INTERP_GAP_S, limit_area="inside")
    gap = gap & ~fillable

    # Genuine stops/auto-pause appear as long timestamp gaps: speed is unknown, not zero
    tidy["gap"] = gap.to_numpy()
    tidy.loc[tidy["gap"], ["speed", "power", "cadence"]] = np.nan
    tidy["speed"] = tidy["speed"].clip(lower=0)
    tidy["power"] = tidy["power"].clip(lower=0)

    tidy.insert(0, "timestamp", tidy.index)
    tidy.insert(0, "t_s", np.arange(len(tidy), dtype=float))
    tidy = tidy.reset_index(drop=True)

    # integrate speed if the device didn't record distance
    if tidy["dist"].isna().all():
        tidy["dist"] = np.nancumsum(tidy["speed"].fillna(0).to_numpy())

    laps = _prepare_laps(laps_raw)
    tidy["lap"] = _assign_laps(tidy["timestamp"], laps)
    return Ride(name=name, df=tidy, laps=laps)


def _prepare_laps(laps_raw: pd.DataFrame) -> pd.DataFrame:
    if laps_raw.empty or "start_time" not in laps_raw.columns:
        return pd.DataFrame(columns=["lap", "start", "end", "duration_s", "distance_km", "avg_power", "avg_speed_kmh"])
    laps = laps_raw.copy()
    laps["start_time"] = pd.to_datetime(laps["start_time"], errors="coerce")  # some devices write junk
    laps = laps.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
    if laps.empty:
        return _prepare_laps(pd.DataFrame())
    start = laps["start_time"]
    dur = pd.to_numeric(
        laps.get("total_elapsed_time", laps.get("total_timer_time", 0.0)), errors="coerce"
    ).fillna(0.0)
    out = pd.DataFrame(
        {
            "lap": np.arange(1, len(laps) + 1),
            "start": start,
            "end": start + pd.to_timedelta(dur, unit="s"),
            "duration_s": dur,
            "distance_km": pd.to_numeric(laps.get("total_distance"), errors="coerce") / 1000.0,
            # float64 always (NaN when missing) so the fast decoder and the fitparse fallback agree on dtype
            "avg_power": pd.to_numeric(laps.get("avg_power"), errors="coerce").astype(float),
            "avg_speed_kmh": _coalesce(laps, "enhanced_avg_speed", "avg_speed") * 3.6,
        }
    )
    return out


def _assign_laps(timestamps: pd.Series, laps: pd.DataFrame) -> np.ndarray:
    if laps.empty:
        return np.ones(len(timestamps), dtype=int)
    starts = laps["start"].to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(starts, timestamps.to_numpy(dtype="datetime64[ns]"), side="right")
    return np.clip(idx, 1, len(laps)).astype(int)


def ride_distance_from_gps(ride: Ride) -> np.ndarray:
    """GPS-integrated distance (m); handy for sanity-checking wheel-sensor distance."""
    d = ride.df.dropna(subset=["lat", "lon"])
    return path_distance_m(d["lat"].to_numpy(), d["lon"].to_numpy())
