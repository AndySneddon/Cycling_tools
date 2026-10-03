from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitparse import FitFile
from scipy.optimize import curve_fit
from scipy.stats import linregress, norm


# ---------------------------
# Configuration / constants
# ---------------------------

CASSETTE_OPTIONS: dict[str, list[int]] = {
    "ultegra_12_speed_11_30_cassette": [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30],
    "ultegra_12_speed_11_32_cassette": [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30, 32],
}
DEFAULT_CASSETTE = "ultegra_12_speed_11_30_cassette"
DEFAULT_TYRE_WIDTH_M = 0.028
BEAD_SEAT_DIAMETER_M = 0.622
SECONDS_PER_MINUTE = 60.0
MIN_POWER_W = 100.0
MIN_CADENCE_RPM = 60.0
FIT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "speed": ("enhanced_speed",),
    "altitude": ("enhanced_altitude",),
    "avg_speed": ("enhanced_avg_speed",),
    "max_speed": ("enhanced_max_speed",),
}


def define_cassette_sizes(cassette: str) -> list[int]:
    """Return the sprocket tooth counts for a named cassette option."""
    return CASSETTE_OPTIONS.get(cassette, []).copy()


def fitfile_to_dataframe(file_path: str | Path, message_name: str = "record") -> pd.DataFrame:
    """Convert FIT messages into a pandas DataFrame."""
    path = Path(file_path)
    try:
        fitfile = FitFile(str(path))
        records: list[dict] = []
        for record in fitfile.get_messages(message_name):
            records.append({data.name: data.value for data in record})
        return pd.DataFrame.from_records(records)
    except Exception as e:
        raise ValueError(f"Error processing {message_name!r} messages in '{path}': {e}") from e


def normalise_fit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical columns for common FIT field aliases."""
    normalised = df.copy()
    for canonical, aliases in FIT_FIELD_ALIASES.items():
        available_aliases = [alias for alias in aliases if alias in normalised.columns]
        if not available_aliases:
            continue

        if canonical not in normalised.columns:
            normalised[canonical] = normalised[available_aliases[0]]
            available_aliases = available_aliases[1:]

        for alias in available_aliases:
            normalised[canonical] = normalised[canonical].fillna(normalised[alias])

    return normalised


def records_with_laps(records_df: pd.DataFrame, laps_df: pd.DataFrame) -> pd.DataFrame:
    """Attach 1-based lap numbers to record rows using FIT lap start times."""
    records = records_df.copy()
    records["lap"] = pd.NA

    if records.empty or laps_df.empty:
        return records

    if "timestamp" not in records.columns or "start_time" not in laps_df.columns:
        return records

    record_times = pd.to_datetime(records["timestamp"], errors="coerce")
    laps = laps_df.copy()
    laps["_lap_start"] = pd.to_datetime(laps["start_time"], errors="coerce")
    laps = laps.dropna(subset=["_lap_start"]).sort_values("_lap_start").reset_index(drop=True)
    if laps.empty:
        return records

    for lap_idx, lap in laps.iterrows():
        lap_number = lap_idx + 1
        start = lap["_lap_start"]
        is_last_lap = lap_idx == len(laps) - 1

        if is_last_lap:
            elapsed = lap.get("total_elapsed_time", lap.get("total_timer_time", 0.0))
            end = start + pd.to_timedelta(float(elapsed or 0.0), unit="s")
            mask = (record_times >= start) & (record_times <= end)
        else:
            end = laps.loc[lap_idx + 1, "_lap_start"]
            mask = (record_times >= start) & (record_times < end)

        records.loc[mask, "lap"] = lap_number

    return records


def _to_float_array(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        array = np.ravel(array)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return array


def _show_figure(fig) -> None:
    if plt.get_backend().lower().endswith("agg"):
        plt.close(fig)
        return
    plt.show()


def estimate_gear_ratio(
    cadence_rpm: np.ndarray | pd.Series | float,
    speed_mps: np.ndarray | pd.Series | float,
    tyre_width_m: float = DEFAULT_TYRE_WIDTH_M,
) -> np.ndarray | float:
    """
    Estimate gear ratio from cadence and speed.

    gear_ratio = wheel revs per second / crank revs per second.
    """
    tyre_diameter_m = BEAD_SEAT_DIAMETER_M + 2 * tyre_width_m
    tyre_circumference_m = tyre_diameter_m * math.pi

    cadence_rpm = np.asarray(cadence_rpm, dtype=float)
    speed_mps = np.asarray(speed_mps, dtype=float)
    cadence_rpm, speed_mps = np.broadcast_arrays(cadence_rpm, speed_mps)

    wheel_revs_per_s = speed_mps / tyre_circumference_m
    crank_revs_per_s = cadence_rpm / SECONDS_PER_MINUTE

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = wheel_revs_per_s / crank_revs_per_s
    ratio = np.where(crank_revs_per_s > 0, ratio, np.nan)
    return float(ratio) if ratio.ndim == 0 else ratio


def calculate_approximate_gears(
    gear_ratios: Sequence[float] | np.ndarray,
    chainring_teeth: int,
    cassette_sizes: Sequence[int],
) -> np.ndarray:
    """Map gear ratio estimates to the nearest cassette sprocket."""
    if not cassette_sizes:
        raise ValueError("cassette_sizes is required to calculate approximate gear")

    ratios = np.asarray(gear_ratios, dtype=float)
    if ratios.ndim != 1:
        ratios = np.ravel(ratios)

    cassette = np.asarray(cassette_sizes, dtype=int)
    expected = chainring_teeth / cassette.astype(float)
    distances = np.abs(ratios[:, np.newaxis] - expected[np.newaxis, :])
    distances[~np.isfinite(distances)] = np.inf

    if np.any(np.all(np.isinf(distances), axis=1)):
        raise ValueError("gear_ratios must contain finite values")

    return cassette[np.argmin(distances, axis=1)]


def calculate_approximate_gear(
    gear_ratio: float,
    chainring_teeth: int,
    cassette_sizes: Sequence[int],
) -> int:
    """
    Given a (wheel/crank) gear ratio estimate and a candidate chainring,
    return the closest cassette sprocket tooth count.
    """
    return int(calculate_approximate_gears([gear_ratio], chainring_teeth, cassette_sizes)[0])


def estimate_cda(
    speed_mps: Iterable[float],
    power_w: Iterable[float],
    mass_kg: float,
    elevation_m: Iterable[float] | None = None,
    wind_speed_mps: float = 0.0,
    wind_direction_deg: float = 45.0,
    rolling_resistance: float = 0.00309,
    air_density: float = 1.225,
    drivetrain_efficiency: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Estimate CdA from speed & power.

    Returns:
        valid_cda, valid_speeds, valid_power, weighted_cda, cda_estimated_fit
    """
    speed = _to_float_array(speed_mps, "speed_mps")
    power = _to_float_array(power_w, "power_w")
    if speed.shape != power.shape:
        raise ValueError("speed_mps and power_w must have the same length.")
    wheel_power = power * drivetrain_efficiency

    g = 9.81
    rel_speed = speed + wind_speed_mps * np.cos(np.radians(wind_direction_deg))

    p_roll = mass_kg * g * rolling_resistance * speed

    p_grade = 0.0
    if elevation_m is not None:
        elev = _to_float_array(elevation_m, "elevation_m")
        if elev.shape != speed.shape:
            raise ValueError("elevation_m must have the same length as speed_mps.")

        with np.errstate(divide="ignore", invalid="ignore"):
            gradient = np.gradient(elev) / speed
        p_grade = mass_kg * g * np.sin(np.arctan(gradient)) * speed
        p_grade = np.nan_to_num(p_grade, nan=0.0, posinf=0.0, neginf=0.0)

    p_aero = wheel_power - p_roll - p_grade
    p_aero = np.maximum(p_aero, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        cda = (2.0 * p_aero) / (air_density * (rel_speed ** 3))

    valid = (
        (speed > 3.0)
        & (wheel_power > 50.0)
        & (rel_speed > 0.0)
        & (cda > 0.0)
        & (cda < 1.0)
        & np.isfinite(cda)
    )
    valid_speeds = speed[valid]
    valid_power = power[valid]
    valid_cda = cda[valid]
    valid_p_aero = p_aero[valid]
    valid_rel_speed = rel_speed[valid]

    if valid_cda.size < 2:
        raise ValueError("Insufficient valid data to estimate CdA.")

    weights = valid_p_aero / np.sum(valid_p_aero) if np.sum(valid_p_aero) > 0 else None
    weighted_cda = float(np.average(valid_cda, weights=weights))

    def aero_model(v: np.ndarray, cda_: float) -> np.ndarray:
        return 0.5 * cda_ * air_density * (v ** 3)

    (cda_fit,), _ = curve_fit(aero_model, valid_rel_speed, valid_p_aero, p0=[weighted_cda])
    cda_fit = float(cda_fit)

    fig, ax = plt.subplots()
    ax.scatter(valid_speeds, valid_cda, label="CdA samples")

    if np.unique(valid_speeds).size > 1:
        slope, intercept, r_value, _, _ = linregress(valid_speeds, valid_cda)
        ax.plot(
            valid_speeds,
            slope * valid_speeds + intercept,
            label=f"Trend (R^2={r_value**2:.2f})",
        )

    ax.axhline(cda_fit, linestyle="--", label="Power-space fit")
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("CdA")
    ax.set_title("CdA vs Speed")
    ax.legend()
    fig.tight_layout()
    _show_figure(fig)

    return valid_cda, valid_speeds, valid_power, weighted_cda, cda_fit


def plot_cda_scatter(
    speeds: Iterable[float],
    cda: Iterable[float],
    *,
    title: str = "CdA vs Speed (valid samples)",
    show_binned_median: bool = True,
    bins: int = 12,
    show_hist_inset: bool = True,
) -> None:
    """
    Cluster-friendly CdA scatter plot.

    - Scatter: CdA vs speed with transparency to reveal clusters
    - Optional: binned median + IQR bands to show structure
    - Optional: small histogram inset to spot multimodality
    """
    speeds = _to_float_array(speeds, "speeds")
    cda = _to_float_array(cda, "cda")
    if speeds.shape != cda.shape:
        raise ValueError("speeds and cda must have the same length.")

    mask = np.isfinite(speeds) & np.isfinite(cda)
    speeds = speeds[mask]
    cda = cda[mask]
    if speeds.size == 0:
        raise ValueError("No finite speed/CdA samples to plot.")

    fig, ax = plt.subplots()

    ax.scatter(speeds, cda, s=10, alpha=0.25)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("CdA")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    if show_binned_median and speeds.size >= 20:
        bins = max(1, int(bins))
        edges = np.linspace(speeds.min(), speeds.max(), bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        med = np.full(bins, np.nan)
        q25 = np.full(bins, np.nan)
        q75 = np.full(bins, np.nan)

        for i in range(bins):
            in_bin = (speeds >= edges[i]) & (speeds < edges[i + 1])
            if i == bins - 1:
                in_bin = (speeds >= edges[i]) & (speeds <= edges[i + 1])
            if np.any(in_bin):
                med[i] = np.median(cda[in_bin])
                q25[i] = np.quantile(cda[in_bin], 0.25)
                q75[i] = np.quantile(cda[in_bin], 0.75)

        ok = np.isfinite(med)
        ax.plot(centers[ok], med[ok], linewidth=2, label="Binned median")
        ax.fill_between(centers[ok], q25[ok], q75[ok], alpha=0.15, label="Binned IQR")
        ax.legend()

    if show_hist_inset and cda.size >= 20:
        inset = ax.inset_axes([0.72, 0.62, 0.25, 0.3])
        inset.hist(cda, bins=25, alpha=0.8)
        inset.set_title("CdA hist", fontsize=9)
        inset.grid(True, alpha=0.2)

    fig.tight_layout()
    _show_figure(fig)


# ---------------------------
# Main parser
# ---------------------------

@dataclass
class GearSummary:
    chainring_teeth: int
    minutes_in_middle_2: float
    minutes_in_middle_4: float


class FitFileParser:
    def __init__(self, fitfile_path: str | Path, cassette: str = DEFAULT_CASSETTE):
        self.fitfile_path = str(fitfile_path)
        self.laps_df = normalise_fit_columns(fitfile_to_dataframe(fitfile_path, "lap"))
        self.records_df = normalise_fit_columns(fitfile_to_dataframe(fitfile_path, "record"))
        self.records_df = records_with_laps(self.records_df, self.laps_df)
        self.df = self.records_df.copy()

        self.cassette_chosen: list[int] = define_cassette_sizes(cassette)
        self.cassette_labels: list[str] = [str(t) for t in self.cassette_chosen]

        self.teeth_to_investigate: list[int] = list(range(40, 67, 2))
        self.teeth_labels: list[str] = [str(t) for t in self.teeth_to_investigate]

        self.time_in_middle_2_gears: list[float] = []
        self.time_in_middle_4_gears: list[float] = []
        self.gear_summaries: list[GearSummary] = []

    def _middle_gears(self) -> tuple[set[int], set[int]]:
        if len(self.cassette_chosen) < 8:
            raise ValueError("Cassette must have at least 8 sprockets for middle-gear analysis.")

        middle_2 = {self.cassette_chosen[5], self.cassette_chosen[6]}
        middle_4 = {
            self.cassette_chosen[4],
            self.cassette_chosen[5],
            self.cassette_chosen[6],
            self.cassette_chosen[7],
        }
        return middle_2, middle_4

    def _prepare_filtered_df(self) -> pd.DataFrame:
        """Filter FIT records used for the chainring/cassette analysis."""
        if self.records_df.empty:
            raise ValueError("FIT file produced an empty DataFrame.")

        required = {"power", "cadence", "speed", "altitude"}
        missing = required - set(self.records_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in FIT data: {sorted(missing)}")

        df = self.records_df.copy()
        df["altitude_change"] = df["altitude"].diff()

        df = df.loc[
            (df["power"] > MIN_POWER_W)
            & (df["cadence"] > MIN_CADENCE_RPM)
            & (df["altitude_change"] > 0)
        ].copy()

        df["gear_ratio"] = estimate_gear_ratio(df["cadence"].to_numpy(), df["speed"].to_numpy())
        df.loc[~np.isfinite(df["gear_ratio"]), "gear_ratio"] = np.nan
        df.dropna(subset=["gear_ratio"], inplace=True)

        return df

    def calculate_best_gear(self) -> list[GearSummary]:
        if not self.cassette_chosen:
            raise ValueError("Cassette is not defined (unknown cassette option).")

        df = self._prepare_filtered_df()
        middle_2, middle_4 = self._middle_gears()

        self.time_in_middle_2_gears.clear()
        self.time_in_middle_4_gears.clear()

        summaries: list[GearSummary] = []
        ratios = df["gear_ratio"].to_numpy()

        for teeth in self.teeth_to_investigate:
            col = f"gear_with_{teeth}"
            df[col] = calculate_approximate_gears(ratios, teeth, self.cassette_chosen)

            counts = df[col].value_counts().reindex(self.cassette_chosen, fill_value=0)
            minutes_middle_2 = round(
                float(counts.loc[list(middle_2)].sum()) / SECONDS_PER_MINUTE,
                1,
            )
            minutes_middle_4 = round(
                float(counts.loc[list(middle_4)].sum()) / SECONDS_PER_MINUTE,
                1,
            )

            self.time_in_middle_2_gears.append(minutes_middle_2)
            self.time_in_middle_4_gears.append(minutes_middle_4)

            summaries.append(GearSummary(teeth, minutes_middle_2, minutes_middle_4))

        self.df = df
        self.gear_summaries = summaries
        return summaries

    def lap_summary(self) -> pd.DataFrame:
        if self.laps_df.empty:
            raise ValueError("No lap messages found in FIT file.")

        summary = pd.DataFrame({"lap": np.arange(1, len(self.laps_df) + 1)})
        if "start_time" in self.laps_df.columns:
            summary["start_time"] = self.laps_df["start_time"]
        if "total_timer_time" in self.laps_df.columns:
            summary["duration_min"] = (
                self.laps_df["total_timer_time"].astype(float) / SECONDS_PER_MINUTE
            )
        elif "total_elapsed_time" in self.laps_df.columns:
            summary["duration_min"] = (
                self.laps_df["total_elapsed_time"].astype(float) / SECONDS_PER_MINUTE
            )
        if "total_distance" in self.laps_df.columns:
            summary["distance_km"] = self.laps_df["total_distance"].astype(float) / 1000.0
        if "avg_power" in self.laps_df.columns:
            summary["avg_power_w"] = self.laps_df["avg_power"]
        if "avg_speed" in self.laps_df.columns:
            summary["avg_speed_kmh"] = self.laps_df["avg_speed"].astype(float) * 3.6
        if "avg_cadence" in self.laps_df.columns:
            summary["avg_cadence_rpm"] = self.laps_df["avg_cadence"]
        if "avg_heart_rate" in self.laps_df.columns:
            summary["avg_heart_rate_bpm"] = self.laps_df["avg_heart_rate"]

        numeric_cols = summary.select_dtypes(include=np.number).columns.difference(["lap"])
        summary[numeric_cols] = summary[numeric_cols].round(1)
        return summary

    def _cda_records(
        self,
        mass_kg: float,
        *,
        wind_speed_mps: float = 0.0,
        wind_direction_deg: float = 45.0,
        rolling_resistance: float = 0.00309,
        air_density: float = 1.225,
        drivetrain_efficiency: float = 1.0,
        use_elevation: bool = False,
        elevation_smoothing_seconds: int = 60,
        min_speed_mps: float = 3.0,
        min_power_w: float = 50.0,
        max_cda: float = 1.0,
    ) -> pd.DataFrame:
        required = {"speed", "power"}
        missing = required - set(self.records_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in FIT data: {sorted(missing)}")

        df = self.records_df.copy()
        speed = pd.to_numeric(df["speed"], errors="coerce").to_numpy(dtype=float)
        power = pd.to_numeric(df["power"], errors="coerce").to_numpy(dtype=float)
        wheel_power = power * drivetrain_efficiency

        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
            dt_s = timestamps.diff().dt.total_seconds()
            fallback_dt = dt_s.loc[dt_s > 0].median()
            if not np.isfinite(fallback_dt):
                fallback_dt = 1.0
            dt_s = dt_s.fillna(fallback_dt).clip(lower=0.1).to_numpy(dtype=float)
        else:
            dt_s = np.ones(len(df), dtype=float)

        g = 9.81
        rel_speed = speed + wind_speed_mps * np.cos(np.radians(wind_direction_deg))
        p_roll = mass_kg * g * rolling_resistance * speed

        p_grade = np.zeros_like(speed)
        if use_elevation and "altitude" in df.columns:
            elevation = pd.to_numeric(
                df["altitude"],
                errors="coerce",
            ).interpolate(limit_direction="both")
            window = max(1, int(elevation_smoothing_seconds))
            if window > 1:
                elevation = elevation.rolling(window, center=True, min_periods=1).median()
            vertical_speed = elevation.diff().fillna(0.0).to_numpy(dtype=float) / dt_s
            p_grade = mass_kg * g * vertical_speed
            p_grade = np.nan_to_num(p_grade, nan=0.0, posinf=0.0, neginf=0.0)

        p_aero = np.maximum(wheel_power - p_roll - p_grade, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cda = (2.0 * p_aero) / (air_density * (rel_speed ** 3))

        df["dt_s"] = dt_s
        df["speed_kmh"] = speed * 3.6
        df["relative_speed"] = rel_speed
        df["wheel_power"] = wheel_power
        df["rolling_power"] = p_roll
        df["grade_power"] = p_grade
        df["cda"] = cda
        df["aero_power"] = p_aero
        valid = (
            (speed > min_speed_mps)
            & (wheel_power > min_power_w)
            & (rel_speed > 0.0)
            & (cda > 0.0)
            & (cda < max_cda)
            & np.isfinite(cda)
        )
        return df.loc[valid].copy()

    def lap_cda_summary(
        self,
        mass_kg: float,
        *,
        include_laps: Sequence[int] | None = None,
        wind_speed_mps: float = 0.0,
        wind_direction_deg: float = 45.0,
        rolling_resistance: float = 0.00309,
        air_density: float = 1.225,
        drivetrain_efficiency: float = 1.0,
        use_elevation: bool = False,
        elevation_smoothing_seconds: int = 60,
        min_speed_mps: float = 3.0,
        min_power_w: float = 50.0,
        max_cda: float = 1.0,
    ) -> pd.DataFrame:
        data = self._cda_records(
            mass_kg,
            wind_speed_mps=wind_speed_mps,
            wind_direction_deg=wind_direction_deg,
            rolling_resistance=rolling_resistance,
            air_density=air_density,
            drivetrain_efficiency=drivetrain_efficiency,
            use_elevation=use_elevation,
            elevation_smoothing_seconds=elevation_smoothing_seconds,
            min_speed_mps=min_speed_mps,
            min_power_w=min_power_w,
            max_cda=max_cda,
        )
        if include_laps is not None:
            data = data.loc[data["lap"].isin(include_laps)].copy()
        if data.empty:
            raise ValueError("No valid CdA samples available for the requested lap selection.")

        summaries: list[dict] = []
        for lap, group in data.groupby("lap", dropna=False, sort=True):
            label = "Unlapped" if pd.isna(lap) else int(lap)
            aero_power_sum = group["aero_power"].sum()
            weights = group["aero_power"] / aero_power_sum if aero_power_sum > 0 else None
            weighted_cda = float(np.average(group["cda"], weights=weights))

            relative_speed = group["relative_speed"].to_numpy(dtype=float)
            dt_s = group["dt_s"].to_numpy(dtype=float)
            x = 0.5 * air_density * (relative_speed ** 3)
            y = group["aero_power"].to_numpy(dtype=float)
            energy_denominator = float(np.sum(x * dt_s))
            denominator = float(np.dot(x, x))
            energy_cda = (
                float(np.sum(y * dt_s) / energy_denominator)
                if energy_denominator > 0
                else np.nan
            )
            least_squares_cda = float(np.dot(x, y) / denominator) if denominator > 0 else np.nan

            summaries.append(
                {
                    "lap": label,
                    "valid_samples": len(group),
                    "median_cda": group["cda"].median(),
                    "mean_cda": group["cda"].mean(),
                    "weighted_cda": weighted_cda,
                    "energy_cda": energy_cda,
                    "least_squares_cda": least_squares_cda,
                    "avg_power_w": group["power"].mean(),
                    "avg_wheel_power_w": group["wheel_power"].mean(),
                    "avg_aero_power_w": group["aero_power"].mean(),
                    "avg_grade_power_w": group["grade_power"].mean(),
                    "avg_speed_kmh": group["speed_kmh"].mean(),
                }
            )

        summary = pd.DataFrame(summaries)
        numeric_cols = summary.select_dtypes(include=np.number).columns.difference(
            ["lap", "valid_samples"]
        )
        summary[numeric_cols] = summary[numeric_cols].round(3)
        return summary

    # ---------------------------
    # Plotting helpers
    # ---------------------------

    def plot_lap_cda_clusters(
        self,
        mass_kg: float,
        *,
        include_laps: Sequence[int] | None = None,
        wind_speed_mps: float = 0.0,
        wind_direction_deg: float = 45.0,
        rolling_resistance: float = 0.00309,
        air_density: float = 1.225,
        drivetrain_efficiency: float = 1.0,
        use_elevation: bool = False,
        elevation_smoothing_seconds: int = 60,
        min_speed_mps: float = 3.0,
        min_power_w: float = 50.0,
        max_cda: float = 1.0,
    ) -> None:
        data = self._cda_records(
            mass_kg,
            wind_speed_mps=wind_speed_mps,
            wind_direction_deg=wind_direction_deg,
            rolling_resistance=rolling_resistance,
            air_density=air_density,
            drivetrain_efficiency=drivetrain_efficiency,
            use_elevation=use_elevation,
            elevation_smoothing_seconds=elevation_smoothing_seconds,
            min_speed_mps=min_speed_mps,
            min_power_w=min_power_w,
            max_cda=max_cda,
        )
        if include_laps is not None:
            data = data.loc[data["lap"].isin(include_laps)].copy()
        if data.empty:
            raise ValueError("No valid CdA samples available for the requested lap selection.")

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")

        grouped = data.groupby("lap", dropna=False, sort=True)
        for i, (lap, group) in enumerate(grouped):
            if pd.isna(lap):
                label = "Unlapped"
            else:
                label = f"Lap {int(lap)}"
            ax.scatter(
                group["speed_kmh"],
                group["cda"],
                s=12,
                alpha=0.28,
                color=cmap(i % cmap.N),
                label=label,
            )

            if len(group) >= 20:
                median_cda = group["cda"].median()
                median_speed = group["speed_kmh"].median()
                ax.scatter(
                    [median_speed],
                    [median_cda],
                    s=90,
                    color=cmap(i % cmap.N),
                    edgecolor="black",
                    linewidth=0.8,
                )

        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("CdA")
        ax.set_title("CdA clusters by lap")
        ax.grid(True, alpha=0.2)
        ax.legend(title="FIT lap")
        fig.tight_layout()
        _show_figure(fig)

    def plot_lap_energy_cda(
        self,
        mass_kg: float,
        *,
        include_laps: Sequence[int] | None = None,
        wind_speed_mps: float = 0.0,
        wind_direction_deg: float = 45.0,
        rolling_resistance: float = 0.00309,
        air_density: float = 1.225,
        drivetrain_efficiency: float = 1.0,
        use_elevation: bool = False,
        elevation_smoothing_seconds: int = 60,
        min_speed_mps: float = 3.0,
        min_power_w: float = 50.0,
        max_cda: float = 1.0,
    ) -> None:
        data = self._cda_records(
            mass_kg,
            wind_speed_mps=wind_speed_mps,
            wind_direction_deg=wind_direction_deg,
            rolling_resistance=rolling_resistance,
            air_density=air_density,
            drivetrain_efficiency=drivetrain_efficiency,
            use_elevation=use_elevation,
            elevation_smoothing_seconds=elevation_smoothing_seconds,
            min_speed_mps=min_speed_mps,
            min_power_w=min_power_w,
            max_cda=max_cda,
        )
        if include_laps is not None:
            data = data.loc[data["lap"].isin(include_laps)].copy()
        if data.empty:
            raise ValueError("No valid CdA samples available for the requested lap selection.")

        fig, (ax_curve, ax_bar) = plt.subplots(
            1,
            2,
            figsize=(13, 5),
            gridspec_kw={"width_ratios": [2.2, 1.0]},
        )
        cmap = plt.get_cmap("tab10")
        final_labels: list[str] = []
        final_values: list[float] = []
        final_colors: list = []

        for i, (lap, group) in enumerate(data.groupby("lap", dropna=False, sort=True)):
            if "timestamp" in group.columns:
                group = group.sort_values("timestamp")

            label = "Unlapped" if pd.isna(lap) else f"Lap {int(lap)}"
            color = cmap(i % cmap.N)
            dt_s = group["dt_s"].to_numpy(dtype=float)
            relative_speed = group["relative_speed"].to_numpy(dtype=float)
            aero_power = group["aero_power"].to_numpy(dtype=float)
            demand_power = 0.5 * air_density * (relative_speed ** 3)

            cumulative_aero_work = np.cumsum(aero_power * dt_s)
            cumulative_demand = np.cumsum(demand_power * dt_s)
            valid = cumulative_demand > 0
            if not np.any(valid):
                continue

            elapsed_min = np.cumsum(dt_s) / SECONDS_PER_MINUTE
            cumulative_cda = cumulative_aero_work[valid] / cumulative_demand[valid]

            ax_curve.plot(
                elapsed_min[valid],
                cumulative_cda,
                color=color,
                linewidth=2,
                label=label,
            )

            final_labels.append(label)
            final_values.append(float(cumulative_cda[-1]))
            final_colors.append(color)

        ax_curve.set_xlabel("Elapsed valid sample time (min)")
        ax_curve.set_ylabel("Cumulative energy CdA")
        ax_curve.set_title("Energy CdA convergence by lap")
        ax_curve.grid(True, alpha=0.2)
        ax_curve.legend(title="FIT lap")

        x = np.arange(len(final_values))
        ax_bar.bar(x, final_values, color=final_colors, alpha=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(final_labels, rotation=30, ha="right")
        ax_bar.set_ylabel("Final energy CdA")
        ax_bar.set_title("Final lap value")
        ax_bar.grid(True, axis="y", alpha=0.2)

        for idx, value in enumerate(final_values):
            ax_bar.text(idx, value, f"{value:.3f}", ha="center", va="bottom")

        fig.tight_layout()
        _show_figure(fig)

    def plot_teeth(self) -> None:
        if not self.time_in_middle_2_gears or not self.time_in_middle_4_gears:
            raise ValueError("Run calculate_best_gear() first.")

        fig, ax = plt.subplots()
        ax.bar(self.teeth_labels, self.time_in_middle_4_gears, label="Middle 4 gears")
        ax.bar(self.teeth_labels, self.time_in_middle_2_gears, label="Middle 2 gears")
        ax.set_xlabel("Chainring teeth")
        ax.set_ylabel("Time (minutes)")
        ax.set_title("Time in middle 2 and 4 cassette gears")
        ax.legend()
        fig.tight_layout()
        _show_figure(fig)

    def plot_power_distribution(self, max_threshold: float = 300) -> None:
        if "power" not in self.df.columns:
            raise ValueError("Power data is missing in the DataFrame.")

        power = self.df.loc[self.df["power"] < max_threshold, "power"].dropna().astype(float)
        if power.empty:
            raise ValueError("No power samples found below max_threshold.")

        fig, ax = plt.subplots()
        ax.hist(power, bins=50, alpha=0.7)
        ax.set_xlabel("Power (W)")
        ax.set_ylabel("Frequency")
        ax.set_title("Power Distribution")
        fig.tight_layout()
        _show_figure(fig)

    def plot_cadence_distribution(
        self,
        min_threshold: float = 50,
        max_threshold: float = 120,
    ) -> None:
        if "cadence" not in self.df.columns:
            raise ValueError("Cadence data is missing in the DataFrame.")

        df_trimmed = self.df.loc[
            (self.df["cadence"] > min_threshold)
            & (self.df["cadence"] < max_threshold)
        ]
        cadence = df_trimmed["cadence"].dropna().astype(float)
        if cadence.empty:
            raise ValueError("No cadence samples found within the requested thresholds.")

        fig, ax = plt.subplots()
        ax.hist(cadence, bins=35, alpha=0.7, density=True, label="Cadence histogram")
        mu, std = norm.fit(cadence)
        if std > 0:
            x = np.linspace(min_threshold, max_threshold, 200)
            ax.plot(x, norm.pdf(x, mu, std), label=f"Normal fit (mu={mu:.1f}, sigma={std:.1f})")
        ax.set_xlabel("Cadence (rpm)")
        ax.set_ylabel("Density")
        ax.set_title("Cadence Distribution")
        ax.legend()
        fig.tight_layout()
        _show_figure(fig)

    def plot_time_in_each_cassette_tooth(self, teeth: int = 56) -> None:
        col = f"gear_with_{teeth}"
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found. Run calculate_best_gear() first.")

        counts = self.df[col].value_counts().reindex(self.cassette_chosen, fill_value=0)
        middle_2, middle_4 = self._middle_gears()
        middle_4_only = middle_4 - middle_2

        cassette = np.asarray(self.cassette_chosen)
        values = counts.to_numpy()
        x = np.arange(len(cassette))
        mask_middle_2 = np.isin(cassette, list(middle_2))
        mask_middle_4 = np.isin(cassette, list(middle_4_only))
        mask_other = ~(mask_middle_2 | mask_middle_4)

        fig, ax = plt.subplots()
        ax.bar(x[mask_other], values[mask_other], width=0.6, label="Other gears")
        ax.bar(x[mask_middle_4], values[mask_middle_4], width=0.6, label="Middle 4")
        ax.bar(x[mask_middle_2], values[mask_middle_2], width=0.6, label="Middle 2")
        ax.set_xticks(x)
        ax.set_xticklabels(self.cassette_labels)
        ax.set_xlabel("Cassette sprocket (teeth)")
        ax.set_ylabel("Time (samples / seconds)")
        ax.set_title(f"Time in each cassette gear (chainring={teeth}T)")
        ax.legend()
        fig.tight_layout()
        _show_figure(fig)

    def plot_speed_distribution(self) -> None:
        if "speed" not in self.df.columns:
            raise ValueError("Speed data is missing in the DataFrame.")

        speed_kmh = (self.df["speed"].dropna().astype(float) * 3.6)
        if speed_kmh.empty:
            raise ValueError("No speed samples found.")

        bins = max(10, int(np.nanmax(speed_kmh) - np.nanmin(speed_kmh)))
        fig, ax = plt.subplots()
        ax.hist(speed_kmh, bins=bins, alpha=0.7)
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("Frequency")
        ax.set_title("Speed Distribution")
        fig.tight_layout()
        _show_figure(fig)


def main() -> None:
    fitfile_path = Path(__file__).with_name("Manchester_District_TTA_50_WU_CD.fit")
    parser = FitFileParser(fitfile_path)
    mass_kg = 100.0
    drivetrain_efficiency = 0.97

    summaries = parser.calculate_best_gear()
    print("Gear summary:")
    print(pd.DataFrame([asdict(summary) for summary in summaries]).head().to_string(index=False))
    print("\nLap summary:")
    print(parser.lap_summary().to_string(index=False))
    print("\nLap CdA summary:")
    print(
        parser.lap_cda_summary(
            mass_kg=mass_kg,
            drivetrain_efficiency=drivetrain_efficiency,
        ).to_string(index=False)
    )

    parser.plot_teeth()
    parser.plot_time_in_each_cassette_tooth(teeth=42)
    parser.plot_lap_cda_clusters(
        mass_kg=mass_kg,
        drivetrain_efficiency=drivetrain_efficiency,
    )
    parser.plot_lap_energy_cda(
        mass_kg=mass_kg,
        drivetrain_efficiency=drivetrain_efficiency,
    )
    return parser


if __name__ == "__main__":
    parser = main()
