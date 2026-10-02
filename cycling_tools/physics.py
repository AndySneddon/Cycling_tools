"""Shared cycling physics: air density, resistive forces, wind geometry."""

from __future__ import annotations

import numpy as np

G = 9.80665
R_DRY_AIR = 287.058
R_VAPOUR = 461.495
WHEEL_INERTIA_KG = 0.14  # equivalent mass of rotating wheels


def standard_pressure_pa(altitude_m: np.ndarray | float) -> np.ndarray | float:
    """ISA station pressure at altitude."""
    return 101325.0 * (1.0 - 2.25577e-5 * np.asarray(altitude_m, dtype=float)) ** 5.25588


def air_density(
    temp_c: np.ndarray | float,
    pressure_pa: np.ndarray | float = 101325.0,
    rel_humidity: np.ndarray | float = 0.5,
) -> np.ndarray | float:
    """Moist-air density in kg/m^3 (humidity as a 0-1 fraction)."""
    t = np.asarray(temp_c, dtype=float)
    t_k = t + 273.15
    p_sat = 610.94 * np.exp(17.625 * t / (t + 243.04))  # Magnus
    p_vap = np.clip(rel_humidity, 0.0, 1.0) * p_sat
    p_dry = np.asarray(pressure_pa, dtype=float) - p_vap
    return p_dry / (R_DRY_AIR * t_k) + p_vap / (R_VAPOUR * t_k)


def headwind_component(
    wind_speed_mps: np.ndarray | float,
    wind_from_deg: np.ndarray | float,
    heading_deg: np.ndarray | float,
) -> np.ndarray | float:
    """
    Wind component along the direction of travel (positive = headwind).

    ``wind_from_deg`` is the meteorological direction the wind blows FROM.
    """
    rel = np.radians(np.asarray(wind_from_deg, dtype=float) - np.asarray(heading_deg, dtype=float))
    return np.asarray(wind_speed_mps, dtype=float) * np.cos(rel)


def crosswind_component(
    wind_speed_mps: np.ndarray | float,
    wind_from_deg: np.ndarray | float,
    heading_deg: np.ndarray | float,
) -> np.ndarray | float:
    """Wind component perpendicular to travel (positive = from the rider's right)."""
    rel = np.radians(np.asarray(wind_from_deg, dtype=float) - np.asarray(heading_deg, dtype=float))
    return np.asarray(wind_speed_mps, dtype=float) * np.sin(rel)


def aero_force(cda: float, rho: np.ndarray | float, v_air: np.ndarray | float) -> np.ndarray | float:
    """Signed aerodynamic drag force (N) for air speed ``v_air`` along the direction of travel."""
    v_air = np.asarray(v_air, dtype=float)
    return 0.5 * cda * rho * v_air * np.abs(v_air)


def power_required(
    v_ground: np.ndarray | float,
    grade: np.ndarray | float,
    *,
    mass_kg: float,
    cda: float,
    crr: float,
    rho: float,
    headwind_mps: np.ndarray | float = 0.0,
    drivetrain_eff: float = 1.0,
) -> np.ndarray | float:
    """Steady-state crank power to hold ``v_ground`` on ``grade`` (rise/run)."""
    v = np.asarray(v_ground, dtype=float)
    theta = np.arctan(grade)
    f_roll = mass_kg * G * crr * np.cos(theta)
    f_grade = mass_kg * G * np.sin(theta)
    f_aero = aero_force(cda, rho, v + headwind_mps)
    return (f_roll + f_grade + f_aero) * v / drivetrain_eff


def steady_speed(
    power_w: float,
    grade: float,
    *,
    mass_kg: float,
    cda: float,
    crr: float,
    rho: float,
    headwind_mps: float = 0.0,
    drivetrain_eff: float = 1.0,
) -> float:
    """Steady-state ground speed (m/s) for a given crank power (bisection)."""
    lo, hi = 0.01, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = power_required(
            mid, grade, mass_kg=mass_kg, cda=cda, crr=crr, rho=rho,
            headwind_mps=headwind_mps, drivetrain_eff=drivetrain_eff,
        )
        if p > power_w:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def normalised_power(power_1hz: np.ndarray) -> float:
    """Coggan normalised power from a 1 Hz power series (30 s rolling average, 4th-power mean)."""
    p = np.nan_to_num(np.asarray(power_1hz, dtype=float), nan=0.0)
    if p.size < 30:
        return float(np.mean(p)) if p.size else 0.0
    kernel = np.ones(30) / 30.0
    rolling = np.convolve(p, kernel, mode="valid")
    return float(np.mean(rolling ** 4) ** 0.25)
