"""Geographic helpers: distance, bearing, smoothing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

EARTH_RADIUS_M = 6371008.8
SEMICIRCLE_TO_DEG = 180.0 / 2 ** 31


def semicircles_to_degrees(values):
    return np.asarray(values, dtype=float) * SEMICIRCLE_TO_DEG


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing (0 = north, clockwise) from point 1 to point 2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlmb = np.radians(lon2) - np.radians(lon1)
    x = np.sin(dlmb) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def path_distance_m(lat, lon):
    """Cumulative horizontal distance along a lat/lon path."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    step = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.concatenate([[0.0], np.cumsum(step)])


def smooth_heading_deg(heading_deg, window: int):
    """Smooth a heading series by averaging its unit vectors (avoids 359/1 wrap issues)."""
    h = np.radians(np.asarray(heading_deg, dtype=float))
    s = uniform_filter1d(np.sin(h), size=max(1, window), mode="nearest")
    c = uniform_filter1d(np.cos(h), size=max(1, window), mode="nearest")
    return (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Signed smallest difference a - b in degrees (-180, 180]."""
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0
