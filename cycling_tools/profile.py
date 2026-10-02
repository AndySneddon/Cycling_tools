"""Persistent rider profile shared between the CdA, gearing and race-planner tools."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROFILE_PATH = Path(__file__).resolve().parent.parent / "rider_profile.json"


@dataclass
class RiderProfile:
    # Mass of rider + bike + kit + bottles (kg)
    mass_kg: float = 100.0
    cda: float = 0.25
    crr: float = 0.0031
    drivetrain_eff: float = 0.97
    # Fraction of 10 m weather-model wind that reaches the rider
    wind_scale: float = 0.7
    # Cassette sprockets, smallest to largest
    cassette: list[int] = field(default_factory=lambda: [11, 12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30])
    chainring: int = 56
    tyre_circumference_m: float = 2.13
    # Preferred cadence (rpm) on the flat at race effort, and slope vs power
    cadence_flat: float = 88.0
    cadence_per_100w: float = 0.0
    ftp: float | None = None
    notes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = PROFILE_PATH) -> "RiderProfile":
        if not Path(path).exists():
            return cls()
        try:
            raw = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path = PROFILE_PATH) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
