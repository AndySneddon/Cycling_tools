"""Named, saved race plans.

Each race is stored as ``<slug>.json`` (the settings) plus ``<slug>.gpx`` (a copy of the course, so uploaded courses
survive) in ``race_profiles/``. The folder is git-ignored because courses contain real locations. Set the
``CYCLING_TOOLS_RACE_PROFILES`` environment variable to use a different folder (the tests do).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
MAX_NAME_LEN = 80


class RaceProfileError(ValueError):
    """Bad name or unreadable profile."""


def store_dir() -> Path:
    return Path(os.environ.get("CYCLING_TOOLS_RACE_PROFILES", ROOT / "race_profiles"))


def slugify(name: str) -> str:
    """File-safe identifier for a race name ('Challenge Almere 2027!' -> 'challenge-almere-2027')."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]


@dataclass
class RaceProfile:
    name: str
    settings: dict
    course_name: str = ""
    notes: str = ""
    saved_at: str = ""
    last_prediction: dict = field(default_factory=dict)
    schema: int = SCHEMA_VERSION

    @property
    def slug(self) -> str:
        return slugify(self.name)


def _paths(slug: str, store: Path | None) -> tuple[Path, Path]:
    d = store or store_dir()
    return d / f"{slug}.json", d / f"{slug}.gpx"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_name(name: str) -> str:
    """Return the cleaned display name or raise :class:`RaceProfileError`."""
    clean = " ".join(name.split())
    if not clean:
        raise RaceProfileError("Enter a name for the race.")
    if len(clean) > MAX_NAME_LEN:
        raise RaceProfileError(f"Keep the name under {MAX_NAME_LEN} characters.")
    if not slugify(clean):
        raise RaceProfileError("The name needs at least one letter or number.")
    return clean


def exists(name_or_slug: str, store: Path | None = None) -> bool:
    slug = slugify(name_or_slug)
    return bool(slug) and _paths(slug, store)[0].exists()


def save_profile(
    name: str,
    settings: dict,
    *,
    gpx_bytes: bytes | None = None,
    course_name: str = "",
    notes: str = "",
    last_prediction: dict | None = None,
    store: Path | None = None,
) -> RaceProfile:
    """Create or overwrite the profile called ``name``. Names that slugify the same are the same race."""
    clean = validate_name(name)
    profile = RaceProfile(
        name=clean,
        settings=dict(settings),
        course_name=course_name,
        notes=notes.strip(),
        saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        last_prediction=dict(last_prediction or {}),
    )
    json_path, gpx_path = _paths(profile.slug, store)
    _atomic_write(json_path, json.dumps(asdict(profile), indent=2).encode())
    if gpx_bytes:
        _atomic_write(gpx_path, gpx_bytes)
    return profile


def load_profile(name_or_slug: str, store: Path | None = None) -> tuple[RaceProfile, bytes | None]:
    """Return the profile and its saved GPX bytes (None if no course was stored)."""
    slug = slugify(name_or_slug)
    json_path, gpx_path = _paths(slug, store)
    try:
        raw = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RaceProfileError(f"Could not read race profile {name_or_slug!r}: {exc}") from exc
    known = {"name", "settings", "course_name", "notes", "saved_at", "last_prediction", "schema"}
    profile = RaceProfile(**{k: v for k, v in raw.items() if k in known})
    gpx = gpx_path.read_bytes() if gpx_path.exists() else None
    return profile, gpx


def list_profiles(store: Path | None = None) -> list[RaceProfile]:
    """All saved profiles, most recently saved first. Unreadable files are skipped."""
    d = store or store_dir()
    if not d.exists():
        return []
    out = []
    for path in d.glob("*.json"):
        try:
            out.append(load_profile(path.stem, d)[0])
        except RaceProfileError:
            continue
    return sorted(out, key=lambda p: p.saved_at, reverse=True)


def delete_profile(name_or_slug: str, store: Path | None = None) -> None:
    slug = slugify(name_or_slug)
    if not slug:
        raise RaceProfileError("No such race profile.")
    for path in _paths(slug, store):
        path.unlink(missing_ok=True)
