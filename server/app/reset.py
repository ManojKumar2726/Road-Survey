"""Clear reported data without touching configuration.

Routes are configuration -- they come from `edge/routes/*.geojson` and are
re-seeded on every startup, so wiping them would only mean reloading them.
Everything else (sightings, the defects clustered from them, the bus roster
derived from them, and the stored crops) is *reported* data, and that is what
a fresh start should drop.

Written with SQLAlchemy deletes rather than by removing the SQLite file, so it
behaves the same against Postgres.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .db import MEDIA_DIR
from .models import Bus, Defect, Event

CROP_DIR = MEDIA_DIR / "crops"


def _clear_crops() -> int:
    """Drop stored crops. They're only reachable via event rows being deleted."""
    if not CROP_DIR.is_dir():
        return 0
    n = 0
    for f in CROP_DIR.glob("*.jpg"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass  # a file held open elsewhere shouldn't fail the reset
    return n


def reset_data(db: Session, drop_crops: bool = True) -> dict[str, Any]:
    """Delete every sighting, defect and bus. Returns what was removed."""
    counts = {
        "events": db.scalar(select(func.count(Event.id))) or 0,
        "defects": db.scalar(select(func.count(Defect.id))) or 0,
        "buses": db.scalar(select(func.count(Bus.id))) or 0,
    }

    # Events reference defects, so clear the link before deleting either.
    db.execute(Event.__table__.update().values(defect_id=None))
    db.execute(delete(Event))
    db.execute(delete(Defect))
    db.execute(delete(Bus))
    db.commit()

    counts["crops"] = _clear_crops() if drop_crops else 0
    return counts


def summarise(counts: dict[str, Any]) -> str:
    return (
        f"{counts['events']} events, {counts['defects']} defects, "
        f"{counts['buses']} buses, {counts['crops']} crops"
    )
