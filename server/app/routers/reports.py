"""The maintenance queue.

A dashboard tells you where the damage is. A work order needs it *ranked*, and
ranking is where the fleet data earns its keep: severity says how bad the
defect is, confirmation says how sure we are it exists, and age says how long
it has been ignored. A one-off low-confidence crack from this morning should
not outrank a pothole four buses have reported for a fortnight.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import taxonomy as tax
from ..db import get_db
from ..models import Defect, Event

router = APIRouter(prefix="/api", tags=["reports"])


def _utc(dt: datetime | None) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def priority(defect: Defect, now: datetime) -> float:
    """severity x confidence x corroboration x age.

    Multiplicative on purpose: a defect that is severe but unconfirmed, or
    confirmed but trivial, should not float to the top on one strong factor.
    Corroboration and age are capped so an old, much-reported minor crack
    cannot outrank a fresh pothole.
    """
    age_days = max(0.0, (now - _utc(defect.first_seen)).total_seconds() / 86400.0)
    corroboration = min(2.0, 1.0 + 0.25 * max(0, (defect.distinct_buses or 1) - 1))
    age_factor = min(2.0, 1.0 + age_days / 30.0)
    size = min(1.5, 1.0 + (defect.peak_area_pct or 0.0) / 20.0)
    return (
        (defect.severity or 0.3)
        * (defect.max_confidence or 0.3)
        * corroboration
        * age_factor
        * size
        * 100.0
    )


def _rows(db: Session, status: str | None, route_id: str | None, limit: int):
    now = datetime.now(timezone.utc)
    q = select(Defect)
    if status:
        q = q.where(Defect.status == status)
    else:
        # Unconfirmed defects are candidates, not work. Repaired ones are done.
        q = q.where(Defect.status == "open")
    if route_id:
        q = q.where(Defect.route_id == route_id)

    out = []
    for d in db.scalars(q):
        age = (now - _utc(d.first_seen)).days
        out.append(
            {
                "rank": 0,
                "defect_id": d.id,
                "priority": round(priority(d, now), 1),
                "damage_type": d.damage_type,
                "damage_label": tax.label_of(d.damage_type),
                "severity": round(d.severity or 0.0, 2),
                "status": d.status,
                "confidence": round(d.max_confidence or 0.0, 3),
                "sightings": d.sightings,
                "distinct_buses": d.distinct_buses,
                "days_open": age,
                "peak_area_pct": round(d.peak_area_pct or 0.0, 2),
                "route_id": d.route_id or "",
                "lat": round(d.lat, 6),
                "lon": round(d.lon, 6),
                "first_seen": _utc(d.first_seen).isoformat(timespec="seconds"),
                "last_seen": _utc(d.last_seen).isoformat(timespec="seconds"),
                "map_link": f"https://www.openstreetmap.org/?mlat={d.lat}&mlon={d.lon}#map=19/{d.lat}/{d.lon}",
            }
        )
    out.sort(key=lambda r: -r["priority"])
    for i, r in enumerate(out[:limit], 1):
        r["rank"] = i
    return out[:limit]


@router.get("/report")
def report(
    db: Session = Depends(get_db),
    status: str | None = Query(None, description="Defaults to open only"),
    route_id: str | None = None,
    limit: int = Query(200, le=2000),
):
    """Ranked maintenance queue, worst first."""
    rows = _rows(db, status, route_id, limit)
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["damage_type"]] = by_type.get(r["damage_type"], 0) + 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(rows),
        "by_type": by_type,
        "total_events": db.scalar(select(Event.id).order_by(Event.id.desc())) or 0,
        "items": rows,
    }


@router.get("/report.csv")
def report_csv(
    db: Session = Depends(get_db),
    status: str | None = None,
    route_id: str | None = None,
    limit: int = Query(2000, le=5000),
):
    """The same queue as a CSV, for whoever actually schedules the crews."""
    rows = _rows(db, status, route_id, limit)
    buf = io.StringIO()
    fields = [
        "rank", "defect_id", "priority", "damage_label", "severity", "status",
        "confidence", "sightings", "distinct_buses", "days_open",
        "peak_area_pct", "route_id", "lat", "lon", "first_seen", "last_seen",
        "map_link",
    ]
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    buf.seek(0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="road-maintenance-{stamp}.csv"'
        },
    )
