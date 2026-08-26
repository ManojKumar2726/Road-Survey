"""Event ingest and the live feed.

Ingest is idempotent on `event_uid`: the publisher retries after a timeout it
can't distinguish from a failure, and a duplicated pothole would quietly
corrupt every count downstream.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..clustering import assign_defect
from ..db import MEDIA_DIR, get_db
from ..models import Bus, Event, utcnow
from ..schemas import EventBatch, EventIn, EventOut, IngestResult

router = APIRouter(prefix="/api", tags=["events"])

CROP_DIR = MEDIA_DIR / "crops"
MAX_CROP_BYTES = 2 * 1024 * 1024


def _save_crop(uid: str, b64: str | None) -> str | None:
    """Decode and store a crop. A bad crop must not sink the event."""
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw or len(raw) > MAX_CROP_BYTES:
        return None
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    path = CROP_DIR / f"{uid}.jpg"
    path.write_bytes(raw)
    return f"/media/crops/{path.name}"


def _as_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return utcnow()
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _ingest_one(db: Session, e: EventIn, result: IngestResult) -> None:
    exists = db.scalar(select(Event.id).where(Event.event_uid == e.event_uid))
    if exists:
        result.duplicates += 1
        return

    row = Event(
        event_uid=e.event_uid,
        bus_id=e.bus_id or "UNKNOWN",
        route_id=e.route_id or None,
        damage_type=e.damage_type,
        severity=e.severity,
        confidence=e.confidence,
        mean_confidence=e.mean_confidence,
        area_pct_frame=e.area_pct_frame,
        lat=e.lat,
        lon=e.lon,
        bearing=e.bearing,
        speed_kmh=e.speed_kmh,
        captured_at=_as_utc(e.captured_at),
        received_at=utcnow(),
        frame_idx=e.frame_idx,
        first_frame=e.first_frame,
        last_frame=e.last_frame,
        frames_seen=e.frames_seen,
        track_id=e.track_id,
        source_clip=e.source_clip,
        model_id=e.model_id,
        crop_path=_save_crop(e.event_uid, e.crop_b64),
    )
    db.add(row)
    db.flush()  # assign row.id before clustering links to it

    defect = assign_defect(db, row)
    if defect is not None:
        row.defect_id = defect.id
        result.defect_ids.append(defect.id)

    # Keep the fleet roster current without a separate registration step.
    bus = db.get(Bus, row.bus_id)
    if bus is None:
        bus = Bus(id=row.bus_id, label=row.bus_id, route_id=row.route_id)
        db.add(bus)
    bus.last_seen = row.received_at
    bus.events_reported = (bus.events_reported or 0) + 1
    if row.route_id:
        bus.route_id = row.route_id

    result.accepted += 1
    result.event_ids.append(row.id)


@router.post("/events", response_model=IngestResult)
def ingest(payload: EventBatch | list[EventIn], db: Session = Depends(get_db)):
    """Accept a batch of events. Takes a bare list too, for curl convenience."""
    events = payload.events if isinstance(payload, EventBatch) else list(payload)
    if not events:
        return IngestResult(accepted=0, duplicates=0)

    result = IngestResult(accepted=0, duplicates=0)
    for e in events:
        try:
            _ingest_one(db, e, result)
        except Exception as exc:  # one bad event must not reject the batch
            db.rollback()
            result.rejected += 1
            result.messages.append(f"{e.event_uid}: {type(exc).__name__}: {exc}")
            continue
    db.commit()
    return result


@router.get("/events", response_model=list[EventOut])
def list_events(
    db: Session = Depends(get_db),
    since: int = Query(0, description="Return events with id greater than this"),
    limit: int = Query(200, le=1000),
    bus_id: str | None = None,
    route_id: str | None = None,
    damage_type: str | None = None,
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    """The live feed. The dashboard polls this with `since=<last id it has>`."""
    q = select(Event)
    if since:
        q = q.where(Event.id > since)
    if bus_id:
        q = q.where(Event.bus_id == bus_id)
    if route_id:
        q = q.where(Event.route_id == route_id)
    if damage_type:
        q = q.where(Event.damage_type == damage_type)
    # Ascending when following a cursor -- otherwise a burst larger than the
    # limit would hand back the newest rows and skip the middle for good.
    ascending = order == "asc" or bool(since)
    q = q.order_by(Event.id.asc() if ascending else Event.id.desc()).limit(limit)
    return list(db.scalars(q))


@router.get("/events/{event_id}", response_model=EventOut)
def get_event(event_id: int, db: Session = Depends(get_db)):
    row = db.get(Event, event_id)
    if row is None:
        raise HTTPException(404, f"No event {event_id}")
    return row


@router.get("/events/by-defect/{defect_id}", response_model=list[EventOut])
def events_for_defect(defect_id: int, db: Session = Depends(get_db)):
    """Sighting history for one defect -- which bus saw it, when, how sure."""
    q = (
        select(Event)
        .where(Event.defect_id == defect_id)
        .order_by(Event.captured_at.asc())
    )
    return list(db.scalars(q))
