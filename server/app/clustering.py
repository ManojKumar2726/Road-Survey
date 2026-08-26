"""Collapse repeat sightings into the physical defect they belong to.

Route 21G runs every fifteen minutes. Without this, one pothole becomes forty
pins stacked on five metres of road, the KPI tile counts *sightings* rather than
problems, and there is nothing to hand a maintenance crew.

With it, three things become expressible that a per-event map cannot say:

  confirmation  "seen by 4 buses on 11 passes" outranks "the model said 0.61",
                so a one-off low-confidence hit stays unconfirmed and greys out
                while a real defect promotes itself
  age           first_seen -> today gives "14 days open", which is what ranks a
                maintenance queue
  closure       a defect that stops being reported on a route that is still
                driven is a repair that verified itself

Known limitation, stated rather than solved: a radius wide enough to absorb
5-10 m of urban GPS drift will merge two genuinely separate potholes 10 m
apart. That is the trade, and it is why RADIUS_M is a tunable constant.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Defect, Event, utcnow

# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #

# How close two sightings must be to count as the same defect. Sized for GPS
# drift, not for pothole geometry.
RADIUS_M = float(os.environ.get("ROADSURVEY_CLUSTER_RADIUS_M", 15.0))

# Distinct buses required before a defect is treated as real. Two independent
# units agreeing is a far stronger signal than one unit twice.
CONFIRM_BUSES = int(os.environ.get("ROADSURVEY_CONFIRM_BUSES", 2))

# ...or this many sightings from a single bus on separate passes.
CONFIRM_SIGHTINGS = int(os.environ.get("ROADSURVEY_CONFIRM_SIGHTINGS", 3))

# A defect not seen for this long, on a route still being driven, is presumed
# repaired. Long enough that a quiet route doesn't close its own defects.
STALE_AFTER = timedelta(days=int(os.environ.get("ROADSURVEY_STALE_DAYS", 14)))

EARTH_R = 6_371_000.0

# Cracks are linear and a tracker can split one along its length, so they get a
# little more slack than a pothole, which is a point feature.
RADIUS_SCALE = {
    "pothole": 1.0,
    "alligator_crack": 1.2,
    "longitudinal_crack": 1.6,
    "transverse_crack": 1.3,
    "crack": 1.5,
}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def radius_for(damage_type: str) -> float:
    return RADIUS_M * RADIUS_SCALE.get(damage_type, 1.0)


def _as_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return utcnow()
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def _candidates(db: Session, event: Event, radius_m: float) -> list[Defect]:
    """Same damage type, inside a bounding box a little larger than the radius.

    The box is a cheap index-friendly prefilter; the haversine below does the
    actual circle. Without the box this would scan every defect in the city on
    every insert.
    """
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(0.01, math.cos(math.radians(event.lat))))
    q = select(Defect).where(
        Defect.damage_type == event.damage_type,
        Defect.status != "repaired",
        Defect.lat >= event.lat - dlat,
        Defect.lat <= event.lat + dlat,
        Defect.lon >= event.lon - dlon,
        Defect.lon <= event.lon + dlon,
    )
    return list(db.scalars(q))


def find_match(db: Session, event: Event) -> Defect | None:
    """Nearest existing defect within the radius, or None."""
    r = radius_for(event.damage_type)
    best, best_d = None, r
    for d in _candidates(db, event, r):
        dist = haversine_m(event.lat, event.lon, d.lat, d.lon)
        if dist <= best_d:
            best, best_d = d, dist
    return best


def _distinct_buses(db: Session, defect_id: int, plus_bus: str) -> int:
    seen = set(
        db.scalars(
            select(Event.bus_id).where(Event.defect_id == defect_id).distinct()
        )
    )
    seen.add(plus_bus)
    return len(seen)


def _restatus(defect: Defect) -> None:
    """Promote to open once corroborated. Never demote an open defect."""
    if defect.status == "repaired":
        return
    if defect.distinct_buses >= CONFIRM_BUSES or defect.sightings >= CONFIRM_SIGHTINGS:
        defect.status = "open"
    else:
        defect.status = "unconfirmed"


def assign_defect(db: Session, event: Event) -> Defect | None:
    """Find or create the defect this sighting belongs to.

    Called from ingest with the event already flushed, so `event.id` exists.
    An event with no position can't be clustered -- it stays in the raw log
    unattached rather than being dropped.
    """
    if event.lat is None or event.lon is None:
        return None

    captured = _as_utc(event.captured_at)
    defect = find_match(db, event)

    if defect is None:
        defect = Defect(
            damage_type=event.damage_type,
            severity=event.severity,
            lat=event.lat,
            lon=event.lon,
            route_id=event.route_id,
            sightings=1,
            distinct_buses=1,
            first_seen=captured,
            last_seen=captured,
            max_confidence=event.confidence,
            peak_area_pct=event.area_pct_frame,
            best_crop=event.crop_path,
            status="unconfirmed",
        )
        _restatus(defect)
        db.add(defect)
        db.flush()
        return defect

    # ---- merge this sighting in
    n = defect.sightings or 0
    # Running mean, weighted by sighting count: later passes refine the
    # position rather than the newest fix winning outright.
    defect.lat = (defect.lat * n + event.lat) / (n + 1)
    defect.lon = (defect.lon * n + event.lon) / (n + 1)
    defect.sightings = n + 1
    defect.distinct_buses = _distinct_buses(db, defect.id, event.bus_id)

    defect.first_seen = min(_as_utc(defect.first_seen), captured)
    defect.last_seen = max(_as_utc(defect.last_seen), captured)

    # Keep the best evidence, not the latest.
    if event.confidence > (defect.max_confidence or 0.0):
        defect.max_confidence = event.confidence
        if event.crop_path:
            defect.best_crop = event.crop_path
    defect.peak_area_pct = max(defect.peak_area_pct or 0.0, event.area_pct_frame or 0.0)
    if not defect.route_id and event.route_id:
        defect.route_id = event.route_id

    _restatus(defect)
    db.flush()
    return defect


# --------------------------------------------------------------------------- #
# Closure
# --------------------------------------------------------------------------- #


def close_stale(db: Session, now: datetime | None = None) -> int:
    """Mark long-unreported defects as repaired.

    Only closes a defect on a route that is *still being driven* -- otherwise a
    route the fleet stopped serving would report all its potholes fixed. That
    distinction is the whole reason this is safe to run automatically.

    Not demonstrable from the prototype footage: nothing in the clips ever gets
    repaired. Covered by unit test instead -- see tests/test_clustering.py.
    """
    now = now or utcnow()
    cutoff = now - STALE_AFTER
    closed = 0

    # Latest activity per route, so a dormant route can't close its own defects.
    route_activity = dict(
        db.execute(
            select(Event.route_id, func.max(Event.captured_at)).group_by(Event.route_id)
        ).all()
    )

    for d in db.scalars(select(Defect).where(Defect.status == "open")):
        last = _as_utc(d.last_seen)
        if last > cutoff:
            continue
        route_last = route_activity.get(d.route_id)
        if route_last is None or _as_utc(route_last) <= cutoff:
            continue  # nobody has driven this route lately; absence proves nothing
        d.status = "repaired"
        closed += 1

    if closed:
        db.commit()
    return closed


def recluster_all(db: Session) -> dict[str, int]:
    """Rebuild every defect from the raw event log.

    The events table is the source of truth, so changing RADIUS_M or the
    confirmation rule is a re-run rather than a migration.
    """
    db.query(Defect).delete()
    db.execute(Event.__table__.update().values(defect_id=None))
    db.flush()

    n = 0
    for e in db.scalars(select(Event).order_by(Event.captured_at, Event.id)):
        d = assign_defect(db, e)
        if d is not None:
            e.defect_id = d.id
            n += 1
    db.commit()
    return {
        "events_clustered": n,
        "defects": db.scalar(select(func.count(Defect.id))) or 0,
    }
