"""Clustering rules, on an in-memory database.

Auto-close gets a test rather than a demo because nothing in the prototype
footage ever gets repaired -- see V1-Plan.md, "What the footage cannot show".
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

SERVER_DIR = Path(__file__).resolve().parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app import clustering  # noqa: E402
from app.clustering import assign_defect, close_stale, haversine_m  # noqa: E402
from app.db import Base  # noqa: E402
from app.models import Defect, Event  # noqa: E402

BASE_LAT, BASE_LON = 13.0500, 80.2480
T0 = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, future=True)()
    try:
        yield session
    finally:
        session.close()


def add_event(db, *, lat=BASE_LAT, lon=BASE_LON, bus="BUS_001", kind="pothole",
              conf=0.6, when=T0, route="route-21g", uid=None):
    """Insert one sighting and cluster it, the way ingest does."""
    n = db.scalar(select(func.count(Event.id))) or 0
    e = Event(
        event_uid=uid or f"uid{n:04d}",
        bus_id=bus,
        route_id=route,
        damage_type=kind,
        severity=1.0 if kind == "pothole" else 0.75,
        confidence=conf,
        area_pct_frame=2.0,
        lat=lat,
        lon=lon,
        captured_at=when,
    )
    db.add(e)
    db.flush()
    d = assign_defect(db, e)
    if d is not None:
        e.defect_id = d.id
    db.commit()
    return e, d


def metres_north(m: float) -> float:
    return BASE_LAT + m / 111_320.0


# --------------------------------------------------------------------- basics


def test_first_sighting_creates_unconfirmed_defect(db):
    _, d = add_event(db)
    assert d is not None
    assert d.sightings == 1
    assert d.distinct_buses == 1
    assert d.status == "unconfirmed"


def test_nearby_sighting_joins_existing_defect(db):
    _, d1 = add_event(db, bus="BUS_001")
    _, d2 = add_event(db, lat=metres_north(8), bus="BUS_002", when=T0 + timedelta(hours=2))
    assert d2.id == d1.id
    assert d2.sightings == 2
    assert db.scalar(select(func.count(Defect.id))) == 1


def test_two_buses_confirm(db):
    add_event(db, bus="BUS_001")
    _, d = add_event(db, lat=metres_north(6), bus="BUS_002", when=T0 + timedelta(hours=1))
    assert d.distinct_buses == 2
    assert d.status == "open"


def test_one_bus_twice_does_not_confirm_but_three_times_does(db):
    add_event(db, bus="BUS_001", when=T0)
    _, d = add_event(db, bus="BUS_001", when=T0 + timedelta(hours=1))
    assert d.distinct_buses == 1
    assert d.status == "unconfirmed", "one unit seeing it twice is weak evidence"

    _, d = add_event(db, bus="BUS_001", when=T0 + timedelta(hours=2))
    assert d.sightings == 3
    assert d.status == "open"


def test_far_sighting_creates_separate_defect(db):
    _, d1 = add_event(db)
    _, d2 = add_event(db, lat=metres_north(60), bus="BUS_002")
    assert d2.id != d1.id
    assert db.scalar(select(func.count(Defect.id))) == 2


def test_different_damage_types_never_merge(db):
    _, d1 = add_event(db, kind="pothole")
    _, d2 = add_event(db, kind="alligator_crack", bus="BUS_002")
    assert d1.id != d2.id


def test_event_without_position_is_not_clustered(db):
    e = Event(event_uid="nofix", bus_id="BUS_001", damage_type="pothole",
              confidence=0.5, captured_at=T0)
    db.add(e)
    db.flush()
    assert assign_defect(db, e) is None
    assert db.scalar(select(func.count(Defect.id))) == 0


# ------------------------------------------------------------------- evidence


def test_defect_keeps_best_evidence_not_latest(db):
    add_event(db, bus="BUS_001", conf=0.85)
    _, d = add_event(db, bus="BUS_002", conf=0.30, when=T0 + timedelta(hours=1))
    assert d.max_confidence == pytest.approx(0.85)


def test_centroid_moves_toward_new_sightings(db):
    _, d1 = add_event(db, bus="BUS_001")
    start = d1.lat
    _, d2 = add_event(db, lat=metres_north(10), bus="BUS_002")
    assert start < d2.lat < metres_north(10), "centroid should be a running mean"


def test_first_and_last_seen_span_all_sightings(db):
    add_event(db, bus="BUS_001", when=T0 + timedelta(days=3))
    _, d = add_event(db, bus="BUS_002", when=T0)
    assert d.first_seen.replace(tzinfo=timezone.utc) == T0
    assert d.last_seen.replace(tzinfo=timezone.utc) == T0 + timedelta(days=3)


def test_linear_cracks_get_a_wider_radius_than_potholes(db):
    assert clustering.radius_for("longitudinal_crack") > clustering.radius_for("pothole")


# ---------------------------------------------------------------------- close


def test_stale_defect_on_active_route_is_closed(db):
    """The loop-closing claim: a defect that stops being reported reads as repaired."""
    old = T0
    add_event(db, bus="BUS_001", when=old)
    _, d = add_event(db, bus="BUS_002", when=old)
    assert d.status == "open"

    # The route is still driven -- another defect elsewhere on it, reported now.
    now = old + timedelta(days=30)
    add_event(db, lat=metres_north(400), bus="BUS_003", when=now)

    assert close_stale(db, now=now) == 1
    db.refresh(d)
    assert d.status == "repaired"


def test_stale_defect_on_dormant_route_is_not_closed(db):
    """Absence of evidence is only evidence when somebody is still looking."""
    old = T0
    add_event(db, bus="BUS_001", when=old)
    _, d = add_event(db, bus="BUS_002", when=old)

    now = old + timedelta(days=30)  # nobody has driven this route since
    assert close_stale(db, now=now) == 0
    db.refresh(d)
    assert d.status == "open"


def test_unconfirmed_defect_is_not_closed(db):
    """A one-off hit was never established, so it can't be 'repaired'."""
    old = T0
    _, d = add_event(db, bus="BUS_001", when=old)
    assert d.status == "unconfirmed"
    now = old + timedelta(days=30)
    add_event(db, lat=metres_north(400), bus="BUS_003", when=now)
    close_stale(db, now=now)
    db.refresh(d)
    assert d.status == "unconfirmed"


def test_repaired_defect_reopens_on_a_new_sighting(db):
    """If the pothole comes back, so does the work order."""
    old = T0
    add_event(db, bus="BUS_001", when=old)
    _, d = add_event(db, bus="BUS_002", when=old)
    now = old + timedelta(days=30)
    add_event(db, lat=metres_north(400), bus="BUS_003", when=now)
    close_stale(db, now=now)
    db.refresh(d)
    assert d.status == "repaired"

    # A repaired defect is excluded from matching, so this opens a new one --
    # deliberate: a recurrence is a fresh work order with its own first_seen.
    _, d2 = add_event(db, bus="BUS_004", when=now + timedelta(days=1))
    assert d2.id != d.id
    assert d2.status == "unconfirmed"


# ------------------------------------------------------------------- geodesy


def test_haversine_matches_known_offset():
    d = haversine_m(BASE_LAT, BASE_LON, metres_north(100), BASE_LON)
    assert d == pytest.approx(100, abs=0.5)
