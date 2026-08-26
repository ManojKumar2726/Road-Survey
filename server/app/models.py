"""Tables.

Two of them carry the interesting design decision:

`events` is the raw log -- one row per sighting, per bus, per pass. Never
merged, never deleted. It's the evidence trail: which bus saw what, when, at
what confidence, with the crop that proves it.

`defects` is the physical thing in the road. Many events collapse into one
defect, which is what the map plots. Without it, a route driven four times a
day stacks forty pins on one pothole and the count on the dashboard is a count
of *sightings*, not of problems.

`defect_id` is nullable so ingest works before clustering exists (phase 3
stores raw events; phase 6 fills this in).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Fleet
# --------------------------------------------------------------------------- #


class Route(Base):
    __tablename__ = "routes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    city: Mapped[str] = mapped_column(String(100), default="")
    speed_kmh: Mapped[float] = mapped_column(Float, default=25.0)
    length_m: Mapped[float] = mapped_column(Float, default=0.0)
    # The polyline, stored as a GeoJSON string. The dashboard draws it; nothing
    # server-side queries into it, so a text column is honest and portable.
    geojson: Mapped[str] = mapped_column(Text, default="")


class Bus(Base):
    __tablename__ = "buses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(String(200), default="")
    route_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("routes.id"), nullable=True
    )
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    events_reported: Mapped[int] = mapped_column(Integer, default=0)


# --------------------------------------------------------------------------- #
# Damage
# --------------------------------------------------------------------------- #


class Defect(Base):
    """One physical defect in the road, confirmed by one or more sightings."""

    __tablename__ = "defects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    damage_type: Mapped[str] = mapped_column(String(40), index=True)
    severity: Mapped[float] = mapped_column(Float, default=0.0)

    # Running centroid of member events.
    lat: Mapped[float] = mapped_column(Float, index=True)
    lon: Mapped[float] = mapped_column(Float, index=True)
    route_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    sightings: Mapped[int] = mapped_column(Integer, default=0)
    distinct_buses: Mapped[int] = mapped_column(Integer, default=0)

    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    max_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    peak_area_pct: Mapped[float] = mapped_column(Float, default=0.0)
    best_crop: Mapped[str | None] = mapped_column(String(300), nullable=True)

    # unconfirmed -> seen once, could be a false positive
    # open        -> corroborated, treat as real
    # repaired    -> stopped being reported on a route that is still driven
    status: Mapped[str] = mapped_column(String(20), default="unconfirmed", index=True)

    events: Mapped[list["Event"]] = relationship(back_populates="defect")

    __table_args__ = (Index("ix_defects_bbox", "lat", "lon"),)


class Event(Base):
    """One sighting. The raw log -- never merged, never deleted."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # The edge generates this. Ingest is idempotent on it, so a publisher retry
    # after a timeout can't double-insert.
    event_uid: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    bus_id: Mapped[str] = mapped_column(String(64), index=True)
    route_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    damage_type: Mapped[str] = mapped_column(String(40), index=True)
    severity: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    mean_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    area_pct_frame: Mapped[float] = mapped_column(Float, default=0.0)

    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    bearing: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)

    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # ---- provenance: any row traces back to a frame of footage
    frame_idx: Mapped[int] = mapped_column(Integer, default=0)
    first_frame: Mapped[int] = mapped_column(Integer, default=0)
    last_frame: Mapped[int] = mapped_column(Integer, default=0)
    frames_seen: Mapped[int] = mapped_column(Integer, default=0)
    track_id: Mapped[int] = mapped_column(Integer, default=-1)
    source_clip: Mapped[str] = mapped_column(String(200), default="")
    model_id: Mapped[str] = mapped_column(String(80), default="")
    crop_path: Mapped[str | None] = mapped_column(String(300), nullable=True)

    defect_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("defects.id"), nullable=True, index=True
    )
    defect: Mapped[Defect | None] = relationship(back_populates="events")
