"""Request and response bodies.

`EventIn` mirrors `edgecore.events.RoadEvent.to_dict()`. Fields the edge may not
have (GPS on a unit with no route configured) are optional, so a partial event
is stored rather than rejected -- a sighting with no fix is still evidence.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


class EventIn(BaseModel):
    event_uid: str
    bus_id: str = ""
    route_id: str | None = None

    damage_type: str = "unknown"
    severity: float = 0.0
    confidence: float = 0.0
    mean_confidence: float = 0.0
    area_pct_frame: float = 0.0

    lat: float | None = None
    lon: float | None = None
    bearing: float | None = None
    speed_kmh: float | None = None
    captured_at: datetime | None = None

    frame_idx: int = 0
    first_frame: int = 0
    last_frame: int = 0
    frames_seen: int = 0
    track_id: int = -1
    source_clip: str = ""
    model_id: str = ""

    # Crop travels inline as base64 -- one round trip per batch beats a
    # multipart upload per event over a flaky link.
    crop_b64: str | None = None

    # Ignored on input, present so an events.json round-trips without a fuss.
    damage_label: str | None = None
    bbox: list[float] | None = None


class EventBatch(BaseModel):
    events: list[EventIn] = Field(default_factory=list)


class IngestResult(BaseModel):
    accepted: int
    duplicates: int
    rejected: int = 0
    event_ids: list[int] = Field(default_factory=list)
    defect_ids: list[int] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_uid: str
    bus_id: str
    route_id: str | None
    damage_type: str
    severity: float
    confidence: float
    area_pct_frame: float
    lat: float | None
    lon: float | None
    bearing: float | None
    speed_kmh: float | None
    captured_at: datetime | None
    received_at: datetime | None
    frames_seen: int
    frame_idx: int
    source_clip: str
    model_id: str
    crop_path: str | None
    defect_id: int | None


class DefectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    damage_type: str
    severity: float
    lat: float
    lon: float
    route_id: str | None
    sightings: int
    distinct_buses: int
    first_seen: datetime | None
    last_seen: datetime | None
    max_confidence: float
    peak_area_pct: float
    best_crop: str | None
    status: str


class RouteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    city: str
    speed_kmh: float
    length_m: float


class BusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    label: str
    route_id: str | None
    events_reported: int
    last_seen: datetime | None


class Stats(BaseModel):
    events: int
    defects: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    buses_reporting: int
    routes: int
    latest_event_id: int = 0
    damage_score: float = 0.0
