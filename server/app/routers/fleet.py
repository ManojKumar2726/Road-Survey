"""Routes, buses, taxonomy, and the KPI tiles."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import taxonomy as tax
from ..db import get_db
from ..models import Bus, Defect, Event, Route
from ..schemas import BusOut, RouteOut, Stats

router = APIRouter(prefix="/api", tags=["fleet"])


@router.get("/routes", response_model=list[RouteOut])
def list_routes(db: Session = Depends(get_db)):
    return list(db.scalars(select(Route).order_by(Route.id)))


@router.get("/routes.geojson")
def routes_geojson(db: Session = Depends(get_db)):
    """Route polylines for the map underlay."""
    feats = []
    for r in db.scalars(select(Route)):
        if not r.geojson:
            continue
        try:
            feats.append(json.loads(r.geojson))
        except json.JSONDecodeError:
            continue
    return {"type": "FeatureCollection", "features": feats}


@router.get("/buses", response_model=list[BusOut])
def list_buses(db: Session = Depends(get_db)):
    return list(db.scalars(select(Bus).order_by(Bus.id)))


@router.get("/taxonomy")
def taxonomy():
    """Damage types, colours and severity weights.

    The dashboard styles itself from this rather than hardcoding hex, so a
    pothole stays the same red as in the onboard video overlay.
    """
    return tax.as_payload()


@router.get("/stats", response_model=Stats)
def stats(db: Session = Depends(get_db)):
    by_type = dict(
        db.execute(
            select(Event.damage_type, func.count(Event.id)).group_by(Event.damage_type)
        ).all()
    )
    by_status = dict(
        db.execute(
            select(Defect.status, func.count(Defect.id)).group_by(Defect.status)
        ).all()
    )
    n_defects = db.scalar(select(func.count(Defect.id))) or 0
    n_events = db.scalar(select(func.count(Event.id))) or 0

    # Severity-weighted total. Falls back to events while clustering is a stub,
    # so the tile means something from phase 3 rather than reading zero.
    if n_defects:
        score = db.scalar(select(func.sum(Defect.severity * Defect.max_confidence))) or 0.0
    else:
        score = db.scalar(select(func.sum(Event.severity * Event.confidence))) or 0.0

    return Stats(
        events=n_events,
        defects=n_defects,
        by_type={k: v for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])},
        by_status=by_status,
        buses_reporting=db.scalar(select(func.count(Bus.id))) or 0,
        routes=db.scalar(select(func.count(Route.id))) or 0,
        latest_event_id=db.scalar(select(func.max(Event.id))) or 0,
        damage_score=round(float(score), 2),
    )
