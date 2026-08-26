"""Clustered defects -- what the map plots once phase 6 is in."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Defect
from ..schemas import DefectOut

router = APIRouter(prefix="/api", tags=["defects"])


@router.get("/defects", response_model=list[DefectOut])
def list_defects(
    db: Session = Depends(get_db),
    bbox: str | None = Query(
        None, description="min_lon,min_lat,max_lon,max_lat -- viewport filter"
    ),
    damage_type: str | None = None,
    status: str | None = Query(None, description="unconfirmed | open | repaired"),
    route_id: str | None = None,
    min_sightings: int = 0,
    limit: int = Query(2000, le=10000),
):
    q = select(Defect)
    if bbox:
        try:
            min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox.split(","))
        except ValueError:
            raise HTTPException(400, "bbox must be min_lon,min_lat,max_lon,max_lat")
        q = q.where(
            Defect.lon >= min_lon,
            Defect.lon <= max_lon,
            Defect.lat >= min_lat,
            Defect.lat <= max_lat,
        )
    if damage_type:
        q = q.where(Defect.damage_type == damage_type)
    if status:
        q = q.where(Defect.status == status)
    if route_id:
        q = q.where(Defect.route_id == route_id)
    if min_sightings:
        q = q.where(Defect.sightings >= min_sightings)
    # Worst first, so a truncated result keeps the defects that matter.
    q = q.order_by(Defect.severity.desc(), Defect.sightings.desc()).limit(limit)
    return list(db.scalars(q))


@router.get("/defects/{defect_id}", response_model=DefectOut)
def get_defect(defect_id: int, db: Session = Depends(get_db)):
    row = db.get(Defect, defect_id)
    if row is None:
        raise HTTPException(404, f"No defect {defect_id}")
    return row
