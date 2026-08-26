"""Maintenance operations on the clustering layer.

The events table is the source of truth, so retuning the clustering rule is a
re-run rather than a migration.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import clustering
from ..db import get_db
from ..reset import reset_data

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/reset")
def reset(db: Session = Depends(get_db), keep_crops: bool = False):
    """Clear every sighting, defect and bus. Routes are configuration and stay.

    Lets you start a clean pass without restarting the server.
    """
    return reset_data(db, drop_crops=not keep_crops)


@router.post("/recluster")
def recluster(db: Session = Depends(get_db)):
    """Rebuild every defect from the raw event log."""
    return clustering.recluster_all(db)


@router.post("/close-stale")
def close_stale(db: Session = Depends(get_db)):
    """Mark long-unreported defects on still-active routes as repaired."""
    return {"closed": clustering.close_stale(db)}


@router.get("/clustering")
def settings():
    """What the clustering layer is currently tuned to."""
    return {
        "radius_m": clustering.RADIUS_M,
        "radius_scale": clustering.RADIUS_SCALE,
        "confirm_buses": clustering.CONFIRM_BUSES,
        "confirm_sightings": clustering.CONFIRM_SIGHTINGS,
        "stale_after_days": clustering.STALE_AFTER.days,
    }
