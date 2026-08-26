"""Event -> defect assignment.

Phase 3 stores raw events only; this is the seam where phase 6 collapses repeat
sightings of the same pothole into one defect. Returning None leaves
`Event.defect_id` null, which every read path already tolerates.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from .models import Defect, Event


def assign_defect(db: Session, event: Event) -> Defect | None:
    """Find or create the defect this sighting belongs to.

    Not implemented until phase 6 -- see V1-Plan.md.
    """
    return None
