"""Centralized system -- the API and the dashboard it serves.

    uvicorn app.main:app --reload --port 8000

Window 2 of the demo. Buses post events here; the dashboard reads them back.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db as dbmod
from .db import Base, SessionLocal, engine, init_dirs
from .models import Route
from .routers import admin, defects, events, fleet, reports

SERVER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVER_DIR.parent
STATIC_DIR = SERVER_DIR / "static"
ROUTES_DIR = REPO_ROOT / "edge" / "routes"


def seed_routes() -> int:
    """Load `edge/routes/*.geojson` into the routes table.

    Idempotent -- it refreshes geometry on restart so an edited polyline shows
    up without a migration.
    """
    if not ROUTES_DIR.is_dir():
        return 0

    # The lab-side loader already parses and measures these; reuse it rather
    # than writing a second GeoJSON reader that can disagree with the first.
    import sys

    edge_dir = REPO_ROOT / "edge"
    if str(edge_dir) not in sys.path:
        sys.path.append(str(edge_dir))
    from edgecore.gps import load_routes  # noqa: E402

    n = 0
    with SessionLocal() as session:
        for rid, r in load_routes(ROUTES_DIR).items():
            row = session.get(Route, rid)
            if row is None:
                row = Route(id=rid)
                session.add(row)
            row.name = r.name
            row.city = r.city
            row.speed_kmh = r.speed_kmh
            row.length_m = r.length_m
            row.geojson = json.dumps(r.as_geojson())
            n += 1
        session.commit()
    return n


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dirs()
    Base.metadata.create_all(engine)
    n = seed_routes()
    print(f"  database : {dbmod.describe()}")
    print(f"  routes   : {n} seeded from {ROUTES_DIR}")
    print(f"  media    : {dbmod.MEDIA_DIR}")
    yield


app = FastAPI(
    title="Road Survey — centralized system",
    version="1.0",
    description=(
        "Aggregates road-damage events from the bus fleet, clusters repeat "
        "sightings into defects, and serves the GIS dashboard."
    ),
    lifespan=lifespan,
)

# The onboard unit is a separate origin in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events.router)
app.include_router(defects.router)
app.include_router(fleet.router)
app.include_router(reports.router)
app.include_router(admin.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "database": dbmod.describe()}


dbmod.init_dirs()
app.mount("/media", StaticFiles(directory=dbmod.MEDIA_DIR), name="media")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard():
        index = STATIC_DIR / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {"status": "ok", "note": "dashboard not built yet — see /docs"}
