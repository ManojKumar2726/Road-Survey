"""Simulated GPS: turn a frame index into a position on a real road.

The prototype has dashcam footage but no telemetry, so position is synthesised
by walking a bus along a traced route polyline at a nominal speed. Frame index
maps to distance travelled, distance maps to a point on the line.

Why a real polyline rather than jittered points around a centre: events have to
land *on roads* or the map, the heatmap and any route-level aggregation are
visibly fiction. Interpolating a real corridor costs nothing extra and the
result survives someone zooming in.

`RouteReplay` and `TrackReplay` share one interface, so swapping simulated
position for a real GPX/CSV track later is a config change, not a rewrite.

Honesty note for the report: simulated fixes are exact. Real GPS drifts 5-10 m
in an urban canyon, which is what forces the server's clustering radius to be
generous -- see `--gps-noise`.
"""

from __future__ import annotations

import csv
import json
import math
import random
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

EARTH_R = 6_371_000.0  # metres

ROUTES_DIR = Path(__file__).resolve().parent.parent / "routes"


# --------------------------------------------------------------------------- #
# Geodesy -- small-angle helpers, adequate at city scale
# --------------------------------------------------------------------------- #


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def offset_m(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Shift a fix by a metre offset. Flat-earth, fine over a few tens of metres."""
    dlat = north_m / 111_320.0
    dlon = east_m / (111_320.0 * max(0.01, math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


# --------------------------------------------------------------------------- #
# A fix
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Fix:
    lat: float
    lon: float
    bearing: float
    speed_kmh: float
    timestamp: str  # ISO-8601 UTC
    distance_m: float = 0.0  # travelled along the route

    def as_dict(self) -> dict[str, Any]:
        return {
            "lat": round(self.lat, 7),
            "lon": round(self.lon, 7),
            "bearing": round(self.bearing, 1),
            "speed_kmh": round(self.speed_kmh, 1),
            "timestamp": self.timestamp,
        }


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #


@dataclass
class Route:
    """A traced road corridor, indexed by cumulative distance."""

    route_id: str
    name: str
    coords: list[tuple[float, float]]  # (lat, lon), in travel order
    speed_kmh: float = 25.0
    city: str = ""

    def __post_init__(self) -> None:
        self.cum: list[float] = [0.0]
        for i in range(1, len(self.coords)):
            a, b = self.coords[i - 1], self.coords[i]
            self.cum.append(self.cum[-1] + haversine_m(a[0], a[1], b[0], b[1]))

    @property
    def length_m(self) -> float:
        return self.cum[-1] if self.cum else 0.0

    def at(self, distance_m: float) -> tuple[float, float, float]:
        """(lat, lon, bearing) at a distance along the route.

        Past either end the bus stays put at the terminus rather than
        extrapolating off the road.
        """
        if len(self.coords) < 2:
            lat, lon = self.coords[0]
            return lat, lon, 0.0

        d = min(max(0.0, distance_m), self.length_m)
        i = max(1, min(bisect_right(self.cum, d), len(self.coords) - 1))
        a, b = self.coords[i - 1], self.coords[i]
        seg = self.cum[i] - self.cum[i - 1]
        t = ((d - self.cum[i - 1]) / seg) if seg > 0 else 0.0
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            bearing_deg(a[0], a[1], b[0], b[1]),
        )

    def as_geojson(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {
                "route_id": self.route_id,
                "name": self.name,
                "speed_kmh": self.speed_kmh,
                "city": self.city,
                "length_m": round(self.length_m, 1),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lat, lon in self.coords],
            },
        }

    # ------------------------------------------------------------------ load

    @classmethod
    def from_geojson(cls, path: str | Path) -> "Route":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        feat = raw["features"][0] if raw.get("type") == "FeatureCollection" else raw
        props = feat.get("properties", {})
        coords = [(lat, lon) for lon, lat in feat["geometry"]["coordinates"]]
        if len(coords) < 2:
            raise ValueError(f"{path}: route needs at least two coordinates")
        return cls(
            route_id=props.get("route_id") or Path(path).stem,
            name=props.get("name") or Path(path).stem,
            coords=coords,
            speed_kmh=float(props.get("speed_kmh") or 25.0),
            city=props.get("city", ""),
        )


def load_routes(folder: str | Path = ROUTES_DIR) -> dict[str, Route]:
    """Every route in `edge/routes/`, keyed by route_id."""
    out: dict[str, Route] = {}
    p = Path(folder)
    if not p.is_dir():
        return out
    for f in sorted(p.glob("*.geojson")):
        try:
            r = Route.from_geojson(f)
        except Exception as exc:  # a malformed route shouldn't kill the run
            print(f"  warning: skipping {f.name}: {exc}")
            continue
        out[r.route_id] = r
    return out


def get_route(route_id: str, folder: str | Path = ROUTES_DIR) -> Route:
    routes = load_routes(folder)
    if route_id not in routes:
        known = ", ".join(sorted(routes)) or "(none found)"
        raise KeyError(f"Unknown route '{route_id}'. Available: {known}")
    return routes[route_id]


# --------------------------------------------------------------------------- #
# Replays
# --------------------------------------------------------------------------- #


class RouteReplay:
    """Walk a route at a nominal speed, mapping frame index to a fix.

    `bind()` is called by the pipeline once the clip's fps and stride are
    known, so frame index converts to elapsed seconds correctly.
    """

    def __init__(
        self,
        route: Route,
        speed_kmh: float | None = None,
        start_offset_m: float = 0.0,
        start_time: datetime | None = None,
        gps_noise_m: float = 0.0,
        speed_jitter: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.route = route
        self.base_speed = float(speed_kmh or route.speed_kmh)
        self.start_offset_m = float(start_offset_m)
        self.start_time = start_time or datetime.now(timezone.utc)
        self.gps_noise_m = float(gps_noise_m)
        self._rng = random.Random(seed)

        # One jitter draw per pass, not per fix: a bus drives a stretch at
        # roughly one speed. Per-fix jitter would model a juddering bus.
        if speed_jitter:
            self.speed_kmh = self.base_speed * (
                1.0 + self._rng.uniform(-speed_jitter, speed_jitter)
            )
        else:
            self.speed_kmh = self.base_speed

        self.fps = 30.0
        self.stride = 1
        self.phase = 0

    # ------------------------------------------------------------------ bind

    def bind(self, fps: float = 30.0, stride: int = 1, phase: int = 0) -> None:
        self.fps = float(fps) or 30.0
        self.stride = max(1, int(stride))
        self.phase = int(phase)

    # ------------------------------------------------------------------- fix

    def fix_for(self, frame_idx: int) -> Fix | None:
        if frame_idx < 0:
            frame_idx = 0
        elapsed_s = frame_idx / self.fps
        dist = self.start_offset_m + (self.speed_kmh / 3.6) * elapsed_s
        lat, lon, brg = self.route.at(dist)

        if self.gps_noise_m:
            # Gaussian in both axes -- a rough stand-in for urban multipath.
            lat, lon = offset_m(
                lat,
                lon,
                self._rng.gauss(0.0, self.gps_noise_m),
                self._rng.gauss(0.0, self.gps_noise_m),
            )

        return Fix(
            lat=lat,
            lon=lon,
            bearing=brg,
            speed_kmh=self.speed_kmh,
            timestamp=(
                self.start_time + timedelta(seconds=elapsed_s)
            ).isoformat(timespec="seconds"),
            distance_m=dist,
        )

    # ------------------------------------------------------------------ meta

    def describe(self) -> dict[str, Any]:
        return {
            "route_id": self.route.route_id,
            "route_name": self.route.name,
            "route_length_m": round(self.route.length_m, 1),
            "speed_kmh": round(self.speed_kmh, 1),
            "start_offset_m": round(self.start_offset_m, 1),
            "gps_noise_m": self.gps_noise_m,
            "start_time": self.start_time.isoformat(timespec="seconds"),
        }

    def span_m(self, frames: int) -> float:
        """How much road a clip of this many frames actually covers."""
        return (self.speed_kmh / 3.6) * (frames / self.fps)


class TrackReplay:
    """Replay real fixes from a GPX or CSV sidecar.

    Same interface as `RouteReplay`, so nothing downstream changes when real
    telemetry turns up. CSV wants columns lat, lon and one of time/timestamp;
    GPX wants ordinary <trkpt lat= lon=> elements.
    """

    def __init__(self, path: str | Path, start_time: datetime | None = None) -> None:
        self.path = Path(path)
        self.points: list[tuple[float, float, float | None]] = []  # lat, lon, t_s
        self._load()
        if not self.points:
            raise ValueError(f"No track points found in {path}")
        self.start_time = start_time or datetime.now(timezone.utc)
        self.fps = 30.0
        self.stride = 1
        self.phase = 0

    def _load(self) -> None:
        suffix = self.path.suffix.lower()
        if suffix == ".gpx":
            import xml.etree.ElementTree as ET

            root = ET.parse(self.path).getroot()
            # Namespace-agnostic: GPX files vary and the tag is what matters.
            for pt in root.iter():
                if not pt.tag.endswith("trkpt"):
                    continue
                lat, lon = float(pt.attrib["lat"]), float(pt.attrib["lon"])
                self.points.append((lat, lon, None))
        else:
            with open(self.path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    keys = {k.lower(): k for k in row}
                    if "lat" not in keys or "lon" not in keys:
                        continue
                    self.points.append(
                        (float(row[keys["lat"]]), float(row[keys["lon"]]), None)
                    )

    def bind(self, fps: float = 30.0, stride: int = 1, phase: int = 0) -> None:
        self.fps = float(fps) or 30.0
        self.stride = max(1, int(stride))
        self.phase = int(phase)

    def fix_for(self, frame_idx: int) -> Fix | None:
        if not self.points:
            return None
        # Spread the track evenly across the clip; without per-point times
        # that's the only defensible mapping.
        t = max(0.0, frame_idx) / max(1, self.fps)
        total_s = len(self.points) / max(1.0, self.fps)
        i = min(len(self.points) - 1, int((t / total_s) * (len(self.points) - 1)))
        lat, lon, _ = self.points[i]
        j = min(len(self.points) - 1, i + 1)
        brg = bearing_deg(lat, lon, self.points[j][0], self.points[j][1])
        return Fix(
            lat=lat,
            lon=lon,
            bearing=brg,
            speed_kmh=0.0,
            timestamp=(self.start_time + timedelta(seconds=t)).isoformat(
                timespec="seconds"
            ),
        )

    def describe(self) -> dict[str, Any]:
        return {"track": str(self.path), "points": len(self.points)}


# --------------------------------------------------------------------------- #
# Relative time parsing, for --at
# --------------------------------------------------------------------------- #


def parse_when(raw: str | None) -> datetime:
    """'-2h', '-1d', '-30m', an ISO timestamp, or now.

    Backdating is how the prototype fakes fleet history: the same clip replayed
    as an earlier pass by another bus.
    """
    now = datetime.now(timezone.utc)
    if not raw:
        return now
    s = raw.strip()
    if s and s[0] in "+-" and s[-1] in "mhd":
        try:
            n = float(s[1:-1])
        except ValueError:
            raise ValueError(f"--at: could not parse '{raw}'")
        unit = {"m": "minutes", "h": "hours", "d": "days"}[s[-1]]
        delta = timedelta(**{unit: n})
        return now - delta if s[0] == "-" else now + delta
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(
            f"--at: expected a relative offset like '-2h' or an ISO timestamp, got '{raw}'"
        )
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
