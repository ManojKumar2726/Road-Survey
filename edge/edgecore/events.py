"""Track lifecycle: turn a stream of per-frame detections into discrete events.

The lab's `survey.build_report()` is a *batch* collapse -- it groups detection
rows by track once the pass is over. An onboard unit can't wait for the end of
the journey, so this module does the same collapse incrementally: it watches
each track, and the moment a track is finished it emits one `RoadEvent`.

That's the whole bandwidth argument from the problem statement. A pothole
visible for two seconds at 30 fps is ~60 detection rows; it leaves the bus as
one event plus one JPEG crop. The video never leaves the bus.

Deliberately stricter than the lab's report: `MIN_FRAMES` is 3 here vs. 2
there, because a false event costs somebody a work order rather than a row in
a table.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from . import _labpath  # noqa: F401
from labcore import taxonomy as tax
from labcore.detector import Detection

# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #

# Sightings before a track is worth reporting. Counted in *processed* frames,
# so this means the same thing at stride 1 and stride 3.
MIN_FRAMES = 3

# Processed frames a track may go unseen before it's considered finished. Too
# low and one occlusion splits a pothole into two events; too high and every
# event arrives late.
MISS_TOLERANCE = 15

# Crop geometry: pad the box a little so the defect has visible road around it.
CROP_PAD = 0.18
CROP_MAX_EDGE = 640
CROP_JPEG_QUALITY = 85


# --------------------------------------------------------------------------- #
# The event
# --------------------------------------------------------------------------- #


@dataclass
class RoadEvent:
    """One road defect, observed once, by one bus, on one pass.

    This is what crosses the wire. Everything in it is either measured from the
    track (damage, confidence, size, provenance) or attached from the metadata
    layer (bus, route, GPS, time) -- see `gps.py`.
    """

    # ---- identity
    event_uid: str
    bus_id: str = ""
    route_id: str = ""

    # ---- what was seen
    damage_type: str = tax.UNKNOWN_KEY
    severity: float = 0.0
    confidence: float = 0.0  # peak across the track
    mean_confidence: float = 0.0
    area_pct_frame: float = 0.0  # at the peak frame

    # ---- where and when (filled by the metadata layer)
    lat: float | None = None
    lon: float | None = None
    bearing: float | None = None
    speed_kmh: float | None = None
    captured_at: str = ""  # ISO-8601 UTC

    # ---- provenance, so any event can be traced back to the footage
    frame_idx: int = 0  # peak frame, where GPS and crop were sampled
    first_frame: int = 0
    last_frame: int = 0
    frames_seen: int = 0
    track_id: int = -1
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    source_clip: str = ""
    model_id: str = ""

    # ---- payload, not part of the JSON body
    crop_jpeg: bytes | None = field(default=None, repr=False)

    @property
    def damage_label(self) -> str:
        return tax.label_of(self.damage_type)

    @property
    def has_fix(self) -> bool:
        return self.lat is not None and self.lon is not None

    def to_dict(self, include_crop: bool = False) -> dict[str, Any]:
        """JSON-safe body. The crop travels separately unless asked for."""
        d = asdict(self)
        d.pop("crop_jpeg", None)
        d["damage_label"] = self.damage_label
        d["bbox"] = [round(v, 1) for v in self.bbox]
        if include_crop and self.crop_jpeg:
            import base64

            d["crop_b64"] = base64.b64encode(self.crop_jpeg).decode("ascii")
        return d

    @property
    def wire_bytes(self) -> int:
        """Roughly what this event costs to send -- the bandwidth headline."""
        import json

        return len(json.dumps(self.to_dict()).encode()) + len(self.crop_jpeg or b"")


# --------------------------------------------------------------------------- #
# Per-track accumulator
# --------------------------------------------------------------------------- #


@dataclass
class _TrackState:
    """One open track, accumulating until it closes."""

    canon: str
    track_id: int
    first_frame: int
    last_frame: int
    last_step: int
    frames_seen: int = 1
    conf_sum: float = 0.0
    peak_conf: float = 0.0
    peak_area_pct: float = 0.0
    peak_frame: int = 0
    peak_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    peak_crop: bytes | None = None

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / self.frames_seen if self.frames_seen else 0.0


def _encode_crop(frame: np.ndarray, box: Sequence[float]) -> bytes | None:
    """Cut the defect out of the frame and JPEG it.

    Padded so there's road around the defect -- a tight crop of a pothole is
    surprisingly hard for a human to judge.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    pw, ph = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(w, int(x2 + pw))
    y2 = min(h, int(y2 + ph))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    crop = frame[y1:y2, x1:x2]
    long_edge = max(crop.shape[:2])
    if long_edge > CROP_MAX_EDGE:
        s = CROP_MAX_EDGE / long_edge
        crop = cv2.resize(
            crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(
        ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), CROP_JPEG_QUALITY]
    )
    return buf.tobytes() if ok else None


# --------------------------------------------------------------------------- #
# The tracker
# --------------------------------------------------------------------------- #


class EventTracker:
    """Watches tracks across a pass and emits an event as each one finishes.

    Feed it one call per processed frame. It returns the events that closed on
    that frame -- usually none, occasionally one or two.
    """

    def __init__(
        self,
        bus_id: str = "BUS_000",
        route_id: str = "",
        source_clip: str = "",
        model_id: str = "",
        min_frames: int = MIN_FRAMES,
        miss_tolerance: int = MISS_TOLERANCE,
        capture_crops: bool = True,
    ) -> None:
        self.bus_id = bus_id
        self.route_id = route_id
        self.source_clip = source_clip
        self.model_id = model_id
        self.min_frames = int(min_frames)
        self.miss_tolerance = int(miss_tolerance)
        self.capture_crops = capture_crops

        self._open: dict[tuple[str, int], _TrackState] = {}
        self._step = 0  # processed-frame counter; stride-invariant

        # ---- honesty counters, mirroring the lab's `unassigned_boxes`
        self.boxes_seen = 0
        self.boxes_unassigned = 0  # tracker never gave these an ID
        self.tracks_opened = 0
        self.tracks_dropped = 0  # closed below min_frames -- flicker
        self.events_emitted = 0

    # ------------------------------------------------------------------ feed

    def update(
        self,
        frame_idx: int,
        detections: Iterable[Detection],
        frame: np.ndarray | None = None,
    ) -> list[RoadEvent]:
        """Absorb one frame. Returns events for tracks that closed here."""
        self._step += 1
        h, w = (frame.shape[:2] if frame is not None else (0, 0))

        touched: set[tuple[str, int]] = set()
        for d in detections:
            self.boxes_seen += 1
            if d.track_id is None:
                # The tracker emits boxes before it commits them to a track.
                # They're real detections but can't be collapsed, and promoting
                # each to its own event would badly inflate the count.
                self.boxes_unassigned += 1
                continue

            key = (d.canon, int(d.track_id))
            touched.add(key)
            area_pct = d.area_pct(w, h) if w and h else 0.0

            st = self._open.get(key)
            if st is None:
                st = _TrackState(
                    canon=d.canon,
                    track_id=int(d.track_id),
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    last_step=self._step,
                    peak_frame=frame_idx,
                )
                self._open[key] = st
                self.tracks_opened += 1
            else:
                st.frames_seen += 1
                st.last_frame = frame_idx
                st.last_step = self._step

            st.conf_sum += d.conf
            st.peak_conf = max(st.peak_conf, d.conf)

            # The largest the box ever gets is the closest the bus ever came to
            # the defect -- so that frame gives both the best crop and the most
            # accurate position estimate.
            if area_pct >= st.peak_area_pct:
                st.peak_area_pct = area_pct
                st.peak_frame = frame_idx
                st.peak_bbox = d.xyxy
                if self.capture_crops and frame is not None:
                    crop = _encode_crop(frame, d.xyxy)
                    if crop:
                        st.peak_crop = crop

        # ---- close tracks that have been missing long enough
        closed: list[RoadEvent] = []
        for key, st in list(self._open.items()):
            if key in touched:
                continue
            if self._step - st.last_step >= self.miss_tolerance:
                ev = self._close(key)
                if ev is not None:
                    closed.append(ev)
        return closed

    def flush(self) -> list[RoadEvent]:
        """Close everything still open. Call once at end of clip."""
        out = []
        for key in list(self._open):
            ev = self._close(key)
            if ev is not None:
                out.append(ev)
        return out

    # ----------------------------------------------------------------- close

    def _close(self, key: tuple[str, int]) -> RoadEvent | None:
        st = self._open.pop(key, None)
        if st is None:
            return None
        if st.frames_seen < self.min_frames:
            self.tracks_dropped += 1
            return None

        self.events_emitted += 1
        return RoadEvent(
            event_uid=uuid.uuid4().hex,
            bus_id=self.bus_id,
            route_id=self.route_id,
            damage_type=st.canon,
            severity=tax.severity_of(st.canon),
            confidence=round(st.peak_conf, 4),
            mean_confidence=round(st.mean_conf, 4),
            area_pct_frame=round(st.peak_area_pct, 4),
            frame_idx=st.peak_frame,
            first_frame=st.first_frame,
            last_frame=st.last_frame,
            frames_seen=st.frames_seen,
            track_id=st.track_id,
            bbox=st.peak_bbox,
            source_clip=self.source_clip,
            model_id=self.model_id,
            crop_jpeg=st.peak_crop,
        )

    # ----------------------------------------------------------------- report

    @property
    def open_tracks(self) -> int:
        return len(self._open)

    def stats(self) -> dict[str, Any]:
        """Counters for the run summary -- including what was thrown away."""
        return {
            "steps": self._step,
            "boxes_seen": self.boxes_seen,
            "boxes_unassigned": self.boxes_unassigned,
            "tracks_opened": self.tracks_opened,
            "tracks_dropped_as_flicker": self.tracks_dropped,
            "events_emitted": self.events_emitted,
            "still_open": self.open_tracks,
        }


def stamp_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
