"""The onboard run loop, with no UI attached.

`app_edge.py` (Streamlit) and `run_edge.py` (headless) are both thin front-ends
over this -- the same split that lets the lab's Streamlit app and `run_cli.py`
share `labcore`.

    for upd in Pipeline(cfg).run("clip.mp4"):
        show(upd.annotated)
        for ev in upd.new_events:
            publish(ev)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import _labpath  # noqa: F401
from labcore import taxonomy as tax
from labcore.detector import Detection, Detector, device_label
from labcore.draw import Annotator, DrawOptions
from labcore.registry import load_registry
from labcore.video import VideoSource, probe_video

from .events import EventTracker, RoadEvent, stamp_now, MIN_FRAMES, MISS_TOLERANCE


@dataclass
class PipelineConfig:
    """Everything one pass needs to know."""

    # ---- model
    model_id: str = "rdd-yolo12s"
    conf: float | None = None  # None -> the model's own default_conf
    iou: float | None = None
    imgsz: int | None = None
    device: str = "auto"
    half: bool = False
    track: bool = True
    tracker: str = "bytetrack.yaml"
    only: list[str] | None = None  # canonical damage keys to keep

    # ---- sampling
    stride: int = 1
    phase: int = 0  # start-frame offset; see run_edge --phase
    width: int = 0  # downscale to this width, 0 = native
    max_frames: int = 0

    # ---- identity
    bus_id: str = "BUS_001"
    route_id: str = ""

    # ---- lifecycle
    min_frames: int = MIN_FRAMES
    miss_tolerance: int = MISS_TOLERANCE
    capture_crops: bool = True

    # ---- output
    annotate: bool = True


@dataclass
class FrameUpdate:
    """One processed frame's worth of everything a front-end might want."""

    frame_idx: int
    step: int
    total_steps: int
    detections: list[Detection] = field(default_factory=list)
    new_events: list[RoadEvent] = field(default_factory=list)
    annotated: np.ndarray | None = None
    infer_ms: float = 0.0
    fps: float = 0.0

    @property
    def progress(self) -> float:
        return (self.step / self.total_steps) if self.total_steps else 0.0


class Pipeline:
    """Clip in, annotated frames and road events out."""

    def __init__(self, cfg: PipelineConfig, gps: Any | None = None) -> None:
        self.cfg = cfg
        self.gps = gps

        spec = load_registry().get(cfg.model_id)
        self.spec = spec
        self.detector = Detector(spec, device=cfg.device, half=cfg.half)

        self.conf = cfg.conf if cfg.conf is not None else spec.default_conf
        self.iou = cfg.iou if cfg.iou is not None else spec.default_iou
        self.imgsz = cfg.imgsz or spec.default_imgsz
        self.class_ids = (
            self.detector.ids_for_canon(cfg.only) if cfg.only else None
        )

        self.tracker_: EventTracker | None = None
        self._annotator = Annotator(DrawOptions())
        self._ms: list[float] = []

    # ------------------------------------------------------------------ meta

    @property
    def device(self) -> str:
        return self.detector.device

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.spec.id,
            "model_name": self.spec.name,
            "device": device_label(self.detector.device),
            "conf": self.conf,
            "imgsz": self.imgsz,
            "damage_types": self.detector.canon_keys,
        }

    # ------------------------------------------------------------------- run

    def run(self, source: str | int | Path) -> Iterator[FrameUpdate]:
        cfg = self.cfg
        clip = Path(str(source)).name if not isinstance(source, int) else f"cam{source}"

        resize = None
        if cfg.width and not isinstance(source, int):
            info = probe_video(str(source))
            if info.width > cfg.width:
                s = cfg.width / info.width
                # Keep it even -- some encoders reject odd dimensions.
                resize = (cfg.width, int(round(info.height * s / 2) * 2))

        self.detector.reset_tracker()
        self.tracker_ = EventTracker(
            bus_id=cfg.bus_id,
            route_id=cfg.route_id,
            source_clip=clip,
            model_id=self.spec.id,
            min_frames=cfg.min_frames,
            miss_tolerance=cfg.miss_tolerance,
            capture_crops=cfg.capture_crops,
        )
        self._ms = []

        src = VideoSource(
            source,
            stride=cfg.stride,
            start_frame=cfg.phase,
            max_frames=cfg.max_frames or None,
            resize_to=resize,
        )
        with src:
            total = src.planned_frames
            fps_in = (src.info.fps if src.info else 30.0) or 30.0
            if self.gps is not None and hasattr(self.gps, "bind"):
                self.gps.bind(fps=fps_in, stride=cfg.stride, phase=cfg.phase)

            step = 0
            for frame_idx, frame in src:
                step += 1
                res = self.detector.infer(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    track=cfg.track,
                    tracker=cfg.tracker,
                    classes=self.class_ids,
                )
                self._ms.append(res.infer_ms)

                new_events = self.tracker_.update(frame_idx, res.detections, frame)
                for ev in new_events:
                    self._stamp(ev)

                annotated = None
                if cfg.annotate:
                    annotated = self._annotator.draw(
                        frame,
                        res.detections,
                        hud=self._hud(frame_idx, step, total, res.infer_ms),
                    )

                yield FrameUpdate(
                    frame_idx=frame_idx,
                    step=step,
                    total_steps=total,
                    detections=res.detections,
                    new_events=new_events,
                    annotated=annotated,
                    infer_ms=res.infer_ms,
                    fps=self.fps,
                )

            # Tracks still open at the end of the clip would otherwise be lost.
            tail = self.tracker_.flush()
            for ev in tail:
                self._stamp(ev)
            if tail:
                yield FrameUpdate(
                    frame_idx=-1,
                    step=step,
                    total_steps=total,
                    new_events=tail,
                    fps=self.fps,
                )

    # -------------------------------------------------------------- helpers

    def _stamp(self, ev: RoadEvent) -> None:
        """Attach position and time, sampled at the event's peak frame."""
        if self.gps is not None:
            fix = self.gps.fix_for(ev.frame_idx)
            if fix is not None:
                ev.lat, ev.lon = fix.lat, fix.lon
                ev.bearing, ev.speed_kmh = fix.bearing, fix.speed_kmh
                ev.captured_at = fix.timestamp
        if not ev.captured_at:
            ev.captured_at = stamp_now()

    def _hud(self, frame_idx: int, step: int, total: int, ms: float) -> dict[str, str]:
        t = self.tracker_
        return {
            "title": f"{self.cfg.bus_id} · {self.spec.name}",
            "device": device_label(self.detector.device, short=True),
            "frame": f"{step}" + (f" / {total}" if total else ""),
            "events": str(t.events_emitted if t else 0),
            "open": str(t.open_tracks if t else 0),
            "infer": f"{ms:.1f} ms",
            "fps": f"{self.fps:.1f}",
            "conf": f"{self.conf:.2f}",
        }

    @property
    def fps(self) -> float:
        if not self._ms:
            return 0.0
        mean = float(np.mean(self._ms[-30:]))
        return 1000.0 / mean if mean > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        s = dict(self.tracker_.stats()) if self.tracker_ else {}
        s["mean_infer_ms"] = round(float(np.mean(self._ms)), 2) if self._ms else 0.0
        s["fps"] = round(self.fps, 1)
        return s
