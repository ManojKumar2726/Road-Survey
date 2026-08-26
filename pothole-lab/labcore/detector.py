"""A thin, uniform wrapper around Ultralytics YOLO.

Every model in the registry -- detect or segment, v8 / v9 / v11, HF or local --
comes out of here as the same `list[Detection]`, so the UI and the drawing code
never has to care which checkpoint is loaded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .registry import ModelSpec


def _supports_quantize() -> bool:
    """Ultralytics >= 8.4 replaced the `half` flag with a unified `quantize` arg."""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        return "quantize" in DEFAULT_CFG_DICT
    except Exception:
        return False


USE_QUANTIZE = _supports_quantize()


# --------------------------------------------------------------------------- #
# Detection record
# --------------------------------------------------------------------------- #


@dataclass
class Detection:
    """One box in one frame, normalised across model families."""

    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    label: str
    track_id: int | None = None

    @property
    def x1(self) -> float:
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        return self.xyxy[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def area_pct(self, frame_w: int, frame_h: int) -> float:
        """Box area as a percentage of the frame -- a crude 'how big' signal."""
        denom = float(frame_w * frame_h) or 1.0
        return 100.0 * self.area / denom

    def as_row(self, frame_idx: int, frame_w: int, frame_h: int) -> dict[str, Any]:
        """Flat dict for CSV / dataframe export."""
        return {
            "frame": frame_idx,
            "track_id": self.track_id if self.track_id is not None else -1,
            "label": self.label,
            "cls_id": self.cls_id,
            "conf": round(self.conf, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "w": round(self.width, 1),
            "h": round(self.height, 1),
            "area_px": round(self.area, 1),
            "area_pct_frame": round(self.area_pct(frame_w, frame_h), 3),
        }


@dataclass
class InferResult:
    detections: list[Detection] = field(default_factory=list)
    infer_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.infer_ms + self.postprocess_ms


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #


def available_devices() -> list[str]:
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] + devices
    except Exception:
        pass
    return devices


def resolve_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def device_label(device: str, short: bool = False) -> str:
    """Human-readable device name. `short` trims it to fit a video overlay."""
    if device.startswith("cuda"):
        try:
            import torch

            idx = int(device.split(":")[1]) if ":" in device else 0
            name = torch.cuda.get_device_name(idx)
            if short:
                # "NVIDIA GeForce RTX 3060" -> "RTX 3060"
                for marker in ("RTX", "GTX", "Tesla", "Quadro", "A100", "H100"):
                    if marker in name:
                        return name[name.index(marker):]
                return name
            return f"{device} · {name}"
        except Exception:
            return device
    return "cpu"


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #


class Detector:
    """Loads one checkpoint and runs it, with or without tracking."""

    def __init__(
        self,
        spec: ModelSpec,
        device: str = "auto",
        half: bool = False,
        fuse: bool = True,
    ) -> None:
        from ultralytics import YOLO

        self.spec = spec
        self.device = resolve_device(device)
        self.half = bool(half) and self.device.startswith("cuda")
        self.weights_path: Path = spec.resolve()

        self.model = YOLO(str(self.weights_path))
        if fuse:
            try:
                self.model.fuse()
            except Exception:
                pass  # already fused, or not fusable for this head

        raw_names = getattr(self.model, "names", None) or {}
        if spec.class_names:
            raw_names = {**raw_names, **spec.class_names}
        self.names: dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}

    # ------------------------------------------------------------------ meta

    @property
    def class_names(self) -> list[str]:
        return [self.names[k] for k in sorted(self.names)]

    def describe(self) -> dict[str, Any]:
        n_params = None
        try:
            n_params = sum(p.numel() for p in self.model.model.parameters())
        except Exception:
            pass
        return {
            "id": self.spec.id,
            "name": self.spec.name,
            "task": getattr(self.model, "task", self.spec.task),
            "origin": self.spec.origin,
            "weights": str(self.weights_path),
            "device": device_label(self.device),
            "half": self.half,
            "classes": self.class_names,
            "params_m": round(n_params / 1e6, 2) if n_params else None,
        }

    # ------------------------------------------------------------- inference

    def reset_tracker(self) -> None:
        """Forget all track IDs. Call this between videos or on re-run."""
        try:
            predictor = getattr(self.model, "predictor", None)
            trackers = getattr(predictor, "trackers", None) if predictor else None
            if trackers:
                for t in trackers:
                    t.reset()
                return
        except Exception:
            pass
        # Fall back to dropping the predictor entirely -- next call rebuilds it.
        try:
            self.model.predictor = None
        except Exception:
            pass

    def warmup(self, imgsz: int = 640) -> None:
        blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(blank, imgsz=imgsz, device=self.device, verbose=False)
        except Exception:
            pass

    def infer(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        track: bool = True,
        tracker: str = "bytetrack.yaml",
        classes: Sequence[int] | None = None,
        max_det: int = 300,
        agnostic_nms: bool = False,
    ) -> InferResult:
        """Run one frame. Returns normalised detections plus timing."""
        kwargs: dict[str, Any] = dict(
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            device=self.device,
            max_det=int(max_det),
            agnostic_nms=bool(agnostic_nms),
            verbose=False,
        )
        if USE_QUANTIZE:
            if self.half:
                kwargs["quantize"] = 16  # FP16; omitted entirely means FP32
        else:
            kwargs["half"] = self.half
        if classes:
            kwargs["classes"] = list(classes)

        t0 = time.perf_counter()
        if track:
            results = self.model.track(
                frame, persist=True, tracker=tracker, **kwargs
            )
        else:
            results = self.model.predict(frame, **kwargs)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            return InferResult([], wall_ms)

        r = results[0]
        speed = getattr(r, "speed", None) or {}
        out = InferResult(
            # `model.track` registers a postprocess callback that stays attached to
            # the model, so a later `predict` still comes back with ids. Ignore them
            # unless this call actually asked for tracking.
            detections=self._unpack(r, with_ids=track),
            infer_ms=float(speed.get("inference", wall_ms) or wall_ms),
            preprocess_ms=float(speed.get("preprocess", 0.0) or 0.0),
            postprocess_ms=float(speed.get("postprocess", 0.0) or 0.0),
        )
        return out

    def _unpack(self, result: Any, with_ids: bool = True) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        clss = (
            boxes.cls.cpu().numpy().astype(int)
            if boxes.cls is not None
            else np.zeros(len(xyxy), dtype=int)
        )
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if with_ids and boxes.id is not None
            else None
        )
        if ids is not None and len(ids) != len(xyxy):
            ids = None  # shouldn't happen, but never mis-pair an id with a box

        names = getattr(result, "names", None) or self.names

        dets: list[Detection] = []
        for i in range(len(xyxy)):
            cid = int(clss[i])
            dets.append(
                Detection(
                    xyxy=tuple(float(v) for v in xyxy[i]),  # type: ignore[arg-type]
                    conf=float(confs[i]),
                    cls_id=cid,
                    label=str(names.get(cid, f"class_{cid}")),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return dets


# --------------------------------------------------------------------------- #
# Small aggregation helper used by the UI
# --------------------------------------------------------------------------- #


class RunStats:
    """Running totals for one video pass."""

    def __init__(self) -> None:
        self.frames = 0
        self.total_dets = 0
        self.unique_ids: set[int] = set()
        self.conf_sum = 0.0
        self.ms: list[float] = []
        self.per_class: dict[str, int] = {}

    def update(self, dets: Iterable[Detection], infer_ms: float) -> None:
        self.frames += 1
        self.ms.append(infer_ms)
        for d in dets:
            self.total_dets += 1
            self.conf_sum += d.conf
            self.per_class[d.label] = self.per_class.get(d.label, 0) + 1
            if d.track_id is not None:
                self.unique_ids.add(d.track_id)

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / self.total_dets if self.total_dets else 0.0

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.ms)) if self.ms else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "detections": self.total_dets,
            "unique_potholes": len(self.unique_ids),
            "mean_conf": round(self.mean_conf, 3),
            "mean_infer_ms": round(self.mean_ms, 2),
            "fps": round(self.fps, 1),
            "per_class": dict(self.per_class),
        }
