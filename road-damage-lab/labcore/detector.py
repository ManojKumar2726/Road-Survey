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

from . import taxonomy as tax
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
    """One box in one frame, normalised across model families.

    `label` is whatever the checkpoint called this class; `canon` is the
    lab-wide damage key it maps onto. Compare models on `canon`, never on
    `cls_id` -- raw ids don't agree between checkpoints.
    """

    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    label: str
    track_id: int | None = None
    canon: str = tax.UNKNOWN_KEY

    # ------------------------------------------------------------- taxonomy

    @property
    def damage(self) -> tax.DamageClass:
        return tax.get(self.canon)

    @property
    def canon_label(self) -> str:
        return self.damage.label

    @property
    def severity(self) -> float:
        return self.damage.severity

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
            "damage": self.canon,
            "damage_label": self.canon_label,
            "severity": self.severity,
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

        # Raw ids are meaningless across checkpoints -- everything downstream
        # works in canonical keys, resolved once here.
        self.canon_by_id: dict[int, str] = tax.build_class_map(
            self.names, spec.class_map
        )
        self.map_warnings: list[str] = tax.validate_class_map(
            self.names, spec.class_map
        )

    # ------------------------------------------------------------------ meta

    @property
    def class_names(self) -> list[str]:
        return [self.names[k] for k in sorted(self.names)]

    @property
    def canon_keys(self) -> list[str]:
        """Canonical damage types this model can emit, worst-first."""
        return tax.sort_keys(self.canon_by_id.values())

    def ids_for_canon(self, keys: Iterable[str]) -> list[int]:
        """Raw class ids for a set of canonical keys.

        This is what makes "show only potholes" work in compare mode: each
        model resolves the same canonical selection to its own indices.
        """
        wanted = set(keys)
        return sorted(c for c, k in self.canon_by_id.items() if k in wanted)

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
            "class_map": {
                cid: f"{self.names[cid]} -> {self.canon_by_id[cid]}"
                for cid in sorted(self.names)
            },
            "damage_types": self.canon_keys,
            "params_m": round(n_params / 1e6, 2) if n_params else None,
            "map_warnings": self.map_warnings,
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
            raw_label = str(names.get(cid, f"class_{cid}"))
            dets.append(
                Detection(
                    xyxy=tuple(float(v) for v in xyxy[i]),  # type: ignore[arg-type]
                    conf=float(confs[i]),
                    cls_id=cid,
                    label=raw_label,
                    track_id=int(ids[i]) if ids is not None else None,
                    # `result.names` can differ from the names captured at load
                    # time, so fall back to matching on the label itself.
                    canon=self.canon_by_id.get(cid) or tax.canon_from_name(raw_label),
                )
            )
        return dets


# --------------------------------------------------------------------------- #
# Small aggregation helper used by the UI
# --------------------------------------------------------------------------- #


class RunStats:
    """Running totals for one video pass, broken down by damage type.

    A single "unique objects" count stops being useful once a model reports
    five damage types, so the per-class unique-ID tallies are the number that
    actually answers "what's wrong with this road".
    """

    def __init__(self) -> None:
        self.frames = 0
        self.total_dets = 0
        self.unique_ids: set[int] = set()
        self.conf_sum = 0.0
        self.ms: list[float] = []
        self.per_class: dict[str, int] = {}  # canonical key -> boxes seen
        self.per_raw_label: dict[str, int] = {}  # the model's own names
        self.unique_by_class: dict[str, set[int]] = {}
        self.conf_sum_by_class: dict[str, float] = {}
        self.area_pct_by_class: dict[str, float] = {}

    def update(
        self,
        dets: Iterable[Detection],
        infer_ms: float,
        frame_w: int = 0,
        frame_h: int = 0,
    ) -> None:
        self.frames += 1
        self.ms.append(infer_ms)
        for d in dets:
            self.total_dets += 1
            self.conf_sum += d.conf
            k = d.canon
            self.per_class[k] = self.per_class.get(k, 0) + 1
            self.per_raw_label[d.label] = self.per_raw_label.get(d.label, 0) + 1
            self.conf_sum_by_class[k] = self.conf_sum_by_class.get(k, 0.0) + d.conf
            if frame_w and frame_h:
                self.area_pct_by_class[k] = self.area_pct_by_class.get(
                    k, 0.0
                ) + d.area_pct(frame_w, frame_h)
            if d.track_id is not None:
                self.unique_ids.add(d.track_id)
                self.unique_by_class.setdefault(k, set()).add(d.track_id)

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / self.total_dets if self.total_dets else 0.0

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.ms)) if self.ms else 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    @property
    def classes_seen(self) -> list[str]:
        """Canonical keys with at least one detection, worst-first."""
        return tax.sort_keys(self.per_class)

    def unique_counts(self) -> dict[str, int]:
        """Canonical key -> unique track IDs. The headline survey number."""
        return {k: len(self.unique_by_class.get(k, ())) for k in self.classes_seen}

    def mean_conf_of(self, key: str) -> float:
        n = self.per_class.get(key, 0)
        return self.conf_sum_by_class.get(key, 0.0) / n if n else 0.0

    @property
    def severity_score(self) -> float:
        """Severity-weighted damage rate: weighted unique finds per 100 frames.

        Weights come from the taxonomy, so a pothole counts for ~3x a
        transverse crack. Comparable only across runs on the same clip.
        """
        if not self.frames:
            return 0.0
        # Without tracking every frame re-counts the same defect, so fall back
        # to box counts and let the per-frame rate carry the meaning.
        counts = self.unique_counts() if self.unique_ids else self.per_class
        weighted = sum(tax.severity_of(k) * n for k, n in counts.items())
        return 100.0 * weighted / self.frames

    def summary(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "detections": self.total_dets,
            "unique_objects": len(self.unique_ids),
            "unique_by_class": self.unique_counts(),
            "mean_conf": round(self.mean_conf, 3),
            "mean_infer_ms": round(self.mean_ms, 2),
            "fps": round(self.fps, 1),
            "severity_score": round(self.severity_score, 2),
            "per_class": dict(self.per_class),
            "per_raw_label": dict(self.per_raw_label),
        }
