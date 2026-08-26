"""Model registry: turns entries in models.yaml into resolvable weight files.

The point of this module is that adding a model to the lab never means touching
the app -- you add a block to models.yaml and it shows up in the picker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = ROOT / "models.yaml"
WEIGHTS_DIR = ROOT / "weights"
HF_CACHE = WEIGHTS_DIR / "hf"


@dataclass
class ModelSpec:
    """One switchable model in the lab."""

    id: str
    name: str
    source: str  # "hf" | "local" | "ultralytics"
    repo_id: str | None = None
    filename: str | None = None
    path: str | None = None
    weights: str | None = None
    revision: str | None = None
    task: str = "detect"
    default_conf: float = 0.25
    default_iou: float = 0.45
    default_imgsz: int = 640
    class_names: dict[int, str] | None = None
    # Raw class id -> canonical damage key (see labcore/taxonomy.py). Without
    # this the lab falls back to matching the checkpoint's own class names,
    # which is fine for obvious ones ("pothole") and wrong for D-codes.
    class_map: dict[int, str] | None = None
    notes: str = ""
    enabled: bool = True

    # ---------------------------------------------------------------- helpers

    @property
    def origin(self) -> str:
        """Short human-readable description of where the weights come from."""
        if self.source == "hf":
            return f"hf:{self.repo_id}/{self.filename}"
        if self.source == "local":
            return f"local:{self.path}"
        return f"ultralytics:{self.weights}"

    def is_cached(self) -> bool:
        """True when the weights are already on disk (no download needed)."""
        try:
            return self._peek_local() is not None
        except Exception:
            return False

    def _peek_local(self) -> Path | None:
        if self.source == "local":
            p = Path(self.path or "")
            p = p if p.is_absolute() else ROOT / p
            return p if p.exists() else None

        if self.source == "ultralytics":
            p = WEIGHTS_DIR / "ultralytics" / (self.weights or "")
            return p if p.exists() else None

        if self.source == "hf":
            from huggingface_hub import try_to_load_from_cache

            hit = try_to_load_from_cache(
                repo_id=self.repo_id,
                filename=self.filename,
                cache_dir=str(HF_CACHE),
                revision=self.revision,
            )
            return Path(hit) if isinstance(hit, str) else None

        return None

    def resolve(self) -> Path:
        """Return a local path to the weights, downloading them if needed."""
        cached = self._peek_local()
        if cached is not None:
            return cached

        if self.source == "hf":
            from huggingface_hub import hf_hub_download

            HF_CACHE.mkdir(parents=True, exist_ok=True)
            got = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                revision=self.revision,
                cache_dir=str(HF_CACHE),
            )
            return Path(got)

        if self.source == "ultralytics":
            target_dir = WEIGHTS_DIR / "ultralytics"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / (self.weights or "")
            try:
                from ultralytics.utils.downloads import attempt_download_asset

                got = attempt_download_asset(self.weights, retry=1)
                got = Path(got)
                if got.exists() and got.resolve() != target.resolve():
                    got.replace(target)
                    return target
                return got
            except Exception:
                # Let ultralytics deal with the name at load time.
                return Path(self.weights or "")

        if self.source == "local":
            p = Path(self.path or "")
            p = p if p.is_absolute() else ROOT / p
            raise FileNotFoundError(
                f"Model '{self.id}': local weights not found at {p}. "
                "Drop the .pt there or fix `path:` in models.yaml."
            )

        raise ValueError(f"Model '{self.id}': unknown source '{self.source}'")


@dataclass
class Registry:
    specs: list[ModelSpec] = field(default_factory=list)

    def __iter__(self):
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def get(self, model_id: str) -> ModelSpec:
        for s in self.specs:
            if s.id == model_id:
                return s
        known = ", ".join(s.id for s in self.specs) or "<none>"
        raise KeyError(f"No model with id '{model_id}'. Known ids: {known}")


_ALLOWED = set(ModelSpec.__dataclass_fields__.keys())


def _coerce(raw: dict[str, Any]) -> ModelSpec:
    unknown = set(raw) - _ALLOWED
    if unknown:
        raise ValueError(
            f"models.yaml entry '{raw.get('id', '?')}' has unknown key(s): "
            f"{sorted(unknown)}"
        )
    for required in ("id", "name", "source"):
        if not raw.get(required):
            raise ValueError(f"models.yaml entry is missing required key '{required}'")

    # A typo'd canonical key would silently colour boxes grey and drop them out
    # of the survey report, so catch it at parse time rather than at run time.
    cmap = raw.get("class_map")
    if cmap:
        from .taxonomy import ORDER, TAXONOMY

        bad = {k: v for k, v in cmap.items() if str(v) not in TAXONOMY}
        if bad:
            raise ValueError(
                f"models.yaml entry '{raw['id']}' maps to unknown canonical "
                f"key(s): {bad}. Valid keys: {', '.join(ORDER)}"
            )

    return ModelSpec(**raw)


def load_registry(path: str | Path | None = None) -> Registry:
    """Parse models.yaml into a Registry. Disabled entries are dropped."""
    p = Path(path) if path else MODELS_YAML
    if not p.exists():
        raise FileNotFoundError(f"Model registry not found: {p}")

    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = doc.get("models") or []
    specs = [_coerce(e) for e in entries]

    seen: set[str] = set()
    for s in specs:
        if s.id in seen:
            raise ValueError(f"Duplicate model id in models.yaml: '{s.id}'")
        seen.add(s.id)

    return Registry([s for s in specs if s.enabled])


def list_specs(path: str | Path | None = None) -> list[ModelSpec]:
    return list(load_registry(path).specs)


def get_spec(model_id: str, path: str | Path | None = None) -> ModelSpec:
    return load_registry(path).get(model_id)
