"""Onboard unit settings: what this bus is, and where it reports.

Precedence is CLI > environment > default, so a demo can be driven entirely
from flags while a real deployment sets `ROADSURVEY_*` once in the unit's
environment and never passes an argument.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

EDGE_DIR = Path(__file__).resolve().parent.parent

ENV_PREFIX = "ROADSURVEY_"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(ENV_PREFIX + name.upper(), default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


@dataclass
class EdgeConfig:
    """Identity and connectivity for one onboard unit."""

    # ---- who
    bus_id: str = "BUS_001"
    route_id: str = ""

    # ---- where it posts
    api_url: str = "http://127.0.0.1:8000"
    api_timeout_s: float = 5.0
    batch_size: int = 1  # 1 = post as events fire, which is what the demo wants
    spool_dir: Path = EDGE_DIR / "spool"

    # ---- detection
    model_id: str = "rdd-yolo12s"
    conf: float | None = None

    @classmethod
    def from_env(cls) -> "EdgeConfig":
        return cls(
            bus_id=_env("bus_id", "BUS_001"),
            route_id=_env("route_id", ""),
            api_url=_env("api_url", "http://127.0.0.1:8000").rstrip("/"),
            api_timeout_s=_env_float("api_timeout_s", 5.0),
            batch_size=int(_env_float("batch_size", 1)),
            spool_dir=Path(_env("spool_dir") or (EDGE_DIR / "spool")),
            model_id=_env("model_id", "rdd-yolo12s"),
            conf=float(_env("conf")) if _env("conf") else None,
        )

    def merge_cli(self, **overrides: Any) -> "EdgeConfig":
        """Apply non-None CLI values over whatever the environment gave."""
        for k, v in overrides.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)
        if isinstance(self.api_url, str):
            self.api_url = self.api_url.rstrip("/")
        return self

    @property
    def events_endpoint(self) -> str:
        return f"{self.api_url}/api/events"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["spool_dir"] = str(self.spool_dir)
        return d
