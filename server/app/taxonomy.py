"""Re-export the lab's damage taxonomy so the server never redefines it.

A pothole has to be the same red in window 1's video overlay and on window 2's
map. Two hardcoded palettes drift the first time somebody adds a damage type,
so both windows resolve colour and severity from `labcore.taxonomy` -- the edge
by importing it, the dashboard by fetching `/api/taxonomy`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

SERVER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVER_DIR.parent
LAB_DIR = Path(os.environ.get("ROADLAB_DIR") or (REPO_ROOT / "road-damage-lab"))

if (LAB_DIR / "labcore").is_dir():
    if str(LAB_DIR) not in sys.path:
        sys.path.append(str(LAB_DIR))

from labcore import taxonomy as tax  # noqa: E402

TAXONOMY = tax.TAXONOMY
ORDER = tax.ORDER
UNKNOWN_KEY = tax.UNKNOWN_KEY

severity_of = tax.severity_of
label_of = tax.label_of
sort_keys = tax.sort_keys
get = tax.get


def as_payload() -> list[dict[str, Any]]:
    """The full taxonomy, worst-first, for the dashboard to style itself from."""
    return [
        {
            "key": d.key,
            "label": d.label,
            "short": d.short,
            "code": d.code,
            "hex": d.hex,
            "severity": d.severity,
            "description": d.description,
        }
        for d in (TAXONOMY[k] for k in ORDER)
    ]


def hex_of(key: str) -> str:
    return get(key).hex
