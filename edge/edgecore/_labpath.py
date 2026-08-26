"""Put the sibling `road-damage-lab/` on the import path.

The lab is a checked-in sibling directory rather than an installed package, so
`labcore` isn't importable by default. Rather than duplicate the detector,
taxonomy and video code here, the edge agent adds the lab to `sys.path` once,
at import time.

Set `ROADLAB_DIR` to point somewhere else (e.g. if the lab gets installed
properly later). Importing this module is idempotent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EDGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EDGE_DIR.parent

LAB_DIR = Path(os.environ.get("ROADLAB_DIR") or (REPO_ROOT / "road-damage-lab"))


def _ensure_on_path() -> Path:
    if not (LAB_DIR / "labcore").is_dir():
        raise ImportError(
            f"Could not find `labcore` under {LAB_DIR}. The edge agent reuses the "
            "detector and taxonomy from road-damage-lab. Set ROADLAB_DIR if the "
            "lab lives elsewhere."
        )
    p = str(LAB_DIR)
    if p not in sys.path:
        # Appended, not prepended: the edge agent's own modules must win any
        # name collision with the lab's.
        sys.path.append(p)
    return LAB_DIR


_ensure_on_path()
