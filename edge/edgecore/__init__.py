"""Onboard agent for the road survey prototype.

Wraps the research bench in `road-damage-lab/labcore` with the three things an
onboard unit needs and a benchmark doesn't: a track lifecycle that emits
discrete events during a pass, simulated GPS/metadata, and a publisher that
posts to the central server and survives losing the network.

`labcore` is imported from the sibling lab directory -- see `_labpath.py`.
"""

from __future__ import annotations

from . import _labpath  # noqa: F401  -- must precede any labcore import

__all__ = ["events", "gps", "config", "pipeline", "publisher"]
