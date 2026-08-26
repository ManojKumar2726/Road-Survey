"""Canonical road-damage taxonomy.

Every model in the registry names its classes differently, and -- worse -- some
of them order the *same* classes differently:

    SreekarAditya  0:longitudinal_crack  1:transverse_crack  2:alligator_crack  3:pothole
    rezzzq         0:D00                 1:D10               2:D20              3:D40  4:Repair
    ozair23        0:alligator crack     1:transverse crack  2:longitudinal crack  3:other  4:Pothole

`ozair23` has longitudinal and alligator swapped relative to the others. So a
raw class id means nothing on its own: comparing two models, colouring boxes by
type, or filtering "show me only potholes" all need a model-independent key.

That key is a canonical damage type. Each model maps its raw ids onto these
keys -- explicitly via `class_map:` in models.yaml, or by name-matching as a
fallback -- and everything downstream (colours, filters, stats, CSV, survey
report) works in canonical terms.

The RDD2022 D-codes are the closest thing this field has to a standard, so they
anchor the taxonomy; the extra keys cover models that don't follow it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# The taxonomy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DamageClass:
    """One canonical road-damage type."""

    key: str
    label: str  # for tables and the UI
    short: str  # for the video overlay, where space is tight
    code: str  # RDD2022 code, "" when the type isn't part of RDD
    color_rgb: tuple[int, int, int]
    severity: float  # 0-1 weight used by the survey score
    description: str

    @property
    def bgr(self) -> tuple[int, int, int]:
        """OpenCV wants BGR."""
        r, g, b = self.color_rgb
        return (b, g, r)

    @property
    def hex(self) -> str:
        return "#%02x%02x%02x" % self.color_rgb


UNKNOWN_KEY = "unknown"

# Ordered worst-first: this is the display order in legends and tables, and the
# colours read as a heat ramp (red = safety-critical, blue/green = cosmetic).
TAXONOMY: dict[str, DamageClass] = {
    d.key: d
    for d in (
        DamageClass(
            key="pothole",
            label="Pothole",
            short="pothole",
            code="D40",
            color_rgb=(231, 76, 60),
            severity=1.00,
            description="Bowl-shaped surface failure. Safety-critical.",
        ),
        DamageClass(
            key="alligator_crack",
            label="Alligator crack",
            short="alligator",
            code="D20",
            color_rgb=(243, 156, 18),
            severity=0.75,
            description="Interconnected fatigue cracking. Precursor to potholing.",
        ),
        DamageClass(
            key="longitudinal_crack",
            label="Longitudinal crack",
            short="long. crack",
            code="D00",
            color_rgb=(241, 196, 15),
            severity=0.40,
            description="Crack running with the direction of travel.",
        ),
        DamageClass(
            key="transverse_crack",
            label="Transverse crack",
            short="trans. crack",
            code="D10",
            color_rgb=(52, 152, 219),
            severity=0.35,
            description="Crack running across the carriageway.",
        ),
        DamageClass(
            key="crack",
            label="Crack (unspecified)",
            short="crack",
            code="",
            color_rgb=(26, 188, 156),
            severity=0.40,
            description="Crack from a model that doesn't distinguish orientation.",
        ),
        DamageClass(
            key="other",
            label="Other damage",
            short="other",
            code="",
            color_rgb=(155, 89, 182),
            severity=0.30,
            description="Damage a model reports without a specific type.",
        ),
        DamageClass(
            key="repair",
            label="Repaired area",
            short="repair",
            code="",
            color_rgb=(46, 204, 113),
            severity=0.10,
            description="Previously patched surface. Not damage -- context.",
        ),
        DamageClass(
            key=UNKNOWN_KEY,
            label="Unmapped",
            short="unmapped",
            code="",
            color_rgb=(149, 165, 166),
            severity=0.30,
            description="Raw class this lab could not map onto the taxonomy.",
        ),
    )
}

ORDER: list[str] = list(TAXONOMY)
"""Canonical keys, worst-first. Use this anywhere classes need a stable order."""

DAMAGE_KEYS: list[str] = [k for k in ORDER if k not in ("repair", UNKNOWN_KEY)]
"""Keys that represent actual damage -- `repair` is context, not a defect."""


def get(key: str) -> DamageClass:
    """Look up a canonical class, falling back to `unknown` rather than raising."""
    return TAXONOMY.get(key, TAXONOMY[UNKNOWN_KEY])


def label_of(key: str) -> str:
    return get(key).label


def short_of(key: str) -> str:
    return get(key).short


def color_of(key: str) -> tuple[int, int, int]:
    """BGR, for OpenCV drawing."""
    return get(key).bgr


def severity_of(key: str) -> float:
    return get(key).severity


def sort_keys(keys) -> list[str]:
    """Order an arbitrary set of canonical keys worst-first."""
    return sorted(set(keys), key=lambda k: (ORDER.index(k) if k in ORDER else 99, k))


# --------------------------------------------------------------------------- #
# Name-based fallback mapping
# --------------------------------------------------------------------------- #

# Exact matches on a normalised name ("Alligator Crack" -> "alligator_crack").
# Checked before the looser keyword rules below.
_ALIASES: dict[str, str] = {
    # RDD2022 codes, in the spellings that show up in the wild
    "d00": "longitudinal_crack",
    "d01": "longitudinal_crack",
    "d10": "transverse_crack",
    "d11": "transverse_crack",
    "d20": "alligator_crack",
    "d40": "pothole",
    "d43": "other",
    "d44": "other",
    "d50": "other",
    # plain-language names
    "pothole": "pothole",
    "potholes": "pothole",
    "pot hole": "pothole",
    "alligator crack": "alligator_crack",
    "alligator cracks": "alligator_crack",
    "fatigue crack": "alligator_crack",
    "crocodile crack": "alligator_crack",
    "block crack": "alligator_crack",
    "longitudinal crack": "longitudinal_crack",
    "lateral crack": "transverse_crack",
    "transverse crack": "transverse_crack",
    "crack": "crack",
    "cracks": "crack",
    "surface crack": "crack",
    "linear crack": "crack",
    "repair": "repair",
    "repaired": "repair",
    "patch": "repair",
    "patched": "repair",
    "repaired area": "repair",
    "other corruption": "other",
    "damage": "other",
    "road damage": "other",
    "surface damage": "other",
}

# Ordered keyword rules, first match wins. Longitudinal/transverse are checked
# before the generic "crack" rule so they don't get collapsed into it.
_KEYWORD_RULES: list[tuple[str, str]] = [
    ("pothole", "pothole"),
    ("pot hole", "pothole"),
    ("alligator", "alligator_crack"),
    ("crocodile", "alligator_crack"),
    ("fatigue", "alligator_crack"),
    ("longitudinal", "longitudinal_crack"),
    ("transverse", "transverse_crack"),
    ("lateral", "transverse_crack"),
    ("repair", "repair"),
    ("patch", "repair"),
    ("crack", "crack"),
    ("corruption", "other"),
    ("damage", "other"),
]


def _normalise(raw: str) -> str:
    """'Alligator-Crack_02' -> 'alligator crack 02'."""
    s = str(raw).strip().lower()
    s = re.sub(r"[_\-/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canon_from_name(raw_name: str) -> str:
    """Best-effort canonical key for a raw class name.

    Used when a model has no explicit `class_map:`. Returns `unknown` rather
    than guessing wildly -- an unmapped class still draws and still counts, it
    just doesn't claim to be a damage type it might not be.
    """
    s = _normalise(raw_name)
    if not s:
        return UNKNOWN_KEY
    if s in _ALIASES:
        return _ALIASES[s]
    # A bare RDD code with a suffix, e.g. "D20_alligator".
    m = re.match(r"^(d\d{2})\b", s)
    if m and m.group(1) in _ALIASES:
        return _ALIASES[m.group(1)]
    for needle, key in _KEYWORD_RULES:
        if needle in s:
            return key
    return UNKNOWN_KEY


def build_class_map(
    names: dict[int, str],
    class_map: dict[int, str] | None = None,
) -> dict[int, str]:
    """Resolve a model's raw class ids onto canonical keys.

    `class_map` (from models.yaml) wins where it's given -- it's ground truth
    written against the checkpoint's actual `model.names`. Ids it doesn't cover
    fall back to name matching.
    """
    resolved: dict[int, str] = {}
    explicit = {int(k): str(v) for k, v in (class_map or {}).items()}
    for cid, raw in names.items():
        cid = int(cid)
        key = explicit.get(cid) or canon_from_name(raw)
        resolved[cid] = key if key in TAXONOMY else UNKNOWN_KEY
    return resolved


def validate_class_map(
    names: dict[int, str], class_map: dict[int, str] | None
) -> list[str]:
    """Warnings for a `class_map:` that doesn't match the loaded checkpoint.

    Catches the two mistakes that matter: mapping an id the model doesn't have
    (usually a stale entry after a re-train), and naming a canonical key that
    doesn't exist (usually a typo).
    """
    problems: list[str] = []
    for raw_id, key in (class_map or {}).items():
        if int(raw_id) not in names:
            problems.append(
                f"class_map has id {raw_id}, but the checkpoint only defines "
                f"{sorted(names)}"
            )
        if str(key) not in TAXONOMY:
            problems.append(
                f"class_map id {raw_id} -> unknown canonical key '{key}'. "
                f"Valid keys: {', '.join(ORDER)}"
            )
    return problems
