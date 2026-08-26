"""Turn a pass of raw detections into a road-condition summary.

The detection table answers "what did the model see in frame 412". A survey
answers "what is wrong with this road, and how badly" -- which needs the boxes
collapsed per tracked defect, weighted by damage type, and located along the
clip.

Everything here works off canonical damage keys, so a report is comparable
between two models even when their raw class ids disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import taxonomy as tax

# A defect seen in only one or two frames is usually a flicker rather than a
# real find. Kept low -- at high stride a genuine defect can be brief.
MIN_FRAMES_FOR_CONFIRMED = 2


@dataclass
class DefectRecord:
    """One tracked defect, collapsed from every frame it appeared in."""

    track_id: int
    canon: str
    frames: int
    first_frame: int
    last_frame: int
    mean_conf: float
    max_conf: float
    peak_area_pct: float
    mean_area_pct: float

    @property
    def label(self) -> str:
        return tax.label_of(self.canon)

    @property
    def severity(self) -> float:
        return tax.severity_of(self.canon)

    @property
    def confirmed(self) -> bool:
        return self.frames >= MIN_FRAMES_FOR_CONFIRMED

    @property
    def weighted_score(self) -> float:
        """Severity scaled by how big the defect got and how sure the model was.

        Area matters because a 6% -of-frame pothole is a different repair job
        from a 0.2% one, and confidence keeps marginal detections from
        dominating the total.
        """
        size = min(1.0, self.peak_area_pct / 5.0)  # 5% of frame ~= "large"
        return self.severity * (0.5 + 0.5 * size) * self.max_conf

    def as_row(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "damage": self.canon,
            "damage_label": self.label,
            "severity": round(self.severity, 2),
            "frames": self.frames,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "mean_conf": round(self.mean_conf, 3),
            "max_conf": round(self.max_conf, 3),
            "peak_area_pct": round(self.peak_area_pct, 3),
            "mean_area_pct": round(self.mean_area_pct, 3),
            "score": round(self.weighted_score, 3),
            "confirmed": self.confirmed,
        }


@dataclass
class ClassSummary:
    """Per-damage-type totals across the whole pass."""

    canon: str
    defects: int  # unique tracked defects
    boxes: int  # raw detections
    mean_conf: float
    peak_area_pct: float
    total_score: float

    @property
    def label(self) -> str:
        return tax.label_of(self.canon)

    @property
    def hex(self) -> str:
        return tax.get(self.canon).hex

    def as_row(self) -> dict[str, Any]:
        return {
            "damage": self.canon,
            "damage_label": self.label,
            "defects": self.defects,
            "boxes": self.boxes,
            "mean_conf": round(self.mean_conf, 3),
            "worst_area_pct": round(self.peak_area_pct, 3),
            "score": round(self.total_score, 2),
        }


@dataclass
class SurveyReport:
    """A road-condition summary for one model over one clip."""

    model_id: str
    model_name: str
    frames: int
    tracked: bool
    defects: list[DefectRecord] = field(default_factory=list)
    by_class: list[ClassSummary] = field(default_factory=list)
    total_boxes: int = 0
    # Boxes the tracker never assigned an ID to. Real detections, but they
    # can't be collapsed into defects, so counting them as unique finds would
    # inflate the totals -- they're reported separately instead.
    unassigned_boxes: int = 0

    # ------------------------------------------------------------- headlines

    @property
    def total_defects(self) -> int:
        return len(self.defects)

    @property
    def confirmed_defects(self) -> int:
        return sum(1 for d in self.defects if d.confirmed)

    @property
    def damage_score(self) -> float:
        """Severity-weighted damage per 100 frames.

        Only comparable between runs over the same clip at the same stride --
        it's a relative measure for ranking models or road segments, not an
        absolute pavement-condition index.
        """
        if not self.frames:
            return 0.0
        return 100.0 * sum(d.weighted_score for d in self.defects) / self.frames

    @property
    def grade(self) -> str:
        """Coarse letter grade off the damage score. Deliberately blunt."""
        s = self.damage_score
        if s <= 0:
            return "A - no damage found"
        if s < 2:
            return "B - light"
        if s < 6:
            return "C - moderate"
        if s < 15:
            return "D - poor"
        return "E - severe"

    @property
    def worst(self) -> DefectRecord | None:
        return max(self.defects, key=lambda d: d.weighted_score, default=None)

    def headline(self) -> str:
        """One-line plain-language verdict."""
        if not self.defects:
            return "No road damage detected in this pass."
        parts = [
            f"{c.defects} x {c.label.lower()}"
            for c in self.by_class
            if c.defects and c.canon != "repair"
        ]
        if not parts:
            parts = [f"{c.boxes} x {c.label.lower()}" for c in self.by_class]
        return f"{self.grade} - " + ", ".join(parts)

    # ---------------------------------------------------------------- tables

    def defect_rows(self, confirmed_only: bool = False) -> list[dict[str, Any]]:
        rows = [
            d.as_row()
            for d in sorted(self.defects, key=lambda x: -x.weighted_score)
            if not confirmed_only or d.confirmed
        ]
        return rows

    def class_rows(self) -> list[dict[str, Any]]:
        return [c.as_row() for c in self.by_class]

    def hotspots(self, bins: int = 12) -> list[dict[str, Any]]:
        """Damage score bucketed along the clip -- where the bad stretches are.

        With a forward-facing camera, frame index is a proxy for distance
        travelled, so these buckets approximate road segments.
        """
        if not self.defects or self.frames <= 0:
            return []
        bins = max(1, min(bins, self.frames))
        width = max(1, self.frames // bins)
        buckets: list[dict[str, Any]] = [
            {
                "segment": i + 1,
                "start_frame": i * width,
                "end_frame": (i + 1) * width - 1,
                "defects": 0,
                "score": 0.0,
            }
            for i in range(bins)
        ]
        for d in self.defects:
            # Attribute a defect to where it was first seen.
            i = min(bins - 1, d.first_frame // width)
            buckets[i]["defects"] += 1
            buckets[i]["score"] += d.weighted_score
        for b in buckets:
            b["score"] = round(b["score"], 3)
        return buckets

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "frames": self.frames,
            "tracked": self.tracked,
            "total_boxes": self.total_boxes,
            "unassigned_boxes": self.unassigned_boxes,
            "defects": self.total_defects,
            "confirmed": self.confirmed_defects,
            "damage_score": round(self.damage_score, 2),
            "grade": self.grade,
            "by_class": {c.canon: c.defects or c.boxes for c in self.by_class},
        }


# --------------------------------------------------------------------------- #
# Building a report
# --------------------------------------------------------------------------- #


def build_report(
    rows: Sequence[dict[str, Any]] | Iterable[dict[str, Any]],
    model_id: str = "",
    model_name: str = "",
    frames: int = 0,
) -> SurveyReport:
    """Collapse per-detection rows (`Detection.as_row`) into a survey.

    Untracked runs have no IDs to collapse on, so every box becomes its own
    one-frame defect. The per-class box counts stay honest; the unique-defect
    count doesn't, which is why the report carries `tracked`.
    """
    rows = list(rows)
    report = SurveyReport(
        model_id=model_id,
        model_name=model_name or model_id,
        frames=frames,
        tracked=any(r.get("track_id", -1) >= 0 for r in rows),
        total_boxes=len(rows),
    )
    if not rows:
        return report

    if not report.frames:
        report.frames = max(int(r.get("frame", 0)) for r in rows) + 1

    # ---- group by tracked defect (or by individual box when untracked)
    #
    # In a tracked run the tracker still emits boxes it hasn't confirmed into a
    # track yet. Those carry no ID, so they can't be collapsed -- and turning
    # each into its own "unique defect" would badly inflate the count (on a
    # 60-frame clip that turned 9 real potholes into 43). They're counted as
    # boxes and reported via `unassigned_boxes`, not as defects.
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for i, r in enumerate(rows):
        canon = str(r.get("damage") or tax.UNKNOWN_KEY)
        tid = int(r.get("track_id", -1))
        if tid < 0:
            if report.tracked:
                report.unassigned_boxes += 1
                continue
            # Untracked run: no IDs exist at all, so every box is its own
            # one-frame defect and the caller is told via `tracked`.
            key = (canon, -(i + 1))
        else:
            # An ID is only unique within a class: trackers reuse numbers
            # across classes, and two damage types sharing one must not merge.
            key = (canon, tid)
        groups.setdefault(key, []).append(r)

    for (canon, tid), grp in groups.items():
        confs = [float(g.get("conf", 0.0)) for g in grp]
        areas = [float(g.get("area_pct_frame", 0.0)) for g in grp]
        fr = [int(g.get("frame", 0)) for g in grp]
        report.defects.append(
            DefectRecord(
                track_id=tid if tid >= 0 else -1,
                canon=canon,
                frames=len(grp),
                first_frame=min(fr),
                last_frame=max(fr),
                mean_conf=sum(confs) / len(confs),
                max_conf=max(confs),
                peak_area_pct=max(areas) if areas else 0.0,
                mean_area_pct=(sum(areas) / len(areas)) if areas else 0.0,
            )
        )

    # ---- per-class rollup, worst damage type first
    per_class: dict[str, list[DefectRecord]] = {}
    box_counts: dict[str, int] = {}
    conf_sums: dict[str, float] = {}
    for r in rows:
        k = str(r.get("damage") or tax.UNKNOWN_KEY)
        box_counts[k] = box_counts.get(k, 0) + 1
        conf_sums[k] = conf_sums.get(k, 0.0) + float(r.get("conf", 0.0))
    for d in report.defects:
        per_class.setdefault(d.canon, []).append(d)

    for key in tax.sort_keys(box_counts):
        ds = per_class.get(key, [])
        n_boxes = box_counts.get(key, 0)
        report.by_class.append(
            ClassSummary(
                canon=key,
                defects=len(ds),
                boxes=n_boxes,
                # Averaged over boxes, not over defects, so this matches the
                # run's headline mean confidence.
                mean_conf=(conf_sums.get(key, 0.0) / n_boxes) if n_boxes else 0.0,
                peak_area_pct=max((d.peak_area_pct for d in ds), default=0.0),
                total_score=sum(d.weighted_score for d in ds),
            )
        )

    return report
