"""Headless runner -- same pipeline as the app, for batch jobs and benchmarks.

Examples
--------
  python run_cli.py --list
  python run_cli.py --inspect                       # real class names + class_map check
  python run_cli.py --model rdd-yolo12s --source data/videos/road.mp4
  python run_cli.py --model rdd-yolo12s --source road.jpg --conf 0.15
  python run_cli.py --model rdd-yolo12s -s road.mp4 --only pothole,alligator_crack
  python run_cli.py --benchmark --source data/videos/road.mp4 --max-frames 200
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import labcore  # noqa: F401  -- must precede cv2: quiets OpenCV/ffmpeg logging
import cv2

from labcore import taxonomy as tax
from labcore.detector import Detector, RunStats, device_label
from labcore.draw import Annotator, DrawOptions, draw_empty_notice
from labcore.registry import load_registry
from labcore.survey import build_report
from labcore.video import VideoSink, VideoSource, probe_video

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Road damage lab -- CLI runner")
    p.add_argument("--list", action="store_true", help="List registered models and exit")
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Load each model, print its real class names, and audit its class_map",
    )
    p.add_argument("--model", "-m", action="append", help="Model id (repeat to run several)")
    p.add_argument("--source", "-s", help="Video file, image file, or webcam index")
    p.add_argument(
        "--only",
        help="Comma-separated canonical damage types to keep, e.g. "
        "'pothole,alligator_crack'. Resolved per model, so it works across "
        "checkpoints with different class orders.",
    )
    p.add_argument(
        "--no-survey", action="store_true", help="Skip the road-condition summary"
    )
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--half", action="store_true")
    p.add_argument("--no-track", action="store_true", help="Disable persistent IDs")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--width", type=int, default=0, help="Downscale frames to this width")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--no-save", action="store_true", help="Don't write annotated media")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--show", action="store_true", help="Open an OpenCV preview window")
    p.add_argument("--benchmark", action="store_true", help="Run every enabled model")
    return p


def cmd_list() -> None:
    print(f"{'id':24} {'task':8} {'cached':7} {'damage types':46} origin")
    print("-" * 132)
    for s in load_registry():
        # Straight from models.yaml -- no weights are loaded, so this stays
        # instant even when nothing is cached yet.
        types = (
            ", ".join(tax.short_of(k) for k in tax.sort_keys(s.class_map.values()))
            if s.class_map
            else "(matched by name at load)"
        )
        print(f"{s.id:24} {s.task:8} {str(s.is_cached()):7} {types:46} {s.origin}")


def cmd_inspect(model_ids: list[str] | None = None, device: str = "cpu") -> int:
    """Print each checkpoint's real class names and audit its class_map.

    Model cards lie about class ordering often enough that every `class_map:`
    in models.yaml should be written from this, not from a README.
    """
    registry = load_registry()
    specs = [registry.get(m) for m in model_ids] if model_ids else list(registry)
    problems = 0

    for spec in specs:
        print(f"\n=== {spec.id} ===")
        print(f"    {spec.origin}")
        try:
            det = Detector(spec, device=device, fuse=False)
        except Exception as exc:
            print(f"    LOAD FAILED: {type(exc).__name__}: {exc}")
            problems += 1
            continue

        n_params = None
        try:
            n_params = sum(p.numel() for p in det.model.model.parameters())
        except Exception:
            pass
        print(
            f"    task={getattr(det.model, 'task', spec.task)}"
            + (f"  params={n_params / 1e6:.2f}M" if n_params else "")
            + f"  classes={len(det.names)}"
        )
        for cid in sorted(det.names):
            key = det.canon_by_id.get(cid, tax.UNKNOWN_KEY)
            src = "yaml" if (spec.class_map or {}).get(cid) else "name-match"
            flag = "  <-- UNMAPPED" if key == tax.UNKNOWN_KEY else ""
            print(f"      {cid:>3}  {det.names[cid]:<24} -> {key:<20} [{src}]{flag}")
            if key == tax.UNKNOWN_KEY and spec.id != "yolov8n-coco":
                problems += 1

        for w in det.map_warnings:
            print(f"    WARNING: {w}")
            problems += 1

    print(
        f"\n{len(specs)} model(s) inspected, {problems} issue(s)."
        if problems
        else f"\n{len(specs)} model(s) inspected, all classes mapped."
    )
    return 1 if problems else 0


def parse_only(raw: str | None) -> list[str] | None:
    """'pothole, alligator_crack' -> validated canonical keys."""
    if not raw:
        return None
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    bad = [k for k in keys if k not in tax.TAXONOMY]
    if bad:
        raise SystemExit(
            f"--only: unknown damage type(s) {bad}. Valid: {', '.join(tax.ORDER)}"
        )
    return keys


def print_survey(report, indent: str = "  ") -> None:
    """Compact road-condition summary for the terminal."""
    print(f"{indent}--- survey ---")
    print(f"{indent}{report.headline()}")
    if not report.defects:
        return
    print(
        f"{indent}damage score {report.damage_score:.2f} / 100 frames"
        + ("" if report.tracked else "   (untracked: counts are per-box)")
    )
    if report.unassigned_boxes:
        print(
            f"{indent}{report.unassigned_boxes} of {report.total_boxes} boxes "
            "never got a track ID and aren't counted as defects"
        )
    print(f"{indent}{'damage type':<22} {'defects':>8} {'boxes':>7} {'conf':>6} {'score':>8}")
    for c in report.by_class:
        print(
            f"{indent}{c.label:<22} {c.defects:>8} {c.boxes:>7} "
            f"{c.mean_conf:>6.2f} {c.total_score:>8.2f}"
        )
    worst = report.worst
    if worst is not None:
        print(
            f"{indent}worst: {worst.label} at frame {worst.first_frame} "
            f"(conf {worst.max_conf:.2f}, {worst.peak_area_pct:.1f}% of frame)"
        )


def run_image(det: Detector, path: Path, args, out_dir: Path) -> dict:
    frame = cv2.imread(str(path))
    if frame is None:
        raise IOError(f"Could not read image: {path}")
    if args.width and frame.shape[1] > args.width:
        s = args.width / frame.shape[1]
        frame = cv2.resize(frame, (args.width, int(frame.shape[0] * s)))

    H, W = frame.shape[:2]
    only = parse_only(args.only)
    res = det.infer(
        frame,
        conf=args.conf if args.conf is not None else det.spec.default_conf,
        iou=args.iou if args.iou is not None else det.spec.default_iou,
        imgsz=args.imgsz or det.spec.default_imgsz,
        track=False,
        classes=det.ids_for_canon(only) if only else None,
    )
    ann = Annotator(DrawOptions(show_trails=False))
    img = ann.draw(
        frame,
        res.detections,
        hud={
            "title": det.spec.name,
            "device": device_label(det.device, short=True),
            "in frame": str(len(res.detections)),
            "infer": f"{res.infer_ms:.1f} ms",
        },
    )
    if not res.detections:
        img = draw_empty_notice(img)

    if not args.no_save:
        out = out_dir / f"{path.stem}__{det.spec.id}.jpg"
        cv2.imwrite(str(out), img)
        print(f"  wrote {out}")

    for d in res.detections:
        print(
            f"    {d.canon_label:<22} (raw: {d.label:<20}) conf={d.conf:.3f}  "
            f"box={tuple(round(v) for v in d.xyxy)}"
        )

    rows = [d.as_row(0, W, H) for d in res.detections]
    per_class: dict[str, int] = {}
    for d in res.detections:
        per_class[d.canon] = per_class.get(d.canon, 0) + 1

    if rows and not args.no_survey:
        print_survey(
            build_report(rows, det.spec.id, det.spec.name, frames=1), indent="    "
        )

    return {
        "detections": len(res.detections),
        "infer_ms": round(res.infer_ms, 2),
        "per_class": per_class,
    }


def run_video(det: Detector, source, args, out_dir: Path) -> dict:
    resize = None
    if isinstance(source, str) and args.width:
        info = probe_video(source)
        if info.width > args.width:
            scale = args.width / info.width
            resize = (args.width, int(round(info.height * scale / 2) * 2))

    stem = Path(str(source)).stem if isinstance(source, str) else f"cam{source}"
    det.reset_tracker()
    ann = Annotator(DrawOptions())
    stats = RunStats()
    rows: list[dict] = []

    conf = args.conf if args.conf is not None else det.spec.default_conf
    iou = args.iou if args.iou is not None else det.spec.default_iou
    imgsz = args.imgsz or det.spec.default_imgsz
    only = parse_only(args.only)
    class_ids = det.ids_for_canon(only) if only else None
    if only and not class_ids:
        print(
            f"  note: {det.spec.id} has no classes matching {only} -- "
            "it will report nothing."
        )

    sink: VideoSink | None = None
    t0 = time.perf_counter()

    src = VideoSource(
        source,
        stride=args.stride,
        max_frames=args.max_frames or None,
        resize_to=resize,
    )
    try:
        with src:
            total = src.planned_frames
            out_fps = max(1.0, (src.info.fps if src.info else 30.0) / max(1, args.stride))

            for idx, frame in src:
                H, W = frame.shape[:2]
                res = det.infer(
                    frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    track=not args.no_track,
                    tracker=args.tracker,
                    classes=class_ids,
                )
                stats.update(res.detections, res.infer_ms, W, H)
                rows.extend(d.as_row(idx, W, H) for d in res.detections)

                img = ann.draw(
                    frame,
                    res.detections,
                    hud={
                        "title": det.spec.name,
                        "device": device_label(det.device, short=True),
                        "frame": f"{idx}" + (f" / {total}" if total else ""),
                        "in frame": str(len(res.detections)),
                        "unique IDs": str(len(stats.unique_ids)),
                        "infer": f"{res.infer_ms:.1f} ms",
                        "fps": f"{stats.fps:.1f}",
                        "conf": f"{conf:.2f}",
                        "imgsz": str(imgsz),
                    },
                )
                if not res.detections:
                    img = draw_empty_notice(img)

                if not args.no_save:
                    if sink is None:
                        sink = VideoSink(
                            out_dir / f"{stem}__{det.spec.id}.mp4",
                            out_fps,
                            (img.shape[1], img.shape[0]),
                        )
                        sink.__enter__()
                    sink.write(img)

                if args.show:
                    cv2.imshow(f"lab · {det.spec.id}", img)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

                if stats.frames % 25 == 0:
                    pct = f"{100 * stats.frames / total:5.1f}%" if total else "  --  "
                    live = " ".join(
                        f"{tax.short_of(k)}={v}"
                        for k, v in stats.unique_counts().items()
                    ) or "no finds"
                    sys.stdout.write(
                        f"\r  {pct}  frame {stats.frames}  "
                        f"dets {stats.total_dets}  {live}  "
                        f"{stats.fps:.1f} fps          "
                    )
                    sys.stdout.flush()
    finally:
        if sink is not None:
            sink.__exit__(None, None, None)
        if args.show:
            cv2.destroyAllWindows()

    wall = time.perf_counter() - t0
    print()
    summary = stats.summary()
    summary["wall_s"] = round(wall, 2)
    print(
        f"  {stats.frames} frames, {stats.total_dets} boxes, "
        f"mean conf {stats.mean_conf:.2f}, {stats.mean_ms:.1f} ms/frame "
        f"({stats.fps:.1f} fps), wall {wall:.1f}s"
    )

    report = build_report(rows, det.spec.id, det.spec.name, frames=stats.frames)
    summary["grade"] = report.grade
    summary["damage_score"] = round(report.damage_score, 2)
    if not args.no_survey:
        print_survey(report)

    if not args.no_save and sink is not None:
        print(f"  wrote {out_dir / f'{stem}__{det.spec.id}.mp4'} ({sink.fourcc_used})")

    if rows and not args.no_csv:
        csv_path = out_dir / f"{stem}__{det.spec.id}_detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {csv_path}")

        # The per-defect table is what a survey actually consumes -- one row
        # per tracked defect rather than one per frame it appeared in.
        defect_rows = report.defect_rows()
        if defect_rows:
            dpath = out_dir / f"{stem}__{det.spec.id}_defects.csv"
            with open(dpath, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(defect_rows[0]))
                w.writeheader()
                w.writerows(defect_rows)
            print(f"  wrote {dpath}")

    return summary


def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        cmd_list()
        return 0

    if args.inspect:
        return cmd_inspect(args.model, device=args.device)

    if not args.source:
        build_parser().print_help()
        return 2

    registry = load_registry()
    if args.benchmark:
        ids = [s.id for s in registry]
    elif args.model:
        ids = args.model
    else:
        ids = [registry.specs[0].id]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source: str | int = args.source
    if str(args.source).isdigit():
        source = int(args.source)
    is_image = isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS

    results: dict[str, dict] = {}
    for mid in ids:
        spec = registry.get(mid)
        print(f"\n=== {spec.name}  ({spec.origin}) ===")
        det = Detector(spec, device=args.device, half=args.half)
        det.warmup(args.imgsz or spec.default_imgsz)
        if is_image:
            results[mid] = run_image(det, Path(source), args, out_dir)
        else:
            results[mid] = run_video(det, source, args, out_dir)

    if len(results) > 1:
        _print_comparison(results)

    return 0


def _print_comparison(results: dict[str, dict]) -> None:
    """Two tables: headline metrics, then unique defects per damage type.

    The per-class table is the one that matters now -- a model can win on
    total detections purely by over-firing on longitudinal cracks.
    """
    print("\n=== comparison ===")
    headline = [
        ("frames", "frames"),
        ("detections", "boxes"),
        ("mean_conf", "conf"),
        ("mean_infer_ms", "ms"),
        ("fps", "fps"),
        ("damage_score", "score"),
        ("grade", "grade"),
    ]
    keys = [k for k, _ in headline if any(k in v for v in results.values())]
    heads = dict(headline)
    print(f"{'model':28} " + " ".join(f"{heads[k]:>12}" for k in keys))
    print("-" * (28 + 13 * len(keys)))
    for mid, v in results.items():
        print(
            f"{mid:28} "
            + " ".join(f"{str(v.get(k, '-'))[:12]:>12}" for k in keys)
        )

    # ---- per damage type
    seen: set[str] = set()
    for v in results.values():
        seen.update((v.get("unique_by_class") or v.get("per_class") or {}))
    cols = tax.sort_keys(seen)
    if not cols:
        return

    print(f"\n--- unique defects per damage type ---")
    print(f"{'model':28} " + " ".join(f"{tax.short_of(c):>13}" for c in cols))
    print("-" * (28 + 14 * len(cols)))
    for mid, v in results.items():
        per = v.get("unique_by_class") or v.get("per_class") or {}
        print(
            f"{mid:28} "
            + " ".join(f"{per.get(c, 0):>13}" for c in cols)
        )
    print(
        "\nCounts are unique tracked defects where tracking ran, raw boxes "
        "otherwise. Only comparable across models on the same clip and stride."
    )


if __name__ == "__main__":
    raise SystemExit(main())
