"""Headless runner -- same pipeline as the app, for batch jobs and benchmarks.

Examples
--------
  python run_cli.py --list
  python run_cli.py --model samdutse-yolov8 --source data/videos/road.mp4
  python run_cli.py --model samdutse-yolov8 --source road.jpg --conf 0.3
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

from labcore.detector import Detector, RunStats, device_label
from labcore.draw import Annotator, DrawOptions, draw_empty_notice
from labcore.registry import load_registry
from labcore.video import VideoSink, VideoSource, probe_video

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pothole detection lab -- CLI runner")
    p.add_argument("--list", action="store_true", help="List registered models and exit")
    p.add_argument("--model", "-m", action="append", help="Model id (repeat to run several)")
    p.add_argument("--source", "-s", help="Video file, image file, or webcam index")
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
    print(f"{'id':28} {'task':9} {'cached':7} origin")
    print("-" * 90)
    for s in load_registry():
        print(f"{s.id:28} {s.task:9} {str(s.is_cached()):7} {s.origin}")


def run_image(det: Detector, path: Path, args, out_dir: Path) -> dict:
    frame = cv2.imread(str(path))
    if frame is None:
        raise IOError(f"Could not read image: {path}")
    if args.width and frame.shape[1] > args.width:
        s = args.width / frame.shape[1]
        frame = cv2.resize(frame, (args.width, int(frame.shape[0] * s)))

    res = det.infer(
        frame,
        conf=args.conf if args.conf is not None else det.spec.default_conf,
        iou=args.iou if args.iou is not None else det.spec.default_iou,
        imgsz=args.imgsz or det.spec.default_imgsz,
        track=False,
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
        print(f"    {d.label:12} conf={d.conf:.3f}  box={tuple(round(v) for v in d.xyxy)}")

    return {"detections": len(res.detections), "infer_ms": round(res.infer_ms, 2)}


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
                )
                stats.update(res.detections, res.infer_ms)
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
                    sys.stdout.write(
                        f"\r  {pct}  frame {stats.frames}  "
                        f"dets {stats.total_dets}  ids {len(stats.unique_ids)}  "
                        f"{stats.fps:.1f} fps"
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
    print(f"  {summary}")

    if not args.no_save and sink is not None:
        print(f"  wrote {out_dir / f'{stem}__{det.spec.id}.mp4'} ({sink.fourcc_used})")

    if rows and not args.no_csv:
        csv_path = out_dir / f"{stem}__{det.spec.id}_detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {csv_path}")

    return summary


def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        cmd_list()
        return 0

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
        print("\n=== comparison ===")
        keys = sorted({k for v in results.values() for k in v if k != "per_class"})
        print(f"{'model':28} " + " ".join(f"{k:>16}" for k in keys))
        for mid, v in results.items():
            print(f"{mid:28} " + " ".join(f"{str(v.get(k, '')):>16}" for k in keys))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
