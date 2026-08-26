"""Headless onboard agent -- replay a clip as if it were a bus on a route.

The Streamlit app (`app_edge.py`) is the demo front door; this is its twin, for
seeding fleet history before a demo and for batch runs during development.

    python run_edge.py -s ../road-damage-lab/data/videos/62_10-07-2023.mp4
    python run_edge.py -s clip.mp4 --bus BUS_002 --phase 1 --out runs/pass2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edgecore.pipeline import Pipeline, PipelineConfig

EDGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = EDGE_DIR / "runs"


def _clear_line() -> None:
    """Wipe the in-place progress line so an event line doesn't land on top."""
    sys.stdout.write("\r" + " " * 78 + "\r")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Onboard agent -- headless runner")
    p.add_argument("--source", "-s", required=True, help="Video file or webcam index")
    p.add_argument("--model", "-m", default="rdd-yolo12s")
    p.add_argument("--bus", default="BUS_001", help="Simulated bus id")
    p.add_argument("--route", default="", help="Route id (see edge/routes/)")

    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--half", action="store_true")
    p.add_argument("--no-track", action="store_true")
    p.add_argument("--only", help="Comma-separated canonical damage types to keep")

    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--phase",
        type=int,
        default=0,
        help="Start-frame offset. Two passes at different phases sample "
        "different frames of the same clip, so they genuinely detect "
        "differently -- that's what makes repeat passes informative.",
    )
    p.add_argument("--width", type=int, default=0, help="Downscale frames to this width")
    p.add_argument("--max-frames", type=int, default=0)

    p.add_argument("--min-frames", type=int, default=None, help="Sightings before a track counts")
    p.add_argument("--miss-tolerance", type=int, default=None, help="Frames unseen before a track closes")

    p.add_argument("--out", default=None, help="Output directory (default edge/runs/<clip>__<bus>)")
    p.add_argument("--no-crops", action="store_true", help="Don't extract or write crops")
    p.add_argument("--quiet", "-q", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    source: str | int = int(args.source) if str(args.source).isdigit() else args.source
    clip_stem = Path(str(source)).stem if not isinstance(source, int) else f"cam{source}"

    out_dir = Path(args.out) if args.out else DEFAULT_OUT / f"{clip_stem}__{args.bus}"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = out_dir / "crops"

    cfg = PipelineConfig(
        model_id=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        track=not args.no_track,
        only=[k.strip() for k in args.only.split(",")] if args.only else None,
        stride=args.stride,
        phase=args.phase,
        width=args.width,
        max_frames=args.max_frames,
        bus_id=args.bus,
        route_id=args.route,
        capture_crops=not args.no_crops,
        annotate=False,  # nothing to show; skip the draw cost
        **{
            k: v
            for k, v in (
                ("min_frames", args.min_frames),
                ("miss_tolerance", args.miss_tolerance),
            )
            if v is not None
        },
    )

    pipe = Pipeline(cfg)
    if not args.quiet:
        d = pipe.describe()
        print(f"=== {d['model_name']}  on {d['device']} ===")
        print(f"    {args.bus}" + (f" · route {args.route}" if args.route else ""))
        print(f"    conf {d['conf']}  stride {args.stride}  phase {args.phase}")

    events = []
    for upd in pipe.run(source):
        for ev in upd.new_events:
            events.append(ev)
            if not args.quiet:
                where = (
                    f"{ev.lat:.5f},{ev.lon:.5f}" if ev.has_fix else "no fix"
                )
                _clear_line()
                print(
                    f"  [{len(events):3d}] {ev.damage_label:<20} "
                    f"conf {ev.confidence:.2f}  {ev.area_pct_frame:5.2f}% frame  "
                    f"frames {ev.frames_seen:3d}  @{ev.frame_idx:<5d} {where}"
                )
        if not args.quiet and upd.step % 25 == 0 and upd.frame_idx >= 0:
            pct = f"{100 * upd.progress:5.1f}%" if upd.total_steps else "  --  "
            sys.stdout.write(
                f"\r  {pct}  frame {upd.step}  events {len(events)}  "
                f"{upd.fps:5.1f} fps          "
            )
            sys.stdout.flush()

    if not args.quiet:
        print()

    # ---- write events + crops
    if not args.no_crops and any(e.crop_jpeg for e in events):
        crop_dir.mkdir(exist_ok=True)
        for ev in events:
            if ev.crop_jpeg:
                (crop_dir / f"{ev.event_uid}.jpg").write_bytes(ev.crop_jpeg)

    payload = [e.to_dict() for e in events]
    (out_dir / "events.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    stats = pipe.stats()
    wire = sum(e.wire_bytes for e in events)
    stats["wire_bytes"] = wire
    (out_dir / "run.json").write_text(
        json.dumps({"config": vars(args), "stats": stats}, indent=2, default=str),
        encoding="utf-8",
    )

    if not args.quiet:
        print(f"  {stats['steps']} frames processed, {stats['boxes_seen']} boxes")
        print(
            f"  {stats['tracks_opened']} tracks opened, "
            f"{stats['tracks_dropped_as_flicker']} dropped as flicker, "
            f"{stats['events_emitted']} events emitted"
        )
        if stats["boxes_unassigned"]:
            print(
                f"  {stats['boxes_unassigned']} boxes never got a track ID "
                "(not counted as events)"
            )
        print(f"  wrote {out_dir / 'events.json'}  ({len(events)} events, {wire / 1024:.0f} KB on the wire)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
