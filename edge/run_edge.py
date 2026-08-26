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

from edgecore.config import EdgeConfig
from edgecore.gps import RouteReplay, TrackReplay, get_route, parse_when
from edgecore.pipeline import Pipeline, PipelineConfig
from edgecore.publisher import EventPublisher

EDGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = EDGE_DIR / "runs"


def _clear_line() -> None:
    """Wipe the in-place progress line so an event line doesn't land on top."""
    sys.stdout.write("\r" + " " * 78 + "\r")


def build_gps(args) -> object | None:
    """A position source for this pass, or None if no route was given."""
    if args.track_file:
        return TrackReplay(args.track_file, start_time=parse_when(args.at))
    if not args.route:
        return None
    return RouteReplay(
        get_route(args.route),
        speed_kmh=args.speed,
        start_offset_m=args.start_offset,
        start_time=parse_when(args.at),
        gps_noise_m=args.gps_noise,
        speed_jitter=args.speed_jitter,
        seed=args.seed,
    )


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

    g = p.add_argument_group("position and time (simulated)")
    g.add_argument("--speed", type=float, default=None, help="Override the route's nominal km/h")
    g.add_argument(
        "--start-offset",
        type=float,
        default=0.0,
        help="Metres along the route where this pass begins. A 10 s clip covers "
        "only ~85 m, so offsets are how different passes cover different "
        "stretches of the same corridor.",
    )
    g.add_argument(
        "--gps-noise",
        type=float,
        default=0.0,
        help="Metres of Gaussian scatter per fix, simulating urban GPS drift",
    )
    g.add_argument(
        "--speed-jitter",
        type=float,
        default=0.0,
        help="Fractional speed variation for this pass, e.g. 0.1 for +/-10%%",
    )
    g.add_argument(
        "--at",
        default=None,
        help="Backdate the pass. Must use the '=' form because the value looks "
        "like a flag: --at=-2h, --at=-1d, or --at=2026-08-20T09:15",
    )
    g.add_argument("--track-file", default=None, help="Replay a real GPX/CSV track instead of simulating")
    g.add_argument("--seed", type=int, default=None, help="Seed the noise/jitter RNG for a repeatable pass")

    p.add_argument("--min-frames", type=int, default=None, help="Sightings before a track counts")
    p.add_argument("--miss-tolerance", type=int, default=None, help="Frames unseen before a track closes")

    n = p.add_argument_group("central system")
    n.add_argument("--post", action="store_true", help="POST events to the central system as they fire")
    n.add_argument("--api", default=None, help="API base URL (default $ROADSURVEY_API_URL or localhost:8000)")
    n.add_argument("--no-post-crops", action="store_true", help="Post events without their crops")

    p.add_argument("--out", default=None, help="Output directory (default edge/runs/<clip>__<bus>)")
    p.add_argument("--no-crops", action="store_true", help="Don't extract or write crops")
    p.add_argument("--quiet", "-q", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # Phase selects which frames of each stride group this pass sees, so it only
    # means anything when there is more than one to choose from. At stride 1
    # every phase samples the same frames and two "different" passes come back
    # bit-identical -- which silently makes confirmation counts meaningless.
    if args.phase and args.stride < 2:
        print(
            f"  warning: --phase {args.phase} has no effect at --stride 1 -- every "
            "phase sees the same frames.\n"
            "           Use --stride 2 or 3 to make passes detect differently.",
            file=sys.stderr,
        )
    elif args.phase >= args.stride > 1:
        print(
            f"  warning: --phase {args.phase} wraps at --stride {args.stride}; "
            f"it behaves as phase {args.phase % args.stride}.",
            file=sys.stderr,
        )

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

    gps = build_gps(args)

    pipe = Pipeline(cfg, gps=gps)
    if not args.quiet:
        d = pipe.describe()
        print(f"=== {d['model_name']}  on {d['device']} ===")
        print(f"    {args.bus}" + (f" · route {args.route}" if args.route else ""))
        print(f"    conf {d['conf']}  stride {args.stride}  phase {args.phase}")
        if gps is not None:
            g = gps.describe()
            if "route_id" in g:
                print(
                    f"    {g['route_name']}  ({g['route_length_m']:.0f} m)  "
                    f"{g['speed_kmh']} km/h from +{g['start_offset_m']:.0f} m"
                )
            print(f"    clock starts {g.get('start_time', '-')}")
        else:
            print("    no route -- events will have no position")

    pub: EventPublisher | None = None
    if args.post:
        econf = EdgeConfig.from_env().merge_cli(api_url=args.api)
        pub = EventPublisher(
            api_url=econf.api_url,
            spool_dir=econf.spool_dir,
            batch_size=econf.batch_size,
            timeout_s=econf.api_timeout_s,
            include_crops=not args.no_post_crops,
        )
        reachable = pub.ping()
        if not args.quiet:
            state = "reachable" if reachable else "NOT reachable — events will spool"
            print(f"    posting to {econf.api_url}  ({state})")
        pub.start()

    events = []
    for upd in pipe.run(source):
        if pub is not None and upd.new_events:
            pub.publish(upd.new_events)
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

    if pub is not None:
        pub.close()  # flush the queue; anything unsent lands in the spool

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

    # Drop this straight into geojson.io to check the fixes land on real road.
    located = [e for e in events if e.has_fix]
    if located:
        features = [
            {
                "type": "Feature",
                "properties": {
                    k: v
                    for k, v in e.to_dict().items()
                    if k not in ("lat", "lon", "bbox")
                },
                "geometry": {"type": "Point", "coordinates": [e.lon, e.lat]},
            }
            for e in located
        ]
        if gps is not None and hasattr(gps, "route"):
            features.append(gps.route.as_geojson())
        (out_dir / "events.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
            encoding="utf-8",
        )

    stats = pipe.stats()
    wire = sum(e.wire_bytes for e in events)
    stats["wire_bytes"] = wire
    if located and gps is not None and hasattr(gps, "span_m"):
        stats["road_covered_m"] = round(gps.span_m(stats.get("steps", 0) * args.stride), 1)
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "config": vars(args),
                "gps": gps.describe() if gps is not None else None,
                "stats": stats,
            },
            indent=2,
            default=str,
        ),
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
        if pub is not None:
            s = pub.stats
            print(
                f"  posted {s.posted}, {s.duplicates} duplicate, "
                f"{s.drained} drained from spool, {s.bytes_sent / 1024:.0f} KB sent"
            )
            if pub.spool_depth:
                print(
                    f"  {pub.spool_depth} batch(es) still spooled at {pub.spool_dir} "
                    "— they'll drain on the next run that connects"
                )
            if s.last_error:
                print(f"  last error: {s.last_error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
