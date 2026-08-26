"""Populate the central system with plausible fleet history before a demo.

An empty map is a bad opening slide. This drives several buses over several
routes on several days, so window 2 starts out looking like a fleet has been
working -- and, more importantly, so the live pass in window 1 lands on defects
that are *already there* and visibly bumps their confirmation counts.

    python seed_demo.py --api http://127.0.0.1:8000

Passes run at stride 3 with different phases, which is what makes them detect
differently: at stride 1 every phase samples the same frames and the passes
come back identical, making confirmation counts meaningless. See V1-Plan.md.

The model is loaded once and reused across every pass -- reloading per pass
dominates the runtime otherwise.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

EDGE_DIR = Path(__file__).resolve().parent
if str(EDGE_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_DIR))

from edgecore.config import EdgeConfig
from edgecore.gps import RouteReplay, get_route, parse_when
from edgecore.pipeline import Pipeline, PipelineConfig
from edgecore.publisher import EventPublisher

CLIPS = EDGE_DIR.parent / "road-damage-lab" / "data" / "videos"

# (clip, route, start offset m, [(bus, phase, backdate), ...])
#
# Offsets differ per pass group so the fleet covers several stretches of each
# corridor rather than stacking every finding on one 85 m segment. Within a
# group the offset is identical -- that's what lets clustering recognise the
# same defect across passes.
PLAN = [
    ("62_10-07-2023.mp4", "route-21g", 900, [
        ("BUS_101", 0, "-6d"), ("BUS_102", 1, "-4d"), ("BUS_103", 2, "-2d"),
    ]),
    ("62_10-07-2023.mp4", "route-21g", 2600, [
        ("BUS_101", 1, "-5d"), ("BUS_104", 2, "-3d"),
    ]),
    ("138_10-07-2023.mp4", "route-570", 1400, [
        ("BUS_201", 0, "-5d"), ("BUS_202", 1, "-3d"), ("BUS_203", 2, "-1d"),
    ]),
    ("138_10-07-2023.mp4", "route-570", 3300, [
        ("BUS_202", 0, "-2d"), ("BUS_203", 1, "-8h"),
    ]),
    ("mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4", "route-51m", 800, [
        ("BUS_301", 0, "-7d"), ("BUS_302", 1, "-4d"), ("BUS_303", 2, "-1d"),
    ]),
    ("mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4", "route-51m", 3000, [
        ("BUS_301", 2, "-3d"), ("BUS_303", 0, "-5h"),
    ]),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Seed demo fleet history")
    p.add_argument("--api", default=None, help="API base URL")
    p.add_argument("--model", default="rdd-yolo12s")
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--gps-noise", type=float, default=6.0)
    p.add_argument("--speed-jitter", type=float, default=0.12)
    p.add_argument("--dry-run", action="store_true", help="Process but don't post")
    args = p.parse_args()

    conf = EdgeConfig.from_env().merge_cli(api_url=args.api)

    missing = {c for c, *_ in PLAN if not (CLIPS / c).is_file()}
    if missing:
        print(f"  missing clips in {CLIPS}:")
        for m in sorted(missing):
            print(f"    {m}")
        return 1

    pub = None
    if not args.dry_run:
        pub = EventPublisher(
            api_url=conf.api_url, spool_dir=conf.spool_dir, batch_size=8
        )
        if not pub.ping():
            print(f"  {conf.api_url} is not reachable — seeded events would spool.")
            return 2
        pub.start()
        print(f"  posting to {conf.api_url}")

    cfg = PipelineConfig(model_id=args.model, stride=args.stride, annotate=False)
    print(f"  loading {args.model} once for every pass…")
    pipe = Pipeline(cfg, gps=None)
    print(f"  {pipe.describe()['device']}\n")

    total_events = 0
    total_passes = sum(len(g) for *_, g in PLAN)
    n = 0

    for clip, route_id, offset, group in PLAN:
        route = get_route(route_id)
        for bus, phase, when in group:
            n += 1
            # run() reads these off cfg at call time, so mutating in place
            # reuses the loaded model rather than reloading it per pass.
            cfg.bus_id = bus
            cfg.route_id = route_id
            cfg.phase = phase
            pipe.gps = RouteReplay(
                route,
                start_offset_m=offset,
                start_time=parse_when(when),
                gps_noise_m=args.gps_noise,
                speed_jitter=args.speed_jitter,
                seed=hash((bus, phase, offset)) % 10_000,
            )

            events = []
            for upd in pipe.run(str(CLIPS / clip)):
                if pub is not None and upd.new_events:
                    pub.publish(upd.new_events)
                events.extend(upd.new_events)
            total_events += len(events)

            print(
                f"  [{n:2d}/{total_passes}] {bus:<8} {route_id:<10} "
                f"+{offset:>5}m  phase {phase}  {when:>5}  "
                f"-> {len(events):2d} events   ({clip[:28]})"
            )

    if pub is not None:
        pub.close()
        s = pub.stats
        print(
            f"\n  {total_events} events from {total_passes} passes; "
            f"posted {s.posted}, {s.duplicates} duplicate, "
            f"{s.bytes_sent / 1024:.0f} KB sent"
        )
        if pub.spool_depth:
            print(f"  {pub.spool_depth} batch(es) spooled — server went away mid-seed")
    else:
        print(f"\n  {total_events} events from {total_passes} passes (dry run)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
