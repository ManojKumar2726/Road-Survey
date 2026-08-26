"""Start the central system on the port the onboard unit expects.

    python run.py                 # clean slate, http://127.0.0.1:8010
    python run.py --keep          # keep what's already reported
    python run.py --port 8000     # if you'd rather use the conventional port
    python run.py --reload        # develop against it

**Starts empty by default.** Test passes accumulate into a map nobody can read,
so every start clears reported data -- sightings, the defects clustered from
them, the bus roster and the stored crops. Routes survive: they're
configuration, re-seeded from `edge/routes/` on every startup.

The consequence worth knowing: seeded demo history is reported data too, so
seed *after* starting, or start with `--keep` on demo day.

Exists so both halves of the project agree on a port without anyone passing a
flag. Uvicorn defaults to 8000, which is a crowded port -- if something else is
already listening there the onboard unit posts its events into whatever that
happens to be, and the failure looks like a network problem rather than a
misconfiguration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

DEFAULT_PORT = 8010


def port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.6)
        return s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0


def clear_reported() -> None:
    """Drop reported data so the map starts empty."""
    from app.db import Base, SessionLocal, engine, init_dirs
    from app.reset import reset_data, summarise

    init_dirs()
    Base.metadata.create_all(engine)  # first run has nothing to clear yet
    with SessionLocal() as db:
        counts = reset_data(db)

    if any(counts.values()):
        print(f"  cleared: {summarise(counts)}")
    else:
        print("  starting empty")


def main() -> int:
    p = argparse.ArgumentParser(description="Road Survey — central system")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--reload", action="store_true", help="Restart on code changes")
    p.add_argument(
        "--keep",
        action="store_true",
        help="Keep previously reported sightings and defects. Use this on demo "
        "day so seeded fleet history survives a restart.",
    )
    args = p.parse_args()

    if port_in_use(args.host, args.port):
        print(
            f"  Port {args.port} is already in use.\n"
            f"  Something else is listening there -- start on another port with\n"
            f"      python run.py --port {args.port + 1}\n"
            f"  and set the same URL in the onboard app's sidebar.",
            file=sys.stderr,
        )
        return 1

    # Wipe here rather than in the app's lifespan: with --reload, uvicorn
    # restarts the app on every code change, and clearing the database each
    # time you saved a file would be maddening.
    if not args.keep:
        clear_reported()
    else:
        print("  --keep: leaving previously reported data in place")

    import uvicorn

    print(f"\n  Road Survey — control room on http://{args.host}:{args.port}")
    print(f"  Onboard units should post to http://127.0.0.1:{args.port}\n")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(SERVER_DIR / "app")] if args.reload else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
