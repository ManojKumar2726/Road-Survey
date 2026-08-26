"""Get events off the bus and into the central system.

A bus loses signal -- under a flyover, in a depot, on a stretch with no
coverage. So the publisher never treats a failed POST as a lost event: it
writes the batch to a disk spool and drains it on the next successful call.
That is a small amount of code and the honest answer to "what happens in a
tunnel", which is the first question anyone asks about an onboard system.

Posting happens on a worker thread. Inference is the expensive thing in the
loop and must not wait on a network round trip.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .events import RoadEvent


@dataclass
class PublishStats:
    """What actually left the bus. Drives the bandwidth readout in window 1."""

    posted: int = 0
    duplicates: int = 0
    rejected: int = 0
    failed_batches: int = 0
    spooled: int = 0
    drained: int = 0
    bytes_sent: int = 0
    last_error: str = ""
    connected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "posted": self.posted,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "spooled": self.spooled,
            "drained": self.drained,
            "failed_batches": self.failed_batches,
            "bytes_sent": self.bytes_sent,
            "kb_sent": round(self.bytes_sent / 1024, 1),
            "connected": self.connected,
            "last_error": self.last_error,
        }


class EventPublisher:
    """POSTs events to the central system, spooling whatever won't go.

    Use it as a context manager so the worker is joined and the queue flushed:

        with EventPublisher(api_url, spool_dir) as pub:
            pub.publish(events)
    """

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000",
        spool_dir: str | Path = "spool",
        batch_size: int = 1,
        timeout_s: float = 5.0,
        include_crops: bool = True,
        enabled: bool = True,
    ) -> None:
        self.endpoint = api_url.rstrip("/") + "/api/events"
        self.spool_dir = Path(spool_dir)
        self.batch_size = max(1, int(batch_size))
        self.timeout_s = float(timeout_s)
        self.include_crops = include_crops
        self.enabled = enabled

        self.stats = PublishStats()
        self._q: queue.Queue = queue.Queue()
        self._pending: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> "EventPublisher":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if not self.enabled or self._worker is not None:
            return
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._worker = threading.Thread(
            target=self._run, name="event-publisher", daemon=True
        )
        self._worker.start()

    def close(self, timeout: float = 15.0) -> None:
        """Flush the queue, then stop. Anything still unsent is spooled."""
        if self._worker is None:
            return
        self._q.put(None)  # sentinel: flush and exit
        self._worker.join(timeout=timeout)
        self._worker = None

    # ---------------------------------------------------------------- submit

    def publish(self, events: Iterable[RoadEvent]) -> None:
        """Hand events to the worker. Never blocks the inference loop."""
        for ev in events:
            body = ev.to_dict(include_crop=self.include_crops)
            if not self.enabled:
                self._spool([body])
                continue
            self._q.put(body)

    # ---------------------------------------------------------------- worker

    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                self._maybe_send(force=False)
                continue

            if item is None:  # shutdown
                self._maybe_send(force=True)
                break
            self._pending.append(item)
            self._maybe_send(force=False)

    def _maybe_send(self, force: bool) -> None:
        with self._lock:
            if not self._pending:
                return
            if not force and len(self._pending) < self.batch_size:
                return
            batch, self._pending = self._pending, []
        self._send(batch)

    # ------------------------------------------------------------------ wire

    def _send(self, batch: Sequence[dict[str, Any]]) -> None:
        if not batch:
            return
        payload = json.dumps({"events": list(batch)}).encode("utf-8")
        try:
            result = self._post(payload)
        except Exception as exc:
            self.stats.connected = False
            self.stats.failed_batches += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            self._spool(batch)
            return

        self.stats.connected = True
        self.stats.last_error = ""
        self.stats.bytes_sent += len(payload)
        self.stats.posted += int(result.get("accepted", 0))
        self.stats.duplicates += int(result.get("duplicates", 0))
        self.stats.rejected += int(result.get("rejected", 0))
        # Only drain once something has actually got through -- otherwise a
        # long outage would retry the whole backlog on every batch.
        self._drain_spool()

    def _post(self, payload: bytes) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")

    # ----------------------------------------------------------------- spool

    def _spool(self, batch: Sequence[dict[str, Any]]) -> None:
        """Park a batch on disk. Named by time so it drains in order."""
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        name = f"{time.time():.6f}_{uuid.uuid4().hex[:8]}.json"
        (self.spool_dir / name).write_text(
            json.dumps(list(batch)), encoding="utf-8"
        )
        self.stats.spooled += len(batch)

    def _drain_spool(self, max_files: int = 20) -> None:
        files = sorted(self.spool_dir.glob("*.json"))[:max_files]
        for f in files:
            try:
                batch = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                f.unlink(missing_ok=True)  # unreadable spool helps nobody
                continue
            try:
                result = self._post(json.dumps({"events": batch}).encode("utf-8"))
            except Exception:
                return  # still down; leave the rest for next time
            self.stats.posted += int(result.get("accepted", 0))
            self.stats.duplicates += int(result.get("duplicates", 0))
            self.stats.drained += len(batch)
            self.stats.spooled = max(0, self.stats.spooled - len(batch))
            f.unlink(missing_ok=True)

    # ------------------------------------------------------------------ meta

    @property
    def spool_depth(self) -> int:
        return len(list(self.spool_dir.glob("*.json"))) if self.spool_dir.is_dir() else 0

    def ping(self) -> bool:
        """Is the central system reachable right now?"""
        import urllib.request

        try:
            url = self.endpoint.rsplit("/api/", 1)[0] + "/api/health"
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                self.stats.connected = resp.status == 200
        except Exception:
            self.stats.connected = False
        return self.stats.connected
