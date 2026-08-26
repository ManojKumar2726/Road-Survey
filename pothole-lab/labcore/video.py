"""Video I/O helpers -- frame iteration with stride/limit, and encoding output."""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@contextlib.contextmanager
def _quiet_native_stderr():
    """Mute C-level stderr.

    Probing codecs makes OpenCV's ffmpeg plugin complain about encoders that
    aren't available on this machine. Those messages come from native code in a
    separate plugin, so neither `cv2.setLogLevel` nor Python's `sys.stderr`
    reaches them -- only redirecting file descriptor 2 does.
    """
    try:
        saved = os.dup(2)
    except (OSError, ValueError, AttributeError):
        yield  # no real fd to redirect (notebook, embedded interpreter, ...)
        return

    devnull = None
    try:
        sys.stderr.flush()
    except Exception:
        pass
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            os.dup2(saved, 2)
        finally:
            os.close(saved)
            if devnull is not None:
                os.close(devnull)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def probe_video(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return VideoInfo(
            path=str(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 0 else 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
    finally:
        cap.release()


class VideoSource:
    """Iterate `(frame_index, frame)` from a file, webcam index, or stream URL."""

    def __init__(
        self,
        source: str | int | Path,
        stride: int = 1,
        max_frames: int | None = None,
        start_frame: int = 0,
        resize_to: tuple[int, int] | None = None,
    ) -> None:
        self.source = source if isinstance(source, int) else str(source)
        self.stride = max(1, int(stride))
        self.max_frames = max_frames
        self.start_frame = max(0, int(start_frame))
        self.resize_to = resize_to
        self.cap: cv2.VideoCapture | None = None
        self.info: VideoInfo | None = None

    def __enter__(self) -> "VideoSource":
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video source: {self.source}")
        if self.start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.info = VideoInfo(
            path=str(self.source),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 0 else 30.0,
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
        if self.resize_to and self.info:
            self.info.width, self.info.height = self.resize_to
        return self

    def __exit__(self, *exc) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def planned_frames(self) -> int:
        """Best-effort count of how many frames this pass will yield."""
        if not self.info or self.info.frame_count <= 0:
            return self.max_frames or 0
        remaining = max(0, self.info.frame_count - self.start_frame)
        n = (remaining + self.stride - 1) // self.stride
        return min(n, self.max_frames) if self.max_frames else n

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        if self.cap is None:
            raise RuntimeError("VideoSource must be used as a context manager")

        raw_idx = self.start_frame
        yielded = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            if (raw_idx - self.start_frame) % self.stride == 0:
                if self.resize_to:
                    frame = cv2.resize(frame, self.resize_to, interpolation=cv2.INTER_AREA)
                yield raw_idx, frame
                yielded += 1
                if self.max_frames and yielded >= self.max_frames:
                    break
            raw_idx += 1


class VideoSink:
    """Write annotated frames to an .mp4.

    Tries H.264 first (plays inline in a browser); falls back to mp4v, which
    every OpenCV build can write but some browsers refuse to play. Check
    `.browser_friendly` if you care.
    """

    FOURCCS = ("avc1", "H264", "mp4v")

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.size = (int(size[0]), int(size[1]))
        self.writer: cv2.VideoWriter | None = None
        self.fourcc_used: str | None = None
        self.frames_written = 0

    @property
    def browser_friendly(self) -> bool:
        return self.fourcc_used in ("avc1", "H264")

    def _codec_works(self, cc: str) -> bool:
        """`isOpened()` lies on some builds -- write a few frames and check bytes."""
        probe = self.path.with_name(f".probe_{cc}_{self.path.stem}.mp4")
        try:
            writer = cv2.VideoWriter(
                str(probe), cv2.VideoWriter_fourcc(*cc), self.fps, self.size
            )
            if not writer.isOpened():
                writer.release()
                return False
            blank = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
            for _ in range(3):
                writer.write(blank)
            writer.release()
            return probe.exists() and probe.stat().st_size > 256
        except Exception:
            return False
        finally:
            try:
                probe.unlink()
            except OSError:
                pass

    def __enter__(self) -> "VideoSink":
        with _quiet_native_stderr():
            for cc in self.FOURCCS:
                if not self._codec_works(cc):
                    continue
                writer = cv2.VideoWriter(
                    str(self.path), cv2.VideoWriter_fourcc(*cc), self.fps, self.size
                )
                if writer.isOpened():
                    self.writer = writer
                    self.fourcc_used = cc
                    return self
                writer.release()

        raise IOError(
            f"No usable video codec found for {self.path}. "
            "Try `pip install opencv-python` (full build) or disable video saving."
        )

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            raise RuntimeError("VideoSink must be used as a context manager")
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(frame)
        self.frames_written += 1

    def __exit__(self, *exc) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def list_local_videos(folder: str | Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)


def list_local_images(folder: str | Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
