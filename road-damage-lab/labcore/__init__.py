"""Core building blocks for the road-damage lab."""

import os as _os

# OpenCV's bundled ffmpeg is chatty about codecs it can't load (openh264 in
# particular). Quiet it before cv2 is imported anywhere, unless the user has
# already asked for a specific level.
_os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET

# Windows without Developer Mode can't make symlinks, so the HF cache falls back
# to copies. That's fine here (a handful of small .pt files) -- don't warn about it.
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from . import taxonomy
from .registry import ModelSpec, load_registry, get_spec, list_specs
from .detector import Detection, Detector, resolve_device
from .draw import Annotator, DrawOptions
from .survey import SurveyReport, build_report
from .video import VideoSource, VideoSink, probe_video

__all__ = [
    "ModelSpec",
    "load_registry",
    "get_spec",
    "list_specs",
    "Detection",
    "Detector",
    "resolve_device",
    "Annotator",
    "DrawOptions",
    "SurveyReport",
    "build_report",
    "taxonomy",
    "VideoSource",
    "VideoSink",
    "probe_video",
]
