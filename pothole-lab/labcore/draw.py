"""Overlay rendering: boxes, ID/confidence chips, trails and the HUD.

Everything is drawn with plain OpenCV so the same annotator works in the
Streamlit app, in the CLI, and on saved video.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .detector import Detection

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_THIN = cv2.FONT_HERSHEY_SIMPLEX

# Vivid, well-separated colours (RGB) -> stored as BGR for OpenCV.
_PALETTE_RGB = [
    (255, 92, 51),
    (255, 187, 51),
    (46, 204, 113),
    (52, 152, 219),
    (155, 89, 182),
    (241, 90, 141),
    (26, 188, 156),
    (247, 220, 60),
    (255, 138, 101),
    (129, 212, 250),
]
PALETTE = [(b, g, r) for (r, g, b) in _PALETTE_RGB]

HUD_BG = (26, 22, 18)
HUD_FG = (238, 238, 238)
HUD_DIM = (168, 168, 168)
ACCENT = (80, 200, 120)


def color_for(key: int) -> tuple[int, int, int]:
    return PALETTE[int(key) % len(PALETTE)]


# OpenCV's Hershey fonts are ASCII-only -- anything else renders as "?".
_ASCII_MAP = str.maketrans(
    {
        "·": "-",
        "•": "*",
        "×": "x",
        "→": "->",
        "←": "<-",
        "⬇": "v",
        "–": "-",
        "—": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "°": "deg",
        "≈": "~",
        "±": "+/-",
    }
)


def ascii_safe(text: str) -> str:
    """Make a string renderable by cv2.putText."""
    return str(text).translate(_ASCII_MAP).encode("ascii", "replace").decode("ascii")


def _luma(bgr: tuple[int, int, int]) -> float:
    b, g, r = bgr
    return 0.299 * r + 0.587 * g + 0.114 * b


def _text_on(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return (20, 20, 20) if _luma(bgr) > 150 else (255, 255, 255)


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #


@dataclass
class DrawOptions:
    show_labels: bool = True
    show_conf: bool = True
    show_track_id: bool = True
    show_dimensions: bool = True
    show_conf_bar: bool = True
    show_trails: bool = True
    show_center_dot: bool = True
    show_hud: bool = True
    corner_box: bool = True
    trail_length: int = 32
    box_thickness: int = 0  # 0 = auto from frame size
    font_scale: float = 0.0  # 0 = auto from frame size
    color_by: str = "track"  # "track" | "class" | "confidence"
    hud_position: str = "top-left"  # "top-left" | "top-right"
    hud_detail: str = "compact"  # "compact" = live stats only | "full" = + settings
    hud_scale: float = 0.85  # multiplier on the auto-computed HUD text size


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def _panel(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int] = HUD_BG,
    alpha: float = 0.62,
    radius: int = 8,
) -> None:
    """Alpha-blended rounded rectangle, clipped to the frame."""
    H, W = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]
    layer = np.zeros_like(roi)
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(layer, (r, 0), (x2 - x1 - r, y2 - y1), color, -1)
    cv2.rectangle(layer, (0, r), (x2 - x1, y2 - y1 - r), color, -1)
    for cx, cy in ((r, r), (x2 - x1 - r, r), (r, y2 - y1 - r), (x2 - x1 - r, y2 - y1 - r)):
        cv2.circle(layer, (cx, cy), r, color, -1)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, roi)


def _corner_box(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Thin full rectangle plus heavy corner brackets -- reads well on video."""
    x1, y1 = p1
    x2, y2 = p2
    cv2.rectangle(img, p1, p2, color, max(1, thickness - 1), cv2.LINE_AA)

    arm = max(8, int(min(x2 - x1, y2 - y1) * 0.22))
    t = thickness + 2
    for (cx, cy), (dx, dy) in (
        ((x1, y1), (1, 1)),
        ((x2, y1), (-1, 1)),
        ((x1, y2), (1, -1)),
        ((x2, y2), (-1, -1)),
    ):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), color, t, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), color, t, cv2.LINE_AA)


def _text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    font: int = FONT,
) -> None:
    cv2.putText(img, ascii_safe(text), org, font, scale, color, thickness, cv2.LINE_AA)


def _measure(text: str, scale: float, thickness: int = 1, font: int = FONT):
    (w, h), base = cv2.getTextSize(ascii_safe(text), font, scale, thickness)
    return w, h, base


Rect = tuple[int, int, int, int]  # x, y, w, h


def _hits(rect: Rect, blocked: Sequence[Rect]) -> bool:
    x, y, w, h = rect
    for bx, by, bw, bh in blocked:
        if x < bx + bw and bx < x + w and y < by + bh and by < y + h:
            return True
    return False


# --------------------------------------------------------------------------- #
# Annotator
# --------------------------------------------------------------------------- #


class Annotator:
    """Stateful renderer -- keeps per-track trails across frames."""

    def __init__(self, options: DrawOptions | None = None) -> None:
        self.opt = options or DrawOptions()
        self._trails: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.opt.trail_length)
        )
        self._seen_ids: set[int] = set()

    def reset(self) -> None:
        self._trails.clear()
        self._seen_ids.clear()

    # ----------------------------------------------------------------- scale

    def _metrics(self, frame: np.ndarray) -> tuple[float, int]:
        h, w = frame.shape[:2]
        base = max(h, w)
        scale = self.opt.font_scale or max(0.42, min(1.05, base / 1500.0))
        thick = self.opt.box_thickness or max(2, int(base / 640))
        return scale, thick

    def _det_color(self, det: Detection, idx: int) -> tuple[int, int, int]:
        mode = self.opt.color_by
        if mode == "class":
            return color_for(det.cls_id)
        if mode == "confidence":
            # red -> amber -> green as confidence rises
            c = max(0.0, min(1.0, det.conf))
            if c < 0.5:
                t = c / 0.5
                return (int(60 * t), int(80 + 130 * t), 235)
            t = (c - 0.5) / 0.5
            return (int(60 + 40 * t), int(210 + 20 * t), int(235 - 150 * t))
        key = det.track_id if det.track_id is not None else idx
        return color_for(key)

    # ------------------------------------------------------------------ main

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        hud: dict[str, Any] | None = None,
        copy: bool = True,
    ) -> np.ndarray:
        img = frame.copy() if copy else frame
        H, W = img.shape[:2]
        scale, thick = self._metrics(img)

        # Register IDs before laying the HUD out so its counter isn't a frame behind.
        for det in detections:
            if det.track_id is not None:
                self._seen_ids.add(det.track_id)

        # Lay the HUD out first (without drawing it) so box labels can dodge it.
        layout = None
        blocked: list[Rect] = []
        if self.opt.show_hud and hud:
            layout = self._hud_layout(img, hud, scale)
            blocked = [layout["panel"]] + ([layout["footer"]] if layout["footer"] else [])

        for i, det in enumerate(detections):
            self._draw_one(img, det, self._det_color(det, i), scale, thick, W, H, blocked)

        if self.opt.show_trails:
            self._draw_trails(img, detections, thick)

        if layout is not None:
            self._render_hud(img, layout)

        return img

    # ------------------------------------------------------------ single box

    def _draw_one(
        self,
        img: np.ndarray,
        det: Detection,
        color: tuple[int, int, int],
        scale: float,
        thick: int,
        W: int,
        H: int,
        blocked: Sequence[Rect] = (),
    ) -> None:
        x1 = int(max(0, min(W - 1, det.x1)))
        y1 = int(max(0, min(H - 1, det.y1)))
        x2 = int(max(0, min(W - 1, det.x2)))
        y2 = int(max(0, min(H - 1, det.y2)))
        if x2 <= x1 or y2 <= y1:
            return

        if self.opt.corner_box:
            _corner_box(img, (x1, y1), (x2, y2), color, thick)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        if self.opt.show_center_dot:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(img, (cx, cy), max(2, thick), color, -1, cv2.LINE_AA)

        if self.opt.show_conf_bar:
            self._conf_bar(img, x1, x2, y2, det.conf, color, thick)

        if not self.opt.show_labels:
            return

        # ---- primary chip: "#7  pothole  0.87"
        parts: list[str] = []
        if self.opt.show_track_id and det.track_id is not None:
            parts.append(f"#{det.track_id}")
        parts.append(det.label)
        if self.opt.show_conf:
            parts.append(f"{det.conf:.2f}")
        chip = "  ".join(parts)

        tw, th, base = _measure(chip, scale, 1)
        pad_x, pad_y = int(8 * scale) + 4, int(6 * scale) + 4
        chip_w, chip_h = tw + 2 * pad_x, th + base + 2 * pad_y

        # ---- secondary line: geometry, smaller and dimmer
        sub = sub_w = sub_h = spad = sh = None
        sub_scale = scale * 0.78
        if self.opt.show_dimensions:
            sub = f"{int(det.width)}x{int(det.height)}px  {det.area_pct(W, H):.1f}% frame"
            sw, sh, sbase = _measure(sub, sub_scale, 1, FONT_THIN)
            spad = int(5 * scale) + 3
            sub_w, sub_h = sw + 2 * spad, sh + sbase + 2 * spad

        # The chip and the geometry line move as one block, so they never split
        # across the box edge.
        block_w = max(chip_w, sub_w or 0)
        block_h = chip_h + ((sub_h + 2) if sub_h else 0)
        bx, by = self._place_label(x1, y1, x2, y2, block_w, block_h, W, H, blocked)

        _panel(img, bx, by, chip_w, chip_h, color, alpha=0.92, radius=int(4 * scale) + 3)
        _text(img, chip, (bx + pad_x, by + pad_y + th), scale, _text_on(color), 1)

        if sub:
            sy = by + chip_h + 2
            _panel(img, bx, sy, sub_w, sub_h, HUD_BG, alpha=0.72, radius=3)
            _text(img, sub, (bx + spad, sy + spad + sh), sub_scale, HUD_FG, 1, FONT_THIN)

    @staticmethod
    def _place_label(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        bw: int,
        bh: int,
        W: int,
        H: int,
        blocked: Sequence[Rect],
    ) -> tuple[int, int]:
        """Pick the first label position that stays on-screen and clear of the HUD."""
        xs = [max(0, min(x1, W - bw)), max(0, min(x2 - bw, W - bw))]
        for bx, by, bbw, _bbh in blocked:  # also try just past each blocked panel
            xs.append(max(0, min(bx + bbw + 6, W - bw)))
        ys = [y1 - bh, y1, y2 + 4, y2 - bh]

        fallback = (xs[0], max(0, ys[0]))
        for y in ys:
            if y < 0 or y + bh > H:
                continue
            for x in xs:
                if not _hits((x, y, bw, bh), blocked):
                    return x, y
        return fallback

    def _conf_bar(
        self,
        img: np.ndarray,
        x1: int,
        x2: int,
        y2: int,
        conf: float,
        color: tuple[int, int, int],
        thick: int,
    ) -> None:
        bar_h = max(3, thick + 1)
        top = min(img.shape[0] - bar_h - 1, y2 + 3)
        full = max(1, x2 - x1)
        cv2.rectangle(img, (x1, top), (x1 + full, top + bar_h), (48, 48, 48), -1)
        filled = int(full * max(0.0, min(1.0, conf)))
        if filled > 0:
            cv2.rectangle(img, (x1, top), (x1 + filled, top + bar_h), color, -1)

    # ---------------------------------------------------------------- trails

    def _draw_trails(
        self, img: np.ndarray, detections: Sequence[Detection], thick: int
    ) -> None:
        live = set()
        for det in detections:
            if det.track_id is None:
                continue
            live.add(det.track_id)
            cx, cy = det.center
            self._trails[det.track_id].append((int(cx), int(cy)))

        for tid in live:
            pts = list(self._trails[tid])
            if len(pts) < 2:
                continue
            color = color_for(tid)
            n = len(pts)
            for i in range(1, n):
                # Fade the tail out toward the oldest point.
                w = max(1, int(thick * (i / n)))
                cv2.line(img, pts[i - 1], pts[i], color, w, cv2.LINE_AA)

    # ------------------------------------------------------------------- HUD

    def _hud_layout(
        self, img: np.ndarray, hud: dict[str, Any], scale: float
    ) -> dict[str, Any]:
        """Measure the HUD without drawing it, so labels can be routed around it."""
        H, W = img.shape[:2]
        # Floor is low enough that the size slider still bites on small frames;
        # below ~0.3 the Hershey font gets hard to read, which is the user's call.
        s = max(0.24, scale * 0.86 * max(0.4, self.opt.hud_scale))
        title = str(hud.get("title", "pothole lab"))

        # Compact keeps only what changes frame to frame; the static settings
        # live in the sidebar anyway. "full" stamps them onto exported video.
        keys = (
            ("frame", "in frame", "unique IDs", "fps")
            if self.opt.hud_detail == "compact"
            else (
                "device",
                "frame",
                "in frame",
                "unique IDs",
                "infer",
                "fps",
                "conf",
                "imgsz",
                "tracker",
            )
        )
        rows = [
            (k, str(hud[k])) for k in keys if k in hud and hud[k] not in (None, "")
        ]

        tw, th, tb = _measure(title, s * 1.05, 1)
        label_w = max((_measure(f"{k}", s, 1, FONT_THIN)[0] for k, _ in rows), default=0)
        value_w = max((_measure(f"{v}", s, 1)[0] for _, v in rows), default=0)
        gap = int(10 * s) + 6
        pad = int(8 * s) + 5
        line_h = int(th + tb + 5 * s) + 3

        panel_w = max(tw, label_w + gap + value_w) + 2 * pad
        panel_h = pad + (th + tb) + int(5 * s) + line_h * len(rows) + pad // 2

        px = W - panel_w - 12 if self.opt.hud_position == "top-right" else 12
        px = max(0, px)

        footer: Rect | None = None
        foot_text = ""
        foot_th = 0
        if self._seen_ids:
            foot_text = f"unique potholes seen: {len(self._seen_ids)}"
            fw, foot_th, fb = _measure(foot_text, s, 1)
            fpad = int(6 * s) + 4
            footer = (
                12,
                H - (foot_th + fb + 2 * fpad) - 12,
                fw + 2 * fpad,
                foot_th + fb + 2 * fpad,
            )

        return {
            "panel": (px, 12, panel_w, panel_h),
            "footer": footer,
            "footer_text": foot_text,
            "footer_th": foot_th,
            "title": title,
            "rows": rows,
            "s": s,
            "pad": pad,
            "gap": gap,
            "line_h": line_h,
            "label_w": label_w,
            "th": th,
        }

    def _render_hud(self, img: np.ndarray, L: dict[str, Any]) -> None:
        px, py, pw, ph = L["panel"]
        s, pad, th = L["s"], L["pad"], L["th"]

        _panel(img, px, py, pw, ph, HUD_BG, alpha=0.66, radius=int(7 * s) + 3)

        y = py + pad + th
        _text(img, L["title"], (px + pad, y), s * 1.05, ACCENT, 1)
        y += int(5 * s)

        for k, v in L["rows"]:
            y += L["line_h"]
            _text(img, k, (px + pad, y), s, HUD_DIM, 1, FONT_THIN)
            _text(img, v, (px + pad + L["label_w"] + L["gap"], y), s, HUD_FG, 1)

        # tiny footer, bottom-left: total unique potholes so far
        if L["footer"]:
            fx, fy, fw, fh = L["footer"]
            fpad = int(6 * s) + 4
            _panel(img, fx, fy, fw, fh, HUD_BG, 0.62, int(6 * s) + 3)
            _text(
                img,
                L["footer_text"],
                (fx + fpad, fy + fpad + L["footer_th"]),
                s,
                HUD_FG,
                1,
            )


def draw_empty_notice(frame: np.ndarray, text: str = "no detections") -> np.ndarray:
    """Small badge for frames where the model found nothing."""
    img = frame
    H, W = img.shape[:2]
    s = max(0.45, max(H, W) / 1700.0)
    tw, th, tb = _measure(text, s, 1)
    pad = int(10 * s) + 6
    x, y = W - (tw + 2 * pad) - 12, 12
    _panel(img, x, y, tw + 2 * pad, th + tb + 2 * pad, (40, 40, 90), 0.6, 8)
    _text(img, text, (x + pad, y + pad + th), s, (210, 210, 235), 1)
    return img
