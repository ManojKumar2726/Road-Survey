"""Road Damage Lab -- swap YOLO models, feed video, watch the boxes.

Detects the full RDD2022 damage taxonomy (longitudinal / transverse /
alligator cracking and potholes), not just potholes, and normalises every
model's class ids onto one canonical vocabulary so any two can be compared.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import labcore  # noqa: F401  -- must precede cv2: quiets OpenCV/ffmpeg logging
import cv2
import pandas as pd
import streamlit as st

from labcore import taxonomy as tax
from labcore.detector import (
    Detector,
    RunStats,
    available_devices,
    device_label,
    resolve_device,
)
from labcore.draw import Annotator, DrawOptions, draw_empty_notice
from labcore.registry import ModelSpec, load_registry
from labcore.survey import build_report
from labcore.video import (
    VideoSink,
    VideoSource,
    list_local_images,
    list_local_videos,
    probe_video,
)

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "data" / "videos"
IMAGE_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "outputs"
for d in (VIDEO_DIR, IMAGE_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Road Damage Lab",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
      div[data-testid="stMetricValue"] { font-size: 1.5rem; }
      .lab-note { color: #8b8b8b; font-size: 0.82rem; line-height: 1.5; }
      .lab-pill { display:inline-block; padding:2px 9px; border-radius:99px;
                  background:#2b2b2b; color:#d6d6d6; font-size:0.74rem;
                  margin-right:6px; border:1px solid #3d3d3d; }
      .dmg-chip { display:inline-flex; align-items:center; gap:6px;
                  padding:3px 10px; border-radius:99px; background:#232323;
                  border:1px solid #3a3a3a; font-size:0.78rem; color:#e4e4e4;
                  margin:0 6px 6px 0; }
      .dmg-dot { width:10px; height:10px; border-radius:3px; display:inline-block; }
      .dmg-grade { font-size:1.35rem; font-weight:600; letter-spacing:-0.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)


def damage_chips(counts: dict[str, int], suffix: str = "") -> str:
    """Coloured per-damage-type chips, worst-first. Colours match the overlay."""
    out = []
    for key in tax.sort_keys(counts):
        n = counts.get(key, 0)
        if not n:
            continue
        d = tax.get(key)
        out.append(
            f'<span class="dmg-chip"><span class="dmg-dot" '
            f'style="background:{d.hex}"></span>{d.label} <b>{n}</b>{suffix}</span>'
        )
    return "".join(out) or '<span class="lab-note">nothing detected</span>'


# --------------------------------------------------------------------------- #
# Streamlit compatibility helpers
# --------------------------------------------------------------------------- #


def show_image(slot, img_bgr, caption: str | None = None):
    """st.image across versions -- the width API changed more than once."""
    rgb = img_bgr[:, :, ::-1]
    try:
        slot.image(rgb, caption=caption, width="stretch")
    except Exception:
        slot.image(rgb, caption=caption, use_container_width=True)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_detector(model_id: str, device: str, half: bool) -> Detector:
    spec = load_registry().get(model_id)
    return Detector(spec, device=device, half=half)


@st.cache_data(show_spinner=False)
def registry_specs() -> list[dict]:
    return [
        {
            "id": s.id,
            "name": s.name,
            "origin": s.origin,
            "task": s.task,
            "notes": s.notes,
            "cached": s.is_cached(),
            "default_conf": s.default_conf,
            "default_iou": s.default_iou,
            "default_imgsz": s.default_imgsz,
            # Declared in models.yaml, so the picker can show what a model
            # covers without downloading its weights first.
            "damage_types": tax.sort_keys((s.class_map or {}).values()),
        }
        for s in load_registry()
    ]


# --------------------------------------------------------------------------- #
# Sidebar -- the control panel
# --------------------------------------------------------------------------- #

try:
    SPECS = registry_specs()
except Exception as exc:  # bad models.yaml shouldn't nuke the whole page
    st.error(f"Could not load models.yaml: {exc}")
    st.stop()

if not SPECS:
    st.error("No enabled models in models.yaml. Set `enabled: true` on at least one.")
    st.stop()

BY_ID = {s["id"]: s for s in SPECS}
IDS = list(BY_ID)

sb = st.sidebar
sb.title("🛣️ Road Damage Lab")

# ---- models
sb.subheader("Model")
mode = sb.radio(
    "Mode",
    ["Single model", "Compare two"],
    horizontal=True,
    label_visibility="collapsed",
)
compare = mode == "Compare two"


def fmt_model(mid: str) -> str:
    s = BY_ID[mid]
    return f"{s['name']}{'' if s['cached'] else '  ⬇'}"


model_a = sb.selectbox("Model A", IDS, format_func=fmt_model, key="model_a")
model_b = None
if compare:
    default_b = 1 if len(IDS) > 1 else 0
    model_b = sb.selectbox(
        "Model B", IDS, index=default_b, format_func=fmt_model, key="model_b"
    )

spec_a = BY_ID[model_a]
if spec_a["notes"]:
    sb.caption(spec_a["notes"])
if spec_a["damage_types"]:
    sb.caption(
        "detects: " + ", ".join(tax.short_of(k) for k in spec_a["damage_types"])
    )
sb.caption(f"`{spec_a['origin']}`" + ("" if spec_a["cached"] else "  · downloads on first run"))

# Comparing a 4-class model against a pothole-only one is a legitimate thing to
# do -- but the pothole model can't lose on cracks it was never trained to see,
# so say so rather than letting the numbers imply it.
if compare and model_b:
    a_types, b_types = set(spec_a["damage_types"]), set(BY_ID[model_b]["damage_types"])
    if a_types and b_types and a_types != b_types:
        only_a = tax.sort_keys(a_types - b_types)
        only_b = tax.sort_keys(b_types - a_types)
        bits = []
        if only_a:
            bits.append(f"only A: {', '.join(tax.short_of(k) for k in only_a)}")
        if only_b:
            bits.append(f"only B: {', '.join(tax.short_of(k) for k in only_b)}")
        sb.warning(
            "These models cover different damage types (" + "; ".join(bits) + "). "
            "Filter to the shared types below for a fair comparison.",
            icon="⚖️",
        )

# ---- inference
sb.subheader("Inference")
conf = sb.slider("Confidence threshold", 0.01, 0.95, float(spec_a["default_conf"]), 0.01)
iou = sb.slider("NMS IoU", 0.10, 0.95, float(spec_a["default_iou"]), 0.05)
imgsz = sb.select_slider("Image size", [320, 416, 512, 640, 768, 960, 1280], value=int(spec_a["default_imgsz"]))
max_det = sb.number_input("Max detections / frame", 1, 1000, 100, 10)

# ---- tracking
sb.subheader("Tracking")
use_track = sb.toggle("Assign persistent IDs", value=True)
tracker = sb.selectbox(
    "Tracker",
    ["bytetrack.yaml", "botsort.yaml"],
    disabled=not use_track,
    help="ByteTrack is faster; BoT-SORT re-identifies better through occlusion.",
)
if not use_track:
    sb.caption("Without tracking, boxes have no ID and repeat counts per frame.")

# ---- runtime
sb.subheader("Runtime")
devices = available_devices()
device = sb.selectbox("Device", devices, index=0)
half = sb.toggle("FP16 (half precision)", value=device.startswith("cuda"), disabled=not device.startswith("cuda"))
stride = sb.number_input("Frame stride", 1, 30, 1, help="Process every Nth frame.")
max_frames = sb.number_input("Max frames (0 = all)", 0, 100000, 0, 50)
proc_width = sb.select_slider(
    "Downscale frames to width",
    [0, 480, 640, 854, 960, 1280, 1600],
    value=0,
    format_func=lambda v: "native" if v == 0 else f"{v}px",
)
save_video = sb.toggle("Save annotated video", value=True)
realtime = sb.toggle("Throttle to source FPS", value=False, help="Play back at the video's real speed instead of as fast as possible.")

# ---- overlay
sb.subheader("Overlay")
opts = DrawOptions(
    show_track_id=sb.checkbox("Track ID", True),
    show_conf=sb.checkbox("Confidence", True),
    show_dimensions=sb.checkbox("Box size / % of frame", True),
    show_conf_bar=sb.checkbox("Confidence bar", True),
    show_trails=sb.checkbox("Motion trails", True),
    show_center_dot=sb.checkbox("Centre dot", True),
    show_hud=sb.checkbox("HUD panel", True),
    corner_box=sb.checkbox("Corner-bracket boxes", True),
)
opts.show_legend = sb.checkbox(
    "Damage legend",
    True,
    help="Bottom-left tally per damage type, colour-matched to the boxes.",
)
# Default is "class": with four damage types on screen, colouring by type is
# what makes the overlay readable. Track colouring is still there for
# following one defect through a clip.
opts.color_by = sb.selectbox("Colour boxes by", ["class", "track", "confidence"], index=0)
opts.label_mode = sb.selectbox(
    "Box label",
    ["canonical", "raw"],
    index=0,
    help="canonical = the lab's damage name (comparable across models). "
    "raw = whatever the checkpoint calls it, e.g. 'D40' — use this to audit a class_map.",
)
opts.hud_position = sb.selectbox(
    "HUD corner", ["top-left", "top-right"], index=0, disabled=not opts.show_hud
)
opts.hud_detail = sb.selectbox(
    "HUD detail",
    ["compact", "full"],
    index=0,
    disabled=not opts.show_hud,
    help="compact = live stats only. full = also stamps device and settings onto the frame.",
)
opts.hud_scale = sb.slider(
    "HUD size", 0.5, 1.5, 0.85, 0.05, disabled=not opts.show_hud
)
opts.trail_length = sb.slider("Trail length", 4, 90, 32, disabled=not opts.show_trails)


# --------------------------------------------------------------------------- #
# Source selection
# --------------------------------------------------------------------------- #

st.title("Road Damage Lab")
st.markdown(
    '<span class="lab-pill">cracks + potholes</span>'
    '<span class="lab-pill">switch models</span>'
    '<span class="lab-pill">persistent IDs</span>'
    '<span class="lab-pill">side-by-side compare</span>'
    '<span class="lab-pill">condition report</span>',
    unsafe_allow_html=True,
)

tab_video, tab_image, tab_models, tab_tax = st.tabs(
    ["🎬  Video", "🖼️  Image", "🧰  Models", "🏷️  Damage types"]
)


def stage_upload(uploaded, folder: Path) -> Path:
    """Persist an uploaded file next to the other lab assets."""
    dest = folder / uploaded.name
    if dest.exists():
        dest = folder / f"{Path(uploaded.name).stem}_{int(time.time())}{Path(uploaded.name).suffix}"
    with open(dest, "wb") as fh:
        shutil.copyfileobj(uploaded, fh)
    return dest


# --------------------------------------------------------------------------- #
# Core run loops
# --------------------------------------------------------------------------- #


def build_hud(det: Detector, stats: RunStats, idx: int, total: int, n_now: int) -> dict:
    return {
        "title": det.spec.name,
        "device": device_label(det.device, short=True) + (" fp16" if det.half else ""),
        "frame": f"{idx}" + (f" / {total}" if total else ""),
        "in frame": str(n_now),
        "unique IDs": str(len(stats.unique_ids)) if use_track else "—",
        "infer": f"{stats.ms[-1]:.1f} ms" if stats.ms else "—",
        "fps": f"{stats.fps:.1f}",
        "conf": f"{conf:.2f}",
        "imgsz": str(imgsz),
        "tracker": tracker.replace(".yaml", "") if use_track else "off",
    }


def render_survey(report, key_prefix: str) -> None:
    """Road-condition panel: grade, per-type chips, defect table, hotspots."""
    st.markdown(f'<div class="dmg-grade">{report.grade}</div>', unsafe_allow_html=True)
    counts = {c.canon: (c.defects if report.tracked else c.boxes) for c in report.by_class}
    st.markdown(damage_chips(counts), unsafe_allow_html=True)

    c = st.columns(4)
    c[0].metric("damage score", f"{report.damage_score:.1f}", help="Severity-weighted defects per 100 frames. Comparable only within the same clip.")
    c[1].metric("defects" if report.tracked else "boxes", report.total_defects)
    c[2].metric("confirmed", report.confirmed_defects, help="Seen in 2+ frames — less likely to be a flicker.")
    worst = report.worst
    c[3].metric("worst find", worst.label if worst else "—")

    if not report.tracked:
        st.caption(
            "Tracking is off, so every box counts separately — these are "
            "per-frame totals, not unique defects."
        )
    elif report.unassigned_boxes:
        st.caption(
            f"{report.unassigned_boxes} of {report.total_boxes} boxes never got "
            "a track ID (the tracker hadn't confirmed them yet) and aren't "
            "counted as defects — which is why boxes exceed defects."
        )

    cls_rows = report.class_rows()
    if cls_rows:
        st.markdown("**By damage type**")
        st.dataframe(pd.DataFrame(cls_rows), width="stretch", hide_index=True)

    spots = report.hotspots()
    if spots and report.frames > 24:
        st.markdown("**Where the damage is**")
        st.caption(
            "Damage score per segment of the clip. With a forward-facing "
            "camera, segment ≈ distance along the road."
        )
        hs = pd.DataFrame(spots).set_index("segment")
        st.bar_chart(hs["score"], height=170)

    drows = report.defect_rows()
    if drows:
        with st.expander(f"Every defect · {len(drows)} rows"):
            st.dataframe(pd.DataFrame(drows), width="stretch", hide_index=True, height=300)
            st.download_button(
                "⬇  defects CSV",
                pd.DataFrame(drows).to_csv(index=False).encode(),
                file_name=f"{key_prefix}_defects.csv",
                mime="text/csv",
                key=f"defcsv_{key_prefix}",
            )


# Caps how often the browser gets a new frame/metrics push, independent of how
# fast inference runs. Without this, a fast model pushes a full annotated
# frame over the websocket 60+ times a second -- on a proxied connection (e.g.
# a VS Code webview / port-forward) that back-pressure shows up as the video
# preview stalling and reconnecting rather than as a clean slowdown.
_UI_FPS_CAP = 15.0
_UI_MIN_INTERVAL = 1.0 / _UI_FPS_CAP


def run_video(
    source,
    detectors: list[Detector],
    class_filter: list[str] | None,
    out_stem: str,
):
    if class_filter == []:
        st.warning("Select at least one damage type to run.")
        return

    # Canonical keys -> each model's own class indices. Resolved once, per
    # detector, because the models disagree on class ordering.
    per_det_classes: list[list[int] | None] = [
        det.ids_for_canon(class_filter) if class_filter else None for det in detectors
    ]
    for det, ids in zip(detectors, per_det_classes):
        if class_filter and not ids:
            st.info(
                f"**{det.spec.name}** doesn't detect any of the selected damage "
                "types — it will report nothing this pass."
            )

    resize = None
    info = None
    if isinstance(source, (str, Path)):
        try:
            info = probe_video(source)
            if proc_width and info.width > proc_width:
                scale = proc_width / info.width
                resize = (proc_width, int(round(info.height * scale / 2) * 2))
        except IOError as exc:
            st.error(str(exc))
            return

    annotators = [Annotator(opts) for _ in detectors]
    stats = [RunStats() for _ in detectors]
    rows: list[list[dict]] = [[] for _ in detectors]
    for d in detectors:
        d.reset_tracker()

    header = st.container()
    stop_col, _ = header.columns([1, 5])
    stop_col.button("⏹  Stop", key=f"stop_{out_stem}", help="Interrupts the current pass.")

    if len(detectors) == 1:
        slots = [st.empty()]
    else:
        cols = st.columns(2)
        st.caption(f"Left: **{detectors[0].spec.name}** · Right: **{detectors[1].spec.name}**")
        slots = [c.empty() for c in cols]

    progress = st.progress(0.0, text="starting…")
    metric_slot = st.empty()

    sinks: list[VideoSink | None] = [None] * len(detectors)
    out_paths: list[Path | None] = [None] * len(detectors)

    src = VideoSource(
        source,
        stride=int(stride),
        max_frames=int(max_frames) or None,
        resize_to=resize,
    )

    t_start = time.perf_counter()
    processed = 0
    last_ui_ts = 0.0
    last_annotated: list[Any] = [None] * len(detectors)
    try:
        with src:
            total = src.planned_frames
            src_fps = src.info.fps if src.info else 30.0
            out_fps = max(1.0, src_fps / max(1, int(stride)))
            frame_delay = 1.0 / src_fps if realtime else 0.0

            for idx, frame in src:
                H, W = frame.shape[:2]
                t_frame = time.perf_counter()

                for k, det in enumerate(detectors):
                    res = det.infer(
                        frame,
                        conf=conf,
                        iou=iou,
                        imgsz=int(imgsz),
                        track=use_track,
                        tracker=tracker,
                        classes=per_det_classes[k],
                        max_det=int(max_det),
                    )
                    stats[k].update(res.detections, res.infer_ms, W, H)
                    rows[k].extend(d.as_row(idx, W, H) for d in res.detections)

                    annotated = annotators[k].draw(
                        frame,
                        res.detections,
                        hud=build_hud(det, stats[k], idx, src.info.frame_count if src.info else 0, len(res.detections)),
                    )
                    if not res.detections:
                        annotated = draw_empty_notice(annotated)

                    if save_video:
                        if sinks[k] is None:
                            p = OUT_DIR / f"{out_stem}__{det.spec.id}.mp4"
                            sink = VideoSink(p, out_fps, (annotated.shape[1], annotated.shape[0]))
                            sink.__enter__()
                            sinks[k] = sink
                            out_paths[k] = p
                        sinks[k].write(annotated)

                    # Every frame is processed, written and stats-updated above
                    # regardless of speed -- only the browser push is throttled.
                    last_annotated[k] = annotated

                processed += 1
                now = time.perf_counter()
                if now - last_ui_ts >= _UI_MIN_INTERVAL:
                    last_ui_ts = now
                    for k in range(len(detectors)):
                        show_image(slots[k], last_annotated[k])

                    if total:
                        progress.progress(
                            min(1.0, processed / total),
                            text=f"frame {processed} / {total}",
                        )
                    else:
                        progress.progress(0.0, text=f"frame {processed}")

                    with metric_slot.container():
                        for k, det in enumerate(detectors):
                            s = stats[k]
                            if len(detectors) > 1:
                                st.caption(f"**{det.spec.name}**")
                            mcols = st.columns(4)
                            mcols[0].metric("boxes total", s.total_dets)
                            mcols[1].metric(
                                "unique defects", len(s.unique_ids) if use_track else "—"
                            )
                            mcols[2].metric("mean conf", f"{s.mean_conf:.2f}")
                            mcols[3].metric("infer fps", f"{s.fps:.1f}")
                            # The live per-type tally -- the number that says
                            # what is actually wrong with the road.
                            counts = s.unique_counts() if use_track else dict(s.per_class)
                            st.markdown(damage_chips(counts), unsafe_allow_html=True)

                if frame_delay:
                    spent = time.perf_counter() - t_frame
                    if spent < frame_delay:
                        time.sleep(frame_delay - spent)

            # The throttle above can leave the preview up to one interval
            # behind when the loop ends -- show the true last frame.
            for k in range(len(detectors)):
                if last_annotated[k] is not None:
                    show_image(slots[k], last_annotated[k])
    finally:
        for s in sinks:
            if s is not None:
                s.__exit__(None, None, None)

    wall = time.perf_counter() - t_start
    progress.progress(1.0, text=f"done · {processed} frames in {wall:.1f}s")

    st.session_state["last_run"] = {
        "stem": out_stem,
        "models": [d.spec.name for d in detectors],
        "ids": [d.spec.id for d in detectors],
        "stats": [s.summary() for s in stats],
        "rows": rows,
        "outputs": [str(p) if p else None for p in out_paths],
        "browser_ok": [bool(s and s.browser_friendly) for s in sinks],
        "wall_s": round(wall, 2),
        "settings": {
            "conf": conf,
            "iou": iou,
            "imgsz": int(imgsz),
            "tracker": tracker if use_track else "off",
            "stride": int(stride),
            "device": device,
            "half": half,
        },
    }


def run_image(image_path: Path, detectors: list[Detector], class_filter):
    if class_filter == []:
        st.warning("Select at least one damage type to run.")
        return

    frame = cv2.imread(str(image_path))
    if frame is None:
        st.error(f"Could not read image: {image_path}")
        return
    if proc_width and frame.shape[1] > proc_width:
        s = proc_width / frame.shape[1]
        frame = cv2.resize(frame, (proc_width, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)

    H, W = frame.shape[:2]
    cols = st.columns(len(detectors))
    all_rows = []
    for k, det in enumerate(detectors):
        det.reset_tracker()
        res = det.infer(
            frame, conf=conf, iou=iou, imgsz=int(imgsz),
            track=False,
            classes=det.ids_for_canon(class_filter) if class_filter else None,
            max_det=int(max_det),
        )
        stats = RunStats()
        stats.update(res.detections, res.infer_ms, W, H)
        ann = Annotator(opts)
        img = ann.draw(
            frame,
            res.detections,
            hud={
                "title": det.spec.name,
                "device": device_label(det.device, short=True),
                "in frame": str(len(res.detections)),
                "infer": f"{res.infer_ms:.1f} ms",
                "conf": f"{conf:.2f}",
                "imgsz": str(imgsz),
            },
        )
        if not res.detections:
            img = draw_empty_notice(img)
        with cols[k]:
            show_image(st, img, caption=det.spec.name)
            st.caption(
                f"{len(res.detections)} detection(s) · mean conf "
                f"{stats.mean_conf:.2f} · {res.infer_ms:.1f} ms"
            )
            st.markdown(damage_chips(dict(stats.per_class)), unsafe_allow_html=True)

        rows = [d.as_row(0, W, H) for d in res.detections]
        for r in rows:
            r["model"] = det.spec.id
        all_rows.extend(rows)

        out = OUT_DIR / f"{image_path.stem}__{det.spec.id}.jpg"
        cv2.imwrite(str(out), img)

    if all_rows:
        st.dataframe(pd.DataFrame(all_rows), width="stretch", hide_index=True)
    else:
        st.info(
            "No detections above the current confidence threshold. The RDD "
            "models peak around 0.10–0.20, below the YOLO default of 0.25."
        )


def load_selected() -> list[Detector]:
    ids = [model_a] + ([model_b] if compare and model_b else [])
    dets = []
    for mid in ids:
        with st.spinner(f"Loading {BY_ID[mid]['name']}…"):
            dets.append(get_detector(mid, device, half))
    return dets


def class_filter_ui(detectors: list[Detector]) -> list[str] | None:
    """Pick damage types to keep, in canonical terms.

    Returns canonical keys, not class indices. Each detector converts them to
    its own ids at inference time -- the models here genuinely disagree on
    ordering (ozair-yolov8s has alligator and longitudinal swapped relative to
    the rdd-* set), so one shared index list would filter the wrong classes.
    """
    available = tax.sort_keys(k for d in detectors for k in d.canon_keys)
    if len(available) <= 1:
        return None

    shared = set(detectors[0].canon_keys)
    for d in detectors[1:]:
        shared &= set(d.canon_keys)

    picked = st.multiselect(
        "Damage types to keep",
        available,
        default=available,
        format_func=lambda k: tax.label_of(k)
        + ("" if k in shared or len(detectors) == 1 else "  (one model only)"),
        help="Applied per model, so this stays correct even when two "
        "checkpoints order their classes differently.",
    )
    if not picked:
        st.warning("No damage types selected — nothing will be reported.")
        return []
    if len(picked) == len(available):
        return None
    return picked


# --------------------------------------------------------------------------- #
# Video tab
# --------------------------------------------------------------------------- #

with tab_video:
    left, right = st.columns([2, 1])

    with left:
        src_kind = st.radio(
            "Source",
            ["Upload a video", "From data/videos", "Webcam"],
            horizontal=True,
        )

        chosen: str | int | None = None
        stem = "run"

        if src_kind == "Upload a video":
            up = st.file_uploader(
                "Drop a road video",
                type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
            )
            keep = st.checkbox("Keep a copy in data/videos", value=True)
            if up is not None:
                if keep:
                    path = stage_upload(up, VIDEO_DIR)
                else:
                    tmp = Path(tempfile.gettempdir()) / f"lab_{int(time.time())}_{up.name}"
                    tmp.write_bytes(up.getbuffer())
                    path = tmp
                chosen, stem = str(path), Path(path).stem

        elif src_kind == "From data/videos":
            vids = list_local_videos(VIDEO_DIR)
            if not vids:
                st.info(f"Drop videos into `{VIDEO_DIR.relative_to(ROOT)}` and they show up here.")
            else:
                pick = st.selectbox("Video", vids, format_func=lambda p: p.name)
                chosen, stem = str(pick), pick.stem

        else:
            cam = st.number_input("Camera index", 0, 8, 0)
            st.caption("Set a frame cap in the sidebar — a webcam stream never ends on its own.")
            chosen, stem = int(cam), f"webcam{int(cam)}"

    with right:
        if isinstance(chosen, str):
            try:
                info = probe_video(chosen)
                st.markdown("**Source**")
                st.markdown(
                    f'<div class="lab-note">{Path(chosen).name}<br>'
                    f"{info.resolution} · {info.fps:.1f} fps · "
                    f"{info.frame_count} frames · {info.duration_s:.1f}s</div>",
                    unsafe_allow_html=True,
                )
                planned = info.frame_count // max(1, int(stride))
                if max_frames:
                    planned = min(planned, int(max_frames))
                st.markdown(
                    f'<div class="lab-note">this pass: <b>{planned}</b> frames</div>',
                    unsafe_allow_html=True,
                )
                st.video(chosen)
            except IOError as exc:
                st.warning(str(exc))

    if chosen is not None:
        detectors = load_selected()
        cfilter = class_filter_ui(detectors)
        go = st.button("▶  Run detection", type="primary", width="stretch")
        if go:
            run_video(chosen, detectors, cfilter, stem)

    # ---- results of the last pass
    last = st.session_state.get("last_run")
    if last:
        st.divider()
        st.subheader("Last run")
        st.caption(
            " · ".join(f"{k}={v}" for k, v in last["settings"].items())
            + f" · wall {last['wall_s']}s"
        )

        for k, name in enumerate(last["models"]):
            s = last["stats"][k]
            st.markdown(f"### {name}")
            c = st.columns(6)
            c[0].metric("frames", s["frames"])
            c[1].metric("boxes", s["detections"])
            c[2].metric("unique defects", s["unique_objects"])
            c[3].metric("mean conf", s["mean_conf"])
            c[4].metric("mean infer", f"{s['mean_infer_ms']} ms")
            c[5].metric("fps", s["fps"])

            rows = last["rows"][k]
            if rows:
                df = pd.DataFrame(rows)
                report = build_report(
                    rows, last["ids"][k], name, frames=s["frames"]
                )

                st.markdown("#### Road condition")
                render_survey(report, key_prefix=f"{last['stem']}_{last['ids'][k]}")

                with st.expander(f"Raw detections · {len(df)} rows"):
                    st.dataframe(df, width="stretch", hide_index=True, height=320)

                    # Detections per frame, split by damage type -- shows which
                    # types the model fires on continuously vs occasionally.
                    by_type = (
                        df.groupby(["frame", "damage_label"])
                        .size()
                        .unstack(fill_value=0)
                    )
                    st.markdown("**Detections per frame, by type**")
                    st.line_chart(by_type, height=200)

                st.download_button(
                    f"⬇  detections CSV · {last['ids'][k]}",
                    df.to_csv(index=False).encode(),
                    file_name=f"{last['stem']}__{last['ids'][k]}_detections.csv",
                    mime="text/csv",
                    key=f"csv_{k}_{last['stem']}",
                )
            else:
                st.info(
                    "No detections in this pass — try lowering the confidence "
                    "threshold. The RDD models peak around 0.10–0.20, well "
                    "below the YOLO default."
                )

            out = last["outputs"][k]
            if out and Path(out).exists():
                if last["browser_ok"][k]:
                    st.video(out)
                else:
                    st.caption(
                        "Annotated video saved with the mp4v codec — most browsers "
                        "won't play it inline, so download it instead."
                    )
                st.download_button(
                    f"⬇  annotated video · {last['ids'][k]}",
                    Path(out).read_bytes(),
                    file_name=Path(out).name,
                    mime="video/mp4",
                    key=f"vid_{k}_{last['stem']}",
                )
                st.caption(f"saved to `{out}`")

        # ---- head to head, when two models ran the same clip
        if len(last["models"]) > 1:
            st.divider()
            st.subheader("Head to head")
            comp = []
            for k, name in enumerate(last["models"]):
                s = last["stats"][k]
                row = {
                    "model": name,
                    "boxes": s["detections"],
                    "unique defects": s["unique_objects"],
                    "mean conf": s["mean_conf"],
                    "ms/frame": s["mean_infer_ms"],
                    "fps": s["fps"],
                }
                # Per-type unique counts: a model can lead on totals purely by
                # over-firing on one easy class.
                for key, n in s["unique_by_class"].items():
                    row[tax.short_of(key)] = n
                comp.append(row)
            cdf = pd.DataFrame(comp).fillna(0)
            st.dataframe(cdf, width="stretch", hide_index=True)
            st.caption(
                "Per-type columns are unique tracked defects. A model that "
                "doesn't cover a damage type shows 0 there — check the "
                "sidebar warning before reading that as a miss."
            )


# --------------------------------------------------------------------------- #
# Image tab
# --------------------------------------------------------------------------- #

with tab_image:
    kind = st.radio("Source", ["Upload", "From data/images"], horizontal=True, key="img_src")
    img_path: Path | None = None

    if kind == "Upload":
        up = st.file_uploader("Road image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img_up")
        if up is not None:
            img_path = stage_upload(up, IMAGE_DIR)
    else:
        imgs = list_local_images(IMAGE_DIR)
        if not imgs:
            st.info(f"Drop images into `{IMAGE_DIR.relative_to(ROOT)}`.")
        else:
            img_path = st.selectbox("Image", imgs, format_func=lambda p: p.name)

    if img_path is not None:
        dets = load_selected()
        cf = class_filter_ui(dets)
        if st.button("▶  Detect", type="primary", key="img_go"):
            run_image(img_path, dets, cf)


# --------------------------------------------------------------------------- #
# Models tab
# --------------------------------------------------------------------------- #

with tab_models:
    st.markdown(
        "Models come from **models.yaml**. Add a block there and it appears in the "
        "sidebar — no code changes. `⬇` in the picker means the weights aren't "
        "cached yet and will download on first run."
    )
    mdf = pd.DataFrame(SPECS)
    mdf["damage types"] = mdf["damage_types"].apply(
        lambda ks: ", ".join(tax.short_of(k) for k in ks) if ks else "(matched by name)"
    )
    st.dataframe(
        mdf[["id", "name", "task", "damage types", "cached", "origin", "notes"]],
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.markdown("**Loaded model details**")
    st.caption(
        "Shows how each raw class id maps onto the lab's taxonomy. "
        "`python run_cli.py --inspect` does the same for every model at once."
    )
    if st.button("Inspect selected model(s)"):
        for d in load_selected():
            info = d.describe()
            st.markdown(f"### {info['name']}")
            if info["map_warnings"]:
                for w in info["map_warnings"]:
                    st.error(w)
            st.json(info)

    st.divider()
    st.code(
        """models:
  - id: my-model
    name: "My road damage run"
    source: local            # hf | local | ultralytics
    path: weights/local/best.pt
    task: detect
    default_conf: 0.15
    # Raw class id -> canonical damage key. Read the ids off the
    # checkpoint with `python run_cli.py --inspect -m my-model`.
    class_map: {0: longitudinal_crack, 1: transverse_crack,
                2: alligator_crack, 3: pothole}""",
        language="yaml",
    )


# --------------------------------------------------------------------------- #
# Damage types tab
# --------------------------------------------------------------------------- #

with tab_tax:
    st.markdown(
        "Every model's classes are normalised onto this vocabulary, so colours, "
        "filters, statistics and the condition report mean the same thing "
        "whichever checkpoint produced them. Two of the registered models order "
        "the *same* four RDD2022 classes differently — without this layer they'd "
        "silently disagree on every box."
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "key": d.key,
                    "label": d.label,
                    "RDD code": d.code or "—",
                    "severity": d.severity,
                    "colour": d.hex,
                    "what it is": d.description,
                }
                for d in (tax.get(k) for k in tax.ORDER)
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "".join(
            f'<span class="dmg-chip"><span class="dmg-dot" '
            f'style="background:{tax.get(k).hex}"></span>{tax.get(k).label}</span>'
            for k in tax.ORDER
        ),
        unsafe_allow_html=True,
    )

    st.caption(
        "**Severity** weights the damage score — a pothole counts for about "
        "three times a transverse crack. `repair` is context rather than a "
        "defect and barely scores; `unknown` catches classes the lab couldn't "
        "map, which draw grey and are worth investigating with "
        "`run_cli.py --inspect`."
    )

    st.divider()
    st.markdown("**Which models cover what**")
    cover = []
    for s in SPECS:
        row = {"model": s["id"]}
        for key in tax.ORDER:
            if key == tax.UNKNOWN_KEY:
                continue
            row[tax.short_of(key)] = "●" if key in s["damage_types"] else ""
        cover.append(row)
    st.dataframe(pd.DataFrame(cover), width="stretch", hide_index=True)
