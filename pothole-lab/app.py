"""Pothole Detection Lab -- swap YOLO models, feed video, watch the boxes.

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

from labcore.detector import (
    Detector,
    RunStats,
    available_devices,
    device_label,
    resolve_device,
)
from labcore.draw import Annotator, DrawOptions, draw_empty_notice
from labcore.registry import ModelSpec, load_registry
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
    page_title="Pothole Detection Lab",
    page_icon="🕳️",
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
    </style>
    """,
    unsafe_allow_html=True,
)


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
sb.title("🕳️ Pothole Lab")

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
sb.caption(f"`{spec_a['origin']}`" + ("" if spec_a["cached"] else "  · downloads on first run"))

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
opts.color_by = sb.selectbox("Colour boxes by", ["track", "class", "confidence"], index=0)
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

st.title("Pothole Detection Lab")
st.markdown(
    '<span class="lab-pill">switch models</span>'
    '<span class="lab-pill">persistent IDs</span>'
    '<span class="lab-pill">per-box confidence</span>'
    '<span class="lab-pill">side-by-side compare</span>',
    unsafe_allow_html=True,
)

tab_video, tab_image, tab_models = st.tabs(["🎬  Video", "🖼️  Image", "🧰  Models"])


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


def run_video(
    source,
    detectors: list[Detector],
    class_filter: list[int] | None,
    out_stem: str,
):
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
                        classes=class_filter,
                        max_det=int(max_det),
                    )
                    stats[k].update(res.detections, res.infer_ms)
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

                    show_image(slots[k], annotated)

                processed += 1
                if total:
                    progress.progress(
                        min(1.0, processed / total),
                        text=f"frame {processed} / {total}",
                    )
                else:
                    progress.progress(0.0, text=f"frame {processed}")

                with metric_slot.container():
                    mcols = st.columns(4 * len(detectors))
                    for k, det in enumerate(detectors):
                        s = stats[k]
                        base = 4 * k
                        mcols[base].metric("dets total", s.total_dets)
                        mcols[base + 1].metric(
                            "unique IDs", len(s.unique_ids) if use_track else "—"
                        )
                        mcols[base + 2].metric("mean conf", f"{s.mean_conf:.2f}")
                        mcols[base + 3].metric("infer fps", f"{s.fps:.1f}")

                if frame_delay:
                    spent = time.perf_counter() - t_frame
                    if spent < frame_delay:
                        time.sleep(frame_delay - spent)
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
            track=False, classes=class_filter, max_det=int(max_det),
        )
        stats = RunStats()
        stats.update(res.detections, res.infer_ms)
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
        rows = [d.as_row(0, W, H) for d in res.detections]
        for r in rows:
            r["model"] = det.spec.id
        all_rows.extend(rows)

        out = OUT_DIR / f"{image_path.stem}__{det.spec.id}.jpg"
        cv2.imwrite(str(out), img)

    if all_rows:
        st.dataframe(pd.DataFrame(all_rows), width="stretch", hide_index=True)
    else:
        st.info("No detections above the current confidence threshold.")


def load_selected() -> list[Detector]:
    ids = [model_a] + ([model_b] if compare and model_b else [])
    dets = []
    for mid in ids:
        with st.spinner(f"Loading {BY_ID[mid]['name']}…"):
            dets.append(get_detector(mid, device, half))
    return dets


def class_filter_ui(detectors: list[Detector]) -> list[int] | None:
    names = detectors[0].class_names
    if len(names) <= 1:
        return None
    picked = st.multiselect("Classes to keep", names, default=names)
    if len(picked) == len(names):
        return None
    lookup = {v: k for k, v in detectors[0].names.items()}
    return [lookup[p] for p in picked]


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
            st.markdown(f"**{name}**")
            c = st.columns(6)
            c[0].metric("frames", s["frames"])
            c[1].metric("detections", s["detections"])
            c[2].metric("unique potholes", s["unique_potholes"])
            c[3].metric("mean conf", s["mean_conf"])
            c[4].metric("mean infer", f"{s['mean_infer_ms']} ms")
            c[5].metric("fps", s["fps"])

            rows = last["rows"][k]
            if rows:
                df = pd.DataFrame(rows)
                with st.expander(f"Detections table · {len(df)} rows"):
                    st.dataframe(df, width="stretch", hide_index=True, height=320)

                    if "track_id" in df and (df["track_id"] >= 0).any():
                        per_id = (
                            df[df.track_id >= 0]
                            .groupby("track_id")
                            .agg(
                                frames=("frame", "count"),
                                first_frame=("frame", "min"),
                                last_frame=("frame", "max"),
                                mean_conf=("conf", "mean"),
                                max_conf=("conf", "max"),
                                mean_area_pct=("area_pct_frame", "mean"),
                            )
                            .round(3)
                            .reset_index()
                        )
                        st.markdown("**Per-pothole summary**")
                        st.dataframe(per_id, width="stretch", hide_index=True)

                    st.line_chart(
                        df.groupby("frame").size().rename("detections"),
                        height=180,
                    )

                st.download_button(
                    f"⬇  detections CSV · {last['ids'][k]}",
                    df.to_csv(index=False).encode(),
                    file_name=f"{last['stem']}__{last['ids'][k]}_detections.csv",
                    mime="text/csv",
                    key=f"csv_{k}_{last['stem']}",
                )
            else:
                st.info("No detections in this pass — try lowering the confidence threshold.")

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
    st.dataframe(
        pd.DataFrame(SPECS)[["id", "name", "task", "origin", "cached", "notes"]],
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.markdown("**Loaded model details**")
    if st.button("Inspect selected model(s)"):
        for d in load_selected():
            info = d.describe()
            st.markdown(f"### {info['name']}")
            st.json(info)

    st.divider()
    st.code(
        """models:
  - id: my-model
    name: "My YOLOv11 pothole run"
    source: local            # hf | local | ultralytics
    path: weights/local/best.pt
    task: detect
    default_conf: 0.3""",
        language="yaml",
    )
