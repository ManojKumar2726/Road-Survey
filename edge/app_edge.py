"""Window 1 -- the onboard system.

Upload a clip, watch it process, watch events leave the bus.

    streamlit run app_edge.py

Deliberately shaped like an onboard unit rather than a research tool: the
sidebar sets what the *bus* is (id, route, where it reports), and the main
panel shows what it is seeing and what it has sent. The model comparison bench
lives next door in road-damage-lab and stays there.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

EDGE_DIR = Path(__file__).resolve().parent
if str(EDGE_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_DIR))

from edgecore.config import EdgeConfig
from edgecore.gps import get_route, load_routes, parse_when
from edgecore.pipeline import Pipeline, PipelineConfig
from edgecore.publisher import EventPublisher
from labcore import taxonomy as tax
from labcore.registry import load_registry
from labcore.video import list_local_videos

LAB_VIDEOS = EDGE_DIR.parent / "road-damage-lab" / "data" / "videos"
UPLOAD_DIR = EDGE_DIR / "data" / "uploads"

st.set_page_config(
    page_title="Onboard — Road Survey",
    page_icon="🚌",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 1rem;}
      div[data-testid="stMetricValue"] {font-size: 1.45rem;}
      .ev-card {border-left: 4px solid #888; background: rgba(128,128,128,.09);
                padding: .45rem .6rem; margin-bottom: .4rem; border-radius: 4px;}
      .ev-head {font-weight: 600; font-size: .92rem;}
      .ev-meta {font-size: .78rem; opacity: .75; font-family: ui-monospace, monospace;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Sidebar -- what this bus is
# --------------------------------------------------------------------------- #

env = EdgeConfig.from_env()
routes = load_routes()

with st.sidebar:
    st.markdown("### 🚌 Onboard unit")

    bus_id = st.text_input("Bus ID", value=env.bus_id)
    route_ids = list(routes)
    route_id = None
    if route_ids:
        route_id = st.selectbox(
            "Route",
            route_ids,
            index=route_ids.index(env.route_id) if env.route_id in route_ids else 0,
            format_func=lambda r: routes[r].name,
            help="Position is simulated by walking this route at its nominal speed.",
        )
    else:
        st.error("No routes found in edge/routes/ — events will have no position.")
    route = routes.get(route_id) if route_id else None
    if route:
        st.caption(f"{route.length_m / 1000:.1f} km · {route.speed_kmh:.0f} km/h nominal")

    start_offset = st.slider(
        "Start offset along route (m)",
        0,
        int(route.length_m) if route else 1000,
        min(900, int(route.length_m) if route else 900),
        step=50,
        help="A 10-second clip covers only ~85 m, so the offset decides which "
        "stretch of the corridor this pass reports on.",
    )

    st.divider()
    st.markdown("### 📡 Central system")
    api_url = st.text_input("API URL", value=env.api_url)
    post = st.toggle("Post events", value=True)
    send_crops = st.toggle("Include crops", value=True)

    st.divider()
    with st.expander("Detection", expanded=False):
        registry = load_registry()
        model_ids = [s.id for s in registry]
        model_id = st.selectbox(
            "Model",
            model_ids,
            index=model_ids.index(env.model_id) if env.model_id in model_ids else 0,
        )
        spec = registry.get(model_id)
        conf = st.slider(
            "Confidence", 0.05, 0.80, float(spec.default_conf), 0.01,
            help="RDD models peak around 0.10-0.20, not the YOLO default of 0.25.",
        )
        stride = st.select_slider("Frame stride", [1, 2, 3, 4, 5], value=1)
        width = st.select_slider(
            "Downscale width", [0, 960, 1280, 1600], value=0,
            format_func=lambda w: "native" if w == 0 else f"{w}px",
        )
        max_frames = st.number_input("Max frames (0 = all)", 0, 10000, 0, step=50)

    with st.expander("Simulation", expanded=False):
        gps_noise = st.slider("GPS noise (m)", 0.0, 15.0, 0.0, 0.5,
                              help="Gaussian scatter per fix, standing in for urban drift.")
        speed_jitter = st.slider("Speed jitter", 0.0, 0.3, 0.0, 0.05)
        phase = st.number_input(
            "Phase", 0, 5, 0,
            help="Which frames of each stride group to sample. Only has an "
            "effect at stride 2 or more.",
        )
        backdate = st.text_input("Backdate pass", value="",
                                 placeholder="-2h, -1d, or blank for now")

    if phase and stride < 2:
        st.warning("Phase does nothing at stride 1 — every phase sees the same frames.")


# --------------------------------------------------------------------------- #
# Source
# --------------------------------------------------------------------------- #

st.markdown("## Onboard system — live road damage detection")

upload = None
choice = None

src_col, act_col = st.columns([3, 1])
with src_col:
    tab_up, tab_local = st.tabs(["Upload a clip", "Pick a local clip"])
    with tab_up:
        upload = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm"])
    with tab_local:
        # The lab directory accumulates re-encoded duplicates of each clip
        # (name_1787740606.mp4); collapse them to one entry per source.
        seen, picks = set(), []
        for p in list_local_videos(LAB_VIDEOS):
            base = p.stem.split("_17")[0]
            if base not in seen:
                seen.add(base)
                picks.append(p)
        if picks:
            choice = st.selectbox("Clip", picks, format_func=lambda p: p.name)
        else:
            st.caption(f"No clips found in {LAB_VIDEOS}")

source_path: str | None = None
if upload is not None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / upload.name
    dest.write_bytes(upload.getbuffer())
    source_path = str(dest)
elif choice is not None:
    source_path = str(choice)

with act_col:
    st.write("")
    st.write("")
    go = st.button("▶ Start pass", type="primary", width="stretch",
                   disabled=source_path is None)
    if source_path:
        st.caption(Path(source_path).name)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def event_card(ev) -> str:
    colour = tax.get(ev.damage_type).hex
    where = f"{ev.lat:.5f}, {ev.lon:.5f}" if ev.has_fix else "no fix"
    return (
        f"<div class='ev-card' style='border-left-color:{colour}'>"
        f"<div class='ev-head' style='color:{colour}'>{ev.damage_label}"
        f" · {ev.confidence:.0%}</div>"
        f"<div class='ev-meta'>{where}<br>"
        f"{ev.captured_at[11:19] if ev.captured_at else '—'} · "
        f"{ev.frames_seen} frames · {ev.area_pct_frame:.1f}% of frame</div>"
        f"</div>"
    )


if go and source_path:
    cfg = PipelineConfig(
        model_id=model_id,
        conf=conf,
        stride=int(stride),
        phase=int(phase),
        width=int(width),
        max_frames=int(max_frames),
        bus_id=bus_id,
        route_id=route_id or "",
        capture_crops=send_crops,
        annotate=True,
    )

    gps = None
    if route is not None:
        from edgecore.gps import RouteReplay

        gps = RouteReplay(
            route,
            start_offset_m=float(start_offset),
            start_time=parse_when(backdate or None),
            gps_noise_m=float(gps_noise),
            speed_jitter=float(speed_jitter),
        )

    pub = None
    if post:
        pub = EventPublisher(
            api_url=api_url,
            spool_dir=env.spool_dir,
            batch_size=1,  # post as they fire; this is what makes window 2 live
            include_crops=send_crops,
        )
        reachable = pub.ping()
        if not reachable:
            st.warning(
                f"**{api_url} is not reachable.** Events will spool to disk and "
                "drain on the next pass that connects — nothing is lost.\n\n"
                "Most likely the control room is on a different port. Start it "
                "with `python run.py` in `server/`, and make sure the **API URL** "
                "above matches the port it prints."
            )
        pub.start()

    with st.spinner("Loading model…"):
        pipe = Pipeline(cfg, gps=gps)

    info = pipe.describe()
    st.caption(
        f"{info['model_name']} · {info['device']} · conf {info['conf']} · "
        f"stride {stride}" + (f" · {route.name}" if route else " · no route")
    )

    prog = st.progress(0.0)
    kpi = st.columns(5)
    k_frames, k_events, k_open, k_fps, k_sent = (c.empty() for c in kpi)

    view_col, feed_col = st.columns([2.2, 1])
    with view_col:
        canvas = st.empty()
    with feed_col:
        st.markdown("#### Events sent")
        feed = st.container(height=560)

    events: list = []
    t0 = time.perf_counter()
    last_draw = 0.0

    for upd in pipe.run(source_path):
        if pub is not None and upd.new_events:
            pub.publish(upd.new_events)
        for ev in upd.new_events:
            events.append(ev)
            with feed:
                st.markdown(event_card(ev), unsafe_allow_html=True)
                if ev.crop_jpeg:
                    st.image(ev.crop_jpeg, width="stretch")

        # Throttle the redraw: Streamlit reruns are far slower than inference,
        # and at 60 fps the UI, not the model, becomes the bottleneck.
        now = time.perf_counter()
        if upd.annotated is not None and (now - last_draw > 0.06):
            canvas.image(upd.annotated, channels="BGR", width="stretch")
            last_draw = now

        if upd.total_steps:
            prog.progress(min(1.0, upd.progress))
        k_frames.metric("Frames", upd.step)
        k_events.metric("Events", len(events))
        k_open.metric("Open tracks", pipe.tracker_.open_tracks if pipe.tracker_ else 0)
        k_fps.metric("FPS", f"{upd.fps:.0f}")
        sent = sum(e.wire_bytes for e in events)
        k_sent.metric("Sent", f"{sent / 1024:.0f} KB")

    prog.progress(1.0)
    if pub is not None:
        pub.close()

    # ---- the bandwidth argument, measured
    wall = time.perf_counter() - t0
    clip_bytes = Path(source_path).stat().st_size
    sent = sum(e.wire_bytes for e in events)
    stats = pipe.stats()

    st.success(
        f"Pass complete — {len(events)} events from {stats['steps']} frames "
        f"in {wall:.1f}s"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Clip size", f"{clip_bytes / 1e6:.1f} MB")
    c2.metric("Sent to control room", f"{sent / 1024:.0f} KB")
    c3.metric(
        "Bandwidth saved",
        f"{100 * (1 - sent / clip_bytes):.1f}%",
        help="Video never leaves the bus — only one event and one crop per "
        "defect. This is the edge-processing argument, measured on this pass.",
    )

    with st.expander("Run detail"):
        st.json(
            {
                "pipeline": stats,
                "publisher": pub.stats.as_dict() if pub else None,
                "gps": gps.describe() if gps else None,
            }
        )
        if pub is not None and pub.spool_depth:
            st.warning(
                f"{pub.spool_depth} batch(es) spooled to {pub.spool_dir} — "
                "they will drain on the next pass that connects."
            )

elif source_path is None:
    st.info("Upload a clip or pick a local one to start a pass.")
