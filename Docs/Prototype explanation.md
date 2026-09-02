# Prototype explanation

What this project is, what it contains, how each piece works, what it is built
with, and the complete path a pothole takes from a frame of bus footage to a
ranked line in a maintenance queue.

Companion documents: [Base.md](Base.md) has the SIH problem statement and the
architecture diagrams, [V1-Plan.md](V1-Plan.md) has the build plan and the
phase-by-phase verification log, [Problem explanation.md](Problem%20explanation.md)
has the background research on bus camera fleets.

---

## 1. What it is

Public transport buses already drive every major road in a city every day, and
most of them already carry cameras. Today those cameras only record incidents.
This prototype turns them into **road-condition sensors**: an onboard unit
watches the forward view, detects road damage, tags each defect with position
and time, and posts **one small event per defect** to a central platform that
clusters repeat sightings from different buses into physical defects, maps them,
and ranks a maintenance queue.

Built for SIH problem statement **26124** — *AI-Powered Mobile Urban
Intelligence Platform Using Public Transport Fleet*. The prototype implements
the left branch of the target architecture: **camera feed → road damage
detection → metadata → central DB → GIS dashboard**. Traffic analysis, ANPR,
waterlogging, signboards and pedestrian detection are out of scope for V1.

The two ideas the prototype exists to demonstrate:

| Idea | How it shows up |
|---|---|
| **Edge processing saves bandwidth** | Video never leaves the bus. A defect visible for 2 s at 30 fps is ~60 detection rows; it leaves as 1 JSON event + 1 ~9 KB JPEG crop. Measured live in window 1: **39 KB sent against a 12.3 MB clip — 99.7 % saved**. |
| **Fleet redundancy beats model confidence** | The map plots *defects*, not sightings. "Seen 8 times by 3 buses over 4 days" is a stronger claim than "the model said 0.61". A one-off stays `unconfirmed` and greys out; a corroborated one promotes itself to `open`. |

---

## 2. What the repository contains

```
road-damage-lab/        the research bench — where the model was chosen
  labcore/              taxonomy, registry, detector, video, draw, survey
  models.yaml           27 registered YOLO checkpoints
  app.py, run_cli.py    Streamlit bench + CLI benchmark
  docs/                 the model-comparison writeup with visual evidence

edge/                   ONBOARD SYSTEM  →  window 1  (Streamlit :8501)
  edgecore/
    config.py           bus identity, API URL, thresholds (env + CLI)
    pipeline.py         clip → annotated frames + road events (UI-agnostic)
    events.py           track lifecycle → one RoadEvent per defect
    gps.py              route interpolation / GPX-CSV replay
    publisher.py        background POST, retry, offline disk spool
  routes/*.geojson      three traced Chennai corridors
  app_edge.py           Streamlit onboard UI — the demo front door
  run_edge.py           headless CLI twin
  seed_demo.py          15 scripted passes that pre-populate fleet history

server/                 CENTRALIZED SYSTEM  →  window 2  (FastAPI :8010)
  app/
    main.py db.py models.py schemas.py taxonomy.py reset.py
    clustering.py       sighting → physical defect assignment
    routers/            events, defects, fleet, reports, admin
  static/               dark control-room dashboard (Leaflet, vendored)
  tests/                16 unit tests over the clustering rules
  run.py                launcher: clean slate by default, port 8010

Docs/                   problem statement, plan, architecture diagrams, this file
```

The three components are layered, not duplicated: `edge/` and `server/` both
import `labcore` rather than redefining the detector or the damage taxonomy, so
a pothole is the same red in the video overlay and on the map.

---

## 3. Tech stack

| Layer | Technology | Why |
|---|---|---|
| Detection | **Ultralytics YOLO**, default checkpoint `rdd-yolo12s` (YOLOv12s trained on RDD2022) at `conf 0.15` | Chosen by measured comparison — see [model-comparison](../road-damage-lab/docs/model-comparison-2026-08-26.md) |
| Tracking | **ByteTrack** (`bytetrack.yaml`, via `model.track(persist=True)`) | Gives each defect a persistent id so many frames collapse into one event |
| Vision I/O | **OpenCV** (`cv2`) — frame iteration, stride, resize, crop, JPEG encode, overlay drawing | No second imaging dependency |
| Weights hosting | **Hugging Face Hub** (`huggingface_hub`), cached under `road-damage-lab/weights/hf` | Lazy download on first use |
| Model registry | **PyYAML** over `models.yaml` | Adding a model never means touching app code |
| Onboard UI | **Streamlit** ≥ 1.50 | Upload-and-run demo front door on `:8501` |
| Onboard networking | **`urllib` from the standard library**, on a worker thread | One less dependency on a unit that has to boot in a bus |
| API | **FastAPI** + **Uvicorn** on `:8010` | Async server, automatic OpenAPI docs at `/docs` |
| Validation | **Pydantic v2** (`schemas.py`) | Partial events are stored rather than rejected |
| ORM / DB | **SQLAlchemy 2.0** with **SQLite** by default, **PostgreSQL** via `DATABASE_URL` (`psycopg`) | Runs on a laptop with nothing installed; no Postgres-only types are used, so the two are interchangeable |
| Geospatial | Plain `lat`/`lon` float columns + bounding-box prefilter + haversine | **No PostGIS in V1** — adequate at prototype scale, avoids a painful Windows install |
| Dashboard | Static **HTML + CSS + vanilla JS** with **Leaflet** (vendored, not CDN), **OpenStreetMap** tiles darkened in CSS | No build step, no second runtime, and a demo must not depend on venue wifi for its own map library |
| Live updates | **2 s polling** on a cursor (`GET /api/events?since=<last_id>`) | Cheap, debuggable, survives a mid-demo server restart — SSE would look identical to the audience |
| Tests | **pytest** against an in-memory SQLite database | 16 tests covering clustering, confirmation and auto-close |
| Language | **Python 3.10+** (`from __future__ import annotations`, `X | None` unions) | — |

Footage source: [RADRoad Anomaly Detection dataset](https://www.kaggle.com/datasets/rohitsuresh15/radroad-anomaly-detection/data)
(Kaggle), plus a stock rural-road clip.

---

## 4. The complete flow

```
   ┌────────────────────────── ON THE BUS (edge/) ──────────────────────────┐
   │                                                                        │
   │  1  video clip / camera                                                │
   │        │  labcore.video.VideoSource  — stride, phase, optional resize  │
   │        ▼                                                               │
   │  2  YOLO inference + ByteTrack        labcore.detector.Detector.infer  │
   │        │  → list[Detection]  (xyxy, conf, cls_id, track_id, canon)     │
   │        ▼                                                               │
   │  3  event lifecycle                   edgecore.events.EventTracker     │
   │        │  open → confirm at 3 sightings → close after 15 missed frames │
   │        │  peak-area frame = closest approach = best crop + best fix    │
   │        ▼  one RoadEvent per defect, with a padded 640 px JPEG crop     │
   │  4  metadata stamp                    edgecore.gps.RouteReplay         │
   │        │  frame_idx → (lat, lon, bearing, speed, ISO timestamp)        │
   │        ▼                                                               │
   │  5  publisher (worker thread)         edgecore.publisher               │
   │        │  POST succeeds → done                                         │
   │        └  POST fails     → spool to disk, drain on next connection     │
   └────────────────────────────────┬───────────────────────────────────────┘
                                    │  HTTP  POST /api/events
                                    │  {"events":[{…, "crop_b64": "…"}]}
   ┌────────────────────────────────▼─── CONTROL ROOM (server/) ────────────┐
   │  6  ingest            routers/events.py                                │
   │        │  idempotent on event_uid · crop → /media/crops/<uid>.jpg      │
   │        │  row inserted into `events` (the raw log, never merged)       │
   │        ▼                                                               │
   │  7  clustering        clustering.assign_defect                         │
   │        │  bbox prefilter → haversine ≤ 15 m × type scale, same type    │
   │        │  match  → sightings++, distinct_buses, centroid, best crop    │
   │        │  no match → new defect, status=unconfirmed                    │
   │        │  ≥2 buses or ≥3 sightings → status=open                       │
   │        ▼                                                               │
   │  8  bus roster auto-updated; `defects` table is what the map plots     │
   │        ▼                                                               │
   │  9  dashboard         static/app.js, polling every 2 s                 │
   │        │  KPI tiles · Leaflet map · live DB table · defect detail      │
   │        │  arrival ping · severity heatmap · client-side alerts         │
   │        ▼                                                               │
   │ 10  outputs           GET /api/report  ·  GET /api/report.csv          │
   │        ranked maintenance queue: severity × confidence × corroboration │
   │                                   × age × size                         │
   │ 11  closure           POST /api/admin/close-stale                      │
   │        an open defect unreported for 14 days *on a route still being   │
   │        driven* is presumed repaired — the loop closes itself           │
   └────────────────────────────────────────────────────────────────────────┘
```

**One timing detail worth knowing while demoing:** an event fires when its track
*closes*, i.e. `MISS_TOLERANCE` (15) processed frames after the defect leaves
the frame. So the pin lands on the map about half a second after you see the
pothole pass in window 1. That lag is real behaviour, not a bug.

---

## 5. Component deep dive

### 5.1 `road-damage-lab/` — the bench

Unchanged by the rest of the project; it is where the model was chosen and it
stays as the evidence behind that choice. `edge/` and `server/` import
`labcore` from it.

**[`labcore/taxonomy.py`](../road-damage-lab/labcore/taxonomy.py) — the canonical vocabulary.**
Different checkpoints name and *order* the same classes differently (`ozair23`
has longitudinal and alligator swapped relative to the `rdd-*` family; `rezzzq`
uses raw RDD codes `D00/D10/D20/D40`). A raw class id therefore means nothing on
its own. Every model maps its ids onto canonical keys — explicitly via
`class_map:` in `models.yaml`, or by name matching as a fallback — and
everything downstream works in canonical terms.

| key | RDD code | severity | colour | notes |
|---|---|---:|---|---|
| `pothole` | D40 | 1.00 | red `#e74c3c` | safety-critical |
| `alligator_crack` | D20 | 0.75 | orange `#f39c12` | fatigue cracking, precursor to potholing |
| `longitudinal_crack` | D00 | 0.40 | yellow `#f1c40f` | runs with the direction of travel |
| `transverse_crack` | D10 | 0.35 | blue `#3498db` | runs across the carriageway |
| `crack` | — | 0.40 | teal | models that don't split by orientation |
| `other` | — | 0.30 | purple | unspecified damage |
| `repair` | — | 0.10 | green | patched surface — context, not a defect |
| `unknown` | — | 0.30 | grey | unmapped class |

Colours are ordered worst-first so they read as a heat ramp. The server
re-exports this module ([`server/app/taxonomy.py`](../server/app/taxonomy.py))
and publishes it at `GET /api/taxonomy`; the dashboard styles itself from that
payload instead of hardcoding hex, which is what keeps both windows in agreement.

**[`labcore/registry.py`](../road-damage-lab/labcore/registry.py)** parses
`models.yaml` into `ModelSpec` objects that know how to resolve their own
weights (`hf` / `local` / `ultralytics`), downloading lazily and caching under
`weights/`. A typo'd canonical key in a `class_map:` fails at parse time rather
than silently colouring boxes grey. **27 checkpoints are registered**, 20
enabled — eight RDD2022 architectures trained on identical data (v5s → v12m),
five crack-segmentation sizes, several pothole specialists, and a COCO baseline.

**[`labcore/detector.py`](../road-damage-lab/labcore/detector.py)** is a thin
uniform wrapper over Ultralytics YOLO. Every checkpoint — detect or segment,
v8/v9/v11/v12, HF or local — comes out as the same `list[Detection]` carrying
`xyxy`, `conf`, `cls_id`, raw `label`, `track_id` and the canonical `canon` key,
plus helpers like `area_pct(w, h)`. It also handles device resolution, FP16, the
Ultralytics ≥ 8.4 `quantize` rename, and `reset_tracker()` between clips.

**[`labcore/video.py`](../road-damage-lab/labcore/video.py)** iterates frames
with `stride`, `start_frame` (= phase) and `max_frames`, and can resize.
**[`labcore/draw.py`](../road-damage-lab/labcore/draw.py)** draws boxes, id and
confidence chips, trails and the HUD in plain OpenCV.
**[`labcore/survey.py`](../road-damage-lab/labcore/survey.py)** collapses a
whole pass into a road-condition report with a severity-weighted damage score
and a blunt letter grade — this is the *batch* twin of what the edge does
incrementally.

**Model choice, measured.** On the 692-frame stock clip `rdd-yolo12s` found the
most distinct potholes of five RDD architectures (47 vs. 38–44). On real survey
footage it loses on raw pothole count to a pothole-only specialist (5 vs. 8) but
wins overall because it also caught 7 alligator cracks and 2 longitudinal cracks
the specialist is architecturally blind to. The writeup is also honest about the
weak spot: under a flyover shadow it fires `pothole 0.29` on plain pavement —
which is precisely the kind of call the server's confirmation rule is designed
to leave `unconfirmed`.

### 5.2 `edge/` — the onboard system (window 1)

`labcore` gives per-frame detections. The edge adds the three things an onboard
unit needs that a benchmark does not.

**Event lifecycle — [`edgecore/events.py`](../edge/edgecore/events.py).**
The one genuinely new piece of onboard logic. `EventTracker.update(frame_idx,
detections, frame)` is fed once per processed frame and returns the events that
closed on that frame. Per `(canon, track_id)`:

1. **Open** on first sighting.
2. **Confirm** at `MIN_FRAMES = 3` sightings — below that it is tracker flicker.
   (Deliberately stricter than the lab's report, which uses 2: a false event
   costs somebody a work order rather than a row in a table.)
3. **Close** after `MISS_TOLERANCE = 15` processed frames without a sighting.
4. On close, if confirmed, **emit one `RoadEvent`**; `flush()` closes whatever is
   still open at the end of the clip.

The **peak-area frame** — the frame where the box was largest — is the closest
the bus ever came to the defect, so it gives simultaneously the best crop and
the most accurate position estimate. GPS, timestamp, bbox and image are all
sampled there. Crops are padded 18 %, capped at a 640 px long edge and JPEG'd at
quality 85 (~9 KB each).

The tracker also keeps honesty counters — `boxes_seen`, `boxes_unassigned`
(boxes the tracker never gave an id, which are *not* promoted to events),
`tracks_opened`, `tracks_dropped_as_flicker`, `events_emitted` — which surface
in the run summary.

A `RoadEvent` carries identity (`event_uid`, `bus_id`, `route_id`), what was
seen (`damage_type`, `severity`, peak and mean `confidence`, `area_pct_frame`),
where and when (`lat`, `lon`, `bearing`, `speed_kmh`, `captured_at`), full
provenance back to the footage (`frame_idx`, `first_frame`, `last_frame`,
`frames_seen`, `track_id`, `bbox`, `source_clip`, `model_id`) and the crop.
`wire_bytes` reports what the event costs to send — the bandwidth headline.

**Simulated GPS — [`edgecore/gps.py`](../edge/edgecore/gps.py).**
The prototype has dashcam footage but no telemetry, so `RouteReplay` walks a
traced route polyline at a nominal speed: frame index → elapsed seconds →
distance travelled → interpolated point on the line, indexed by cumulative
haversine distance. Three real Chennai corridors are stored as GeoJSON
`LineString`s in [`edge/routes/`](../edge/routes/) — 21G Anna Salai, 570 OMR,
51M GST Road — so events land *on real roads*, which is what makes the map and
any route-level aggregation credible. Perturbation knobs (`gps_noise_m`,
`speed_jitter`, `start_offset_m`, `parse_when("-2h" | "-1d" | ISO)`) simulate
urban drift, traffic and repeat passes. `TrackReplay` reads a real GPX or CSV
sidecar behind the identical interface, so swapping in real telemetry later is a
config change rather than a rewrite.

**Publisher — [`edgecore/publisher.py`](../edge/edgecore/publisher.py).**
Posts from a worker thread so inference never waits on a network round trip.
On failure the batch is **written to a disk spool** (`edge/spool/`, named by
timestamp so it drains in order) and retried after the next successful POST —
the honest answer to "what happens in a tunnel", which is the first question
anyone asks about an onboard system. It tracks `posted`, `duplicates`,
`rejected`, `spooled`, `drained`, `failed_batches` and `bytes_sent`, and `ping()`
checks `/api/health` before a pass so the UI can warn instead of silently
spooling.

**Pipeline — [`edgecore/pipeline.py`](../edge/edgecore/pipeline.py).**
The UI-agnostic run loop: `VideoSource` → `Detector.infer` → `EventTracker` →
GPS stamp, yielding `FrameUpdate(frame_idx, step, total_steps, detections,
new_events, annotated, infer_ms, fps)`. Both front-ends are thin wrappers over
it — the same split that lets the lab's app and `run_cli.py` share `labcore`.

**Front-ends.**
[`app_edge.py`](../edge/app_edge.py) is the Streamlit demo door: sidebar sets
what the *bus* is (id, route, start offset, API URL, model, confidence, stride,
simulation knobs), the main panel streams annotated frames with a live
events-sent feed and crop thumbnails, and the footer shows clip size vs. bytes
sent vs. percentage saved. Redraws are throttled to ~16 fps because Streamlit
reruns are slower than inference. [`run_edge.py`](../edge/run_edge.py) is the
headless twin used for seeding and batch development; it writes `events.json`,
`events.geojson` (droppable straight into geojson.io to verify fixes land on
road), `run.json` and a `crops/` folder. [`seed_demo.py`](../edge/seed_demo.py)
runs a fixed 15-pass plan — 3 clips × 3 routes × several buses, phases 0/1/2 at
stride 3, backdated from −7 d to −5 h, GPS noise 6 m — loading the model once
and reusing it across passes.

**Getting repeat passes out of three clips.** The dataset has no repeat
traversals, so one clip = one traversal and a repeat pass is the same clip
replayed as a different bus at a different time — *perturbed*, because a
bit-identical replay would put every defect at exactly `sightings=N` and the
confirmation count would carry no information. `--stride N --phase K` samples
*different frames* and therefore genuinely detects differently. This was
measured, not assumed: at stride 1 phases 0 and 1 come back with identical
confidences to two decimal places (`run_edge.py` warns if you ask for a phase at
stride 1); at stride 3 on `62_10-07-2023.mp4` the three phases give **5 / 4 / 5
events**, and phase 1 misses a pothole the other two catch.

### 5.3 `server/` — the centralized system (window 2)

**Two tables doing two jobs.** [`models.py`](../server/app/models.py):

`events` — the raw log. One row per sighting, per bus, per pass. Never merged,
never deleted. This is the evidence trail: which bus saw what, when, at what
confidence, with the crop that proves it, and which frame of which clip it came
from.

`defects` — the physical thing in the road. Many events collapse into one
defect, and this is what the map plots. Without it, a route driven four times a
day stacks forty pins on one pothole and the KPI tile counts *sightings* rather
than problems.

| `defects` column | meaning |
|---|---|
| `damage_type`, `severity` | canonical key and its taxonomy weight |
| `lat`, `lon` | running centroid of member events |
| `sightings`, `distinct_buses` | corroboration |
| `first_seen`, `last_seen` | age, and staleness for auto-close |
| `max_confidence`, `peak_area_pct`, `best_crop` | best evidence, not latest |
| `status` | `unconfirmed` \| `open` \| `repaired` |

Plus `buses` (roster, auto-maintained on ingest — no separate registration step)
and `routes` (id, name, city, speed, length, GeoJSON polyline, re-seeded from
`edge/routes/` on every startup).

**Ingest — [`routers/events.py`](../server/app/routers/events.py).**
`POST /api/events` takes `{"events": [...]}` or a bare list. Per event: reject
duplicates on `event_uid` (a publisher retry after a timeout it cannot
distinguish from a failure must not double-count), decode and store the base64
crop under `server/media/crops/<uid>.jpg` (capped at 2 MB; a bad crop must not
sink the event), insert the row, flush to get an id, cluster it, update the bus
roster. One bad event is rolled back and counted as `rejected` rather than
failing the whole batch. The response reports `accepted / duplicates / rejected`
plus the ids touched.

**Clustering — [`clustering.py`](../server/app/clustering.py).**
On ingest: *is there a non-repaired defect of the same `damage_type` within the
radius?* A bounding-box query on the indexed `lat`/`lon` columns is the cheap
prefilter; haversine does the actual circle. The base radius is **15 m**, sized
for GPS drift rather than pothole geometry, scaled per type because cracks are
linear features a tracker can split along their length (pothole ×1.0, alligator
×1.2, transverse ×1.3, generic crack ×1.5, longitudinal ×1.6).

- **No match** → create a defect, `status = unconfirmed`.
- **Match** → attach; centroid becomes a sighting-weighted running mean (later
  passes refine the position rather than the newest fix winning outright),
  `sightings++`, `distinct_buses` recomputed from the raw log, `first_seen` /
  `last_seen` widened, the *highest-confidence* crop kept.
- **Promotion** → `distinct_buses ≥ 2` **or** `sightings ≥ 3`. An open defect is
  never demoted.

`close_stale()` marks an open defect `repaired` if it has not been seen for 14
days **and its route is still being driven** — otherwise a route the fleet
stopped serving would report all its potholes fixed. That distinction is the
whole reason the rule is safe to run automatically.
`recluster_all()` rebuilds every defect from the raw event log, so retuning the
radius or the confirmation rule is a re-run rather than a migration.

**Reports — [`routers/reports.py`](../server/app/routers/reports.py).**
A dashboard tells you where the damage is; a work order needs it *ranked*.

```
priority = severity × max_confidence × corroboration × age × size × 100
           corroboration = min(2.0, 1 + 0.25 × (distinct_buses − 1))
           age           = min(2.0, 1 + days_since_first_seen / 30)
           size          = min(1.5, 1 + peak_area_pct / 20)
```

Multiplicative on purpose: a defect that is severe but unconfirmed, or confirmed
but trivial, should not float to the top on one strong factor. Corroboration and
age are capped so an old, much-reported minor crack cannot outrank a fresh
pothole. Only `open` defects are queued by default — unconfirmed ones are
candidates, repaired ones are done. `GET /api/report.csv` hands the same queue,
with an OpenStreetMap deep link per row, to whoever actually schedules crews.

**Dashboard — [`static/`](../server/static/).**
Dark control-room shell: KPI tiles on top, Leaflet map, live database table
directly below, defect detail panel, alert toasts. Notable behaviours:

- Markers are coloured from `/api/taxonomy`, sized by severity nudged by
  sighting count, greyed and hollow when `unconfirmed`, dashed green when
  `repaired`. Route polylines underlay them.
- New arrivals get a brief expanding ring. Markers render as **SVG, not canvas** —
  with `preferCanvas: true` circle markers have no `_path` element and the ping
  animated nothing.
- Tiles are plain OSM inverted in CSS, scoped to the tile pane so marker colours
  stay true. CARTO's dark basemap now watermarks **API KEY REQUIRED** across
  every tile and a demo should not depend on a key.
- The table is a literal database view — columns mirror the schema, newest
  first, `defects` / `sightings` toggle, click a row to pan the map.
- Events arrive on a cursor (`?since=<last_id>`); **defects are re-read whole**
  because clustering mutates existing rows rather than only appending.
- If `latest_event_id` goes *backwards* the server was reset while the page
  stayed open, so the client clears its cursor and state instead of silently
  skipping every new event.
- If nothing has been clustered yet but sightings exist, it opens on the
  Sightings view — an empty Defects panel reads as a broken dashboard.
- Alerts are a client-side severity threshold (`severity ≥ 0.75` and
  `confidence ≥ 0.55`). Real SMS/WhatsApp delivery is an integration cost a
  prototype cannot justify, so this shows the rule *firing* rather than
  pretending a message was sent.

**Reset — [`reset.py`](../server/app/reset.py) and [`run.py`](../server/run.py).**
`run.py` **starts empty by default**: test passes otherwise pile up into a map
nobody can read, so each start clears reported data — sightings, the defects
clustered from them, the bus roster and stored crops. Routes survive; they are
configuration. `--keep` preserves everything (use it on demo day so seeded
history survives a restart), and `POST /api/admin/reset` clears without
restarting — the dashboard notices and empties itself. Deletes are written in
SQLAlchemy rather than by removing the SQLite file, so they behave the same
against Postgres.

**Why port 8010.** 8000 is crowded. If something else is already listening
there, the onboard unit posts its events into whatever that happens to be and
the failure looks like a network problem rather than a misconfiguration. Both
halves default to 8010, and `run.py` refuses to start if the port is taken.

---

## 6. API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/events` | Ingest a batch of sightings + base64 crops. Idempotent on `event_uid`. |
| `GET` | `/api/events?since=&limit=&bus_id=&route_id=&damage_type=&order=` | The live feed. Ascending when following a cursor, so a burst larger than the limit can't skip the middle. |
| `GET` | `/api/events/{id}` | One sighting. |
| `GET` | `/api/events/by-defect/{defect_id}` | Sighting history for a defect — who saw it, when, how sure. |
| `GET` | `/api/defects?bbox=&damage_type=&status=&route_id=&min_sightings=&limit=` | Clustered defects, worst first. |
| `GET` | `/api/defects/{id}` | One defect. |
| `GET` | `/api/routes`, `/api/routes.geojson` | Route metadata and polylines for the map underlay. |
| `GET` | `/api/buses` | Fleet roster with per-bus event counts. |
| `GET` | `/api/taxonomy` | Damage keys, labels, colours, severities — the dashboard styles itself from this. |
| `GET` | `/api/stats` | KPI tiles: events, defects, by type, by status, buses, routes, `latest_event_id`, severity-weighted damage score. |
| `GET` | `/api/report`, `/api/report.csv` | Ranked maintenance queue, JSON or CSV download. |
| `POST` | `/api/admin/reset` | Clear reported data; keep routes. |
| `POST` | `/api/admin/recluster` | Rebuild every defect from the raw log. |
| `POST` | `/api/admin/close-stale` | Run the auto-close rule. |
| `GET` | `/api/admin/clustering` | What the clustering layer is currently tuned to. |
| `GET` | `/api/health` | Liveness + redacted database URL; what the publisher pings. |
| `GET` | `/`, `/static/*`, `/media/*` | Dashboard, assets, stored crops. |

Interactive OpenAPI docs at `http://127.0.0.1:8010/docs`.

---

## 7. Configuration

Precedence on the edge is **CLI > environment > default**, so a demo runs off
flags while a real deployment sets `ROADSURVEY_*` once in the unit's environment.

| Variable | Default | Effect |
|---|---|---|
| `ROADSURVEY_BUS_ID` | `BUS_001` | Onboard unit identity |
| `ROADSURVEY_ROUTE_ID` | — | Which corridor to simulate position along |
| `ROADSURVEY_API_URL` | `http://127.0.0.1:8010` | Where events are posted |
| `ROADSURVEY_API_TIMEOUT_S` | `5.0` | POST timeout |
| `ROADSURVEY_BATCH_SIZE` | `1` | 1 = post as events fire, which is what makes window 2 live |
| `ROADSURVEY_SPOOL_DIR` | `edge/spool` | Offline spool location |
| `ROADSURVEY_MODEL_ID` / `_CONF` | `rdd-yolo12s` / model default `0.15` | Detector selection |
| `DATABASE_URL` | `sqlite:///server/data/roadsurvey.db` | e.g. `postgresql+psycopg://user:pw@localhost/roadsurvey` |
| `ROADSURVEY_CLUSTER_RADIUS_M` | `15.0` | Base clustering radius |
| `ROADSURVEY_CONFIRM_BUSES` | `2` | Distinct buses required to promote to `open` |
| `ROADSURVEY_CONFIRM_SIGHTINGS` | `3` | ...or sightings from a single bus |
| `ROADSURVEY_STALE_DAYS` | `14` | Silence before an open defect is presumed repaired |
| `ROADLAB_DIR` | `../road-damage-lab` | Where the server finds `labcore` |

Edge-side tuning constants live in code: `MIN_FRAMES = 3`,
`MISS_TOLERANCE = 15`, `CROP_PAD = 0.18`, `CROP_MAX_EDGE = 640`,
`CROP_JPEG_QUALITY = 85`.

---

## 8. Running it

Torch first, matching your CUDA version, then everything else:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r road-damage-lab/requirements.txt
pip install -r edge/requirements.txt
pip install -r server/requirements.txt
```

**Window 2 — the control room:**

```bash
cd server
python run.py            # starts with an empty map, http://127.0.0.1:8010
python run.py --keep     # ...or keep what was already reported
```

**Window 1 — the bus:**

```bash
cd edge
streamlit run app_edge.py    # http://127.0.0.1:8501
```

Pick a clip, press **Start pass**, watch pins land in window 2.

Optional but worth doing before showing anyone — give the map a fleet history so
the live pass lands on defects that already exist (seed *after* starting the
server, since seeded history is reported data):

```bash
cd edge
python seed_demo.py          # 15 passes, ~2 min
```

Headless single pass, and the tests:

```bash
python run_edge.py -s ../road-damage-lab/data/videos/62_10-07-2023.mp4 \
  --route route-21g --bus BUS_002 --stride 3 --phase 1 --at=-1d \
  --gps-noise 6 --speed-jitter 0.1 --post

cd server && python -m pytest tests/     # 16 tests
```

---

## 9. What is real and what is simulated

Stated plainly rather than left for a reviewer to find.

**Real** — the footage, the detections, the tracking, the event lifecycle, the
clustering and confirmation logic, the bandwidth reduction, the offline spool,
the ranking, the database.

**Simulated** — GPS position, timestamps, that there is more than one bus, the
multi-day history, and repair events.

**Known limitations, stated rather than solved:**

- A 15 m radius chosen to absorb GPS drift will merge two genuinely separate
  potholes 10 m apart.
- Simulated fixes are exact; real GPS drifts 5–10 m in an urban canyon, which is
  what forces the radius to be generous in the first place.
- Route corridors are hand-plotted to roughly ±30 m, not traced from Overpass.
- **Auto-close is not demonstrable from this footage** — nothing in the clips
  ever gets repaired. The rule ships with unit tests instead
  ([`tests/test_clustering.py`](../server/tests/test_clustering.py)), including
  the two cases that matter: a stale defect on an *active* route closes, and one
  on a dormant route does not, because absence of evidence is only evidence when
  somebody is still looking.

## 10. Measured outcomes

| Checkpoint | Result |
|---|---|
| Event count sane vs. the lab | Exact — 412 boxes, 208 unassigned, 14 tracks vs. the lab's 14 defects; 12 events after flicker rejection; the worst defect resolves to the same peak frame (135) and confidence (0.69) |
| Fixes land on real road; phases differ | Fixes on Anna Salai; `--phase` does nothing at stride 1, at stride 3 the phases give 5/4/5 events |
| Rows land; spool survives restart | 12 events + 12 crops (105 KB) stored; server killed mid-run spooled 10 batches which drained on reconnect; re-post returned 4 duplicates, 0 accepted |
| Upload → stream → post | 60-frame pass, 3 events, **39 KB sent against a 12.3 MB clip — 99.7 % saved** |
| Both windows live together | KPI advanced 32 → 33 → 36 → 37 → 41 → 44 during a pass, zero JS errors |
| One pin per defect | 14 sightings → **5 defects**, 4 open + 1 unconfirmed; 16 unit tests pass |
| Populated map, live pass lands on it | Seeded 92 sightings → 39 defects across 10 buses; a live pass added 11 sightings but only **1 new defect** — 5 existing defects went 3 → 4 buses, 4 alerts fired |

## 11. Out of scope for V1, and what scales from here

Not implemented: traffic analysis (the right branch of the architecture),
number-plate recognition, waterlogging, signboards, pedestrian detection. The
onboard agent has exactly one detector branch.

The architecture, however, is shaped to take them: the edge is a single
multi-class detector whose output is converted into **task-specific event
formats** per class, so adding a capability is adding classes and an event
shape, not a second inference pipeline. The same three-layer split holds — the
bus answers *"what is in front of me"*, the central platform answers *"what is
happening across the city"*, and the central platform is where deduplication,
corroboration, GIS, trends and prioritisation live. See
[Problem explanation.md](Problem%20explanation.md) for the full capability map
this prototype is one vertical slice of.
