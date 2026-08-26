# V1 — Build plan

Scope, component specs and build order for the prototype described in
[Base.md](Base.md).

**In scope:** the left branch of [Architecture.jpeg](Architecture.jpeg) — camera
feed → road damage detection → metadata → central DB → map dashboard.

**Out of scope for V1:** traffic analysis (the right branch), number-plate
recognition, waterlogging, signboards, pedestrian detection. The onboard agent
has exactly one detector branch.

---

## Target output

Two windows, side by side on one screen. This is the deliverable — everything
below exists to serve it.

**Window 1 — Onboard system.** Streamlit, `localhost:8501`. Upload a video (or
pick one from `data/videos/`), hit run. Annotated frames stream live with boxes,
track IDs and the damage legend. A panel shows events firing as tracks close,
with the simulated bus ID, route, GPS fix and the crop that was sent.

**Window 2 — Centralized system.** Browser tab, `localhost:8000`. Leaflet map
with damage markers, and **the database rendered as a live table directly below
the map**. Both update in real time as window 1 processes — new rows append,
new pins drop, KPI tiles tick up.

The two are connected only by `POST /api/events` over HTTP, which is the point:
window 1 is the bus, window 2 is the control room, and the only thing crossing
between them is a small JSON event plus a crop — never video.

**One timing detail to expect.** An event fires when its track *closes*, i.e.
`MISS_TOLERANCE` frames after the defect leaves the frame. So the pin lands on
the map about half a second after you see the pothole pass in window 1. That
lag is real behaviour, not a bug — worth mentioning while demoing rather than
having someone spot it.

---

## What already exists

[`road-damage-lab/`](../road-damage-lab/) covers stages 1–3 of
[Flow.jpeg](Flow.jpeg) and is not being rewritten. V1 imports it:

| lab component | role in V1 |
|---|---|
| `labcore.video.VideoSource` | frame iteration, stride, resize |
| `labcore.detector.Detector` | model load + `infer()` → normalised `Detection` |
| `labcore.taxonomy` | canonical damage keys, severity weights, colours |
| `labcore.survey` | per-pass report; also the shape the event record follows |
| `labcore.draw` | annotated frames + defect crops |

`labcore` has no Streamlit dependency, so the edge agent imports it directly.
The Streamlit app and `run_cli.py` stay as the research bench — they're the
evidence behind the model choice
([model-comparison](../road-damage-lab/docs/model-comparison-2026-08-26.md))
and shouldn't be disturbed.

**Model:** `rdd-yolo12s` at `conf 0.15`, tracking on (ByteTrack).

---

## The one new onboard concept: event lifecycle

`survey.build_report()` is batch — it collapses detection rows *after* a pass.
The architecture calls for real-time posting, so V1 adds a track-lifecycle
watcher that emits events *during* the pass.

Per `(canon, track_id)`:

1. **Open** on first sighting.
2. **Confirm** at `MIN_FRAMES` (≥3) sightings — below that it's tracker flicker.
3. **Close** after `MISS_TOLERANCE` (≈15) frames without a sighting.
4. On close, if confirmed, **emit one `RoadEvent`**.

A track's **peak frame** is the frame with the largest box area. On a
forward-facing camera that is the closest approach to the defect, which makes
it simultaneously the best crop and the most accurate position estimate — so
GPS, timestamp and image are all sampled there.

This is the bandwidth argument from the problem statement, and it's
quantifiable: a pothole visible for 2s at 30fps is ~60 detection rows →
**1 event + 1 JPEG crop (~30 KB)**. Video never leaves the bus.

Note the deliberate difference from `build_report`, which counts *every*
confirmed track. The edge watcher is stricter (`MIN_FRAMES` 3 vs. 2) because a
false event costs a work order, not a table row.

---

## Simulated GPS

`edgecore/gps.py` maps `frame_idx → (lat, lon, bearing, speed_kmh, timestamp)`
by interpolating along a GeoJSON `LineString` at an assumed speed.

Routes are traced from **real city bus routes** (Chennai, matching the
13.08 °N / 80.27 °E in the Flow mockup) and stored in `edge/routes/*.geojson`.
Events then land on real roads, which is what makes the heatmap and any
route-level aggregation credible.

The same interface backs a GPX/CSV sidecar, so swapping in real telemetry later
is a config change rather than a rewrite.

**Honesty note for the report:** simulated GPS is exact. Real GPS drifts 5–10 m,
especially in urban canyons — which is what forces the clustering radius below
to be generous.

---

## Event and defect model

Two tables, doing two different jobs.

### `events` — the raw log

One row per sighting per bus. Never merged, never deleted; this is the evidence
trail.

```
id, event_uid, bus_id, route_id,
damage_type, severity, confidence, area_pct_frame,
lat, lon, bearing, speed_kmh,
captured_at, frame_idx, track_id,
source_clip, model_id, crop_path,
defect_id  -> the defect it was clustered into
```

### `defects` — the physical thing in the road

One row per pothole-in-the-world. This is what the map plots.

```
id, damage_type, severity,
lat, lon,                    running centroid of member events
sightings, distinct_buses,
first_seen, last_seen,
max_confidence, best_crop_path,
status: unconfirmed | open | repaired
```

Plus `buses` and `routes` (id, name, geojson) for joins and the map overlay.

**Postgres** via `DATABASE_URL`, with a SQLite fallback so the demo runs on a
laptop with no server. **No PostGIS in V1** — a bounding-box query on plain
`lat`/`lon` floats is enough at prototype scale and avoids a painful Windows
install.

### Clustering rule

On ingest: *is there an open defect within `RADIUS` (15 m) of this event, of the
same `damage_type`?*

- **No** → create a defect, `status=unconfirmed`.
- **Yes** → attach, bump `sightings`, update `last_seen`, keep the
  highest-confidence crop, nudge the centroid.

Promotion and closure:

- `sightings ≥ 2` from `distinct_buses ≥ 2` → `open` (confirmed).
- Not seen on the last `N` passes of its route → `repaired`.

Three things fall out of this that a per-event map cannot express:

- **Confirmation.** "Seen by 4 buses on 11 passes" is a stronger claim than
  "the model said 0.61". A one-off low-confidence hit stays `unconfirmed` and
  greys out; a real defect promotes itself. This directly addresses the
  shadowed-pavement false positive at conf 0.29 flagged in the
  [model comparison](../road-damage-lab/docs/model-comparison-2026-08-26.md) —
  repeat sightings are free extra evidence, so the edge can afford to stay
  permissive.
- **Age.** `first_seen` → today gives "14 days open", which is what actually
  ranks a maintenance queue.
- **Closure.** A defect that stops being reported is a repair that verified
  itself, with nobody filing a report. That closes the loop back to *Action by
  Authorities → Issue Resolved* in the Flow diagram — and it only works because
  buses re-drive the same roads.

**Known limitation:** a 15 m radius chosen to absorb GPS drift will merge two
genuinely separate potholes 10 m apart. Stated, not solved.

---

## Layout

```
road-damage-lab/          unchanged — the research bench
edge/                     onboard system  →  WINDOW 1
  edgecore/
    events.py             track lifecycle -> RoadEvent
    gps.py                route interpolation / GPX replay
    publisher.py          batch, POST, retry, offline spool
    pipeline.py           clip -> annotated frames + events (UI-agnostic)
    config.py             bus id, route id, API URL, thresholds
  routes/*.geojson        traced bus routes
  app_edge.py             Streamlit upload-and-run UI      ← the demo front door
  run_edge.py             headless CLI — seeds phased fleet history
server/                   centralized system  →  WINDOW 2
  app/
    main.py  db.py  models.py  schemas.py
    clustering.py         event -> defect assignment
    routers/  events.py  defects.py  stats.py  fleet.py
  static/                 map + live DB table, served by FastAPI
```

`pipeline.py` holds the run loop and yields `(annotated_frame, new_events)` so
`app_edge.py` and `run_edge.py` are both thin front-ends over it — the same
split that keeps `labcore` usable by both the lab app and `run_cli.py`.

---

## Edge → server contract

`POST /api/events` takes a batch (list of events + base64 crops, or multipart).

The publisher **spools to disk on failure and retries** — an onboard system on
a moving bus loses connectivity, and a demo that dies when the server blinks is
worse than no demo. This is a small amount of code and a disproportionately
good answer to "what happens in a tunnel".

Dashboard reads `GET /api/defects?bbox=&type=&status=`,
`GET /api/events?since=` for the live feed, `GET /api/stats` for the KPI tiles.
Polling every 2s to start; SSE if there's time.

---

## Dashboard — window 2

Static HTML + Leaflet + Tailwind, served by FastAPI — no build step, no second
runtime, and it can be made to look like the dark control-room mockup in
[Flow.jpeg](Flow.jpeg).

Layout is **map on top, database table directly below it**, per the target
output.

- **KPI tiles** (top strip) — total defects, by type, open vs. unconfirmed,
  buses reporting. Tick up live.
- **Live map** — markers coloured by `taxonomy` damage colour, so a pothole is
  the same red here as in window 1's video overlay. Sized by severity, greyed
  when `unconfirmed`. Route polylines underneath. New pins drop as they arrive
  and pulse briefly so you can see which one is new.
- **Database table** (below the map) — the literal DB contents, newest first,
  auto-scrolling as rows land. Toggle between `events` (raw sightings) and
  `defects` (clustered). Columns mirror the schema so it reads as a database
  view, not a prettified summary; clicking a row pans the map to it.
- **Defect detail** — click a marker: best crop, sighting history, which buses,
  days open.
- **Heatmap layer** — severity-weighted, toggleable over the route polylines.

Live updates by polling `GET /api/events?since=<last_id>` every 2s. Cheap,
debuggable, survives a server restart mid-demo. SSE only if there's time left
over — it looks no different to the audience.

---

## Demo mode

The live demo runs through **window 1** (`app_edge.py`): upload a clip, watch it
process, watch pins land in window 2. Processing paces naturally to inference
speed, so no artificial delay is needed.

`run_edge.py` is the headless twin, used for two things the UI shouldn't do:
seeding fleet history before the demo, and batch runs during development.

### Getting repeat passes out of three clips

Clustering needs the same road driven more than once, and the dataset has no
repeat traversals. So one clip = one traversal, and repeat passes are the same
clip replayed as a different bus at a different time — **perturbed**, because a
bit-identical replay would put every defect at exactly `sightings=N` and the
confirmation count would carry no information at all.

| flag | simulates | effect |
|---|---|---|
| `--stride N --phase K` | never hitting a defect from the same frame twice | samples *different frames* → different detections |
| `--gps-noise 6` | urban GPS drift | fixes land 5–10 m apart across passes |
| `--speed-jitter 0.1` | traffic, signals, driver | shifts the frame→position mapping |
| `--at=-2h` | the bus came round again | backdates the run |
| `--start-offset M` | a different stretch of the same corridor | moves the pass along the route |

**`--phase` only works with `--stride ≥ 2`, and this was measured, not assumed.**
At stride 1, phase 0 and phase 1 sample frames 0,1,2… and 1,2,3… — essentially
the same set — and the two passes came back with *identical* damage counts and
identical confidences to two decimal places. Confirmation counts computed off
that would be pure decoration. `run_edge.py` warns if you ask for a phase at
stride 1.

At stride 3, the three phases genuinely diverge on `62_10-07-2023.mp4`:

| pass | events | mix | confidences |
|---|---:|---|---|
| `--stride 3 --phase 0` | 5 | 3 alligator, 1 pothole, 1 long. | 0.46 0.59 0.65 0.69 0.75 |
| `--stride 3 --phase 1` | 4 | 3 alligator, 1 long. | 0.48 0.62 0.70 0.77 |
| `--stride 3 --phase 2` | 5 | 3 alligator, 1 pothole, 1 long. | 0.52 0.62 0.63 0.67 0.76 |

Phase 1 misses the pothole the other two catch. That is a real disagreement
between passes over the same road, and it is what makes a confirmation count
carry information — some defects reach `distinct_buses=3`, others don't.

### The demo split

Stride costs detections (12 events at stride 1, 4–5 at stride 3), so the two
jobs use different settings:

- **Seeded history** — `--stride 3`, phases 0/1/2, backdated, GPS noise on.
  Three or four buses per route, run before the demo. Varied confirmation
  counts, and it's 3× faster to seed.
- **The live pass** — window 1 at `--stride 1`, full quality, smooth video,
  maximum detections.

Which gives a better demo than uniform passes would: the defects the live bus
reports land on pins that are **already there**, and their confirmation counts
tick up in front of the audience. *"That pothole was already reported twice
this week"* is the whole fleet argument in one sentence.

```bash
# seed history: three buses, three phases, backdated
python run_edge.py -s <clip> --route route-21g --bus BUS_002 --stride 3 --phase 0 --at=-3d --gps-noise 6 --speed-jitter 0.1
python run_edge.py -s <clip> --route route-21g --bus BUS_003 --stride 3 --phase 1 --at=-1d --gps-noise 6 --speed-jitter 0.1
python run_edge.py -s <clip> --route route-21g --bus BUS_004 --stride 3 --phase 2 --at=-2h --gps-noise 6 --speed-jitter 0.1
```

`--at` needs the `=` form: argparse reads a bare `-2h` as a flag.

Three clips across three routes, each with a few phased passes, is enough to
populate a map that behaves like a fleet.

### What the footage cannot show

**Auto-close is not demonstrable from this data** — nothing in the clips ever
gets repaired. Ship it as the rule plus a unit test, and one clearly-labelled
scripted pass with a defect suppressed standing in for a fixed pothole. Call it
staged in the report.

### The simulation boundary

State this plainly rather than letting a reviewer find it:

- **Real** — the footage, the detections, the tracking, the clustering logic,
  the confirmation counting, the bandwidth reduction.
- **Simulated** — GPS, timestamps, that there is more than one bus, multi-day
  history, repair events.

---

## Build plan, phase by phase

Seven phases. Each one ends in something you can run and show, so a deadline
that lands mid-plan still leaves a working demo. **Phase 5 is the first time
both windows are live together** — that's the minimum viable deliverable;
phases 6–7 are what make it a platform rather than a dashcam with a map.

Phase 0 can be done in parallel with 1–2 by anyone, since it's not code.

---

### Phase 0 — Route tracing *(no code, do it early)*

Not blocked on anything, and it blocks phase 2.

**Build**
- Trace 2–3 real Chennai bus routes as GeoJSON `LineString`s
  (geojson.io, or Overpass for an OSM relation) → `edge/routes/`.
- Each file: `route_id`, `name`, coordinate array, nominal `speed_kmh`.
- Pick which clip maps to which route and note the assumed direction of travel.

**Done when** all three clips have a route assigned and the polylines render on
a map at the right scale — a 306-frame clip at 30 km/h covers ~85 m, so the
route segment has to be about that long or the events bunch into a dot.

---

### Phase 1 — Event lifecycle *(the core new logic)*

Offline, no GPS, no server. Get the hard part right in isolation.

**Build**
- `RoadEvent` dataclass — schema per [Event and defect model](#event-and-defect-model), GPS fields left null.
- `edgecore/events.py`: `EventTracker.update(frame_idx, detections) -> list[RoadEvent]`.
  Open / confirm at `MIN_FRAMES` / close after `MISS_TOLERANCE` / emit on close.
  Track peak-area frame per track; `flush()` at end of clip for still-open tracks.
- `edgecore/pipeline.py`: `VideoSource` → `Detector.infer` → `EventTracker`,
  yielding `(frame_idx, annotated_frame, new_events)`.
- Crop extraction at the peak frame, JPEG, ~640px long edge.
- `run_edge.py` writing `events.json` + `crops/`.

**Done when** a full pass over `62_10-07-2023.mp4` emits an event count in the
same neighbourhood as the lab's own defect count for that clip
(`run_cli.py -m rdd-yolo12s -s 62_10-07-2023.mp4`), and the crops actually show
the damage. Expect slightly fewer events than lab defects — `MIN_FRAMES` is
stricter here on purpose.

**Watch for:** this is the phase where the numbers can silently go wrong.
`MIN_FRAMES` and `MISS_TOLERANCE` are guesses; tune them against a clip where
you have counted the real potholes by eye. Everything downstream inherits
whatever this produces.

---

### Phase 2 — GPS and metadata

**Build**
- `edgecore/gps.py`: load a route GeoJSON, cumulative-distance index, then
  `fix_for(frame_idx, fps) -> (lat, lon, bearing, speed_kmh, timestamp)` by
  interpolating along the polyline. Same interface for a GPX/CSV sidecar.
- Perturbation flags: `--phase`, `--gps-noise`, `--speed-jitter`, `--at`.
- `edgecore/config.py`: bus id, route id, API URL, thresholds — env + CLI.
- Wire the fix at each event's **peak frame** into `RoadEvent`.

**Done when** `events.json` exported as GeoJSON drops onto real road geometry in
geojson.io, and two runs with different `--phase` produce visibly different but
overlapping point sets.

---

### Phase 3 — Server, database, ingest

Raw events only. No clustering yet — that's phase 6.

**Build**
- FastAPI app, SQLAlchemy models for `buses`, `routes`, `events`
  (`defect_id` nullable for now).
- `DATABASE_URL` with SQLite fallback; `create_all` on startup, seed
  buses/routes from `edge/routes/`.
- `POST /api/events` — batch, validates, stores, saves crops under
  `server/media/`. Idempotent on `event_uid` so a retry can't double-insert.
- `GET /api/events?since=`, `GET /api/stats`, `GET /api/routes`.
- `edgecore/publisher.py` — batches, POSTs, and **spools to disk on failure with
  retry**. Kill the server mid-run, restart it, watch the backlog drain.

**Done when** `run_edge.py --post` puts rows in the DB, and the spool survives a
server restart.

---

### Phase 4 — Window 1: onboard UI

**Build**
- `edge/app_edge.py`, Streamlit. Reuse the lab app's upload/pick pattern —
  it already does upload, `data/videos/` picker and streaming annotated frames.
- Sidebar: bus ID, route, confidence, stride, `--phase`, API URL, a connection
  indicator.
- Main: live annotated frame + progress.
- Right panel: events as they fire — damage type, confidence, GPS fix, crop
  thumbnail, POST status per event.
- Footer: frames processed, events emitted, **bytes sent vs. clip size** —
  the bandwidth argument, on screen, live.

**Done when** you can upload a clip in a browser, watch boxes stream, and see
events posting with a running byte count.

---

### Phase 5 — Window 2: map + live database  ← *minimum viable demo*

**Build**
- `server/static/` — Tailwind dark shell matching [Flow.jpeg](Flow.jpeg).
- KPI tiles, Leaflet map with route polylines, markers coloured from
  `labcore.taxonomy` (export the palette as JSON so both windows agree).
- **Live DB table below the map** — newest first, auto-scroll, `events` /
  `defects` toggle, click-row-to-pan.
- 2s polling on `?since=<last_id>`; new pins pulse on arrival.

**Done when** both windows are open side by side, you upload a clip in window 1,
and pins and rows appear in window 2 while it runs. **This is the deliverable.**
Everything after this is upside.

---

### Phase 6 — Defect clustering

The map switches from plotting sightings to plotting *things in the road*.

**Build**
- `defects` table + `server/app/clustering.py`. On ingest: bounding-box
  prefilter, then haversine within `RADIUS` (15 m) and same `damage_type`.
- Attach or create; bump `sightings` / `distinct_buses`, nudge centroid, keep
  best crop.
- Promote to `open` at `distinct_buses ≥ 2`.
- `GET /api/defects`; dashboard defaults to defects, events behind the toggle.
- Marker style by status — `unconfirmed` greys out.
- Auto-close job + unit test (see [What the footage cannot show](#what-the-footage-cannot-show)).

**Done when** three phased passes over one clip produce **one pin per real
defect** with `distinct_buses=3`, and the conf-0.29 shadow artefact from the
model comparison sits there greyed out as `unconfirmed`.

---

### Phase 7 — Polish, in priority order

Take these top-down and stop wherever the deadline lands.

1. **Defect detail panel** — click a marker: crop, sighting history, which
   buses, days open. Highest demo value per hour.
2. **Seed the fleet history** — run all 9 phased passes so the map opens
   populated rather than empty.
3. **Heatmap layer** — severity-weighted, toggleable.
4. **Reports** — `GET /api/report` → maintenance queue ranked by
   severity × persistence, CSV export.
5. **Alerts** — a severity-threshold rule firing into a dashboard toast. Real
   SMS/WhatsApp is not worth the integration cost for a prototype; say so
   rather than half-wiring it.
6. **SSE** instead of polling. Looks identical to the audience — genuinely last.

---

### Critical path

```
Phase 0 ─┐
         ├─> 2 ─> 3 ─> 4 ─┐
Phase 1 ─┘                ├─> 5 (demo works) ─> 6 ─> 7
                          ┘
```

Phase 1 is the one to get right and the one most likely to eat time. Phase 5 is
the one that must land.
