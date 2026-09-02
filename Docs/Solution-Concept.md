# AI-Powered Mobile Urban Intelligence Platform

### Solution Concept and Technical Rationale

Smart India Hackathon — Problem Statement **26124**

---

## 1. Executive summary

City authorities today learn about road defects, congestion and unsafe driving
from fixed CCTV, manual inspections and citizen complaints. All three are slow,
sparse and reactive.

At the same time, public transport buses already drive across nearly every major
road in a city, every day, and a large share of them already carry cameras —
installed and paid for under passenger-safety programmes.

This project proposes to reuse that existing camera infrastructure. Each bus runs
lightweight AI perception onboard, converts what it sees into compact structured
events, and transmits only those events to a central platform. The central
platform aggregates observations from the entire fleet and turns them into
city-level intelligence: road-condition maps, congestion heat maps, incident
alerts and maintenance priorities.

In short:

> **Edge:** *"I saw a pothole at this location."*
>
> **Central:** *"This pothole has been observed 37 times by 12 buses over 8 days,
> on a high-traffic corridor — it should be prioritised for repair."*

---

## 2. The existing camera base

The proposal does not assume new hardware. Camera systems have already been
deployed at scale across Indian state transport fleets, largely under women's
safety and Nirbhaya-fund programmes.

| City / State | Transport body | Camera deployment status |
| --- | --- | --- |
| Chennai, Tamil Nadu | MTC | ~2,500 buses targeted or equipped under the Nirbhaya safety project; newer projects are adding 360° cameras |
| Bengaluru, Karnataka | BMTC | CCTV and panic buttons deployed across buses; ADAS forward-facing camera pilot conducted |
| Delhi | DTC and Cluster buses | CCTV installation undertaken across the fleet |
| Mumbai, Maharashtra | BEST | Large-scale CCTV deployment undertaken |
| Rajasthan | RSRTC | Nirbhaya buses equipped with CCTV, vehicle tracking and panic buttons |
| Thoothukudi, Tamil Nadu | TNSTC | CCTV installed in selected government buses as early as 2015 |

Tamil Nadu has additionally announced plans to install **360° cameras and driver
monitoring systems** in government buses, which would further widen the field of
view available to a system of this kind.

### 2.1 What these cameras are used for today

| # | Current use | Description |
| --- | --- | --- |
| 1 | Passenger surveillance | Monitoring activity inside the bus. MTC's system is explicitly specified as a CCTV surveillance system for women's safety. |
| 2 | Crime and harassment investigation | Recorded footage is used as investigative material after an incident is reported. |
| 3 | Emergency response | Cameras are integrated with panic buttons, GPS tracking and command centres; a panic-button press raises an alert that can be routed to police, fire or ambulance services. |
| 4 | Post-incident evidence | Footage is retained on bus-mounted recorders. MTC's technical specification calls for a 4-channel mobile NVR with a minimum of 30 days of storage. |
| 5 | Fleet and location monitoring | CCTV combined with GPS vehicle-location tracking lets authorities monitor buses from command centres. |
| 6 | Driver and road-safety monitoring | A more advanced and less common use — some buses have been equipped or trialled with forward-facing vision cameras for ADAS, letting the camera observe the road rather than only the cabin. |

**The gap this project addresses.** Every one of these uses treats the camera as
a record of *the bus and its passengers*. None of them treat it as a sensor for
*the city the bus is driving through*. The footage is stored, reviewed only when
something goes wrong, and then overwritten. This project uses the same feed to
detect public infrastructure and traffic problems continuously.

---

## 3. Why a bus fleet makes a good sensing network

| # | Property | Why it matters |
| --- | --- | --- |
| 1 | Continuous travel on public roads | Buses repeatedly traverse predefined routes, so a bus-mounted camera can collect road-condition data across a large geographic area without deploying dedicated survey vehicles. |
| 2 | Large coverage from existing assets | Public transport fleets run into the thousands of vehicles — Delhi's bus-safety programme alone deployed CCTV across thousands of DTC and Cluster buses. A fleet of that size can act as a distributed sensing network. |
| 3 | Existing recording and network path | Modern bus CCTV architectures already run cameras → mobile NVR → wireless network → central command centre. That chain can be extended for AI-based road perception instead of being rebuilt. |
| 4 | Repeated observation of the same road | A bus may cover the same route several times a day. This allows observations to be compared over time — detecting that a road section has developed a pothole, or that a traffic sign has disappeared. |
| 5 | Location is already available | Bus tracking systems already carry GPS. Combining bus location with an AI detection yields a directly actionable record: *damaged zebra crossing → coordinates → timestamp → image*. |
| 6 | Bus frequency is a proxy for road usage | A high concentration of bus trips on a corridor indicates heavy public-transport use and, typically, greater overall traffic exposure. This feeds naturally into hazard-priority scoring. |
| 7 | Substantially cheaper than dedicated survey vehicles | No separate inspection fleet needs to be procured or operated when buses already cover the same roads on a daily schedule. |

---

## 4. What can be detected

The camera feed can support a broad set of detection tasks. These fall into five
categories.

| Category | Detectable items |
| --- | --- |
| **A. Road defects** | Potholes; damaged road surfaces; missing road dividers; damaged road dividers; missing zebra crossings; damaged or faded zebra crossings; missing traffic signs; damaged traffic signs; waterlogging; other road hazards |
| **B. Traffic** | Vehicle detection, classification and counting; vehicle density; traffic congestion; bottlenecks; traffic flow; route delays |
| **C. Pedestrian safety** | Vulnerable pedestrian situations; school children crossing roads |
| **D. Incidents** | Hit-and-run detection; offending vehicle detection and tracking; rash or dangerous driving; vehicle registration / ANPR; incident evidence generation |
| **E. City-level intelligence** | Road-condition maps; congestion heat maps; infrastructure deficiency maps; origin–destination traffic patterns; fleet-wide observation aggregation; historical and repeated-problem analysis |

Categories A–D are perception tasks performed on the bus. Category E is derived
centrally from the accumulated output of A–D.

---

## 5. Detection approach

Almost all of the perception tasks above reduce to object detection, tracking or
segmentation, and a substantial body of pretrained models already exists for
them.

| Problem | Approach | Reference model / resource |
| --- | --- | --- |
| Potholes | Object detection | [PotholeNet-YOLO11m](https://huggingface.co/Vansh180/PotholeNet-V1) |
| Damaged roads / cracks | Object detection | [YOLOv8 Road Damage Detector](https://huggingface.co/vinothvikas1987/pothole-detection-yolov8), trained on the Road Damage Dataset |
| Missing / damaged road dividers | Detection or segmentation | YOLO fine-tuned on a road-scene dataset |
| Missing / damaged zebra crossings | Detection or segmentation | [Traffic Sign YOLOv8](https://huggingface.co/yahyagul/traffic-sign-yolov8), which includes a `ped_zebra_cross` class |
| Damaged / missing traffic signs | Detection plus condition classification | [Traffic Sign Condition Detector](https://huggingface.co/Erpix3lt/traffic-sign-detection), with good / moderate / bad condition classes |
| Traffic sign recognition | Detection plus classification | [YOLOv11 Traffic Sign Detection](https://huggingface.co/cvtechniques/TrafficSignDetection) |
| Waterlogging | Segmentation or detection | Waterlogging-specific or fine-tuned segmentation model |
| Other road hazards | Detection or segmentation | YOLO fine-tuned on the relevant hazard classes |
| Vehicles | Detection plus classification | YOLO pretrained on COCO |
| Vehicle counting | Detection plus tracking | YOLO with an object tracker |
| Pedestrians | Detection plus tracking | YOLO pretrained on COCO |
| School children crossing | Person detection plus tracking | Person detector with tracker |
| Traffic density | Detection plus counting | YOLO with tracker |
| Traffic congestion | Detection, tracking and temporal analysis | Vehicle detector with tracker |
| Rash driving | Detection, tracking and trajectory analysis | Vehicle detector with tracker |
| Hit-and-run | Detection plus multi-object tracking | Vehicle detector with multi-object tracker |
| Licence plate detection | Plate detection | [YOLOv5 License Plate Detection](https://github.com/yakhyo/yolov5-license-plate-detection) |
| Licence plate recognition | Plate detection plus OCR | YOLO with EasyOCR or PaddleOCR — see [YOLOv8 + EasyOCR ANPR](https://github.com/AarohiSingla/Automatic-Number-Plate-Recognition--ANPR-) |

### 5.1 Detection coverage in the prototype

The prototype implements **road damage and pothole detection only** — the onboard
agent has exactly one detector branch. The surrounding architecture — event
formatting, metadata enrichment, transmission, ingestion, storage, clustering and
dashboard — is built to be detection-agnostic, so extending coverage to the
remaining categories is a matter of adding models to the perception layer rather
than redesigning the system. Section 9 sets out in full what is and is not built.

---

## 6. System architecture

![Full system architecture](Full%20architecture.png)

The architecture has two halves, separated by a single narrow interface: the
structured event.

**Onboard (edge), on the bus.** The camera feed is sampled and passed to an AI
perception layer that detects potholes, road damage, zebra crossings,
waterlogging, vehicles, traffic signs, pedestrians and licence plates. Every
detection, regardless of class, is converted into a **common event format** and
enriched with GPS coordinates, timestamp, bus ID, route ID, event type,
confidence and severity, and an evidence image or clip. Only these enriched
events are transmitted, over 4G/5G or Wi-Fi.

**Central platform (cloud or server).** An ingestion service receives events from
the fleet in real time. Event processing validates, deduplicates and aggregates
them; a geospatial database stores them; an analytics engine produces
aggregations, heat maps and trend analysis; and a GIS and analytics dashboard
presents live maps, alerts, reports, infrastructure insights and route/traffic
analytics to the authority.

---

## 7. How edge computing is achieved

The governing principle is to **perform perception on the bus and transmit only
structured observations**, rather than continuously streaming raw video to a
central server.

### 7.1 Multi-task perception in a single pass

Running several heavy specialised models independently on an onboard device does
not scale:

```text
Video
  ↓
Pothole model  →  Vehicle model  →  Pedestrian model  →  Sign model  →  ...
```

Instead, a **single multi-class detector** runs on the onboard system and detects
all classes in one inference pass. The raw detections are then converted into
class-specific event formats downstream:

```text
                Multi-class detector
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
    Vehicles       Pedestrians     Road damage
        ↓               ↓               ↓
   count / speed     tracking        severity
```

For example:

- Vehicle → count, class, speed
- Pothole → location, confidence, severity
- Pedestrian → position, tracking information
- Licence plate → plate number, confidence

This avoids maintaining a separate inference pipeline for every individual urban
problem, and keeps the onboard compute budget bounded as detection coverage
grows.

### 7.2 Accuracy versus edge efficiency

A single multi-class model will generally be **less accurate on any one task**
than a specialised model trained solely for that task. Rather than treating this
purely as a defect, the system manages it with a **priority hierarchy** and with
**fleet redundancy**.

**Higher priority** — detections that warrant greater emphasis in training and
inference, and that must remain reliable:

1. Pedestrian safety
2. Hit-and-run and dangerous driving
3. Traffic incidents
4. Vehicle and traffic detection

**Lower priority** — detections that can tolerate lower per-observation accuracy:

5. Potholes and road damage
6. Missing dividers
7. Road markings
8. Other infrastructure defects

This ordering is not arbitrary: higher-priority subjects tend to occupy a larger
share of the frame and are therefore more likely to be captured reliably, while
smaller and more distant infrastructure defects are more likely to be missed on
any single pass.

The key mitigation is that **the same object is observed repeatedly by multiple
buses**:

```text
Bus 01 ──┐
Bus 07 ──┤
Bus 14 ──┼──→ Same pothole
Bus 23 ──┤
Bus 31 ──┘
```

A single imperfect detection is therefore not the final conclusion. Repeated
independent observations raise confidence that a problem genuinely exists.

> **Fleet redundancy compensates, to a significant extent, for imperfect
> individual observations.**

This is stated deliberately as *to a significant extent* rather than as a claim
that accuracy does not matter. Safety-critical detections still require high
per-observation reliability, because they cannot wait for corroboration.

### 7.3 Reducing the inference workload

The camera captures at 30 FPS, but urban infrastructure does not change at 30
FPS. Frames are therefore sampled before inference:

```text
30 FPS camera
      ↓
Frame sampling
      ↓
5–10 FPS
      ↓
AI inference
```

The sampling rate can be tuned per detection task. This materially reduces
CPU/GPU utilisation, power draw, thermal load and inference cost, while retaining
sufficient temporal coverage for slow-changing road conditions.

### 7.4 Edge produces observations; central produces intelligence

This separation of responsibility is the core architectural decision.

**On the bus — perception.** The edge system answers *"What is happening in front
of this bus?"*

```text
Camera → AI model → Detection
                        ↓
        GPS + timestamp, bus ID + route ID,
        confidence + severity, evidence
                        ↓
                Structured event
```

Only this compact record is transmitted — not the video.

**On the server — intelligence.** The central system answers *"What is happening
across the city?"*

```text
Observations from many buses
            ↓
     Central database
            ↓
  Aggregation, deduplication,
  temporal analysis, spatial analysis
            ↓
      GIS + analytics
```

This is where congestion heat maps, repeated-defect identification, fleet-wide
traffic analysis, infrastructure deficiency analysis, route delays and historical
trends are produced.

### 7.5 Edge strategy at a glance

> **Run lightweight, multi-task perception on each bus and transmit only
> structured observations to the central platform.**

| Strategy | Purpose |
| --- | --- |
| Single multi-class detector | Detect multiple urban objects and problems in one inference pipeline |
| Task-specific output processing | Convert each detection class into an appropriate event format |
| Priority hierarchy | Allocate greater emphasis to safety-critical detections |
| Frame sampling | Reduce 30 FPS video to a lower, task-appropriate inference rate |
| Fleet redundancy | Use repeated observations across buses to build confidence in persistent problems |
| Edge perception | Generate detections locally without continuously transmitting raw video |
| Central analytics | Perform aggregation, correlation, heat maps and city-level intelligence centrally |

### 7.6 Statement of novelty

Multi-task and multi-model inference are established techniques, and the novelty
of this work is not simply that several detections run at once. It is stated more
precisely as:

> **A fleet-scale edge perception architecture that uses a unified multi-class
> detector and task-specific event extraction to convert existing public
> transport cameras into distributed urban sensors, while shifting
> computationally intensive city-level analytics to a centralised platform.**

The accompanying design insight:

> **Individual buses do not need to produce perfect knowledge of the city. They
> act as distributed, redundant observers whose imperfect observations are
> combined centrally into reliable urban intelligence.**

---

## 8. The central platform

The central platform is where raw detections from individual buses become
city-level intelligence. Its capabilities are organised in layers.

### 8.1 Event management

Every bus sends structured events of the form:

```text
Event:      Pothole
GPS:        13.0827, 80.2707
Time:       2026-08-14 09:12:41
Bus:        42
Route:      17
Confidence: 0.87
Evidence:   image
```

The platform stores these events, validates incoming data, discards invalid or
noisy detections, deduplicates repeated observations, groups nearby detections
into a single real-world issue, and maintains the history of each issue.

### 8.2 Fleet-level observation aggregation

This is the layer that gives the concept its value.

```text
Bus 12 ──┐
Bus 18 ──┤
Bus 27 ──┼──→ Same road defect
Bus 41 ──┘
```

From clustered observations the platform derives how many buses saw the issue,
how often it is seen, whether it persists, which routes encounter it, a
confidence estimate based on repetition, and the first and most recent
observation times.

The output therefore changes from a single-vehicle report:

> "Bus 12 detected a pothole."

into a corroborated finding an authority can act on:

> **"This pothole has been independently observed 37 times by 12 buses over 8
> days."**

### 8.3 GIS intelligence

The map is the primary interface. It presents individual events — potholes, road
damage, traffic signs, zebra crossings, waterlogging, traffic incidents — as well
as derived spatial layers:

- Road-condition map
- Infrastructure-deficiency map
- Traffic-density map
- Congestion heat map
- Accident and incident hotspots
- Frequently problematic road segments

with filtering by issue type, severity, date and time, route, bus, confidence and
road segment.

### 8.4 Road-condition intelligence

Rather than plotting isolated potholes, defects are aggregated onto **road
segments**:

```text
Road A
├── 17 potholes
├── 4 damaged sections
├── 2 missing dividers
└── observed by 23 buses
```

This makes it possible to identify problematic roads, recurring defects,
high-severity segments, roads deteriorating over time, and infrastructure-deficient
areas — and it is what makes the heat map meaningful rather than decorative.

### 8.5 Traffic intelligence

Vehicle detections reported by buses are aggregated into:

- **Vehicle density** per road segment (for example, *Road A → 82 vehicles/min*)
- **Congestion patterns** — frequently congested locations, peak periods,
  bottlenecks and hotspots
- **Traffic trends** over the day:

```text
Road A

08:00 → Heavy
10:00 → Moderate
13:00 → Low
18:00 → Very heavy
```

### 8.6 Route analytics

Because every observation carries bus ID, route ID, GPS and timestamp, the
platform can also analyse the public transport network itself:

```text
Route 17
Expected travel time: 42 min
Observed average:     57 min
Delay:                15 min
```

This surfaces routes experiencing delays, the sections causing them, recurring
delay locations, average travel time by route, and time-dependent congestion.

### 8.7 Origin–destination and movement analysis

With sufficient fleet coverage, broader movement patterns can be inferred —
major corridors, high-demand directions, flows between areas, and changes in
those patterns over time. This is a more advanced capability and is not central
to the prototype.

### 8.8 Incident intelligence

For hit-and-run and dangerous-driving events:

```text
Detection → Tracking → Incident event → Central platform
```

The central system combines vehicle identity or plate, time, GPS, bus ID, route,
evidence, and observations from other nearby buses — enabling **cross-bus
corroboration** of a single incident.

### 8.9 Alert generation

Observations are turned into actionable alerts, so the dashboard reports what
needs attention rather than only what exists:

> **Severe waterlogging detected**
> Location: XYZ Road
> Observed by: 6 buses
> First detected: 08:32
> Latest observation: 09:14

> **Recurring road defect**
> 24 observations across 9 buses in the past 7 days.

### 8.10 Maintenance prioritisation

Rather than presenting a thousand defects as equals, defects are scored:

```text
                 PRIORITY
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
    Severity     Frequency      Traffic
       ↓             ↓             ↓
       └─────────────┼─────────────┘
                     ↓
            Maintenance priority
```

| Road | Severity | Observations | Traffic exposure | Priority |
| --- | ---: | ---: | ---: | --- |
| A | High | 42 | High | Critical |
| B | Medium | 5 | Low | Low |
| C | High | 18 | High | Critical |

This is what moves the system from *"AI detects potholes"* to *"AI helps
authorities decide what to fix first."*

### 8.11 Historical analytics

Because all observations are retained centrally, the platform can compare today
against yesterday, this week against last week, and conditions before and after a
repair — exposing seasonal variation, recurring problem locations and
infrastructure deterioration:

```text
Road A

January  →  4 observations
February →  9
March    → 17
April    → 31
```

A trend of this shape indicates a road segment in active decline.

### 8.12 Dashboard and visualisation

```text
                    CENTRAL PLATFORM
                           │
       ┌───────────────────┼───────────────────┐
       ↓                   ↓                   ↓
   Live events        Fleet analytics     Historical data
       │                   │                   │
       ↓                   ↓                   ↓
    GIS map          Traffic analysis     Trend analysis
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ↓
                  Authority dashboard
```

### 8.13 Core central capabilities

Reduced to its essentials, the central platform provides:

1. Event ingestion and storage
2. Deduplication and cross-bus aggregation
3. Confidence and reliability estimation
4. GIS visualisation
5. Road-condition and infrastructure analysis
6. Traffic and congestion analysis
7. Route-delay analysis
8. Historical and trend analysis
9. Incident alerts
10. Maintenance prioritisation

---

## 9. Prototype scope — covered and not covered

Everything described in sections 6 to 8 is the intended platform. What follows is
the boundary of what has actually been built, stated plainly rather than left for
a reviewer to discover.

The guiding principle for the prototype was to build **one detection class end to
end** — camera through to ranked maintenance queue — rather than several classes
half way. The pipeline either exists or it does not; the perception layer is the
part that is deliberately narrow.

### 9.1 What the prototype covers

| Architecture stage | Built in the prototype |
| --- | --- |
| **Camera input** | Video clip ingestion with configurable frame stride. Real road footage from the RADRoad Anomaly Detection dataset. |
| **AI perception** | Road damage and pothole detection using `rdd-yolo12s` at confidence 0.15 with ByteTrack tracking. Model selected by benchmarking 27 YOLO checkpoints over a single canonical damage taxonomy. |
| **Event lifecycle** | Per-track open / confirm / close logic emitting **one event per physical defect**, not one row per frame. Position, timestamp and evidence crop are all sampled at the track's peak-area frame — the closest approach to the defect. |
| **Common event format** | A single `RoadEvent` schema carrying damage type, severity, confidence, frame area, position, bearing, speed, capture time, track ID, source clip, model ID and crop reference. |
| **Metadata enrichment** | Bus ID, route ID, GPS fix, bearing, speed and timestamp attached to every event. |
| **Transmission** | Batched HTTP POST from a background worker thread so inference never blocks on the network, with **disk spooling and retry when connectivity is lost** — the backlog drains on the next pass that connects. |
| **Bandwidth reduction** | Measured, not asserted: a 60-frame pass sent 39 KB against a 12.3 MB clip — **99.7% reduction**. Video never leaves the bus. |
| **Ingestion service** | FastAPI `POST /api/events`, batch capable, storing events and evidence crops. Idempotent on `event_uid`, so a publisher retry cannot double-count. |
| **Database** | Two tables — `events` as the immutable raw sighting log, `defects` as the physical thing in the road — plus buses and routes. SQLite by default, PostgreSQL via `DATABASE_URL`. |
| **Deduplication and clustering** | On ingest, a sighting joins the nearest same-type defect within a 15 m radius, or creates a new one; sightings count, distinct-bus count, centroid, first/last seen and best evidence crop are all maintained. Measured: 92 sightings from 10 buses collapse into 43 defects. |
| **Cross-bus confirmation** | A defect promotes from `unconfirmed` to `open` once seen by two or more distinct buses. Single-bus sightings stay greyed out on the map. This is the fleet-redundancy argument of §7.2, working. |
| **Repair closure** | The auto-close rule — a defect that stops being reported on a route still being driven is treated as repaired — is implemented and covered by unit tests. |
| **GIS dashboard** | Leaflet map with defects plotted by taxonomy colour, sized by severity and greyed when unconfirmed; route polylines; live KPI tiles; and the database rendered as a live table below the map, updating in real time as a pass runs. |
| **Defect detail** | Click a defect for its evidence: best crop, full sighting history, which buses reported it, and days open. |
| **Heat map** | Severity-weighted, toggleable over the route polylines. |
| **Maintenance prioritisation** | A maintenance queue ranked by severity × persistence, with CSV export. |
| **Alerts** | Severity-threshold rules firing into the dashboard. |

### 9.2 What is real and what is simulated

Within the covered scope above, an important distinction:

| | |
| --- | --- |
| **Real** | The footage, the detections, the tracking, the event lifecycle, the clustering and confirmation logic, the bandwidth reduction, and the offline spool. |
| **Simulated** | GPS position, timestamps, the existence of more than one bus, the multi-day fleet history, and repair events. |

GPS is interpolated along hand-plotted real Chennai bus corridors, so events land
on real roads at real coordinates — but the fix is generated, not received. Real
GPS drifts 5–10 m in urban canyons, which is precisely why the clustering radius
is set generously at 15 m.

The dataset contains no repeat traversals of the same road, so repeat passes are
the same clip replayed as a different bus at a different frame phase. This is not
a cosmetic trick: different phases sample **different frames** and therefore
genuinely detect differently — measured at 5, 4 and 5 events across three phases
of one clip, with phase 1 missing a pothole the other two catch. Confirmation
counts therefore carry real information rather than being decoration.

**Auto-close cannot be demonstrated from this footage**, because nothing in the
clips is ever repaired. The rule ships with unit tests instead.

### 9.3 What the prototype does not cover

| Area | Not built | Why |
| --- | --- | --- |
| **Traffic analysis** | Vehicle detection, classification, counting, density, congestion, bottlenecks and traffic flow | The entire right branch of the architecture. Out of scope for V1. |
| **Pedestrian safety** | Pedestrian detection and tracking, school-children-crossing detection | Requires a second detector branch. |
| **Incidents** | Hit-and-run detection, offending-vehicle tracking, rash-driving detection, number-plate recognition and ANPR | The most complex branch — needs multi-object tracking plus OCR plus an evidence chain. |
| **Other road defects** | Waterlogging, signboards, zebra crossings, road dividers | Models exist (§5) but are not integrated. |
| **Multi-class single detector** | The unified multi-class model described in §7.1 | The architecture is designed for it; the prototype runs one branch, so the multi-class trade-off is reasoned about rather than measured. |
| **Multi-camera input** | Rear, side and cabin feeds | Only a single forward-facing stream is processed. |
| **Real onboard hardware** | Deployment to an in-vehicle compute unit | Runs on a development machine with a GPU, not on bus hardware; no thermal, power or vibration validation. |
| **Real fleet integration** | Live camera feeds, real GPS receivers, real bus telemetry, transport-authority systems | No access to an operating fleet. |
| **Secure transmission** | Authentication, encryption, device identity and key management on the edge-to-server link | Events post over plain HTTP on a local network. A production deployment would require all of this. |
| **Route and delay analytics** | Route-delay estimation, expected-versus-observed travel time | Needs real timestamps from real traversals; simulated time makes the result meaningless. |
| **Origin–destination analysis** | Movement corridors and inter-area flow inference | Requires fleet-wide coverage over a long period. Explicitly deferred as an advanced feature. |
| **Scalable geospatial storage** | PostGIS and spatial indexing | A bounding-box query on plain latitude/longitude floats is sufficient at prototype scale. |
| **Historical trend analytics** | Month-over-month deterioration, before/after repair comparison, seasonal analysis | Needs genuine multi-day history rather than backdated replays. |
| **External alert delivery** | SMS, WhatsApp and email dispatch | Not worth the integration cost for a prototype; alerts surface in the dashboard instead. |
| **Live push transport** | Server-sent events or WebSockets | Two-second polling is used. Functionally identical to a viewer, and more robust to a mid-demo server restart. |

### 9.4 Where the boundary actually sits

The uncovered items fall into three groups, and they are not equally difficult.

1. **Additional detector branches** — traffic, pedestrians, waterlogging,
   signboards, plates. Pretrained models for most of these already exist (§5),
   and the event schema, transport, ingestion, clustering and dashboard are all
   detection-agnostic. This is integration work, not redesign.
2. **Things that need a real fleet** — genuine GPS, real timestamps, real repeat
   traversals, route-delay and origin–destination analytics, historical trends.
   These are blocked on data access rather than on engineering, and the
   interfaces to swap simulation for reality already exist: GPS replays from GPX
   or CSV behind the same interface used by the simulator.
3. **Production hardening** — onboard hardware, secure transmission, device
   identity, scalable geospatial storage. Straightforward but substantial, and
   deliberately not attempted at prototype stage.

What the prototype does demonstrate is the part of the concept that could
genuinely have failed: that **imperfect detections from multiple passes cluster
into confirmed physical defects**, and that the bandwidth cost of doing so is
small enough for the approach to be viable on a real fleet.

---

## 10. References

**Camera deployment across transport fleets**

- [MTC Chennai — 2,330 buses with CCTV, MNVR and panic buttons](https://www.dtnext.in/news/chennai/udhayanidhi-inaugurates-iccc-to-monitor-cameras-in-mtc-buses)
- [MTC Chennai — CCTV installation details](https://www.newindianexpress.com/cities/chennai/2021/Aug/05/mtc-commences-works-to-install-cctv-cameras-2340351.html)
- [MTC Chennai — Annual Report 2020–21](https://mtcbus.tn.gov.in/uploads/Annualreports/MTC_49th_ANNUAL_REPORT_2020-21.pdf)
- [MTC — CCTV surveillance system for women's safety, technical specification](https://mtcbus.tn.gov.in/asset/tenders/Volume_1_-_MTC1.pdf)
- [Tamil Nadu — 360° cameras and driver monitoring systems planned for government buses](https://www.thenewsminute.com/tamil-nadu/tn-to-install-360-degree-cameras-driver-monitoring-systems-in-govt-buses)
- [Delhi Government — CCTV in DTC and Cluster buses](https://delhiplanning.delhi.gov.in/sites/default/files/Planning/generic_multiple_files/wc_7.pdf)
- [DTC — Citizen Charter confirming IP-CCTV in buses](https://dtc.delhi.gov.in/dtc/citizen-charter)
- [Delhi — command centre tracking government buses in real time](https://www.hindustantimes.com/cities/delhi-news/transport-dept-readies-command-centre-to-track-govt-buses-in-realtime-101614190368159.html)
- [BMTC — Government annual report, CCTV planned for 5,000 buses](https://kla.kar.nic.in/council/house/Paperlaid/144/38.pdf)
- [BMTC — ADAS / forward-facing vision camera pilot](https://www.newindianexpress.com/cities/bengaluru/2023/nov/30/bmtc-launches-advanced-passenger-safety-system-2637428.html)
- [BMTC — Economic Survey, CCTV installed in 5,000 buses](https://data.opencity.in/dataset/71fbcff3-2414-4426-ab79-20fcf7cf04e2/resource/abab2550-ee98-4d94-8351-84670c8988a1/download/091ccfaa-bf5b-4e3b-8619-522218e3f89e.pdf)
- [Rajasthan RSRTC — 2025 CCTV specifications for buses](https://transport.rajasthan.gov.in/content/dam/transport/RSRTC/pdf/Tenders/2025/242_241_merged.pdf)
- [PIB — 20 Rajasthan Nirbhaya buses with CCTV and GPS](https://www.pib.gov.in/newsite/PrintRelease.aspx?lang=2&reg=48&relid=145670)
- [Thoothukudi — government buses fitted with CCTV](https://www.ndtv.com/tamil-nadu-news/to-curb-harassment-of-women-and-communal-clashes-tamil-nadu-buses-to-get-cctv-cameras-763846)

**Road asset management**

- [World Bank — Road Asset Management](https://www.worldbank.org/en/topic/transport/brief/road-asset-management)

**Prototype data and model selection**

- [RADRoad Anomaly Detection dataset](https://www.kaggle.com/datasets/rohitsuresh15/radroad-anomaly-detection/data) — source of the road footage
- [Model comparison writeup](../road-damage-lab/docs/model-comparison-2026-08-26.md) — the evidence behind the `rdd-yolo12s` choice

**Related project documents**

- [Project base and problem statement](Base.md)
- [V1 implementation plan](V1-Plan.md) — component specs, build phases and verified outcomes
