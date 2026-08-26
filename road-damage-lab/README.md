# Road Damage Lab

A bench for trying YOLO-based road-damage detectors on road video. Pick a model
from a dropdown, feed it a clip, and watch the boxes land — each carrying a
persistent track ID, a damage type, a confidence score, and its size relative to
the frame. Two models can run side by side on the same footage, and every pass
ends with a road-condition report.

It covers the full RDD2022 taxonomy — longitudinal, transverse and alligator
cracking plus potholes — not just potholes.

```
streamlit run app.py
```

---

## The taxonomy layer

Models don't agree on class ids. Two of the registered checkpoints order the
*same* four RDD2022 classes differently:

```
rdd-*          0:longitudinal  1:transverse  2:alligator     3:pothole
ozair-yolov8s  0:alligator     1:transverse  2:longitudinal  3:other   4:pothole
```

and a third names them `D00 / D10 / D20 / D40`. So a raw class id means nothing
on its own — comparing two models, colouring boxes by type, or filtering "show
me only potholes" all need a model-independent key.

Every model maps its ids onto a canonical vocabulary
([`labcore/taxonomy.py`](labcore/taxonomy.py)) via `class_map:` in
`models.yaml`. Colours, class filters, statistics, CSVs and the condition report
all work in canonical terms, so they mean the same thing whichever checkpoint
produced them.

| key | RDD code | severity | notes |
|---|---|---|---|
| `pothole` | D40 | 1.00 | safety-critical |
| `alligator_crack` | D20 | 0.75 | fatigue cracking; precursor to potholing |
| `longitudinal_crack` | D00 | 0.40 | runs with the direction of travel |
| `transverse_crack` | D10 | 0.35 | runs across the carriageway |
| `crack` | — | 0.40 | models that don't split by orientation |
| `other` | — | 0.30 | unspecified damage |
| `repair` | — | 0.10 | patched surface — context, not a defect |
| `unknown` | — | 0.30 | unmapped; draws grey |

Every `class_map` in `models.yaml` was read off the actual checkpoint with
`--inspect`, not off a model card. The cards get the ordering wrong.

```bash
python run_cli.py --inspect            # audit every model's class map
python run_cli.py --inspect -m my-run  # just one
```

---

## What's in the box

**27 checkpoints registered** (`models.yaml`), 20 of them enabled, weights
downloading lazily from the Hugging Face Hub on first use and caching under
`weights/`. Flip `enabled:` to trim the picker.

### Multi-class road damage

| id | source | notes |
|---|---|---|
| `rdd-yolo12s` … `rdd-yolov5s` | [SreekarAditya/yolo-rdd2022-benchmark](https://huggingface.co/SreekarAditya/yolo-rdd2022-benchmark) | **eight architectures on identical data** — v5s/v8s/v9s/v10s/v11s/v12n/v12s/v12m |
| `rdd-yolo12s-480`, `-800` | ↑ same repo | resolution ablation, off by default |
| `rezzzq-yolo12s` | [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022) | independent RDD2022 run; adds a `Repair` class |
| `ozair-yolov8s` | [ozair23/yolov8-road-damage-detector](https://huggingface.co/ozair23/yolov8-road-damage-detector) | swapped class order + `other corruption` |

The eight `rdd-*` entries are the apples-to-apples set: same split, same
protocol, same seed. Their published sweep puts peak F1 at **conf 0.10–0.20**,
not the YOLO default of 0.25 — these entries default to 0.15, and detecting
nothing usually means the threshold is too high.

`rdd-yolov10s` scores worst in the published benchmark because of its NMS-free
head; this lab applies standard NMS, so expect it to beat its own paper numbers.

### Crack segmentation

`crackseg-yolov8{n,s,m,l,x}` — [OpenSistemas/YOLOv8-crack-seg](https://huggingface.co/OpenSistemas/YOLOv8-crack-seg),
five sizes on the Ultralytics crack-seg set. Single class, so they map onto the
generic `crack` key rather than claiming to know orientation. Only `s` is on by
default.

### Pothole specialists

`samdutse-yolov8`, `engjameso-yolov{8n,8fpn,9c,11,12n}`,
`keremberke-yolov8{n,s,m}-seg`. Single-class, so they can't tell you a road is
cracking — but on potholes they're trained harder than the RDD models, which
spread capacity across four classes. Filter an RDD model to `pothole` and put
one of these beside it.

---

**On every box:** track ID, damage type, confidence, a confidence bar, pixel
dimensions, and area as a percentage of the frame. Colour is keyed to the damage
type by default, so a pothole is the same red in every model's output.

**On every frame:** a HUD with the model name, device, frame counter, detections
in frame, unique IDs, inference time and live FPS — plus a bottom-left legend
tallying each damage type, colour-matched to the boxes.

**Per run:** an annotated `.mp4`, a per-detection CSV, a per-defect CSV, and a
road-condition report.

---

## The condition report

Collapses a pass into what a survey actually wants
([`labcore/survey.py`](labcore/survey.py)):

- **Damage score** — severity-weighted defects per 100 frames. A pothole counts
  for about three times a transverse crack, scaled by how large the defect got
  and how confident the model was. Comparable *only* between runs over the same
  clip at the same stride; it ranks models and road segments, it is not an
  absolute pavement index.
- **Grade** — A (clean) through E (severe), off that score. Deliberately blunt.
- **Per damage type** — unique defects, raw boxes, mean confidence, worst size.
- **Hotspots** — score bucketed along the clip. With a forward-facing camera,
  frame index proxies distance, so buckets approximate road segments.
- **Every defect** — one row per tracked defect, not per frame it appeared in.

Two honesty notes the report surfaces itself:

- Trackers emit boxes they haven't yet confirmed into a track. Those carry no
  ID and are **not** counted as unique defects — the report says how many were
  dropped, which is why boxes exceed defects.
- Without tracking there are no IDs to collapse on, so every box counts
  separately. The report flags this rather than implying unique counts.

---

## Setup

Torch first, matching your CUDA version — then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verified on Python 3.10, torch 2.7.1+cu128, ultralytics 8.4.129, RTX 3060.
CPU-only works too; it's just slower.

All 20 enabled checkpoints load on mainline Ultralytics — including the YOLOv12
ones, despite one model card asking for a fork.

> If your editor flags the imports as missing, it's pointing at a different
> interpreter than the one these packages went into. Select the Python 3.10 that
> `python --version` resolves to on your PATH.

---

## Using it

**Video tab** — upload a clip, pick one in `data/videos/`, or point at a webcam
index. Frames stream in annotated as they're processed.

**Image tab** — same overlay on stills. Handy for eyeballing a threshold before
committing to a long video.

**Models tab** — what's registered, what's cached, which damage types each
covers, and a JSON dump of the loaded model including its full id → damage-type
mapping.

**Damage types tab** — the taxonomy, the severity weights, and a coverage matrix
of which models detect what.

### Sidebar controls that matter

- **Mode: Compare two** renders two models on the same frames. If they cover
  different damage types the sidebar says so — a pothole-only model can't lose
  on cracks it was never trained to see.
- **Damage types to keep** is applied *per model*, resolved through the
  taxonomy. This is what makes filtering correct across checkpoints with
  different class orders.
- **Confidence threshold** — the first knob to reach for. RDD models want
  0.10–0.20; the pothole models are calibrated higher.
- **Colour boxes by: class** (default) keys colour to damage type. Switch to
  `track` to follow one defect through a clip.
- **Box label: raw** shows what the checkpoint actually calls a class (`D40`
  rather than `pothole`) — use it to audit a `class_map`.
- **Frame stride** — process every Nth frame. The quickest way to skim a clip.
- **Downscale frames to width** — big speedup on 4K dashcam footage; costs you
  small distant defects, and cracks are small.

Outputs land in `outputs/`, named `<clip>__<model-id>.mp4` /
`_detections.csv` / `_defects.csv`.

---

## CLI

Same pipeline, no browser:

```bash
python run_cli.py --list
python run_cli.py --inspect
python run_cli.py -m rdd-yolo12s -s data/videos/road.mp4
python run_cli.py -m rdd-yolo12s -s road.jpg --conf 0.15
python run_cli.py -m rdd-yolo12s -m ozair-yolov8s -s road.mp4 --only pothole
python run_cli.py --benchmark -s data/videos/road.mp4 --max-frames 200 --no-save
```

`--benchmark` runs every enabled model over the same clip and prints two
comparison tables: headline metrics, then unique defects per damage type. The
second is the one that matters — a model can win on total detections purely by
over-firing on one easy class.

`--only` takes canonical damage types and is resolved per model, so it stays
correct across checkpoints with different class orders.

Other flags: `--half`, `--stride`, `--width`, `--no-track`, `--tracker`,
`--device`, `--show`, `--no-survey`.

---

## Adding a model

Append to `models.yaml`. That's the whole procedure.

```yaml
  - id: my-run-01
    name: "My road damage run 01"
    source: local              # hf | local | ultralytics
    path: weights/local/best.pt
    task: detect
    default_conf: 0.15
    class_map: {0: longitudinal_crack, 1: transverse_crack,
                2: alligator_crack, 3: pothole}
    notes: "trained on our own dashcam frames"
```

Then confirm the mapping against the actual checkpoint:

```bash
python run_cli.py --inspect -m my-run-01
```

Without `class_map`, classes are matched by name — which works for obvious ones
(`pothole`, `crack`) and fails on D-codes. Unmapped classes still draw and still
count; they land on `unknown`, render grey, and `--inspect` flags them.

Anything Ultralytics can load works — v5/v8/v9/v10/v11/v12, detect or segment.
Add `enabled: false` to keep an entry listed but out of the picker.

---

## Layout

```
app.py            Streamlit UI
run_cli.py        headless runner / benchmark / --inspect
models.yaml       the model registry
labcore/
  taxonomy.py     canonical damage types, colours, severity weights
  registry.py     models.yaml -> resolvable weight files (HF download + cache)
  detector.py     uniform wrapper over Ultralytics; normalised Detection records
  draw.py         boxes, ID/confidence chips, trails, HUD, damage legend
  survey.py       detections -> road-condition report
  video.py        frame iteration, codec-probing writer
data/videos/      drop clips here
data/images/      drop stills here
outputs/          annotated video, stills, CSVs
weights/          downloaded checkpoints (gitignored)
```

`labcore` has no Streamlit dependency — the app and the CLI are both just
front-ends to it.

---

## Notes

- **Confidence defaults differ by family.** RDD models default to 0.15, pothole
  models to 0.25. Comparing two families at one threshold favours whichever is
  calibrated for it; the sidebar threshold applies to both, so check both
  families at their own defaults too.
- **Codec.** `VideoSink` probes `avc1` → `H264` → `mp4v` by actually writing a
  few frames, because `VideoWriter.isOpened()` returns true on some builds for
  codecs that then fail. If it lands on `mp4v`, the app says so — most browsers
  won't play it inline, so use the download button.
- **Fonts.** OpenCV's Hershey fonts are ASCII-only; overlay text is
  transliterated before drawing, so non-ASCII in a model `name:` won't turn
  into `??`.
- **Tracker reset.** Track IDs reset between runs. Comparing defect counts
  across models is only meaningful for the same clip and the same stride.
