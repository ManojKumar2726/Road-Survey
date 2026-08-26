# Pothole Detection Lab

A bench for trying YOLO-based pothole detectors on road video. Pick a model from
a dropdown, feed it a video, and watch the boxes land — each one carrying a
persistent track ID, a class label, a confidence score, and its size relative to
the frame. Two models can run side by side on the same footage.

```
streamlit run app.py
```

---

## What's in the box

**Model switching.** Nine pothole checkpoints are registered out of the box
(`models.yaml`). Weights download lazily from the Hugging Face Hub on first use
and cache under `weights/`.

| id | source | notes |
|---|---|---|
| `samdutse-yolov8` | [Samdutse/pothole-yolov8](https://huggingface.co/Samdutse/pothole-yolov8) | YOLOv8s, 11.1M params — the starting point |
| `engjameso-yolov11` | [EngJamesO/pothole-detector](https://huggingface.co/EngJamesO/pothole-detector) | newer backbone |
| `engjameso-yolov12n` | ↑ same repo | nano YOLOv12, edge-device proxy |
| `engjameso-yolov9c` | ↑ same repo | YOLOv9 compact |
| `engjameso-yolov8n` | ↑ same repo | nano baseline |
| `engjameso-yolov8fpn` | ↑ same repo | FPN neck — try it on small/distant potholes |
| `keremberke-yolov8{n,s,m}-seg` | [keremberke/…-pothole-segmentation](https://huggingface.co/keremberke/yolov8n-pothole-segmentation) | segmentation heads; the lab draws their boxes |

The five `engjameso-*` entries are trained on the same data, so they're the
closest thing here to an apples-to-apples architecture comparison.

**On every box:** track ID, class, confidence, a confidence bar, pixel
dimensions, and area as a percentage of the frame. Colour is keyed to the track
ID (or class, or confidence — your pick).

**On every frame:** a HUD with the model name, device, frame counter,
detections in frame, unique IDs so far, inference time, and live FPS. Box labels
route themselves around the HUD rather than hiding under it.

**Per run:** an annotated `.mp4`, a per-detection CSV, and a per-pothole summary
table (first/last frame, mean and max confidence, mean size).

---

## Setup

Torch first, matching your CUDA version — then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verified on Python 3.10, torch 2.7.1+cu128, ultralytics 8.4.129, RTX 3060.
CPU-only works too; it's just slower.

> If your editor flags the imports as missing, it's pointing at a different
> interpreter than the one these packages went into. Select the Python 3.10 that
> `python --version` resolves to on your PATH.

---

## Using it

```
streamlit run app.py
```

**Video tab** — upload a clip, pick one already in `data/videos/`, or point at a
webcam index. Hit **Run detection** and frames stream in annotated as they're
processed.

**Image tab** — same overlay on stills. Handy for eyeballing a threshold before
committing to a long video.

**Models tab** — what's registered, what's cached, and a JSON dump of the loaded
model (params, classes, resolved weights path, device).

### Sidebar controls that matter

- **Mode: Compare two** renders two models on the same frames, side by side.
- **Confidence threshold** — the first knob to reach for. `keremberke-yolov8m-seg`
  scores much lower than the others on the same footage; that's a calibration
  difference, not necessarily worse detection.
- **Tracking** gives each pothole a stable ID so "unique potholes" is a real
  count rather than a per-frame tally. ByteTrack is faster; BoT-SORT survives
  occlusion better.
- **Frame stride** — process every Nth frame. The quickest way to skim a long clip.
- **Downscale frames to width** — big speedup on 4K dashcam footage; costs you
  small distant potholes.

Outputs land in `outputs/`, named `<clip>__<model-id>.mp4` / `_detections.csv`.

---

## CLI

Same pipeline, no browser:

```bash
python run_cli.py --list
python run_cli.py -m samdutse-yolov8 -s data/videos/road.mp4
python run_cli.py -m samdutse-yolov8 -s road.jpg --conf 0.3
python run_cli.py --benchmark -s data/videos/road.mp4 --max-frames 200 --no-save
```

`--benchmark` runs every enabled model over the same clip and prints a
comparison table (detections, unique IDs, mean confidence, ms/frame, FPS).
Other flags: `--half`, `--stride`, `--width`, `--no-track`, `--tracker`,
`--device`, `--show`.

---

## Adding a model

Append to `models.yaml`. That's the whole procedure.

```yaml
  - id: my-run-01
    name: "My YOLOv11 run 01"
    source: local              # hf | local | ultralytics
    path: weights/local/best.pt
    task: detect
    default_conf: 0.3
    notes: "trained on our own dashcam frames"
```

From the Hub instead:

```yaml
  - id: someone-yolov8
    source: hf
    repo_id: someone/their-pothole-model
    filename: best.pt
```

Anything Ultralytics can load works — v8/v9/v11/v12, detect or segment.
Add `enabled: false` to keep an entry listed but out of the picker.

---

## Layout

```
app.py            Streamlit UI
run_cli.py        headless runner / benchmark
models.yaml       the model registry
labcore/
  registry.py     models.yaml -> resolvable weight files (HF download + cache)
  detector.py     uniform wrapper over Ultralytics; normalised Detection records
  draw.py         boxes, ID/confidence chips, trails, HUD
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

- **Codec.** `VideoSink` probes `avc1` → `H264` → `mp4v` by actually writing a
  few frames, because `VideoWriter.isOpened()` returns true on some builds for
  codecs that then fail. If it lands on `mp4v`, the app says so — most browsers
  won't play it inline, so use the download button.
- **Fonts.** OpenCV's Hershey fonts are ASCII-only; overlay text is transliterated
  before drawing, so non-ASCII in a model `name:` won't turn into `??`.
- **Tracker reset.** Track IDs reset between runs. Comparing "unique potholes"
  across models is only meaningful for the same clip and the same stride.
