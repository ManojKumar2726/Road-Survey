# Road Survey

Road-condition survey work. Components live in their own directories.

## Components

### [`pothole-lab/`](pothole-lab/)

A bench for trying YOLO-based pothole detectors on road video. Switch between
nine registered checkpoints from a dropdown, feed it a clip, and watch the boxes
land with track IDs, confidence and size — or run two models side by side on the
same footage.

```bash
cd pothole-lab
pip install -r requirements.txt
streamlit run app.py
```

See [pothole-lab/README.md](pothole-lab/README.md) for the model list, the CLI
runner, and how to register your own weights.
