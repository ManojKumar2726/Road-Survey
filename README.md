# Road Survey

Road-condition survey work. Components live in their own directories.

## Components

### [`road-damage-lab/`](road-damage-lab/)

A bench for trying YOLO-based road-damage detectors on road video. Covers the
full RDD2022 taxonomy — longitudinal, transverse and alligator cracking plus
potholes — not just potholes. Switch between 27 registered checkpoints from a
dropdown, feed it a clip, and watch the boxes land with track IDs, confidence
and size, then read a road-condition report off the pass. Two models can run
side by side on the same footage.

Every model's class ids are normalised onto one canonical damage vocabulary, so
comparisons hold even between checkpoints that order their classes differently.

```bash
cd road-damage-lab
pip install -r requirements.txt
streamlit run app.py
```

See [road-damage-lab/README.md](road-damage-lab/README.md) for the model list,
the damage taxonomy, the CLI runner, and how to register your own weights.
