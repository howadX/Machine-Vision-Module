# Phase 19 Visual Spatial Replay

This directory is the executable contract baseline for Phase 19.0. Fixtures
are deterministic synthetic replay descriptions, not live-camera measurements
and not evidence that a production spatial expert exists.

The baseline checks honesty, reference-frame authority, source freshness and
bounded graph-delta envelopes. Runtime motion, semantic surfaces, occupancy and
mapping remain gated by later Phase 19 slices.

Run:

```bash
python3 benchmark/visual-spatial/evaluate.py
```

`labels.jsonl` remains the only durable semantic history. These fixtures are a
frozen benchmark corpus and never become a runtime store.
