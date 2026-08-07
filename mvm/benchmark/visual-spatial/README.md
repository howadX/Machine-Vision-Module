# Phase 19 Visual Spatial Replay

This directory is the executable contract baseline for Phase 19.0. Fixtures
are deterministic synthetic replay descriptions, not live-camera measurements
and not evidence that a production spatial expert exists.

The baseline checks honesty, reference-frame authority, source freshness and
bounded graph-delta envelopes. Runtime occupancy and mapping remain gated by
later Phase 19 slices. Additional executable gates live in:

- `motion-19.1/` — selected classical camera-motion expert;
- `semantic-19.2.1/` — semantic taxonomy and synthetic outcome corpus, with no
  runtime semantic expert yet;
- `semantic-19.2.2/` — measured PP-MobileSeg-Tiny and SegFormer-B0 export/parity,
  cadence-aware CPU/integrity spike and executable decision records.

Run:

```bash
python3 benchmark/visual-spatial/evaluate.py
python3 benchmark/visual-spatial/motion-19.1/evaluate.py
python3 benchmark/visual-spatial/semantic-19.2.1/evaluate.py
python3 benchmark/visual-spatial/semantic-19.2.2/evaluate.py
```

`labels.jsonl` remains the only durable semantic history. These fixtures are a
frozen benchmark corpus and never become a runtime store.
