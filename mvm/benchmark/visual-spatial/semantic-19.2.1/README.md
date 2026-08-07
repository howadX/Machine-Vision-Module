# Phase 19.2.1 Semantic Surface Contract

This is the executable taxonomy and synthetic outcome corpus for the future
semantic surface expert. It does **not** contain model measurements and does not
claim that runtime surfaces are available.

## Taxonomy boundary

The scored surface classes match the existing canonical `SurfaceKind` values:

```text
floor | wall | ceiling | support | opening
```

Expert labels are normalized before Fusion sees them. `table`/`table_top` map
to a support candidate; `door`/`window` map to opening geometry only.
`background`, `unlabeled`, OOD and unreliable pixels map to `unknown`.
`unknown` and `ignore` are never canonical surfaces. An `opening` never implies
open state, free space, clearance, safety or traversability.

## Corpus and gates

Nine compact pixel grids cover the five surface classes, partial openings,
support candidates, person occlusion, a no-floor counterfactual, ignored labels,
blur, glare, low light, mirrors and OOD/background inputs.

The evaluator computes per-class IoU, mean surface IoU, exact-grid boundary F1
for floor/opening, and unknown abstention recall. It fails immediately if any
truth pixel marked `unknown` becomes a surface, including floor. Thresholds in
`manifest.json` are the acceptance contract for the 19.2.2 model spike.

The committed predictions are a `synthetic_oracle`, so their 1.0 scores validate
the evaluator and corpus wiring only. Runtime availability remains 0/9.

Run:

```bash
python3 benchmark/visual-spatial/semantic-19.2.1/evaluate.py
python3 benchmark/visual-spatial/semantic-19.2.1/test_evaluate.py
```
