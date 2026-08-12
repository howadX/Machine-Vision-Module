# Iteration C — Semantic Gate Contract

Iteration C repairs the acceptance boundary without rewriting Iteration B. Commit
`b9125c73` remains the truthful measured rejection of SegFormer-B0 for canonical
surface and runtime authority. The only decision emitted here is:

```text
authorize_shadow_runtime_only
```

This authorizes implementation of a default-off, measurement-only shadow path in
19.2.3a. It does **not** claim that the shadow runtime is implemented, that indoor
production quality has passed, or that canonical `SurfaceFact`, occupancy, free
space, passability, model distribution, or a live A/B is authorized.

## Contract boundary

The executable contract separates:

```text
raw_observation
  → selective_surface_evidence
  → fused_surface_fact
  → occupancy_free_space
```

Known indoor pixels outside the target surface taxonomy are
`other_non_surface`. Invalid, non-finite, low-confidence, high-entropy, corrupted,
or OOD evidence is `withheld`; abstention is not a negative class. Candidate
labels remain candidates: an opening label is not an open/free opening, and a
support-region label is not support geometry.

Future computed quality artifacts must pin ground truth and predictions with
source revisions and SHA-256 hashes, keep scene/session-level calibration and test
splits separate, and report per-class support, confusion, boundary metrics,
precision/recall/F1/IoU, ECE/Brier, and selective risk/coverage. The tiny files in
`fixtures/` exercise that contract end to end: the evaluator hashes GT and
predictions, computes exact-label accuracy, then verifies the expected result and
metric-artifact hash. They are explicitly `contract_fixture_only`: never quality
evidence and never production acceptance. ADE20K remains reference-only;
production acceptance requires a NYUv2/ScanNet-equivalent indoor RGB-D corpus plus
an MVM-camera holdout.

The evaluator pins and loads both Iteration-B `measurements.json` and
`quality_measurement.json`. `baseline.json` is a summarized snapshot derived from
those source fields; their SHA-256 pins preserve the complete artifacts. The
snapshot retains FP32/INT8 bytes, p50/p95/RSS, scheduling/amortized/aggregate
outcomes, all selected per-class IoUs and abstention/count/violation fields, and
both fine-tune results. Boundary metrics and an integrated live A/B remain
explicitly unmeasured. The zero `unknown`/occluded/stale → `free` requirement is
enforced at the authoritative occupancy boundary in Phase 19.3, not fabricated as
a raw semantic-pixel metric.

## Run

From `mvm/`:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 benchmark/visual-spatial/semantic-19.2.2/iteration-c/test_evaluate.py
PYTHONDONTWRITEBYTECODE=1 python3 benchmark/visual-spatial/semantic-19.2.2/iteration-c/evaluate.py
```

The test runner also sets `sys.dont_write_bytecode` before dynamically importing
the evaluator, so routine runs do not recreate `iteration-c/__pycache__/`.

The evaluator is stdlib-only and fail-closed: contract drift, missing provenance
requirements, a promoted dataset role, fabricated measurement status, or any
canonical/occupancy authorization exits non-zero.

Implementation sequencing and remaining gates are in
[`docs/phase-19.2.2c-plan.md`](../../../../docs/phase-19.2.2c-plan.md).
