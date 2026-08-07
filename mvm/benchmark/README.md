# MVM Visual QA Benchmark

This benchmark is part of MVM's visual subsystem: it evaluates evidence-backed
facts, bounded visual memory and advisory Cortex behavior. It is not a robot
controller, safety case, planner benchmark, or open-world VLM evaluation.

This directory is the executable evaluation contract for phase 18. Phase 19.0
adds its separate contract-only corpus under `benchmark/visual-spatial/`; Phase
19.1 adds the measured classical motion decision gate under `motion-19.1/`. Neither
benchmark directory is a runtime history store or competes with Dataset Factory
`labels.jsonl`.

## Layout

- `scenarios/manifest.json` is the canonical scenario index.
- `scenarios/<id>.json` contains replay inputs and expected observable claims.
- `questions/temporal-v1.json` defines deterministic QA prompts and answer keys.
- `schemas/` pins the JSON fixture envelopes for facts and Cortex output.
- `baselines/v0.3.0.md` records the pre-Cortex baseline without inventing
  unavailable measurements.
- `eval/` documents the evaluator contract.

Scenario fixtures start as skeletons in 18.0. A skeleton fixes identity,
required signals, questions and expected events; captured/replayed frame packets
are added before the feature whose gate consumes them. `status="skeleton"` must
never be reported as a measured benchmark result.

Synthetic replay fixtures 013–015 and 023–025 exercise the 18.5 forecast comparator
at its published 1 s and 2 s horizons. Fixture
016 exercises the 18.6 uncertainty inventory with positive and negative cases;
its score is a deterministic contract result, not a live-camera measurement.
Fixture 017 exercises the 18.7 hand-interaction affordance boundary, including
withheld attention and blocked claims; it is likewise synthetic replay only.
Fixture 018 exercises the 18.8 evidence-backed attention ranking signals. Its
top-3 coverage is a deterministic synthetic-replay contract, not a live-camera
event-recall measurement.
Fixture 019 exercises the 18.8b bounded re-observation lifecycle: semantic
dedupe, expiry, one-shot Depth permits, single resolution, explicit unavailable
external execution and overload deferral. It does not substitute for the
same-hardware live throughput gate required before expanding adaptive scheduling.
Fixtures 020–022 are the closure loops: each requires
`Accepted → Dispatched → Resolved` only after a newer stamped observation.
Neither the request nor its resolution can rewrite canonical facts.

Phase 18 is closed. The bounded live-camera result for commit `0159c8a5` is recorded in
`baselines/18.8b-live-2026-08-05.md`. It confirms real Scene/Depth permit
execution and zero agent errors, while leaving the controlled ≤5% A/B throughput gate for phase 18.8b.1.

The completed 18.8b.1 adaptive scheduling spike is recorded in
`baselines/18.8b.1-adaptive-mesh-2026-08-05.md`. It compares fixed, Scene-only,
and competing Scene+Depth permits on the same camera profile and under 4/2
CPU-core affinity. Runtime metrics remain diagnostic and are not canonical scene
facts or durable semantic history.

The 18.9 report-surface live validation is recorded in
`baselines/18.9-cortex-report-live-2026-08-05.md`. `current_cortex.json` is an
atomic latest cognitive artifact, while `current_state.md` remains facts-only
and `labels.jsonl` remains the sole durable semantic history.

## Invariants

1. Canonical facts are deterministic under sink cadence changes.
2. `withheld` and `unknown` are missing claims, never negative classes.
3. A source result newer than the fused camera tick is invalid.
4. Event consumers are non-destructive and report an explicit `EventGap`.
5. Runtime performance may regress by at most 5% on the same hardware/profile.

Run the contract gate:

```bash
cargo test -p mvm-core canonical_scene_contract
```

The full phase gate remains:

```bash
cargo test --workspace
```
