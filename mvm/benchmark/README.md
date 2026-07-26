# MVM Visual QA Benchmark

This directory is the executable evaluation contract for phase 18. It is not a
runtime history store and never competes with Dataset Factory `labels.jsonl`.

## Layout

- `scenarios/manifest.json` is the canonical scenario index.
- `scenarios/<id>.json` contains replay inputs and expected observable claims.
- `questions/temporal-v1.json` defines deterministic QA prompts and answer keys.
- `schemas/` pins the JSON fixture envelopes for facts and future Cortex output.
- `baselines/v0.3.0.md` records the pre-Cortex baseline without inventing
  unavailable measurements.
- `eval/` documents the evaluator contract.

Scenario fixtures start as skeletons in 18.0. A skeleton fixes identity,
required signals, questions and expected events; captured/replayed frame packets
are added before the feature whose gate consumes them. `status="skeleton"` must
never be reported as a measured benchmark result.

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
