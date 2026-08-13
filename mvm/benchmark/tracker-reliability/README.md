# MVM tracker reliability benchmark

This directory is the active evidence root for
`docs/tracker-reliability-rgbd-plan.md`. It starts at T0 as a fail-closed
registry: historical evidence is indexed, while calibration and held-out splits
remain explicitly unavailable until qualifying captured sessions are recorded.

## Authority

- Rust `PersonTracker` and `ObjectTracker` behavior is the production subject.
- `CanonicalFusion` remains the only canonical entity/incarnation authority.
- `labels.jsonl` remains the only durable semantic history.
- Python is limited to offline manifest, annotation, and reporting checks. The
  historical Python A/B/C adapter is not production parity and cannot promote a
  runtime policy.

## Layout

- `contract.json`: fail-closed benchmark and promotion contract.
- `manifests/`: development, calibration, and held-out split registries.
- `inventory.md`: T0 caller, configuration, and downstream integration map.
- `capture-protocol.md`: requirements for future qualifying captures.
- `test_contract.py`: verifies schemas, split state, paths, and SHA-256 pins.
- `replay/`: frozen image-free inputs for the real Rust tracker entrypoints.
- `reports/`: reproducibility records and separate person/object aggregates.

Historical Block I inputs remain under `benchmark/tracker-depth/` until T1
replaces the adapter with production Rust replay. They are referenced by hash,
not silently relabeled as active acceptance evidence.

## T0 verification

```bash
python3 -m unittest discover -s benchmark/tracker-reliability -p 'test_*.py'
python3 -m unittest discover -s benchmark/tracker-depth -p 'test_*.py'
git diff --check
```

T0 changes no tracker runtime behavior.

## T1 production replay

```bash
cargo run -p mvm-cli --example tracker_replay -- \
  benchmark/tracker-reliability/replay/production-baseline.json \
  --expected-sha256 b99dfab66f9615b7b3119ee06ae7f95f25924a7cc0f96e20b976e9e571fca8b8
cargo test -p mvm-cli --example tracker_replay
```

The replay calls the same bound tracker methods as AgentMesh. Each decision event carries its exact
RGB stamp and, for paired packets, the exact depth stamp, skew and calibration revision. The output
contains no wall-time field, so repeated executions are byte-comparable. This development replay
refuses the held-out split.
