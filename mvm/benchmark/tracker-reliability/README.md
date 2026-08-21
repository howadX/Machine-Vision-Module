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
- `corpus-plan.md`: operator-facing scripted scene list, split isolation, truth and approval workflow.
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

Timestamp-aware PersonTracker regression:

```bash
cargo run -p mvm-cli --example tracker_replay -- \
  benchmark/tracker-reliability/replay/person-irregular-cadence.json \
  --expected-sha256 9175befdfe700b016752a996d72307d114ad810e12be5a03fd5fb195861df721
```

The fixture covers fast motion, skipped source-frame ids, an empty Human tick and re-entry at an
irregular monotonic cadence. It is synthetic development evidence only. Captured replay reports
baseline/candidate births, single-person fragmentation/ID-switch proxies, phantom current ticks,
inter-tick Human cadence and dropped source-frame ratio. Schemas `1.3`/`1.4` replay the exact
recorded asynchronous enrichment boundary; `1.4` additionally records the typed Scene display
conflict used during person birth. Historical `1.1`/`1.2` captures lack that timing; they exercise
the real fast tracker entrypoint with captured pose applied after the tick and are labelled
`fast_legacy_captured_pose_one_tick_simulation` in report schema `1.1`. Historical captures also do
not contain Human processing duration, so processing p50/p95 remain explicit `null`; false re-links
remain `null` unless multi-person truth can adjudicate them.

## Fast TrackingAgent calibration

The production default remains `--person-tracking-mode legacy`. `shadow` keeps the legacy Human
observation authoritative and runs the detector-only TrackingAgent as diagnostics; `candidate`
makes TrackingAgent the sole PersonTracker owner and moves pose/identity/emotion/hands to a bounded
latest-wins Human enrichment worker. Candidate runtime reports `tracking_ms` separately from
`human_ms`, plus processed tracking ticks and source-frame drop ratio.

Opt-in candidate captures also write `human-enrichment-applications.jsonl`. Each record binds the
Human source packet and generation-safe internal handle to the exact tracking packet before which
the semantic patch was applied. The patch contains no bbox, Kalman, mask, or metric-position field.
Capture replay schemas `1.3`/`1.4` require this file to be pinned with
`--expected-enrichment-sha256`; the runner applies every recorded patch before its exact
`applied_before_binding` tick and fails on a missing record, ordering drift or outcome mismatch.
This file is calibration evidence, not `labels.jsonl` semantic history. It does not promote the
candidate without captured truth, a frozen held-out split, and controlled runtime A/B.

Candidate detector post-processing follows MediaPipe weighted NMS, then applies a strict
alignment-point same-body gate before the BYTE high/low split. Degenerate full-body scale geometry
is withheld before it can create a track. Kalman association deliberately retains the stable raw
detector anchor: a 2026-08-19 calibration take showed that using the full-body alignment ROI directly
introduced enough keypoint jitter to fragment a continuous person. Tick capture schema `1.4` stores
the raw detector box, all four alignment keypoints, and the explicit association-bbox kind; replay
continues to admit historical `1.1`–`1.3` formats.

## T3 lifecycle and publication

```bash
cargo test --workspace --all-targets --no-fail-fast
cargo test -p mvm-cli --example tracker_replay
cargo run -q -p mvm-cli --example tracker_replay -- \
  benchmark/tracker-reliability/replay/production-baseline.json \
  --expected-sha256 b99dfab66f9615b7b3119ee06ae7f95f25924a7cc0f96e20b976e9e571fca8b8
```

Person lifecycle evidence is split into high hits, low recoveries, consecutive high hits and
monotonic time since the last high/any observation. In the fast TrackingAgent path, admission needs
three timestamp-valid high hits plus a fresh generation-bound pose/body vote. Low recovery may
continue motion but cannot confirm birth or publish current presence; metric depth may veto an
impossible association but cannot prove personhood. Lost tracks are excluded from ordinary matching
and use only the bounded relink path. Tentative and established retention use separate monotonic
TTLs. ObjectTracker retains its existing two-tick policy. Observation packets carry occluded
evidence, retained IDs, and tick-local retirements separately; retention preserves an existing
`EntityKey` but cannot mint or publish an entity. See `reports/T3-lifecycle-publication.md`.

The opt-in fast path adds one independent birth-evidence gate: an anonymous detector/pose hypothesis
must either carry face evidence or move by a scale-relative amount during one uninterrupted high
run. Motion-only birth is withheld when the fresh Scene packet reports a confirmed current or
retained `tv`/`laptop` covering at least 20% of that proposal. The conflict is typed, stamped, and
recorded by capture schema `1.4`; it cannot delete an established person, and face evidence remains
stronger. This gate applies only before first confirmation; a confirmed person may remain motionless.
Tracker-local `tracker_id` is the association/lifecycle/canonical key, while `track_id` remains the
gallery-aware display id. Late recognition can change the display id without retiring or
reincarnating the canonical person. See `reports/T4-fast-admission-and-id-authority.md`.
