# Qualifying tracker capture protocol

The historical one-person procedure remains at
`benchmark/tracker-depth/CAPTURE.md`. It is exploratory and does not satisfy
this active protocol.

The concrete operator choreography, split allocation, per-take checklist and
minimum first tranche are defined in `corpus-plan.md`.

## Required producer evidence

- record every Human association tick and every Scene detection tick;
- retain the exact high/low detections, current masks/landmarks, source stamp,
  admitted metric geometry, candidate decisions, lifecycle transitions, and
  publication outcome;
- keep RGB/depth artifacts in canonical Dataset Factory sessions with explicit
  consent and selected privacy mode;
- record sensor identity, registration, pairing skew, calibration revision,
  health/drop counters, binary revision, config/profile revision, and model
  digests;
- bound diagnostic traces and keep them outside `labels.jsonl`.

## Split admission

A session may enter calibration or held-out only when:

1. it closes cleanly with `session.toml`;
2. strict Dataset check and applicable pixel validator pass;
3. every referenced artifact exists and is SHA-256 pinned;
4. annotation coverage and reviewer provenance are complete;
5. production Rust replay can reconstruct every tracker tick;
6. all required scenarios for its declared domain are present;
7. no person, room setup, object instance, or parent session crosses splits.

Held-out data is frozen before a candidate is evaluated and remains unread
during feature/threshold selection. A result that lacks enough critical events
is `insufficient_evidence`, never a pass inferred from zero observed failures.
