# T1 production Rust replay record

- Date: 2026-08-13
- Schema: `mvm-tracker-production-replay/1.0`
- Split: `development`
- Input: `replay/production-baseline.json`
- Input SHA-256: `b99dfab66f9615b7b3119ee06ae7f95f25924a7cc0f96e20b976e9e571fca8b8`
- Tracker entrypoints: `PersonTracker::tick_motion_bound`, `ObjectTracker::update_bound`
- Frames: 4 (2 exact registered RGB-D packets, 2 explicit RGB-only packets)
- Determinism: two independent executions produced byte-identical JSON stdout; timing is absent from the schema
- Trace bounds: 16,384 events/domain; 0 dropped

## Separate domain aggregates

| Domain | admission | prediction | candidate | assignment | birth | low recovery | lifecycle | miss | publication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| person | 4 | 3 | 3 | 2 | 1 | 1 | 1 | 0 | 4 |
| object | 4 | 3 | 2 | 2 | 1 | 0 | 0 | 1 | 3 |

Final person track `10` is `confirmed`, age 4, missed 0. Final object track `10000` is `cup`, age 3,
missed 0. Every event carries the exact RGB stamp; paired events also carry the exact depth stamp,
measured skew and calibration revision.

This is production-parity synthetic contract evidence, not captured accuracy evidence and not a
promotion record. The object trace explicitly labels the current implementation
`matched_greedy_baseline`; T7 owns replacement of that policy.

## Fail-closed gates

- input bytes must match caller-provided lower-case SHA-256;
- serde rejects unknown fields;
- schema is exact-versioned;
- only `development` and `calibration` execute; `held-out` is refused by this development tool;
- sensor envelopes and exact RGB-D pairing validate before either tracker tick;
- RGB source sequence must be strictly increasing;
- bbox, object score and COCO class are validated.

No trace is written to `labels.jsonl`; replay output is an explicit benchmark/debug artifact only.
