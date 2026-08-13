# T2 pre-association measurement record

- Date: 2026-08-13
- Production seam: Human/Scene detector tick -> `AssociationObservation` -> domain tracker -> Fusion
- Measurement owner: `mvm-depth::measure_association_geometry`
- Sampling: current mask when present; otherwise 20%-inset bbox; bounded to 4,096 samples
- Durable-data impact: none; `labels.jsonl` remains the sole semantic history

## Contract result

Each observation carries the detector-local key, bbox and score plus `AssociationGeometry` stamped
with the exact RGB frame and, when admitted, its paired registered depth frame, measured skew and
calibration reference. No track ID, identity or canonical entity key exists at measurement time.

PersonTracker and ObjectTracker retain the existing RGB association result, map detection key to the
resulting track ID, and forward that same geometry value. Human and Scene packets therefore preserve
their independent source ticks. Fusion consumes the carried values and contains no owner-unprojection
or ROI/mask sampling implementation.

## Deterministic production replay

- Input: `replay/production-baseline.json`
- Input SHA-256: `b99dfab66f9615b7b3119ee06ae7f95f25924a7cc0f96e20b976e9e571fca8b8`
- Output SHA-256: `7feb15a9b78151598caff470230ee55dc284633866495fea0e7c6c1a2f18f478`
- Two independent executions: byte-identical
- Person measurements: 4 (2 paired `withheld/missing_owner_association`, 2 RGB-only
  `unknown/depth_unavailable`, including the low-score recovery tick)
- Object measurements: 3 (same intentional ownerless/RGB-only negative coverage)

The frozen replay intentionally has no registered owner map, so it proves exact packet binding and
honest negative propagation, not positive metric accuracy. Positive 2.0 m owner-unprojection is
covered by the Rust primitive test. Captured calibration/held-out accuracy remains a T4 gate.

## Gates

- mask and inner-bbox paths, including empty-mask withholding;
- stale depth, excessive skew, missing owner and calibration mismatch fail soft with typed reasons;
- RGB-only association parity for person and object trackers;
- mapping preserves the outward person display ID and object track ID;
- Fusion rejects future and duplicate per-track geometry before canonical mutation;
- grep audit finds sampling/unprojection calls only in `mvm-depth/src/association.rs`;
- replay schema/hash/split and exact-pair validation remain fail-closed;
- no new dependency, model, tracker table or durable store.
