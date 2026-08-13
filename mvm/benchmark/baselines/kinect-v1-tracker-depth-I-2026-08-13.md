# Kinect v1 block I — depth-assisted tracker comparison

Date: 2026-08-13 (Europe/Moscow)

## Method

The frozen corpus is `benchmark/tracker-depth/replay.json`, SHA-256
`16de552ebe25532502cb4a9196a0e2b5c009dd20acd33fe0bbdad2d77bbbc959`. It is explicitly
`synthetic_replay`, not captured Kinect ground truth. The stdlib-only evaluator compares:

- A: current bbox IoU plus 2D motion adapter;
- B: A plus a bounded impossible metric-Z jump gate;
- C: B plus bounded XYZ constant-velocity scoring.

The 13 scenarios cover crossing, fast motion, enter/exit, short/long occlusion, partial body,
sitting/lying, false detection, animal, object churn, depth holes, skew and packet loss. Person and
object aggregates are reported separately. `IDF1`, `AssA` and `HOTA` values are named `*_proxy`:
this compact synthetic adapter does not claim TrackEval-equivalent benchmark scores.

## Results

| Variant | IDF1 proxy | ID switches | Fragmentations | False births | False merges | Runtime delta |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.750 | 9 | 1 | 2 | 4 | baseline |
| B | 0.750 | 9 | 1 | 2 | 0 | +8.51% |
| C | 0.750 | 9 | 1 | 2 | 0 | +18.10% |

The object-only churn slice was identical for all variants (`IDF1/AssA/HOTA proxy = 1.0`, zero
switches/fragments/births/merges). Person results account for every observed difference. B and C
remove synthetic cross-depth false merges, but neither improves the required target metrics
(ID switches, fragmentation, false births). C adds no tracking benefit over B. Both measured Python
adapter runtime deltas exceed the plan's bounded 5% comparison gate; these microbenchmarks are only
relative evaluator costs, not end-to-end MVM latency.

## Verdict

**Select A; reject production depth cues for now.** The plan explicitly allows rejection. Promotion
is blocked by both absent captured/annotated Kinect tracker replay and lack of target-metric improvement.
`mvm-core::PersonTracker`, object tracker association, activities, zones and Cortex were not changed.
The synthetic result does show a candidate future hypothesis: re-measure B on captured crossing data,
but it is not sufficient authority to enable B.

Historical note (added after this frozen Block I decision): an exploratory 39-sample one-person
captured replay was subsequently recorded. It did not reverse the verdict: A/B were unchanged and C
regressed, while multi-person, object/mask and production-parity coverage remained absent. The active
corpus, implementation and promotion contract is now `docs/tracker-reliability-rgbd-plan.md`; this file
continues to describe the exact synthetic decision at the time it was made.

## Verification

```text
python3 -m unittest discover -s mvm/benchmark/tracker-depth -p 'test_*.py'
python3 mvm/benchmark/tracker-depth/evaluate.py --loops 5000 \
  --output mvm/benchmark/tracker-depth/results.json
cargo test --workspace --features kinect-v1 --no-fail-fast
cargo clippy --workspace --all-targets --features kinect-v1 -- -D warnings
cargo fmt --all -- --check
git diff --check
```
