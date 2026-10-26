# T4 fast admission and person ID authority record

- Date: 2026-08-20
- Runtime scope: opt-in `candidate`; legacy remains the default
- Durable-data impact: none; `labels.jsonl` remains the sole semantic history
- Dependencies/models: unchanged

## Root causes

1. Personhood used lifetime `high_hits`, so intermittent high detections could accumulate the three
   hits required for publication. Admission now uses an uninterrupted high run and stores a sticky
   result only after the full gate succeeds.
2. The fast detector and pose expert can agree on a static background object. Candidate birth now
   also requires independent evidence: face/identity evidence or scale-relative detector motion in
   the same uninterrupted high run. This applies only before first confirmation.
3. `PersonState.track_id` is a gallery-aware display id and can change after late recognition.
   `CanonicalFusion` previously used it as the entity key, causing a false retirement/reincarnation.
   `PersonState.tracker_id` now carries the stable association id used by lifecycle handoff, metric
   geometry, and canonical identity; display id remains external metadata.
4. The first live gate showed that detector motion alone could eventually promote the monitor.
   Scene already tracked the same region as `tv`/`laptop`, but detections were intermittent. Candidate
   admission now consumes the Scene-owned current or retained display lifecycle as a typed conflict;
   it blocks motion-only anonymous birth while face evidence and established people remain valid.

## C03 calibration regression

`C03-fast-motion-edges-20260813-a` is a schema `1.1` capture. It lacks exact asynchronous Human
application timing, so report schema `1.1` labels this run
`fast_legacy_captured_pose_one_tick_simulation`. Its provisional manual truth still has no operator
signoff and cannot select a production threshold or promote candidate.

| Metric | captured baseline | fast candidate |
|---|---:|---:|
| phantom truth-absent ticks | 125 | 0 |
| missed truth-present ticks | 29 | 83 |
| duplicate-present ticks | 16 | 0 |
| visible IDs | 9 | 1 |
| fragmentation proxy | 8 | 0 |
| ID-switch proxy | 14 | 0 |
| birth latency, source frames | 0 | 61 |

The zero-phantom result is useful only as a calibration regression. The large recall loss reflects
the historical Human cadence (p50 165.9 ms, p95 400.0 ms, 80.8% dropped source frames) and blocks
promotion. A qualifying schema `1.4` high-cadence capture plus signed truth and held-out/live gates
remain required.

## Live Kinect validation

Three bounded 600-frame runs used the current feature-enabled binary, candidate tracking, registered
Kinect metric depth, `--no-vision-log`, and no pixel/tracker capture. The first two runs exposed the
remaining display-gap failure: anonymous id `10` appeared for three snapshots over the monitor, with
horizontal/lying pose and no verified face. That evidence rejected both the motion-only gate and the
current-object-only conflict variant.

The third run used the retained Scene conflict. Across 82 snapshots it had 71 empty snapshots while
the monitor was repeatedly tracked as `tv`, then one real-person interval of three consecutive
snapshots under the single display id `10`, followed by eight empty snapshots. Entry carried a
verified face, upright upper-body geometry, head direction and wrists; motion reached 2.374
torso-lengths/s. There was one empty→occupied transition, one occupied→empty transition, zero
fragmentation, zero ID switch and zero post-exit phantom. Runtime processed 599 tracking ticks with
0.0000 dropped-frame ratio; tracking p50/p95 was 9.383/12.356 ms, and every agent reported zero
errors. This is a local live regression, not held-out promotion evidence.

A separate empty-room stress run then processed 1,800 Kinect frames (1,799 tracking ticks) over
65 seconds. All 259 snapshots remained `empty`: zero published people and zero person IDs. This was
not an inactive-detector pass—the Scene expert published the known display as `tv` in 32 snapshots
and `laptop` in three. Tracking dropped-frame ratio remained 0.0000, p50/p95 was 9.273/12.936 ms,
and every agent again reported zero errors. This shows the conflict prevents rather than merely
delays the known monitor phantom on the tested scene.

The final pre-capture movement/object-interaction gate exposed and then reproduced a detector
postprocess liveness bug. Weighted NMS re-tested its seed through IoU; a degenerate or non-finite
seed had self-IoU `0`/`NaN`, was returned to the remaining set, and could loop forever. The fixed
implementation consumes every seed unconditionally and rejects non-finite bbox geometry. On the
fresh release binary after that fix, Kinect delivered all 1,200 frames, Tracking processed 1,199
ticks with a 0.0000 dropped-frame ratio, and every agent reported zero errors. Tracking p50/p95 was
11.010/23.844 ms.

That run still failed the continuity gate. Its 201 snapshots contained 14 occupied snapshots split
across five runs with display IDs `12`, `10`, and `11`; there was never more than one published
person at once, and the 70-snapshot empty tail contained no phantom. Person/object overlap was
observed for laptop, bottle, and the display region, but no reliable manipulation activity was
published. Compact output cannot determine whether the gaps are withheld publication on one
retained internal track or actual retire/rebirth. The next qualifying diagnostic is therefore an
explicitly consented schema `1.4` capture with exact association, enrichment, birth-conflict, and
internal-ID decisions; no threshold or candidate promotion is selected from this failed gate.

## Historical replay compatibility

The T3 report remains an immutable record of the earlier lifecycle contract. The stricter
consecutive-high admission intentionally changes the current output for its synthetic
`production-baseline.json`: the final person is now `tentative`, because a low-confidence recovery
interrupts the high run before three consecutive accepted highs. The input is unchanged
(`b99dfab66f9615b7b3119ee06ae7f95f25924a7cc0f96e20b976e9e571fca8b8`); the current output SHA-256
is `5677a364204aa4f65866c77700c6ff0fafc53fba782c6ca785ed5058fe473d43`, byte-identical across two
executions. This is an explicit admission-policy change, not baseline parity.

## Regression gates

- intermittent high boxes cannot accumulate personhood;
- a static anonymous fast-path hypothesis remains tracker-local;
- independent motion or face evidence can admit a candidate without weakening the body vote;
- current/retained Scene display evidence blocks motion-only anonymous birth and is captured exactly
  in schema `1.4`;
- late gallery recognition preserves `EntityKey` and incarnation and emits no false exit/enter;
- retained and retired person IDs are internal tracker IDs;
- object replay remains byte-equivalent to its captured decisions.
