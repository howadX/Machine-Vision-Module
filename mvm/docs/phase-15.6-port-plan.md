# Phase 15.6 — Port winners → prod + release v0.2.0 (port runbook)

*Source of truth for the 15.6 Rust port. Every act is TDD (failing golden/unit test first →
implement → `cargo test --workspace` green). Pauses between acts, per phase-15 discipline.
Lab decisions being ported: `kaggle-portfolio/research/LJ{1..5}-DESIGN.md` + `CHECKPOINT.md`.*

Baseline: **12 crates, 597 workspace tests** green; prod = v0.1.0. Ship artifacts (gitignored,
regenerable via `spike_pose/lj1_build_ship.py`): `ship/pose_rf_2dgrip_v2.bin` (220 KB),
`ship/pose_maha.bin` (1.9 KB).

## Bit-exact contracts (from the Python lab — golden tests pin these)

- **Features:** `mocap_sanity_gate.features_13(lm)` with **z zeroed**, **12 features** in order
  `torso_inc, thigh_inc, shank_inc, knee_flex, hip_flex, elbow_flex, vext, hext, aspect,
  nose_above, ank_below, kn_below`. vext/hext computed over the **13 GRIP joints only**
  (`GRIP_IDX`), z=0 ⇒ radial extent uses |x−hip_x|. (Current prod = 14 feats WITH z + 2 asymmetry
  terms → both dropped.)
- **Forest** `pose_rf_2dgrip_v2.bin`: little-endian `<I` n_trees · per tree `<I` n_nodes · per node
  `<ifii3f` = feature(i32), threshold(f32), left(i32), right(i32), proba×3. **Proba columns are in
  sklearn `classes_` order = alphabetical `[lying, sitting, standing]`.** Same format as prod
  `Forest::load` already parses (classifier.rs:194) — only the file + N_FEAT change.
- **Mahalanobis** `pose_maha.bin`: `<II` (n_classes=3, n_features=12) · `<f` thr (99th pct of ID) ·
  per class: mean 12×f32 + inv_cov 12×12×f32 row-major. score = min_c √((F−μ_c)ᵀ P_c (F−μ_c));
  **OOD iff score > thr**.
  ⚠️ **Class order UNIFIED to the forest = alphabetical `[lying, sitting, standing]`** (decided
  2026-07-16). Rationale: the score is a pure `min` over classes — the maha class order is
  *functionally inert* (no positional index, no per-class label ever emitted; verified in
  `lj1_ood.py:63`, `lj1_build_ship.py:76`, and every bench consumer), whereas the forest column
  order is fixed by sklearn `classes_` **and already load-bearing in prod** (`POSE_LABELS`,
  `PoseResult.probs`, gold-R3). So we move the free side: one `POSE_LABELS`/`CLASS_ORDER` constant
  drives both loaders + golden. **Do NOT unify the other way** (forest→bench order) — that permutes
  forest columns and touches the prod contract (R1) for zero gain.
  ⚠️ **The real bit-invariant golden must pin is NOT the class order** (score is permutation-
  invariant, so a reorder is silent) **but the `mean_c ↔ inv_cov_c` pairing** — mean[i] must stay
  zipped with inv_cov[i]. Golden asserts `maha_score` (order-invariant) + a paired-integrity check.
- **Flexion** = feature[0] `torso_inc` raw (arithmetic, +0 weight). Exposed as `torso_flexion`.
- **Anomaly E2** (15.4): fall = *reached floor* AND *peak downward pelvis velocity ≳ 1.05 m/s @15fps*;
  scene-level, retires the per-id standing→lying window.
- **Tracker** (15.3): center-relink may recover an identified track through a **stale** identity
  unless that name is **claimed by a confident detection this frame**. One branch at tracker.rs:465.
- **Stabilizer** (15.2): dwell + per-edge veto `sl_dwell=12` on the `standing↔lying` edge.
- **Identity** (15.5): feed SFace the **aligned** crop (`mvm_emotion::align_face(frame, lm5, 112)`),
  not the raw bbox (pipeline.rs:473).
- **Vocab v2** (15.4): fold `typing/using_mouse/watching_tv` → `using_object(class)`, drop
  `watching_tv` (0 VRD instances). fusion.rs:213-223.

---

## Acts (sequenced; each ends at a green `cargo test --workspace`)

### P0 — Foundation: parity fixtures (non-production)
- **Unify maha class order first:** in `lj1_build_ship.py` change the maha loop `for c in bench.CLASSES`
  → `for c in POSE_LABELS` (alphabetical `[lying, sitting, standing]`), so `pose_maha.bin` and the
  forest share one order. Also fix the stale print (`:82` says "14-d" — the committed .bin is 12-d:
  1884 B = `8 + 4 + 3·(12·4 + 12·12·4)`). Cosmetic but the log must not lie.
- Regenerate ship artifacts deterministically (`lj1_build_ship.py`), confirm sha vs committed numbers
  (note: unifying the maha order changes `pose_maha.bin` bytes — re-record its sha as the new baseline).
- New Python parity exporter `spike_pose/lj6_golden.py`: for a fixed set of landmark inputs
  (reuse `pose_p1_golden` inputs + a lying/sitting/standing/bend spread) emit reference
  `(features12[z=0], forest_probs3, maha_score, flexion)` → `crates/mvm-posture/src/pose_v2_golden.in`.
  Replaces the stale `pose_p1_golden.in` (14-feat+VAE).
- **Gate:** exporter deterministic (empty diff on re-run); fixtures committed; maha `mean↔inv_cov`
  paired-integrity assertion drafted for the P1c loader test (class order alone is silent — see the
  Mahalanobis contract note).

### P1 — Pose port (load-bearing) — crate `mvm-posture` + pipeline OOD + state contract
- **P1a** `compute_features` → 12 feats, z=0, Python order. `N_FEAT 14→3`-consumers audited.
  Failing `features_v2_match_python` (tol 1e-4) → implement.
- **P1b** embed `pose_rf_2dgrip_v2.bin` (replace `pose_rf_forest.bin`), `Forest::predict(&[f32;12])`.
  Test probs match golden; gold-R3 semantics unchanged (labels lying/sit/stand).
- **P1c** Mahalanobis replaces VAE: new `MahaOod` loader (`pose_maha.bin`), **drop
  `pose_vae_public.bin` + the whole `PoseHead` VAE stack**. `PoseResult.plausibility` semantics →
  OOD from maha thr. Rewire the downstream gate (pipeline.rs `POSTURE_OOD_LOGPZ=170` → maha) and
  the anomaly pose-OOD trigger (anomaly.rs:95 `plausibility==Some("low")`).
- **P1d** flexion: add `torso_flexion: Option<f32>` to `PoseResult` = feature[0]; thread to
  `PersonState` → StateFile/snapshot/recap/MCP (state.rs:117 + populate sites 381/927/1225 +
  statefile.rs). **Contract change #1 — document.**
- **Gate:** all mvm-posture tests green (incl. desk-occlusion oracle lib.rs:438); bit-exact golden;
  workspace green.

### P2 — Tracker stale-identity center-relink — `mvm-core/tracker.rs`
- Change guard at tracker.rs:465 `if t.identity.is_some() { continue; }` → skip only if the track's
  identity name is **claimed this frame** (confident detection in `matched_ids`/claimed-names set).
- **Gate:** new unit reproduces walking fragmentation then shows the fix drops it; golden G6
  `lone_walker_never_meets_himself` stays green; no swap regression.

### P3 — Posture SL-veto — `mvm-core/posture.rs`
- Add per-edge veto: a `standing↔lying` flip needs `sl_dwell=12` consecutive confident frames
  (vs default dwell 3 elsewhere). Licensed by measured A[standing,lying]≈0.
- **Gate:** unit — false SL flip under occlusion stress → 0; no aggregate dwell regression.

### P4 — Anomaly E2 scene-level fall — `mvm-core/anomaly.rs` ⚠️ needs plumbing

**P4.0 — RESOLVED (measured, `spike_pose/lj6_fall_ablation.py`, CMU 6 falls / 35 trials).**
The question "does the absolute floor matter, or can a floor-FREE relative signal do its job?" is
answered by data, not opinion. Three rules scored against ground truth (v-threshold 1.13 m/s):

| rule | TP | FP | FN | note |
|---|---|---|---|---|
| 1 velocity-only | 6 | **1** | 0 | FP = `01_08` fast descent from height (jumped/vaulted, never reached floor) |
| 2 velocity + **abs floor** (E2 shipped) | **6** | **0** | 0 | floor gate rejects the vault cleanly |
| 3 velocity + **rel net-drop** (floor-free) | 5 | 1 | 1 | conflates "dropped from high" with "ended low" |

Single-signal AUROC: velocity **0.994** > abs-endpoint **0.960** > rel-net-drop **0.891**. The floor
carries real information the relative signal loses: `01_08` drops a lot relative to itself
(1.63→0.85, net 0.78) yet is NOT a fall because it never reached the ground; abs-floor sees that,
rel-net-drop doesn't (and it also misses stunt-fall `85_15` that bounces back up).

**North-star framing:** ground-plane perception is itself a visual primitive — the eye *uses* the
floor (from head/body pose priors), it does not survey it. So a floor reference is human-analog, not
a surveyor's abstraction; "floor-free at any cost" would *lose* human-likeness, not gain it.

**Decision (locked):** ship **velocity (primary, prediction-error salience) + a coarse ground-endpoint
veto derived from the known camera mount** (config `camera_height_m≈1.75`, ~level floor) — recovers
E2's 6/0 power without a surveyor's calibration. Velocity-only stays the **documented fallback** if
the proxy proves too noisy live (scale wobble). True bed/couch-vs-floor disambiguation is explicitly
**deferred to the affordance layer (north-star 7.D)** — a context task, not an anomaly heuristic.
Honest caveat → passport: the proxy inherits the uncalibrated scale (tail #3, ±15-30%) so the 0.35 m
gate is approximate; CMU falls are stunt-only (threshold bounds clinical from above).

**Port work:**
- Add per-track pelvis-y ring buffer (history) + `camera_height_m` to `Config`; compute pelvis
  height-above-floor = `camera_height_m − position_3d.y` (sign per MediaPipe y-down; verify live).
- Replace the per-id standing→lying window (anomaly.rs:73-93) with the scene-level E2 rule
  (peak downward pelvis velocity ≳ recalibrated-to-prod-fps ~1.05 AND ground-endpoint ≤ ~0.35 m).
- **Gate:** golden FLOOR subset fires; brisk lie-down / fast crouch stays quiet; velocity-only
  fallback path unit-tested; threshold recalibrated to prod fps.

**SHIPPED (P4, 2026-07-17).** `Config.camera_height_m=1.75`; `AnomalyScorer` keeps a per-track
pelvis-height ring (`E2_LOOKBACK_S=3.0`, bounded) and fires `ANOMALY_FALL=0.6` when peak windowed
descent ≥ `E2_FALL_VEL_MS=1.05` (median-smoothed k=3, time-based `E2_WIN_S=0.4` window — fps-robust)
AND floor endpoint ≤ `E2_FLOOR_M=0.35`. `e2_fall(samples, require_floor)` — the shipped path passes
`true`; `false` is the velocity-only fallback (both unit-tested: the vault fires velocity-only but the
floor gate vetoes it). Per-id standing→lying window retired. **Coordinate unify (y-down) — the crux,
resolved:** prod `position_3d` is y-DOWN (`space::pixel_to_3d`), so `height = camera_height_m − y`; the
golden fixture carries `pos3d` as world Y-up pelvis height (floor=0), so `golden.rs` converts at the
injection boundary (`y_cam = CAM_H − height`, `CAM_H=1.75`) — mirrors production, no lab/JSON edit.
**LJ4 notebook verified unaffected:** its own E2 reads `pelvis3d[1]` (Y-up, `lj4_bench.py:341/421`) not
`pos3d`; `people_together` uses 3D distance (sign-invariant under the y flip). Validated on the golden
data itself (lab recipe @15 fps): G4 descent **3.19 m/s** / min-pelvis **0.077** → fires; G3 controlled
**0.095 m/s** and every other control ≤ 0.13 → quiet (huge margin). Workspace **607 tests** green.

### P5 — SFace align-fix — `mvm-cli/pipeline.rs:473`
- Feed `id.identify_margin` the aligned crop: `face_lm5_px.and_then(|lm|
  mvm_emotion::align_face(frame, lm, 112)).unwrap_or_else(|| face.clone())` (mirrors emotion path
  437-440). +0 weight/latency.
- **Gate:** aligned path taken (unit on crop selection); no crash on degenerate lm; live identity-lock
  validated by owner (the +19.5 pt is the LJ5 held-out number, not re-measured in prod).

**SHIPPED (P5, 2026-07-17).** `pipeline.rs`: new `identity_crop(frame, face_lm5_px, face)` helper feeds
`id.identify_margin` the ArcFace-aligned 112 crop (`mvm_emotion::align_face(frame, lm, 112)`), raw-crop
fallback on absent/degenerate landmarks — mirrors the emotion align path. SFace only `resize_exact`s its
input (no internal alignment), so this is the whole fix, +0 weight/latency. Unit
`identity_crop_aligns_with_landmarks_and_falls_back_without` pins all three paths (aligned 112×112 /
no-lm raw / degenerate raw). The +19.5 pt (LFW held-out 0.783→0.978) is the LJ5 number — owner
re-validates identity-lock live. Workspace **608 tests** green.

### P6 — Vocab v2 — `mvm-cli/fusion.rs:213-223`
- `interaction_label`: `keyboard|laptop|mouse|cell phone|remote|tv → using_object(<class>)`; **drop
  `watching_tv`**; keep `holding_drink`, `reading`, relational (`people_together`,
  `seated_at_workstation`). Update golden episode-label asserts + fusion tests. **Contract change #2.**
- **Gate:** fusion + golden tests updated & green.

**SHIPPED (P6, 2026-07-17).** `interaction_label` → `Option<String>`, vocab v2: `holding_drink` **kept**
(cup/bottle/wine glass/bowl — 145 VRD instances, the one real specialization); `remote|tv` **drop**
(watching is head-facing, not a wrist contact); `keyboard|laptop|mouse|cell phone|book` →
`using_object(<class>)` (the flat verbs typing/using_mouse/using_phone/reading had 0–15 supervisable
instances — unsupervisable names). `activity.rs::max_dwell_s` folds the old `using_phone` 1800 s cap
into the whole `using_object(...)` family by prefix. Golden regenerated from `lj4_scenarios.py`
(`label="typing"` → `"using_object(keyboard)"` at G1/G2) — **byte-diff = exactly the 2 label strings**,
all scenario numbers identical (regeneration verified deterministic). Python lab mirror
`lj4_fusion.py` updated in lockstep (interaction_label + max_dwell_s) so the lab stays self-consistent.
Rust fusion + `state.rs` sync test + golden asserts updated. Workspace **608 tests** green. **Note:** the
published Kaggle `mvm-golden-scenarios` dataset + LJ4 notebook are **frozen at v1 vocab by design** (the
contract change lands in 15.6, not retroactively) — a future LJ4-v2 re-push must carry the regenerated
npz + updated `lj4_fusion.py`.

### P7 — Golden refresh + FSM exit-dwell sweep decision + full workspace green
- Golden `tests/data/golden_scenarios.json` was already regenerated deterministically in P6 (vocab v2).
- **FSM exit-dwell sweep (LJ4 candidate 5/8→12/15) — DEFERRED (not shipped).** Verified 12/15 keeps the
  golden CI green (FLOOR P/R 1.000, idle G6 clean — no regression); the 4 activity-FSM units that
  hard-code the old dwell counts would need mechanical updates. Not shipped, deliberately: the golden CI
  is FLOOR-only so it **cannot adjudicate the benefit** (stress precision med 0.133→0.667 / heavy
  0.029→0.545 exists only against LJ4's synthetic stress model), while under normal load 5/8 already
  scores 1.000 → the sweep is an always-on cost (~0.5 s later event finalization + sub-0.8 s episode
  merges) for a stress-only, synthetically-measured gain. Logged as a one-line flip (5→12, 8→15) in
  `activity.rs` for the owner to enable after live multi-person/occlusion validation. Kept at 5/8.

**SHIPPED (P7, 2026-07-17).** No code behavior change — golden was regenerated in P6; P7 is the sweep
decision (defer, documented in `activity.rs` next to `ACTIVITY_EXIT_DWELL_OLD`). Workspace **608 tests**
green, `agents::golden` green, negative controls hold.
- **Gate:** `cargo test --workspace` (597+ green), `agents::golden` green, negative controls hold. ✅

### P8 — Release v0.2.0
- Bump `Cargo.toml:20` 0.1.0→0.2.0. Quality passport in README/AGENT.md (numbers from LJ0-5 with
  journal links). CHANGELOG. Document contract changes (#1 flexion, #2 vocab-v2, posture OOD
  semantics). Owner live checklist: sit/stand/lie/bend × azimuths + walking + identity lock.
- **Passport carry-items from P1 (record honestly):**
  - **z-free sit↔stand limit:** 2D-GRIP drops MediaPipe z (cross-domain liability, LJ1) → loses the
    seated-thigh depth cue → a *frontal* desk-sitter can read `standing` at a near-tie (measured
    `p_V_06` .42 vs .48). Safety invariant (never false `lying`) holds; oracle bounds confusion ≤1.
  - **anomaly pose-OOD now live:** plausibility unified to `"low"/"ok"`; the previously-dead
    `anomaly.rs` pose-OOD trigger (weight 0.3) now fires on Mahalanobis OOD. `recon_error` is
    deprecated → always `None` (VAE removed) but kept in the state schema.
  - **anomaly fall rule is now kinematic (E2), not label-transition (P4):** fires on a fast
    pelvis descent that ends on the floor (velocity ≥1.05 m/s + floor ≤0.35 m), not on a
    standing→lying label flip. New config `camera_height_m` (default 1.75, eye-level rig) is the
    floor reference — inherits the uncalibrated 3D scale (tail #3, ±15–30%) so the floor gate is
    approximate; CMU falls are stunts → threshold bounds clinical from above; velocity-only is the
    documented fallback if the scale proxy is too noisy live.
- Release via existing CI (two jobs, `cargo build -p mvm-cli`, NOT `xtask bundle`). Then **LJ6 —
  "Shipping v0.2.0: the scorecard"** (before/after on the three axes).
- **Gate:** tests green, golden green, live checklist passed, release built by CI.

---

## Contract changes (document in P8)
1. **`torso_flexion`** — new optional field on posture output (StateFile/JSON/MCP/snapshot).
2. **Vocab v2** — activity labels `typing/using_mouse` → `using_object(class)`, `watching_tv` dropped.
3. **Posture OOD semantics** — `plausibility` now derived from Mahalanobis distance, not VAE logpz.

## Risks
- **R1** N_FEAT 14→12 ripples into `PoseResult.probs`, golden, oracle tests — contained in mvm-posture.
- **R2 (biggest)** E2 floor reference absent in prod → P4.0 gate; may reduce to velocity-only + honest gap.
- **R3** align +19.5 pt is LJ5 held-out; prod only asserts path + live identity-lock.
- **R4** FSM exit-dwell sweep is a latency↔stress-phantom knob — ship only if golden confirms, no idle regression.

## Order / dependencies
```
P0 ─► P1 ─┬─► P2
          ├─► P3
          ├─► P4 (needs P1 posture + position_3d; P4.0 first)
          ├─► P5
          └─► P6
   P2..P6 ─► P7 ─► P8
```
