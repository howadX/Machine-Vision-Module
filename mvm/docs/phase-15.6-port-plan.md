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
  per class **in `bench.CLASSES` order = `[standing, sitting, lying]`**: mean 12×f32 + inv_cov
  12×12×f32 row-major. score = min_c √((F−μ_c)ᵀ P_c (F−μ_c)); **OOD iff score > thr**.
  ⚠️ **Class order differs from the forest** — golden test must catch a mismatched map.
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
- Regenerate ship artifacts deterministically (`lj1_build_ship.py`), confirm sha vs committed numbers.
- New Python parity exporter `spike_pose/lj6_golden.py`: for a fixed set of landmark inputs
  (reuse `pose_p1_golden` inputs + a lying/sitting/standing/bend spread) emit reference
  `(features12[z=0], forest_probs3, maha_score, flexion)` → `crates/mvm-posture/src/pose_v2_golden.in`.
  Replaces the stale `pose_p1_golden.in` (14-feat+VAE).
- **Gate:** exporter deterministic (empty diff on re-run); fixtures committed.

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

### P5 — SFace align-fix — `mvm-cli/pipeline.rs:473`
- Feed `id.identify_margin` the aligned crop: `face_lm5_px.and_then(|lm|
  mvm_emotion::align_face(frame, lm, 112)).unwrap_or_else(|| face.clone())` (mirrors emotion path
  437-440). +0 weight/latency.
- **Gate:** aligned path taken (unit on crop selection); no crash on degenerate lm; live identity-lock
  validated by owner (the +19.5 pt is the LJ5 held-out number, not re-measured in prod).

### P6 — Vocab v2 — `mvm-cli/fusion.rs:213-223`
- `interaction_label`: `keyboard|laptop|mouse|cell phone|remote|tv → using_object(<class>)`; **drop
  `watching_tv`**; keep `holding_drink`, `reading`, relational (`people_together`,
  `seated_at_workstation`). Update golden episode-label asserts + fusion tests. **Contract change #2.**
- **Gate:** fusion + golden tests updated & green.

### P7 — Golden refresh + full workspace green
- Regenerate `tests/data/golden_scenarios.json` from `spike_pose/lj4_scenarios.py` (never hand-edit).
- Decide FSM exit-dwell sweep (15.4 candidate 5/8→12/15): ship only if golden stress subset improves
  with no idle regression; else defer with a logged note.
- **Gate:** `cargo test --workspace` (597+ green), `agents::golden` green, negative controls hold.

### P8 — Release v0.2.0
- Bump `Cargo.toml:20` 0.1.0→0.2.0. Quality passport in README/AGENT.md (numbers from LJ0-5 with
  journal links). CHANGELOG. Document contract changes (#1 flexion, #2 vocab-v2, posture OOD
  semantics). Owner live checklist: sit/stand/lie/bend × azimuths + walking + identity lock.
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
