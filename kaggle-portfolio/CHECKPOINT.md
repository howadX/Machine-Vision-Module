# Checkpoint — 2026-07-16

Resume point. Branch: **`kaggle`**. Active phase: **15 — ML Refactor → v0.2.0**
(plan: `mvm/docs/phase-15-plan.md`; frozen bench: `research/BENCH-PROTOCOL.md`).
15.6 port runbook: `mvm/docs/phase-15.6-port-plan.md`.

## ⭐ Where we are: 15.0–15.5 CLOSED+PUBLISHED; 15.6 P0–P8 DONE + LIVE-VALIDATED — release v0.2.0 pending owner tag-push + LJ6

**Live-validated 2026-07-17** (real camera, v0.2.0 binary, Вадим id0 in frame, 2× `mvm run`, 0 agent
errors): P5 align-fix → `Вадим ✓` locked every tick; P1 `torso_flexion` live 5.7–11.7° + plausibility
`ok` (Mahalanobis); P6 vocab v2 → `using_object(mouse/keyboard/laptop/cell phone)` + `holding_drink`
kept + **`watching_tv` gone** (tv detected, no activity) + zero legacy verbs; P4 anomaly 0.00 calm.
Fall (E2) not staged (stunt-calibrated threshold, unsafe live). Details: `mvm/docs/phase-15.6-port-plan.md`
§Live validation.

| stage | status |
|---|---|
| 15.0 — lab foundation (CMU+OpenImages conveyors, frozen bench, LJ0) | ✅ closed |
| 15.1 — pose (5 experiments + integration + LJ1) | ✅ closed |
| 15.2 — transition kinematics (stabilizer + fall threshold, LJ2) | ✅ closed |
| 15.3 — tracker on true trajectories (fragmentation + dead-zone fix, LJ3) | ✅ closed |
| 15.4 — golden-scenario simulator (fusion/FSM/anomaly, LJ4) | ✅ closed |
| 15.5 — expert passport + diet (weight/speed, LJ5) | ✅ closed |
| 15.6 — Rust port + release v0.2.0 | 🔨 **P0–P8 done+committed+LIVE-VALIDATED**; release pending owner tag-push + LJ6 |

## 15.6 progress (act-by-act on branch `kaggle`, pause after each; workspace green each act)

Baseline was 597+2 tests; now **608 passed / 0 failed / 1 ignored**. Commits:
- **P0** `6f90e20e` — parity fixtures + maha class-order unify. `lj1_build_ship.py`: maha class
  order unified to forest alphabetical `[lying,sitting,standing]` (byte-only reorder, score/thr
  identical). New `spike_pose/lj6_golden.py` reads the shipped `.bin`s (forest traversal + min-maha)
  → `crates/mvm-posture/src/pose_v2_golden.in` (8 real-npz inputs, deterministic, compiles). New maha
  sha `bb85d87c`; forest sha unchanged `c902bf9b`.
- **P1** `1c7e730b` (+ `2fbc1a86` passport notes) — pose port. `compute_features` → 12 z-free feats
  (GRIP vext/hext over 13 joints); forest v2 embedded (`pose_rf_2dgrip_v2.bin`, dropped 873 KB one);
  `MahaOod` replaces the whole VAE stack (dropped `pose_vae_public.bin`, `normalize_landmarks`,
  `rodrigues`, `Linear/Cursor`); `torso_flexion` (feat[0]) threaded PoseResult→Detection→PersonTrack
  →PersonState→Scene JSON (`JsonFace.torso_flexion`) + snapshot. **Decisions taken:** plausibility
  vocab unified to `"low"/"ok"` (was 3 inconsistent) → `anomaly.rs:95` pose-OOD trigger **now live**
  (was dead); `posture_trusted` moved `"unusual"→"low"`; `recon_error` now always-None (VAE gone),
  field kept for schema stability. **Passport carry-items (→P8):** (a) z-free sit↔stand limit —
  `p_V_06` frontal desk-sitter reads `standing` at a near-tie (.42/.48); oracle relaxed to
  never-`lying` invariant + sit↔stand confusion ≤1; (b) anomaly pose-OOD now fires on maha OOD.
- **P2** `db1808b5` — tracker stale-identity center-relink (LJ3, Δ med −4.70/heavy −8.25). `tracker.rs`:
  identified track eligible for anonymous center re-link UNLESS its name is claimed by a confident
  detection this frame (`claimed_names`); tier-3 entry also admits stale low-conf identity. New unit
  reproduces phantom (count 2→1). Golden G6 + full golden stack green.
- **P3** `05e07bea` — posture SL-veto (LJ2, Δ CI [−3.62,−0.43]). `posture.rs`: direct `standing↔lying`
  commit needs `SL_DWELL=12` (vs dwell 3); licensed by `A[standing,lying]≈0.0013`; no aggregate
  regression by construction. 3 units.

- **P4** `1591b407` — anomaly E2 scene-level fall (LJ2/LJ4). `anomaly.rs`: per-id standing→lying
  window retired → per-track pelvis-height ring + kinematic rule. Fires `ANOMALY_FALL=0.6` when peak
  windowed descent ≥ `E2_FALL_VEL_MS=1.05` (median k=3, **time-based** `E2_WIN_S=0.4` window → fps-robust)
  AND floor endpoint ≤ `E2_FLOOR_M=0.35`; `E2_LOOKBACK_S=3.0` bounded ring. `e2_fall(samples, require_floor)`
  — shipped path `true`, velocity-only fallback `false` (both unit-tested: vault fires velocity-only, floor
  gate vetoes). New `Config.camera_height_m=1.75`; `FusionAgent` wires it; height = `camera_height_m − y`.
  **Coordinate crux resolved (unify y-down):** prod `position_3d` is y-DOWN (`pixel_to_3d`); golden fixture
  `pos3d` is world Y-up pelvis (floor=0) → `golden.rs` converts at injection (`y_cam=CAM_H−h`, CAM_H=1.75),
  no lab/JSON edit. **LJ4 notebook verified unaffected** (its E2 reads `pelvis3d`, not `pos3d`;
  people_together sign-invariant). Validated on golden data (lab recipe @15fps): G4 3.19 m/s/min 0.077 fires,
  G3 0.095 + all controls ≤0.13 quiet. 8 anomaly units (−2 old +6 new).

- **P5** `ce33e6a6` — SFace align-fix (LJ5 ⭐). `pipeline.rs`: new `identity_crop(frame,
  face_lm5_px, face)` feeds `id.identify_margin` the ArcFace-aligned 112 crop
  (`mvm_emotion::align_face(frame, lm, 112)`), raw-crop fallback on absent/degenerate lm (mirrors the
  emotion align path). SFace only `resize_exact`s its input, so this is the whole fix, +0 weight/latency;
  LFW held-out +19.5 pt (0.783→0.978; TAR@FAR=1e-3 0.173→0.949). Unit pins all 3 paths (aligned 112² /
  no-lm raw / degenerate raw). Owner re-validates identity-lock live.

- **P6** `9d15a4fb` — Vocab v2 (LJ4/S3 VRD audit, contract change #2). `fusion.rs`
  `interaction_label` → `Option<String>`: **keep** `holding_drink` (cup/bottle/wine glass/bowl, 145 VRD);
  **drop** `remote|tv` watching_tv (head-facing ≠ wrist contact); `keyboard|laptop|mouse|cell phone|book`
  → `using_object(<class>)` (flat verbs 0–15 instances, unsupervisable names). `activity.rs::max_dwell_s`
  folds using_phone's 1800 s cap into the `using_object(` family by prefix. Golden regenerated from
  `lj4_scenarios.py` (byte-diff = exactly 2 label strings, numbers identical — regen deterministic);
  Python mirror `lj4_fusion.py` updated in lockstep. `state.rs` sync test + fusion asserts updated.
  **Published `mvm-golden-scenarios`/LJ4 frozen at v1 by design** — the contract change lands in 15.6, not
  retroactively; a future LJ4-v2 re-push carries the regenerated npz + updated lj4_fusion.py.

- **P7** `eff1d778` — FSM exit-dwell sweep decision = **DEFER** (golden regenerated already in P6).
  Verified 12/15 keeps golden CI green (FLOOR P/R 1.000, idle G6 clean — no regression; 4 activity-FSM
  units hard-code old dwell counts, would need mechanical updates). NOT shipped: golden CI is FLOOR-only →
  can't adjudicate the benefit (stress precision 0.133→0.667 med / 0.029→0.545 heavy exists only vs LJ4's
  synthetic stress model); under normal load 5/8 already 1.000, so the sweep is an always-on cost (~0.5 s
  later event finalization + sub-0.8 s episode merges) for a stress-only synthetic gain. Logged as a
  one-line flip (5→12, 8→15) in `activity.rs` next to `ACTIVITY_EXIT_DWELL_OLD` for the owner post
  live-validation. Kept 5/8. No behavior change → workspace 608 green, golden green.

- **P8** `3bab0ce1` — Release v0.2.0 prep (final act, code side). `Cargo.toml:20` 0.1.0→0.2.0
  (version flows everywhere via `CARGO_PKG_VERSION`: `mvm --version`→`mvm 0.2.0`, MCP serverInfo). New
  `mvm/CHANGELOG.md` (v0.2.0 + v0.1.0). README: version/phase/tests badges, **Паспорт качества v0.2.0**
  table (LJ0–5 numbers + the 3 contract changes), honest-limits += z-free sit↔stand + fall-ceiling, Roadmap
  row 15, `using_mouse`→`using_object(mouse)`. AGENT.md: vocab-v2 + torso_flexion + `## Contract changes in
  v0.2.0`. mvm/CLAUDE.md status + phase-15 row updated (608 tests). Workspace 608 green.

### RESUME HERE → owner-only tail (release + LJ6) — NOT agent work
The code side of 15.6 is complete on branch `kaggle`. Remaining is owner/human:
1. **Merge `kaggle`→`master`** (or tag on `kaggle`), then **push tag `v0.2.0`** → CI `release.yml` builds
   x86_64 (+aarch64 once public) via `cargo build -p mvm-cli` and attaches assets. Version comes from the
   tag, not Cargo.toml.
2. **Live checklist** (real camera): sit/stand/lie/bend × azimuths + walking + identity-lock (verify the
   SFace align-fix locks Вадим) + a fall (E2 fires) + confirm `torso_flexion`/`using_object(...)` render.
3. **LJ6 — "Shipping v0.2.0: the scorecard"** — ✅ **AUTHORED + Run-All verified locally** (17.07):
   `notebooks/lj6/_build_lj6.py` → `lj6_shipping_the_scorecard.ipynb` (16 cells), kctl entry `lj6`
   (id `howadx/lj6-shipping-the-scorecard`, title "LJ6 — Shipping the Scorecard", **no mounts** — recomputes
   nothing, scorecard baked from `results/metrics.json` at build time so numbers can't drift; offline-safe).
   Sections: the LJ0–5 arc · 3-axis scorecard (accuracy/weight/speed before→after) · the 8 port acts P0–P8 +
   3 contract changes · **the honest ledger of 6 measured-and-rejected/deferred decisions** · live-camera
   validation · north-star close. **Remaining owner step: `kctl push lj6`** (external publish — not agent
   work). This closes phase 15.
See `mvm/docs/phase-15.6-port-plan.md` (P8) + `mvm/CHANGELOG.md`.

Every ML number runs through the frozen protocol (`spike_pose/bench.py`, SEED=0, grouped
bootstrap CI, data+code passport). Rust production is still **v0.1.0** — the port is 15.6.

## Published on Kaggle (private, account `howadx`)

- Notebooks (Run-All **COMPLETE**, numbers verified on Kaggle):
  `lj0-building-the-lab`, `lj1-the-pose-model-rebuilt-honestly`,
  `lj2-physics-of-posture-transitions`, `lj3-why-walking-people-multiply`,
  `lj4-a-camera-that-never-lies`, `lj5-weighing-the-experts`.
- Datasets (derived keypoints/boxes/models only, no images/BVH): `cmu-mocap-views`,
  `mvm-pose-landmarks` (+`landmarks_v2.npz`), `mvm-animal-ood`, `mvm-mocap-transitions`,
  `mvm-mocap-tracks`, `mvm-golden-scenarios`, `mvm-openimages-vrd-meta` (VRD CSVs, CC BY 4.0),
  `mvm-onnx-experts` (Apache/MIT ONNX), `mvm-lfw-verification`. RAF-DB is mounted from the
  community `shuvoalok/raf-db-dataset` (research-only license — never re-host).
- Gotchas hit + baked in: dataset title ≤50 chars; kernel `id` must equal `slugify(title)`;
  Kaggle auto-runs on push so datasets must finish processing first; `_find` does a recursive
  `/kaggle/input` search (robust to `/kaggle/input/datasets/<owner>/<slug>/` nesting);
  **API-pushed kernel runs resolve no DNS even with `enable_internet=true`** — no fetches AND
  no `pip install` (that's why VRD CSVs/LFW/ONNX are re-hosted and LJ5 runs experts through
  `cv2.dnn`, byte-equivalent to ort, max|Δ|≈2e-6); the Kaggle base image has no `onnxruntime`;
  push with kctl only (it resolves the authed **tool-venv** `.venv/bin/kaggle` itself and
  recreates `.stage/<nb>/` from scratch — a raw `kaggle kernels push -p .stage/...` once
  shipped a stale LJ5; both fixed in kctl 2026-07-16).

## The lab (15.0) — reusable tools

- **`spike_pose/mocap_to_views.py`** — CMU BVH → FK → 13 GRIP joints → virtual eye-level camera
  (175 cm). Emits **true metric `P3D`** so azimuth is a REAL viewpoint. Subjects 111+113 span
  every posture → class ⊥ subject. Bend trials included; lying-gate needs low pelvis (deep bend =
  standing). BVH pulled per-subject from `una-dinosauria/cmu-mocap` (gitignored cache).
  `mocap_views.npz` = 4425 poses / 34 trials.
- **`mocap_sanity_gate.py`** (+`features_13` = **12 features**, 13 joints, z-free variant),
  **`bench.py`** (frozen metrics + passport), **`openimages_{relations,fetch}.py`** (relations
  inventory + S3 fetch; `--mid/--exclude` for OOD sourcing), **`research/BENCH-PROTOCOL.md`**.

## Pose (15.1) — decisions, all measured

- **Ship 2D-GRIP (z=0)**: MediaPipe z is a cross-domain liability (3D collapses 0.955 in-domain →
  0.62 on gold, sitting recall 0.01; 2D holds 0.83). Prod `mvm-posture` still uses z → real fix.
- **mocap is a TRAINING asset**, not just a bench: lab+mocap gold R3 **0.894** (vs lab-only 0.83);
  synthetic azimuth aug refuted (needs the untrustworthy z; old "0.90–0.95" was circular).
- **Flexion channel** = continuous `torso_flexion`, pure arithmetic (2D `torso_inc`, MAE ~6°
  frontal / 3.5° profile; regressor no better) → +0 model weight.
- **OOD: Mahalanobis replaces the 195 KB VAE** — on real animal-OOD 0.825 vs recon 0.702
  (Denouden blind spot). ~1.8 KB, pure arithmetic.
- **Held-out azimuth** (group-disjoint by trial): lab+mocap flat 3.0 pt, ~0.92 on unseen people.
- **Ship artifacts** (`spike_pose/ship/`, gitignored, regen via `lj1_build_ship.py`): forest
  **220 KB** + Mahalanobis 1.8 KB + flexion 0 KB. **Pose footprint 1068 KB → 222 KB (−79%).**
  Experiment scripts: `spike_pose/lj1_{2d_vs_3d,augmentation,flexion,ood,azimuth_heldout,build_ship}.py`.
- **Honest limits carried forward:** lying recall ~0.78 (scarce class); 2D-GRIP ties raw (not beats)
  cross-domain — GRIP's value is invariance + interpretability + $0-Rust.

## 15.2 done (LJ2) — decisions to port in 15.6

- **E1 stabilizer:** forward-filter prior ties hand-tuned dwell on aggregate flip-rate (H-prior
  NOT established — honest negative). Ships as a **hybrid**: dwell + one per-edge veto
  (`sl_dwell=12` on the `standing↔lying` edge, licensed by measured `A[standing,lying]≈0`) →
  drives the false standing↔lying flip to **0** under occlusion stress (Δ CI [−3.62,−0.43],
  established), no aggregate regression. That edge = the fall-relevant one.
- **E2 anomaly fall rule:** *reached floor (min pelvis ≤0.35 m)* AND *peak downward pelvis
  velocity ≳1.13 m/s* (AUROC 1.000, 6 falls vs 8 controlled; velocity alone insufficient).
  Caveat: CMU falls stunt-only → threshold bounds clinical from above.
- Tools: `spike_pose/lj2_{transitions,fall_velocity,transition_matrix,stabilizer}.py`; artifact
  `mocap_transitions.npz` (35 trials @60fps); design `research/LJ2-DESIGN.md`. Numbers in
  `results/metrics.json["lj2"]`, passport commit `e0514bed`.

## 15.3 done (LJ3) — decisions to port in 15.6

- ⚠️ Needed a **new extractor** `lj3_trajectories.py`: `mocap_to_views` re-centers the pelvis per
  frame → kills walk translation. New one keeps world x,z (single global offset), tall-narrow
  YOLO-like bbox. Artifact `mocap_tracks.npz` (24 walk + 2 idle + 2 cross @15fps).
- Tracker ported to Python (`lj3_tracker.py`, byte-faithful to `tracker.rs`/`kalman.rs`), bench
  `lj3_bench.py` (bursty-occlusion emitter, grouped bootstrap CI).
- **Root cause = structural re-link DEAD-ZONE**, not the knobs: one early confident face sticks
  `track.identity` on a body-primary walker → the anonymous center re-link is skipped forever
  (`if t.identity is not None: continue`) → every IoU gap spawns a phantom (**57%** of frag).
- **Fix (+0 deps, one branch):** center re-link may recover an identified track through a stale
  identity unless that name is claimed by a confident detection this frame. Δ(fix−base): med
  **−4.70 [−8.78,−0.82]**, heavy **−8.25 [−12.79,−3.50]** (established); below no-face baseline.
- **Honest negatives:** Kalman-Q calibrated to real accel var (~2e-4 vs prod 0.01) is **worse**
  (7.75 vs 5.43) → refuted; gate marginal → **Kalman/gate constants unchanged in 15.6**.
- **T2:** idle 0/0; crossings no swap increase (2/2, 0/0). Numbers `results/metrics.json["lj3"]`.

## 15.4 done (LJ4) — decisions to port in 15.6

- **Lab:** bit-faithful fusion-stack ports (`lj4_fusion.py`, 39 Rust self-check scenarios) +
  scenario compiler (`lj4_scenarios.py`: 28 instances G1–G6, world-3D composition, 5 m camera —
  at LJ3's 3 m a lying body leaves the frame; settling frames dropped — they inject a phantom
  fall at t=0) + bench (`lj4_bench.py`, per-source RNG streams for paired ablations).
- **S1 gate taken:** FLOOR event-P/R = **1.000/1.000**, 0 phantoms/swaps. Under stress FP =
  100% fragments / 0 foreign; P/R invariant to tracker/stabilizer/face/flip — the knob is the
  **FSM exit dwell** (5/8→12/15 lifts P med 0.133→0.667, heavy 0.029→0.545) → 15.6 candidate.
- **S2 verdict measured:** prod fall rule = 3/10 FA on brisk lie-downs + 4/6 under stress (the
  fall itself churns the track id → per-id standing memory dies). **Ships: E2 scene-level,
  threshold recalibrated to 15 fps (~1.05) → 0/10 FA + 6/6 on FLOOR and STRESS** (thin margin
  0.99 vs 1.12 stated; CMU falls are stunts). Per-id window rule retires.
- **S3 verdict (directive confirmed by data):** 110k VRD instances — typing/using_mouse/
  watching_tv = **0**, using_phone 15, reading 24, holding_drink 145; holds generic 2153, at
  Table/Desk 1360; **no lying state in VRD** (CMU stays the only fall source). Vocabulary v2
  table in `research/LJ4-VRD-AUDIT.md` (fold → `using_object(class)`, drop watching_tv);
  contract change lands in 15.6.
- **CI gate:** `mvm-cli agents::golden` replays a 6-instance FLOOR subset (same compiled file
  the notebook scores) through the production crates — 597+2 tests green, negative control
  verified. LJ4 notebook is built FROM the lab sources (import surgery) — no code drift possible.
- Numbers: `results/metrics.json["lj4"]`; design+results: `research/LJ4-DESIGN.md`.

## 15.5 done (LJ5) — CLOSED + PUBLISHED, Run-All COMPLETE, numbers bit-for-bit

Design+RESULTS `research/LJ5-DESIGN.md`; numbers `results/metrics.json["lj5"]`; tools
`spike_pose/lj5_{latency,emotion,identity,detect,objects}.py`; kernel
`howadx/lj5-weighing-the-experts` (v4, verified 2026-07-16: status COMPLETE, remote == local
source 20 cells code-identical, log numbers == metrics.json to the last digit).
- **A latency/weight:** human chain 16.7 ms all-cores (<50 PASS, 3×); dominant SFace 36.9 MiB/5.2 ms;
  total ONNX 121.7 MiB; depth 232 ms / YOLO cadenced (off-budget). Reference-hardware (i7-12700K);
  presented in the journal as tables (Kaggle image has no onnxruntime).
- **B1 emotion (RAF-DB held-out):** neg-control HEALTHY (10.1 bug not reproduced); balanced-acc
  0.665 [0.643,0.688] / overall 0.726. Re-ran live on Kaggle via cv2.dnn ✓.
- **B2 identity (SFace/LFW):** aligned 0.9783 = published, **but prod ships UNALIGNED 0.7833 →
  +19.5 pt on the table** (`pipeline.rs:470`); TAR@FAR=1e-3 0.173→0.949. ⭐ Top 15.6 fix:
  `align_face(lm5,112)` on the id crop, +0 cost. Re-ran live on Kaggle ✓.
- **B3 detect (OpenImages val):** YuNet face recall 0.805; YOLO object passport 18 classes —
  interaction core 0.6–0.8, weak tail = cross-taxonomy. KEEP.
- **C diet:** SFace INT8 = **measured NEGATIVE** (lossless acc 0.979, weight −74%, latency ×6 →
  REJECT). Gate MET: ONNX flat, arithmetic 1068→222 KB (−79%). Verdicts for 15.6: align-fix SHIP +
  VAE-drop + forest-220 KB; keep all ONNX experts.

**kctl bug fixed (2026-07-16, the LJ5 stale-push root cause):** KAGGLE was resolved from
`sys.executable` → wrong/broken CLI under the main repo venv (kagglesdk broken on py3.14) →
kctl push unusable → manual `kaggle kernels push -p .stage/lj5` bypassed `_stage()` and shipped
a stale stage. Fix: kctl now resolves the tool-venv `.venv/bin/kaggle` explicitly (fails loudly
if absent) and `_stage`/`_stage_data` recreate the stage dir from scratch. Verified: build lj5
deterministic (20 cells), stale files wiped on stage, dry-push OK under the broken interpreter.

## 15.6 in progress — Rust port + release v0.2.0 (act-by-act with pauses)

Port runbook = **`mvm/docs/phase-15.6-port-plan.md`** (source of truth: bit-exact contracts, 8 acts
P0–P8, dependencies, risks). Ship artifacts present: `spike_pose/ship/pose_rf_2dgrip_v2.bin` (220 KB),
`ship/pose_maha.bin` (1.9 KB). Acts: P0 parity fixtures → P1 pose (feats12/z=0 · forest v2 ·
Mahalanobis↔VAE drop · flexion output) → P2 tracker stale-relink → P3 posture SL-veto → P4 anomaly
E2 → P5 SFace align-fix → P6 vocab v2 → P7 golden refresh → P8 release. **Mode: act-by-act, pause
after each.** RESUME at **P0**.

**⚠️ Bit gotcha (golden pins it):** forest proba columns are sklearn order `[lying,sitting,standing]`;
Mahalanobis means/inv_cov are `bench.CLASSES` order `[standing,sitting,lying]` — **different**.

**P4.0 RESOLVED (measured, `spike_pose/lj6_fall_ablation.py`):** does absolute floor matter, or can
a floor-free relative signal replace it? CMU 6 falls/35 trials, rules @v≥1.13 m/s: velocity-only 6TP/
**1FP** (vault `01_08`), velocity+abs-floor **6/0**, velocity+rel-net-drop 5/1/1. Single AUROC:
velocity 0.994 > abs-endpoint 0.960 > rel-net-drop 0.891. **Floor carries real info** the relative
signal loses (it conflates "dropped from height" with "ended on the ground"). Decision (locked):
ship **velocity primary + coarse ground-endpoint veto from the known camera mount** (config
`camera_height_m≈1.75`) — recovers 6/0 without a surveyor's calibration; **velocity-only = documented
fallback** if the proxy is too noisy live; bed/couch-vs-floor deferred to affordance layer (north-star
7.D). North-star framing: ground-plane perception is a visual primitive (the eye *uses* the floor),
so a floor reference is human-analog, not a surveyor's abstraction.

## Open debts (non-blocking)

- Kaggle assets are **private** → flip to public for portfolio when ready.
- Doc debt "0.959"/"12 КБ" fixed across skeletons + research maps (15.07).

## Key files
- Tools/experiments: `spike_pose/{mocap_to_views,mocap_sanity_gate,bench,openimages_*,lj1_*,lj2_*,
  lj3_*,lj4_*}.py` + `extract_images.py` (shared MPPose runner). `kctl.py` — Kaggle driver + DATASETS/NOTEBOOKS.
- Rust CI gate: `mvm/crates/mvm-cli/src/agents/golden.rs` + `tests/data/golden_scenarios.json`
  (exported by `lj4_scenarios.py` — regenerate there, never hand-edit).
- Docs: `research/{BENCH-PROTOCOL,openimages-relations-inventory,DATA-EXPANSION-DESIGN,LJ2-DESIGN,
  LJ3-DESIGN,LJ4-DESIGN,LJ4-VRD-AUDIT}.md`, `mvm/docs/phase-15-plan.md`.
  Journals: `notebooks/{lj0,lj1,lj2,lj3,lj4}/`.
- Data (gitignored, regenerable): `data_probe/*.npz`, `mocap_bvh/`, `oi_images/`, `oi_meta/`,
  `ship/`. Gold raw images: `/home/dawoh/goldwork/{lyingv2,sitstand}/`.
