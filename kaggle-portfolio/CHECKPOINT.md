# Checkpoint — 2026-07-16

Resume point. Branch: **`kaggle`**. Active phase: **15 — ML Refactor → v0.2.0**
(plan: `mvm/docs/phase-15-plan.md`; frozen bench: `research/BENCH-PROTOCOL.md`).

## ⭐ Where we are: 15.0–15.4 CLOSED and PUBLISHED; next = 15.5 (expert passport + diet)

| stage | status |
|---|---|
| 15.0 — lab foundation (CMU+OpenImages conveyors, frozen bench, LJ0) | ✅ closed |
| 15.1 — pose (5 experiments + integration + LJ1) | ✅ closed |
| 15.2 — transition kinematics (stabilizer + fall threshold, LJ2) | ✅ closed |
| 15.3 — tracker on true trajectories (fragmentation + dead-zone fix, LJ3) | ✅ closed |
| 15.4 — golden-scenario simulator (fusion/FSM/anomaly, LJ4) | ✅ closed |
| 15.5 — expert passport + diet (weight/speed) | ⬜ next |
| 15.6 — Rust port + release v0.2.0 | ⬜ |

Every ML number runs through the frozen protocol (`spike_pose/bench.py`, SEED=0, grouped
bootstrap CI, data+code passport). Rust production is still **v0.1.0** — the port is 15.6.

## Published on Kaggle (private, account `howadx`)

- Notebooks (Run-All **COMPLETE**, numbers verified on Kaggle):
  `lj0-building-the-lab`, `lj1-the-pose-model-rebuilt-honestly`,
  `lj2-physics-of-posture-transitions`, `lj3-why-walking-people-multiply`,
  `lj4-a-camera-that-never-lies`.
- Datasets (derived keypoints/boxes only, no images/BVH): `cmu-mocap-views`,
  `mvm-pose-landmarks` (+`landmarks_v2.npz`), `mvm-animal-ood`, `mvm-mocap-transitions`,
  `mvm-mocap-tracks`, `mvm-golden-scenarios`, `mvm-openimages-vrd-meta` (VRD CSVs, CC BY 4.0).
- Gotchas hit + baked in: dataset title ≤50 chars; kernel `id` must equal `slugify(title)`;
  Kaggle auto-runs on push so datasets must finish processing first; `_find` does a recursive
  `/kaggle/input` search (robust to dataset nesting); **API-pushed kernel runs resolve no DNS
  even with `enable_internet=true`** — anything a notebook fetches must be a mounted dataset
  (that's why the VRD CSVs are re-hosted).

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

## Next actions — 15.5 (expert passport + diet) → journal LJ5

- Per-agent passport: latency (reference hardware), weight, honest accuracy on a public bench
  (emotion/identity/detect) — where feasible, no retraining.
- Diet per 15.1: drop the 195 KB VAE (Mahalanobis won), forest size already 220 KB (−79%);
  revisit quantizations. Gate: total model weight and latency do NOT grow vs v0.1.0.

## Open debts (non-blocking)

- Kaggle assets are **private** → flip to public for portfolio when ready.
- 15.6 will port the 15.1 ship artifacts to Rust (bit-exact golden tests; posture output gains a
  `torso_flexion` field — contract change to document).
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
