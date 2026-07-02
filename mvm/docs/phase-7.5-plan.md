# Phase 7.5 — Python-zero: native pose + bridge removal

> **Goal:** the last Python responsibility — **pose** — goes native, then
> `mvm_bridge.py` + the venv are removed. End state: one self-contained Rust
> binary, no Python anywhere, multi-person pose (fixes #3).
>
> **Principle (unchanged):** CPU-first, $0, native Rust. Footprint grows only for
> new ONNX experts; the VAE/classifier port is closed-form arithmetic (zero new
> deps). Builds on the Python-light hot-path shipped in Phase 7 (`--native-camera`,
> 25 fps, zero Python except pose).

---

## What is already known (grounded 2026-06-25)

### The pose chain (reference implementation to port)
```
Frame → person ROI → 33 BlazePose landmarks (x,y,z) → 99-vec
      → PoseVAE.encode (99 → z32)  → PoseClassifier (z32 → 6) → 6 poses
                          └→ decode → reconstruction_error → plausibility
```
Multi-persona = run per detected person ROI (this is what fixes #3: MediaPipe's
`.task` returned `poses=1`; we now drive the ROI loop ourselves).

### G1 — landmark models: **SOLVED** (OpenCV Zoo, ONNX, ort-loadable)
The PoseVAE is trained on **33 MediaPipe-topology** landmarks (`vposer_coco33.pth`,
input 99 = 33×3; hip indices 23/24 = BlazePose convention). There is **no
training pipeline** in the repo, so the landmark model MUST output that exact
33-landmark topology to reuse the weights. OpenCV Zoo provides exactly this,
TFLite→ONNX, same family as our YuNet/SFace:

| Model | Role | Notes |
|-------|------|-------|
| `person_detection_mediapipe_2023mar.onnx` | person ROI | the detection stage |
| `pose_estimation_mediapipe_2023mar.onnx` | 33 landmarks + seg mask | BlazePose, runs on ROI |

Two-stage: detector → ROI crop → landmark model (per person).

### G2 — weights to port (tiny, pure arithmetic)
From the checkpoints (already on disk):
- **PoseVAE** (`vposer_coco33.pth`, 206 KB): `BatchNorm1d(99)` → `Linear(99→128)`
  → ReLU → `Linear(128→64)` → ReLU → `fc_mu(64→32)` = z. Decoder
  `Linear(32→64)→Linear(64→128)→Linear(128→99)` for recon_error. (config:
  input_dim 99, latent 32, hidden [128,64], β 0.001.)
- **PoseClassifier** (`pose_classifier.pth`, 21 KB): `Linear(32→64)` → ReLU →
  `Linear(64→32)` → ReLU → `Linear(32→6)`. Dropout is identity at inference.
- Largest matrix is 128×99 → trivially hand-rolled in Rust, **zero new deps**.

### G7 — labels: **trivial** (already done upstream)
EN labels are in the checkpoint: `class_names = ['standing','sitting',
'lying_back','lying_side','arms_crossed','sitting_at_desk']` (RU variant also
present). Use `class_names` directly — no RU→EN mapping work.

### Plumbing — already in place
`Detection` and `PersonTrack` already carry `posture`, `posture_confidence`,
`recon_error`, `plausibility`. Native pose fills the existing fields; no new
magistrale. `mvm-posture` is an empty `Module` stub → becomes production.

### Footprint note
+2 ONNX experts (person detector + BlazePose, a few MB each). Justified per the
philosophy (footprint grows only for new ONNX experts). int8bq variants exist if
size matters.

---

## Stages

### 7.5.0 — Spike: de-risk the model FIRST 🔴 *main unknown*
Mirror the camera-spike discipline: prove the models load and produce correct
landmarks before building anything.
- Load both ONNX in `ort`; run person-detector on a test image → ROI; run
  BlazePose on the ROI → 33 landmarks.
- **Verify the 33-landmark order matches MediaPipe/PoseVAE training** (e.g. idx
  11/12 shoulders, 23/24 hips) — feed the landmarks to the *Python* PoseVAE and
  confirm a sane class; this is the topology gate.
- Nail preprocessing: ROI crop/rotation, input size, coordinate normalisation,
  z-scale convention.
- **Gate:** native landmarks on a test person ≈ MediaPipe's, and the Python
  PoseVAE classifies them correctly. If topology/preproc can't be matched →
  reconsider (retrain head, or different model).

### 7.5.1 — Acquire + bundle models
Download the two ONNX into `models/`; wire into `cargo xtask bundle` (embed).
- **Gate:** models load in a release build; bundle still produces `dist/mvm`.

### 7.5.2 — `mvm-posture` production: two-stage landmark inference
Stub → real. Person-detector → ROI(s) → BlazePose → `Vec<[f32;3];33]` per person.
TDD: black-image → no person; test-person image → 33 plausible landmarks.
- **Gate:** landmarks extracted natively per detected person.

### 7.5.3 — Port PoseVAE encoder + classifier (hand-rolled)
Closed-form Rust: BN(99) → Lin → ReLU → Lin → ReLU → fc_mu = z32; classifier
z32 → 6 logits → softmax; decoder → MSE = recon_error → plausibility. Export raw
weights (JSON/bincode) from the `.pth`; validate Rust output **bit-close to
Python** on fixed landmark inputs.
- **Gate:** Rust pose label + recon_error match Python (±ε) on recorded inputs.

### 7.5.4 — Multi-persona + wire into `--native-camera`
Run the pose chain per person ROI; fill `posture/posture_confidence/recon_error/
plausibility` on each detection. EN labels.
- **Gate:** pose on 2+ people simultaneously (fixes #3); appears in StateFile +
  `--dump-json`; `cam_test.py two_people` shows posture coverage > 0.

### 7.5.5 — G6: pose accuracy oracle
Build the missing quantitative check: native pose vs Python MediaPipe on a set of
saved frames / hand-labeled postures. Record per-class agreement.
- **Gate:** native pose agreement with MediaPipe ≥ acceptable on the labeled set
  (define threshold with the operator).

### 7.6 — Python-zero: remove the bridge ⚠️ *consent-gated*
Camera + detection + identity + emotion + 3D + **pose** now all native →
`mvm_bridge.py` has no remaining role. Remove it + the venv dependency; update
`cargo xtask bundle`; consider making `--native-camera` the default (it now has
pose).
- **HARD GATE — explicit user consent before deleting any Python** (standing
  rule). Pause and confirm, as in Phase 7.4.
- **Gate:** `mvm-cli` runs end-to-end with zero Python; venv not required to build
  or run; one self-contained binary.

---

## Order & dependencies
```
7.5.0 spike (model de-risk) ── do FIRST, gates everything
  → 7.5.1 acquire/bundle
  → 7.5.2 landmark inference ── 7.5.3 VAE/classifier port (parallel-ish)
        → 7.5.4 multi-persona + wire
        → 7.5.5 oracle
        → 7.6 bridge removal (consent)
```

## Open decisions (confirm before/at spike)
1. **ROI source:** `person_detection_mediapipe` (2nd ONNX, robust, MediaPipe-native)
   vs reuse the YuNet **face** bbox expanded to a body ROI (lighter, one fewer
   model, rougher for non-frontal/seated). **Recommend** the person detector
   (proven topology pairing); spike can A/B both.
2. **VAE/classifier:** hand-rolled raw-weights (closed-form, zero deps — matches
   philosophy) vs export to ONNX (less transcription risk, +files, ort already
   present). **Recommend** hand-rolled, validated bit-close to Python.
3. **`--native-camera` default:** flip to default once pose is native (7.6)?
4. **Bridge removal:** explicit consent at 7.6 (do NOT delete Python before).

## Risks
- **Topology/preproc mismatch** (BlazePose ROI rotation, z-scale, landmark order)
  — the spike (7.5.0) exists to kill this risk early; it's the one real unknown.
- **ROI for seated/ceiling-angle users** — person detector trained on upright
  full-body may underperform at ~45° ceiling view; validate in spike.
- **Footprint** — +2 ONNX; use int8bq variants if the bundle grows too much.

## What carries over from Phase 7 (assets)
- `tools/cam_test.py --native-camera` harness (posture coverage already reported).
- `mvm-pipeline::native` camera; `--native-camera` hot-path to extend.
- Plumbing fields (`posture`/`recon_error`/`plausibility`) already wired end-to-end.
