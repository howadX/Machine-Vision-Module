# Checkpoint — 2026-07-15

Resume point. Branch: **`kaggle`**.

## ⭐ Strategic pivot (15.07, evening): Phase 15 = ML Refactor → v0.2.0

Two reference datasets anchor a lab-first rebuild of ALL MVM ML logic: **CMU mocap** (kinematic
truth: honest azimuth bench, transition/fall ground truth, tracker trajectories, golden scenarios)
+ **OpenImages** (static breadth, relations, measured-oracle). Full plan: **`mvm/docs/phase-15-plan.md`**
(stages 15.0 lab → 15.1 pose → 15.2 transitions → 15.3 tracker → 15.4 simulator → 15.5 experts
passport/diet → 15.6 port+release). Dataset Factory renumbered to **phase 16** (`phase-16-plan.md`).
Every stage closes ONLY with a narrative Kaggle lab journal (LJ0–LJ6, English) — these journals are
the foundation for a future Habr article (separate phase). Track A folds into 15.0/15.1; R3 gold
protocol stays the load-bearing pose test.

**Pose-method decision (research 15.07):** representation = **2D-GRIP** (z=0, features_13 variant);
invariance moves from features into DATA (view augmentation: local experiment — unaugmented 2D-GRIP
collapses on synthetic azimuth 0.94→0.44, azimuth-augmented train recovers 0.90–0.95 at all angles,
standing recall 0.92@90°); **NB3b's azimuth test was CIRCULAR** (3D-GRIP is invariant to y-rotation
by construction; honest bench needs true 3D = mocap); **bending = NOT a 4th class** but a continuous
torso-flexion channel (OWAS/REBA precedent), base 3-class + flexion_deg output; **OOD: Mahalanobis
on 14 features is the candidate** to replace the 195KB VAE (Denouden 2018: recon-error blind spot).
Scripts: `spike_pose/azimuth_2dgrip_test.py`, `azimuth_aug_test.py`.

Previous active track: **Data Expansion (Track A)** — see [DATA-EXPANSION-DESIGN.md](./research/DATA-EXPANSION-DESIGN.md)
and [NARRATIVE-STRUCTURE.md](./research/NARRATIVE-STRUCTURE.md).

## Where we are

- **NB3b = GRIP published (Kaggle v5)** — reference-grade research notebook (bootstrap CI, adversarial,
  calibration, HPO, ablation, per-class azimuth). Real metrics in `results/metrics.json`. Committed `3fc9cd0d`.
- **Pose data massively expanded.** From the **26-group lab probe** to
  **`spike_pose/data_probe/landmarks_v2.npz` = 8309 poses / 5580 groups**, all through OUR MPPose extractor
  (`spike_pose/extract_images.py`, cv2.dnn, consistent with the lab). Sources: lab (URFD+MultiPosture 1035) +
  gold#5 lyingv2 (1598, human 3-class) + gold#6 sitting/standing (5676, noisy binary).
- **Initial baseline trained + serialized:** **2D-GRIP on the CLEAN CORE (lab+gold#5, 2633 poses, 1417 groups)
  → grouped-CV 0.942 [0.919,0.960], balanced 0.941, per-class recall lying .94 / sitting .96 / standing .92.**
  Artifact `spike_pose/pose_rf_2dgrip.bin` (~840 KB). Big step up from the original one-person model.

## ⚠️ Pose-model problems we uncovered (the important part)

1. **The confound (class↔source).** Lab set had each posture from ONE dataset (sitting=MultiPosture,
   standing+lying=URFD) → adversarial RF predicts the *source* with AUC=1.0 → raw-coords' clean-data "win"
   was partly reading rig identity, not posture. (Diagnosed in NB3b.)
2. **⭐ The z-channel is a cross-domain liability — the biggest finding.** GRIP's 3D features (`hext`, `aspect`,
   3D angles) depend on MediaPipe's **z**, which is unreliable and domain-dependent. Train-lab → test-gold:
   **3D-GRIP collapsed to 0.662** (sitting recall 0.14, calling gold-sitting "lying"), while **2D-GRIP (z=0)
   = 0.834** and *beat raw*. Bootstrap CIs (1391 groups) don't overlap; effect is **direction-symmetric**
   (also gold→lab). → **The representation to ship is 2D-GRIP (z-free), not the current 3D GRIP.** Production
   `mvm-posture` still uses z → a real, actionable improvement (a 2D-GRIP variant).
3. **Narrow / single-person origin.** Original models trained on `poses_dataset.csv` = Vadim's own webcam
   captures (hand-labeled 0-9, single person) → memorized him → "worked so-so". CSVs are gone; only models remain.
4. **The old weak-labeler is broken.** `extract_coco_poses.py::classify_pose_heuristic` used `abs(knee-hip)`
   (sign loss) + desk-webcam assumptions ("no hips/knees → standing") → on lab true-labels **standing
   precision/recall = 0.00** (all standing → sitting). Reimagined LF v2 (gravity-referenced, signed, abstain)
   recovers standing, BUT a *good* geometric LF converges on GRIP's own features → weak-labels must NOT bear
   the raw-vs-GRIP comparison (would rig it). Hence **R3+R1**: clean human labels carry the confound test,
   weak labels are scale-only.
5. **Noisy data hurts.** Adding noisy gold#6 dropped full-set balanced-acc from ~0.94 (clean core) to ~0.82.
   → more data is not automatically better; label quality gates the gain. **lying is scarce (818)**.
6. **Extraction context matters.** Crop-vs-full-frame changes MediaPipe's input scale (~14pt swing); always
   extract full-frame to match the lab; `grp` must be namespaced by split (fixed) and ideally by source video.

## 💡 OpenImages thought (recorded)

OpenImages V7 (9.1M images; 1.9M dense; **16M boxes/600 cls**, 2.8M masks, **3.3M visual relationships /
1466 triplets**, 675K localized narratives, 38M points; images CC BY 2.0, annots CC BY 4.0) is **structurally
the right partner for MVM-as-data-factory**, not just another training set:

- **Its structure == MVM's output.** MVM is *Structured Decomposed Picture-to-Text* (entities + relationships
  + text). OpenImages is the only large corpus with **human-labeled visual relationships** (`holds/plays/rides/
  wears/on/at`) and **region-grounded narratives** → direct supervision for MVM's *relational* scene-graph and
  LLM-render, which almost nobody exploits this way.
- **Three roles in the factory (Phase 15):** (a) **breadth-seed / cold-start** for every non-temporal component
  (pose, objects, face/hand, masks); (b) **⭐ measured oracle** — run MVM's ensemble on OpenImages, measure how
  well its auto-labels match human labels → turns "the factory labels data" from a claim into a *measured*
  system (senior data-centric framing); (c) **domain-bridge** so the factory doesn't overfit one home.
- **Honest limits:** NO pose keypoints (still weak-label for pose — OpenImages gives diverse Person *crops*,
  not GRIP labels), NO temporal/video (tracking, activity-FSM, fall can't be supervised), NO identity/emotion/
  depth, fixed vocab, images CC BY 2.0 (publish only derived features). Sourcing works: annotations = direct
  Google-Storage CSVs; images = `s3://open-images-dataset/<split>/<id>.jpg` (public, no key — verified).
- Full analysis lives in this session; candidate to persist as `mvm/docs/openimages-for-mvm.md` (Phase-15 doc).

## Next actions

1. **OpenImages pose pilot** (Track A scale step): pull ~2-3K Person crops → MPPose → **consensus label**
   (geometric LF v2 ∩ 2D-GRIP baseline high-conf) → add to clean core → test on **held-out clean gold**.
   Question E3: does noisy scale help or hurt? (We KNOW noise can hurt — gold#6.) Pilot before scaling.
2. **Consider more CLEAN gold** (Roboflow #3/#4, 3-class) — likely a better quality lever than noisy weak data,
   but Roboflow needs an API key (access gate — ask Vadim).
3. **Fold findings into portfolio:** NB3b addendum / Article-2 material — "real cross-domain data exposed a
   hidden z-liability in clever 3D features; 2D-invariants + more data are the answer" (data-centric thesis).
4. **Portfolio Stage A** (parallel): NB3 → NB2 → NB4 → NB1 → NB0 still carry `[fill]`; nb_bench dashboard.

## Open debts / TODOs

- **2D-GRIP → production:** `mvm-posture` uses z; the z-ablation says a z-free variant generalizes better.
- **`grp` by source-video** (not just image_id/split) for fully-clean gold grouped-CV.
- **Cherry-pick `a1433c5a` (GRIP rename) into `master`**; reconcile `mvm/CLAUDE.md` "0.959"; fix "12 КБ" →
  ~270/873 KB in `research/ARTIFACT-MAP.md`, `research/METHODOLOGY-AUDIT.md`, skeleton docs.
- Data-expansion scripts + design docs committed `ed2007d1`; `data_probe/*.npz` stay gitignored (regenerable).

## Key files
- `spike_pose/extract_images.py` — MPPose runner (folder + coco + no-crop modes). `r3_confound_test.py`,
  `gold6_zablation.py`, `assemble_and_train.py`, `train_baseline_2dgrip.py` — the experiment scripts.
- `spike_pose/data_probe/landmarks_v2.npz` — assembled 8309-pose set. `pose_rf_2dgrip.bin` — baseline model.
- `research/DATA-EXPANSION-DESIGN.md` — the plan (R3+R1 framing, source inventory §11). `kctl.py` — Kaggle driver.
- Gold raw images: `/home/dawoh/goldwork/{lyingv2,sitstand}/` (outside repo).
