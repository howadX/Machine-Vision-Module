# Checkpoint — 2026-07-14

Resume point for the Kaggle ML-portfolio work. Branch: **`kaggle`**.

## Where we are

- **Kaggle platform fully wired.** `kctl.py` driver (tool-venv `.venv/`) does the whole loop:
  `build → push (=Save&Run All) → runstatus → fetch (metrics.json → results/metrics.json)`.
  See [WORKFLOW.md](./WORKFLOW.md).
- **All 6 notebooks + the `mvm-pose-landmarks` dataset are LIVE (private) on Kaggle.**
- **NB3b = GRIP is DONE** — the pilot, finished as an honest research notebook (real numbers, jitter-robust
  prose, trade-off frame). It is the only notebook with **real measured metrics**; NB0–NB2, NB4 still carry
  `[fill]` placeholders.
- **"method C" renamed to GRIP** (Gravity-Referenced Invariant Pose features) everywhere — portfolio, mvm
  docs, classifier.rs, memory. Kernel slug `grip-gravity-referenced-invariant-pose-features` (old deleted).

### Real metrics captured so far (`results/metrics.json`)
- **GRIP (NB3b), grouped 5-fold CV, 26 groups:** GRIP ≈0.93 ± 0.05 · raw coords ≈0.98 ± 0.03 (raw *wins*
  on clean data) · VAE-latent(~PCA32) ≈0.93. Azimuth 0→90°: GRIP flat 0.909; raw collapses ≈0.94→≈0.35.
  Forest ~270 KB (nb) / 873 KB (prod). The quoted 0.959 and "12 KB" did **not** reproduce.

## Next actions (in order)

1. **Stage A continues — polish remaining notebooks to Run-All + real metrics**, order by Run-All risk:
   **NB3 → NB2 → NB4 → NB1 → NB0**. For each: make Run-All-clean on Kaggle (attach datasets, emit
   `/kaggle/working/metrics.json` like GRIP does), run via `kctl push --yes`, `fetch`, replace `[fill]`.
   NB3 needs the COCO-2017 keypoints dataset attached (add to its `dataset_sources`).
2. **nb_bench dashboard** (task): once ≥3 notebooks have real numbers, a dedicated Kaggle notebook reading
   `results/metrics.json`, dataviz-styled (acc/latency/size bars, calibration). See [RESEARCH-PLAN.md](./RESEARCH-PLAN.md) Phase 0.1.
3. Then **Phase 1** of [RESEARCH-PLAN.md](./RESEARCH-PLAN.md) (honest re-measurement) → Phase 2 (per-model
   deepening) → Phase 3 (new models: temporal fall detector, calibrated emotion).

## Open debts / TODOs (owner or next session)

- **Cherry-pick `a1433c5a` (mvm GRIP rename) into `master`** so mvm-agents see GRIP. It touches only `mvm/`.
- **Reconcile `mvm/CLAUDE.md:220` "grouped 5-fold 0.959"** — reproduced ≈0.93 on the public snapshot; verify
  against `spike_pose/train_pose_p1.py` (likely a different/augmented data snapshot).
- **"12 КБ" honesty debt** still in `ARTIFACT-MAP.md`, `METHODOLOGY-AUDIT.md:170`, and skeletons
  (`nb3b-grip-skeleton.md`, `nb3-weak-supervision-skeleton.md`). Real: ~270 KB (nb) / 873 KB (prod). The
  published NB3b notebook is already correct; these internal docs are not.
- Skeleton docs still carry pre-measurement projections (0.959 etc.) — superseded by real runs; treat as
  historical scaffolding, not truth.

## Key files
- `kctl.py` — driver + NOTEBOOKS/DATASETS manifest (single source of truth).
- `results/metrics.json` — real numbers harvested from Kaggle runs.
- `RESEARCH-PLAN.md` — the ordered revision campaign. `METHODOLOGY-AUDIT.md` — the diagnosis.
- `_build_nb3b.py` — reference pattern for a finished notebook (metrics emit, honest framing).
