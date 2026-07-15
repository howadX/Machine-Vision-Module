# Kaggle workflow — how this portfolio is maintained

The portfolio is **generated, not hand-edited**. Each notebook lives in its own
folder `notebooks/<nb>/` holding a deterministic build script (`_build_nbX.py` →
`nbX_*.ipynb`) and its design skeleton (`nbX-*-skeleton.md`). A single driver,
`kctl.py`, builds them and syncs them to Kaggle. Kernel identity, titles, folders,
and dataset links live in one manifest inside `kctl.py`, so a push *updates* the
existing kernel instead of forking a duplicate.

## Layout

```
kaggle-portfolio/
├── kctl.py                # the driver (build/push/pull/status) + manifest
├── README.md WORKFLOW.md CHECKPOINT.md
├── notebooks/<nb>/        # one folder per notebook: build script + .ipynb + skeleton
├── research/              # ML-development research & design notes (not notebooks)
└── results/metrics.json   # real numbers fetched from Kaggle runs
```

## One-time setup

```bash
cd kaggle-portfolio
python3 -m venv .venv
./.venv/bin/pip install kaggle          # CLI 2.2.x
# Auth: ~/.kaggle/kaggle.json must exist (chmod 600). Already configured for user howadx.
./.venv/bin/kaggle kernels list --user howadx    # read-only smoke test
```

`.venv/`, `.stage/`, `pulled/` are git-ignored — only source (build scripts under
`notebooks/<nb>/`, the `.ipynb`, docs, `kctl.py`) is committed.

## The loop

```bash
V=./.venv/bin/python

$V kctl.py build [nb|all]      # regenerate notebook(s) from _build_nbX.py
$V kctl.py status              # local built? live on Kaggle? at a glance
$V kctl.py push  <nb|all> --yes   # upload; --yes required (publishing is outward-facing)
$V kctl.py pull  <nb|all>      # fetch remote copy into pulled/<nb>/ (for diffing)
$V kctl.py list                # my kernels on Kaggle
```

`nb` keys: `nb0 nb1 nb2 nb3 nb3b nb4`. A push overwrites the remote kernel with the
local notebook — **the local build is authoritative**. Kaggle allows one
`kernel-metadata.json` per folder, so push/pull stage each notebook into a throwaway
`.stage/<nb>/` with a freshly generated metadata file.

### Typical iteration

```bash
# edit notebooks/nb3b/_build_nb3b.py …
$V kctl.py build nb3b          # regenerate notebooks/nb3b/nb3b_grip.ipynb
# open nb3b_grip.ipynb on kaggle.com, Run All, read the printed metrics
# paste real numbers into the [fill] placeholders in _build_nb3b.py, rebuild
$V kctl.py push nb3b --yes     # publish the update
```

## Datasets

NB3b depends on a published, **keypoints-only** dataset (`mvm-pose-landmarks`,
derived from URFD + Zenodo MultiPosture — no images). It is defined in the `DATASETS`
manifest and sourced from `../spike_pose/data_probe/landmarks.npz`.

```bash
$V kctl.py push-data landmarks --create --yes   # first publish
$V kctl.py push-data landmarks --yes            # later: push a new version
```

After the dataset exists, NB3b's staged metadata automatically lists it under
`dataset_sources`, so Run-All on Kaggle finds it at
`/kaggle/input/mvm-pose-landmarks/landmarks.npz`.

## Metrics discipline

Build scripts never fabricate numbers. Prose placeholders are marked `[fill]`; the
code cells `print(...)` the real values on a Kaggle run, and those are what get pasted
back into the build script. This keeps every headline number traceable to an actual
execution — the point of a "living document" portfolio.

## Current state

| nb | kernel id | dataset |
|----|-----------|---------|
| nb0  | `howadx/emozid-an-svm-on-facial-geometry-2019` | CK+ via kagglehub |
| nb1  | `howadx/distilling-a-cnn-into-a-geometric-mlp` | FER2013 via kagglehub |
| nb2  | `howadx/a-pose-manifold-that-knows-when-it-s-surprised` | COCO keypoints |
| nb3  | `howadx/weak-supervision-at-scale` | COCO keypoints (publishes a derived dataset) |
| nb3b | `howadx/grip-gravity-referenced-invariant-pose-features` | `howadx/mvm-pose-landmarks` |
| nb4  | `howadx/identity-embeddings-are-wrong-for-emotion` | written teardown |

**All 6 kernels + the `mvm-pose-landmarks` dataset are live (private) as of 2026-07-13.**
Next step per notebook: open on kaggle.com, Run All, paste real metrics into the `[fill]`
placeholders in `_build_nbX.py`, rebuild, re-push.

**Kaggle gotchas baked into `kctl.py`:** the kernel `id` slug must equal `slugify(title)`,
and the title must be ≤ 50 characters (kctl asserts this before staging).
