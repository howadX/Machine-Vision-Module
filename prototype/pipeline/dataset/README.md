# MVM Dataset Factory — exporters (phase 16.4)

Turn recorded MVM sessions into ML-ready datasets. The Rust side
(`mvm run --record`) accumulates self-describing sessions; this Python layer
compiles them into PyTorch / TFRecord / WebDataset views. Python is the research
consumer's ecosystem, so the ML-framework deps live **here**, never in the Rust
core (invariant #2).

## Input: a recorded session

```
<root>/sessions/<YYYY-MM-DD_HHMMSS>/
  frames/NNNNNN.jpg      # RGB, privacy-blurred per [dataset].privacy
  depth/NNNNNN.png       # 16-bit relative depth (optional), reversible via depth_range
  labels.jsonl           # one CanonicalObservation per saved frame
  session.toml           # closed-session summary
```

Produce sessions with:

```bash
mvm run --record                 # privacy=safe by default
mvm dataset list                 # see what you have
```

## Modules

| File | Role |
|------|------|
| `records.py` | Parse `labels.jsonl` → typed `Record`/`Person`/`Obj`; lazy frame/depth load; RLE mask decode. **No torch/tf.** |
| `filters.py` | `Filters` predicate over a `Record`, loadable from `dataset_config.yaml`. Quality-aware; `withheld` ≠ negative. |
| `export_pytorch.py` | `MVMSessionDataset` (`IterableDataset`), worker-sharded. |
| `export_tfrecord.py` | TFRecord shards (needs `tensorflow`). |
| `export_webdataset.py` | WebDataset `.tar` shards (writes with stdlib `tarfile` — no dep). |
| `dataset_config.yaml` | Export filters + task presets. |
| `test_dataset.py` | Pure-logic + PyTorch tests on a synthetic session. |

## Quick start (PyTorch)

```python
from export_pytorch import MVMSessionDataset
from filters import Filters
import torch

ds = MVMSessionDataset(
    root="~/.local/share/mvm/datasets",
    filters=Filters(postures=["sitting"], require_depth=True),
)
for sample in torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2):
    frame  = sample["frame"]    # H×W×3 uint8
    depth  = sample.get("depth")# H×W float32 (relative; larger = nearer)
    labels = sample["labels"]   # dict: people/objects/activities/quality
    ...
```

> **Multi-worker note.** With `num_workers > 0`, wrap your entry point in
> `if __name__ == "__main__":` — Python 3.14's default `forkserver`/`spawn`
> re-imports your module in each worker (standard multiprocessing requirement).
> The dataset itself is picklable and worker-shards each frame exactly once.

Smoke it without writing code:

```bash
cd prototype/pipeline/dataset
python export_pytorch.py ~/.local/share/mvm/datasets --config dataset_config.yaml --limit 8
python export_webdataset.py ~/.local/share/mvm/datasets /tmp/mvm_wds --config dataset_config.yaml
python export_tfrecord.py  ~/.local/share/mvm/datasets /tmp/mvm_tfr --config dataset_config.yaml
```

Each exporter also accepts a **single session dir** in place of `<root>`.

## Labels

`record_to_labels()` keeps the recorder's structure and *status*:

```json
{
  "frame_id": 388, "seq": 12,
  "people": [{"identity": "Вадим", "is_anonymous": false,
              "posture": "sitting", "posture_conf": 0.95, "posture_status": "observed",
              "emotion": "Neutral", "emotion_conf": 0.87, "emotion_entropy": 0.18}],
  "objects": [{"class": "cup", "confidence": 0.72, "has_mask": true}],
  "activities": ["using_object(cell phone)"],
  "quality": {"people_count": 1, "object_count": 3, "anomaly_score": 0.0}
}
```

`posture_status` is honest: `observed` / `withheld` (lower body occluded) /
`unknown`. A filter for `postures: [sitting]` will not accept a `withheld` frame —
absence of a posture is never a negative label (§16.0.2).

## Depth

`depth/NNNNNN.png` is a 16-bit min-max normalized relative-depth map; the raw
range is in each label's `frame.depth_range = [min, max]`. `Record.load_depth()`
de-normalizes back to the recorder's affine-relative scale (larger = nearer; not
metres). Frames without depth simply have no `depth` key.

## Feeding the training scripts

These views feed the existing trainers in `prototype/pipeline/` with **own-domain**
data (your room, your lighting) instead of only public datasets — the active-learning
loop of the north-star: record → export → train → deploy → record.

## Tests

```bash
cd prototype/pipeline/dataset && python test_dataset.py
```

Builds a synthetic session in a temp dir and checks RLE round-trip, parsing,
filters, and the PyTorch dataset. TFRecord/WebDataset writers are import-guarded
(tensorflow / a tar reader are optional).
