"""PyTorch ``IterableDataset`` over MVM Dataset Factory sessions (phase 16.4).

Streams saved frames + structured labels from ``labels.jsonl`` and lazily loads
pixels on demand — no need to fit a session in memory. Filters (from
``dataset_config.yaml``) select which frames enter the view.

Usage as a library::

    from export_pytorch import MVMSessionDataset
    from filters import Filters
    ds = MVMSessionDataset(root="~/.local/share/mvm/datasets",
                           filters=Filters(postures=["sitting"], require_depth=True))
    for sample in torch.utils.data.DataLoader(ds, batch_size=None):
        sample["frame"], sample["depth"], sample["labels"]

Or as a smoke CLI::

    python export_pytorch.py <root> [--config dataset_config.yaml] [--limit 8]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

from filters import Filters
from records import Record, iter_dataset, iter_session


def record_to_labels(rec: Record) -> dict:
    """Flatten a record's structured labels into a JSON-friendly dict. Preserves
    label *status* (``posture_status``) and per-signal quality (confidence,
    entropy) — a training head chooses how to treat ``withheld``/``unknown``."""
    return {
        "session_id": rec.session_id,
        "frame_id": rec.frame_id,
        "seq": rec.seq,
        "people": [
            {
                "identity": p.identity,
                "is_anonymous": p.is_anonymous,
                "posture": p.posture,
                "posture_conf": p.posture_conf,
                "posture_status": p.posture_status,
                "emotion": p.emotion,
                "emotion_conf": p.emotion_conf,
                "emotion_entropy": p.emotion_entropy,
            }
            for p in rec.people
        ],
        "objects": [
            {"class": o.cls, "confidence": o.confidence, "has_mask": o.has_mask}
            for o in rec.objects
        ],
        "activities": rec.activities,
        "quality": rec.quality,
    }


def record_to_sample(rec: Record, with_frame: bool = True, with_depth: bool = True) -> dict:
    sample: dict = {"labels": record_to_labels(rec)}
    if with_frame:
        sample["frame"] = rec.load_frame()  # H×W×3 uint8
    if with_depth and rec.has_depth:
        sample["depth"] = rec.load_depth()  # H×W float32 (relative)
    return sample


class MVMSessionDataset:
    """An ``IterableDataset`` (subclassed lazily so torch is a soft dependency).

    Point it at a whole dataset ``root`` (all sessions) or a single ``session_dir``.
    Multi-worker ``DataLoader`` is supported: records are round-robin sharded by
    worker so each frame is emitted exactly once.
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        session_dir: Optional[str | Path] = None,
        session_ids: Optional[list[str]] = None,
        filters: Optional[Filters] = None,
        with_frame: bool = True,
        with_depth: bool = True,
        transform: Optional[Callable[[dict], dict]] = None,
    ):
        if (root is None) == (session_dir is None):
            raise ValueError("pass exactly one of root= or session_dir=")
        self.root = Path(root) if root else None
        self.session_dir = Path(session_dir) if session_dir else None
        self.session_ids = session_ids
        self.filters = filters
        self.with_frame = with_frame
        self.with_depth = with_depth
        self.transform = transform

    def _records(self):
        if self.session_dir:
            return iter_session(self.session_dir)
        return iter_dataset(self.root, self.session_ids)

    def __iter__(self):
        # Round-robin shard across DataLoader workers (if any).
        wid, nworkers = 0, 1
        try:
            import torch.utils.data as tud

            info = tud.get_worker_info()
            if info is not None:
                wid, nworkers = info.id, info.num_workers
        except Exception:
            pass

        for idx, rec in enumerate(self._records()):
            if nworkers > 1 and idx % nworkers != wid:
                continue
            if self.filters and not self.filters.passes(rec):
                continue
            sample = record_to_sample(rec, self.with_frame, self.with_depth)
            if self.transform:
                sample = self.transform(sample)
            yield sample


def _as_iterable_dataset(ds: MVMSessionDataset):
    """Wrap into a real ``torch.utils.data.IterableDataset`` when torch is present,
    so it drops straight into a ``DataLoader``. Falls back to the plain object."""
    try:
        from torch.utils.data import IterableDataset
    except Exception:
        return ds

    class _Wrapped(IterableDataset):
        def __iter__(self_inner):
            return iter(ds)

    return _Wrapped()


def build_dataset(root: str | Path, config: Optional[str | Path] = None, **kw):
    """Convenience: dataset over ``root`` with filters from a YAML config."""
    filters = Filters.from_yaml(config) if config else None
    return _as_iterable_dataset(MVMSessionDataset(root=root, filters=filters, **kw))


def _main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-iterate an MVM PyTorch dataset")
    ap.add_argument("root", help="dataset root (…/datasets) or a single session dir")
    ap.add_argument("--config", help="dataset_config.yaml with filters")
    ap.add_argument("--limit", type=int, default=4, help="samples to materialize")
    ap.add_argument("--no-frame", action="store_true", help="labels only (skip pixels)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    filters = Filters.from_yaml(args.config) if args.config else None
    is_session = (root / "labels.jsonl").exists()
    ds = MVMSessionDataset(
        session_dir=root if is_session else None,
        root=None if is_session else root,
        filters=filters,
        with_frame=not args.no_frame,
    )

    n = 0
    for sample in ds:
        n += 1
        lb = sample["labels"]
        shape = sample.get("frame").shape if "frame" in sample else "—"
        dshape = sample.get("depth").shape if "depth" in sample else "—"
        print(
            f"[{n}] frame_id={lb['frame_id']} people={len(lb['people'])} "
            f"objs={len(lb['objects'])} frame={shape} depth={dshape}"
        )
        if n >= args.limit:
            break
    print(f"\nmaterialized {n} sample(s)" + (" (filtered)" if filters else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
