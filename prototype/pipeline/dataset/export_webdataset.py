"""Export MVM Dataset Factory sessions to WebDataset shards (phase 16.4).

WebDataset shards are plain POSIX tar files whose members share a key stem, e.g.::

    000000.jpg  000000.depth.png  000000.json

so this writer uses ``tarfile`` directly — **no dependency to write**. Reading
back is the usual ``webdataset.WebDataset("shard-{000000..N}.tar")`` (that side
needs the ``webdataset`` package).

    python export_webdataset.py <root> <out_dir> [--config dataset_config.yaml] \
        [--shard-size 512]
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Optional

from filters import Filters
from export_pytorch import record_to_labels
from records import Record, iter_dataset, iter_session


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _write_sample(tar: tarfile.TarFile, key: str, rec: Record) -> None:
    labels = record_to_labels(rec)
    if rec.depth_range:
        labels["depth_range"] = list(rec.depth_range)
    if rec.depth_meta:
        labels["depth"] = rec.depth_meta
    _add(tar, f"{key}.json", json.dumps(labels).encode())
    if rec.frame_path and rec.frame_path.exists():
        _add(tar, f"{key}.jpg", rec.frame_path.read_bytes())
    if rec.has_depth and rec.depth_path and rec.depth_path.exists():
        _add(tar, f"{key}.depth.png", rec.depth_path.read_bytes())


def write_webdataset(
    root: str | Path,
    out_dir: str | Path,
    filters: Optional[Filters] = None,
    shard_size: int = 512,
    session_dir: Optional[str | Path] = None,
) -> int:
    """Write filtered records to ``out_dir/shard-NNNNN.tar``. Returns count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = iter_session(session_dir) if session_dir else iter_dataset(root)

    n = 0
    shard = 0
    tar: Optional[tarfile.TarFile] = None
    for rec in src:
        if filters and not filters.passes(rec):
            continue
        if tar is None or n % shard_size == 0:
            if tar is not None:
                tar.close()
            tar = tarfile.open(out_dir / f"shard-{shard:05d}.tar", "w")
            shard += 1
        # Globally unique, sortable key.
        _write_sample(tar, f"{rec.session_id}_{rec.seq:06d}", rec)
        n += 1
    if tar is not None:
        tar.close()
    return n


def _main() -> int:
    ap = argparse.ArgumentParser(description="Export MVM sessions to WebDataset")
    ap.add_argument("root")
    ap.add_argument("out_dir")
    ap.add_argument("--config")
    ap.add_argument("--shard-size", type=int, default=512)
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    is_session = (root / "labels.jsonl").exists()
    filters = Filters.from_yaml(args.config) if args.config else None
    n = write_webdataset(
        root=None if is_session else root,
        session_dir=root if is_session else None,
        out_dir=args.out_dir,
        filters=filters,
        shard_size=args.shard_size,
    )
    print(f"wrote {n} sample(s) → {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
