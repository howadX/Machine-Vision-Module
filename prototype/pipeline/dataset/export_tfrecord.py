"""Export MVM Dataset Factory sessions to TFRecord shards (phase 16.4).

Each ``tf.train.Example`` carries the JPEG bytes, optional depth PNG bytes, and
the structured labels as a JSON string feature (kept whole so no signal — status,
confidence, entropy — is lost in a flat schema). Requires ``tensorflow``.

    python export_tfrecord.py <root> <out_dir> [--config dataset_config.yaml] \
        [--shard-size 512]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from filters import Filters
from export_pytorch import record_to_labels
from records import Record, iter_dataset, iter_session


def _example(rec: Record):
    import tensorflow as tf

    def _bytes(v: bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _int(v: int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

    feats = {
        "frame_id": _int(rec.frame_id),
        "session_id": _bytes(rec.session_id.encode()),
        "labels_json": _bytes(json.dumps(record_to_labels(rec)).encode()),
    }
    if rec.frame_path and rec.frame_path.exists():
        feats["image"] = _bytes(rec.frame_path.read_bytes())
    if rec.has_depth and rec.depth_path and rec.depth_path.exists():
        feats["depth_png"] = _bytes(rec.depth_path.read_bytes())
        if rec.depth_meta:
            feats["depth_metadata_json"] = _bytes(json.dumps(rec.depth_meta, sort_keys=True).encode())
        if rec.depth_range:
            lo, hi = rec.depth_range
            feats["depth_min"] = tf.train.Feature(float_list=tf.train.FloatList(value=[lo]))
            feats["depth_max"] = tf.train.Feature(float_list=tf.train.FloatList(value=[hi]))
    return tf.train.Example(features=tf.train.Features(feature=feats))


def write_tfrecords(
    root: str | Path,
    out_dir: str | Path,
    filters: Optional[Filters] = None,
    shard_size: int = 512,
    session_dir: Optional[str | Path] = None,
) -> int:
    """Write filtered records to ``out_dir/shard-NNNNN.tfrecord``. Returns count."""
    import tensorflow as tf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = iter_session(session_dir) if session_dir else iter_dataset(root)

    n = 0
    shard = 0
    writer = None
    for rec in src:
        if filters and not filters.passes(rec):
            continue
        if writer is None or n % shard_size == 0:
            if writer is not None:
                writer.close()
            path = out_dir / f"shard-{shard:05d}.tfrecord"
            writer = tf.io.TFRecordWriter(str(path))
            shard += 1
        writer.write(_example(rec).SerializeToString())
        n += 1
    if writer is not None:
        writer.close()
    return n


def _main() -> int:
    ap = argparse.ArgumentParser(description="Export MVM sessions to TFRecord")
    ap.add_argument("root")
    ap.add_argument("out_dir")
    ap.add_argument("--config")
    ap.add_argument("--shard-size", type=int, default=512)
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    is_session = (root / "labels.jsonl").exists()
    filters = Filters.from_yaml(args.config) if args.config else None
    n = write_tfrecords(
        root=None if is_session else root,
        session_dir=root if is_session else None,
        out_dir=args.out_dir,
        filters=filters,
        shard_size=args.shard_size,
    )
    print(f"wrote {n} example(s) → {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
