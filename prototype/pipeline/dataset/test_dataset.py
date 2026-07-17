"""Tests for the MVM dataset exporters (phase 16.4).

Builds a synthetic session in a temp dir — matching the Rust recorder's on-disk
schema — and checks parsing, RLE round-trip, filters, and the PyTorch dataset.
Run: ``python test_dataset.py`` (needs numpy, PIL, yaml, torch).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import records
from filters import Filters
from records import MASK_RES, decode_rle, iter_session, record_from_obs


def encode_rle(grid: np.ndarray) -> list[int]:
    """Inverse of records.decode_rle — alternating runs starting with 0."""
    flat = (grid.reshape(-1) != 0).astype(np.uint8)
    rle, cur, run = [], 0, 0
    for v in flat:
        if v == cur:
            run += 1
        else:
            rle.append(run)
            cur = int(v)
            run = 1
    rle.append(run)
    return rle


def _obs(seq, frame_id, *, identity=None, posture=("sitting", 0.9),
         posture_status="observed", entropy=0.1, with_depth=True, mask_grid=None,
         activities=()):
    attrs = {"posture_status": posture_status, "emotion_entropy": entropy}
    if identity:
        attrs["identity"] = {"label": identity, "confidence": 0.92}
    if posture:
        attrs["posture"] = {"label": posture[0], "confidence": posture[1]}
    people = [{"entity_id": "p1", "track_id": 1, "group": "human",
               "kind": "person", "attributes": attrs}]
    objects = []
    obj_attrs = {"class": "cup", "confidence": 0.7, "age": 3, "remembered": False}
    if mask_grid is not None:
        obj_attrs["mask"] = {"res": MASK_RES,
                             "coverage": float((mask_grid != 0).mean()),
                             "rle": encode_rle(mask_grid)}
    objects.append({"entity_id": "o10000", "track_id": 10000, "group": "scene",
                    "kind": "object", "attributes": obj_attrs})
    frame = {"timestamp": f"2026-07-17T15:00:{seq:02d}.000",
             "frame_path": f"frames/{seq:06d}.jpg"}
    if with_depth:
        frame["depth_path"] = f"depth/{seq:06d}.png"
        frame["depth_range"] = [0.5, 9.0]
    return {
        "dataset_id": "mvm_sessions", "schema_version": "16.0.1",
        "session_id": "2026-07-17_150000", "frame_id": frame_id,
        "frame": frame, "session": {"seq_in_session": seq},
        "entities": people + objects, "relations": [], "events": [],
        "semantic_groups": ["human", "scene", "quality"],
        "derived_targets": None,
        "quality": {"frame_ok": True, "people_count": 1, "object_count": 1,
                    "anomaly_score": 0.1},
        "provenance": [{"model": "mvm-dataset:router", "version": "16.0.1",
                        "source": "router"}],
        "activities": list(activities),
    }


def build_session(tmp: Path) -> Path:
    sdir = tmp / "sessions" / "2026-07-17_150000"
    (sdir / "frames").mkdir(parents=True)
    (sdir / "depth").mkdir(parents=True)

    grid = np.zeros((MASK_RES, MASK_RES), dtype=np.uint8)
    grid[:8, :8] = 1  # top-left block

    obs = [
        _obs(0, 100, identity="Вадим", posture=("sitting", 0.95),
             posture_status="observed", entropy=0.1, mask_grid=grid,
             activities=["using_object(cup)"]),
        _obs(1, 130, identity=None, posture=None, posture_status="withheld",
             entropy=0.8, with_depth=False),  # anon, occluded, near-chance, no depth
        _obs(2, 160, identity="Вадим", posture=("standing", 0.7),
             posture_status="observed", entropy=0.2),
    ]
    with open(sdir / "labels.jsonl", "w", encoding="utf-8") as fh:
        for o in obs:
            fh.write(json.dumps(o) + "\n")

    # A gradient RGB frame + a 16-bit depth PNG for the depth-carrying frames.
    for seq in (0, 1, 2):
        arr = np.zeros((48, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
        Image.fromarray(arr, "RGB").save(sdir / "frames" / f"{seq:06d}.jpg")
    for seq in (0, 2):
        d = np.linspace(0, 65535, 48 * 64, dtype=np.uint16).reshape(48, 64)
        Image.fromarray(d).save(sdir / "depth" / f"{seq:06d}.png")  # uint16 → I;16
    return sdir


def test_rle_round_trip():
    grid = np.zeros((MASK_RES, MASK_RES), dtype=np.uint8)
    grid[:8, :8] = 1
    assert np.array_equal(decode_rle(encode_rle(grid)), grid)
    # Grid opening on a set cell → first run is 0.
    g2 = np.ones((MASK_RES, MASK_RES), dtype=np.uint8)
    assert encode_rle(g2)[0] == 0
    assert np.array_equal(decode_rle(encode_rle(g2)), g2)


def test_parse_records(sdir: Path):
    recs = list(iter_session(sdir))
    assert len(recs) == 3
    r0 = recs[0]
    assert r0.frame_id == 100 and r0.seq == 0
    p = r0.people[0]
    assert p.identity == "Вадим" and not p.is_anonymous
    assert p.posture == "sitting" and p.posture_status == "observed"
    assert abs(p.emotion_entropy - 0.1) < 1e-6
    # object + mask
    o = r0.objects[0]
    assert o.cls == "cup" and o.has_mask
    m = o.mask()
    assert m.shape == (MASK_RES, MASK_RES) and m[:8, :8].all() and m.sum() == 64
    # withheld anon frame
    assert recs[1].people[0].is_anonymous
    assert recs[1].people[0].posture is None
    assert recs[1].people[0].posture_status == "withheld"
    assert not recs[1].has_depth


def test_pixel_loaders(sdir: Path):
    r0 = next(iter_session(sdir))
    frame = r0.load_frame()
    assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
    depth = r0.load_depth()
    assert depth is not None and depth.shape == (48, 64)
    # de-normalized to [min,max] = [0.5, 9.0]
    assert abs(depth.min() - 0.5) < 1e-2 and abs(depth.max() - 9.0) < 1e-2
    # frame with no depth → None
    r1 = list(iter_session(sdir))[1]
    assert r1.load_depth() is None


def test_filters(sdir: Path):
    recs = list(iter_session(sdir))

    # postures: only frames with a sitting/standing person; withheld frame drops.
    f = Filters(postures=["sitting", "standing"])
    kept = [r for r in recs if f.passes(r)]
    assert [r.seq for r in kept] == [0, 2], "withheld-posture frame is excluded"

    # require_identity drops the anon frame.
    assert [r.seq for r in recs if Filters(require_identity=True).passes(r)] == [0, 2]

    # require_depth drops the depthless frame.
    assert [r.seq for r in recs if Filters(require_depth=True).passes(r)] == [0, 2]

    # emotion_entropy_max=0.5 drops the near-chance (0.8) frame.
    assert [r.seq for r in recs if Filters(emotion_entropy_max=0.5).passes(r)] == [0, 2]

    # activities filter.
    got = [r.seq for r in recs if Filters(activities=["using_object(cup)"]).passes(r)]
    assert got == [0]

    # no filters → everything passes.
    assert all(Filters().passes(r) for r in recs)


def test_filters_from_yaml():
    cfg = Path(__file__).parent / "dataset_config.yaml"
    f = Filters.from_yaml(cfg)
    assert f.emotion_entropy_max == 1.0 and f.require_depth is False


def test_pytorch_dataset(sdir: Path):
    from export_pytorch import MVMSessionDataset

    ds = MVMSessionDataset(session_dir=sdir, filters=Filters(require_depth=True))
    samples = list(iter(ds))
    assert len(samples) == 2, "require_depth keeps 2 of 3"
    s = samples[0]
    assert s["frame"].shape == (48, 64, 3)
    assert "depth" in s and s["depth"].shape == (48, 64)
    assert s["labels"]["people"][0]["identity"] == "Вадим"
    assert s["labels"]["objects"][0]["has_mask"] is True

    # labels-only mode skips pixels.
    ds2 = MVMSessionDataset(session_dir=sdir, with_frame=False, with_depth=False)
    s2 = next(iter(ds2))
    assert "frame" not in s2 and "depth" not in s2 and "labels" in s2


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sdir = build_session(tmp)
        test_rle_round_trip()
        test_parse_records(sdir)
        test_pixel_loaders(sdir)
        test_filters(sdir)
        test_filters_from_yaml()
        test_pytorch_dataset(sdir)
    print("all dataset exporter tests passed ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
