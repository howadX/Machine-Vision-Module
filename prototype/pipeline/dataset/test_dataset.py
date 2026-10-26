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
    # Writable → torch.as_tensor won't warn / forbid writes (caught in live e2e).
    assert frame.flags.writeable
    depth = r0.load_depth()
    assert depth is not None and depth.shape == (48, 64)
    # de-normalized to [min,max] = [0.5, 9.0]
    assert abs(depth.min() - 0.5) < 1e-2 and abs(depth.max() - 9.0) < 1e-2
    # frame with no depth → None
    r1 = list(iter_session(sdir))[1]
    assert r1.load_depth() is None


def test_metric_depth_loader_preserves_exact_u16_mm():
    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td)
        (sdir / "depth").mkdir()
        o = _obs(0, 100, with_depth=False)
        o["schema_version"] = "16.8.0"
        o["frame"]["depth_path"] = "depth/000000.png"
        o["frame"]["depth"] = {
            "encoding": "metric_mm_u16", "units": "millimetres",
            "width": 3, "height": 2, "invalid_value": 0,
        }
        expected = np.array([[0, 500, 1000], [1500, 2047, 65535]], dtype=np.uint16)
        Image.fromarray(expected).save(sdir / "depth/000000.png")
        rec = record_from_obs(o, sdir)
        actual = rec.load_depth()
        assert actual.dtype == np.uint16
        assert np.array_equal(actual, expected)


def test_validate_metric_depth_decodes_uint16_without_crashing():
    from validate import validate_session

    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td)
        (sdir / "frames").mkdir()
        (sdir / "depth").mkdir()
        Image.new("RGB", (2, 2)).save(sdir / "frames/000000.jpg")
        depth = np.array([[0, 500], [1000, 2000]], dtype=np.uint16)
        Image.fromarray(depth).save(sdir / "depth/000000.png")
        obs = _obs(0, 1)
        obs["schema_version"] = "16.8.0"
        obs["frame"].pop("depth_range")
        obs["frame"]["depth"] = {
            "encoding": "metric_mm_u16",
            "units": "millimetres",
            "width": 2,
            "height": 2,
            "invalid_value": 0,
            "valid_ratio": 0.75,
        }
        (sdir / "labels.jsonl").write_text(json.dumps(obs) + "\n")

        report = validate_session(sdir)
        assert report.errors == 0, report.issues


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

    # Picklable → survives DataLoader workers under forkserver (Python 3.14+).
    # (A local IterableDataset subclass would fail here.)
    import pickle

    reloaded = pickle.loads(pickle.dumps(ds))
    assert len(list(iter(reloaded))) == 2

    # Real DataLoader (single-process is enough to exercise the integration).
    import torch

    from torch.utils.data import IterableDataset

    assert isinstance(ds, IterableDataset), "must subclass IterableDataset when torch present"
    batches = list(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0))
    assert len(batches) == 2


def test_validate_clean(sdir: Path):
    from validate import validate_dataset

    # `sdir` is a session dir (has labels.jsonl) → validated as a single session.
    reps = validate_dataset(sdir)
    assert len(reps) == 1
    r = reps[0]
    assert r.errors == 0, [i.__dict__ for i in r.issues]
    st = r.stats
    assert st.frames == 3 and st.people == 3
    assert st.gallery == 2 and st.anon == 1
    assert st.withheld_posture == 1 and abs(st.withheld_rate - 1 / 3) < 1e-6
    assert st.masks == 1 and st.with_depth == 2
    assert st.posture_hist.get("sitting") == 1 and st.posture_hist.get("standing") == 1


def test_validate_catches_missing_frame():
    from validate import validate_session

    with tempfile.TemporaryDirectory() as td:
        sdir = build_session(Path(td))
        (sdir / "frames" / "000000.jpg").unlink()  # break a referenced frame
        r = validate_session(sdir)
        assert r.errors >= 1
        assert any(i.code == "frame-missing" for i in r.issues)


def test_validate_catches_bad_mask():
    from validate import validate_session

    with tempfile.TemporaryDirectory() as td:
        sdir = Path(td) / "sessions" / "2026-07-17_150000"
        (sdir / "frames").mkdir(parents=True)
        o = _obs(0, 100, identity="Вадим", with_depth=False)
        # An RLE that doesn't cover MASK_RES² cells → decode_rle raises → finding.
        o["entities"][1]["attributes"]["mask"] = {"res": MASK_RES, "coverage": 0.0, "rle": [10, 20]}
        (sdir / "labels.jsonl").write_text(json.dumps(o) + "\n", encoding="utf-8")
        Image.fromarray(np.zeros((48, 64, 3), np.uint8), "RGB").save(sdir / "frames" / "000000.jpg")
        r = validate_session(sdir)
        assert any(i.code == "mask-decode" for i in r.issues), [i.__dict__ for i in r.issues]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sdir = build_session(tmp)
        test_rle_round_trip()
        test_parse_records(sdir)
        test_pixel_loaders(sdir)
        test_metric_depth_loader_preserves_exact_u16_mm()
        test_validate_metric_depth_decodes_uint16_without_crashing()
        test_filters(sdir)
        test_filters_from_yaml()
        test_pytorch_dataset(sdir)
        test_validate_clean(sdir)
    test_validate_catches_missing_frame()
    test_validate_catches_bad_mask()
    print("all dataset exporter tests passed ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
