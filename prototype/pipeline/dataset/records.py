"""Parse MVM Dataset Factory sessions into typed Python records (phase 16.4).

The Rust recorder (`mvm run --record`) writes each session as a self-describing
folder::

    sessions/<id>/
      frames/NNNNNN.jpg      # RGB, privacy-blurred per [dataset].privacy
      depth/NNNNNN.png       # 16-bit relative depth (optional)
      labels.jsonl           # one CanonicalObservation per saved frame
      session.toml           # closed-session summary

This module turns each ``labels.jsonl`` line (a CanonicalObservation, schema
16.x) into typed records the exporters consume. It is the Python side of the
"interpretable encoding" contract: the raw JSON attributes become typed fields,
label *status* is preserved (``posture_status`` etc.), and ``withheld``/``unknown``
are never silently coerced into a negative class.

Pure parsing + numpy; images are loaded lazily (PIL) only when a sample is
materialized, so iterating labels is cheap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# Instance-mask grid side (mirrors mvm_core::mask::MASK_RES).
MASK_RES = 64


def decode_rle(rle: list[int], res: int = MASK_RES) -> np.ndarray:
    """Decode a run-length mask back to a ``res × res`` uint8 grid.

    The RLE is alternating run lengths over the row-major grid, **starting with a
    run of 0s** (first run is 0 if the grid opens on a set cell) — the exact
    encoding the Rust router emits, so this round-trips it bit-for-bit.
    """
    flat = np.empty(res * res, dtype=np.uint8)
    pos = 0
    sym = 0
    for run in rle:
        flat[pos : pos + run] = sym
        pos += run
        sym ^= 1
    if pos != res * res:
        raise ValueError(f"RLE covers {pos} cells, expected {res * res}")
    return flat.reshape(res, res)


@dataclass
class Person:
    entity_id: str
    track_id: Optional[int]
    identity: Optional[str]
    identity_conf: Optional[float]
    posture: Optional[str]
    posture_conf: Optional[float]
    posture_status: str  # observed | withheld | unknown
    emotion: Optional[str]
    emotion_conf: Optional[float]
    emotion_entropy: Optional[float]
    bbox: Optional[list]

    @property
    def is_anonymous(self) -> bool:
        return self.identity is None


@dataclass
class Obj:
    entity_id: str
    cls: str
    confidence: float
    mask_rle: Optional[list]

    @property
    def has_mask(self) -> bool:
        return self.mask_rle is not None

    def mask(self) -> Optional[np.ndarray]:
        """Decode this object's instance mask, or ``None`` if it has none."""
        if self.mask_rle is None:
            return None
        return decode_rle(self.mask_rle)


@dataclass
class Record:
    """One saved frame: pixels (lazy) + its full structured labels."""

    session_id: str
    frame_id: int
    seq: int
    timestamp: str
    session_dir: Path
    frame_rel: Optional[str]
    depth_rel: Optional[str]
    depth_range: Optional[tuple[float, float]]
    people: list[Person]
    objects: list[Obj]
    activities: list[str]
    semantic_groups: list[str]
    quality: dict

    # --- paths ---
    @property
    def frame_path(self) -> Optional[Path]:
        return self.session_dir / self.frame_rel if self.frame_rel else None

    @property
    def depth_path(self) -> Optional[Path]:
        return self.session_dir / self.depth_rel if self.depth_rel else None

    @property
    def has_depth(self) -> bool:
        return self.depth_rel is not None

    # --- lazy pixel loaders ---
    def load_frame(self) -> np.ndarray:
        """RGB frame as a ``H×W×3`` uint8 array (PIL loaded on demand)."""
        from PIL import Image

        with Image.open(self.frame_path) as im:
            return np.asarray(im.convert("RGB"))

    def load_depth(self) -> Optional[np.ndarray]:
        """Relative depth as a ``H×W`` float32 array, de-normalized via
        ``depth_range`` so values match the recorder's raw affine-relative scale
        (larger = nearer). ``None`` when the frame has no depth."""
        if not self.has_depth:
            return None
        from PIL import Image

        with Image.open(self.depth_path) as im:
            u16 = np.asarray(im, dtype=np.float32)  # 0..65535
        lo, hi = self.depth_range if self.depth_range else (0.0, 1.0)
        return lo + (u16 / 65535.0) * (hi - lo)


def _person_from_entity(e: dict) -> Person:
    a = e.get("attributes", {})
    ident = a.get("identity")
    posture = a.get("posture")
    emotion = a.get("emotion")
    return Person(
        entity_id=e["entity_id"],
        track_id=e.get("track_id"),
        identity=ident["label"] if ident else None,
        identity_conf=ident.get("confidence") if ident else None,
        posture=posture["label"] if posture else None,
        posture_conf=posture.get("confidence") if posture else None,
        posture_status=a.get("posture_status", "unknown"),
        emotion=emotion["label"] if emotion else None,
        emotion_conf=emotion.get("confidence") if emotion else None,
        emotion_entropy=a.get("emotion_entropy"),
        bbox=a.get("bbox"),
    )


def _obj_from_entity(e: dict) -> Obj:
    a = e.get("attributes", {})
    mask = a.get("mask")
    return Obj(
        entity_id=e["entity_id"],
        cls=a.get("class", "unknown"),
        confidence=a.get("confidence", 0.0),
        mask_rle=mask["rle"] if mask else None,
    )


def record_from_obs(obs: dict, session_dir: Path) -> Record:
    """Build a :class:`Record` from one parsed CanonicalObservation dict."""
    entities = obs.get("entities", [])
    people = [_person_from_entity(e) for e in entities if e.get("kind") == "person"]
    objects = [_obj_from_entity(e) for e in entities if e.get("kind") == "object"]
    frame = obs.get("frame", {})
    dr = frame.get("depth_range")
    return Record(
        session_id=obs.get("session_id", session_dir.name),
        frame_id=obs.get("frame_id", 0),
        seq=obs.get("session", {}).get("seq_in_session", 0),
        timestamp=frame.get("timestamp", ""),
        session_dir=session_dir,
        frame_rel=frame.get("frame_path"),
        depth_rel=frame.get("depth_path"),
        depth_range=tuple(dr) if dr else None,
        people=people,
        objects=objects,
        activities=_activities(obs),
        semantic_groups=obs.get("semantic_groups", []),
        quality=obs.get("quality", {}),
    )


def _activities(obs: dict) -> list[str]:
    """Activity labels for a frame. The observation's ``activities`` section is
    reserved for §16.3 typed events; today activity labels live on the session
    summary, so per-frame we surface any that the router attached (forward-compat:
    accept both a list of strings and a list of {label} objects)."""
    acts = obs.get("activities", [])
    out = []
    for a in acts:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict) and "label" in a:
            out.append(a["label"])
    return out


def iter_session(session_dir: str | Path) -> Iterator[Record]:
    """Yield every saved frame of one session as a :class:`Record`."""
    session_dir = Path(session_dir)
    labels = session_dir / "labels.jsonl"
    with open(labels, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield record_from_obs(json.loads(line), session_dir)


def iter_dataset(
    root: str | Path, session_ids: Optional[list[str]] = None
) -> Iterator[Record]:
    """Yield records across sessions under ``<root>/sessions`` (chronological).

    ``session_ids`` restricts to a subset; otherwise every session is streamed.
    """
    sessions_dir = Path(root) / "sessions"
    if not sessions_dir.is_dir():
        return
    ids = session_ids or sorted(p.name for p in sessions_dir.iterdir() if p.is_dir())
    for sid in ids:
        sdir = sessions_dir / sid
        if (sdir / "labels.jsonl").exists():
            yield from iter_session(sdir)
