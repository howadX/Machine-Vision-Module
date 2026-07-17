"""Frame filters for MVM dataset export (phase 16.4).

A :class:`Filters` decides which frames enter a training view — the mechanism the
plan's ``dataset_config.yaml`` drives. Filters are *quality-aware*: they read the
per-signal confidence / entropy / status the recorder preserved, and they never
turn a ``withheld`` posture into a usable "negative" — a withheld posture simply
does not satisfy a ``postures`` requirement.

Pure predicate over :class:`records.Record`; no torch/PIL import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from records import Person, Record


@dataclass
class Filters:
    # Person-level (a frame passes if ≥1 person satisfies the active person filters).
    postures: Optional[list[str]] = None  # None/empty = any posture
    min_posture_confidence: float = 0.0
    min_emotion_confidence: float = 0.0
    emotion_entropy_max: float = 1.0  # 1.0 = keep all (drop near-chance below this)
    require_identity: bool = False  # keep only frames with ≥1 gallery-identified person
    # Frame-level.
    activities: Optional[list[str]] = None  # keep frames tagged with ≥1 of these
    require_depth: bool = False
    min_people: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Filters":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        f = doc.get("filters", {}) or {}
        known = {fld.name for fld in field_list()}
        unknown = set(f) - known
        if unknown:
            raise ValueError(f"unknown filter keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in f.items() if k in known})

    # --- predicate ---
    def _person_active(self) -> bool:
        """Whether any person-level constraint is set (else people don't gate)."""
        return bool(
            self.postures
            or self.require_identity
            or self.min_emotion_confidence > 0.0
            or self.emotion_entropy_max < 1.0
            or self.min_posture_confidence > 0.0
        )

    def _person_ok(self, p: Person) -> bool:
        if self.postures:
            # A withheld/unknown posture is honestly absent — it cannot match a
            # required posture (§16.0.2: absence is not a class).
            if p.posture is None or p.posture not in self.postures:
                return False
        if p.posture_conf is not None and p.posture_conf < self.min_posture_confidence:
            return False
        if self.require_identity and p.identity is None:
            return False
        if p.emotion_conf is not None and p.emotion_conf < self.min_emotion_confidence:
            return False
        if p.emotion_entropy is not None and p.emotion_entropy > self.emotion_entropy_max:
            return False
        return True

    def passes(self, rec: Record) -> bool:
        if self.require_depth and not rec.has_depth:
            return False
        if len(rec.people) < self.min_people:
            return False
        if self.activities:
            if not any(a in self.activities for a in rec.activities):
                return False
        if self._person_active():
            if not any(self._person_ok(p) for p in rec.people):
                return False
        return True


def field_list():
    return list(Filters.__dataclass_fields__.values())
