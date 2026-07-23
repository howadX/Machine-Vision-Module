"""Data-quality gate — pixel side (phase 16.6-A).

Companion to the Rust ``mvm dataset check``. That gate is authoritative for
*structure*: it parses ``labels.jsonl``, checks binding keys, referenced-file
existence, mask-RLE cell counts, depth ranges, and privacy (no raw embeddings in
labels) — all without an image crate (invariant #2). This module does the part
Rust deliberately can't: it **actually decodes the pixels** the exporter is about
to feed a trainer —

  * every ``frame_path`` opens as an H×W×3 image,
  * every ``depth_path`` opens and de-normalizes to a 2-D float map,
  * every object mask RLE decodes back to a ``MASK_RES²`` grid,

and it folds the same quality-as-data stats (identity split, posture balance,
withheld-rate) so the consumer sees, at export time, what they're training on.
A ``withheld`` posture is honest absence (§16.0.2) — counted, never an error.

Run it before an export as a CI gate::

    python validate.py ~/.local/share/mvm/datasets            # whole dataset
    python validate.py <session_dir> --strict                 # one session, warns fail too

Exit code is non-zero when the gate fails (any error; with ``--strict``, any
warning too), so it drops straight into CI. Reuses ``records.py`` — needs
numpy + PIL, no torch/tf.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from records import MASK_RES, Record, decode_rle, iter_session


@dataclass
class Issue:
    severity: str  # "ERROR" | "WARN"
    code: str
    seq: Optional[int]
    msg: str


@dataclass
class Stats:
    frames: int = 0
    people: int = 0
    gallery: int = 0
    anon: int = 0
    withheld_posture: int = 0
    with_depth: int = 0
    objects: int = 0
    masks: int = 0
    posture_hist: dict = field(default_factory=dict)
    emotion_hist: dict = field(default_factory=dict)
    object_hist: dict = field(default_factory=dict)

    @property
    def withheld_rate(self) -> float:
        return self.withheld_posture / self.people if self.people else 0.0


@dataclass
class SessionReport:
    session_id: str
    issues: list[Issue] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)

    @property
    def errors(self) -> int:
        return sum(i.severity == "ERROR" for i in self.issues)

    @property
    def warns(self) -> int:
        return sum(i.severity == "WARN" for i in self.issues)


def _bump(hist: dict, key: Optional[str]) -> None:
    if key:
        hist[key] = hist.get(key, 0) + 1


def _check_frame_pixels(rec: Record, rep: SessionReport) -> None:
    """Frame must open and look like an H×W×3 image."""
    if rec.frame_rel is None:
        return
    if not rec.frame_path.exists():
        rep.issues.append(Issue("ERROR", "frame-missing", rec.seq, f"{rec.frame_rel} not on disk"))
        return
    try:
        arr = rec.load_frame()
    except Exception as e:  # noqa: BLE001 — surface any decode failure as a finding
        rep.issues.append(Issue("ERROR", "frame-unreadable", rec.seq, f"{rec.frame_rel}: {e}"))
        return
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        rep.issues.append(
            Issue("ERROR", "frame-shape", rec.seq, f"{rec.frame_rel}: expected H×W×3, got {arr.shape}")
        )


def _check_depth_pixels(rec: Record, rep: SessionReport) -> None:
    """Depth (when present) must carry a reversible range and open as a 2-D map."""
    if not rec.has_depth:
        return
    rep.stats.with_depth += 1
    if rec.depth_range is None:
        rep.issues.append(Issue("ERROR", "depth-range-missing", rec.seq, "depth_path set but no depth_range"))
    else:
        lo, hi = rec.depth_range
        if not (lo <= hi):
            rep.issues.append(Issue("ERROR", "depth-range-bad", rec.seq, f"depth_range [{lo}, {hi}] not [min ≤ max]"))
    if not rec.depth_path.exists():
        rep.issues.append(Issue("ERROR", "depth-missing", rec.seq, f"{rec.depth_rel} not on disk"))
        return
    try:
        d = rec.load_depth()
    except Exception as e:  # noqa: BLE001
        rep.issues.append(Issue("ERROR", "depth-unreadable", rec.seq, f"{rec.depth_rel}: {e}"))
        return
    if d is None or d.ndim != 2 or d.size == 0:
        rep.issues.append(
            Issue("ERROR", "depth-shape", rec.seq, f"{rec.depth_rel}: expected 2-D map, got {None if d is None else d.shape}")
        )


def _check_person(rec: Record, rep: SessionReport) -> None:
    for p in rec.people:
        rep.stats.people += 1
        if p.is_anonymous:
            rep.stats.anon += 1
        else:
            rep.stats.gallery += 1
        _bump(rep.stats.posture_hist, p.posture)
        _bump(rep.stats.emotion_hist, p.emotion)
        # Honest absence, never a class and never an error (§16.0.2).
        if p.posture_status == "withheld":
            rep.stats.withheld_posture += 1


def _check_objects(rec: Record, rep: SessionReport) -> None:
    for o in rec.objects:
        rep.stats.objects += 1
        _bump(rep.stats.object_hist, o.cls)
        if not o.has_mask:
            continue
        rep.stats.masks += 1
        try:
            grid = decode_rle(o.mask_rle)  # raises if it doesn't cover MASK_RES² cells
        except Exception as e:  # noqa: BLE001
            rep.issues.append(Issue("ERROR", "mask-decode", rec.seq, f"object {o.entity_id}: {e}"))
            continue
        if grid.shape != (MASK_RES, MASK_RES):
            rep.issues.append(
                Issue("ERROR", "mask-shape", rec.seq, f"object {o.entity_id}: mask {grid.shape} ≠ ({MASK_RES},{MASK_RES})")
            )


def validate_session(session_dir: str | Path) -> SessionReport:
    """Decode-validate one recorded session; never raises on bad data — findings
    are collected into the report so the whole dataset can still be gated."""
    session_dir = Path(session_dir)
    rep = SessionReport(session_id=session_dir.name)
    if not (session_dir / "labels.jsonl").exists():
        rep.issues.append(Issue("WARN", "no-labels", None, f"no labels.jsonl under {session_dir}"))
        return rep
    for rec in iter_session(session_dir):
        rep.stats.frames += 1
        _check_frame_pixels(rec, rep)
        _check_depth_pixels(rec, rep)
        _check_person(rec, rep)
        _check_objects(rec, rep)
    return rep


def validate_dataset(root: str | Path) -> list[SessionReport]:
    """Validate every ``<root>/sessions/<id>`` — or, if ``root`` is itself a
    session dir (has labels.jsonl), just that one."""
    root = Path(root)
    if (root / "labels.jsonl").exists():
        return [validate_session(root)]
    sessions_dir = root / "sessions"
    if not sessions_dir.is_dir():
        return []
    ids = sorted(p.name for p in sessions_dir.iterdir() if p.is_dir())
    return [validate_session(sessions_dir / sid) for sid in ids]


def _print_report(rep: SessionReport) -> None:
    st = rep.stats
    print(f"── {rep.session_id} ──")
    for i in rep.issues:
        tag = "E" if i.severity == "ERROR" else "W"
        where = f" frame {i.seq}" if i.seq is not None else ""
        print(f"  [{tag}] {i.code:<20}{where}: {i.msg}")
    print(
        f"  stats: {st.frames} frames · {st.people} people "
        f"({st.gallery} gallery / {st.anon} anon) · {st.objects} objects · "
        f"{st.masks} masks · {st.with_depth} depth"
    )
    print(
        f"         withheld-posture {st.withheld_rate * 100:.0f}% · "
        f"postures {_fmt_hist(st.posture_hist)} · emotions {_fmt_hist(st.emotion_hist)} · "
        f"objects {_fmt_hist(st.object_hist)}"
    )


def _fmt_hist(h: dict) -> str:
    if not h:
        return "—"
    items = sorted(h.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
    return " ".join(f"{k}×{c}" for k, c in items)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MVM dataset pixel-level quality gate (16.6-A)")
    ap.add_argument("root", help="dataset root (…/datasets) or a single session dir")
    ap.add_argument("--strict", action="store_true", help="fail the gate on warnings too")
    args = ap.parse_args(argv)

    reports = validate_dataset(args.root)
    if not reports:
        print(f"No sessions under {args.root} — nothing to validate.")
        return 0
    for rep in reports:
        _print_report(rep)

    errs = sum(r.errors for r in reports)
    warns = sum(r.warns for r in reports)
    print(f"\n{len(reports)} session(s) · {errs} error(s) · {warns} warning(s)")
    if errs or (args.strict and warns):
        why = f"{errs} error(s)" + (f" + {warns} warning(s)" if args.strict and warns else "")
        print(f"FAILED — data-quality gate: {why}")
        return 1
    print("OK — pixel-level validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
