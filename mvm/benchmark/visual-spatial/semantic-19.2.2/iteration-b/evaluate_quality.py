#!/usr/bin/env python3
"""Fail-closed evaluator for the measured ADE20K quality rejection."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


REJECT = "reject_candidate_defer_runtime_expert"
SURFACES = {"wall", "floor", "ceiling", "support", "opening"}


def evaluate(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("quality_measurement_version") != "19.2.2-b.2":
        raise ValueError("quality measurement version drift")
    dataset = value.get("dataset", {})
    if dataset.get("archive_sha256") != "7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29":
        raise ValueError("ADE20K archive hash drift")
    if dataset.get("measured_frames") != 200 or dataset.get("selection") != "lexicographic_every_tenth_frame":
        raise ValueError("quality subset drift")
    baseline = value.get("selected_baseline", {})
    iou = baseline.get("per_class_iou", {})
    if set(iou) != SURFACES or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in iou.values()):
        raise ValueError("invalid surface IoU record")
    failed = (
        baseline.get("mean_surface_iou", 0.0) < 0.60
        or iou["floor"] < 0.75
        or iou["opening"] < 0.55
        or baseline.get("unknown_abstention_recall", 0.0) < 0.90
        or baseline.get("unknown_discipline_violations") != 0
        or baseline.get("false_floor_on_unknown") != 0
    )
    if not failed or baseline.get("passed") is not False:
        raise ValueError("quality-failing baseline cannot pass")
    fine_tunes = value.get("conditional_fine_tunes")
    if not isinstance(fine_tunes, list) or len(fine_tunes) != 2:
        raise ValueError("both bounded fine-tune records are required")
    if any(run.get("passed") is not False for run in fine_tunes):
        raise ValueError("failed fine-tune cannot be selected")
    if any(run.get("mean_surface_iou", 1.0) >= baseline["mean_surface_iou"] for run in fine_tunes):
        raise ValueError("recorded fine-tune unexpectedly improves baseline")
    if value.get("decision") != REJECT or value.get("model_or_dataset_bytes_committed") is not False:
        raise ValueError("quality failure must reject without committing artifacts")
    return {
        "evaluator_version": "19.2.2-b.2",
        "decision": REJECT,
        "measured_frames": 200,
        "mean_surface_iou": baseline["mean_surface_iou"],
        "floor_iou": iou["floor"],
        "opening_iou": iou["opening"],
        "unknown_abstention_recall": baseline["unknown_abstention_recall"],
        "unknown_discipline_violations": baseline["unknown_discipline_violations"],
        "fine_tunes_rejected": len(fine_tunes),
        "runtime_semantic_expert": "not_implemented_in_19.2.2",
    }


def main() -> int:
    try:
        result = evaluate(Path(__file__).with_name("quality_measurement.json"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"semantic quality evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
