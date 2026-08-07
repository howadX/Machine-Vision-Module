#!/usr/bin/env python3
"""Fail-closed Phase 19.1 motion-spike decision gate."""

import json
import sys
from pathlib import Path


REQUIRED = {
    "static_room_camera_pan",
    "moving_person_fixed_room",
    "moving_object_fixed_room",
    "blur_low_texture",
    "dropped_frames",
    "dynamic_crowd",
}


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def evaluate(root: Path) -> dict:
    manifest = read_object(root / "manifest.json")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("manifest scenarios must be a list")
    names = {item.get("name") for item in scenarios if isinstance(item, dict)}
    if names != REQUIRED or len(scenarios) != len(REQUIRED):
        raise ValueError("motion fixture set differs from the six required scenarios")
    for item in scenarios:
        if item.get("expected") == "withheld" and not item.get("reason"):
            raise ValueError(f"{item.get('id')}: withheld result needs an explicit reason")

    baseline = read_object(root / "baseline.json")
    if baseline.get("model_dependency_added") is not False:
        raise ValueError("19.1 classical decision must not add a model dependency")
    if baseline.get("scenario_count") != len(REQUIRED):
        raise ValueError("baseline scenario count drift")
    if baseline.get("expected_outcomes_passed") != len(REQUIRED):
        raise ValueError("not every required outcome passed")
    if baseline.get("false_ego_motion_rate") != 0.0:
        raise ValueError("false ego-motion gate failed")
    if baseline.get("labelled_flow_epe_px", float("inf")) > 1.0:
        raise ValueError("labelled flow EPE gate failed")
    if baseline.get("static_background_consensus", 0.0) < 0.60:
        raise ValueError("static-background consensus gate failed")
    if baseline.get("cpu_release_p95_ms", float("inf")) > baseline.get("cpu_budget_ms", 0.0):
        raise ValueError("CPU p95 gate failed")
    if baseline.get("decision") != "select_classical_runtime_expert":
        raise ValueError("decision gate is not closed")
    return baseline


def main() -> int:
    try:
        result = evaluate(Path(__file__).resolve().parent)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"motion 19.1 evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
