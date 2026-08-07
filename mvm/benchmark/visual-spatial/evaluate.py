#!/usr/bin/env python3
"""Fail-closed Phase 19.0 spatial contract evaluator (stdlib only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ASSERTED = {"observed", "derived", "temporal", "remembered", "weak"}
MISSING = {"withheld", "unknown"}


class EvaluationError(ValueError):
    pass


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{path}: expected an object")
    return value


def validate_fixture(path: Path, fixture: dict[str, Any]) -> tuple[int, int]:
    required = {
        "fixture_version",
        "scenario_id",
        "name",
        "status",
        "measurement_scope",
        "inputs",
        "expected_facts",
        "honesty_assertions",
    }
    if missing := sorted(required - fixture.keys()):
        raise EvaluationError(f"{path}: missing fields {missing}")
    if fixture["fixture_version"] != "19.0.0":
        raise EvaluationError(f"{path}: unsupported fixture version")
    if fixture["status"] != "synthetic_replay" or fixture["measurement_scope"] != "contract_only":
        raise EvaluationError(f"{path}: fixture must remain contract-only synthetic replay")
    inputs = fixture["inputs"]
    facts = fixture["expected_facts"]
    assertions = fixture["honesty_assertions"]
    if not isinstance(inputs, dict) or not isinstance(facts, list) or not isinstance(assertions, list) or not assertions:
        raise EvaluationError(f"{path}: invalid inputs/facts/assertions shape")

    for fact in facts:
        if not isinstance(fact, dict) or fact.get("status") not in ASSERTED | MISSING:
            raise EvaluationError(f"{path}: invalid fact status")
        status = fact["status"]
        if status in MISSING and ("value" in fact or "confidence" in fact):
            raise EvaluationError(f"{path}: withheld/unknown fact carries a value or confidence")
        if status in ASSERTED and "value" not in fact:
            raise EvaluationError(f"{path}: asserted fact has no value")
        if fact.get("kind") == "occupancy" and fact.get("value") == "free":
            if status not in ASSERTED or fact.get("positive_evidence") is not True:
                raise EvaluationError(f"{path}: free occupancy lacks positive evidence")
        if fact.get("reference_frame") == "agent_local" and fact.get("metric") is True and status in ASSERTED:
            if inputs.get("intrinsics") is not True or inputs.get("extrinsics") is not True:
                raise EvaluationError(f"{path}: asserted agent-local metric fact lacks calibration")

    previous = None
    for delta in fixture.get("graph_deltas", []):
        if not isinstance(delta, dict):
            raise EvaluationError(f"{path}: graph delta must be an object")
        start = delta.get("from_revision")
        end = delta.get("to_revision")
        count = delta.get("operation_count")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            raise EvaluationError(f"{path}: graph delta does not advance")
        if not isinstance(count, int) or not 0 <= count <= 256:
            raise EvaluationError(f"{path}: graph delta exceeds operation bound")
        if previous is not None and start != previous:
            raise EvaluationError(f"{path}: graph delta revision gap")
        previous = end

    source_frames = [source.get("source_frame_id") for source in inputs.get("sources", [])]
    if any(not isinstance(frame, int) for frame in source_frames):
        raise EvaluationError(f"{path}: invalid source frame")
    return len(assertions), len(facts)


def evaluate(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = read_object(manifest_path)
    entries = manifest.get("scenarios")
    if not isinstance(entries, list) or len(entries) < 6:
        raise EvaluationError(f"{manifest_path}: at least six scenarios are required")
    seen: set[str] = set()
    assertion_count = 0
    fact_count = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise EvaluationError(f"{manifest_path}: invalid scenario entry")
        scenario_id = entry["id"]
        if scenario_id in seen:
            raise EvaluationError(f"{manifest_path}: duplicate scenario {scenario_id}")
        seen.add(scenario_id)
        fixture_path = root / "scenarios" / f"{scenario_id}.json"
        fixture = read_object(fixture_path)
        if fixture.get("scenario_id") != scenario_id or fixture.get("name") != entry.get("name"):
            raise EvaluationError(f"{fixture_path}: manifest mismatch")
        assertions, facts = validate_fixture(fixture_path, fixture)
        assertion_count += assertions
        fact_count += facts
    return {
        "evaluator_version": "19.0.0",
        "fixture_version": manifest.get("fixture_version"),
        "measurement_scope": "contract_only_no_runtime_expert",
        "scenario_count": len(seen),
        "expected_fact_count": fact_count,
        "honesty_assertion_count": assertion_count,
        "contract": {
            "passed": True,
            "unknown_not_free_violations": 0,
            "missing_claim_payload_violations": 0,
            "agent_local_calibration_violations": 0,
            "graph_revision_or_capacity_violations": 0,
        },
        "runtime_spatial_expert": "not_implemented_in_19.0",
        "baseline_spatial_availability": {"available": 0, "total": len(seen)},
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        result = evaluate(root)
    except EvaluationError as exc:
        print(f"spatial replay evaluation failed: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    baseline = read_object(root / "baseline-19.0.json")
    if baseline != result:
        print("spatial replay evaluation failed: committed baseline drift", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
