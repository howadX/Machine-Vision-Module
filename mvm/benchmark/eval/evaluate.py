#!/usr/bin/env python3
"""Run the phase-18.10 replay-contract evaluation.

The evaluator is deliberately stdlib-only and consumes frozen fixtures, never
live mutable MVM state.  It fails closed on malformed replay envelopes and
never treats a skeleton fixture as a measured score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


CORE_CLAIMS = (
    "temporal_qa",
    "identity_continuity",
    "object_permanence",
    "uncertainty_honesty",
    "forecast_iou",
    "attention_recall_at_3",
)
REPLAY_STATUSES = {"synthetic_replay", "captured_replay"}
KNOWN_STATUSES = REPLAY_STATUSES | {"skeleton"}


class EvaluationError(ValueError):
    """A fixture or manifest violates the evaluator contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{path}: top-level value must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{label}: expected a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationError(f"{label}: non-finite number")
    return result


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise EvaluationError(f"{label}: bbox must contain four numbers")
    result = tuple(_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise EvaluationError(f"{label}: bbox must have positive area")
    return result  # type: ignore[return-value]


def bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0.0 else 0.0


def validate_frame(frame: Any, fixture_path: Path, previous_id: int | None) -> int:
    if not isinstance(frame, dict):
        raise EvaluationError(f"{fixture_path}: frame must be an object")
    frame_id = frame.get("frame_id")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise EvaluationError(f"{fixture_path}: frame_id must be a non-negative integer")
    if previous_id is not None and frame_id <= previous_id:
        raise EvaluationError(f"{fixture_path}: frame_id sequence is not strictly increasing")

    entities = frame.get("entities", [])
    if not isinstance(entities, list):
        raise EvaluationError(f"{fixture_path}: entities must be an array")
    seen_entities: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict) or not isinstance(entity.get("entity_key"), str):
            raise EvaluationError(f"{fixture_path}: every entity needs an entity_key")
        key = entity["entity_key"]
        if key in seen_entities:
            raise EvaluationError(f"{fixture_path}: duplicate entity_key {key}")
        seen_entities.add(key)
        if entity.get("kind") not in {"person", "object", "animal"}:
            raise EvaluationError(f"{fixture_path}: unsupported entity kind")
        _bbox(entity.get("bbox"), f"{fixture_path} entity {key} bbox")
        if entity.get("existence") not in {"present", "occluded", "absent"}:
            raise EvaluationError(f"{fixture_path}: unsupported entity existence")
        if entity.get("observation_status") not in {"observed", "remembered", "withheld", "unknown"}:
            raise EvaluationError(f"{fixture_path}: unsupported observation status")

    sources = frame.get("sources", [])
    if not isinstance(sources, list):
        raise EvaluationError(f"{fixture_path}: sources must be an array")
    for source in sources:
        if not isinstance(source, dict):
            raise EvaluationError(f"{fixture_path}: source stamp must be an object")
        source_frame = source.get("source_frame_id")
        if isinstance(source_frame, bool) or not isinstance(source_frame, int):
            raise EvaluationError(f"{fixture_path}: source_frame_id must be an integer")
        if source_frame > frame_id:
            raise EvaluationError(f"{fixture_path}: future source stamp {source_frame} > frame {frame_id}")

    events = frame.get("events", [])
    if not isinstance(events, list):
        raise EvaluationError(f"{fixture_path}: events must be an array")
    previous_event: int | None = None
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("event_seq"), int):
            raise EvaluationError(f"{fixture_path}: every event needs integer event_seq")
        event_seq = event["event_seq"]
        if previous_event is not None and event_seq <= previous_event:
            raise EvaluationError(f"{fixture_path}: event sequence is duplicate or non-monotonic")
        previous_event = event_seq
    if frame.get("event_gap") is True and frame.get("event_gap_reported") is not True:
        raise EvaluationError(f"{fixture_path}: event gap must be reported explicitly")
    return frame_id


def validate_fixture(fixture: dict[str, Any], fixture_path: Path) -> None:
    required = ("fixture_version", "scenario_id", "status")
    missing = [key for key in required if key not in fixture]
    if missing:
        raise EvaluationError(f"{fixture_path}: missing required fields {missing}")
    if not isinstance(fixture["scenario_id"], str) or not fixture["scenario_id"]:
        raise EvaluationError(f"{fixture_path}: scenario_id must be a non-empty string")
    status = fixture["status"]
    if status not in KNOWN_STATUSES:
        raise EvaluationError(f"{fixture_path}: unsupported status {status!r}")
    frames = fixture.get("frames", [])
    if not isinstance(frames, list):
        raise EvaluationError(f"{fixture_path}: frames must be an array")
    previous_id: int | None = None
    for frame in frames:
        previous_id = validate_frame(frame, fixture_path, previous_id)


def load_fixtures(repo_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    benchmark = repo_root / "mvm" / "benchmark"
    manifest_path = benchmark / "scenarios" / "manifest.json"
    manifest = _read_json(manifest_path)
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise EvaluationError(f"{manifest_path}: scenarios must be a non-empty array")
    loaded: dict[str, dict[str, Any]] = {}
    for entry in scenarios:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise EvaluationError(f"{manifest_path}: every scenario needs a string id")
        scenario_id = entry["id"]
        if scenario_id in loaded:
            raise EvaluationError(f"{manifest_path}: duplicate scenario id {scenario_id}")
        fixture_path = benchmark / "scenarios" / f"{scenario_id}.json"
        fixture = _read_json(fixture_path)
        if fixture.get("scenario_id") != scenario_id:
            raise EvaluationError(f"{fixture_path}: scenario_id does not match manifest")
        if fixture.get("status") != entry.get("status"):
            raise EvaluationError(f"{fixture_path}: status does not match manifest")
        validate_fixture(fixture, fixture_path)
        loaded[scenario_id] = fixture
    return manifest, loaded


def forecast_metric(fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scenario_id in ("013", "014", "015"):
        fixture = fixtures[scenario_id]
        source = _bbox(fixture["source_bbox"], f"scenario {scenario_id} source_bbox")
        observed = _bbox(fixture["observed_bbox"], f"scenario {scenario_id} observed_bbox")
        velocity = fixture.get("bbox_velocity_per_second")
        if not isinstance(velocity, list) or len(velocity) != 2:
            raise EvaluationError(f"scenario {scenario_id}: velocity must contain two numbers")
        vx, vy = (_finite_number(value, f"scenario {scenario_id} velocity") for value in velocity)
        horizon = _finite_number(fixture.get("horizon_ms"), f"scenario {scenario_id} horizon_ms") / 1000.0
        predicted = (source[0] + vx * horizon, source[1] + vy * horizon, source[2] + vx * horizon, source[3] + vy * horizon)
        candidate = bbox_iou(predicted, observed)
        baseline = bbox_iou(source, observed)
        rows.append({"scenario_id": scenario_id, "candidate_iou": candidate, "baseline_iou": baseline})
    return {
        "status": "measured_synthetic_replay",
        "count": len(rows),
        "candidate_mean": sum(row["candidate_iou"] for row in rows) / len(rows),
        "baseline_mean": sum(row["baseline_iou"] for row in rows) / len(rows),
        "improved_cases": sum(row["candidate_iou"] > row["baseline_iou"] for row in rows),
        "cases": rows,
    }


def _frames(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    return fixture.get("frames", [])


def core_replay_metrics(fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    temporal_cases = [
        any(event.get("kind") == "person_entered" for event in _frames(fixtures["001"])[-1].get("events", [])),
        any(event.get("kind") == "entity_appeared" and event.get("entity_key", "").startswith("run/animal/") for event in _frames(fixtures["009"])[-1].get("events", [])),
    ]
    permanence_frame = _frames(fixtures["004"])[-1]
    permanence_entity = next(entity for entity in permanence_frame["entities"] if entity["kind"] == "object")
    permanence_correct = permanence_entity["existence"] == "occluded" and permanence_entity["observation_status"] == "remembered"

    crossing_frames = _frames(fixtures["007"])
    identity_by_display: dict[str, set[str]] = {}
    for frame in crossing_frames:
        for entity in frame["entities"]:
            identity_by_display.setdefault(entity["display_id"], set()).add(entity["entity_key"])
    identity_correct = all(len(keys) == 1 for keys in identity_by_display.values()) and len(identity_by_display) == 2

    return {
        "temporal_qa": {"status": "measured_synthetic_replay", "correct": sum(temporal_cases), "total": len(temporal_cases), "scenarios": ["001", "009"]},
        "identity_continuity": {"status": "measured_synthetic_replay", "correct": int(identity_correct), "total": 1, "scenarios": ["001", "007"]},
        "object_permanence": {"status": "measured_synthetic_replay", "correct": int(permanence_correct), "total": 1, "scenarios": ["004"]},
    }


def evaluate(repo_root: Path) -> dict[str, Any]:
    manifest, fixtures = load_fixtures(repo_root)
    included = sorted(scenario_id for scenario_id, fixture in fixtures.items() if fixture["status"] in REPLAY_STATUSES)
    excluded = sorted(scenario_id for scenario_id, fixture in fixtures.items() if fixture["status"] == "skeleton")
    loop_scenarios = sorted(
        scenario_id
        for scenario_id, fixture in fixtures.items()
        if fixture.get("loop", {}).get("trigger") in {"uncertainty", "prediction_error"}
    )
    if len(loop_scenarios) < 3:
        raise EvaluationError("phase loop gate requires at least three trigger scenarios")
    metrics: dict[str, Any] = {
        **core_replay_metrics(fixtures),
        "uncertainty_honesty": {"status": "contract_evidence", "scenario_id": "016", "source": "mvm-cortex Rust replay test"},
        "forecast_iou": forecast_metric(fixtures),
        "attention_recall_at_3": {"status": "contract_evidence", "scenario_id": "018", "source": "mvm-cortex Rust replay test"},
        "affordance_accuracy": {"status": "contract_evidence", "scenario_id": "017", "source": "mvm-cortex Rust replay test"},
        "memory_precision_at_5": {"status": "contract_evidence", "source": "mvm-cortex Rust unit replay test"},
        "self_model_claim_accuracy": {"status": "contract_evidence", "source": "mvm-cortex Rust unit replay test"},
        "advisory_reobservation": {"status": "contract_evidence", "scenario_ids": loop_scenarios, "source": "mvm-cli Rust replay test"},
    }
    return {
        "evaluator_version": "18.10.0",
        "fixture_version": manifest.get("fixture_version"),
        "core_claims": list(CORE_CLAIMS),
        "included_scenarios": included,
        "excluded_scenarios": excluded,
        "loop_scenarios": loop_scenarios,
        "metrics": metrics,
        "honesty": {"withheld_unknown_are_missing_claims": True, "malformed_input": "fail_closed"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.repo_root.resolve())
    except EvaluationError as exc:
        print(f"MVM phase-18.10 evaluator FAILED: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
