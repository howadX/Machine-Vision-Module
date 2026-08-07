#!/usr/bin/env python3
"""Fail-closed Phase 19.2.1 semantic-surface contract evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_CLASSES = {"floor", "wall", "ceiling", "support", "opening", "unknown", "ignore"}
REQUIRED_SURFACES = {"floor", "wall", "ceiling", "support", "opening"}
REQUIRED_BOUNDARIES = {"floor", "opening"}
REQUIRED_CORRUPTIONS = {"blur", "glare", "low_light", "mirror", "ood"}


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


def validate_taxonomy(path: Path, taxonomy: dict[str, Any]) -> None:
    if taxonomy.get("taxonomy_version") != "19.2.1":
        raise EvaluationError(f"{path}: unsupported taxonomy version")
    if set(taxonomy.get("benchmark_classes", [])) != REQUIRED_CLASSES:
        raise EvaluationError(f"{path}: benchmark class set drift")
    if set(taxonomy.get("surface_classes", [])) != REQUIRED_SURFACES:
        raise EvaluationError(f"{path}: canonical surface class set drift")
    if set(taxonomy.get("boundary_classes", [])) != REQUIRED_BOUNDARIES:
        raise EvaluationError(f"{path}: boundary class set drift")

    canonical = taxonomy.get("benchmark_to_canonical")
    expert = taxonomy.get("expert_to_benchmark")
    honesty = taxonomy.get("honesty")
    if not isinstance(canonical, dict) or not isinstance(expert, dict) or not isinstance(honesty, dict):
        raise EvaluationError(f"{path}: invalid taxonomy mappings")
    if set(canonical) != REQUIRED_CLASSES:
        raise EvaluationError(f"{path}: incomplete canonical mapping")
    if any(canonical.get(label) != label for label in REQUIRED_SURFACES):
        raise EvaluationError(f"{path}: surface mapping must preserve canonical names")
    if canonical.get("unknown") is not None or canonical.get("ignore") is not None:
        raise EvaluationError(f"{path}: unknown/ignore cannot become canonical surfaces")
    if expert.get("background") != "unknown" or expert.get("unlabeled") != "unknown":
        raise EvaluationError(f"{path}: background/unlabeled must map to unknown")
    if not set(expert.values()).issubset(REQUIRED_CLASSES - {"ignore"}):
        raise EvaluationError(f"{path}: expert mapping targets an invalid class")
    if honesty.get("unknown_truth_allows_only") != ["unknown"]:
        raise EvaluationError(f"{path}: unknown truth must require abstention")
    if honesty.get("background_is_not_floor") is not True:
        raise EvaluationError(f"{path}: background-to-floor prohibition missing")
    if honesty.get("opening_is_not_traversable") is not True:
        raise EvaluationError(f"{path}: opening cannot imply traversability")


def load_taxonomy(root: Path) -> dict[str, Any]:
    taxonomy_path = root / "taxonomy.json"
    taxonomy = read_object(taxonomy_path)
    validate_taxonomy(taxonomy_path, taxonomy)
    return taxonomy


def _validate_grid(path: Path, name: str, grid: Any, width: int, height: int) -> list[list[str]]:
    if not isinstance(grid, list) or len(grid) != height:
        raise EvaluationError(f"{path}: {name} height mismatch")
    for row in grid:
        if not isinstance(row, list) or len(row) != width or not all(isinstance(value, str) for value in row):
            raise EvaluationError(f"{path}: {name} is not a rectangular string grid")
    return grid


def _boundary(grid: list[list[str]], label: str, ignored: set[tuple[int, int]]) -> set[tuple[int, int]]:
    height = len(grid)
    width = len(grid[0])
    points: set[tuple[int, int]] = set()
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value != label or (x, y) in ignored:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    points.add((x, y))
                    break
                if (nx, ny) not in ignored and grid[ny][nx] != label:
                    points.add((x, y))
                    break
    return points


def validate_fixture(path: Path, fixture: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]:
    required = {
        "fixture_version", "scenario_id", "name", "status", "measurement_scope",
        "width", "height", "truth", "prediction", "prediction_source",
        "corruptions", "honesty_assertions",
    }
    if missing := sorted(required - fixture.keys()):
        raise EvaluationError(f"{path}: missing fields {missing}")
    if fixture["fixture_version"] != "19.2.1":
        raise EvaluationError(f"{path}: unsupported fixture version")
    if fixture["status"] != "synthetic_contract" or fixture["measurement_scope"] != "semantic_outcome_contract":
        raise EvaluationError(f"{path}: fixture must remain a synthetic semantic contract")
    if fixture["prediction_source"] != "synthetic_oracle":
        raise EvaluationError(f"{path}: 19.2.1 cannot claim runtime model measurements")
    width, height = fixture["width"], fixture["height"]
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise EvaluationError(f"{path}: invalid dimensions")
    truth = _validate_grid(path, "truth", fixture["truth"], width, height)
    prediction = _validate_grid(path, "prediction", fixture["prediction"], width, height)
    classes = set(taxonomy["benchmark_classes"])
    if any(value not in classes for row in truth for value in row):
        raise EvaluationError(f"{path}: truth contains an unmapped label")
    if any(value not in classes - {"ignore"} for row in prediction for value in row):
        raise EvaluationError(f"{path}: prediction contains an unmapped or ignore label")
    if not isinstance(fixture["honesty_assertions"], list) or not fixture["honesty_assertions"]:
        raise EvaluationError(f"{path}: honesty assertions are required")
    if not isinstance(fixture["corruptions"], list) or not all(
        isinstance(value, str) for value in fixture["corruptions"]
    ):
        raise EvaluationError(f"{path}: corruptions must be a string list")

    confusion = {
        actual: {predicted: 0 for predicted in classes - {"ignore"}}
        for actual in classes - {"ignore"}
    }
    ignored: set[tuple[int, int]] = set()
    unknown_total = 0
    unknown_correct = 0
    false_floor_on_unknown = 0
    for y in range(height):
        for x in range(width):
            actual = truth[y][x]
            predicted = prediction[y][x]
            if actual == "ignore":
                ignored.add((x, y))
                continue
            confusion[actual][predicted] += 1
            if actual == "unknown":
                unknown_total += 1
                unknown_correct += int(predicted == "unknown")
                false_floor_on_unknown += int(predicted == "floor")
                if predicted != "unknown":
                    raise EvaluationError(f"{path}: unknown pixel ({x},{y}) became {predicted}")

    boundaries: dict[str, dict[str, int]] = {}
    for label in taxonomy["boundary_classes"]:
        expected = _boundary(truth, label, ignored)
        observed = _boundary(prediction, label, ignored)
        boundaries[label] = {
            "true_positive": len(expected & observed),
            "predicted": len(observed),
            "expected": len(expected),
        }
    return {
        "confusion": confusion,
        "boundaries": boundaries,
        "unknown_total": unknown_total,
        "unknown_correct": unknown_correct,
        "false_floor_on_unknown": false_floor_on_unknown,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _rounded(value: float) -> float:
    return round(value, 6)


def evaluate(root: Path) -> dict[str, Any]:
    taxonomy = load_taxonomy(root)
    manifest_path = root / "manifest.json"
    manifest = read_object(manifest_path)
    if manifest.get("fixture_version") != "19.2.1":
        raise EvaluationError(f"{manifest_path}: unsupported fixture version")
    if manifest.get("measurement_scope") != "synthetic_contract_no_runtime_expert":
        raise EvaluationError(f"{manifest_path}: measurement scope overclaims runtime evidence")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) < 8:
        raise EvaluationError(f"{manifest_path}: at least eight scenarios are required")
    if set(manifest.get("required_corruptions", [])) != REQUIRED_CORRUPTIONS:
        raise EvaluationError(f"{manifest_path}: corruption coverage drift")
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict):
        raise EvaluationError(f"{manifest_path}: thresholds are required")

    classes = taxonomy["benchmark_classes"]
    aggregate = {
        actual: {predicted: 0 for predicted in classes if predicted != "ignore"}
        for actual in classes if actual != "ignore"
    }
    boundary_counts = {
        label: {"true_positive": 0, "predicted": 0, "expected": 0}
        for label in taxonomy["boundary_classes"]
    }
    seen_ids: set[str] = set()
    seen_corruptions: set[str] = set()
    seen_truth_classes: set[str] = set()
    honesty_assertion_count = 0
    unknown_total = 0
    unknown_correct = 0
    false_floor_on_unknown = 0
    for entry in scenarios:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not isinstance(entry.get("name"), str):
            raise EvaluationError(f"{manifest_path}: invalid scenario entry")
        scenario_id = entry["id"]
        if scenario_id in seen_ids:
            raise EvaluationError(f"{manifest_path}: duplicate scenario {scenario_id}")
        seen_ids.add(scenario_id)
        fixture_path = root / "scenarios" / f"{scenario_id}.json"
        fixture = read_object(fixture_path)
        if fixture.get("scenario_id") != scenario_id or fixture.get("name") != entry["name"]:
            raise EvaluationError(f"{fixture_path}: manifest mismatch")
        measured = validate_fixture(fixture_path, fixture, taxonomy)
        for actual, row in measured["confusion"].items():
            for predicted, count in row.items():
                aggregate[actual][predicted] += count
        for label, counts in measured["boundaries"].items():
            for key, count in counts.items():
                boundary_counts[label][key] += count
        seen_corruptions.update(fixture["corruptions"])
        seen_truth_classes.update(value for row in fixture["truth"] for value in row)
        honesty_assertion_count += len(fixture["honesty_assertions"])
        unknown_total += measured["unknown_total"]
        unknown_correct += measured["unknown_correct"]
        false_floor_on_unknown += measured["false_floor_on_unknown"]

    if not REQUIRED_CORRUPTIONS.issubset(seen_corruptions):
        raise EvaluationError(f"{manifest_path}: required corruption fixtures are missing")
    if not REQUIRED_CLASSES.issubset(seen_truth_classes):
        raise EvaluationError(f"{manifest_path}: taxonomy coverage is incomplete")

    class_iou: dict[str, float] = {}
    for label in taxonomy["surface_classes"]:
        true_positive = aggregate[label][label]
        false_positive = sum(aggregate[actual][label] for actual in aggregate if actual != label)
        false_negative = sum(aggregate[label][predicted] for predicted in aggregate[label] if predicted != label)
        class_iou[label] = _rounded(_ratio(true_positive, true_positive + false_positive + false_negative))
    mean_surface_iou = _rounded(sum(class_iou.values()) / len(class_iou))

    boundary_f1: dict[str, float] = {}
    for label, counts in boundary_counts.items():
        precision = _ratio(counts["true_positive"], counts["predicted"])
        recall = _ratio(counts["true_positive"], counts["expected"])
        boundary_f1[label] = _rounded(_ratio(2 * precision * recall, precision + recall))
    unknown_recall = _rounded(_ratio(unknown_correct, unknown_total))
    violations = unknown_total - unknown_correct
    passed = (
        mean_surface_iou >= thresholds["mean_surface_iou_min"]
        and class_iou["floor"] >= thresholds["floor_iou_min"]
        and class_iou["opening"] >= thresholds["opening_iou_min"]
        and boundary_f1["floor"] >= thresholds["floor_boundary_f1_min"]
        and boundary_f1["opening"] >= thresholds["opening_boundary_f1_min"]
        and unknown_recall >= thresholds["unknown_abstention_recall_min"]
        and violations <= thresholds["unknown_discipline_violations_max"]
        and false_floor_on_unknown <= thresholds["false_floor_on_unknown_max"]
    )
    if not passed:
        raise EvaluationError("committed synthetic contract does not meet its gates")

    return {
        "evaluator_version": "19.2.1",
        "fixture_version": manifest["fixture_version"],
        "measurement_scope": manifest["measurement_scope"],
        "scenario_count": len(seen_ids),
        "taxonomy": {
            "surface_classes": taxonomy["surface_classes"],
            "unknown_and_ignore_are_not_surfaces": True,
            "opening_does_not_imply_traversability": True,
        },
        "metrics": {
            "per_class_iou": class_iou,
            "mean_surface_iou": mean_surface_iou,
            "boundary_f1": boundary_f1,
            "unknown_abstention_recall": unknown_recall,
        },
        "contract": {
            "passed": True,
            "honesty_assertion_count": honesty_assertion_count,
            "unknown_discipline_violations": violations,
            "false_floor_on_unknown": false_floor_on_unknown,
            "corruptions_covered": sorted(seen_corruptions & REQUIRED_CORRUPTIONS),
        },
        "runtime_semantic_expert": "not_implemented_in_19.2.1",
        "runtime_surface_availability": {"available": 0, "total": len(seen_ids)},
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        result = evaluate(root)
        baseline = read_object(root / "baseline.json")
    except EvaluationError as exc:
        print(f"semantic 19.2.1 evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if baseline != result:
        print("semantic 19.2.1 evaluation failed: committed baseline drift", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
