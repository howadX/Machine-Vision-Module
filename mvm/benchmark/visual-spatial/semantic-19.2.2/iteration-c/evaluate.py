#!/usr/bin/env python3
"""Fail-closed executable contract gate for Phase 19.2.2 Iteration C."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


LABELS = ["floor", "wall", "ceiling", "support_region_candidate", "opening_candidate", "other_non_surface", "withheld"]
AUTHORITY_LEVELS = ["raw_observation", "selective_surface_evidence", "fused_surface_fact", "occupancy_free_space"]
FORBIDDEN_PROMOTIONS = {
    "raw_observation_to_selective_evidence_without_calibration",
    "selective_evidence_to_fused_surface_fact_without_fusion",
    "raw_observation_to_fused_surface_fact",
    "opening_candidate_to_open_or_free",
    "support_region_candidate_to_support_geometry",
    "other_non_surface_to_withheld",
    "withheld_to_other_non_surface",
    "withheld_unknown_occluded_stale_to_free",
}


class EvaluationError(ValueError):
    """The committed gate contract is incomplete or has gained authority."""


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{path}: expected object")
    return value


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvaluationError(f"{path}: cannot hash: {exc}") from exc


def _object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise EvaluationError(f"{key} must be an object")
    return value


def require_path(obj: Any, path: tuple[str | int, ...]) -> Any:
    """Return a required source value, failing closed when any path component is absent."""
    current = obj
    rendered: list[str] = []
    for component in path:
        rendered.append(str(component))
        if isinstance(current, dict) and component in current:
            current = current[component]
        elif isinstance(current, list) and isinstance(component, int) and 0 <= component < len(current):
            current = current[component]
        else:
            raise EvaluationError(f"required Iteration-B source field missing: {'.'.join(rendered)}")
    return current


def _validate_iteration_b_sources(measurements: dict[str, Any], quality: dict[str, Any]) -> None:
    artifact_fields = {
        "input_shape", "output_shape", "onnx_bytes", "onnx_checker_passed",
    }
    for artifact in ("fp32_256", "fp32_320", "int8_256", "int8_320"):
        for field in artifact_fields:
            require_path(measurements, ("artifacts", artifact, field))
    for artifact in ("fp32_256", "fp32_320"):
        for field in ("onnx_sha256", "paddle_or_pytorch_onnx_argmax_agreement", "max_abs_logit_error", "mean_abs_logit_error"):
            require_path(measurements, ("artifacts", artifact, field))
    for artifact in ("int8_256", "int8_320"):
        require_path(measurements, ("artifacts", artifact, "rust_probe_compatible"))

    for field in ("hardware", "warmup_runs", "measured_runs"):
        require_path(measurements, ("rust_release_single_thread", field))
    for artifact in ("fp32_256", "fp32_320", "int8_256", "int8_320"):
        for field in ("p50_ms", "p95_ms", "process_max_rss_kib"):
            require_path(measurements, ("rust_release_single_thread", artifact, field))

    for field in (
        "semantic_cadence_frames", "depth_reference_invocation_ms",
        "aggregate_fp32_256_plus_depth_amortized_ms_per_capture_frame",
        "aggregate_fp32_320_plus_depth_amortized_ms_per_capture_frame",
        "capture_blocking", "max_queue_depth", "stale_results_dropped",
        "source_frame_stamp_required", "controlled_ab_wall_time_delta_pct",
        "controlled_ab_measured",
    ):
        require_path(measurements, ("scheduling_integrity", field))
    for artifact in ("fp32_256", "fp32_320", "int8_256", "int8_320"):
        require_path(measurements, ("scheduling_integrity", "semantic_amortized_ms_per_capture_frame", artifact))

    for field in (
        "name", "source_url", "archive_bytes", "archive_sha256",
        "available_validation_frames", "selection", "measured_frames",
        "ground_truth_label_policy", "evaluation_resolution",
    ):
        require_path(quality, ("dataset", field))
    for field in (
        "name", "confidence_threshold", "mean_surface_iou",
        "unknown_abstention_recall", "unknown_pixels",
        "unknown_discipline_violations", "false_floor_on_unknown", "passed",
    ):
        require_path(quality, ("selected_baseline", field))
    for class_name in ("wall", "floor", "ceiling", "support", "opening"):
        require_path(quality, ("selected_baseline", "per_class_iou", class_name))

    fine_tunes = require_path(quality, ("conditional_fine_tunes",))
    if not isinstance(fine_tunes, list) or len(fine_tunes) != 2:
        raise EvaluationError("Iteration-B requires both conditional fine-tunes")
    for index in range(2):
        for field in (
            "name", "steps", "batch_size", "input_size", "learning_rate",
            "class_weights", "trainable_scope", "checkpoint_bytes",
            "checkpoint_sha256", "mean_surface_iou", "floor_iou", "opening_iou",
            "unknown_abstention_recall", "passed",
        ):
            require_path(quality, ("conditional_fine_tunes", index, field))
    require_path(quality, ("boundary_f1",))
    require_path(quality, ("boundary_gate_status",))


def _verify_pin(root: Path, pin: dict[str, Any], role: str) -> Path:
    path_value = pin.get("path")
    digest = pin.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str) or len(digest) != 64:
        raise EvaluationError(f"{role} requires path and SHA-256")
    relative_path = Path(path_value)
    if relative_path.is_absolute():
        raise EvaluationError(f"{role} path must be relative to the benchmark")
    benchmark_root = root.parent.resolve(strict=True)
    try:
        path = (root / relative_path).resolve(strict=True)
        path.relative_to(benchmark_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise EvaluationError(f"{role} path escapes the benchmark or cannot be read: {path_value}") from exc
    if sha256_file(path) != digest:
        raise EvaluationError(f"{role} SHA-256 mismatch")
    return path


def _evaluate_contract_fixture(root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    fixture = _object(provenance, "contract_fixture")
    if fixture.get("contract_fixture_only") is not True or fixture.get("prohibited_uses") != ["quality_evidence", "production_acceptance"]:
        raise EvaluationError("conformance fixture must remain contract_fixture_only, never quality/acceptance evidence")
    calibration = fixture.get("calibration_scene_ids")
    test = fixture.get("test_scene_ids")
    if not isinstance(calibration, list) or not calibration or not isinstance(test, list) or not test:
        raise EvaluationError("fixture requires calibration and test scene IDs")
    if set(calibration) & set(test):
        raise EvaluationError("calibration/test scene collision")

    inputs = []
    for role in ("ground_truth", "predictions"):
        pin = _object(fixture, role)
        if not pin.get("source") or not pin.get("revision"):
            raise EvaluationError(f"{role} requires source and revision")
        path = _verify_pin(root, pin, role)
        artifact = read_object(path)
        if artifact.get("contract_fixture_only") is not True or artifact.get("artifact_role") != role:
            raise EvaluationError(f"{role} fixture identity drift")
        if artifact.get("scene_id") not in test:
            raise EvaluationError(f"{role} must belong to the pinned test split")
        labels = artifact.get("labels")
        if not isinstance(labels, list) or not labels or any(label not in LABELS for label in labels):
            raise EvaluationError(f"{role} labels are invalid")
        inputs.append((artifact, labels))
    if inputs[0][0].get("scene_id") != inputs[1][0].get("scene_id") or len(inputs[0][1]) != len(inputs[1][1]):
        raise EvaluationError("ground truth and predictions must align")

    correct = sum(expected == predicted for expected, predicted in zip(inputs[0][1], inputs[1][1]))
    metric = {"accuracy": correct / len(inputs[0][1]), "correct": correct, "total": len(inputs[0][1])}
    metric_bytes = (json.dumps(metric, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    metric_pin = _object(fixture, "metric_artifact")
    metric_path = _verify_pin(root, metric_pin, "metric_artifact")
    try:
        committed_bytes = metric_path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"metric artifact cannot be read: {exc}") from exc
    if metric_pin.get("metric") != "exact_label_accuracy" or metric_pin.get("expected_result") != metric:
        raise EvaluationError("computed fixture metric result drift")
    if committed_bytes != metric_bytes or hashlib.sha256(metric_bytes).hexdigest() != metric_pin.get("sha256"):
        raise EvaluationError("computed fixture metric artifact drift")
    return {
        "contract_fixture_only": True,
        "metric": metric,
        "metric_artifact_sha256": metric_pin["sha256"],
        "quality_evidence": False,
        "production_acceptance": False,
    }


def _iteration_b_snapshot(root: Path, iteration_b: dict[str, Any]) -> dict[str, Any]:
    if iteration_b.get("authority") != "truthful_reject_for_canonical_surface_and_runtime_authority":
        raise EvaluationError("Iteration B must remain the canonical authority rejection")
    if iteration_b.get("source_commit") != "b9125c73":
        raise EvaluationError("Iteration-B source commit drift")
    pins = iteration_b.get("source_artifacts")
    expected_paths = ["../iteration-b/measurements.json", "../iteration-b/quality_measurement.json"]
    if not isinstance(pins, list) or [pin.get("path") for pin in pins if isinstance(pin, dict)] != expected_paths:
        raise EvaluationError("both Iteration-B source artifacts must be pinned")
    paths = [_verify_pin(root, pin, f"Iteration-B source {index}") for index, pin in enumerate(pins)]
    measurements, quality = (read_object(path) for path in paths)
    _validate_iteration_b_sources(measurements, quality)
    if measurements.get("quality", {}).get("measurement_file") != paths[1].name:
        raise EvaluationError("measurements.json must point to pinned quality_measurement.json")
    if measurements.get("decision") != "reject_candidate_defer_runtime_expert" or quality.get("decision") != measurements.get("decision"):
        raise EvaluationError("Iteration-B truthful rejection drift")

    measured_quality = _object(measurements, "quality")
    selected = _object(quality, "selected_baseline")
    expected_quality_projection = {
        "measured_frames": selected.get("unknown_pixels") is not None and quality.get("dataset", {}).get("measured_frames"),
        "per_class_iou": selected.get("per_class_iou"),
        "mean_surface_iou": selected.get("mean_surface_iou"),
        "unknown_abstention_recall": selected.get("unknown_abstention_recall"),
        "unknown_discipline_violations": selected.get("unknown_discipline_violations"),
        "false_floor_on_unknown": selected.get("false_floor_on_unknown"),
    }
    actual_quality_projection = {
        "measured_frames": measured_quality.get("measured_frames"),
        "per_class_iou": measured_quality.get("per_class_iou"),
        "mean_surface_iou": measured_quality.get("mean_surface_iou"),
        "unknown_abstention_recall": measured_quality.get("unknown_abstention_recall"),
        "unknown_discipline_violations": measured_quality.get("unknown_discipline_violations"),
        "false_floor_on_unknown": measured_quality.get("false_floor_on_unknown"),
    }
    if actual_quality_projection != expected_quality_projection:
        raise EvaluationError("measurements/quality source fields disagree")
    if iteration_b.get("snapshot_policy") != "summarized_from_sources_while_sha256_pins_complete_artifacts":
        raise EvaluationError("Iteration-B snapshot must declare summarized output and complete source pins")
    if iteration_b.get("not_measured") != ["boundary_metrics", "integrated_live_ab"]:
        raise EvaluationError("unmeasured boundary or live A/B evidence must not be fabricated")

    return {
        "source_artifact_sha256": {pin["path"]: pin["sha256"] for pin in pins},
        "snapshot_is_summarized": True,
        "complete_artifacts_pinned_by_sha256": True,
        "cpu": {
            "artifacts": measurements.get("artifacts"),
            "rust_release_single_thread": measurements.get("rust_release_single_thread"),
            "scheduling_integrity": measurements.get("scheduling_integrity"),
        },
        "quality": {
            "dataset": quality.get("dataset"),
            "selected_baseline": selected,
            "conditional_fine_tunes": quality.get("conditional_fine_tunes"),
            "boundary_f1": quality.get("boundary_f1"),
            "boundary_gate_status": quality.get("boundary_gate_status"),
        },
    }


def evaluate(root: Path, contract_override: dict[str, Any] | None = None) -> dict[str, Any]:
    contract = contract_override if contract_override is not None else read_object(root / "contract.json")
    if contract.get("contract_version") != "19.2.2-c" or contract.get("status") != "contract_gate_only":
        raise EvaluationError("iteration-C contract identity drift")
    if contract.get("decision") != "authorize_shadow_runtime_only":
        raise EvaluationError("decision must remain authorize_shadow_runtime_only")

    taxonomy = _object(contract, "taxonomy")
    if taxonomy.get("labels") != LABELS or taxonomy.get("known_non_target_indoor_pixels") != "other_non_surface":
        raise EvaluationError("taxonomy must contain the seven distinct ordered labels")
    if taxonomy.get("invalid_low_confidence_high_entropy_ood") != "withheld" or taxonomy.get("withheld_is_negative_class") is not False:
        raise EvaluationError("withheld must remain abstention, never a negative class")

    authority = _object(contract, "authority")
    forbidden = _object(authority, "forbidden_promotions")
    if authority.get("levels") != AUTHORITY_LEVELS or set(forbidden) != FORBIDDEN_PROMOTIONS or any(value is not True for value in forbidden.values()):
        raise EvaluationError("four authority levels and all forbidden promotions are required")
    if authority.get("occupancy_honesty_gate_owner") != "19.3":
        raise EvaluationError("unknown/occluded/stale-to-free belongs to occupancy gate 19.3")

    provenance = _object(contract, "computed_artifact_provenance")
    expected_provenance = {
        "required_inputs": ["pinned_ground_truth", "predictions"],
        "ground_truth_and_predictions_pinned": True,
        "sha256_required_for": ["ground_truth", "predictions", "metric_artifact"],
        "source_and_revision_required": True,
        "separate_calibration_and_test_splits": True,
        "split_unit": "scene_or_session",
        "split_leakage_forbidden": True,
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise EvaluationError("computed-artifact provenance requirements drift")
    fixture_result = _evaluate_contract_fixture(root, provenance)

    metrics = _object(contract, "future_measurement_requirements")
    if metrics != {
        "per_class_metrics": ["support", "confusion", "boundary", "precision", "recall", "f1", "iou"],
        "calibration_metrics": ["ece", "brier"],
        "selective_metrics": ["risk", "coverage", "risk_coverage_curve"],
        "metric_source_artifact_required": True,
    }:
        raise EvaluationError("future metric schema drift")
    if _object(contract, "dataset_roles") != {
        "ADE20K": "reference_only",
        "NYUv2_or_ScanNet_equivalent_indoor_RGBD": "production_acceptance_indoor",
        "MVM_camera_holdout": "production_acceptance_camera_domain",
    }:
        raise EvaluationError("dataset roles cannot promote ADE or omit production-domain holdouts")

    measured = _iteration_b_snapshot(root, _object(contract, "iteration_b_baseline"))
    authorization = _object(contract, "authorization")
    if authorization != {
        "shadow_runtime": True,
        "canonical_surfaces": False,
        "occupancy_free_space": False,
        "production_quality_acceptance": False,
        "live_ab_completed": False,
    }:
        raise EvaluationError("Iteration C authorizes only a shadow runtime")
    return {
        "evaluator_version": "19.2.2-c",
        "decision": "authorize_shadow_runtime_only",
        "contract_passed": True,
        "authority_levels": AUTHORITY_LEVELS,
        "taxonomy_labels": LABELS,
        "contract_fixture": fixture_result,
        "iteration_b_measured": measured,
        "boundary_metrics_measured": False,
        "integrated_live_ab_measured": False,
        "canonical_surfaces_authorized": False,
        "occupancy_free_space_authorized": False,
        "production_quality_accepted": False,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        result = evaluate(root)
        baseline = read_object(root / "baseline.json")
    except EvaluationError as exc:
        print(f"semantic 19.2.2-c evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if result != baseline:
        print("semantic 19.2.2-c evaluation failed: committed baseline drift", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
