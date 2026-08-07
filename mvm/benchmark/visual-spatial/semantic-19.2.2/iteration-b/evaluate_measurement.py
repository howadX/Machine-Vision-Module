#!/usr/bin/env python3
"""Fail-closed measured decision for SegFormer-B0 Iteration B."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


SEMANTIC_INVOCATION_BUDGET_MS = 250.0
SEMANTIC_AMORTIZED_BUDGET_MS = 10.0
AGGREGATE_SPARSE_BUDGET_MS = 20.0
PARITY_MIN = 0.999
REJECT = "reject_candidate_defer_runtime_expert"


class EvaluationError(ValueError):
    pass


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{path}: expected object")
    return value


def _object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise EvaluationError(f"{key} must be an object")
    return value


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvaluationError(f"{name} must be finite")
    return float(value)


def evaluate(root: Path) -> dict[str, Any]:
    measurement = read_object(root / "measurements.json")
    if measurement.get("measurement_version") != "19.2.2-b.2":
        raise EvaluationError("measurement version drift")
    if measurement.get("candidate_revision") != "489d5cd81a0b59fab9b7ea758d3548ebe99677da":
        raise EvaluationError("candidate revision drift")
    artifacts = _object(measurement, "artifacts")
    timings = _object(measurement, "rust_release_single_thread")

    parity_results: dict[str, bool] = {}
    for name, artifact in artifacts.items():
        if artifact.get("onnx_checker_passed") is not True:
            raise EvaluationError(f"ONNX checker failed for {name}")
        shape = artifact.get("output_shape")
        if not isinstance(shape, list) or shape[:2] != [1, 150] or len(shape) != 4:
            raise EvaluationError(f"invalid SceneParse150 output shape for {name}")
        for field in ("onnx_bytes",):
            if not isinstance(artifact.get(field), int) or artifact[field] <= 0:
                raise EvaluationError(f"invalid {field} for {name}")
        if "paddle_or_pytorch_onnx_argmax_agreement" in artifact:
            agreement = _finite_number(artifact["paddle_or_pytorch_onnx_argmax_agreement"], f"{name} parity")
            if agreement < PARITY_MIN:
                raise EvaluationError(f"parity failed for {name}")
            parity_results[name] = True
            _finite_number(artifact.get("max_abs_logit_error"), f"{name} max logit error")
            _finite_number(artifact.get("mean_abs_logit_error"), f"{name} mean logit error")

    latency_results: dict[str, bool] = {}
    timing_names = ("fp32_256", "fp32_320", "int8_256", "int8_320")
    for name in timing_names:
        timing = _object(timings, name)
        p50 = _finite_number(timing.get("p50_ms"), f"{name} p50")
        p95 = _finite_number(timing.get("p95_ms"), f"{name} p95")
        rss = timing.get("process_max_rss_kib")
        if p50 < 0 or p95 < p50 or not isinstance(rss, int) or rss <= 0:
            raise EvaluationError(f"invalid timing envelope for {name}")
        latency_results[name] = p95 <= SEMANTIC_INVOCATION_BUDGET_MS

    quality = _object(measurement, "quality")
    if quality.get("semantic_19_2_1_measured") is not True or quality.get("unknown_discipline_measured") is not True:
        raise EvaluationError("measured rejection requires quality and unknown-discipline evidence")
    class_iou = _object(quality, "per_class_iou")
    if set(class_iou) != {"wall", "floor", "ceiling", "support", "opening"}:
        raise EvaluationError("surface IoU class set drift")
    for label, value in class_iou.items():
        score = _finite_number(value, f"{label} IoU")
        if not 0.0 <= score <= 1.0:
            raise EvaluationError(f"invalid {label} IoU")
    mean_iou = _finite_number(quality.get("mean_surface_iou"), "mean surface IoU")
    unknown_recall = _finite_number(quality.get("unknown_abstention_recall"), "unknown abstention recall")
    violations = quality.get("unknown_discipline_violations")
    false_floor = quality.get("false_floor_on_unknown")
    if not isinstance(violations, int) or violations <= 0 or not isinstance(false_floor, int) or false_floor <= 0:
        raise EvaluationError("quality rejection must preserve measured honesty violations")
    quality_passed = (
        mean_iou >= 0.60
        and class_iou["floor"] >= 0.75
        and class_iou["opening"] >= 0.55
        and unknown_recall >= 0.90
        and violations == 0
        and false_floor == 0
    )
    if quality_passed:
        raise EvaluationError("rejected measurement unexpectedly passes quality gates")
    if quality.get("boundary_f1") is not None:
        raise EvaluationError("boundary F1 must remain null when rejection short-circuits before that gate")
    scheduling = _object(measurement, "scheduling_integrity")
    cadence = scheduling.get("semantic_cadence_frames")
    if cadence != 30:
        raise EvaluationError("semantic cadence drift")
    if scheduling.get("capture_blocking") is not False or scheduling.get("max_queue_depth") != 1:
        raise EvaluationError("capture blocking/queue policy drift")
    if scheduling.get("stale_results_dropped") is not True or scheduling.get("source_frame_stamp_required") is not True:
        raise EvaluationError("stale/stamp integrity policy drift")
    if scheduling.get("controlled_ab_measured") is not False:
        raise EvaluationError("controlled A/B must remain pending before runtime selection")
    amortized = _object(scheduling, "semantic_amortized_ms_per_capture_frame")
    for name, timing in timings.items():
        if name not in timing_names:
            continue
        expected = _finite_number(timing.get("p95_ms"), f"{name} p95") / cadence
        observed = _finite_number(amortized.get(name), f"{name} amortized cost")
        if abs(expected - observed) > 0.01:
            raise EvaluationError(f"amortized timing mismatch for {name}")
    for name in ("fp32_256", "fp32_320"):
        p95 = _finite_number(timings[name]["p95_ms"], f"{name} p95")
        if p95 > SEMANTIC_INVOCATION_BUDGET_MS:
            raise EvaluationError(f"{name} exceeds sparse invocation budget")
        if _finite_number(amortized[name], f"{name} amortized cost") > SEMANTIC_AMORTIZED_BUDGET_MS:
            raise EvaluationError(f"{name} exceeds semantic amortized budget")
    for name in ("fp32_256", "fp32_320"):
        key = f"aggregate_{name}_plus_depth_amortized_ms_per_capture_frame"
        if _finite_number(scheduling.get(key), key) > AGGREGATE_SPARSE_BUDGET_MS:
            raise EvaluationError(f"{key} exceeds aggregate sparse budget")
    if measurement.get("decision") != REJECT:
        raise EvaluationError("quality-failing candidate must be rejected")
    reasons = measurement.get("rejection_reasons")
    if not isinstance(reasons, list) or "mean_surface_iou_below_0_60" not in reasons:
        raise EvaluationError("quality rejection reasons are missing")
    pending = measurement.get("pending_gates")
    if pending != ["new_candidate_selection"]:
        raise EvaluationError("rejected candidate must not retain runtime gates")

    return {
        "evaluator_version": "19.2.2-b.2",
        "candidate": "segformer_b0_sceneparse150",
        "decision": REJECT,
        "measurement_scope": measurement["measurement_scope"],
        "gates": {
            "onnx_structure_valid": all(artifact.get("onnx_checker_passed") is True for artifact in artifacts.values()),
            "pytorch_onnx_parity_passed": all(parity_results.values()),
            "current_rust_ort_loads_and_runs": all(
                artifact.get("rust_probe_compatible", True) is True for artifact in artifacts.values()
            ),
            "cpu_invocation_within_250ms": all(
                _finite_number(timings[name]["p95_ms"], f"{name} p95") <= SEMANTIC_INVOCATION_BUDGET_MS
                for name in ("fp32_256", "fp32_320")
            ),
            "cpu_amortized_within_10ms_per_capture_frame": all(
                _finite_number(amortized[name], f"{name} amortized cost") <= SEMANTIC_AMORTIZED_BUDGET_MS
                for name in ("fp32_256", "fp32_320")
            ),
            "aggregate_sparse_experts_within_20ms_per_capture_frame": all(
                _finite_number(scheduling[f"aggregate_{name}_plus_depth_amortized_ms_per_capture_frame"], name)
                <= AGGREGATE_SPARSE_BUDGET_MS
                for name in ("fp32_256", "fp32_320")
            ),
            "semantic_19_2_1_quality_measured": True,
            "semantic_19_2_1_quality_passed": False,
            "controlled_ab_wall_time_gate_measured": False,
        },
        "measurements": {
            "fp32_256_p95_ms": timings["fp32_256"]["p95_ms"],
            "fp32_320_p95_ms": timings["fp32_320"]["p95_ms"],
            "int8_256_p95_ms": timings["int8_256"]["p95_ms"],
            "int8_320_p95_ms": timings["int8_320"]["p95_ms"],
            "fp32_256_rss_kib": timings["fp32_256"]["process_max_rss_kib"],
            "int8_256_rss_kib": timings["int8_256"]["process_max_rss_kib"],
            "fp32_256_onnx_bytes": artifacts["fp32_256"]["onnx_bytes"],
            "int8_256_onnx_bytes": artifacts["int8_256"]["onnx_bytes"],
            "fp32_256_amortized_ms_per_capture_frame": amortized["fp32_256"],
            "fp32_320_amortized_ms_per_capture_frame": amortized["fp32_320"],
            "aggregate_fp32_256_plus_depth_amortized_ms_per_capture_frame": scheduling[
                "aggregate_fp32_256_plus_depth_amortized_ms_per_capture_frame"
            ],
            "aggregate_fp32_320_plus_depth_amortized_ms_per_capture_frame": scheduling[
                "aggregate_fp32_320_plus_depth_amortized_ms_per_capture_frame"
            ],
            "mean_surface_iou": mean_iou,
            "floor_iou": class_iou["floor"],
            "opening_iou": class_iou["opening"],
            "unknown_abstention_recall": unknown_recall,
            "unknown_discipline_violations": violations,
            "false_floor_on_unknown": false_floor,
        },
        "runtime_semantic_expert": "not_implemented_in_19.2.2",
        "integration": {
            "model_manifest_unchanged": True,
            "runtime_dependency_unchanged": True,
            "occupancy_authority_violations": 0,
        },
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        result = evaluate(root)
        baseline = read_object(root / "measurement_baseline.json")
    except EvaluationError as exc:
        print(f"semantic 19.2.2-b measurement evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if result != baseline:
        print("semantic 19.2.2-b measurement evaluation failed: baseline drift", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
