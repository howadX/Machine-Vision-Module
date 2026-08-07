#!/usr/bin/env python3
"""Fail-closed measured decision for SegFormer-B0 Iteration B."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


CPU_BUDGET_MS = 50.0
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
    if measurement.get("measurement_version") != "19.2.2-b.1":
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
        latency_results[name] = p95 <= CPU_BUDGET_MS

    quality = _object(measurement, "quality")
    if quality.get("semantic_19_2_1_measured") is not False:
        raise EvaluationError("quality must remain unmeasured until dataset evaluation exists")
    if quality.get("unknown_discipline_measured") is not False:
        raise EvaluationError("unknown discipline cannot be claimed without predictions")
    if quality.get("per_class_iou") is not None or quality.get("boundary_f1") is not None:
        raise EvaluationError("unmeasured quality fields must remain null")
    if measurement.get("decision") != REJECT:
        raise EvaluationError("failed CPU/quality gates require reject/defer")
    if not measurement.get("rejection_reasons"):
        raise EvaluationError("rejection reasons are required")

    return {
        "evaluator_version": "19.2.2-b.1",
        "candidate": "segformer_b0_sceneparse150",
        "decision": REJECT,
        "measurement_scope": measurement["measurement_scope"],
        "gates": {
            "onnx_structure_valid": all(artifact.get("onnx_checker_passed") is True for artifact in artifacts.values()),
            "pytorch_onnx_parity_passed": all(parity_results.values()),
            "current_rust_ort_loads_and_runs": all(
                artifact.get("rust_probe_compatible", True) is True for artifact in artifacts.values()
            ),
            "cpu_p95_within_50ms": all(latency_results.values()),
            "semantic_19_2_1_quality_measured": False,
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
