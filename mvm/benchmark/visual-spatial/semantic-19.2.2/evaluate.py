#!/usr/bin/env python3
"""Fail-closed Phase 19.2.2 semantic model decision evaluator."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
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


def evaluate(root: Path) -> dict[str, Any]:
    candidate = read_object(root / "candidate.json")
    if candidate.get("record_version") != "19.2.2":
        raise EvaluationError("unsupported model-spike record")
    if candidate.get("candidate") != "pp_mobileseg_tiny_ade20k":
        raise EvaluationError("candidate identity drift")

    source = _object(candidate, "source")
    provenance = _object(candidate, "provenance")
    toolchain = _object(candidate, "toolchain")
    measurements = _object(candidate, "measurements")
    vim = _object(measurements, "official_vim_export")
    exported = _object(measurements, "control_flow_free_onnx")
    rust = _object(measurements, "rust_ort_release")
    quality = _object(candidate, "quality_gate")
    integration = _object(candidate, "integration")

    for field in ("exported_zip_sha256", "training_weights_sha256"):
        if not isinstance(source.get(field), str) or not SHA256.fullmatch(source[field]):
            raise EvaluationError(f"missing immutable source hash: {field}")
    if not isinstance(exported.get("onnx_sha256"), str) or not SHA256.fullmatch(exported["onnx_sha256"]):
        raise EvaluationError("converted ONNX hash is missing")
    if provenance.get("implementation_license") != "Apache-2.0":
        raise EvaluationError("implementation license record drift")
    if provenance.get("training_dataset") != "ADE20K":
        raise EvaluationError("training dataset provenance drift")
    if provenance.get("pretrained_model_terms") != "research_only":
        raise EvaluationError("pretrained-model restriction must remain explicit")
    if provenance.get("production_redistribution_clear") is not False:
        raise EvaluationError("candidate cannot claim production-clear provenance")

    if toolchain.get("paddlepaddle") != "2.6.2" or toolchain.get("paddle2onnx") != "1.3.1":
        raise EvaluationError("measured conversion toolchain drift")
    if toolchain.get("onnx_opset") != 16 or toolchain.get("rust_ort") != "2.0.0-rc.12/api-22":
        raise EvaluationError("ONNX/ORT compatibility envelope drift")

    if vim.get("onnx_checker_passed") is not True or vim.get("parity_passed") is not False:
        raise EvaluationError("official VIM export result is inconsistent")
    if exported.get("onnx_checker_passed") is not True or exported.get("output_shape_matches") is not True:
        raise EvaluationError("control-flow-free ONNX did not pass structural export gates")
    if exported.get("argmax_pixel_agreement") != 0.0:
        raise EvaluationError("committed parity measurement drift")
    if exported.get("paddle_unique_labels") != [0, 8, 14] or exported.get("onnx_unique_labels") != [2]:
        raise EvaluationError("collapsed-output evidence drift")
    if exported.get("parity_passed") is not False:
        raise EvaluationError("collapsed output cannot pass parity")

    if rust.get("current_ort_loads_and_runs") is not True:
        raise EvaluationError("current Rust ORT compatibility was not demonstrated")
    if rust.get("output_shape") != [1, 150, 256, 256]:
        raise EvaluationError("Rust ORT output shape drift")
    p95 = rust.get("p95_ms")
    budget = rust.get("cpu_budget_ms")
    if not isinstance(p95, (int, float)) or not isinstance(budget, (int, float)) or p95 <= budget:
        raise EvaluationError("committed over-budget measurement drift")
    if rust.get("latency_gate_passed") is not False:
        raise EvaluationError("over-budget candidate cannot pass latency")

    if quality.get("contract") != "semantic-19.2.1" or quality.get("reason") != "export_parity_failed":
        raise EvaluationError("quality gate must point to the executable 19.2.1 contract")
    if quality.get("measured_predictions_available") is not False or quality.get("passed") is not False:
        raise EvaluationError("quality cannot pass without parity-valid predictions")
    for field in ("per_class_iou", "boundary_f1", "unknown_discipline_violations"):
        if quality.get(field) is not None:
            raise EvaluationError(f"unmeasured quality field {field} must remain null")

    if any(integration.get(field) is not False for field in (
        "model_bytes_committed",
        "model_manifest_added",
        "runtime_dependency_added",
        "runtime_semantic_expert_enabled",
        "occupancy_or_traversability_authorized",
    )):
        raise EvaluationError("rejected candidate leaked into production integration")
    if candidate.get("decision") != REJECT:
        raise EvaluationError("failed gates require reject/defer decision")

    return {
        "evaluator_version": "19.2.2",
        "candidate": candidate["candidate"],
        "measurement_scope": "single_candidate_export_parity_and_cpu_spike",
        "gates": {
            "source_artifacts_hashed": True,
            "production_provenance_clear": False,
            "onnx_structure_valid": True,
            "paddle_onnx_parity_passed": False,
            "current_rust_ort_loads_and_runs": True,
            "cpu_p95_within_50ms": False,
            "semantic_19_2_1_quality_measured": False,
        },
        "measurements": {
            "input_shape": exported["input_shape"],
            "output_shape": exported["output_shape"],
            "onnx_bytes": exported["onnx_bytes"],
            "argmax_pixel_agreement": exported["argmax_pixel_agreement"],
            "max_abs_logit_error": exported["max_abs_logit_error"],
            "rust_release_single_thread_p50_ms": rust["p50_ms"],
            "rust_release_single_thread_p95_ms": rust["p95_ms"],
            "rust_process_max_rss_kib": rust["process_max_rss_kib"],
        },
        "decision": REJECT,
        "contract": {
            "passed": True,
            "overclaim_violations": 0,
            "model_manifest_unchanged": True,
            "runtime_dependency_unchanged": True,
            "occupancy_authority_violations": 0,
        },
        "runtime_semantic_expert": "not_implemented_after_19.2.2",
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        result = evaluate(root)
        baseline = read_object(root / "baseline.json")
    except EvaluationError as exc:
        print(f"semantic 19.2.2 evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if result != baseline:
        print("semantic 19.2.2 evaluation failed: committed baseline drift", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
