#!/usr/bin/env python3
"""Fail-closed pre-measurement contract for Phase 19.2.2 Iteration B."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


REVISION = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SURFACES = {"floor", "wall", "ceiling", "support", "opening"}
EXACT_THRESHOLDS = {
    "mean_surface_iou_min": 0.6,
    "floor_iou_min": 0.75,
    "opening_iou_min": 0.55,
    "floor_boundary_f1_min": 0.8,
    "opening_boundary_f1_min": 0.7,
    "unknown_abstention_recall_min": 0.9,
    "unknown_discipline_violations_max": 0,
    "false_floor_on_unknown_max": 0,
    "argmax_pixel_agreement_min": 0.999,
    "semantic_invocation_p95_ms_max": 250.0,
    "semantic_amortized_ms_per_capture_frame_max": 10.0,
    "aggregate_sparse_experts_amortized_ms_per_capture_frame_max": 20.0,
    "capture_wall_time_delta_pct_max": 5.0,
}


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
    contract = read_object(root / "contract.json")
    datasets = read_object(root / "datasets.json")
    taxonomy = read_object(root / "taxonomy.json")

    if contract.get("contract_version") != "19.2.2-b" or contract.get("status") != "measurement_pending":
        raise EvaluationError("iteration-B contract identity drift")
    candidate = _object(contract, "candidate")
    if candidate.get("name") != "segformer_b0_sceneparse150":
        raise EvaluationError("candidate identity drift")
    if not isinstance(candidate.get("revision"), str) or not REVISION.fullmatch(candidate["revision"]):
        raise EvaluationError("candidate revision must be immutable")
    source_artifacts = _object(candidate, "source_artifacts")
    for field in ("config_json_sha256", "preprocessor_config_json_sha256", "model_safetensors_sha256"):
        value = source_artifacts.get(field)
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            raise EvaluationError(f"candidate source hash is missing: {field}")
    if source_artifacts.get("model_safetensors_bytes") != 15036944:
        raise EvaluationError("candidate checkpoint size drift")
    if candidate.get("input_channels") != 3 or candidate.get("static_input_sizes") != [256, 320]:
        raise EvaluationError("candidate must retain the planned static RGB inputs")
    if candidate.get("output_classes") != 150 or candidate.get("output") != "raw_logits":
        raise EvaluationError("SceneParse150 raw-logit contract drift")
    if contract.get("thresholds") != EXACT_THRESHOLDS:
        raise EvaluationError("acceptance thresholds cannot drift")

    depth = _object(contract, "depth_policy")
    if depth.get("inference_input") != "rgb_only" or depth.get("stale_depth_is_hidden_input") is not False:
        raise EvaluationError("depth cannot become a required or hidden inference input")
    if depth.get("ground_truth_use") != "auxiliary_training_and_evaluation":
        raise EvaluationError("ground-truth depth policy drift")
    scheduling = _object(contract, "scheduling_policy")
    if scheduling != {
        "semantic_cadence_frames": 30,
        "max_observation_age_frames": 60,
        "max_queue_depth": 1,
        "capture_blocking": False,
        "stale_results_dropped": True,
        "source_frame_stamp_required": True,
        "controlled_ab_required_before_runtime_selection": True,
    }:
        raise EvaluationError("sparse expert integrity policy drift")

    integration = _object(contract, "integration")
    if any(value is not False for value in integration.values()):
        raise EvaluationError("measurement-only slice leaked into production integration")

    if datasets.get("manifest_version") != "19.2.2-b":
        raise EvaluationError("dataset manifest identity drift")
    if datasets.get("split_policy") != "scene_or_session_level":
        raise EvaluationError("adjacent frames may not cross dataset splits")
    if datasets.get("download_policy") != "external_artifacts_not_committed":
        raise EvaluationError("dataset bytes cannot enter git")
    sources = datasets.get("sources")
    if not isinstance(sources, list) or [source.get("name") for source in sources] != [
        "ADE20K", "ScanNet_v2", "Structured3D", "MVM_owned_video"
    ]:
        raise EvaluationError("dataset source set drift")
    for source in sources[:3]:
        terms = _object(source, "terms")
        if terms.get("accepted_for_current_research_phase") is not True:
            raise EvaluationError(f"research terms not explicitly accepted for {source.get('name')}")
        if source.get("archive_sha256") is not None:
            if not isinstance(source["archive_sha256"], str) or not SHA256.fullmatch(source["archive_sha256"]):
                raise EvaluationError(f"invalid archive hash for {source.get('name')}")
        if source.get("archive_sha256_required_before_measurement") is not True:
            raise EvaluationError(f"archive hash gate missing for {source.get('name')}")
    own_terms = _object(sources[3], "terms")
    if own_terms.get("accepted_for_current_research_phase") is not False:
        raise EvaluationError("uncollected MVM video cannot claim completed consent/provenance")

    if taxonomy.get("taxonomy_version") != "19.2.2-b" or taxonomy.get("expert_class_count") != 150:
        raise EvaluationError("taxonomy identity drift")
    config_hash = taxonomy.get("official_id2label_sha256")
    if not isinstance(config_hash, str) or not SHA256.fullmatch(config_hash):
        raise EvaluationError("official id2label config hash is missing")
    if set(taxonomy.get("surface_classes", [])) != SURFACES:
        raise EvaluationError("surface taxonomy drift")
    mappings = [
        _object(taxonomy, "direct_surface_labels"),
        _object(taxonomy, "opening_candidate_labels"),
        _object(taxonomy, "support_candidate_labels"),
    ]
    if any(not set(mapping.values()).issubset(SURFACES) for mapping in mappings):
        raise EvaluationError("expert mapping targets a non-surface class")
    if taxonomy.get("default_unmapped_target") != "unknown":
        raise EvaluationError("unmapped/background pixels must remain unknown")
    confidence = _object(taxonomy, "confidence_policy")
    if any(confidence.get(key) != "unknown" for key in ("low_confidence_target", "high_entropy_target", "ood_target")):
        raise EvaluationError("uncertain pixels must abstain as unknown")
    if confidence.get("ignore_is_scored_surface") is not False:
        raise EvaluationError("ignore cannot become a scored surface")
    authority = _object(taxonomy, "authority")
    required_false = {
        "raw_label_is_canonical_fact",
        "opening_implies_open",
        "opening_implies_free",
        "opening_implies_traversable",
        "support_candidate_is_support_geometry",
        "semantic_object_region_creates_entity",
    }
    if any(authority.get(key) is not False for key in required_false):
        raise EvaluationError("taxonomy exceeded semantic measurement authority")
    if authority.get("fusion_owns_surface_facts") is not True:
        raise EvaluationError("Fusion must remain the surface-facts authority")

    mapping_count = sum(len(mapping) for mapping in mappings)
    return {
        "evaluator_version": "19.2.2-b",
        "candidate": candidate["name"],
        "decision": "measurement_pending",
        "planned_input_sizes": candidate["static_input_sizes"],
        "depth_inference_input": depth["inference_input"],
        "source_count": len(sources),
        "surface_mapping_count": mapping_count,
        "contract": {
            "passed": True,
            "taxonomy_authority_violations": 0,
            "occupancy_authority_violations": 0,
            "dataset_bytes_committed": False,
            "model_manifest_unchanged": True,
            "runtime_dependency_unchanged": True,
        },
        "runtime_semantic_expert": "not_implemented_in_19.2.2",
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        result = evaluate(root)
        baseline = read_object(root / "baseline.json")
    except EvaluationError as exc:
        print(f"semantic 19.2.2-b evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if result != baseline:
        print("semantic 19.2.2-b evaluation failed: committed baseline drift", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
