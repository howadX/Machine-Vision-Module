import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("semantic_19_2_2_c_evaluate", ROOT / "evaluate.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class IterationCSemanticGateTests(unittest.TestCase):
    def setUp(self):
        self.contract = MODULE.read_object(ROOT / "contract.json")

    def evaluate_with(self, contract):
        return MODULE.evaluate(ROOT, contract_override=contract)

    def fixture_copy(self):
        temporary = tempfile.TemporaryDirectory()
        parent = Path(temporary.name) / "semantic-19.2.2"
        shutil.copytree(ROOT.parent, parent, ignore=shutil.ignore_patterns("__pycache__"))
        return temporary, parent / "iteration-c"

    def test_committed_contract_authorizes_shadow_runtime_only(self):
        result = MODULE.evaluate(ROOT)
        self.assertEqual(result["decision"], "authorize_shadow_runtime_only")
        self.assertTrue(result["contract_passed"])
        self.assertFalse(result["canonical_surfaces_authorized"])
        self.assertFalse(result["occupancy_free_space_authorized"])
        self.assertFalse(result["production_quality_accepted"])

    def test_taxonomy_distinguishes_known_non_target_from_abstention(self):
        taxonomy = self.contract["taxonomy"]
        self.assertEqual(
            taxonomy["labels"],
            ["floor", "wall", "ceiling", "support_region_candidate", "opening_candidate", "other_non_surface", "withheld"],
        )
        self.assertEqual(taxonomy["known_non_target_indoor_pixels"], "other_non_surface")
        self.assertEqual(taxonomy["invalid_low_confidence_high_entropy_ood"], "withheld")

    def test_taxonomy_drift_fails_closed(self):
        contract = copy.deepcopy(self.contract)
        contract["taxonomy"]["known_non_target_indoor_pixels"] = "withheld"
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract)

    def test_four_authority_levels_and_forbidden_promotions_are_enforced(self):
        contract = copy.deepcopy(self.contract)
        contract["authority"]["levels"].pop()
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract)
        contract = copy.deepcopy(self.contract)
        contract["authority"]["forbidden_promotions"]["opening_candidate_to_open_or_free"] = False
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract)

    def test_contract_fixture_reads_inputs_computes_and_hashes_expected_metric(self):
        result = MODULE.evaluate(ROOT)
        fixture = result["contract_fixture"]
        self.assertTrue(fixture["contract_fixture_only"])
        self.assertEqual(fixture["metric"], {"accuracy": 0.75, "correct": 6, "total": 8})
        self.assertEqual(len(fixture["metric_artifact_sha256"]), 64)

    def test_contract_fixture_rejects_changed_ground_truth_prediction_or_declared_hash(self):
        for mutation in ("ground_truth", "predictions", "declared_hash"):
            with self.subTest(mutation=mutation):
                temporary, root = self.fixture_copy()
                try:
                    contract = MODULE.read_object(root / "contract.json")
                    fixture = contract["computed_artifact_provenance"]["contract_fixture"]
                    if mutation == "declared_hash":
                        fixture["ground_truth"]["sha256"] = "0" * 64
                    else:
                        path = root / fixture[mutation]["path"]
                        value = MODULE.read_object(path)
                        value["labels"][0] = "wall" if value["labels"][0] != "wall" else "floor"
                        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
                    with self.assertRaises(MODULE.EvaluationError):
                        MODULE.evaluate(root, contract_override=contract)
                finally:
                    temporary.cleanup()

    def test_contract_fixture_rejects_calibration_test_scene_collision(self):
        contract = copy.deepcopy(self.contract)
        fixture = contract["computed_artifact_provenance"]["contract_fixture"]
        fixture["test_scene_ids"] = fixture["calibration_scene_ids"][:]
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract)

    def test_future_measurement_schema_requires_full_selective_metrics(self):
        required = self.contract["future_measurement_requirements"]
        self.assertEqual(required["per_class_metrics"], ["support", "confusion", "boundary", "precision", "recall", "f1", "iou"])
        self.assertEqual(required["calibration_metrics"], ["ece", "brier"])
        self.assertEqual(required["selective_metrics"], ["risk", "coverage", "risk_coverage_curve"])
        contract = copy.deepcopy(self.contract)
        contract["future_measurement_requirements"]["calibration_metrics"].remove("brier")
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract)

    def test_dataset_roles_keep_ade_reference_only_and_require_production_domains(self):
        roles = self.contract["dataset_roles"]
        self.assertEqual(roles["ADE20K"], "reference_only")
        self.assertEqual(roles["NYUv2_or_ScanNet_equivalent_indoor_RGBD"], "production_acceptance_indoor")
        self.assertEqual(roles["MVM_camera_holdout"], "production_acceptance_camera_domain")
        contract = copy.deepcopy(self.contract)
        contract["dataset_roles"]["ADE20K"] = "production_acceptance"
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract)

    def test_iteration_b_pins_and_hashes_both_complete_sources(self):
        sources = self.contract["iteration_b_baseline"]["source_artifacts"]
        self.assertEqual(
            sources,
            [
                {"path": "../iteration-b/measurements.json", "sha256": "c8ceeb42d09cbeba7909ee5b5744ddd7bd265d50e30b5174b499112f30d09c7f"},
                {"path": "../iteration-b/quality_measurement.json", "sha256": "4af2c08924bdfff6b76ab9f31af36141d20556692323bb65c05e4d7cc1f888a0"},
            ],
        )
        contract = copy.deepcopy(self.contract)
        contract["iteration_b_baseline"]["source_artifacts"][1]["sha256"] = "0" * 64
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract)

    def assert_ground_truth_pin_path_rejected(self, root, path_value, external_path=None):
        contract = MODULE.read_object(root / "contract.json")
        pin = contract["computed_artifact_provenance"]["contract_fixture"]["ground_truth"]
        source_path = root / pin["path"]
        if external_path is not None:
            external_path.write_bytes(source_path.read_bytes())
            pin["sha256"] = MODULE.sha256_file(external_path)
        pin["path"] = path_value
        with self.assertRaises(MODULE.EvaluationError):
            MODULE.evaluate(root, contract_override=contract)

    def test_contract_fixture_rejects_absolute_pin_path(self):
        temporary, root = self.fixture_copy()
        try:
            pin = self.contract["computed_artifact_provenance"]["contract_fixture"]["ground_truth"]
            self.assert_ground_truth_pin_path_rejected(root, str((root / pin["path"]).resolve()))
        finally:
            temporary.cleanup()

    def test_contract_fixture_rejects_parent_escape_pin_path(self):
        temporary, root = self.fixture_copy()
        try:
            self.assert_ground_truth_pin_path_rejected(root, "../../escape.json", Path(temporary.name) / "escape.json")
        finally:
            temporary.cleanup()

    def test_contract_fixture_rejects_symlink_escape_pin_path(self):
        temporary, root = self.fixture_copy()
        try:
            external_path = Path(temporary.name) / "escape.json"
            source_path = root / self.contract["computed_artifact_provenance"]["contract_fixture"]["ground_truth"]["path"]
            external_path.write_bytes(source_path.read_bytes())
            link = root / "symlink-escape.json"
            try:
                link.symlink_to(external_path)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unsupported: {exc}")
            self.assert_ground_truth_pin_path_rejected(root, "symlink-escape.json", external_path)
        finally:
            temporary.cleanup()

    def test_iteration_b_snapshot_is_derived_and_preserves_cpu_quality_and_fine_tunes(self):
        measured = MODULE.evaluate(ROOT)["iteration_b_measured"]
        runtime = measured["cpu"]["rust_release_single_thread"]
        self.assertEqual(runtime["fp32_256"], {"p50_ms": 132.446, "p95_ms": 164.979, "process_max_rss_kib": 146644})
        self.assertEqual(runtime["int8_320"], {"p50_ms": 455.554, "p95_ms": 494.692, "process_max_rss_kib": 155724})
        self.assertEqual(measured["cpu"]["artifacts"]["int8_256"]["onnx_bytes"], 4200003)
        self.assertEqual(measured["cpu"]["scheduling_integrity"]["semantic_amortized_ms_per_capture_frame"]["int8_320"], 16.4897)
        quality = measured["quality"]
        self.assertEqual(quality["selected_baseline"]["per_class_iou"], {"wall": 0.597713, "floor": 0.641646, "ceiling": 0.739643, "support": 0.279541, "opening": 0.427946})
        self.assertEqual(quality["selected_baseline"]["unknown_pixels"], 563841)
        self.assertEqual(len(quality["conditional_fine_tunes"]), 2)
        self.assertEqual(quality["conditional_fine_tunes"][1]["checkpoint_bytes"], 1593725)

    def test_measurements_must_point_to_pinned_quality_source(self):
        temporary, root = self.fixture_copy()
        try:
            contract = MODULE.read_object(root / "contract.json")
            measurements_path = root / "../iteration-b/measurements.json"
            measurements = MODULE.read_object(measurements_path)
            measurements["quality"]["measurement_file"] = "other.json"
            measurements_path.write_text(json.dumps(measurements, indent=2) + "\n", encoding="utf-8")
            contract["iteration_b_baseline"]["source_artifacts"][0]["sha256"] = MODULE.sha256_file(measurements_path)
            with self.assertRaises(MODULE.EvaluationError):
                MODULE.evaluate(root, contract_override=contract)
        finally:
            temporary.cleanup()

    def assert_source_mutation_fails_closed(self, source_name, mutate):
        temporary, root = self.fixture_copy()
        try:
            contract = MODULE.read_object(root / "contract.json")
            pin = next(pin for pin in contract["iteration_b_baseline"]["source_artifacts"] if pin["path"].endswith(source_name))
            path = root / pin["path"]
            source = MODULE.read_object(path)
            mutate(source)
            path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
            pin["sha256"] = MODULE.sha256_file(path)
            with self.assertRaises(MODULE.EvaluationError):
                MODULE.evaluate(root, contract_override=contract)
        finally:
            temporary.cleanup()

    def test_iteration_b_rejects_missing_fp32_p95(self):
        self.assert_source_mutation_fails_closed(
            "measurements.json", lambda source: source["rust_release_single_thread"]["fp32_256"].pop("p95_ms")
        )

    def test_iteration_b_rejects_missing_int8_artifact(self):
        self.assert_source_mutation_fails_closed("measurements.json", lambda source: source["artifacts"].pop("int8_256"))

    def test_iteration_b_rejects_missing_scheduling_field(self):
        self.assert_source_mutation_fails_closed(
            "measurements.json", lambda source: source["scheduling_integrity"].pop("source_frame_stamp_required")
        )

    def test_iteration_b_rejects_missing_all_conditional_fine_tunes(self):
        self.assert_source_mutation_fails_closed("quality_measurement.json", lambda source: source.pop("conditional_fine_tunes"))

    def test_iteration_b_rejects_missing_quality_class_iou(self):
        self.assert_source_mutation_fails_closed(
            "quality_measurement.json", lambda source: source["selected_baseline"]["per_class_iou"].pop("opening")
        )

    def test_tdd_evidence_records_observed_reds_without_invented_logs(self):
        evidence = MODULE.read_object(ROOT / "tdd_evidence.json")
        self.assertEqual(evidence["status"], "observed_red_evidence")
        self.assertTrue(evidence["runs"])
        for run in evidence["runs"]:
            self.assertTrue(run["command"])
            self.assertTrue(run["expected_reason"])
            self.assertEqual(run["observed_status"], "nonzero")
        self.assertFalse(evidence["exact_log_text_claimed"])

    def test_dynamic_import_does_not_create_iteration_c_pycache(self):
        self.assertTrue(sys.dont_write_bytecode)
        self.assertFalse((ROOT / "__pycache__").exists())

    def test_canonical_or_occupancy_authorization_fails_closed(self):
        for key in ("canonical_surfaces", "occupancy_free_space", "production_quality_acceptance"):
            contract = copy.deepcopy(self.contract)
            contract["authorization"][key] = True
            with self.assertRaises(MODULE.EvaluationError):
                self.evaluate_with(contract)

    def test_decision_is_exact(self):
        contract = copy.deepcopy(self.contract)
        contract["decision"] = "authorize_canonical_surfaces"
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract)


if __name__ == "__main__":
    unittest.main()
