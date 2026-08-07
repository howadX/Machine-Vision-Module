import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("semantic_19_2_2_b_evaluate", ROOT / "evaluate.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SegFormerDecisionContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = MODULE.read_object(ROOT / "contract.json")
        self.datasets = MODULE.read_object(ROOT / "datasets.json")
        self.taxonomy = MODULE.read_object(ROOT / "taxonomy.json")

    def evaluate_with(self, *, contract=None, datasets=None, taxonomy=None):
        values = {
            "contract.json": contract or self.contract,
            "datasets.json": datasets or self.datasets,
            "taxonomy.json": taxonomy or self.taxonomy,
        }

        def read(path):
            return values[path.name]

        with patch.object(MODULE, "read_object", side_effect=read):
            return MODULE.evaluate(ROOT)

    def test_committed_plan_contract_passes_without_claiming_measurements(self):
        result = self.evaluate_with()
        self.assertTrue(result["contract"]["passed"])
        self.assertEqual(result["decision"], "measurement_pending")
        self.assertEqual(result["runtime_semantic_expert"], "not_implemented_in_19.2.2")

    def test_cpu_threshold_cannot_be_weakened(self):
        contract = copy.deepcopy(self.contract)
        contract["thresholds"]["cpu_end_to_end_p95_ms_max"] = 60.0
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract=contract)

    def test_background_cannot_map_to_floor(self):
        taxonomy = copy.deepcopy(self.taxonomy)
        taxonomy["default_unmapped_target"] = "floor"
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(taxonomy=taxonomy)

    def test_opening_label_cannot_authorize_free_space(self):
        taxonomy = copy.deepcopy(self.taxonomy)
        taxonomy["authority"]["opening_implies_free"] = True
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(taxonomy=taxonomy)

    def test_frame_level_split_leakage_is_forbidden(self):
        datasets = copy.deepcopy(self.datasets)
        datasets["split_policy"] = "random_frames"
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(datasets=datasets)

    def test_research_terms_must_remain_explicit(self):
        datasets = copy.deepcopy(self.datasets)
        datasets["sources"][0]["terms"]["accepted_for_current_research_phase"] = False
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(datasets=datasets)

    def test_measurement_slice_cannot_enable_runtime(self):
        contract = copy.deepcopy(self.contract)
        contract["integration"]["runtime_semantic_expert_enabled"] = True
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract=contract)

    def test_depth_cannot_become_required_inference_input(self):
        contract = copy.deepcopy(self.contract)
        contract["depth_policy"]["inference_input"] = "rgb_plus_depth"
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(contract=contract)


if __name__ == "__main__":
    unittest.main()
