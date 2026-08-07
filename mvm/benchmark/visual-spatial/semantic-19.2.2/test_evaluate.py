import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("semantic_19_2_2_evaluate", ROOT / "evaluate.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SemanticModelDecisionTests(unittest.TestCase):
    def setUp(self):
        self.candidate = MODULE.read_object(ROOT / "candidate.json")

    def evaluate_with(self, candidate):
        with patch.object(MODULE, "read_object", return_value=candidate):
            return MODULE.evaluate(ROOT)

    def test_committed_reject_decision_passes(self):
        result = self.evaluate_with(self.candidate)
        self.assertTrue(result["contract"]["passed"])
        self.assertEqual(result["decision"], "reject_candidate_defer_runtime_expert")

    def test_failed_parity_cannot_select_candidate(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["decision"] = "select_runtime_expert"
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(candidate)

    def test_collapsed_output_cannot_claim_parity(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["measurements"]["control_flow_free_onnx"]["parity_passed"] = True
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(candidate)

    def test_over_budget_latency_cannot_pass(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["measurements"]["rust_ort_release"]["latency_gate_passed"] = True
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(candidate)

    def test_unclear_weights_cannot_enter_manifest(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["integration"]["model_manifest_added"] = True
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(candidate)

    def test_rejected_model_cannot_authorize_occupancy(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["integration"]["occupancy_or_traversability_authorized"] = True
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(candidate)


if __name__ == "__main__":
    unittest.main()
