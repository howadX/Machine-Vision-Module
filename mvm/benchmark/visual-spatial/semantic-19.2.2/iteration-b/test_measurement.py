import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("semantic_19_2_2_b_measurement", ROOT / "evaluate_measurement.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SegFormerMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.measurements = MODULE.read_object(ROOT / "measurements.json")

    def evaluate_with(self, measurements):
        with patch.object(MODULE, "read_object", side_effect=lambda path: measurements):
            return MODULE.evaluate(ROOT)

    def test_committed_measurement_passes_sparse_cpu_budget_but_stays_provisional(self):
        result = self.evaluate_with(self.measurements)
        self.assertEqual(result["decision"], "provisionally_accept_for_quality_gate")
        self.assertTrue(result["gates"]["cpu_invocation_within_250ms"])
        self.assertTrue(result["gates"]["cpu_amortized_within_10ms_per_capture_frame"])
        self.assertFalse(result["gates"]["semantic_19_2_1_quality_measured"])

    def test_sparse_invocation_over_budget_cannot_be_reported_as_pass(self):
        measurements = copy.deepcopy(self.measurements)
        measurements["rust_release_single_thread"]["fp32_256"]["p95_ms"] = 260.0
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(measurements)

    def test_parity_failure_cannot_select_candidate(self):
        measurements = copy.deepcopy(self.measurements)
        measurements["artifacts"]["fp32_256"]["paddle_or_pytorch_onnx_argmax_agreement"] = 0.5
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(measurements)

    def test_unmeasured_quality_cannot_pass(self):
        measurements = copy.deepcopy(self.measurements)
        measurements["quality"]["semantic_19_2_1_measured"] = True
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(measurements)

    def test_quality_pending_candidate_cannot_select_runtime(self):
        measurements = copy.deepcopy(self.measurements)
        measurements["decision"] = "select_sceneparse150_candidate"
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(measurements)

    def test_stale_result_policy_cannot_be_disabled(self):
        measurements = copy.deepcopy(self.measurements)
        measurements["scheduling_integrity"]["stale_results_dropped"] = False
        with self.assertRaises(MODULE.EvaluationError):
            self.evaluate_with(measurements)


if __name__ == "__main__":
    unittest.main()
