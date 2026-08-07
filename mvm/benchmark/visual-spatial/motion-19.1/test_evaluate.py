import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("motion_19_1_evaluate", ROOT / "evaluate.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MotionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = MODULE.read_object(ROOT / "manifest.json")
        self.baseline = MODULE.read_object(ROOT / "baseline.json")

    def evaluate_with(self, manifest=None, baseline=None):
        values = [manifest or self.manifest, baseline or self.baseline]
        with patch.object(MODULE, "read_object", side_effect=values):
            return MODULE.evaluate(ROOT)

    def test_committed_baseline_passes(self):
        self.assertEqual(self.evaluate_with()["expected_outcomes_passed"], 6)

    def test_missing_required_scenario_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["scenarios"].pop()
        with self.assertRaises(ValueError):
            self.evaluate_with(manifest=manifest)

    def test_false_ego_motion_fails(self):
        baseline = dict(self.baseline, false_ego_motion_rate=0.1)
        with self.assertRaises(ValueError):
            self.evaluate_with(baseline=baseline)

    def test_cpu_budget_regression_fails(self):
        baseline = dict(self.baseline, cpu_release_p95_ms=51.0)
        with self.assertRaises(ValueError):
            self.evaluate_with(baseline=baseline)


if __name__ == "__main__":
    unittest.main()
