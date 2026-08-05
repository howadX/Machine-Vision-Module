import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "benchmark" / "eval" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("benchmark_evaluate", MODULE_PATH)
evaluate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate)


class BenchmarkEvaluatorTests(unittest.TestCase):
    def test_current_fixtures_produce_replay_report(self):
        root = Path(__file__).parents[2].parent
        report = evaluate.evaluate(root)
        self.assertEqual(report["fixture_version"], "18.0.0")
        self.assertEqual(report["excluded_scenarios"], [])
        self.assertEqual(report["metrics"]["temporal_qa"]["correct"], 2)
        self.assertEqual(report["metrics"]["temporal_qa"]["total"], 2)
        self.assertEqual(report["metrics"]["identity_continuity"]["correct"], 1)
        self.assertEqual(report["metrics"]["object_permanence"]["correct"], 1)
        self.assertEqual(report["loop_scenarios"], ["019", "020", "021", "022"])
        forecast = report["metrics"]["forecast_iou"]
        self.assertEqual(forecast["at_1s"]["improved_cases"], 3)
        self.assertEqual(forecast["at_2s"]["improved_cases"], 3)
        self.assertAlmostEqual(forecast["at_2s"]["candidate_mean"], 1.0)
        self.assertEqual(report["metrics"]["advisory_reobservation"]["resolved"], 3)
        self.assertEqual(report["phase_gate"]["improved_core_claims"], 6)
        self.assertTrue(report["phase_gate"]["passed"])

    def test_future_source_stamp_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark = root / "mvm" / "benchmark" / "scenarios"
            benchmark.mkdir(parents=True)
            (root / "mvm" / "benchmark" / "scenarios" / "manifest.json").write_text(
                json.dumps({"fixture_version": "18.0.0", "scenarios": [{"id": "x", "status": "captured_replay"}]}),
                encoding="utf-8",
            )
            (benchmark / "x.json").write_text(
                json.dumps({"fixture_version": "18.0.0", "scenario_id": "x", "status": "captured_replay", "frames": [{"frame_id": 1, "sources": [{"source_frame_id": 2}]}]}),
                encoding="utf-8",
            )
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.evaluate(root)

    def test_event_gap_without_explicit_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark = root / "mvm" / "benchmark" / "scenarios"
            benchmark.mkdir(parents=True)
            (root / "mvm" / "benchmark" / "scenarios" / "manifest.json").write_text(
                json.dumps({"fixture_version": "18.0.0", "scenarios": [{"id": "x", "status": "captured_replay"}]}),
                encoding="utf-8",
            )
            (benchmark / "x.json").write_text(
                json.dumps({"fixture_version": "18.0.0", "scenario_id": "x", "status": "captured_replay", "frames": [{"frame_id": 1, "event_gap": True}]}),
                encoding="utf-8",
            )
            with self.assertRaises(evaluate.EvaluationError):
                evaluate.evaluate(root)


if __name__ == "__main__":
    unittest.main()
