import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("spatial_evaluate", ROOT / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


class SpatialEvaluatorTest(unittest.TestCase):
    def fixture(self):
        return EVALUATE.read_object(ROOT / "scenarios" / "vs002.json")

    def test_committed_corpus_passes(self):
        self.assertTrue(EVALUATE.evaluate(ROOT)["contract"]["passed"])

    def test_missing_claim_with_value_fails_closed(self):
        fixture = self.fixture()
        fixture["expected_facts"][1]["value"] = "free"
        with self.assertRaises(EVALUATE.EvaluationError):
            EVALUATE.validate_fixture(Path("fixture.json"), fixture)

    def test_free_without_positive_evidence_fails_closed(self):
        fixture = self.fixture()
        fact = copy.deepcopy(fixture["expected_facts"][0])
        fact["value"] = "free"
        fact.pop("positive_evidence")
        fixture["expected_facts"] = [fact]
        with self.assertRaises(EVALUATE.EvaluationError):
            EVALUATE.validate_fixture(Path("fixture.json"), fixture)

    def test_agent_local_metric_without_extrinsics_fails_closed(self):
        fixture = self.fixture()
        fixture["expected_facts"] = [{
            "kind": "clearance",
            "reference_frame": "agent_local",
            "metric": True,
            "status": "derived",
            "value": 0.5,
        }]
        with self.assertRaises(EVALUATE.EvaluationError):
            EVALUATE.validate_fixture(Path("fixture.json"), fixture)

    def test_graph_revision_gap_fails_closed(self):
        fixture = self.fixture()
        fixture["graph_deltas"] = [
            {"from_revision": 0, "to_revision": 1, "operation_count": 1},
            {"from_revision": 2, "to_revision": 3, "operation_count": 1},
        ]
        with self.assertRaises(EVALUATE.EvaluationError):
            EVALUATE.validate_fixture(Path("fixture.json"), fixture)


if __name__ == "__main__":
    unittest.main()
