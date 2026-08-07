import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("semantic_19_2_1_evaluate", ROOT / "evaluate.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SemanticEvaluationTests(unittest.TestCase):
    def fixture(self):
        return MODULE.read_object(ROOT / "scenarios" / "sf001.json")

    def test_committed_contract_corpus_passes(self):
        result = MODULE.evaluate(ROOT)
        self.assertTrue(result["contract"]["passed"])
        self.assertEqual(result["measurement_scope"], "synthetic_contract_no_runtime_expert")
        self.assertEqual(result["runtime_semantic_expert"], "not_implemented_in_19.2.1")

    def test_unknown_prediction_cannot_carry_surface_value(self):
        fixture = self.fixture()
        fixture["prediction"][0][0] = "floor"
        fixture["truth"][0][0] = "unknown"
        with self.assertRaises(MODULE.EvaluationError):
            MODULE.validate_fixture(Path("fixture.json"), fixture, MODULE.load_taxonomy(ROOT))

    def test_unmapped_expert_label_fails_closed(self):
        fixture = self.fixture()
        fixture["prediction"][0][0] = "road"
        with self.assertRaises(MODULE.EvaluationError):
            MODULE.validate_fixture(Path("fixture.json"), fixture, MODULE.load_taxonomy(ROOT))

    def test_ignore_pixels_do_not_change_iou(self):
        fixture = self.fixture()
        fixture["truth"][0][0] = "ignore"
        first = MODULE.validate_fixture(Path("fixture.json"), fixture, MODULE.load_taxonomy(ROOT))
        fixture["prediction"][0][0] = "support"
        second = MODULE.validate_fixture(Path("fixture.json"), fixture, MODULE.load_taxonomy(ROOT))
        self.assertEqual(first["confusion"], second["confusion"])

    def test_false_floor_on_unknown_fails_even_if_aggregate_gate_would_pass(self):
        fixture = self.fixture()
        fixture["truth"][0][0] = "unknown"
        fixture["prediction"][0][0] = "floor"
        with self.assertRaises(MODULE.EvaluationError):
            MODULE.validate_fixture(Path("fixture.json"), fixture, MODULE.load_taxonomy(ROOT))

    def test_taxonomy_cannot_map_background_to_floor(self):
        taxonomy = copy.deepcopy(MODULE.load_taxonomy(ROOT))
        taxonomy["expert_to_benchmark"]["background"] = "floor"
        with self.assertRaises(MODULE.EvaluationError):
            MODULE.validate_taxonomy(Path("taxonomy.json"), taxonomy)


if __name__ == "__main__":
    unittest.main()
