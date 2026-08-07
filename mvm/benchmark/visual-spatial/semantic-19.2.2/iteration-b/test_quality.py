import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("semantic_quality", ROOT / "evaluate_quality.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QualityDecisionTests(unittest.TestCase):
    def setUp(self):
        self.value = json.loads((ROOT / "quality_measurement.json").read_text())

    def evaluate_with(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.json"
            path.write_text(json.dumps(value))
            return MODULE.evaluate(path)

    def test_committed_quality_record_rejects_candidate(self):
        result = self.evaluate_with(self.value)
        self.assertEqual(result["decision"], MODULE.REJECT)
        self.assertEqual(result["fine_tunes_rejected"], 2)

    def test_honesty_violations_cannot_be_hidden(self):
        value = copy.deepcopy(self.value)
        value["selected_baseline"]["unknown_discipline_violations"] = 0
        value["selected_baseline"]["false_floor_on_unknown"] = 0
        value["selected_baseline"]["mean_surface_iou"] = 0.8
        value["selected_baseline"]["per_class_iou"]["floor"] = 0.8
        value["selected_baseline"]["per_class_iou"]["opening"] = 0.8
        value["selected_baseline"]["passed"] = True
        with self.assertRaises(ValueError):
            self.evaluate_with(value)

    def test_failed_fine_tune_cannot_be_selected(self):
        value = copy.deepcopy(self.value)
        value["conditional_fine_tunes"][0]["passed"] = True
        with self.assertRaises(ValueError):
            self.evaluate_with(value)


if __name__ == "__main__":
    unittest.main()
