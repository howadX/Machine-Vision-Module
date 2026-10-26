import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("tracker_depth_evaluate", ROOT / "evaluate.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluatorContractTests(unittest.TestCase):
    def replay(self):
        return json.loads((ROOT / "replay.json").read_text(encoding="utf-8"))

    def captured_replay(self):
        return json.loads((ROOT / "captured-2026-08-13-replay.json").read_text(encoding="utf-8"))

    def test_frozen_replay_covers_every_required_scenario(self):
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(MODULE.validate(self.replay()), set(manifest["scenarios"]))

    def test_malformed_xyz_fails_closed(self):
        replay = self.replay()
        replay["scenarios"][0]["frames"][0][0]["xyz"] = [1.0, 2.0]
        with self.assertRaises(MODULE.ContractError):
            MODULE.validate(replay)

    def test_synthetic_comparison_cannot_promote_production(self):
        result = MODULE.evaluate(self.replay(), 1)
        self.assertEqual(result["verdict"]["selected"], "A")
        self.assertFalse(result["verdict"]["production_change"])
        self.assertEqual(set(result["variants"]), {"A", "B", "C"})
        self.assertEqual(result["variants"]["A"]["scenario_count"], 13)

    def test_captured_person_only_replay_scores_production_without_fake_object_domain(self):
        replay = {
            "schema": "mvm-depth-tracker-replay/1.0",
            "status": "captured_replay",
            "production_observations": [
                {"truth": "p1", "predicted_track_ids": [10], "selected_track_id": 10},
                {"truth": "p1", "predicted_track_ids": [10, 11], "selected_track_id": 10},
                {"truth": "p1", "predicted_track_ids": [], "selected_track_id": None},
                {"truth": "p1", "predicted_track_ids": [12], "selected_track_id": 12},
                {"truth": None, "predicted_track_ids": [12], "selected_track_id": None},
            ],
            "scenarios": [{
                "id": "captured", "domain": "person", "frames": [
                    [{"truth": "p1", "bbox": [0.1, 0.1, 0.4, 0.9], "xyz": [0, 0, 2]}]
                ]
            }],
        }
        MODULE.validate(replay)
        result = MODULE.evaluate(replay, 1)
        self.assertNotIn("object", result["variants"]["A"]["by_domain"])
        self.assertEqual(result["production_tracker"]["false_negatives"], 1)
        self.assertEqual(result["production_tracker"]["false_positives"], 2)
        self.assertEqual(result["production_tracker"]["id_switches"], 1)
        self.assertEqual(result["production_tracker"]["fragmentations"], 1)

    def test_frozen_captured_replay_has_complete_rgbd_and_review_provenance(self):
        replay = self.captured_replay()
        self.assertEqual(MODULE.validate(replay), {"captured_one_person_motion"})
        observations = replay["production_observations"]
        self.assertEqual(len(observations), 39)
        self.assertEqual([item["seq"] for item in observations], list(range(39)))
        self.assertEqual(sum(item["truth"] is not None for item in observations), 37)
        self.assertTrue(all(len(item["frame_sha256"]) == 64 for item in observations))
        self.assertTrue(all(len(item["depth_sha256"]) == 64 for item in observations))
        for item in observations:
            depth = item["depth_metadata"]
            self.assertEqual(depth["encoding"], "metric_mm_u16")
            self.assertEqual(depth["units"], "millimetres")
            self.assertEqual(depth["geometry"], "registered_to_rgb")
            self.assertLessEqual(depth["rgb_depth_skew_ns"], 20_000_000)

        result = MODULE.evaluate(replay, 1)
        self.assertEqual(result["verdict"]["selected"], "A")
        self.assertFalse(result["verdict"]["production_change"])
        self.assertEqual(result["production_tracker"]["sample_count"], 39)


if __name__ == "__main__":
    unittest.main()
