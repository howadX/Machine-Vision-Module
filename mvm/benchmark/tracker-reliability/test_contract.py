import hashlib
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TrackerReliabilityContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = load(HERE / "contract.json")
        self.manifests = {
            split: load(HERE / "manifests" / f"{split}.json")
            for split in self.contract["required_splits"]
        }

    def test_authority_and_promotion_contract_are_fail_closed(self):
        self.assertEqual(
            self.contract["schema"], "mvm-tracker-reliability-contract/1.0"
        )
        self.assertEqual(
            self.contract["authority"]["durable_semantic_history"], "labels.jsonl"
        )
        promotion = self.contract["promotion"]
        self.assertTrue(promotion["production_rust_replay_required"])
        self.assertTrue(promotion["captured_calibration_required"])
        self.assertTrue(promotion["captured_held_out_required"])
        self.assertFalse(promotion["synthetic_or_adapter_only_may_promote"])
        self.assertEqual(promotion["false_merges_allowed"], 0)
        self.assertEqual(promotion["false_relinks_allowed"], 0)
        self.assertEqual(promotion["honesty_violations_allowed"], 0)
        replay = self.contract["production_replay"]
        self.assertTrue(replay["exact_frame_binding_required"])
        self.assertFalse(replay["held_out_execution_by_development_runner"])
        self.assertFalse(replay["durable_trace_allowed"])

    def test_required_split_manifests_are_distinct_and_not_promotable_at_t0(self):
        self.assertEqual(set(self.manifests), {"development", "calibration", "held-out"})
        for split, manifest in self.manifests.items():
            self.assertEqual(manifest["schema"], "mvm-tracker-reliability-manifest/1.0")
            self.assertEqual(manifest["split"], split)
            self.assertIn(
                manifest["status"], self.contract["allowed_manifest_statuses"]
            )
            self.assertFalse(manifest["production_promotion_eligible"])
        self.assertEqual(self.manifests["calibration"]["status"], "not_collected")
        self.assertEqual(self.manifests["held-out"]["status"], "not_collected")

    def test_every_registered_artifact_exists_and_matches_sha256(self):
        paths = set()
        for manifest in self.manifests.values():
            for artifact in manifest["artifacts"]:
                relative = artifact["path"]
                self.assertNotIn(relative, paths, f"duplicate artifact {relative}")
                paths.add(relative)
                path = REPO / relative
                self.assertTrue(path.is_file(), relative)
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual, artifact["sha256"], relative)

    def test_historical_replays_are_not_relabelled_as_acceptance_splits(self):
        development = self.manifests["development"]
        self.assertEqual(development["status"], "exploratory_only")
        classes = {artifact["evidence_class"] for artifact in development["artifacts"]}
        self.assertIn("synthetic_adapter", classes)
        self.assertIn("captured_exploratory", classes)
        self.assertIn("production_parity_synthetic", classes)
        self.assertFalse(self.manifests["calibration"]["artifacts"])
        self.assertFalse(self.manifests["held-out"]["artifacts"])


if __name__ == "__main__":
    unittest.main()
