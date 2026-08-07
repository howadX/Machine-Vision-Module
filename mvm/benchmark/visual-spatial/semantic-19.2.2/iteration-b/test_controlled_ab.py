import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("controlled_ab", ROOT / "controlled_ab.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ControlledABTests(unittest.TestCase):
    def test_sparse_worker_does_not_block_capture(self):
        result = MODULE.simulate(worker_ms=231.357)
        self.assertFalse(result["capture_blocking"])
        self.assertEqual(result["max_queue_depth"], 1)
        self.assertTrue(result["stale_results_dropped"])
        self.assertTrue(result["source_frame_stamp_required"])
        self.assertLessEqual(result["capture_wall_time_delta_pct"], 5.0)

    def test_stale_result_is_withheld(self):
        result = MODULE.simulate(frames=180, cadence=30, worker_ms=5000.0, max_age_frames=10)
        self.assertGreater(result["stale_results_withheld"], 0)


if __name__ == "__main__":
    unittest.main()
