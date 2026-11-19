import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_kitti_lidar_clearance import ray_votes


class LidarClearanceTest(unittest.TestCase):
    def test_only_beam_crossing_a_voxel_without_foreground_can_clear(self):
        cells = np.array([[4, 0, 0]])
        pose = np.eye(4)
        pose[:3, 3] = [0, 0.5, 0.5]
        for scan, expected in (
            ([[8, 0, 0], [8, 2, 0]], (False, True)),
            ([[8, 0, 0], [2, 0, 0]], (False, False)),
            ([[8, 0, 0], [4.5, 0, 0]], (True, False)),
            ([[8, 2, 0], [8, 3, 0]], (False, False)),
        ):
            with self.subTest(scan=scan):
                hit, free = ray_votes(cells, np.asarray(scan), pose, 1.0, 0.07)
                self.assertEqual((bool(hit[0]), bool(free[0])), expected)

    def test_empty_scan_leaves_clearance_unknown(self):
        hit, free = ray_votes(np.array([[4, 0, 0]]), np.empty((0, 3)), np.eye(4), 1.0, 0.07)
        self.assertFalse(hit[0] or free[0])


if __name__ == "__main__":
    unittest.main()
