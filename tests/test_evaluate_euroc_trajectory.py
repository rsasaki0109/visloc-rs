import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_euroc_trajectory.py"
SPEC = importlib.util.spec_from_file_location("evaluate_euroc_trajectory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvaluateEurocTrajectoryTests(unittest.TestCase):
    def test_timestamp_parser_preserves_nanoseconds(self) -> None:
        self.assertEqual(MODULE.timestamp_ns("1403636579963555584.000000"), 1403636579963555584)

    def test_loads_visloc_csv_and_skips_failed_tracking_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slam_trajectory.csv"
            path.write_text(
                "timestamp_ns,frame_idx,px,py,pz,qw,qx,qy,qz,tracking_success\n"
                "10,0,1,2,3,1,0,0,0,1\n"
                "20,1,4,5,6,1,0,0,0,0\n",
                encoding="utf-8",
            )
            poses = MODULE.load_estimate(path)
        self.assertEqual(len(poses), 1)
        self.assertEqual(poses[0][0], 10)
        np.testing.assert_allclose(poses[0][1], [1.0, 2.0, 3.0])

    def test_loads_standard_tum_seconds_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.tum"
            path.write_text(
                "1403636579.963555584 1 2 3 0 0 0 1\n",
                encoding="utf-8",
            )
            poses = MODULE.load_estimate(path, tum_time_unit="s")
        self.assertEqual(len(poses), 1)
        self.assertEqual(poses[0][0], 1_403_636_579_963_555_584)
        np.testing.assert_allclose(poses[0][1], [1.0, 2.0, 3.0])

    def test_rigidly_transformed_trajectory_has_zero_error(self) -> None:
        angle = 0.4
        align_rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        align_translation = np.asarray([2.0, -1.0, 0.5])
        ground_truth = []
        estimate = []
        for index in range(6):
            stamp = 1_000_000_000 + index * 50_000_000
            est_position = np.asarray([index * 0.2, index * index * 0.03, 0.1 * index])
            est_rotation = np.eye(3)
            gt_position = align_rotation @ est_position + align_translation
            gt_rotation = align_rotation @ est_rotation
            ground_truth.append((stamp, gt_position, gt_rotation))
            estimate.append((stamp + 200, est_position, est_rotation))

        result = MODULE.evaluate(ground_truth, estimate, max_diff_ns=1_000)

        self.assertEqual(result["associated_poses"], 6)
        self.assertAlmostEqual(result["association_ratio"], 1.0)
        self.assertLess(result["ate_translation_se3_m"]["rmse"], 1e-12)
        self.assertLess(result["rpe_translation_consecutive_m"]["rmse"], 1e-12)
        # 1e-6 deg is tighter than cross-platform libm reproducibility
        # (observed 1.7e-6 on x86-64 glibc); 1e-4 deg is still exactness for
        # an identity-vs-rigid-transform comparison.
        self.assertLess(result["rpe_rotation_consecutive_deg"]["rmse"], 1e-4)

    def test_restricts_comparison_to_exact_common_timestamps(self) -> None:
        rotation = np.eye(3)
        first = [
            (stamp, np.asarray([float(stamp), 0.0, 0.0]), rotation)
            for stamp in (1, 2, 3, 4)
        ]
        second = [
            (stamp, np.asarray([float(stamp), 1.0, 0.0]), rotation)
            for stamp in (2, 3, 4, 5)
        ]

        restricted = MODULE.restrict_to_common_timestamps([first, second])

        self.assertEqual([[pose[0] for pose in run] for run in restricted], [[2, 3, 4], [2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
