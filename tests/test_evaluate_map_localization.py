import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_map_localization.py"
SPEC = importlib.util.spec_from_file_location("evaluate_map_localization", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def trajectory(count=20):
    """{timestamp_ns: (p, R)} on a non-degenerate 3D curve."""
    poses = {}
    for index in range(count):
        t = 0.3 * index
        poses[1_000_000_000 + index * 50_000_000] = (
            np.asarray([np.cos(t), np.sin(2 * t), 0.1 * t]),
            rotation_z(0.2 * t),
        )
    return poses


def transformed(poses, rotation, translation):
    return {
        timestamp: (rotation @ position + translation, rotation @ orientation)
        for timestamp, (position, orientation) in poses.items()
    }


class EvaluateMapLocalizationTests(unittest.TestCase):
    def test_tum_seconds_and_euroc_ground_truth_parse_to_nanoseconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tum = Path(directory) / "localization.tum"
            tum.write_text(
                "# timestamp tx ty tz qx qy qz qw\n1403636579.963555584 1 2 3 0 0 0 1\n",
                encoding="utf-8",
            )
            gt = Path(directory) / "data.csv"
            gt.write_text(
                "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n1403636579963555584,1,2,3,1,0,0,0\n",
                encoding="utf-8",
            )
            estimate = MODULE.load_tum_seconds(tum)
            ground_truth = MODULE.load_euroc_ground_truth(gt)
        # Seconds lose sub-microsecond precision; association absorbs it.
        self.assertEqual(len(MODULE.associate(estimate, ground_truth, 1_000)), 1)
        np.testing.assert_allclose(next(iter(estimate.values()))[0], [1.0, 2.0, 3.0])

    def test_association_respects_max_difference(self) -> None:
        estimate = {100: None, 200: None, 400: None}
        reference = {105: None, 260: None}
        self.assertEqual(MODULE.associate(estimate, reference, 10), [(100, 105)])

    def test_direct_mode_measures_without_alignment(self) -> None:
        reference = trajectory()
        offset = transformed(reference, np.eye(3), np.asarray([0.06, 0.0, 0.08]))
        result = MODULE.evaluate("direct", offset, reference, 1_000)
        self.assertAlmostEqual(result["translation_m"]["median"], 0.1)
        self.assertAlmostEqual(result["rotation_deg"]["max"], 0.0, places=6)
        self.assertEqual(result["within_5cm"], 0.0)

    def test_fixed_mode_applies_the_alignment_fitted_on_another_trajectory(self) -> None:
        rotation, translation = rotation_z(0.7), np.asarray([4.0, -1.0, 0.5])
        map_poses = trajectory()
        ground_truth = transformed(map_poses, rotation, translation)
        alignment = MODULE.fit_to_ground_truth(map_poses, ground_truth, 1_000)
        np.testing.assert_allclose(alignment[0], rotation, atol=1e-9)
        np.testing.assert_allclose(alignment[1], translation, atol=1e-9)

        # A query trajectory in the map frame, 2 cm off along x: the fixed
        # alignment keeps that error, while a fit on the query would too here
        # but could absorb a bias in general.
        query = transformed(map_poses, np.eye(3), np.asarray([0.02, 0.0, 0.0]))
        result = MODULE.evaluate("fixed", query, ground_truth, 1_000, alignment)
        self.assertAlmostEqual(result["translation_m"]["rmse"], 0.02, places=9)
        self.assertEqual(result["within_5cm"], 1.0)

    def test_gt_mode_removes_a_rigid_offset(self) -> None:
        reference = trajectory()
        moved = transformed(reference, rotation_z(-0.4), np.asarray([1.0, 2.0, 3.0]))
        result = MODULE.evaluate("gt", moved, reference, 1_000)
        self.assertLess(result["translation_m"]["max"], 1e-9)


if __name__ == "__main__":
    unittest.main()
