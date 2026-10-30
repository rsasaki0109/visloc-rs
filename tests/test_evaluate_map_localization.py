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

    def test_local_mode_scores_against_the_map_around_each_pose(self) -> None:
        # A map driven 40 m out along x and back 2 m to the side, whose heading
        # drifted 0.05 deg per metre: the true path bends away from the map's.
        positions = [np.asarray([0.25 * i, 0.0, 0.0]) for i in range(161)]
        positions += [np.asarray([40.0 - 0.25 * i, 2.0, 0.0]) for i in range(161)]
        map_poses, ground_truth = {}, {}
        true_position, travelled = np.zeros(3), 0.0
        for index, position in enumerate(positions):
            if index:
                step = position - positions[index - 1]
                true_position = true_position + rotation_z(0.0005 * travelled) @ step
                travelled += np.linalg.norm(step)
            timestamp = 1_000_000_000 + index * 50_000_000
            map_poses[timestamp] = (position, np.eye(3))
            ground_truth[timestamp] = (true_position, rotation_z(0.0005 * travelled))
        pairs = MODULE.associate(map_poses, ground_truth, 1_000)
        sources = np.asarray([map_poses[a][0] for a, _ in pairs])
        targets = np.asarray([ground_truth[b][0] for _, b in pairs])

        # Localizations 2 cm above the outbound map poses in the middle 20 m.
        query = {
            timestamp: (position + np.asarray([0.0, 0.0, 0.02]), orientation)
            for timestamp, (position, orientation) in map_poses.items()
            if position[1] == 0.0 and 10.0 <= position[0] <= 30.0
        }
        local = MODULE.evaluate("local", query, ground_truth, 1_000, (sources, targets, 5.0))
        self.assertEqual(local["scored"], len(query))
        self.assertAlmostEqual(local["translation_m"]["rmse"], 0.02, delta=0.005)
        self.assertLess(local["rotation_deg"]["max"], 0.2)
        alignment = MODULE.fit_to_ground_truth(map_poses, ground_truth, 1_000)
        fixed = MODULE.evaluate("fixed", query, ground_truth, 1_000, alignment)
        self.assertGreater(fixed["translation_m"]["rmse"], 0.1)

    def test_local_mode_scores_on_the_pass_facing_the_same_way(self) -> None:
        # Out 40 m along x and back 0.3 m to the side, facing back; the
        # heading drifts 0.2 deg per metre, so the two passes' map->GT
        # transforms differ by up to 0.3 m at the same place.
        positions = [(np.asarray([0.25 * i, 0.0, 0.0]), 0.0) for i in range(161)]
        positions += [(np.asarray([40.0 - 0.25 * i, 0.3, 0.0]), np.pi) for i in range(161)]
        map_poses, ground_truth, drift = {}, {}, {}
        true_position, travelled = np.zeros(3), 0.0
        for index, (position, yaw) in enumerate(positions):
            if index:
                step = position - positions[index - 1][0]
                true_position = true_position + rotation_z(0.0035 * travelled) @ step
                travelled += np.linalg.norm(step)
            timestamp = 1_000_000_000 + index * 50_000_000
            map_poses[timestamp] = (position, rotation_z(yaw))
            ground_truth[timestamp] = (true_position, rotation_z(0.0035 * travelled + yaw))
            drift[timestamp] = rotation_z(0.0035 * travelled)
        pairs = MODULE.associate(map_poses, ground_truth, 1_000)
        alignment = (
            np.asarray([map_poses[a][0] for a, _ in pairs]),
            np.asarray([ground_truth[b][0] for _, b in pairs]),
            5.0,
            np.asarray([ground_truth[b][1] for _, b in pairs]),
        )
        # Localizations on the way back, 0.16 m towards the outbound pass:
        # nearer to it than to their own.
        shift = np.asarray([0.0, -0.16, 0.0])
        query, reference = {}, {}
        for timestamp, (position, orientation) in map_poses.items():
            if position[1] == 0.3 and 10.0 <= position[0] <= 30.0:
                query[timestamp] = (position + shift, orientation)
                truth, truth_orientation = ground_truth[timestamp]
                reference[timestamp] = (truth + drift[timestamp] @ shift, truth_orientation)
        facing = MODULE.evaluate("local", query, reference, 1_000, alignment)
        self.assertEqual(facing["scored"], len(query))
        # (1.4 cm: the drift bends within the 10 m stretch.)
        self.assertLess(facing["translation_m"]["max"], 0.02)
        # Without the orientations the outbound pass is nearer and fits them.
        anywhere = MODULE.evaluate("local", query, reference, 1_000, alignment[:3])
        self.assertGreater(anywhere["translation_m"]["max"], 0.1)

    def test_inlier_filter_keeps_poses_accepted_at_the_threshold(self) -> None:
        estimate = MODULE.load_tum_seconds  # keep the parser's rounding in the loop
        with tempfile.TemporaryDirectory() as directory:
            tum = Path(directory) / "localization.tum"
            tum.write_text(
                "1403636579.963555584 0 0 0 0 0 0 1\n"
                "1403636580.013555456 1 0 0 0 0 0 1\n"
                "1403636580.063555584 2 0 0 0 0 0 1\n",
                encoding="utf-8",
            )
            per_frame = Path(directory) / "per_frame.csv"
            per_frame.write_text(
                "timestamp_ns,success,used_prior,keypoints,matches,inliers,inlier_ratio,latency_ms\n"
                "1403636579963555584,true,false,800,60,45,0.75,70\n"
                "1403636580013555456,true,false,800,40,22,0.55,70\n"
                "1403636580063555584,true,false,800,90,30,0.33,70\n"
                "1403636580113555456,false,false,800,10,4,0.40,70\n",
                encoding="utf-8",
            )
            kept = MODULE.filter_by_inliers(estimate(tum), per_frame, 30)
        self.assertEqual(sorted(position[0] for position, _ in kept.values()), [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
