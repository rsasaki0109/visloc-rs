"""Scoped tests for the bounded observation-geometry diagnostic."""

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "audit_ba_geometry_bins",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_ba_geometry_bins.py",
)
DIAG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DIAG
SPEC.loader.exec_module(DIAG)


def _project(center, point):
    z = point[2] - center[2]
    return 100.0 * (point[0] - center[0]) / z + 320.0, 100.0 * (point[1] - center[1]) / z + 240.0


def _write_model(path, centers, points, observations, errors=None):
    path.mkdir()
    (path / "cameras.txt").write_text("1 PINHOLE 640 480 100 100 320 240\n")
    image_lines = []
    for image_id, center in enumerate(centers, 1):
        image_lines.append(
            f"{image_id} 1 0 0 0 {-center[0]:.17g} {-center[1]:.17g} {-center[2]:.17g} 1 im{image_id}.png\n"
        )
        image_lines.append(
            " ".join(
                f"{observations[image_id - 1][point_id][0]:.17g} "
                f"{observations[image_id - 1][point_id][1]:.17g} {point_id}"
                for point_id in points
            )
            + "\n"
        )
    (path / "images.txt").write_text("".join(image_lines))
    point_lines = []
    for point_index, (point_id, xyz) in enumerate(points.items()):
        error = 0.0 if errors is None else errors.get(point_id, 0.0)
        track = " ".join(
            f"{image_id} {point_index}" for image_id in range(1, len(centers) + 1)
        )
        point_lines.append(
            f"{point_id} {' '.join(f'{value:.17g}' for value in xyz)} 1 2 3 {error:.17g} {track}\n"
        )
    (path / "points3D.txt").write_text("".join(point_lines))


def _make_pair(root, candidate_points=None, candidate_centers=None):
    centers = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    points = {1: (0.0, 0.0, 5.0), 2: (0.2, 0.1, 5.0)}
    targets = {1: (0.1, 0.0, 5.0), 2: (0.2, 0.1, 5.0)}
    observations = [
        {point_id: _project(center, target) for point_id, target in targets.items()}
        for center in centers
    ]
    initial = root / "initial"
    candidate = root / "candidate"
    _write_model(initial, centers, points, observations)
    _write_model(
        candidate,
        centers if candidate_centers is None else candidate_centers,
        points if candidate_points is None else candidate_points,
        observations,
    )
    return initial, candidate, centers, points, targets


class GeometryBinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_pair_angle_uses_maximum_and_normalizes_acute_obtuse(self):
        point = (0.0, 0.0, 1.0)
        centers = [(0.0, 0.0, 0.0), (0.001, 0.0, 0.0), (1.0, 0.0, 0.0)]
        angle = DIAG._track_angle(point, centers)
        self.assertAlmostEqual(angle, 45.0, places=8)
        self.assertEqual(
            DIAG._track_angle((0.0, 0.0, 0.0), [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]),
            0.0,
        )
        self.assertEqual(DIAG._track_angle((0.0, 0.0, 0.0), [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]), 90.0)
        self.assertEqual(DIAG._track_angle((0.0, 0.0, 0.0), [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]), 0.0)
        with self.assertRaises(DIAG.DiagnosticError):
            DIAG._track_angle(point, [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0)])

    def test_bin_boundaries_are_half_open(self):
        self.assertEqual(DIAG._bin_index(2, 0.0), (0, 0))
        self.assertEqual(DIAG._bin_index(3, 0.1), (0, 1))
        self.assertEqual(DIAG._bin_index(4, 1.0), (1, 2))
        self.assertEqual(DIAG._bin_index(16, 5.0), (2, 3))
        self.assertEqual(DIAG._bin_index(17, 90.0), (3, 3))

    def test_report_separates_cost_reduction_and_increase(self):
        targets = {1: (0.1, 0.0, 5.0), 2: (0.2, 0.1, 5.0)}
        initial, candidate, _, points, targets = _make_pair(
            self.root,
            candidate_points={1: targets[1], 2: (0.45, 0.1, 5.0)},
        )
        report = DIAG.diagnose(initial, candidate, "synthetic", include_runtime=False)
        self.assertEqual(len(report["cells"]), 16)
        self.assertEqual(report["global"]["points"], 2)
        self.assertEqual(report["global"]["observations"], 6)
        self.assertGreater(report["global"]["cost_reduction"], 0.0)
        self.assertGreater(report["global"]["cost_increase"], 0.0)
        self.assertEqual(report["membership"]["initial_only"], True)
        self.assertEqual(report["initial_counts"]["w2"], 18)
        self.assertEqual(report["initial_counts"]["pairs"], 6)
        self.assertEqual(report["initial_counts"]["sum_track_lengths"], 6)

    def test_membership_is_initial_only_and_camera_motion_is_observation_weighted(self):
        points = {1: (0.0, 0.0, 5.0), 2: (0.2, 0.1, 5.0)}
        initial, candidate, centers, points, _ = _make_pair(
            self.root,
            candidate_points={1: (0.1, 0.0, 50.0), 2: points[2]},
            candidate_centers=[(0.0, 0.0, 0.0), (1.25, 0.0, 0.0), (2.0, 0.0, 0.0)],
        )
        report = DIAG.diagnose(initial, candidate, "motion", include_runtime=False)
        self.assertGreater(report["global"]["landmark_displacement_sum_m"], 0.0)
        self.assertGreater(report["global"]["camera_center_motion_observation_mean_m"], 0.0)
        self.assertGreater(report["global"]["image_weighted_camera_center_motion"]["mean_m"], 0.0)
        self.assertEqual(sum(cell["points"] for cell in report["cells"]), 2)
        self.assertEqual(sum(cell["observations"] for cell in report["cells"]), 6)
        expected_motion = 2.0 * 0.25 / 6.0
        self.assertAlmostEqual(
            report["global"]["camera_center_motion_observation_mean_m"],
            expected_motion,
            places=14,
        )
        self.assertAlmostEqual(
            report["global"]["image_weighted_camera_center_motion"]["sum_m"],
            0.25,
            places=14,
        )
        self.assertAlmostEqual(
            report["global"]["image_weighted_camera_center_motion"]["mean_m"],
            0.25 / 3.0,
            places=14,
        )
        self_control = DIAG.diagnose(initial, initial, "self-control")
        membership_fields = ("k_stratum", "angle_bin_deg", "points", "observations")
        self.assertEqual(
            [tuple(cell[field] for field in membership_fields) for cell in report["cells"]],
            [tuple(cell[field] for field in membership_fields) for cell in self_control["cells"]],
        )

    def test_repeat_is_deterministic_without_runtime_metadata(self):
        targets = {1: (0.1, 0.0, 5.0), 2: (0.2, 0.1, 5.0)}
        initial, candidate, _, _, targets = _make_pair(
            self.root, candidate_points={1: targets[1], 2: (0.4, 0.1, 5.0)}
        )
        first = DIAG.diagnose(initial, candidate, "repeat", include_runtime=False)
        second = DIAG.diagnose(initial, candidate, "repeat", include_runtime=False)
        self.assertEqual(first, second)
        self.assertNotIn("runtime", first)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_reconciliation_rejects_missing_bin_contribution(self):
        cells = DIAG._empty_cells()
        direct = DIAG._new_totals()
        direct["points"] = 1
        direct["observations"] = 2
        cells[(0, 0)]["points"] = 1
        cells[(0, 0)]["observations"] = 2
        self.assertIsNotNone(DIAG._reconcile(direct, cells, 1, 2))
        cells[(0, 0)]["observations"] = 1
        with self.assertRaises(DIAG.DiagnosticError):
            DIAG._reconcile(direct, cells, 1, 2)

    def test_identity_depth_and_symlink_fail_closed(self):
        initial, candidate, centers, points, targets = _make_pair(self.root)
        valid = DIAG.diagnose(initial, candidate, "valid")
        self.assertEqual(valid["global"]["observations"], 6)
        image_path = candidate / "images.txt"
        original = image_path.read_text()
        image_path.write_text(original.replace("322", "323", 1))
        with self.assertRaisesRegex(DIAG.DiagnosticError, "keypoint association|coordinate"):
            DIAG.diagnose(initial, candidate, "identity")
        image_path.write_text(original)
        _write_model(self.root / "bad", centers, {1: (0.0, 0.0, -1.0), 2: points[2]}, [
            {point_id: _project(center, target) for point_id, target in targets.items()}
            for center in centers
        ])
        with self.assertRaises(DIAG.DiagnosticError):
            DIAG.diagnose(initial, self.root / "bad", "depth")
        link = self.root / "link"
        link.symlink_to(initial, target_is_directory=True)
        with self.assertRaises(DIAG.DiagnosticError):
            DIAG.diagnose(link, candidate, "symlink")

    def test_all_caps_are_rejected_before_pair_traversal(self):
        initial, candidate, _, _, _ = _make_pair(self.root)
        caps = {
            "MAX_CAMERAS": 0,
            "MAX_IMAGES": 0,
            "MAX_POINTS": 0,
            "MAX_OBSERVATIONS": 0,
            "MAX_TOTAL_KEYPOINTS": 0,
            "MAX_TRACK": 2,
            "MAX_FILE_BYTES": 1,
            "MAX_MODEL_BYTES": 1,
            "MAX_LINE_BYTES": 1,
            "MAX_CROSS_WORK": 0,
        }
        originals = {name: getattr(DIAG, name) for name in caps}
        try:
            for name, value in caps.items():
                setattr(DIAG, name, value)
                with self.subTest(cap=name):
                    with self.assertRaises(DIAG.DiagnosticError):
                        DIAG.diagnose(initial, candidate, "cap")
                setattr(DIAG, name, originals[name])
        finally:
            for name, value in originals.items():
                setattr(DIAG, name, value)

    def test_frozen_count_gate_is_explicit(self):
        initial, _, _, _, _ = _make_pair(self.root)
        preflight = DIAG._preflight_model(initial)
        with self.assertRaisesRegex(DIAG.DiagnosticError, "frozen input counts"):
            DIAG._require_frozen_counts(preflight, "synthetic")

    def test_norm_perturbed_unit_quaternion_is_normalized_for_centres(self):
        initial, candidate, _, _, _ = _make_pair(self.root)
        image_path = candidate / "images.txt"
        text = image_path.read_text()
        image_path.write_text(text.replace("1 0 0 0", "1.0000000000000002 0 0 0", 1))
        report = DIAG.diagnose(initial, candidate, "norm-perturbation")
        self.assertEqual(report["global"]["image_weighted_camera_center_motion"]["max_m"], 0.0)

    def test_rigid_transform_preserves_angle(self):
        rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        shift = (3.0, -2.0, 4.0)

        def transform(value):
            rotated = DIAG._mat_vec(rotation, value)
            return tuple(rotated[index] + shift[index] for index in range(3))

        point = (0.2, -0.1, 5.0)
        centers = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        self.assertAlmostEqual(
            DIAG._track_angle(point, centers),
            DIAG._track_angle(transform(point), [transform(center) for center in centers]),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
