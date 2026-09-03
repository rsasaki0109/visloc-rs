import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_rig_metric_motion.py"
SPEC = importlib.util.spec_from_file_location("diagnose_rig_metric_motion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MetricMotionTests(unittest.TestCase):
    def write_fixture(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        model = root / "model"
        model.mkdir()
        (model / "cameras.txt").write_text(
            "1 PINHOLE 100 100 100 100 0 0\n"
            "2 PINHOLE 100 100 100 100 0 0\n",
            encoding="utf-8",
        )
        world_points = [(0.0, 0.0, 5.0), (0.0, 1.0, 5.0), (1.0, 0.0, 4.0)]
        images = [
            (1, 1, "left0.png", 0.0),
            (2, 2, "right0.png", 0.1),
            (3, 1, "left1.png", 1.0),
            (4, 2, "right1.png", 1.1),
        ]
        image_lines = []
        for image_id, camera_id, name, center_x in images:
            image_lines.append(
                f"{image_id} 1 0 0 0 {-center_x} 0 0 {camera_id} {name}\n"
            )
            projected = []
            for point_id, (x, y, z) in enumerate(world_points, 1):
                projected.extend([str(100 * (x - center_x) / z), str(100 * y / z), str(point_id)])
            image_lines.append(" ".join(projected) + "\n")
        (model / "images.txt").write_text("".join(image_lines), encoding="utf-8")
        point_lines = []
        for point_id, point in enumerate(world_points, 1):
            track = " ".join(f"{image_id} {point_id - 1}" for image_id in range(1, 5))
            point_lines.append(
                f"{point_id} {point[0]} {point[1]} {point[2]} 255 255 255 0 {track}\n"
            )
        (model / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")
        manifest = root / "rig-manifest.txt"
        manifest.write_text(
            "S 0 1 100 100 100 100 0 0 1 0 0 0 0 0 0\n"
            "S 1 2 100 100 100 100 0 0 1 0 0 0 -0.1 0 0\n"
            "F 0 left0.png 0\n"
            "F 0 right0.png 1\n"
            "F 1 left1.png 0\n"
            "F 1 right1.png 1\n",
            encoding="utf-8",
        )
        return model, manifest

    def test_recovers_metric_translation_without_ground_truth(self):
        model, manifest = self.write_fixture()
        result = MODULE.diagnose(
            model,
            manifest,
            None,
            min_frame_gap=1,
            max_frame_gap=1,
            min_correspondences=3,
            min_inliers=3,
            min_angle_deg=0.5,
            ransac_threshold_m=1e-6,
            ransac_iterations=16,
            bin_size=10,
        )
        self.assertFalse(result["ground_truth_used"])
        self.assertEqual(result["accepted_frame_pairs"], 1)
        self.assertAlmostEqual(
            result["all_pairs"]["metric_to_mapper_translation_ratio"]["median"], 1.0, places=8
        )
        self.assertAlmostEqual(
            result["all_pairs"]["translation_direction_cosine"]["median"], 1.0, places=8
        )
        self.assertAlmostEqual(
            result["all_pairs"]["rotation_error_deg"]["median"], 0.0, places=5
        )
        self.assertEqual(result["pose_consistent_pairs"]["pair_count"], 1)
        self.assertEqual(
            result["pose_consistent_frame_bins"]["0-9"]["pair_count"], 1
        )


if __name__ == "__main__":
    unittest.main()
