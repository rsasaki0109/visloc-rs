import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parents[1] / "scripts" / "audit_colmap_pinhole_model.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ModelAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "cameras.txt").write_text("1 PINHOLE 100 100 10 10 0 0\n")
        # Two distinct errors (0.1, 0.3) distinguish arithmetic mean from RMS.
        (self.root / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 1 a.png\n0.1 0 1\n"
            "2 1 0 0 0 -1 0 0 1 b.png\n-1.7 0 1\n"
            "3 1 0 0 0 -2 0 0 1 c.png\n\n")
        (self.root / "points3D.txt").write_text("1 0 0 5 255 255 255 0.2 1 0 2 0\n")

    def test_direct_projection_mean_and_support(self):
        report = AUDIT.audit(self.root)
        self.assertAlmostEqual(report["mean_reprojection_px"], 0.2)
        self.assertAlmostEqual(report["max_stored_mean_difference_px"], 0)
        self.assertEqual(report["unsupported_image_ids"], [3])
        self.assertEqual(report["observations"], 2)

    def test_reverse_reference_required(self):
        path = self.root / "images.txt"
        path.write_text(path.read_text().replace("0.1 0 1", "0.1 0 9"))
        with self.assertRaisesRegex(ValueError, "reference mismatch"):
            AUDIT.audit(self.root)

    def test_orphan_image_reference_required(self):
        path = self.root / "images.txt"
        path.write_text(path.read_text().replace("c.png\n\n", "c.png\n0 0 9\n"))
        with self.assertRaisesRegex(ValueError, "image-to-point"):
            AUDIT.audit(self.root)

    def test_duplicate_observation_rejected(self):
        path = self.root / "points3D.txt"
        path.write_text(path.read_text().replace("1 0 2 0", "1 0 1 0"))
        with self.assertRaisesRegex(ValueError, "duplicate or invalid"):
            AUDIT.audit(self.root)

    def test_distinct_same_image_keypoints_reported(self):
        path = self.root / "images.txt"
        path.write_text(path.read_text().replace("0.1 0 1\n", "0.1 0 1 0.3 0 1\n"))
        path = self.root / "points3D.txt"
        path.write_text(path.read_text().replace("1 0 2 0", "1 0 1 1 2 0"))
        report = AUDIT.audit(self.root)
        self.assertEqual(report["tracks_with_multiple_keypoints_in_one_image"], 1)
        self.assertEqual(report["excess_same_image_track_observations"], 1)

    def test_unsupported_distortion_rejected(self):
        (self.root / "cameras.txt").write_text("1 SIMPLE_RADIAL 100 100 10 0 0 0.1\n")
        with self.assertRaisesRegex(ValueError, "PINHOLE required"):
            AUDIT.audit(self.root)

    def test_nonfinite_rejected(self):
        path = self.root / "points3D.txt"
        path.write_text(path.read_text().replace("1 0 0 5", "1 nan 0 5"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            AUDIT.audit(self.root)

    def rig_manifest(self):
        path = self.root / "rig.txt"
        path.write_text(
            "S 0 1 100 100 10 10 0 0 1 0 0 0 0 0 0\n"
            "S 1 1 100 100 10 10 0 0 1 0 0 0 -1 0 0\n"
            "F 10 a.png 0\nF 10 b.png 1\nF 20 c.png 0\n")
        return path

    def test_rig_support_and_fixed_baseline(self):
        report = AUDIT.audit(self.root, self.rig_manifest())["rig"]
        self.assertEqual(report["rig_frames"], 2)
        self.assertEqual(report["supported_rig_frames"], 1)
        self.assertEqual(report["unsupported_rig_frame_ids"], [20])
        self.assertEqual(report["track_connected_component_sizes"], [1, 1])
        self.assertAlmostEqual(report["max_inferred_rig_center_disagreement_m"], 0)

    def test_supported_frames_can_still_be_track_disconnected(self):
        manifest = self.rig_manifest()
        manifest.write_text(manifest.read_text() + "F 30 d.png 0\n")
        path = self.root / "images.txt"
        path.write_text(path.read_text().replace("c.png\n\n", "c.png\n-4 0 2\n")
                        + "4 1 0 0 0 -3 0 0 1 d.png\n-6 0 2\n")
        path = self.root / "points3D.txt"
        path.write_text(path.read_text() + "2 0 0 5 255 255 255 0 3 0 4 0\n")
        report = AUDIT.audit(self.root, manifest)["rig"]
        self.assertEqual(report["supported_rig_frames"], 3)
        self.assertEqual(report["track_connected_components"], 2)
        self.assertEqual(report["track_connected_component_sizes"], [2, 1])

    def test_changed_stereo_baseline_rejected(self):
        manifest = self.rig_manifest()
        path = self.root / "images.txt"
        path.write_text(path.read_text().replace("-1 0 0 1 b.png", "-1.1 0 0 1 b.png"))
        with self.assertRaisesRegex(ValueError, "fixed sensor extrinsics"):
            AUDIT.audit(self.root, manifest)

    def test_rotated_sensor_extrinsic_composition(self):
        manifest = self.rig_manifest()
        manifest.write_text(manifest.read_text().replace(
            "0 0 1 0 0 0 -1 0 0", "0 0 0.7071067811865476 0 0 0.7071067811865476 -1 0 0"))
        path = self.root / "images.txt"
        path.write_text(path.read_text().replace(
            "2 1 0 0 0 -1", "2 0.7071067811865476 0 0 0.7071067811865476 -1"))
        report = AUDIT.audit(self.root, manifest)["rig"]
        self.assertAlmostEqual(report["max_inferred_rig_center_disagreement_m"], 0)
        self.assertLess(report["max_inferred_rig_rotation_disagreement_deg"], 1e-5)

    def test_rig_intrinsics_mismatch_rejected(self):
        manifest = self.rig_manifest()
        manifest.write_text(manifest.read_text().replace("100 100 10 10", "100 100 11 10"))
        with self.assertRaisesRegex(ValueError, "rig camera calibration"):
            AUDIT.audit(self.root, manifest)

    def test_missing_rig_name_rejected(self):
        manifest = self.rig_manifest()
        manifest.write_text(manifest.read_text().replace("c.png", "missing.png"))
        with self.assertRaisesRegex(ValueError, "absent from rig manifest"):
            AUDIT.audit(self.root, manifest)


if __name__ == "__main__":
    unittest.main()
