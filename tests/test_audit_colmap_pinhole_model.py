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

    def test_same_image_track_conflict(self):
        path = self.root / "points3D.txt"
        path.write_text(path.read_text().replace("1 0 2 0", "1 0 1 0"))
        with self.assertRaisesRegex(ValueError, "duplicate observation"):
            AUDIT.audit(self.root)

    def test_unsupported_distortion_rejected(self):
        (self.root / "cameras.txt").write_text("1 SIMPLE_RADIAL 100 100 10 0 0 0.1\n")
        with self.assertRaisesRegex(ValueError, "PINHOLE required"):
            AUDIT.audit(self.root)

    def test_nonfinite_rejected(self):
        path = self.root / "points3D.txt"
        path.write_text(path.read_text().replace("1 0 0 5", "1 nan 0 5"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            AUDIT.audit(self.root)


if __name__ == "__main__":
    unittest.main()
