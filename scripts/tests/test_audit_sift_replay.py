import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "audit_sift_replay", Path(__file__).resolve().parents[1] / "audit_sift_replay.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReplayAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.images, self.reference, self.replay = [root / name for name in
                                                  ("images", "reference", "replay")]
        for directory in (self.images, self.reference, self.replay):
            directory.mkdir()
        (self.images / "camera.png").write_bytes(b"image")
        for directory in (self.reference, self.replay):
            (directory / "camera_features.txt").write_bytes(b"features")
            (directory / "camera_loci.txt").write_bytes(b"loci")

    def audit(self):
        return MODULE.audit(self.images, self.reference, self.replay, 1)

    def test_equal_with_reference_superset(self):
        (self.reference / "other_features.txt").write_bytes(b"other shard")
        result = self.audit()
        self.assertTrue(result["passed"])
        self.assertEqual(result["replay_bytes"], 12)

    def test_same_size_corruption(self):
        (self.replay / "camera_features.txt").write_bytes(b"Features")
        self.assertEqual(self.audit()["mismatches"], ["camera_features.txt"])

    def test_missing_and_extra(self):
        (self.replay / "camera_loci.txt").rename(self.replay / "unexpected.txt")
        result = self.audit()
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing"], ["camera_loci.txt"])
        self.assertEqual(result["extra"], ["unexpected.txt"])

    def test_missing_reference(self):
        (self.reference / "camera_loci.txt").unlink()
        self.assertFalse(self.audit()["passed"])

    def test_directory_not_output_file(self):
        (self.replay / "camera_loci.txt").unlink()
        (self.replay / "camera_loci.txt").mkdir()
        self.assertFalse(self.audit()["passed"])

    def test_count_and_stem_collision(self):
        with self.assertRaises(ValueError):
            MODULE.audit(self.images, self.reference, self.replay, 2)
        (self.images / "camera.jpg").write_bytes(b"image")
        with self.assertRaises(ValueError):
            MODULE.audit(self.images, self.reference, self.replay, 2)


if __name__ == "__main__":
    unittest.main()
