import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_colmap_observation_filter.py"
SPEC = importlib.util.spec_from_file_location("observation_filter_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FilterAuditTests(unittest.TestCase):
    def run_audit(self, old, new):
        with tempfile.TemporaryDirectory() as directory:
            before = Path(directory) / "before.txt"
            after = Path(directory) / "after.txt"
            before.write_text(old, encoding="utf-8")
            after.write_text(new, encoding="utf-8")
            return MODULE.audit(before, after)

    @staticmethod
    def model(keys, pose="0", second=""):
        return f"# images\n1 1 0 0 0 {pose} 0 0 1 cam.png\n{keys}\n{second}"

    def test_renumbering_and_deletion_are_counted(self):
        old = self.model("10 20 4 30 40 8 50 60 -1")
        new = self.model("10 20 1 30 40 -1 50 60 -1", pose="0.1")
        result = self.run_audit(old, new)
        self.assertEqual(result["removed_observations"], 1)
        self.assertEqual(result["removed_points"], 1)
        self.assertEqual(result["changed_pose_images"], 1)
        self.assertEqual(result["supported_images_after"], 1)

    def test_support_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lost support"):
            self.run_audit(self.model("10 20 4"), self.model("10 20 -1"))

    def test_added_observation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "added"):
            self.run_audit(self.model("10 20 -1"), self.model("10 20 1"))

    def test_coordinate_or_count_changes_are_rejected(self):
        for new in ["10 21 4", "10 20 4 30 40 -1"]:
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, "keypoint"):
                self.run_audit(self.model("10 20 4"), self.model(new))

    def test_merge_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "merged"):
            self.run_audit(self.model("10 20 4 30 40 8"), self.model("10 20 1 30 40 1"))

    def test_split_is_rejected_across_images(self):
        second_old = "2 1 0 0 0 0 0 0 1 other.png\n30 40 4\n"
        second_new = "2 1 0 0 0 0 0 0 1 other.png\n30 40 2\n"
        with self.assertRaisesRegex(ValueError, "split"):
            self.run_audit(self.model("10 20 4", second=second_old),
                           self.model("10 20 1", second=second_new))

    def test_empty_keypoint_row_and_missing_row(self):
        self.assertEqual(self.run_audit(self.model(""), self.model(""))["images"], 1)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_audit(self.model(""), self.model("").rstrip() + "\n")


if __name__ == "__main__":
    unittest.main()
