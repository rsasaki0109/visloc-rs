import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "merge_sift_supplements", Path(__file__).resolve().parents[1] / "merge_sift_supplements.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeTests(unittest.TestCase):
    def test_preserves_prefix_and_supplement_variants(self):
        base = [b"# header\n", b"0 0 base\n"]
        additions = [b"1 0 boundary\n", b"1.01 0 first\n", b"1.01 0 second\n"]
        self.assertEqual(list(MODULE.merged_rows(base, additions)),
                         [base[1], additions[1], additions[2]])

    def test_neighbor_cells_and_negative_coordinates(self):
        self.assertEqual(list(MODULE.merged_rows([b"-0.1 -0.1 base\n"],
                                                [b"0.1 0.1 near\n"])), [b"-0.1 -0.1 base\n"])

    def test_empty_base_keeps_all_supplements(self):
        rows = [b"2 3 a\n", b"2 3 b\n"]
        self.assertEqual(list(MODULE.merged_rows([], rows)), rows)

    def test_bad_coordinates_rejected(self):
        for row in (b"nan 0\n", b"0 inf\n", b"1\n"):
            with self.assertRaises(ValueError):
                list(MODULE.merged_rows([], [row]))


if __name__ == "__main__":
    unittest.main()
