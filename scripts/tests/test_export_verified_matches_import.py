import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_verified_matches_import import (
    build_matches_import_lines,
    parse_images_tsv,
    parse_pair_matches,
    parse_pairs_tsv,
)


class ParsePairsTsvTest(unittest.TestCase):
    def test_preserves_row_order_and_ignores_extra_columns(self):
        text = (
            "image_i\timage_j\traw_matches\taccepted_matches\te_inliers\tf_inliers\th_inliers\n"
            "10\t20\t5\t5\t5\t5\t4\n"
            "1\t2\t9\t9\t9\t9\t8\n"
        )
        self.assertEqual(parse_pairs_tsv(text), [(10, 20), (1, 2)])

    def test_rejects_unexpected_header(self):
        with self.assertRaises(ValueError):
            parse_pairs_tsv("wrong header\n1\t2\n")


class ParseImagesTsvTest(unittest.TestCase):
    def test_orders_by_index_not_by_row(self):
        text = "image_index\timage_name\tfeature_count\n1\tb.png\t8\n0\ta.png\t8\n"
        self.assertEqual(parse_images_tsv(text), ["a.png", "b.png"])

    def test_rejects_sparse_index_range(self):
        text = "image_index\timage_name\tfeature_count\n0\ta.png\t8\n2\tc.png\t8\n"
        with self.assertRaises(ValueError):
            parse_images_tsv(text)


class ParsePairMatchesTest(unittest.TestCase):
    def test_stops_at_footer(self):
        stdout = (
            "pair_match_index_i\tpair_match_index_j\n"
            "2\t9\n"
            "3\t6\n"
            "images=5000 verified_pairs=69 accepted_correspondences=4618\n"
            "verifier_config_hash=deadbeef verifier_config=\"mode=Full\"\n"
        )
        self.assertEqual(parse_pair_matches(stdout), [(2, 9), (3, 6)])

    def test_rejects_unexpected_header(self):
        with self.assertRaises(ValueError):
            parse_pair_matches("wrong\n1\t2\n")


class BuildMatchesImportLinesTest(unittest.TestCase):
    def test_matches_import_matches_file_layout(self):
        names = ["a.png", "b.png"]
        pairs = [(0, 1)]
        correspondences_by_pair = {(0, 1): [(3, 4), (5, 6)]}
        lines = build_matches_import_lines(names, pairs, correspondences_by_pair)
        self.assertEqual(
            lines,
            ["2", "a.png", "b.png", "1", "0 1 2", "3 4", "5 6"],
        )

    def test_empty_pair_writes_zero_count_and_no_correspondence_lines(self):
        lines = build_matches_import_lines(["a.png", "b.png"], [(0, 1)], {(0, 1): []})
        self.assertEqual(lines, ["2", "a.png", "b.png", "1", "0 1 0"])


if __name__ == "__main__":
    unittest.main()
