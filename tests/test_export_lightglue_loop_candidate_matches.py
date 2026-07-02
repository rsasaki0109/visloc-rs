from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_lightglue_loop_candidate_matches import (  # noqa: E402
    feature_path,
    parse_image_size,
    read_candidates,
    read_features,
    read_stereo_left_indices,
    stereo_matches_path,
)


class ExportLightGlueLoopCandidateMatchesTest(unittest.TestCase):
    def test_parse_image_size_accepts_common_forms(self) -> None:
        self.assertEqual(parse_image_size("1241x376"), (1241, 376))
        self.assertEqual(parse_image_size("1241,376"), (1241, 376))
        with self.assertRaises(Exception):
            parse_image_size("bad")

    def test_read_candidates_filters_attempted_and_orders_by_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loop_candidate_verifications.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "matched_keyframe_id",
                        "query_frame_id",
                        "score",
                        "attempted",
                        "verified",
                        "failure_reason",
                    ]
                )
                writer.writerow([10, 100, "0.4", "false", "false", "not_attempted"])
                writer.writerow([20, 120, "0.9", "true", "false", "TooFewInliers"])
                writer.writerow([21, 121, "0.7", "true", "true", ""])

            attempted = read_candidates(path, "attempted", None)
            self.assertEqual([(row.older, row.newer) for row in attempted], [(20, 120), (21, 121)])

            failed = read_candidates(path, "failed", None)
            self.assertEqual([(row.older, row.newer) for row in failed], [(20, 120)])

            high_score = read_candidates(path, "all", 0.8)
            self.assertEqual([(row.older, row.newer) for row in high_score], [(20, 120)])

    def test_read_features_truncates_and_validates_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame_000001_left_features.txt"
            path.write_text(
                "# X Y SCORE D0 D1\n"
                "1 2 0.9 0.1 0.2\n"
                "3 4 0.8 0.3 0.4\n",
                encoding="utf-8",
            )
            features = read_features(path, max_keypoints=1)
            self.assertEqual(features.keypoints, [(1.0, 2.0)])
            self.assertEqual(features.scores, [0.9])
            self.assertEqual(features.descriptors, [[0.1, 0.2]])

    def test_stereo_indices_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = stereo_matches_path(root, 7)
            self.assertEqual(path.name, "frame_000007_stereo_matches.txt")
            self.assertEqual(feature_path(root, 7).name, "frame_000007_left_features.txt")
            path.write_text("# q t c\n4 8 0.5\n9 10 0.4\n", encoding="utf-8")
            self.assertEqual(read_stereo_left_indices(path), {4, 9})


if __name__ == "__main__":
    unittest.main()
