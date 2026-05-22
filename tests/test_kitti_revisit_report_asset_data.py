from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kitti_revisit_report_asset_data import (  # noqa: E402
    find_overlay_svg,
    find_pair_images,
    first_summary_int,
    parse_svg_lines,
    read_candidates,
    read_summary,
    strongest_candidate,
)


class KittiRevisitReportAssetDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.report_dir = REPO_ROOT / "target" / "kitti_revisit_report_asset_data_unit"
        if self.report_dir.exists():
            shutil.rmtree(self.report_dir)
        (self.report_dir / "assets").mkdir(parents=True)

    def tearDown(self) -> None:
        if self.report_dir.exists():
            shutil.rmtree(self.report_dir)

    def test_read_summary_extracts_key_value_tokens(self) -> None:
        summary = self.report_dir / "summary.txt"
        summary.write_text(
            "segment_a_frames=50 segment_b_frames=30\nignored-token max_features=200\n",
            encoding="utf-8",
        )

        values = read_summary(summary)

        self.assertEqual(values["segment_a_frames"], "50")
        self.assertEqual(values["segment_b_frames"], "30")
        self.assertEqual(first_summary_int(values, "max_features", 100), 200)
        self.assertEqual(first_summary_int(values, "missing", 100), 100)

    def test_read_candidates_converts_numeric_fields_and_selects_strongest(self) -> None:
        csv_path = self.report_dir / "candidates.csv"
        csv_path.write_text(
            "matched_keyframe_id,query_frame_id,matches,inliers,inlier_ratio,mean_sampson_error,score\n"
            "48,4500,90,50,0.55,0.002,10.0\n"
            "49,4501,95,57,0.60,0.003,42.0\n",
            encoding="utf-8",
        )

        rows = read_candidates(csv_path)
        strongest = strongest_candidate(rows)

        self.assertEqual(rows[0]["matched_keyframe_id"], 48)
        self.assertEqual(rows[1]["inlier_ratio"], 0.60)
        self.assertEqual(strongest["query_frame_id"], 4501)

    def test_strongest_candidate_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(SystemExit, "no accepted candidates"):
            strongest_candidate([])

    def test_find_pair_images_and_overlay_use_candidate_frame_ids(self) -> None:
        candidate = {"matched_keyframe_id": 49, "query_frame_id": 4501}
        assets = self.report_dir / "assets"
        from_path = assets / "deep_from_49.png"
        to_path = assets / "deep_to_4501.png"
        svg_path = assets / "deep_matches_49_4501.svg"
        from_path.write_text("from", encoding="utf-8")
        to_path.write_text("to", encoding="utf-8")
        svg_path.write_text('<line x1="1.5" y1="2" x2="27" y2="4.25" />', encoding="utf-8")

        self.assertEqual(find_pair_images(self.report_dir, candidate), (from_path, to_path))
        self.assertEqual(find_overlay_svg(self.report_dir, candidate), svg_path)
        self.assertEqual(parse_svg_lines(svg_path), [(1.5, 2.0, 27.0, 4.25)])

    def test_find_pair_images_reports_missing_asset_context(self) -> None:
        candidate = {"matched_keyframe_id": 49, "query_frame_id": 4501}

        with self.assertRaisesRegex(SystemExit, "49 -> 4501"):
            find_pair_images(self.report_dir, candidate)


if __name__ == "__main__":
    unittest.main()
