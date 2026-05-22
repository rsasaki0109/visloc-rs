from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kitti_revisit_report_asset_render import (  # noqa: E402
    ReportAssetInputs,
    load_report_asset_inputs,
    render_asset_image,
    save_asset_image,
)


def non_background_pixels(image: Image.Image) -> int:
    background = (248, 250, 252)
    return sum(
        1
        for y in range(image.height)
        for x in range(image.width)
        if image.getpixel((x, y)) != background
    )


def candidate(score: float = 16083.0719) -> dict[str, object]:
    return {
        "matched_keyframe_id": 49,
        "query_frame_id": 4501,
        "matches": 95,
        "inliers": 57,
        "inlier_ratio": 0.6,
        "mean_sampson_error": 0.00213,
        "score": score,
    }


class KittiRevisitReportAssetRenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.report_dir = REPO_ROOT / "target" / "kitti_revisit_report_asset_render_unit"
        if self.report_dir.exists():
            shutil.rmtree(self.report_dir)
        (self.report_dir / "assets").mkdir(parents=True)

    def tearDown(self) -> None:
        if self.report_dir.exists():
            shutil.rmtree(self.report_dir)

    def test_render_asset_image_returns_rgb_canvas_without_file_io(self) -> None:
        inputs = ReportAssetInputs(
            summary={"segment_a_frames": "50", "segment_b_frames": "30", "max_features": "200"},
            candidates=[candidate()],
            strongest=candidate(),
            from_image=Image.new("RGB", (80, 40), (180, 0, 0)),
            to_image=Image.new("RGB", (80, 40), (0, 0, 180)),
            overlay_lines=[(10.0, 10.0, 114.0, 10.0)],
        )

        image = render_asset_image(inputs, 600)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (600, 405))
        self.assertGreater(non_background_pixels(image), 1000)

    def test_load_report_asset_inputs_reads_report_files_and_images(self) -> None:
        (self.report_dir / "summary.txt").write_text(
            "segment_a_frames=50 segment_b_frames=30 max_features=200\n",
            encoding="utf-8",
        )
        (self.report_dir / "candidates.csv").write_text(
            "matched_keyframe_id,query_frame_id,matches,inliers,inlier_ratio,mean_sampson_error,score\n"
            "48,4500,90,50,0.55,0.002,10.0\n"
            "49,4501,95,57,0.60,0.003,42.0\n",
            encoding="utf-8",
        )
        Image.new("RGB", (80, 40), (180, 0, 0)).save(
            self.report_dir / "assets" / "deep_from_49.png"
        )
        Image.new("RGB", (80, 40), (0, 0, 180)).save(
            self.report_dir / "assets" / "deep_to_4501.png"
        )
        (self.report_dir / "assets" / "deep_matches_49_4501.svg").write_text(
            '<line x1="1" y1="2" x2="105" y2="4" />',
            encoding="utf-8",
        )

        inputs = load_report_asset_inputs(self.report_dir)

        self.assertEqual(inputs.strongest["matched_keyframe_id"], 49)
        self.assertEqual(inputs.from_image.mode, "RGB")
        self.assertEqual(inputs.to_image.size, (80, 40))
        self.assertEqual(inputs.overlay_lines, [(1.0, 2.0, 105.0, 4.0)])

    def test_save_asset_image_creates_parent_directories(self) -> None:
        out_path = self.report_dir / "nested" / "asset.jpg"

        save_asset_image(Image.new("RGB", (20, 10), (10, 20, 30)), out_path, quality=85)

        with Image.open(out_path) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (20, 10))


if __name__ == "__main__":
    unittest.main()
