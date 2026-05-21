from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kitti_revisit_report_asset_canvas import draw_badge, draw_verified_overlay, resize_fit  # noqa: E402


def nonwhite_pixels(image: Image.Image) -> int:
    return sum(
        1
        for y in range(image.height)
        for x in range(image.width)
        if image.getpixel((x, y)) != (255, 255, 255)
    )


class KittiRevisitReportAssetCanvasTest(unittest.TestCase):
    def test_resize_fit_preserves_aspect_ratio_and_centers(self) -> None:
        image = Image.new("RGB", (100, 50), (255, 0, 0))

        resized, offset = resize_fit(image, (80, 80))

        self.assertEqual(resized.size, (80, 40))
        self.assertEqual(offset, (0, 20))

    def test_draw_badge_modifies_canvas(self) -> None:
        canvas = Image.new("RGB", (240, 80), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        draw_badge(draw, (10, 10), "57 inliers", ImageFont.load_default(), (15, 118, 110))

        self.assertGreater(nonwhite_pixels(canvas), 100)
        self.assertEqual(canvas.getpixel((30, 25)), (15, 118, 110))

    def test_draw_verified_overlay_renders_pair_images_and_match_line(self) -> None:
        canvas = Image.new("RGB", (260, 120), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        from_img = Image.new("RGB", (40, 20), (200, 0, 0))
        to_img = Image.new("RGB", (40, 20), (0, 0, 200))
        candidate = {
            "matched_keyframe_id": 49,
            "query_frame_id": 4501,
            "inliers": 1,
            "matches": 2,
        }

        draw_verified_overlay(
            canvas,
            draw,
            from_img,
            to_img,
            [(10.0, 10.0, 74.0, 10.0)],
            (10, 10),
            (220, 70),
            ImageFont.load_default(),
            ImageFont.load_default(),
            candidate,
        )

        self.assertGreater(nonwhite_pixels(canvas), 1000)


if __name__ == "__main__":
    unittest.main()
