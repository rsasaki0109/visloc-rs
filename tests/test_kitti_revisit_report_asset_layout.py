from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kitti_revisit_report_asset_layout import ReportAssetLayout  # noqa: E402


class KittiRevisitReportAssetLayoutTest(unittest.TestCase):
    def test_default_layout_matches_readme_asset_geometry(self) -> None:
        layout = ReportAssetLayout(1400)

        self.assertEqual(layout.overlay_size, (1328, 279))
        self.assertEqual(layout.canvas_size, (1400, 573))
        self.assertEqual(layout.overlay_xy, (36, 140))
        self.assertEqual(layout.footer_y, 443)
        self.assertEqual(layout.settings_xy, (36, 501))

    def test_badge_positions_keep_fixed_offsets_from_margin(self) -> None:
        layout = ReportAssetLayout(1000, margin=20, header_height=100, footer_height=80)

        self.assertEqual(layout.badge_xy(0), (20, 326))
        self.assertEqual(layout.badge_xy(1), (290, 326))
        self.assertEqual(layout.badge_xy(2), (575, 326))
        self.assertEqual(layout.badge_xy(3), (765, 326))


if __name__ == "__main__":
    unittest.main()
