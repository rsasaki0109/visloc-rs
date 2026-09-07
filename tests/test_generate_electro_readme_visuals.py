import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_electro_readme_visuals.py"
SPEC = importlib.util.spec_from_file_location("generate_electro_readme_visuals", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ElectroReadmeVisualTests(unittest.TestCase):
    def test_ecdf_note_describes_tighter_direction(self) -> None:
        self.assertEqual(
            MODULE.CDF_DIRECTION_NOTE,
            "higher/left curve = tighter camera-centre agreement",
        )
        self.assertNotIn("lower curve", MODULE.CDF_DIRECTION_NOTE)

    def test_summary_metric_column_stays_inside_card(self) -> None:
        self.assertGreater(MODULE.SUMMARY_CARD_METRICS_X, 0.50)
        self.assertLess(MODULE.SUMMARY_CARD_METRICS_X, 0.59)

    def test_tracked_visual_dimensions_and_frame_count(self) -> None:
        from PIL import Image

        with Image.open(ROOT / "docs/assets/electro_1200_sfm_comparison.png") as png:
            self.assertEqual(png.size, (1690, 936))
        with Image.open(ROOT / "docs/assets/electro_1200_sfm_comparison.gif") as gif:
            self.assertEqual(gif.size, (864, 486))
            self.assertEqual(getattr(gif, "n_frames", 1), 24)


if __name__ == "__main__":
    unittest.main()
