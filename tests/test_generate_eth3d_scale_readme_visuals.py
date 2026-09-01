import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_eth3d_scale_readme_visuals.py"
)
SPEC = importlib.util.spec_from_file_location("generate_eth3d_scale_readme_visuals", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Eth3dScaleVisualTests(unittest.TestCase):
    def test_empty_points2d_rows_do_not_hide_the_next_pose(self) -> None:
        model = "\n".join(
            [
                "# images",
                "1 1 0 0 0 0 0 0 1 cam4_000001.png",
                "",
                "2 1 0 0 0 -1 0 0 1 cam4_000002.png",
                "",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            path.write_text(model, encoding="utf-8")
            centres = MODULE.read_centres(path)

        self.assertEqual(centres.shape, (2, 3))
        self.assertAlmostEqual(float(centres[0, 0]), 0.0)
        self.assertAlmostEqual(float(centres[1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
