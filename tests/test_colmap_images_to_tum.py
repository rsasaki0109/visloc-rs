import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colmap_images_to_tum.py"
SPEC = importlib.util.spec_from_file_location("colmap_images_to_tum", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ColmapImagesToTumTests(unittest.TestCase):
    def test_pose_lines_survive_empty_points2d_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            path.write_text(
                "# Image list\n"
                "1 1 0 0 0 0 0 0 1 000000.png\n"
                "\n"
                "2 1 0 0 0 1 0 0 1 000001.png\n"
                "10 20 7 30 40 -1\n",
                encoding="utf-8",
            )
            rows = MODULE.pose_lines(path)
        self.assertEqual([row[9] for row in rows], ["000000.png", "000001.png"])


if __name__ == "__main__":
    unittest.main()
