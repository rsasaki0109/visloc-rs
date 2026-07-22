import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_gluemap_ssfm_adapter.py"
)
SPEC = importlib.util.spec_from_file_location("run_gluemap_ssfm_adapter", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_png_header(path: Path, width: int = 752, height: int = 480) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height))


class GluemapAdapterTests(unittest.TestCase):
    def test_calibration_only_model_contains_no_input_pose(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            images_dir = root / "images"
            images_dir.mkdir()
            images = [images_dir / "000000.png", images_dir / "000001.png"]
            for image in images:
                write_png_header(image)
            model = root / "calibration"
            MODULE.write_calibration_model(model, images, (436.0, 437.0, 364.0, 257.0))
            self.assertIn("1 PINHOLE 752 480 436 437 364 257", (model / "cameras.txt").read_text())
            text = (model / "images.txt").read_text()
            self.assertIn("no pose ground truth", text)
            pose_rows = MODULE.pose_rows(model / "images.txt")
            self.assertEqual(len(pose_rows), 2)
            self.assertTrue(all(row[1:8] == ["1", "0", "0", "0", "0", "0", "0"] for row in pose_rows))

    def test_official_command_keeps_full_pipeline_and_known_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = SimpleNamespace(
                gluemap_command="gluemap-demo",
                source_dir=root / "source",
                images_path=root / "images",
            )
            command = MODULE.gluemap_command(args, root / "calibration", root / "work")
            self.assertIn("--is_sequential", command)
            self.assertEqual(command[command.index("--sample_frequency") + 1], "1")
            self.assertEqual(command[command.index("--intrinsics_mode") + 1], "SHARED")
            self.assertIn("--use_gt_intrinsics", command)
            self.assertIn("--no-skip_doppelgangers", command)
            self.assertIn("--no-coarse_only", command)

    def test_colmap_world_to_camera_pose_is_written_as_timestamped_tum(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            images = root / "images.txt"
            images.write_text(
                "# image list\n1 1 0 0 0 1 2 3 1 /tmp/MH_03/000007.png\n\n",
                encoding="utf-8",
            )
            timestamps = root / "timestamps.txt"
            timestamps.write_text("7 1400000000\n", encoding="utf-8")
            output = root / "trajectory.tum"
            self.assertEqual(MODULE.write_tum(images, timestamps, output), 1)
            values = [float(value) for value in output.read_text().split()]
            self.assertEqual(len(values), 8)
            self.assertAlmostEqual(values[0], 1.4)
            self.assertEqual(values[1:4], [-1.0, -2.0, -3.0])
            self.assertEqual(values[4:], [0.0, 0.0, 0.0, 1.0])

    def test_rectified_calibration_parser_rejects_non_pinhole_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "calib.txt"
            path.write_text("P0: 436 1 364 0 0 437 257 0 0 0 1 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a rectified pinhole"):
                MODULE.parse_rectified_p0(path)


if __name__ == "__main__":
    unittest.main()
