import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage_openloris_corridor.py"
SPEC = importlib.util.spec_from_file_location("stage_openloris_corridor", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def members(camera: int, timestamps: list[str]) -> list[str]:
    return [f"corridor1-1/fisheye{camera}/{timestamp}.png" for timestamp in timestamps]


class OpenLorisStagingTests(unittest.TestCase):
    def test_py7zr_backpressure_drains_buffer_before_reading_more(self) -> None:
        from py7zr.compressor import SevenZipDecompressor

        MODULE.patch_py7zr_backpressure()
        decompressor = object.__new__(SevenZipDecompressor)
        decompressor.chain = [type("Buffered", (), {"needs_input": False})()]
        source = mock.Mock()

        self.assertEqual(decompressor._read_data(source), b"")
        source.read.assert_not_called()

    def test_extract_callback_releases_completed_folder_decompressors(self) -> None:
        file_info = type("FileInfo", (), {"filename": "scene/frame.png"})()
        folder = type("Folder", (), {"files": [file_info], "decompressor": object()})()
        unpackinfo = type("UnpackInfo", (), {"folders": [folder]})()
        main_streams = type("MainStreams", (), {"unpackinfo": unpackinfo})()
        header = type("Header", (), {"main_streams": main_streams})()
        archive = type("Archive", (), {"header": header})()

        callback = MODULE.releasing_extract_callback(archive)
        callback.report_end("scene/frame.png", "123")

        self.assertIsNone(folder.decompressor)

    def test_selection_is_bounded_and_globally_timestamp_ordered(self) -> None:
        names = [
            *members(1, ["1.000", "2.100", "3.000"]),
            *members(2, ["1.050", "2.000", "3.100"]),
            "corridor1-1/sensors.yaml",
        ]

        selected, targets = MODULE.selected_members(names, 2)

        self.assertEqual(
            [(camera, timestamp) for camera, _, timestamp in selected],
            [(1, "1.000"), (2, "1.050"), (2, "2.000"), (1, "2.100")],
        )
        self.assertEqual(len(targets), 9)
        self.assertTrue(all(member in targets for _, member, _ in selected))

    def test_selection_rejects_a_tier_larger_than_either_camera(self) -> None:
        names = [*members(1, ["1", "2"]), *members(2, ["1"])]
        with self.assertRaisesRegex(ValueError, "1..1"):
            MODULE.selected_members(names, 2)

    def test_complete_raw_reuse_rejects_missing_and_empty_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "present").write_bytes(b"ok")
            (root / "empty").touch()
            with self.assertRaisesRegex(ValueError, "1 targets missing"):
                MODULE.validate_complete_raw(root, ["present", "missing"])
            with self.assertRaisesRegex(ValueError, "1 targets empty"):
                MODULE.validate_complete_raw(root, ["present", "empty"])
            MODULE.validate_complete_raw(root, ["present"])

    def test_calibration_is_intrinsics_only_and_keeps_camera_assignment(self) -> None:
        intrinsics = {
            1: (100.0, 101.0, 50.0, 51.0),
            2: (102.0, 103.0, 52.0, 53.0),
        }
        records = [
            {"camera": 1, "name": "cam1_1.png"},
            {"camera": 2, "name": "cam2_1.png"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            MODULE.write_calibration(root, intrinsics, records)
            cameras = (root / "calibration" / "cameras.txt").read_text()
            images = (root / "calibration" / "images.txt").read_text()

        self.assertIn("1 PINHOLE 848 800 100 101 50 51", cameras)
        self.assertIn("2 PINHOLE 848 800 102 103 52 53", cameras)
        self.assertIn("1 1 0 0 0 0 0 0 1 cam1_1.png", images)
        self.assertIn("2 1 0 0 0 0 0 0 2 cam2_1.png", images)
        self.assertIn("never use as GT", images)

    def test_tier_views_use_prefix_symlinks_and_subset_calibration(self) -> None:
        intrinsics = {1: (100.0, 101.0, 50.0, 51.0)}
        records = [
            {"camera": 1, "name": f"cam1_{index:06}.png"} for index in range(4)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            for record in records:
                (root / "images" / record["name"]).touch()
            with mock.patch.object(MODULE, "TIER_COUNTS", (2, 4)):
                views = MODULE.write_tier_views(root, intrinsics, records)

            first_tier = root / "tiers" / "tier-2"
            links = sorted((first_tier / "images").iterdir())
            link_states = [path.is_symlink() for path in links]
            images_text = (first_tier / "calibration" / "images.txt").read_text()

        self.assertEqual(set(views), {"2", "4"})
        self.assertEqual(len(links), 2)
        self.assertTrue(all(link_states))
        self.assertIn("cam1_000000.png", images_text)
        self.assertIn("cam1_000001.png", images_text)
        self.assertNotIn("cam1_000002.png", images_text)


if __name__ == "__main__":
    unittest.main()
