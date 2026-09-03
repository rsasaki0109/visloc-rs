import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_rig_stereo_track_consistency.py"
SPEC = importlib.util.spec_from_file_location("diagnose_rig_stereo_track_consistency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class StereoTrackConsistencyTests(unittest.TestCase):
    def write_fixture(self, second_right_x=-22.0):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        model = root / "model"
        model.mkdir()
        (model / "cameras.txt").write_text(
            "1 PINHOLE 100 100 100 100 0 0\n"
            "2 PINHOLE 100 100 100 100 0 0\n",
            encoding="utf-8",
        )
        rows = [
            (1, 1, "left0.png", 0.0, 0.0),
            (2, 2, "right0.png", -0.1, -2.0),
            (3, 1, "left1.png", -1.0, -20.0),
            (4, 2, "right1.png", -1.1, second_right_x),
        ]
        image_lines = []
        for image_id, camera_id, name, tx, x in rows:
            image_lines.extend(
                [
                    f"{image_id} 1 0 0 0 {tx} 0 0 {camera_id} {name}\n",
                    f"{x} 0 1\n",
                ]
            )
        (model / "images.txt").write_text("".join(image_lines), encoding="utf-8")
        (model / "points3D.txt").write_text(
            "1 0 0 5 255 255 255 0 1 0 2 0 3 0 4 0\n", encoding="utf-8"
        )
        manifest = root / "rig-manifest.txt"
        manifest.write_text(
            "F 0 left0.png 0\n"
            "F 0 right0.png 1\n"
            "F 1 left1.png 0\n"
            "F 1 right1.png 1\n",
            encoding="utf-8",
        )
        return model, manifest

    def test_exact_two_frame_stereo_track_passes_tight_gate(self):
        model, manifest = self.write_fixture()
        result = MODULE.diagnose(model, manifest, [1e-8, 0.1], 0.5)
        self.assertFalse(result["ground_truth_used"])
        self.assertEqual(result["tracks_with_two_stereo_frames"], 1)
        self.assertEqual(result["observations_on_eligible_tracks"], 4)
        self.assertEqual(
            result["thresholds_by_max_cross_frame_deviation_m"]["1e-08"]["tracks"], 1
        )

    def test_inconsistent_second_stereo_frame_is_rejected(self):
        model, manifest = self.write_fixture(second_right_x=-30.0)
        result = MODULE.diagnose(model, manifest, [0.1], 0.5)
        self.assertGreater(result["max_cross_frame_deviation_m"]["max"], 0.1)
        self.assertEqual(
            result["thresholds_by_max_cross_frame_deviation_m"]["0.1"]["tracks"], 0
        )

    def test_colmap_name_aliases_are_inverted_for_manifest_lookup(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "aliases.tsv"
        path.write_text(
            "flat_name\tcolmap_name\nleft0.png\trig/camera1/time.png\n",
            encoding="utf-8",
        )
        self.assertEqual(
            MODULE.load_image_aliases(path), {"rig/camera1/time.png": "left0.png"}
        )


if __name__ == "__main__":
    unittest.main()
