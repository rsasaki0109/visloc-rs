import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage_eth3d_rig_scene.py"
SPEC = importlib.util.spec_from_file_location("stage_eth3d_rig_scene", SCRIPT)
stage = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stage)


class Eth3dRigStageTests(unittest.TestCase):
    def test_stage_removes_reference_poses_and_flattens_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / "forest"
            calibration = scene / "rig_calibration_undistorted"
            calibration.mkdir(parents=True)
            (calibration / "cameras.txt").write_text("0 PINHOLE 10 8 5 5 5 4\n")
            (calibration / "images.txt").write_text(
                "1 .1 .2 .3 .4 8 9 10 0 images_rig_cam4_undistorted/2.png\n1 2 -1\n"
                "2 .5 .6 .7 .8 11 12 13 0 images_rig_cam4_undistorted/1.png\n\n"
            )
            source = scene / "images" / "images_rig_cam4_undistorted"
            source.mkdir(parents=True)
            (source / "1.png").write_bytes(b"one")
            (source / "2.png").write_bytes(b"two")
            output = root / "staging"
            result = stage.stage(scene, output)
            self.assertEqual(result["image_count"], 2)
            images_text = (output / "calibration" / "images.txt").read_text()
            self.assertIn("1 1 0 0 0 0 0 0 0 cam4_1.png", images_text)
            self.assertIn("2 1 0 0 0 0 0 0 0 cam4_2.png", images_text)
            self.assertNotIn(".1 .2 .3 .4", images_text)
            self.assertEqual((output / "images" / "cam4_1.png").read_bytes(), b"one")

    def test_rejects_malformed_camera_directory(self):
        with self.assertRaises(stage.StageError):
            stage.flattened_name("camera0/1.png")


if __name__ == "__main__":
    unittest.main()
