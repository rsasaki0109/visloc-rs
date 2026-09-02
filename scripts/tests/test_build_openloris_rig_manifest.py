import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_openloris_rig_manifest import ManifestError, build


class RigManifestTests(unittest.TestCase):
    def test_builds_two_sensor_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tier = root / "tier.json"
            rig = root / "rig.json"
            tier.write_text(
                json.dumps(
                    {
                        "schema": "visloc_openloris_corridor_manifest_v1",
                        "images": [
                            {"name": "left.png", "camera": 1, "timestamp": "10"},
                            {"name": "right.png", "camera": 2, "timestamp": "10"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rig.write_text(
                json.dumps(
                    [
                        {
                            "cameras": [
                                {
                                    "image_prefix": "rig/camera1/",
                                    "camera_model_name": "PINHOLE",
                                    "camera_params": [1, 2, 3, 4],
                                    "ref_sensor": True,
                                },
                                {
                                    "image_prefix": "rig/camera2/",
                                    "camera_model_name": "PINHOLE",
                                    "camera_params": [5, 6, 7, 8],
                                    "cam_from_rig_rotation": [1, 0, 0, 0],
                                    "cam_from_rig_translation": [-0.2, 0, 0],
                                },
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output = build(tier, rig, 848, 800)
            self.assertIn("S 1 2 848 800 5 6 7 8 1 0 0 0 -0.20000000000000001 0 0", output)
            self.assertIn("F 0 left.png 0", output)
            self.assertIn("F 0 right.png 1", output)

    def test_rejects_incomplete_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tier = root / "tier.json"
            rig = root / "rig.json"
            tier.write_text(
                json.dumps(
                    {
                        "schema": "visloc_openloris_corridor_manifest_v1",
                        "images": [{"name": "left.png", "camera": 1, "timestamp": "10"}],
                    }
                ),
                encoding="utf-8",
            )
            rig.write_text(
                json.dumps(
                    [
                        {
                            "cameras": [
                                {
                                    "image_prefix": "rig/camera1/",
                                    "camera_model_name": "PINHOLE",
                                    "camera_params": [1, 2, 3, 4],
                                    "ref_sensor": True,
                                },
                                {
                                    "image_prefix": "rig/camera2/",
                                    "camera_model_name": "PINHOLE",
                                    "camera_params": [5, 6, 7, 8],
                                    "cam_from_rig_rotation": [1, 0, 0, 0],
                                    "cam_from_rig_translation": [-0.2, 0, 0],
                                },
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ManifestError):
                build(tier, rig, 848, 800)


if __name__ == "__main__":
    unittest.main()
