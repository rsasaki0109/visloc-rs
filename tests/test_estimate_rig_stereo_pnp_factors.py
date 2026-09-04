import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import estimate_rig_stereo_pnp_factors as factors  # noqa: E402


class StereoPnpFactorTests(unittest.TestCase):
    def test_recovers_bidirectionally_consistent_metric_rig_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "rig.txt"
            manifest.write_text(
                "# generalized-rig-manifest-v1\n"
                "S 0 1 640 480 300 300 320 240 1 0 0 0 0 0 0\n"
                "S 1 2 640 480 300 300 320 240 1 0 0 0 -0.064 0 0\n"
                "F 0 left0.png 0\nF 0 right0.png 1\n"
                "F 32 left32.png 0\nF 32 right32.png 1\n",
                encoding="utf-8",
            )
            angle = math.radians(2.0)
            rotation = np.asarray(
                [
                    [math.cos(angle), 0.0, math.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [-math.sin(angle), 0.0, math.cos(angle)],
                ]
            )
            translation = np.asarray([0.12, -0.01, 0.03])
            sensor_translations = (np.zeros(3), np.asarray([-0.064, 0.0, 0.0]))

            def project(point, frame, sensor):
                local = point if frame == 0 else rotation @ point + translation
                camera = local + sensor_translations[sensor]
                return 300.0 * camera[0] / camera[2] + 320.0, 300.0 * camera[1] / camera[2] + 240.0

            tracks = root / "tracks.tsv"
            fields = ["track"] + [
                f"{name}_{index}"
                for index in range(4)
                for name in ("image", "keypoint", "frame", "sensor", "name", "x", "y")
            ]
            with tracks.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fields, delimiter="\t")
                writer.writeheader()
                for track in range(24):
                    point = np.asarray(
                        [
                            -0.45 + 0.09 * (track % 8),
                            -0.24 + 0.12 * ((track // 8) % 3),
                            2.0 + 0.15 * (track % 5),
                        ]
                    )
                    row = {"track": track}
                    for index, (frame, sensor, name) in enumerate(
                        ((0, 0, "left0.png"), (0, 1, "right0.png"), (32, 0, "left32.png"), (32, 1, "right32.png"))
                    ):
                        x, y = project(point, frame, sensor)
                        row.update(
                            {
                                f"image_{index}": frame * 2 + sensor,
                                f"keypoint_{index}": track,
                                f"frame_{index}": frame,
                                f"sensor_{index}": sensor,
                                f"name_{index}": name,
                                f"x_{index}": f"{x:.17g}",
                                f"y_{index}": f"{y:.17g}",
                            }
                        )
                    writer.writerow(row)

            estimated, report = factors.estimate(
                tracks,
                manifest,
                min_angle_deg=0.5,
                min_correspondences=8,
                min_inliers=6,
                ransac_threshold_px=0.5,
                ransac_iterations=128,
                min_frame_gap=32,
                max_frame_gap=32,
                max_sensor_rotation_error_deg=0.1,
                max_sensor_translation_error_m=0.005,
                max_forward_reverse_rotation_error_deg=0.1,
                max_forward_reverse_translation_error_m=0.005,
            )
            self.assertEqual(report["forward_reverse_consistent_pairs"], 1)
            self.assertEqual(len(estimated), 2)
            forward = next(value for value in estimated if value["direction"] == "forward")
            np.testing.assert_allclose(forward["rotation"], rotation, atol=1e-5)
            np.testing.assert_allclose(forward["translation"], translation, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
