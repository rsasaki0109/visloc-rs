import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "estimate_rig_metric_factors.py"
SPEC = importlib.util.spec_from_file_location("estimate_rig_metric_factors", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MetricFactorTests(unittest.TestCase):
    def test_recovers_calibrated_metric_translation(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        manifest = root / "rig.txt"
        manifest.write_text(
            "S 0 1 100 100 100 100 50 50 1 0 0 0 0 0 0\n"
            "S 1 2 100 100 100 100 50 50 1 0 0 0 -0.2 0 0\n",
            encoding="utf-8",
        )
        fields = ["track"]
        for index in range(4):
            fields.extend(
                [f"image_{index}", f"keypoint_{index}", f"frame_{index}",
                 f"sensor_{index}", f"name_{index}", f"x_{index}", f"y_{index}"]
            )
        tracks = root / "tracks.tsv"
        world = [(0.0, 0.0, 4.0), (0.3, 0.2, 5.0), (-0.4, 0.1, 3.5),
                 (0.2, -0.3, 4.5), (-0.2, -0.2, 5.5), (0.5, 0.4, 6.0),
                 (-0.5, 0.3, 4.2), (0.1, 0.5, 3.8)]
        translation = np.asarray([0.1, -0.02, 0.03])

        def project(point, sensor):
            sensor_t = np.asarray([0.0, 0.0, 0.0]) if sensor == 0 else np.asarray([-0.2, 0.0, 0.0])
            camera = np.asarray(point) + sensor_t
            return 100 * camera[0] / camera[2] + 50, 100 * camera[1] / camera[2] + 50

        with tracks.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, delimiter="\t")
            writer.writeheader()
            for track, point in enumerate(world):
                target = np.asarray(point) + translation
                observations = [
                    (0, track, 0, 0, "l0", *project(point, 0)),
                    (1, track, 0, 1, "r0", *project(point, 1)),
                    (2, track, 32, 0, "l1", *project(target, 0)),
                    (3, track, 32, 1, "r1", *project(target, 1)),
                ]
                row = {"track": track}
                for index, observation in enumerate(observations):
                    for key, value in zip(("image", "keypoint", "frame", "sensor", "name", "x", "y"), observation):
                        row[f"{key}_{index}"] = value
                writer.writerow(row)
        factors, result = MODULE.estimate(
            tracks, manifest, min_angle_deg=0.1, min_correspondences=8,
            min_inliers=6, ransac_threshold_m=1e-6, ransac_iterations=16,
        )
        self.assertEqual(result["accepted_factors"], 1)
        np.testing.assert_allclose(factors[0]["rotation"], np.eye(3), atol=1e-9)
        np.testing.assert_allclose(factors[0]["translation"], translation, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
