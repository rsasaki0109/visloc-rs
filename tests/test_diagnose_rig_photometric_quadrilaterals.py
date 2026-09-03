import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_rig_photometric_quadrilaterals.py"


class PhotometricQuadrilateralTests(unittest.TestCase):
    def test_translation_endpoint_and_two_sensor_gates(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        images = root / "images"
        images.mkdir()

        rng = np.random.default_rng(7)
        base = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        transform = np.float32([[1, 0, 3], [0, 1, 2]])
        moved = cv2.warpAffine(base, transform, (128, 128))
        for name, image in (
            ("left0.png", base),
            ("left1.png", moved),
            ("right0.png", base),
            ("right1.png", moved),
        ):
            self.assertTrue(cv2.imwrite(str(images / name), image))

        fields = ["track"]
        for index in range(4):
            fields.extend(
                [
                    f"image_{index}",
                    f"keypoint_{index}",
                    f"frame_{index}",
                    f"sensor_{index}",
                    f"name_{index}",
                    f"x_{index}",
                    f"y_{index}",
                ]
            )

        def row(track, left_target, right_target):
            observations = [
                (0, 10 + track, 0, 0, "left0.png", 48.0, 52.0),
                (1, 20 + track, 1, 0, "left1.png", *left_target),
                (2, 30 + track, 0, 1, "right0.png", 70.0, 65.0),
                (3, 40 + track, 1, 1, "right1.png", *right_target),
            ]
            values = {"track": track}
            for index, observation in enumerate(observations):
                for key, value in zip(
                    ("image", "keypoint", "frame", "sensor", "name", "x", "y"),
                    observation,
                ):
                    values[f"{key}_{index}"] = value
            return values

        quadrilaterals = root / "quadrilaterals.tsv"
        with quadrilaterals.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, delimiter="\t")
            writer.writeheader()
            writer.writerow(row(0, (51.0, 54.0), (73.0, 67.0)))
            writer.writerow(row(1, (60.0, 60.0), (73.0, 67.0)))
            writer.writerow(row(2, (51.0, 54.0), (82.0, 75.0)))

        output = root / "diagnostic.json"
        accepted = root / "accepted.tsv"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--quadrilaterals-tsv",
                str(quadrilaterals),
                "--images-dir",
                str(images),
                "--output-json",
                str(output),
                "--accepted-tsv",
                str(accepted),
                "--min-frame-gap",
                "1",
                "--max-frame-gap",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["candidate_tracks"], 3)
        self.assertEqual(result["accepted_tracks"], 1)
        self.assertFalse(result["ground_truth_used"])
        with accepted.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual([int(value["track"]) for value in rows], [0])
        self.assertEqual(rows[0]["image_3"], "3")


if __name__ == "__main__":
    unittest.main()
