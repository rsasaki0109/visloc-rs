import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "select_rig_sift_supplements",
    Path(__file__).resolve().parents[1] / "select_rig_sift_supplements.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SelectionTests(unittest.TestCase):
    def test_manifest_sensor_membership_and_filename_collisions(self):
        declarations = "S 0 " + "1 " * 14 + "\nS 1 " + "1 " * 14 + "\n"
        valid = declarations + "F 0 a.png 0\nF 0 b.png 1\n"
        self.assertEqual(MODULE.parse_frames(valid.encode()), {0: ["a.png", "b.png"]})
        for rows in (
            "F 0 a.png 0\n",  # Missing sensor must not look like sufficient support.
            "F 0 a.png 0\nF 0 b.png 2\n",
            "F 0 a.png 0\nF 0 a.jpg 1\n",
            "F 0 a.png 0\nF 0 ../b.png 1\n",
            "F 0 a.png 0\nF 0 b.png 0\n",
        ):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    MODULE.parse_frames((declarations + rows).encode())

    def test_strict_threshold_and_ordinal_halo(self):
        frames = {10: ["a", "b"], 20: ["c", "d"], 40: ["e", "f"], 90: ["g", "h"]}
        counts = dict.fromkeys("abcdefgh", 32)
        self.assertEqual(MODULE.select(frames, counts, 32, 1)["selected_images"], 0)
        counts["c"] = 31
        result = MODULE.select(frames, counts, 32, 1)
        self.assertEqual(result["base_frames"], 1)
        self.assertEqual(result["image_names"], list("abcdef"))

    def test_clipped_overlapping_halos(self):
        frames = {0: ["a"], 1: ["b"], 2: ["c"]}
        result = MODULE.select(frames, {"a": 0, "b": 32, "c": 0}, 32, 1000000)
        self.assertEqual(result["selected_frames"], 3)
        self.assertEqual(result["selected_images"], 3)

    def test_invalid_inputs(self):
        for frames, counts, threshold, halo in [
            ({0: ["a"]}, {}, 32, 8),
            ({0: ["a"], 1: ["a"]}, {"a": 1}, 32, 8),
            ({0: ["a"]}, {"a": -1}, 32, 8),
            ({0: ["a"]}, {"a": 1}, 32, -1),
            ({0: []}, {}, 32, 8),
        ]:
            with self.subTest(frames=frames, counts=counts, halo=halo):
                with self.assertRaises(ValueError):
                    MODULE.select(frames, counts, threshold, halo)


if __name__ == "__main__":
    unittest.main()
