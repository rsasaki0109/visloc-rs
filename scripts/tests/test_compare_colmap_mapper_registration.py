import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_colmap_mapper_registration as diag  # noqa: E402


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _pose_row(image_id: int, center, name: str) -> str:
    # identity rotation => translation = -center (see load_model_centres)
    return f"{image_id} 1 0 0 0 {-center[0]} {-center[1]} {-center[2]} 1 {name}"


class LcsTest(unittest.TestCase):
    def test_length_matches_bruteforce(self):
        rng = range(5)
        for _ in range(30):
            a = [int(x) for x in np.random.randint(0, 4, size=6)]
            b = [int(x) for x in np.random.randint(0, 4, size=6)]
            self.assertEqual(diag.lcs_length(a, b), _brute_lcs(a, b))

    def test_sequence_is_a_common_subsequence(self):
        a = [1, 2, 3, 4, 5]
        b = [2, 3, 5, 6]
        seq = diag.lcs_sequence(a, b)
        self.assertEqual(seq, [2, 3, 5])
        self.assertEqual(diag.lcs_length(a, b), len(seq))

    def test_empty(self):
        self.assertEqual(diag.lcs_length([], [1, 2]), 0)
        self.assertEqual(diag.lcs_sequence([], [1]), [])


def _brute_lcs(a, b):
    best = 0
    for r in range(len(a) + 1):
        for combo in itertools.combinations(a, r):
            it = iter(b)
            if all(any(x == y for y in it) for x in combo):
                best = max(best, r)
    return best


class ParseTest(unittest.TestCase):
    def test_parse_rig_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "m.txt", "S 0 1 1 1 1 1 0 0 0 0 0 0 0\nF 0 cam1_a.png 0\nF 0 cam2_b.png 1\nF 3 cam1_c.png 0\n")
            flat, frames = diag.parse_rig_manifest(path)
            self.assertEqual(flat, {"cam1_a.png": 0, "cam2_b.png": 0, "cam1_c.png": 3})
            self.assertEqual(frames[0], ["cam1_a.png", "cam2_b.png"])

    def test_parse_model_image_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "images.txt", "# header\n5 1 0 0 0 -1 -2 -3 1 a.png\n1.0 2.0 3\n7 1 0 0 0 -4 -5 -6 1 b.png\n1.0 2.0 3\n")
            self.assertEqual(diag.parse_model_image_ids(path), {5: "a.png", 7: "b.png"})

    def test_parse_colmap_log(self):
        log = (
            "Registering initial image pair #1 and #2\n"
            "Discarding reconstruction due to bad initial pair\n"
            "Registering initial image pair #3 and #4\n"
            "Global bundle adjustment\n"
            "Registering image #10 (num_reg_frames=2)\n"
            "Retriangulation and Global bundle adjustment\n"
            "Registering image #11 (num_reg_frames=3)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            parsed = diag.parse_colmap_log(_write(Path(tmp) / "mapper.log", log))
        self.assertEqual(parsed["init_attempts"], [(1, 2), (3, 4)])
        self.assertEqual(parsed["init_pair"], (3, 4))
        self.assertEqual(parsed["order"], [10, 11])
        self.assertEqual(
            [event["kind"] for event in parsed["events"]],
            ["initial_global_ba", "retriangulation_global_ba"],
        )

    def test_parse_ported_log(self):
        log = (
            "INIT_PAIR image1=5 image2=6 frame1=1 frame2=2\n"
            "REGISTER path=B frame=3 inliers=10 correspondences=20\n"
            "TIMING iterative_global_refinement elapsed_ms=1 num_reg_frames=2 points3d=5\n"
            "REGISTER path=A frame=4 inliers=9 correspondences=19\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            parsed = diag.parse_ported_log(_write(Path(tmp) / "stderr.log", log))
        self.assertEqual(parsed["init"], (5, 6, 1, 2))
        self.assertEqual(parsed["order"], [3, 4])
        self.assertEqual(parsed["events"], [{"kind": "iterative_global_refinement", "registered": 1}])

    def test_missing_init_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "stderr.log", "REGISTER path=B frame=3 inliers=1 correspondences=1\n")
            with self.assertRaises(diag.RegistrationDiagError):
                diag.parse_ported_log(path)


class CompareOrdersTest(unittest.TestCase):
    def test_detects_set_equality_and_divergence(self):
        result = diag.compare_orders([0, 1, 2, 3], [1, 2, 0, 3])
        self.assertTrue(result["registered_set_equal"])
        self.assertEqual(result["lcs_length"], 3)
        self.assertAlmostEqual(result["lcs_ratio"], 0.75)
        self.assertEqual(result["first_divergence_index"], 0)
        self.assertEqual(result["equal_positions"], 1)

    def test_unequal_sets(self):
        result = diag.compare_orders([0, 1, 2], [0, 1, 3])
        self.assertFalse(result["registered_set_equal"])
        self.assertEqual(result["lcs_length"], 2)


class PoseDifferenceTest(unittest.TestCase):
    def test_recovers_known_sim3(self):
        ported = {
            "cam1_0.png": np.array([0.0, 0.0, 0.0]),
            "cam1_1.png": np.array([1.0, 0.0, 0.0]),
            "cam1_2.png": np.array([0.0, 2.0, 0.0]),
            "cam1_3.png": np.array([0.0, 0.0, 3.0]),
        }
        scale = 2.5
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([10.0, -4.0, 1.0])
        colmap = {name: scale * rotation @ value + translation for name, value in ported.items()}
        flat_to_colmap = {name: name for name in ported}
        flat_to_frame = {"cam1_0.png": 0, "cam1_1.png": 1, "cam1_2.png": 2, "cam1_3.png": 3}
        result = diag.pose_difference(ported, colmap, flat_to_colmap, flat_to_frame, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertAlmostEqual(result["sim3_scale_ported_to_colmap"], scale, places=6)
        self.assertLess(result["center_error_m"]["max"], 1e-9)
        self.assertEqual(result["matched_images"], 4)

    def test_no_common_names_raises(self):
        with self.assertRaises(diag.RegistrationDiagError):
            diag.pose_difference(
                {"a.png": np.zeros(3)},
                {"b.png": np.zeros(3)},
                {"a.png": "missing.png"},
                {"a.png": 0},
                {0: 0},
            )


class MainEndToEndTest(unittest.TestCase):
    def _fixtures(self, tmp: Path):
        manifest = _write(
            tmp / "rig.txt",
            "F 0 cam1_000000.png 0\nF 0 cam2_000001.png 1\n"
            "F 1 cam1_000002.png 0\nF 1 cam2_000003.png 1\n"
            "F 2 cam1_000004.png 0\nF 2 cam2_000005.png 1\n",
        )
        aliases = _write(
            tmp / "aliases.tsv",
            "flat_name\tcolmap_name\n"
            "cam1_000000.png\trig/camera1/0.png\n"
            "cam2_000001.png\trig/camera2/1.png\n"
            "cam1_000002.png\trig/camera1/2.png\n"
            "cam2_000003.png\trig/camera2/3.png\n"
            "cam1_000004.png\trig/camera1/4.png\n"
            "cam2_000005.png\trig/camera2/5.png\n",
        )
        # COLMAP model: ids 100=cam1 frame0, 102=cam1 frame1, 104=cam1 frame2
        # ported centres are transformed by (scale=2.5, translation=(10,-4,1))
        # into the COLMAP centres, so pose_difference must recover scale 2.5.
        colmap_images = _write(
            tmp / "colmap_images.txt",
            "# header\n"
            + _pose_row(100, (10.0, -4.0, 1.0), "rig/camera1/0.png") + "\n1.0 2.0 3\n"
            + _pose_row(102, (12.5, -4.0, 1.0), "rig/camera1/2.png") + "\n1.0 2.0 3\n"
            + _pose_row(104, (10.0, -1.5, 1.0), "rig/camera1/4.png") + "\n1.0 2.0 3\n",
        )
        ported_images = _write(
            tmp / "ported_images.txt",
            "# header\n"
            + _pose_row(0, (0.0, 0.0, 0.0), "cam1_000000.png") + "\n1.0 2.0 3\n"
            + _pose_row(1, (1.0, 0.0, 0.0), "cam1_000002.png") + "\n1.0 2.0 3\n"
            + _pose_row(2, (0.0, 1.0, 0.0), "cam1_000004.png") + "\n1.0 2.0 3\n",
        )
        colmap_log = _write(
            tmp / "mapper.log",
            "Registering initial image pair #100 and #102\n"
            "Global bundle adjustment\n"
            "Registering image #104 (num_reg_frames=2)\n"
            "Retriangulation and Global bundle adjustment\n",
        )
        ported_log = _write(
            tmp / "stderr.log",
            "INIT_PAIR image1=2 image2=4 frame1=1 frame2=2\n"
            "TIMING iterative_global_refinement elapsed_ms=1 num_reg_frames=2 points3d=5\n"
            "REGISTER path=B frame=0 inliers=10 correspondences=20\n",
        )
        return manifest, aliases, colmap_images, ported_images, colmap_log, ported_log

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest, aliases, colmap_images, ported_images, colmap_log, ported_log = self._fixtures(tmp)
            out = tmp / "out.json"
            argv = [
                "--colmap-log", str(colmap_log),
                "--colmap-images", str(colmap_images),
                "--ported-log", str(ported_log),
                "--ported-images", str(ported_images),
                "--rig-manifest", str(manifest),
                "--image-aliases", str(aliases),
                "--out", str(out),
            ]
            self.assertEqual(diag.main(argv), 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "visloc_colmap_mapper_registration_diag_v1")
            self.assertEqual(payload["initial_pair"]["colmap_successful_frames"], [0, 1])
            self.assertEqual(payload["initial_pair"]["ported_frames"], [1, 2])
            self.assertFalse(payload["initial_pair"]["frames_match"])
            # colmap order [0,1,2], ported [1,2,0]
            self.assertAlmostEqual(payload["registration_order"]["full"]["lcs_ratio"], 2 / 3)
            self.assertTrue(payload["registration_order"]["full"]["registered_set_equal"])
            self.assertAlmostEqual(payload["pose_difference"]["sim3_scale_ported_to_colmap"], 2.5, places=6)
            self.assertLess(payload["pose_difference"]["center_error_m"]["max"], 1e-9)
            self.assertEqual(payload["global_ba_events"]["colmap_count"], 1)
            self.assertEqual(payload["global_ba_events"]["ported_count"], 1)


if __name__ == "__main__":
    unittest.main()
