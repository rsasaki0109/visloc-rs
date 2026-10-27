import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eth3d_pose_errors as eth3d  # noqa: E402
import score_smallcity_reference as smallcity  # noqa: E402


def _quaternion(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def _write_images(path: Path, poses) -> None:
    """COLMAP images.txt from (name, quaternion wxyz, camera centre) triples."""
    with open(path, "w") as out:
        out.write("# images\n")
        for i, (name, q, c) in enumerate(poses, 1):
            r = eth3d.rotation(np.asarray(q, float))
            t = -r @ np.asarray(c, float)
            out.write(f"{i} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {name}\n\n")


def _scene(n=12, seed=0):
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n):
        q = _quaternion(rng.normal(size=3), rng.uniform(0, 0.5))
        poses.append((f"img_{i:03d}.jpg", q, rng.uniform(-5, 5, size=3)))
    return poses


def _similarity(poses, scale, axis, angle, shift):
    """The same cameras in a frame moved by a similarity transform."""
    q_frame = _quaternion(axis, angle)
    r_frame = eth3d.rotation(q_frame)
    out = []
    for name, q, c in poses:
        # World-to-camera rotation R' = R * R_frame^T; centre c' = s R_frame c + shift.
        r_new = eth3d.rotation(np.asarray(q, float)) @ r_frame.T
        w = np.sqrt(max(1e-12, 1 + np.trace(r_new))) / 2
        q_new = np.array(
            [
                w,
                (r_new[2, 1] - r_new[1, 2]) / (4 * w),
                (r_new[0, 2] - r_new[2, 0]) / (4 * w),
                (r_new[1, 0] - r_new[0, 1]) / (4 * w),
            ]
        )
        out.append((name, q_new, scale * r_frame @ np.asarray(c, float) + shift))
    return out


class Eth3dPoseErrorsTest(unittest.TestCase):
    def _run(self, est, gt, names=None):
        with tempfile.TemporaryDirectory() as tmp:
            e, g = Path(tmp) / "est.txt", Path(tmp) / "gt.txt"
            _write_images(e, est)
            _write_images(g, gt)
            argv = ["eth3d_pose_errors.py", str(e), str(g)]
            if names is not None:
                n = Path(tmp) / "image_names.txt"
                n.write_text("".join(f"{a} {b}\n" for a, b in names))
                argv.append(str(n))
            out = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                eth3d.main()
            return out.getvalue()

    def test_a_similar_copy_has_no_error(self):
        gt = _scene()
        est = _similarity(gt, 0.3, [1, 2, 3], 0.7, np.array([4.0, -1.0, 2.0]))
        line = self._run(est, gt)
        self.assertIn("12/12 reg", line)
        self.assertIn("relrot med 0.000 deg", line)
        self.assertIn("centre med 0.00 p90 0.00 max 0.0 cm", line)

    def test_working_copy_names_map_back(self):
        gt = _scene()
        est = [(f"frame_{i:05d}.png", q, c) for i, (_, q, c) in enumerate(gt)]
        names = [(f"frame_{i:05d}.png", f"/photos/{n}") for i, (n, _, _) in enumerate(gt)]
        self.assertIn("12/12 reg", self._run(est, gt, names))

    def test_one_camera_off_is_left_out_of_the_fit(self):
        gt = _scene(20)
        est = [(n, q, np.asarray(c) + (np.array([0.0, 0.0, 2.0]) if i == 3 else 0)) for i, (n, q, c) in enumerate(gt)]
        line = self._run(est, gt)
        self.assertIn("centre med 0.00", line)
        self.assertIn("max 200.0 cm", line)


class SmallCityScoreTest(unittest.TestCase):
    def _score(self, ref, est, names, **flags):
        with tempfile.TemporaryDirectory() as tmp:
            r, e, n = Path(tmp) / "ref.txt", Path(tmp) / "est.txt", Path(tmp) / "names.txt"
            _write_images(r, ref)
            _write_images(e, est)
            n.write_text("".join(f"{a} {b}\n" for a, b in names))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                smallcity.score(str(r), str(e), str(n), **flags)
            return out.getvalue()

    def test_subset_names_and_trimmed_fit(self):
        # A 300-photo loop around a block; the subset is photos 100..199 and
        # one of them is 5 m off.
        ref = [
            (
                f"pass1_{i:04d}.jpg",
                np.array([1.0, 0, 0, 0]),
                np.array([40 * np.cos(i / 48), 25 * np.sin(i / 48), 0.1 * np.sin(i)]),
            )
            for i in range(300)
        ]
        subset = ref[100:200]
        est = []
        names = []
        for k, (name, q, c) in enumerate(subset):
            offset = np.array([0.0, 5.0, 0.0]) if k == 40 else 0.0
            est.append((f"frame_{k:05d}.png", q, 2.0 * c + offset))
            names.append((f"frame_{k:05d}.png", f"/data/{name}"))
        report = self._score(ref, est, names, trimmed=True, profile=True)
        self.assertIn("100 registered, 100 of 300", report)
        self.assertIn("cameras over 1 m (trimmed fit): 1: pass1_0140.jpg", report)
        self.assertIn("centre error median 0.000 m", report)
        self.assertIn("pass1_0100.jpg..: scale", report)


if __name__ == "__main__":
    unittest.main()
