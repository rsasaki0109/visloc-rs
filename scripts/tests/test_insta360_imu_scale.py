import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import insta360_imu_scale as imu_scale  # noqa: E402

G_VEC = np.array([0.0, 9.80665, 0.0])  # gravity along +Y, as in the dataset
B_A = np.array([0.05, -0.1, 0.08])
R_CB = Rot.from_euler("xyz", [10, 80, -5], degrees=True)
LENS1_FROM_LENS0 = Rot.from_euler("y", 180, degrees=True)


def _position(t):
    w1, w2, w3 = 2 * np.pi * 0.85, 2 * np.pi * 1.7, 2 * np.pi / 40
    p = np.c_[1.2 * t + 0.04 * np.sin(w1 * t), 0.03 * np.sin(w2 * t), 8 * np.sin(w3 * t)]
    acc = np.c_[-0.04 * w1**2 * np.sin(w1 * t), -0.03 * w2**2 * np.sin(w2 * t), -8 * w3**2 * np.sin(w3 * t)]
    return p, acc


def _body_rotation(t):
    yaw = 0.4 * np.sin(2 * np.pi * t / 25) + 0.05 * np.sin(2 * np.pi * 0.85 * t + 0.5)
    pitch = 0.04 * np.sin(2 * np.pi * 1.7 * t + 0.3)
    roll = 0.05 * np.sin(2 * np.pi * 0.85 * t + 1.0)
    return Rot.from_euler("yxz", np.c_[yaw, pitch, roll])


def _write_images_bin(path, images):
    """COLMAP binary images.bin from (name, world-from-camera R, centre, camera id)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for i, (name, r_wc, c, cam) in enumerate(images, 1):
            x, y, z, w = Rot.from_matrix(r_wc.T).as_quat()
            t = -r_wc.T @ c
            f.write(struct.pack("<I7dI", i, w, x, y, z, *t, cam))
            f.write(name.encode() + b"\0")
            f.write(struct.pack("<Q", 0))


def _scene(directory, s_true=1.05, period=1 / 3, offset=0.4, seconds=75.0):
    """An OverMaps360_50-like scene: model in units of s_true metres, raw-count IMU."""
    t = np.arange(0, seconds, 1e-3)
    h = 1e-4
    gyro = (_body_rotation(t - h).inv() * _body_rotation(t + h)).as_rotvec() / (2 * h)
    _, acc_w = _position(t)
    spec = _body_rotation(t).inv().apply(acc_w - G_VEC) + B_A
    counts = np.c_[t, np.degrees(gyro) * 16.4, spec / 9.80665 * 1024]
    np.savetxt(
        Path(directory) / "insta360_imu.csv",
        counts,
        delimiter=",",
        header="t_sec,gyro_x_dps,gyro_y_dps,gyro_z_dps,accel_x_mg,accel_y_mg,accel_z_mg",
        comments="",
    )
    # Frames i at t = period * i + offset; a 1.7 s gap splits them into two runs.
    idx = np.array([i for i in range(1, int((seconds - 1 - offset) / period)) if not 120 <= i < 125])
    tf = period * idx + offset
    p, _ = _position(tf)
    r_wc0 = _body_rotation(tf) * R_CB.inv()
    images = []
    for k, i in enumerate(idx):
        for lens, r_wc in ((0, r_wc0[k]), (1, r_wc0[k] * LENS1_FROM_LENS0)):
            for view, yaw in ((0, 0.0), (1, 60.0)):
                r = (r_wc * Rot.from_euler("y", yaw, degrees=True)).as_matrix()
                images.append((f"{2 * i + lens}_perspective_{view:08d}.jpg", r, p[k] / s_true, 1))
    # Phone images: the model's centres and ARKit positions s_true * R x + t.
    rng = np.random.default_rng(0)
    phone = rng.uniform(-10, 10, size=(40, 3)) / s_true
    r_ark = Rot.from_euler("y", 30, degrees=True)
    rows = ["imageName,pos_x,pos_y,pos_z"]
    for n, x in enumerate(phone):
        name = f"405_{n:04d}.jpg"
        images.append((name, np.eye(3), x, 2))
        y = s_true * r_ark.apply(x) + np.array([1.0, 2.0, 3.0])
        rows.append(f"{name},{y[0]},{y[1]},{y[2]}")
    (Path(directory) / "smartphone_poses_405.csv").write_text("\n".join(rows) + "\n")
    _write_images_bin(Path(directory) / "images.bin", images)


class Insta360ImuScaleTest(unittest.TestCase):
    def test_synthetic_scene_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            _scene(tmp)
            r = imu_scale.analyse_scene(tmp, log=lambda m: None)
        self.assertTrue(r["lens_interleaved"])
        self.assertEqual(r["phone_images"], 40)
        self.assertAlmostEqual(r["s_ref"], 1.05, places=6)
        self.assertAlmostEqual(r["accel_one_g_counts"], 1024, delta=15)
        self.assertAlmostEqual(r["gyro_counts_per_dps_fit"], 16.4, delta=0.05)
        self.assertAlmostEqual(r["frame_period"], 1 / 3, delta=1e-5)
        self.assertAlmostEqual(r["frame_offset"], 0.4, delta=2e-3)
        self.assertLess(r["rotation_residual_deg"], 0.01)
        for lens in ("lens0", "lens1"):
            self.assertEqual(r[lens]["runs_kept"], 2)
            self.assertAlmostEqual(r[lens]["s_per_run"] / r["s_ref"], 1, delta=3e-3)
            self.assertAlmostEqual(r[lens]["g_median"], 9.80665, delta=0.15)
        e = imu_scale.scene_errors(r)
        self.assertLess(abs(e["s_per_run"]), 0.3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(r))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                imu_scale.summary([str(path)])
        self.assertIn("per-run         1 scenes", out.getvalue())

    def test_rotation_fit_ignores_outliers(self):
        rng = np.random.default_rng(1)
        r_cb = Rot.from_euler("xyz", [5, -80, 20], degrees=True)
        rb = rng.normal(scale=0.05, size=(200, 3))
        noise = Rot.from_rotvec(rng.normal(scale=1e-4, size=(200, 3)))
        d_rc = noise * r_cb * Rot.from_rotvec(rb) * r_cb.inv()
        bad = Rot.from_rotvec(rng.normal(scale=0.5, size=(20, 3)))
        d_rc = Rot.concatenate([d_rc[:180], bad])
        r, resid, inliers = imu_scale.rotation_fit(rb, d_rc)
        self.assertLess((Rot.from_matrix(r) * r_cb.inv()).magnitude(), 1e-3)
        self.assertGreater(inliers[:180].sum(), 170)
        self.assertEqual(inliers[180:].sum(), 0)


if __name__ == "__main__":
    unittest.main()
