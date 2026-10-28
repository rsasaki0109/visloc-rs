from pathlib import Path
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate_wheel_offset import calibrate, format_pose, main, parse_pose, quat_matrix, read_tum, rotation_quat

# OpenLORIS-Scene's base_link in the T265 IMU frame.
IMU_FROM_BODY = "0.0306,-0.8937,-0.2452,-0.5030,-0.4927,-0.5074,0.4967"


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def drive():
    """Planar wheel odometry: 2 m ahead, an in-place half turn, 2 m back, a quarter turn."""
    yaw, x, y, poses = 0.0, 0.0, 0.0, []
    for steps, speed, turn in ((100, 0.02, 0.0), (90, 0.0, np.pi / 90), (100, 0.02, 0.0), (45, 0.0, np.pi / 90)):
        for _ in range(steps):
            x, y, yaw = x + speed * np.cos(yaw), y + speed * np.sin(yaw), yaw + turn
            poses.append((x, y, yaw))
    return np.array(poses)


def write_tum(path, stamps_ns, positions, rotations):
    lines = []
    for stamp, p, r in zip(stamps_ns, positions, rotations):
        q = rotation_quat(r)
        lines.append(f"{stamp // 1_000_000_000}.{stamp % 1_000_000_000:09d} " + " ".join(f"{v:.9f}" for v in [*p, *q]))
    path.write_text("\n".join(lines) + "\n")


def write_run(directory, offset, scale=1.03, angle=0.4, noise_m=0.0):
    """Wheel odometry and the VIO's IMU poses of a body point `offset` (body frame) from the wheel's."""
    r_ib, t_ib = parse_pose(IMU_FROM_BODY)
    wheel = drive()
    stamps = 1_000_000_000 + np.arange(len(wheel), dtype=np.int64) * 50_000_000
    wheel_p = np.column_stack([wheel[:, :2], np.zeros(len(wheel))])
    wheel_r = np.array([rz(yaw) for yaw in wheel[:, 2]])
    world = rz(angle)
    body_p = np.array([scale * world @ (p + r @ np.append(offset, 0.0)) for p, r in zip(wheel_p, wheel_r)])
    body_p += np.array([0.3, -1.2, 0.5])
    body_p[:, :2] += np.random.default_rng(1).normal(0.0, noise_m, (len(wheel), 2))
    imu_r = np.array([world @ r @ r_ib.T for r in wheel_r])
    imu_p = body_p - np.einsum("nij,j->ni", imu_r, t_ib)
    write_tum(directory / "odom.tum", stamps, wheel_p, wheel_r)
    write_tum(directory / "vio.tum", stamps + 2_000_000, imu_p, imu_r)
    return directory / "vio.tum", directory / "odom.tum"


class CalibrateWheelOffsetTests(unittest.TestCase):
    def test_recovers_the_offset_and_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            vio, odometry = write_run(Path(tmp), offset=np.array([0.041, 0.064]), noise_m=0.002)
            result = calibrate(vio, odometry, parse_pose(IMU_FROM_BODY))
        np.testing.assert_allclose(result["offset_m"], [0.041, 0.064], atol=1e-3)
        self.assertAlmostEqual(result["scale"], 1.03, places=3)
        self.assertLess(result["rms_m"], 0.003)
        self.assertGreater(result["rms_without_offset_m"], 0.02)
        self.assertAlmostEqual(result["turned_deg"], 270.0, delta=2.0)

    def test_corrected_extrinsic_moves_the_body_to_the_wheel_point(self):
        offset = np.array([0.02, -0.05])
        with tempfile.TemporaryDirectory() as tmp:
            vio, odometry = write_run(Path(tmp), offset=offset, scale=1.0)
            r_ib, t_ib = parse_pose(IMU_FROM_BODY)
            result = calibrate(vio, odometry, (r_ib, t_ib))
            corrected = result["imu_from_body"]
            again = calibrate(vio, odometry, corrected)
        np.testing.assert_allclose(corrected[0], r_ib)
        np.testing.assert_allclose(r_ib.T @ (t_ib - corrected[1]), [*offset, 0.0], atol=1e-4)
        np.testing.assert_allclose(again["offset_m"], [0.0, 0.0], atol=1e-4)

    def test_main_prints_the_corrected_extrinsic(self):
        with tempfile.TemporaryDirectory() as tmp:
            vio, odometry = write_run(Path(tmp), offset=np.array([0.013, 0.062]))
            out = io.StringIO()
            with redirect_stdout(out):
                main(["--vio", str(vio), "--odometry", str(odometry), "--imu-from-body", IMU_FROM_BODY])
        last = out.getvalue().strip().splitlines()[-1]
        self.assertTrue(last.startswith("--imu-from-body="))
        rotation, translation = parse_pose(last.split("=", 1)[1])
        # The printed form is accepted back, though it starts with a minus sign.
        with tempfile.TemporaryDirectory() as tmp:
            vio, odometry = write_run(Path(tmp), offset=np.array([0.013, 0.062]))
            with redirect_stdout(io.StringIO()):
                main(["--vio", str(vio), "--odometry", str(odometry), last])
        expected_r, expected_t = parse_pose(IMU_FROM_BODY)
        np.testing.assert_allclose(rotation, expected_r, atol=2e-4)
        np.testing.assert_allclose(translation, expected_t - expected_r @ [0.013, 0.062, 0.0], atol=1e-3)

    def test_quaternion_round_trip(self):
        q = np.array([-0.5030, -0.4927, -0.5074, 0.4967])
        rotation = quat_matrix(*q / np.linalg.norm(q))
        np.testing.assert_allclose(quat_matrix(*rotation_quat(rotation)), rotation, atol=1e-9)
        self.assertEqual(format_pose(np.eye(3), np.array([0.1, 0.0, -0.2])), "0.1000,0.0000,-0.2000,0.0000,0.0000,0.0000,1.0000")

    def test_reads_stamps_in_exponent_notation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slam.tum"
            path.write_text("# timestamp tx ty tz qx qy qz qw\n"
                            "1.560000083949067116e9 0 0 0 0 0 0 1\n"
                            "1560000084.000000001 1 0 0 0 0 0 1\n")
            stamps, positions, _ = read_tum(path)
        self.assertEqual(stamps.tolist(), [1560000083949067116, 1560000084000000001])
        self.assertEqual(positions[1].tolist(), [1.0, 0.0, 0.0])

    def test_too_few_pairs_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            vio, odometry = write_run(Path(tmp), offset=np.zeros(2))
            lines = odometry.read_text().splitlines()
            shifted = []
            for line in lines:
                stamp, rest = line.split(" ", 1)
                shifted.append(f"{float(stamp) + 1000.0:.9f} {rest}")
            odometry.write_text("\n".join(shifted) + "\n")
            with self.assertRaises(SystemExit):
                calibrate(vio, odometry, parse_pose(IMU_FROM_BODY))


if __name__ == "__main__":
    unittest.main()
