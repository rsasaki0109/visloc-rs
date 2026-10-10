from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aria_factory_imu_rectify import extract_record, rectify_csv
import run_lamaria_test_submission as submission

GYRO_R = [[1.01, 0.002, 0.0], [0.0, 0.99, -0.003], [0.001, 0.0, 1.005]]
GYRO_B = [0.004, 0.002, -0.0014]
ACCEL_R = [[0.998, 0.0, 0.004], [-0.002, 1.003, 0.0], [0.0, 0.001, 0.997]]
ACCEL_B = [0.25, 0.21, 0.39]


def sensor(rect, bias):
    return {"Bias": {"Offset": bias}, "Model": {"RectificationMatrix": rect}}


CALIB = {
    "Serial": "TEST0001",
    "CameraCalibrations": [{"Label": "camera-slam-left"}],
    "ImuCalibrations": [
        {"Label": "imu-right", "Gyroscope": sensor(GYRO_R, GYRO_B),
         "Accelerometer": sensor(ACCEL_R, ACCEL_B)},
        {"Label": "imu-left", "Gyroscope": sensor(np.eye(3).tolist(), [0.0] * 3),
         "Accelerometer": sensor(np.eye(3).tolist(), [0.0] * 3)},
    ],
}


def vrs_blob():
    # Binary noise with stray braces around the JSON record, like a .vrs header.
    return b"VRS\x00{\x01\xff}{garbage" + json.dumps(CALIB).encode() + b"\x00{\x02tail"


def write_raw_csv(path, real_gyro, real_acc):
    raw_gyro = (np.array(GYRO_R) @ real_gyro.T).T + GYRO_B
    raw_acc = (np.array(ACCEL_R) @ real_acc.T).T + ACCEL_B
    with open(path, "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for i, (g, a) in enumerate(zip(raw_gyro, raw_acc)):
            f.write(f"{1000000 * i}," + ",".join(f"{x:.12g}" for x in (*g, *a)) + "\n")


class ExtractTests(unittest.TestCase):
    def test_finds_record_among_binary_noise(self):
        self.assertEqual(extract_record(vrs_blob()), CALIB)

    def test_missing_record_raises(self):
        with self.assertRaises(ValueError):
            extract_record(b"VRS\x00{no calibration here}")


class RectifyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.real_gyro = rng.normal(size=(50, 3))
        self.real_acc = rng.normal(size=(50, 3)) + [0.0, 0.0, 9.81]
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_raw_csv(self.dir / "raw.csv", self.real_gyro, self.real_acc)

    def tearDown(self):
        self.tmp.cleanup()

    def test_recovers_real_samples(self):
        n, bg, ba = rectify_csv(CALIB, self.dir / "raw.csv", self.dir / "out.csv")
        self.assertEqual(n, 50)
        np.testing.assert_allclose(bg, GYRO_B)
        np.testing.assert_allclose(ba, ACCEL_B)
        out = np.loadtxt(self.dir / "out.csv", delimiter=",", comments="#")
        np.testing.assert_allclose(out[:, 1:4], self.real_gyro, atol=1e-7)
        np.testing.assert_allclose(out[:, 4:7], self.real_acc, atol=1e-6)
        np.testing.assert_array_equal(out[:, 0], np.arange(50) * 1000000)
        self.assertTrue((self.dir / "out.csv").read_text().startswith("#timestamp [ns]"))

    def test_gyro_only_and_shift(self):
        rectify_csv(CALIB, self.dir / "raw.csv", self.dir / "out.csv", part="gyro", shift_ns=4000)
        raw = np.loadtxt(self.dir / "raw.csv", delimiter=",", comments="#")
        out = np.loadtxt(self.dir / "out.csv", delimiter=",", comments="#")
        np.testing.assert_allclose(out[:, 1:4], self.real_gyro, atol=1e-7)
        np.testing.assert_allclose(out[:, 4:7], raw[:, 4:7])
        np.testing.assert_array_equal(out[:, 0], raw[:, 0] + 4000)

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            rectify_csv(CALIB, self.dir / "raw.csv", self.dir / "out.csv", label="imu-middle")


class SubmissionTests(unittest.TestCase):
    def test_rectify_imu_replaces_csv_in_place(self):
        rng = np.random.default_rng(1)
        real_gyro, real_acc = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inner, work, calib_dir = tmp / "seq", tmp / "work", tmp / "calib"
            (inner / "mav0" / "imu0").mkdir(parents=True)
            work.mkdir()
            calib_dir.mkdir()
            write_raw_csv(inner / "mav0" / "imu0" / "data.csv", real_gyro, real_acc)

            def fake_head(url, dest, nbytes):
                self.assertTrue(url.endswith("/raw_data/test/sequence_1_1.vrs"))
                self.assertEqual(nbytes, 4 * 1024 * 1024)
                dest.write_bytes(vrs_blob())

            with mock.patch.object(submission, "download_head", fake_head):
                submission.rectify_imu("sequence_1_1", inner, work, calib_dir)

            out = np.loadtxt(inner / "mav0" / "imu0" / "data.csv", delimiter=",", comments="#")
            np.testing.assert_allclose(out[:, 1:4], real_gyro, atol=1e-7)
            np.testing.assert_allclose(out[:, 4:7], real_acc, atol=1e-6)
            self.assertEqual(sorted(p.name for p in (inner / "mav0" / "imu0").iterdir()),
                             ["data.csv"])
            saved = json.loads((calib_dir / "sequence_1_1_factory_calib.json").read_text())
            self.assertEqual(saved["Serial"], "TEST0001")


if __name__ == "__main__":
    unittest.main()
