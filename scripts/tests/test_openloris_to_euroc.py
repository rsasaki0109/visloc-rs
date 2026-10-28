import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openloris_to_euroc as conv  # noqa: E402

# office1-1's fisheye1 (sensors.yaml): fx, fy, cx, cy, k1..k4.
KB4 = [284.98089599609375, 286.1023864746094, 425.244384765625, 398.46759033203125,
       -0.007304710801690817, 0.043499931693077087, -0.04128304123878479,
       0.007652460131794214]

TRANS_MATRIX_YAML = """%YAML:1.0
trans_matrix:
   -
      parent_frame: base_link
      child_frame: t265_fisheye1_optical_frame
      matrix: !!opencv-matrix
         rows: 4
         cols: 4
         dt: d
         data: [ 1., 0., 0., 0.1,
             0., 1., 0., 0.2,
             0., 0., 1., 0.3, 0., 0., 0.,
             1. ]
"""

SENSORS_YAML = """%YAML:1.0
t265_fisheye1_optical_frame:
   sensor_type: camera
   width: 848
   height: 800
   intrinsics: !!opencv-matrix
      rows: 1
      cols: 4
      dt: d
      data: [ 2.8e+02, 4.2e+02,
          2.9e+02, 4.0e+02 ]
   distortion_coefficients: !!opencv-matrix
      rows: 1
      cols: 5
      dt: d
      data: [ -7e-03, 4e-02, -4e-02, 7e-03, 0. ]
t265_fisheye2_optical_frame:
   sensor_type: camera
"""


class OpenLorisToEurocTests(unittest.TestCase):
    def test_stamps_are_exact_nanoseconds(self):
        self.assertEqual(conv.stamp_ns("1560000110.95848536"), 1_560_000_110_958_485_360)
        self.assertEqual(conv.stamp_ns("1560000110.947943"), 1_560_000_110_947_943_000)
        self.assertEqual(conv.stamp_ns("12"), 12_000_000_000)

    def test_kannala_brandt_unprojection_inverts_projection(self):
        pixels = np.array([[425.0, 398.0], [10.0, 400.0], [700.0, 100.0], [430.0, 790.0]])
        rays, theta = conv.kb4_unproject(KB4, pixels)
        self.assertGreater(math.degrees(theta[1]), 90.0)  # the lens sees behind its side
        np.testing.assert_allclose(conv.kb4_project(KB4, rays), pixels, atol=1e-6)

    def test_double_sphere_unprojection_inverts_projection(self):
        ds, rms, _, _ = conv.fit_double_sphere(KB4, 848, 800)
        self.assertLess(rms, 0.3)
        grid = np.stack(np.meshgrid(np.arange(0, 848, 13.0), np.arange(0, 800, 13.0)), -1)
        pixels = grid.reshape(-1, 2)
        rays, valid = conv.ds_unproject(ds, pixels)
        self.assertGreater(valid.mean(), 0.9)
        np.testing.assert_allclose(conv.ds_project(ds, rays[valid]), pixels[valid], atol=1e-8)

    def test_resampling_samples_the_kannala_brandt_image_at_each_ray(self):
        ds, _, _, _ = conv.fit_double_sphere(KB4, 848, 800)
        source, inside = conv.resampling_map(KB4, ds, 848, 800)
        self.assertGreater(inside.mean(), 0.95)
        # A ramp image resamples to the source x coordinate.
        ramp = np.tile(np.arange(848) * 255.0 / 847.0, (800, 1)).astype(np.uint8)
        out = conv.resample(ramp, source, inside)
        y, x = 400, 600
        self.assertLessEqual(abs(int(out[y, x]) - source[y, x, 0] * 255.0 / 847.0), 1.5)
        self.assertEqual(out[0, 0], 0)  # outside the lens

    def test_yaml_parsing(self):
        frames = conv.transforms(TRANS_MATRIX_YAML)
        matrix = frames[("base_link", "t265_fisheye1_optical_frame")]
        np.testing.assert_allclose(matrix[:3, 3], [0.1, 0.2, 0.3])
        self.assertEqual(conv.sensor_matrix(SENSORS_YAML, "t265_fisheye1_optical_frame",
                                            "intrinsics"), [280.0, 420.0, 290.0, 400.0])
        self.assertEqual(conv.sensor_matrix(SENSORS_YAML, "t265_fisheye1_optical_frame",
                                            "distortion_coefficients")[:2], [-7e-03, 4e-02])

    def test_imu_interpolates_the_accelerometer_to_the_gyro_stamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp)
            (seq / "t265_gyroscope.txt").write_text(
                "#Time Gx Gy Gz\n1.000 0 0 0\n1.005 0.1 0 0\n1.010 0.2 0 0\n1.100 0.3 0 0\n")
            (seq / "t265_accelerometer.txt").write_text(
                "#Time Ax Ay Az\n1.000 0 9 0\n1.016 1.6 9 0\n1.032 3.2 9 0\n")
            count, _, _ = conv.write_imu(seq, seq / "imu0")
            rows = (seq / "imu0" / "data.csv").read_text().splitlines()[1:]
        self.assertEqual(count, 3)  # 1.100 is after the last accelerometer sample
        stamp, *values = rows[1].split(",")
        self.assertEqual(int(stamp), 1_005_000_000)
        self.assertAlmostEqual(float(values[0]), 0.1)
        self.assertAlmostEqual(float(values[3]), 0.5)  # 1.6 * 5 / 16

    def test_rotations_round_trip_through_quaternions(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            r = conv.quat_to_matrix(*q)
            back = conv.matrix_to_quat(r)
            np.testing.assert_allclose(conv.quat_to_matrix(*back), r, atol=1e-12)
            self.assertGreaterEqual(back[3], 0.0)
        json.dumps(conv.transform_json(np.eye(4)))  # plain floats


if __name__ == "__main__":
    unittest.main()
