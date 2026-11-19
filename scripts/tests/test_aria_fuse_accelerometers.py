import pathlib
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from aria_fuse_accelerometers import fuse_accelerometers


class AccelerometerFusionTest(unittest.TestCase):
    def test_lever_arm_rotation_rectification_and_delay(self):
        rotation = Rotation.from_rotvec([0.2, 0.1, -0.3])
        lever = np.array([0.05, 0.1, -0.02])
        bias = np.array([0.1, -0.2, 0.3])
        matrix = np.diag([0.99, 1.01, 1.02])

        def imu(label, rot, t):
            q = rot.as_quat()
            return {"Label": label, "T_Device_Imu": {"UnitQuaternion": [q[3], q[:3]], "Translation": t},
                    "Accelerometer": {"Bias": {"Offset": bias}, "Model": {"RectificationMatrix": matrix}}}

        calibration = {"ImuCalibrations": [imu("imu-right", Rotation.identity(), np.zeros(3)),
                                           imu("imu-left", rotation, lever)]}
        time_ns = 10**14 + np.arange(10) * 10**6
        sec = (time_ns - time_ns[0]) * 1e-9
        alpha = np.array([10.0, -2.0, 1.5])
        gyro = np.array([3.0, -1.0, 2.0]) + sec[:, None] * alpha
        reference = np.tile([0.2, 0.4, 9.6], (len(sec), 1))
        left = reference + np.cross(alpha, lever) + np.cross(gyro, np.cross(gyro, lever))
        left_raw = rotation.inv().apply(left) @ matrix.T + bias
        right_offset = np.array([0.02, -0.01, 0.04])
        right = np.c_[time_ns, gyro, reference + right_offset]
        left_indices = np.array([1, 2, 3, 4, 8])
        fused, overlap = fuse_accelerometers(right, time_ns[left_indices] + 200, left_raw[left_indices], calibration,
                                            left_delay_ns=200, gyro_smoothing_ms=0)
        np.testing.assert_array_equal(overlap, [False, True, True, True, True, True, True, True, True, False])
        np.testing.assert_array_equal(fused[:, :4], right[:, :4])
        np.testing.assert_array_equal(fused[~overlap, 4:], right[~overlap, 4:])
        np.testing.assert_allclose(fused[left_indices, 4:], reference[left_indices] + right_offset / 2, atol=1e-12)

        # A missing run longer than four sample periods falls back to the right accelerometer.
        indices = np.array([1, 2, 3, 8, 9])
        fused, overlap = fuse_accelerometers(right, time_ns[indices], left_raw[indices], calibration,
                                            gyro_smoothing_ms=0)
        np.testing.assert_array_equal(fused[4:8], right[4:8])
        self.assertFalse(overlap[4:8].any())


if __name__ == "__main__":
    unittest.main()
