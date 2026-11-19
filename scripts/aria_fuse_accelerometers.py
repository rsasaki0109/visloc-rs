#!/usr/bin/env python3
"""Fuse the two Aria accelerometers at the right IMU origin, preserving its gyro and timestamps."""

import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from aria_factory_imu_rectify import ASL_IMU_HEADER, sensor_model


def left_to_right(calibration):
    imus = {entry["Label"]: entry for entry in calibration["ImuCalibrations"]}
    left, right = imus["imu-left"], imus["imu-right"]

    def transform(imu):
        pose = imu["T_Device_Imu"]
        w, xyz = pose["UnitQuaternion"]
        return Rotation.from_quat([*xyz, w]), np.asarray(pose["Translation"])

    left_rotation, left_position = transform(left)
    right_rotation, right_position = transform(right)
    right_from_left = right_rotation.inv() * left_rotation
    lever_arm = right_rotation.inv().apply(left_position - right_position)
    return right_from_left, lever_arm


def fuse_accelerometers(right, left_ns, left_accel, calibration, left_delay_ns=0, gyro_smoothing_ms=5.0):
    """Rectify the left accelerometer, align it to the right clock and compensate for the lever arm.

    The right IMU CSV already contains factory-rectified, time-aligned measurements. The
    left stream uses the factory model and capture timestamps from the VRS. Positive delay
    means a left sample was stamped later than it was measured.
    """
    if len(right) < 3 or len(left_ns) < 2:
        raise ValueError("fusion needs at least three right samples and two left samples")
    if right.shape != (len(right), 7) or left_accel.shape != (len(left_ns), 3):
        raise ValueError("expected seven right IMU columns and three left acceleration columns")
    if not np.all(np.isfinite(right)) or not np.all(np.isfinite(left_accel)):
        raise ValueError("IMU samples must be finite")
    right_ns = right[:, 0].astype(np.int64)
    left_ns = np.asarray(left_ns, dtype=np.int64) - left_delay_ns
    if np.any(np.diff(right_ns) <= 0) or np.any(np.diff(left_ns) <= 0):
        raise ValueError("IMU timestamps must increase strictly")
    if not np.isfinite(gyro_smoothing_ms) or gyro_smoothing_ms < 0:
        raise ValueError("gyro smoothing must be finite and nonnegative")

    left_imu = next(imu for imu in calibration["ImuCalibrations"] if imu["Label"] == "imu-left")
    bias, rectification = sensor_model(left_imu["Accelerometer"])
    rotation, lever_arm = left_to_right(calibration)
    left = rotation.apply((left_accel - bias) @ rectification.T)

    right_time = (right_ns - right_ns[0]) * 1e-9
    gyro = right[:, 1:4]
    if gyro_smoothing_ms:
        gyro = gaussian_filter1d(gyro, gyro_smoothing_ms * 1e-3 / np.median(np.diff(right_time)), axis=0)
    alpha = np.gradient(gyro, right_time, axis=0, edge_order=2)
    rotational_accel = np.cross(alpha, lever_arm) + np.cross(gyro, np.cross(gyro, lever_arm))

    # Keep the right accelerometer at the recording edges and across gaps in the left stream.
    upper = np.searchsorted(left_ns, right_ns, side="left").clip(1, len(left_ns) - 1)
    lower = upper - 1
    gap = left_ns[upper] - left_ns[lower]
    within = (right_ns >= left_ns[lower]) & (right_ns <= left_ns[upper])
    exact = (right_ns == left_ns[lower]) | (right_ns == left_ns[upper])
    valid = within & (exact | (gap <= 4 * np.median(np.diff(left_ns))))
    if not np.any(valid):
        raise ValueError("no overlapping, closely spaced left accelerometer samples")
    fraction = (right_ns[valid] - left_ns[lower[valid]]) / gap[valid]
    aligned_left = left[lower[valid]] + fraction[:, None] * (left[upper[valid]] - left[lower[valid]])
    left_at_right = aligned_left - rotational_accel[valid]
    fused = right.copy()
    fused[valid, 4:7] = 0.5 * (right[valid, 4:7] + left_at_right)
    return fused, valid


def read_left_accelerometer(path):
    from projectaria_tools.core.data_provider import create_vrs_data_provider

    provider = create_vrs_data_provider(str(path))
    if provider is None:
        raise ValueError(f"could not open {path}")
    stream = provider.get_stream_id_from_label("imu-left")
    if stream is None:
        raise ValueError(f"{path} has no imu-left stream")
    total = provider.get_num_data(stream)
    timestamps, samples = np.empty(total, dtype=np.int64), np.empty((total, 3))
    i = 0
    for index in range(total):
        reading = provider.get_imu_data_by_index(stream, index)
        if reading.accel_valid:
            timestamps[i], samples[i] = reading.capture_timestamp_ns, reading.accel_msec2
            i += 1
    return timestamps[:i], samples[:i]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-imu", required=True, help="factory-rectified and time-aligned ASL imu0/data.csv")
    parser.add_argument("--vrs", required=True, help="raw VRS with the complete imu-left stream")
    parser.add_argument("--factory-calib", required=True, help="JSON extracted by aria_factory_imu_rectify.py")
    parser.add_argument("--out", required=True, help="ASL-format CSV with fused accelerometer")
    parser.add_argument("--left-delay-ns", type=int, default=0, help="left accelerometer timestamp delay in ns")
    parser.add_argument("--gyro-smoothing-ms", type=float, default=5.0, help="gyro Gaussian smoothing sigma in ms")
    args = parser.parse_args()
    with open(args.factory_calib) as f:
        calibration = json.load(f)
    right = np.loadtxt(args.right_imu, delimiter=",", comments="#", ndmin=2)
    try:
        left_time, left_accel = read_left_accelerometer(args.vrs)
        fused, valid = fuse_accelerometers(right, left_time, left_accel, calibration,
                                           args.left_delay_ns, args.gyro_smoothing_ms)
    except ValueError as error:
        parser.error(str(error))
    np.savetxt(args.out, fused, fmt=["%d", *["%.17g"] * 6], delimiter=",", header=ASL_IMU_HEADER.strip(), comments="")
    print(f"{args.out}: fused {valid.sum()}/{len(valid)} samples")


if __name__ == "__main__":
    main()
