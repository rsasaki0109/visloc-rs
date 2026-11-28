#!/usr/bin/env python3
"""Convert a Project Aria recording (VRS + MPS SLAM trajectory) to EuRoC.

Used for InCrowd-VI (pedestrian-rich indoor sequences recorded with Aria
glasses, https://github.com/banafshebamdad/InCrowd-VI): the two global-shutter
SLAM cameras become cam0/cam1, one IMU becomes imu0, and the MPS closed-loop
trajectory (device time) becomes the ground truth, moved to that IMU.

The SLAM cameras use Aria's Fisheye624 model. As for OpenLORIS
(`openloris_to_euroc.py`), a Double Sphere model is fitted to each camera's
rays within --max-theta-deg of its axis, and the images are resampled into
it, so the Basalt pipeline sees exact Double Sphere cameras.

    python3 scripts/aria_vrs_to_euroc.py --vrs G6_loop.vrs \\
        --trajectory mps_G6_loop_vrs/slam/closed_loop_trajectory.csv --out data/G6_loop

Needs `projectaria_tools` (pip). Writes mav0/{cam0,cam1,imu0}, the ground
truth in mav0/state_groundtruth_estimate0/data.csv, calibration.json (Basalt,
Double Sphere) and extrinsics.json (the fits).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openloris_to_euroc import (  # noqa: E402
    DEFAULT_ACCEL_BIAS_STD,
    DEFAULT_ACCEL_NOISE_STD,
    DEFAULT_GYRO_BIAS_STD,
    DEFAULT_GYRO_NOISE_STD,
    ds_project,
    ds_unproject,
    matrix_to_quat,
    resample,
    transform_json,
)


def fit_double_sphere_to_rays(pixels, rays, width, height):
    """Double Sphere parameters projecting `rays` onto `pixels`."""
    params = np.array([width / 2.5, width / 2.5, width / 2, height / 2, 0.0, 0.6])

    def residual(p):
        return (ds_project(p, rays) - pixels).ravel()

    damping = 1e-3
    current = residual(params)
    for _ in range(300):
        jacobian = np.empty((current.size, params.size))
        for i in range(params.size):
            step = 1e-6 * max(1.0, abs(params[i]))
            shifted = params.copy()
            shifted[i] += step
            jacobian[:, i] = (residual(shifted) - current) / step
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal)), -gradient)
        candidate = params + delta
        candidate[5] = min(max(candidate[5], 0.0), 0.999)
        trial = residual(candidate)
        if np.isfinite(trial).all() and trial @ trial < current @ current:
            params, current = candidate, trial
            damping = max(damping / 3, 1e-9)
            if np.abs(delta).max() < 1e-10:
                break
        else:
            damping *= 4
    errors = np.linalg.norm(current.reshape(-1, 2), axis=1)
    return params, float(np.sqrt(np.mean(errors**2))), float(errors.max())


def camera_model(calib, width, height, max_theta_deg):
    """The Double Sphere fit of an Aria camera and the resampling map."""
    us, vs = np.meshgrid(np.linspace(0, width - 1, 80), np.linspace(0, height - 1, 60))
    samples = np.stack([us.ravel(), vs.ravel()], axis=1)
    pixels, rays = [], []
    for pixel in samples:
        ray = calib.unproject(pixel)
        if ray is None:
            continue
        ray = np.asarray(ray, dtype=float)
        ray /= np.linalg.norm(ray)
        if np.degrees(np.arccos(np.clip(ray[2], -1, 1))) < max_theta_deg:
            pixels.append(pixel)
            rays.append(ray)
    ds, rms, worst = fit_double_sphere_to_rays(np.array(pixels), np.array(rays), width, height)
    grid_u, grid_v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    target = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)
    target_rays, valid = ds_unproject(ds, target)
    source = np.full((len(target), 2), -1.0)
    for i in np.flatnonzero(valid):
        projected = calib.project(target_rays[i])
        if projected is not None:
            source[i] = projected
    inside = valid & (source[:, 0] >= 0) & (source[:, 0] <= width - 1) & \
        (source[:, 1] >= 0) & (source[:, 1] <= height - 1)
    fit = {"ds": list(map(float, ds)), "fit_rms_px": rms, "fit_max_px": worst,
           "samples": len(pixels), "resampled_pixels_with_data": float(inside.mean())}
    return ds, source.reshape(height, width, 2), inside.reshape(height, width), fit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vrs", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True,
                        help="MPS slam/closed_loop_trajectory.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imu", default="imu-right", help="imu-right (1 kHz) or imu-left")
    parser.add_argument("--max-theta-deg", type=float, default=75.0)
    parser.add_argument("--accel-noise-std", type=float, default=DEFAULT_ACCEL_NOISE_STD)
    parser.add_argument("--gyro-noise-std", type=float, default=DEFAULT_GYRO_NOISE_STD)
    parser.add_argument("--accel-bias-std", type=float, default=DEFAULT_ACCEL_BIAS_STD)
    parser.add_argument("--gyro-bias-std", type=float, default=DEFAULT_GYRO_BIAS_STD)
    args = parser.parse_args()

    from projectaria_tools.core import data_provider  # noqa: PLC0415

    provider = data_provider.create_vrs_data_provider(str(args.vrs))
    device = provider.get_device_calibration()
    imu_calib = device.get_imu_calib(args.imu)
    device_imu = imu_calib.get_transform_device_imu().to_matrix()
    mav0 = args.out / "mav0"

    intrinsics, fits, t_imu_cam, maps = [], [], [], []
    for label in ("camera-slam-left", "camera-slam-right"):
        calib = device.get_camera_calib(label)
        width, height = (int(v) for v in calib.get_image_size())
        ds, source, inside, fit = camera_model(calib, width, height, args.max_theta_deg)
        intrinsics.append({"camera_type": "ds", "intrinsics": dict(zip(
            ("fx", "fy", "cx", "cy", "xi", "alpha"), map(float, ds)))})
        fits.append({"camera": label, "width": width, "height": height, **fit})
        t_imu_cam.append(np.linalg.inv(device_imu) @ calib.get_transform_device_camera().to_matrix())
        maps.append((source, inside))
        print(f"{label}: Double Sphere fit {fit['fit_rms_px']:.3f} px RMS "
              f"(max {fit['fit_max_px']:.3f}), {100 * fit['resampled_pixels_with_data']:.1f}% "
              "of the resampled pixels have data", file=sys.stderr)

    calibration = {"value0": {
        "T_imu_cam": [transform_json(m) for m in t_imu_cam],
        "intrinsics": intrinsics,
        "resolution": [[f["width"], f["height"]] for f in fits],
        "vignette": [],
        "calib_accel_bias": [0.0] * 9,
        "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": 1000.0 if args.imu == "imu-right" else 800.0,
        "accel_noise_std": [args.accel_noise_std] * 3,
        "gyro_noise_std": [args.gyro_noise_std] * 3,
        "accel_bias_std": [args.accel_bias_std] * 3,
        "gyro_bias_std": [args.gyro_bias_std] * 3,
        "T_mocap_world": transform_json(np.eye(4)),
        "T_imu_marker": transform_json(np.linalg.inv(device_imu)),
        "mocap_time_offset_ns": 0,
        "mocap_to_imu_offset_ns": 0,
        "cam_time_offset_ns": 0,
    }}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    # Images: the two SLAM cameras are triggered together; pair by index and
    # stamp both with the left camera's capture time (device clock).
    left = provider.get_stream_id_from_label("camera-slam-left")
    right = provider.get_stream_id_from_label("camera-slam-right")
    count = min(provider.get_num_data(left), provider.get_num_data(right))
    rows = {"cam0": [], "cam1": []}
    worst_skew = 0
    for cam in ("cam0", "cam1"):
        (mav0 / cam / "data").mkdir(parents=True, exist_ok=True)
    for i in range(count):
        image_l, record_l = provider.get_image_data_by_index(left, i)
        image_r, record_r = provider.get_image_data_by_index(right, i)
        stamp = int(record_l.capture_timestamp_ns)
        worst_skew = max(worst_skew, abs(int(record_r.capture_timestamp_ns) - stamp))
        for cam, image, (source, inside) in (("cam0", image_l, maps[0]), ("cam1", image_r, maps[1])):
            pixels = resample(image.to_numpy_array(), source, inside)
            Image.fromarray(pixels).save(mav0 / cam / "data" / f"{stamp}.png")
            rows[cam].append(f"{stamp},{stamp}.png")
    for cam in ("cam0", "cam1"):
        (mav0 / cam / "data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows[cam]) + "\n")

    imu_stream = provider.get_stream_id_from_label(args.imu)
    (mav0 / "imu0").mkdir(parents=True, exist_ok=True)
    lines = ["#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
             "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"]
    for i in range(provider.get_num_data(imu_stream)):
        sample = provider.get_imu_data_by_index(imu_stream, i)
        gyro = imu_calib.raw_to_rectified_gyro(np.asarray(sample.gyro_radsec))
        accel = imu_calib.raw_to_rectified_accel(np.asarray(sample.accel_msec2))
        lines.append(f"{int(sample.capture_timestamp_ns)}," + ",".join(f"{v:.9g}" for v in (*gyro, *accel)))
    (mav0 / "imu0" / "data.csv").write_text("\n".join(lines) + "\n")

    # Ground truth: MPS's device pose at tracking (device) time, moved to the IMU.
    gt = ["#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],"
          "v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],"
          "b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],"
          "b_a_RS_S_z [m s^-2]"]
    with args.trajectory.open() as stream:
        for row in csv.DictReader(stream):
            stamp = int(row["tracking_timestamp_us"]) * 1000
            q = [float(row[k]) for k in ("qx_world_device", "qy_world_device", "qz_world_device",
                                         "qw_world_device")]
            x, y, z, w = q
            rotation = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            world_device = np.eye(4)
            world_device[:3, :3] = rotation
            world_device[:3, 3] = [float(row[k]) for k in ("tx_world_device", "ty_world_device",
                                                            "tz_world_device")]
            world_imu = world_device @ device_imu
            qx, qy, qz, qw = matrix_to_quat(world_imu[:3, :3])
            px, py, pz = world_imu[:3, 3]
            gt.append(f"{stamp},{px:.9f},{py:.9f},{pz:.9f},{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}"
                      + ",0" * 9)
    out = mav0 / "state_groundtruth_estimate0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.csv").write_text("\n".join(gt) + "\n")

    (args.out / "extrinsics.json").write_text(json.dumps({
        "vrs": str(args.vrs), "images": count, "imu": args.imu, "imu_samples": len(lines) - 1,
        "ground_truth_samples": len(gt) - 1, "stereo_worst_skew_ns": worst_skew,
        "double_sphere_fit": fits,
        "T_device_imu": device_imu.tolist(),
    }, indent=2) + "\n")
    print(f"{count} stereo frames, {len(lines) - 1} IMU samples, {len(gt) - 1} ground-truth "
          f"poses, stereo skew <= {worst_skew} ns -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
