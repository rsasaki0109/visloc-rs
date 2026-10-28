#!/usr/bin/env python3
"""Convert an OpenLORIS-Scene sequence's T265 data to the EuRoC/ASL layout.

OpenLORIS-Scene (https://lifelong-robotic-vision.github.io/dataset/scene)
records a wheeled robot with a RealSense T265 (two fisheye cameras and an
IMU), a D435i and wheel odometry. This writes what the Basalt examples
read:

  OUT/mav0/cam0, cam1/    fisheye1, fisheye2 resampled to a Double Sphere
                          camera, data.csv with nanosecond stamps
  OUT/mav0/imu0/data.csv  T265 gyro at its own 200 Hz stamps, with the 62.5
                          Hz accelerometer linearly interpolated to them
  OUT/mav0/state_groundtruth_estimate0/data.csv
                          ground truth (base_link's pose) moved to the T265
                          IMU (EuRoC columns; velocity and biases 0)
  OUT/odom.tum            wheel odometry of base_link, seconds
  OUT/calibration.json    Basalt calibration: the Double Sphere intrinsics of
                          the resampled images, T_imu_cam from
                          trans_matrix.yaml, and IMU noise
  OUT/extrinsics.json     base_link <- T265 IMU and the resampling's
                          parameters

The Basalt port models cameras as Double Sphere only, and no Double Sphere
camera reproduces the T265's Kannala-Brandt lenses over their whole field
of view (fitted out to 85 deg off axis, 1.9 px RMS). So each image is
resampled: every pixel of a Double Sphere camera close to the lens (fitted
within 60 deg, 0.15 px RMS, then used as is) is unprojected to its ray,
projected with the official Kannala-Brandt model and sampled bilinearly.
The calibration is then exact by construction and the field of view is kept.

`--sequence` is an extracted sequence directory (with fisheye1/,
sensors.yaml, ...), or a sequence's .7z from the official package, from
which only the T265 images and the text files are extracted (needs py7zr).
Dataset images stay outside the repository and remain subject to the
dataset's CC BY-ND 4.0 terms.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

T265_FILES = re.compile(
    r"(^|/)(fisheye[12]/.*\.png|fisheye[12]\.txt|t265_(accelerometer|gyroscope)\.txt|"
    r"odom\.txt|groundtruth\.txt|sensors\.yaml|trans_matrix\.yaml)$"
)

# Starting values for the T265's IMU (Bosch BMI055) in Basalt's units; the
# accelerometer runs at 62.5 Hz and is interpolated, so it is given more noise
# than EuRoC's ADIS16448 (0.016 / 0.000282).
DEFAULT_ACCEL_NOISE_STD = 0.03
DEFAULT_GYRO_NOISE_STD = 0.003
DEFAULT_ACCEL_BIAS_STD = 0.001
DEFAULT_GYRO_BIAS_STD = 0.0001


def num(value) -> str:
    """Shortest round-trip text of a float (numpy scalars included)."""
    return repr(float(value))


def stamp_ns(text: str) -> int:
    """Seconds with a decimal fraction, exactly, as integer nanoseconds."""
    whole, _, fraction = text.strip().partition(".")
    return int(whole) * 1_000_000_000 + int((fraction + "000000000")[:9])


# --- OpenCV YAML (the subset these files use) -------------------------------


def sensor_matrix(text: str, sensor: str, field: str) -> list[float]:
    start = text.index(f"\n{sensor}:")
    end = text.find("\n", start + 1)
    next_sensor = re.search(r"\n[A-Za-z0-9_]+:\s*\n", text[end:])
    section = text[start : end + next_sensor.start()] if next_sensor else text[start:]
    match = re.search(field + r":\s*!!opencv-matrix.*?data:\s*\[([^\]]*)\]", section, re.S)
    if not match:
        raise ValueError(f"sensors.yaml: no {field} for {sensor}")
    return [float(v) for v in match.group(1).split(",") if v.strip()]


def transforms(text: str) -> dict[tuple[str, str], np.ndarray]:
    """(parent, child) -> T_parent_child from trans_matrix.yaml."""
    result = {}
    pattern = re.compile(
        r"parent_frame:\s*(\S+)\s*child_frame:\s*(\S+)\s*matrix:.*?data:\s*\[([^\]]*)\]", re.S
    )
    for parent, child, data in pattern.findall(text):
        values = [float(v) for v in data.split(",") if v.strip()]
        result[(parent, child)] = np.array(values).reshape(4, 4)
    return result


# --- Camera models ------------------------------------------------------------


def kb4_project(params, points):
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    x, y, z = points.T
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    t2 = theta * theta
    d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = np.where(r > 1e-12, d / np.maximum(r, 1e-12), 1.0 / np.maximum(z, 1e-12))
    return np.stack([fx * x * scale + cx, fy * y * scale + cy], axis=1)


def kb4_unproject(params, pixels):
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    mx = (pixels[:, 0] - cx) / fx
    my = (pixels[:, 1] - cy) / fy
    d = np.hypot(mx, my)
    theta = d.copy()
    for _ in range(20):
        t2 = theta * theta
        f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - d
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + 9 * t2 * k4)))
        theta = theta - f / df
    s = np.where(d > 1e-12, np.sin(theta) / np.maximum(d, 1e-12), 1.0)
    return np.stack([mx * s, my * s, np.cos(theta)], axis=1), theta


def ds_project(params, points):
    fx, fy, cx, cy, xi, alpha = params
    x, y, z = points.T
    d1 = np.sqrt(x * x + y * y + z * z)
    k = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + k * k)
    denom = alpha * d2 + (1 - alpha) * k
    return np.stack([fx * x / denom + cx, fy * y / denom + cy], axis=1)


def ds_unproject(params, pixels):
    """Unit rays of pixels (Basalt's DoubleSphereCamera::unproject), and
    whether each is in the model's domain."""
    fx, fy, cx, cy, xi, alpha = params
    mx = (pixels[:, 0] - cx) / fx
    my = (pixels[:, 1] - cy) / fy
    r2 = mx * mx + my * my
    valid = np.ones(len(pixels), dtype=bool)
    if alpha > 0.5:
        valid &= r2 <= 1.0 / (2 * alpha - 1)
    r2c = np.where(valid, r2, 0.0)
    mz = (1 - alpha * alpha * r2c) / (alpha * np.sqrt(1 - (2 * alpha - 1) * r2c) + 1 - alpha)
    k = (mz * xi + np.sqrt(mz * mz + (1 - xi * xi) * r2c)) / (mz * mz + r2c)
    rays = np.stack([k * mx, k * my, k * mz - xi], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True), valid


def resampling_map(kb4, ds, width, height):
    """Source (KB4) pixel coordinates for every target (DS) pixel."""
    us, vs = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    pixels = np.stack([us.ravel(), vs.ravel()], axis=1)
    rays, valid = ds_unproject(ds, pixels)
    source = kb4_project(kb4, rays)
    inside = valid & (source[:, 0] >= 0) & (source[:, 0] <= width - 1) & \
        (source[:, 1] >= 0) & (source[:, 1] <= height - 1)
    return source.reshape(height, width, 2), inside.reshape(height, width)


def resample(image, source, inside):
    """Bilinear samples of an 8-bit image at `source`; 0 outside."""
    h, w = image.shape
    x = np.clip(source[..., 0], 0, w - 1.000001)
    y = np.clip(source[..., 1], 0, h - 1.000001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    img = image.astype(np.float32)
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bottom = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    out = top * (1 - fy) + bottom * fy
    return np.where(inside, np.clip(np.rint(out), 0, 255), 0).astype(np.uint8)


def fit_double_sphere(kb4, width, height, max_theta_deg=60.0):
    """Double Sphere parameters reproducing a Kannala-Brandt camera."""
    us, vs = np.meshgrid(np.linspace(0, width - 1, 80), np.linspace(0, height - 1, 80))
    pixels = np.stack([us.ravel(), vs.ravel()], axis=1)
    rays, theta = kb4_unproject(kb4, pixels)
    keep = theta < math.radians(max_theta_deg)
    pixels, rays = pixels[keep], rays[keep]
    fx, fy, cx, cy = kb4[:4]
    params = np.array([fx, fy, cx, cy, 0.0, 0.6])

    def residual(p):
        return (ds_project(p, rays) - pixels).ravel()

    damping = 1e-3
    current = residual(params)
    for _ in range(200):
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
        trial = residual(candidate)
        if np.isfinite(trial).all() and trial @ trial < current @ current:
            params, current = candidate, trial
            damping = max(damping / 3, 1e-9)
            if np.abs(delta).max() < 1e-10:
                break
        else:
            damping *= 4
    errors = np.linalg.norm(current.reshape(-1, 2), axis=1)
    return params, float(np.sqrt(np.mean(errors**2))), float(errors.max()), int(len(pixels))


# --- Poses ------------------------------------------------------------------


def quat_to_matrix(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r):
    """(qx, qy, qz, qw) of a rotation matrix."""
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q) * (1 if q[3] >= 0 else -1)


def pose(t, q):
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(*q)
    m[:3, 3] = t
    return m


def orthonormalize(m):
    """The nearest rotation (the YAML matrices are rounded)."""
    u, _, vt = np.linalg.svd(m[:3, :3])
    out = m.copy()
    out[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return out


def transform_json(m):
    qx, qy, qz, qw = matrix_to_quat(m[:3, :3])
    px, py, pz = m[:3, 3]
    return {"px": px, "py": py, "pz": pz, "qx": qx, "qy": qy, "qz": qz, "qw": qw}


# --- Conversion ---------------------------------------------------------------


def read_rows(path: Path):
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            yield line.split()


def extract(archive: Path, destination: Path) -> Path:
    import py7zr  # optional dependency, only for .7z input

    with py7zr.SevenZipFile(archive) as z:
        targets = [name for name in z.getnames() if T265_FILES.search(name)]
    with py7zr.SevenZipFile(archive) as z:
        z.extract(path=destination, targets=targets)
    roots = {name.split("/")[0] for name in targets}
    if len(roots) != 1:
        raise SystemExit(f"{archive}: expected one sequence directory, found {sorted(roots)}")
    return destination / roots.pop()


def write_camera(sequence: Path, name: str, out: Path, source, inside) -> int:
    from PIL import Image

    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    rows = ["#timestamp [ns],filename"]
    for stamp, relative in read_rows(sequence / f"{name}.txt"):
        ns = stamp_ns(stamp)
        target = data / f"{ns}.png"
        if not target.exists():  # a rerun keeps the images already resampled
            image = np.asarray(Image.open(sequence / relative).convert("L"))
            Image.fromarray(resample(image, source, inside)).save(target)
        rows.append(f"{ns},{ns}.png")
    (out / "data.csv").write_text("\n".join(rows) + "\n")
    return len(rows) - 1


def write_imu(sequence: Path, out: Path) -> tuple[int, float, float]:
    gyro = np.array([[stamp_ns(r[0])] + [float(v) for v in r[1:4]]
                     for r in read_rows(sequence / "t265_gyroscope.txt")], dtype=object)
    accel = np.array([[stamp_ns(r[0])] + [float(v) for v in r[1:4]]
                      for r in read_rows(sequence / "t265_accelerometer.txt")], dtype=object)
    accel_t = np.array([int(t) for t in accel[:, 0]], dtype=np.int64)
    accel_v = np.array(accel[:, 1:], dtype=float)
    rows = ["#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"]
    previous = None
    for t, gx, gy, gz in gyro:
        t = int(t)
        if t < accel_t[0] or t > accel_t[-1] or (previous is not None and t <= previous):
            continue
        a = [np.interp(t - accel_t[0], accel_t - accel_t[0], accel_v[:, i]) for i in range(3)]
        rows.append(",".join([str(t)] + [num(v) for v in (gx, gy, gz, *a)]))
        previous = t
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.csv").write_text("\n".join(rows) + "\n")
    gyro_rate = (len(gyro) - 1) / ((int(gyro[-1, 0]) - int(gyro[0, 0])) * 1e-9)
    accel_rate = (len(accel_t) - 1) / ((accel_t[-1] - accel_t[0]) * 1e-9)
    return len(rows) - 1, gyro_rate, accel_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=Path, required=True,
                        help="extracted sequence directory, or its .7z")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extract-to", type=Path,
                        help="where a .7z is extracted (default: next to it)")
    parser.add_argument("--accel-noise-std", type=float, default=DEFAULT_ACCEL_NOISE_STD)
    parser.add_argument("--gyro-noise-std", type=float, default=DEFAULT_GYRO_NOISE_STD)
    parser.add_argument("--accel-bias-std", type=float, default=DEFAULT_ACCEL_BIAS_STD)
    parser.add_argument("--gyro-bias-std", type=float, default=DEFAULT_GYRO_BIAS_STD)
    args = parser.parse_args()

    sequence = args.sequence
    if sequence.suffix == ".7z":
        sequence = extract(sequence, args.extract_to or sequence.parent)
    sensors = (sequence / "sensors.yaml").read_text()
    frames = transforms((sequence / "trans_matrix.yaml").read_text())
    mav0 = args.out / "mav0"

    # Cameras: a Double Sphere camera close to each official Kannala-Brandt
    # one, which the images are resampled to.
    intrinsics, fits, maps = [], [], []
    for name in ("t265_fisheye1_optical_frame", "t265_fisheye2_optical_frame"):
        fx, cx, fy, cy = sensor_matrix(sensors, name, "intrinsics")  # OpenLORIS order
        k1, k2, k3, k4 = sensor_matrix(sensors, name, "distortion_coefficients")[:4]
        width = int(re.search(rf"\n{name}:.*?width:\s*(\d+)", sensors, re.S).group(1))
        height = int(re.search(rf"\n{name}:.*?height:\s*(\d+)", sensors, re.S).group(1))
        ds, rms, worst, samples = fit_double_sphere([fx, fy, cx, cy, k1, k2, k3, k4], width, height)
        intrinsics.append({"camera_type": "ds", "intrinsics": dict(zip(
            ("fx", "fy", "cx", "cy", "xi", "alpha"), map(float, ds)))})
        source, inside = resampling_map([fx, fy, cx, cy, k1, k2, k3, k4], ds, width, height)
        maps.append((source, inside))
        fits.append({"camera": name, "width": width, "height": height,
                     "kb4": [fx, fy, cx, cy, k1, k2, k3, k4], "ds": list(map(float, ds)),
                     "fit_within_60_deg_rms_px": rms, "fit_within_60_deg_max_px": worst,
                     "samples": samples, "resampled_pixels_with_data": float(inside.mean())})

    f1_f2 = orthonormalize(frames[("t265_fisheye1_optical_frame", "t265_fisheye2_optical_frame")])
    f1_imu = orthonormalize(frames[("t265_fisheye1_optical_frame", "t265_gyroscope")])
    base_f1 = orthonormalize(frames[("base_link", "t265_fisheye1_optical_frame")])
    imu_f1 = np.linalg.inv(f1_imu)
    imu_f2 = imu_f1 @ f1_f2
    base_imu = base_f1 @ f1_imu

    calibration = {"value0": {
        "T_imu_cam": [transform_json(imu_f1), transform_json(imu_f2)],
        "intrinsics": intrinsics,
        "resolution": [[f["width"], f["height"]] for f in fits],
        "vignette": [],
        "calib_accel_bias": [0.0] * 9,
        "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": 200.0,
        "accel_noise_std": [args.accel_noise_std] * 3,
        "gyro_noise_std": [args.gyro_noise_std] * 3,
        "accel_bias_std": [args.accel_bias_std] * 3,
        "gyro_bias_std": [args.gyro_bias_std] * 3,
        "T_mocap_world": transform_json(np.eye(4)),
        "T_imu_marker": transform_json(np.linalg.inv(base_imu)),  # ground truth is base_link
        "mocap_time_offset_ns": 0,
        "mocap_to_imu_offset_ns": 0,
        "cam_time_offset_ns": 0,
    }}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    images = [write_camera(sequence, name, mav0 / cam, *pixel_map)
              for (name, cam), pixel_map in zip((("fisheye1", "cam0"), ("fisheye2", "cam1")), maps)]
    imu_count, gyro_rate, accel_rate = write_imu(sequence, mav0 / "imu0")

    # Ground truth: base_link's pose, moved to the T265 IMU. (Read as the
    # mocap marker's pose instead, office1-1's VIO is 14 cm off rather than
    # 2.7 cm.)
    gt_rows = ["#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],"
               "q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],"
               "b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],"
               "b_a_RS_S_z [m s^-2]"]
    gt_count = 0
    if (sequence / "groundtruth.txt").exists():
        for row in read_rows(sequence / "groundtruth.txt"):
            world_imu = pose([float(v) for v in row[1:4]], [float(v) for v in row[4:8]]) @ base_imu
            qx, qy, qz, qw = matrix_to_quat(world_imu[:3, :3])
            p = world_imu[:3, 3]
            gt_rows.append(",".join([str(stamp_ns(row[0]))]
                                    + [num(v) for v in (*p, qw, qx, qy, qz)]) + ",0" * 9)
            gt_count += 1
        gt_dir = mav0 / "state_groundtruth_estimate0"
        gt_dir.mkdir(parents=True, exist_ok=True)
        (gt_dir / "data.csv").write_text("\n".join(gt_rows) + "\n")

    odom_rows = ["# timestamp tx ty tz qx qy qz qw (base_link in the wheel odometry frame)"]
    for row in read_rows(sequence / "odom.txt"):
        ns = stamp_ns(row[0])
        odom_rows.append(f"{ns // 1_000_000_000}.{ns % 1_000_000_000:09d} " + " ".join(row[1:8]))
    (args.out / "odom.tum").write_text("\n".join(odom_rows) + "\n")

    report = {
        "sequence": str(sequence),
        "images": {"cam0": images[0], "cam1": images[1]},
        "imu_samples": imu_count, "gyro_hz": gyro_rate, "accel_hz": accel_rate,
        "ground_truth_samples": gt_count, "odometry_samples": len(odom_rows) - 1,
        "double_sphere_fit": fits,
        "T_base_imu": base_imu.tolist(),
    }
    (args.out / "extrinsics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("images", "imu_samples", "gyro_hz", "accel_hz",
                                              "ground_truth_samples", "odometry_samples")}))
    for f in fits:
        print(f"{f['camera']}: Double Sphere {np.round(f['ds'], 4).tolist()}, "
              f"{f['resampled_pixels_with_data']:.1%} of the resampled image has data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
