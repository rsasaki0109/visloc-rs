#!/usr/bin/env python3
"""Convert a FoMo (Foret Montmorency) trajectory to the EuRoC/ASL layout.

FoMo (https://fomo.norlab.ulaval.ca, Boxan et al., IJRR 2026) records a
wheeled robot on the same boreal-forest trails over a year: snow and -19 C in
winter, dense vegetation in summer. It is public on AWS Open Data
(s3://fomo-dataset, CC BY 4.0) and needs no account. This writes what the
Basalt examples read:

  OUT/mav0/cam0, cam1/     ZED X left / right, grey, downscaled by --scale,
                           data.csv with nanosecond stamps (only stamps both
                           cameras have)
  OUT/mav0/imu0/data.csv   the chosen IMU (VectorNav by default) at 200 Hz
  OUT/calibration.json     Basalt calibration: the ZED X's rectified pinhole
                           intrinsics as a Double Sphere camera with xi = 0,
                           alpha = 0 (a pinhole), T_imu_cam from
                           calib/transforms.json, IMU noise
  OUT/gt_base.tum          the PPK-GNSS ground truth (positions only; its
                           quaternions are identity), in metres from its
                           first sample, seconds
  OUT/extrinsics.json      T_base_imu (base_link <- IMU), the UTM origin and
                           what was converted

FoMo's transforms.json lists {"to": A, "from": B, position, orientation};
the position is A's origin in B (zedx_right sits 0.12 m along zedx_left's
+x), so each entry is T_B_A. Checked against gravity: with this reading the
VectorNav's +z (up at rest) is the left camera's -y, the camera pitched
18 deg down, which is where the horizon sits in the images.

The ZED X images are already rectified (the published K is shared by both
cameras, no distortion); matched ORB rows still differ by about 2 px at full
resolution, which is the 0.3 deg residual rotation transforms.json gives
between the cameras, so the extrinsic is used as published.

`--trajectory` is an S3 prefix such as
`data/2025-08-20/blue_2025-08-20-11-25` (downloaded over HTTPS, images
converted on the fly so the 3.4 MB colour PNGs are never kept), or a local
directory with the same layout. Dataset images stay outside the repository
and remain subject to the dataset's CC BY 4.0 terms.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

BUCKET_URL = "https://fomo-dataset.s3.us-west-2.amazonaws.com"

# Basalt noise parameters per IMU, in the units of Basalt's calibration
# (continuous-time densities). Starting values: Kalibr's (Allan) densities
# from calib/kalibr-*.yaml, inflated for vibration on rough trails the way
# EuRoC's are, so they are not tuned on the ground truth.
IMU_NOISE = {
    "vectornav": {"accel_noise_std": 0.016, "gyro_noise_std": 0.0005,
                  "accel_bias_std": 0.001, "gyro_bias_std": 0.0001},
    "xsens": {"accel_noise_std": 0.016, "gyro_noise_std": 0.002,
              "accel_bias_std": 0.001, "gyro_bias_std": 0.0001},
}


def num(value) -> str:
    return repr(float(value))


def quat_to_matrix(qx, qy, qz, qw):
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r):
    """(qx, qy, qz, qw) of a rotation matrix, qw >= 0."""
    trace = np.trace(r)
    if trace > 0:
        s = (trace + 1.0) ** 0.5 * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = (1.0 + r[0, 0] - r[1, 1] - r[2, 2]) ** 0.5 * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = (1.0 + r[1, 1] - r[0, 0] - r[2, 2]) ** 0.5 * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = (1.0 + r[2, 2] - r[0, 0] - r[1, 1]) ** 0.5 * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q) * (1 if q[3] >= 0 else -1)


def transform_json(m):
    qx, qy, qz, qw = matrix_to_quat(m[:3, :3])
    px, py, pz = m[:3, 3]
    return {"px": px, "py": py, "pz": pz, "qx": qx, "qy": qy, "qz": qz, "qw": qw}


# --- Source access (S3 over HTTPS, or a local copy) ---------------------------


class Source:
    def __init__(self, trajectory: str):
        self.local = Path(trajectory) if Path(trajectory).is_dir() else None
        self.prefix = trajectory.strip("/")

    def read(self, relative: str) -> bytes:
        if self.local is not None:
            return (self.local / relative).read_bytes()
        url = f"{BUCKET_URL}/{urllib.parse.quote(self.prefix + '/' + relative)}"
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    return response.read()
            except OSError:
                if attempt == 4:
                    raise
        raise AssertionError("unreachable")

    def list(self, directory: str) -> list[str]:
        """File names (not paths) in `directory`, sorted."""
        if self.local is not None:
            return sorted(p.name for p in (self.local / directory).iterdir())
        names, token = [], None
        prefix = f"{self.prefix}/{directory}/"
        while True:
            query = {"list-type": "2", "prefix": prefix}
            if token:
                query["continuation-token"] = token
            with urllib.request.urlopen(f"{BUCKET_URL}/?{urllib.parse.urlencode(query)}",
                                        timeout=60) as response:
                text = response.read().decode()
            names += [k[len(prefix):] for k in re.findall(r"<Key>([^<]*)</Key>", text)]
            match = re.search(r"<NextContinuationToken>([^<]*)<", text)
            if not match:
                return sorted(names)
            token = match.group(1)


# --- Conversion ---------------------------------------------------------------


def frame_transforms(entries) -> dict[tuple[str, str], np.ndarray]:
    """(B, A) -> T_B_A for every {"to": A, "from": B} entry, and its inverse."""
    out = {}
    for e in entries:
        m = np.eye(4)
        o, p = e["orientation"], e["position"]
        m[:3, :3] = quat_to_matrix(o["x"], o["y"], o["z"], o["w"])
        m[:3, 3] = [p["x"], p["y"], p["z"]]
        out[(e["from"], e["to"])] = m
        out[(e["to"], e["from"])] = np.linalg.inv(m)
    return out


def chain(frames, *names) -> np.ndarray:
    m = np.eye(4)
    for a, b in zip(names, names[1:]):
        m = m @ frames[(a, b)]
    return m


def convert_image(data: bytes, scale: float):
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("L")
    if scale != 1.0:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.Resampling.BOX)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory", required=True,
                        help="S3 prefix (data/<date>/<name>) or a local directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imu", choices=sorted(IMU_NOISE), default="vectornav")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="image downscale (0.5: 1920x1200 -> 960x600)")
    parser.add_argument("--start-s", type=float, default=0.0,
                        help="skip this many seconds from the first image")
    parser.add_argument("--duration-s", type=float, help="keep only this many seconds")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    source = Source(args.trajectory)
    mav0 = args.out / "mav0"
    args.out.mkdir(parents=True, exist_ok=True)

    # Cameras: only stamps both cameras have.
    left = {n[:-4] for n in source.list("zedx_left") if n.endswith(".png")}
    right = {n[:-4] for n in source.list("zedx_right") if n.endswith(".png")}
    stamps_us = sorted(int(s) for s in left & right)
    t0 = stamps_us[0] + int(args.start_s * 1e6)
    t1 = t0 + int(args.duration_s * 1e6) if args.duration_s else stamps_us[-1]
    stamps_us = [s for s in stamps_us if t0 <= s <= t1]

    calib_left = json.loads(source.read("calib/zedx_left.json"))
    calib_right = json.loads(source.read("calib/zedx_right.json"))
    frames = frame_transforms(json.loads(source.read("calib/transforms.json")))
    cam_r_l = frames[("zedx_left", "zedx_right")]  # T_left_right
    left_imu = frames[("zedx_left", args.imu)]  # T_left_imu
    imu_left = np.linalg.inv(left_imu)
    imu_right = imu_left @ cam_r_l
    base_imu = chain(frames, "base_link", "robosense", "zedx_left", args.imu)

    def write(name_cam):
        name, cam = name_cam
        data = mav0 / cam / "data"
        data.mkdir(parents=True, exist_ok=True)
        size = [None]

        def one(stamp):
            target = data / f"{stamp * 1000}.png"
            if not target.exists():  # a rerun keeps what is already converted
                image = convert_image(source.read(f"{name}/{stamp}.png"), args.scale)
                tmp = target.with_suffix(".tmp.png")
                image.save(tmp)
                tmp.rename(target)
            return target

        with cf.ThreadPoolExecutor(args.workers) as pool:
            for i, target in enumerate(pool.map(one, stamps_us)):
                if size[0] is None:
                    from PIL import Image
                    size[0] = Image.open(target).size
                if i % 500 == 0:
                    print(f"{cam}: {i}/{len(stamps_us)}", file=sys.stderr, flush=True)
        rows = ["#timestamp [ns],filename"] + [f"{s * 1000},{s * 1000}.png" for s in stamps_us]
        (mav0 / cam / "data.csv").write_text("\n".join(rows) + "\n")
        return size[0]

    sizes = [write(nc) for nc in (("zedx_left", "cam0"), ("zedx_right", "cam1"))]

    intrinsics, resolution = [], []
    for calib, (width, height) in zip((calib_left, calib_right), sizes):
        p = calib["p"]
        s = width / 1920.0
        # Pixel centres: (c + 0.5) * s - 0.5.
        intrinsics.append({"camera_type": "ds", "intrinsics": {
            "fx": p[0] * s, "fy": p[5] * s, "cx": (p[2] + 0.5) * s - 0.5,
            "cy": (p[6] + 0.5) * s - 0.5, "xi": 0.0, "alpha": 0.0}})
        resolution.append([width, height])

    noise = IMU_NOISE[args.imu]
    calibration = {"value0": {
        "T_imu_cam": [transform_json(imu_left), transform_json(imu_right)],
        "intrinsics": intrinsics,
        "resolution": resolution,
        "vignette": [],
        "calib_accel_bias": [0.0] * 9,
        "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": 200.0,
        "accel_noise_std": [noise["accel_noise_std"]] * 3,
        "gyro_noise_std": [noise["gyro_noise_std"]] * 3,
        "accel_bias_std": [noise["accel_bias_std"]] * 3,
        "gyro_bias_std": [noise["gyro_bias_std"]] * 3,
        "T_mocap_world": transform_json(np.eye(4)),
        "T_imu_marker": transform_json(np.linalg.inv(base_imu)),
        "mocap_time_offset_ns": 0,
        "mocap_to_imu_offset_ns": 0,
        "cam_time_offset_ns": 0,
    }}
    (args.out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    # IMU: keep a margin around the images.
    imu = np.loadtxt(io.BytesIO(source.read(f"{args.imu}.csv")), delimiter=",", skiprows=1)
    keep = (imu[:, 0] >= t0 - 1_000_000) & (imu[:, 0] <= t1 + 1_000_000)
    imu = imu[keep]
    imu = imu[np.concatenate([[True], np.diff(imu[:, 0]) > 0])]
    rows = ["#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
            "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"]
    rows += [",".join([str(int(r[0]) * 1000)] + [num(v) for v in r[1:7]]) for r in imu]
    (mav0 / "imu0").mkdir(parents=True, exist_ok=True)
    (mav0 / "imu0" / "data.csv").write_text("\n".join(rows) + "\n")

    # Ground truth: positions only, from the first sample.
    gt = np.loadtxt(io.BytesIO(source.read("gt.txt")))
    gt = gt[(gt[:, 0] >= t0 * 1e-6 - 1) & (gt[:, 0] <= t1 * 1e-6 + 1)]
    origin = gt[0, 1:4].copy()
    lines = ["# timestamp x y z qx qy qz qw (PPK-GNSS positions, UTM minus utm_origin; "
             "orientation not provided)"]
    for r in gt:
        p = r[1:4] - origin
        lines.append(f"{r[0]:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 0 0 1")
    (args.out / "gt_base.tum").write_text("\n".join(lines) + "\n")

    report = {
        "trajectory": args.trajectory, "imu": args.imu, "scale": args.scale,
        "images": len(stamps_us), "first_image_us": stamps_us[0], "last_image_us": stamps_us[-1],
        "imu_samples": len(imu), "ground_truth_samples": len(gt),
        "ground_truth_length_m": float(np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1).sum()),
        "utm_origin": origin.tolist(),
        "T_base_imu": base_imu.tolist(),
        "T_left_right": cam_r_l.tolist(),
    }
    (args.out / "extrinsics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("images", "imu_samples", "ground_truth_samples",
                                              "ground_truth_length_m")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
