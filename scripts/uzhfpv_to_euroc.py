#!/usr/bin/env python3
"""Convert a UZH-FPV drone racing sequence (Snapdragon stereo) to EuRoC/ASL.

UZH-FPV (https://fpv.ifi.uzh.ch, Delmerico et al., ICRA 2019) flies a racing
quadrotor fast and aggressively, indoors and outdoors, with a Leica laser
tracker ground truth. Input: an extracted `*_snapdragon_with_gt.zip`
(img/, left_images.txt, right_images.txt, imu.txt, groundtruth.txt) and the
matching Kalibr calibration directory (`*_calib_snapdragon/`, from
http://rpg.ifi.uzh.ch/datasets/uzh-fpv/calib/).

The cameras are equidistant (Kannala-Brandt) fisheyes; this port of Basalt
reads Double Sphere only, so each camera gets the Double Sphere model
closest to its Kannala-Brandt one and its images are resampled to it
(helpers shared with `openloris_to_euroc.py`). Image stamps are moved onto
the IMU clock with Kalibr's `timeshift_cam_imu` (t_imu = t_cam + shift).

Writes calibration.json, mav0/{cam0,cam1,imu0}, and for `fomo_eval.py`:
gt_base.tum (the ground-truth IMU poses, seconds, moved onto the IMU
clock by the lag that best aligns ground-truth and gyro angular rates) and extrinsics.json
(T_base_imu = identity, as the ground truth is the IMU's pose).

  python3 scripts/uzhfpv_to_euroc.py --sequence /data/fpv/indoor_forward_3 \\
      --calib /data/fpv/calib/indoor_forward_calib_snapdragon --out /data/fpv/euroc/indoor_forward_3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openloris_to_euroc import (  # noqa: E402
    fit_double_sphere, resample, resampling_map, transform_json)


def seconds_to_ns(text: str) -> int:
    return int(Decimal(text) * 1_000_000_000)


def kalibr_camchain(path: Path) -> dict:
    """The few fields we need from a Kalibr camchain-imucam yaml (no PyYAML)."""
    text = path.read_text()
    cams = {}
    for name, body in re.findall(r"^(cam\d):\n((?:  .*\n?)+)", text, re.M):
        def matrix(key):
            m = re.search(rf"{key}:\n((?:\s+- \[.*\]\n)+)", body)
            rows = re.findall(r"\[(.*?)\]", m.group(1))
            return np.array([[float(v) for v in r.split(",")] for r in rows])

        def vector(key):
            m = re.search(rf"{key}: \[(.*?)\]", body, re.S)
            return [float(v) for v in m.group(1).replace("\n", " ").split(",")]

        cams[name] = {
            "T_cam_imu": matrix("T_cam_imu"),
            "intrinsics": vector("intrinsics"),
            "distortion": vector("distortion_coeffs"),
            "resolution": [int(v) for v in vector("resolution")],
            "timeshift": float(re.search(r"timeshift_cam_imu: (\S+)", body).group(1)),
            "model": re.search(r"distortion_model: (\S+)", body).group(1),
        }
    return cams


def gt_time_offset_s(gt_t, gt_q, imu_t, gyro, search_s=0.05):
    """Shift that puts the ground-truth stamps on the IMU clock: the lag
    maximising agreement of |angular rate| (ground truth vs gyro)."""
    q = gt_q / np.linalg.norm(gt_q, axis=1, keepdims=True)
    dot = np.abs((q[:-1] * q[1:]).sum(1)).clip(0, 1)
    rate = 2 * np.arccos(dot) / np.diff(gt_t)
    mid = (gt_t[:-1] + gt_t[1:]) / 2
    gyro_rate = np.linalg.norm(gyro, axis=1)
    lags = np.arange(-search_s, search_s + 1e-9, 0.0005)
    cost = [np.mean(np.abs(rate - np.interp(mid + lag, imu_t, gyro_rate))) for lag in lags]
    return float(lags[int(np.argmin(cost))])


def read_rows(path: Path):
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            yield line.split()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    # Kalibr's imu.yaml gives inflated densities (0.1 / 0.05); these are
    # continuous-time densities for Basalt, chosen on indoor_forward_3.
    parser.add_argument("--accel-noise-std", type=float, default=0.05)
    parser.add_argument("--gyro-noise-std", type=float, default=0.005)
    parser.add_argument("--accel-bias-std", type=float, default=0.002)
    parser.add_argument("--gyro-bias-std", type=float, default=4e-5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from PIL import Image

    chain = kalibr_camchain(next(args.calib.glob("camchain-imucam-*.yaml")))
    intrinsics, t_imu_cam, resolutions, fits, maps = [], [], [], [], []
    for name in ("cam0", "cam1"):
        c = chain[name]
        assert c["model"] == "equidistant", c["model"]
        width, height = c["resolution"]
        kb4 = c["intrinsics"] + c["distortion"]
        ds, rms, worst, samples = fit_double_sphere(kb4, width, height)
        source, inside = resampling_map(kb4, ds, width, height)
        maps.append((source, inside))
        intrinsics.append({"camera_type": "ds", "intrinsics": dict(zip(
            ("fx", "fy", "cx", "cy", "xi", "alpha"), map(float, ds)))})
        t_imu_cam.append(transform_json(np.linalg.inv(c["T_cam_imu"])))
        resolutions.append([width, height])
        fits.append({"camera": name, "kb4": kb4, "ds": list(map(float, ds)),
                     "fit_within_60_deg_rms_px": rms, "fit_within_60_deg_max_px": worst,
                     "resampled_pixels_with_data": float(inside.mean())})
    shift_ns = int(round(chain["cam0"]["timeshift"] * 1e9))

    calibration = {"value0": {
        "T_imu_cam": t_imu_cam,
        "intrinsics": intrinsics,
        "resolution": resolutions,
        "vignette": [],
        "calib_accel_bias": [0.0] * 9,
        "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": 500.0,
        "accel_noise_std": [args.accel_noise_std] * 3,
        "gyro_noise_std": [args.gyro_noise_std] * 3,
        "accel_bias_std": [args.accel_bias_std] * 3,
        "gyro_bias_std": [args.gyro_bias_std] * 3,
        "T_mocap_world": transform_json(np.eye(4)),
        "T_imu_marker": transform_json(np.eye(4)),
        "mocap_time_offset_ns": 0,
        "mocap_to_imu_offset_ns": 0,
        "cam_time_offset_ns": 0,
    }}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    mav0 = args.out / "mav0"

    # Stereo pairs share a stamp; keep stamps present on both sides.
    left = {r[1]: r[2] for r in read_rows(args.sequence / "left_images.txt")}
    right = {r[1]: r[2] for r in read_rows(args.sequence / "right_images.txt")}
    stamps = sorted(set(left) & set(right), key=Decimal)
    for (cam, names), (source, inside) in zip((("cam0", left), ("cam1", right)), maps):
        data = mav0 / cam / "data"
        data.mkdir(parents=True, exist_ok=True)

        def convert(stamp, names=names, data=data, source=source, inside=inside):
            ns = seconds_to_ns(stamp) + shift_ns
            target = data / f"{ns}.png"
            if not target.exists():
                image = np.asarray(Image.open(args.sequence / names[stamp]).convert("L"))
                Image.fromarray(resample(image, source, inside)).save(target)
            return f"{ns},{ns}.png"

        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(convert, stamps))
        (mav0 / cam / "data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows) + "\n")

    imu_rows = ["#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"]
    previous = None
    for r in read_rows(args.sequence / "imu.txt"):
        ns = seconds_to_ns(r[1])
        if previous is not None and ns <= previous:
            continue
        imu_rows.append(",".join([str(ns)] + r[2:8]))
        previous = ns
    (mav0 / "imu0").mkdir(parents=True, exist_ok=True)
    (mav0 / "imu0" / "data.csv").write_text("\n".join(imu_rows) + "\n")

    # The ground-truth stamps lag the IMU by ~15 ms (indoor_forward_3); at
    # 6 m/s that is 9 cm, so estimate the lag from angular rates and remove it.
    gt = [r for r in read_rows(args.sequence / "groundtruth.txt")]
    imu = np.array([[float(v) for v in row.split(",")] for row in imu_rows[1:]])
    gt_arr = np.array([[float(v) for v in r[:8]] for r in gt])
    gt_offset = gt_time_offset_s(gt_arr[:, 0], gt_arr[:, 4:8], imu[:, 0] * 1e-9, imu[:, 1:4])
    (args.out / "gt_base.tum").write_text("\n".join(
        f"{Decimal(r[0]) + Decimal(f"{gt_offset:.4f}")} " + " ".join(r[1:8]) for r in gt) + "\n")
    p = np.array([[float(v) for v in r[1:4]] for r in gt])
    (args.out / "extrinsics.json").write_text(json.dumps({
        "T_base_imu": np.eye(4).tolist(), "note": "ground truth is the Snapdragon IMU pose",
        "timeshift_cam_imu_ns": shift_ns,
        "gt_time_offset_s": gt_offset, "double_sphere_fit": fits}, indent=2) + "\n")
    print(json.dumps({"images": len(stamps), "imu_samples": len(imu_rows) - 1,
                      "ground_truth_samples": len(gt), "gt_time_offset_s": gt_offset,
                      "ground_truth_length_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                      "fit_rms_px": [f["fit_within_60_deg_rms_px"] for f in fits]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
