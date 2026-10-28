#!/usr/bin/env python3
"""Scale the stereo baseline of a Basalt calibration.

A stereo VIO takes its metric scale mostly from the distance between its
cameras. If the calibration gets that distance wrong by a percent, the VIO,
the SLAM and every map built with them come out a percent too long or too
short. On OpenLORIS-Scene the T265 calibration shipped with the dataset
makes the SLAM 1-3.5% long, which is most of the maps' error against the
ground truth.

Wheel odometry measures that without ground truth: its scale is set by
the wheels. `calibrate_wheel_offset.py` prints "VIO scale vs wheel odometry"
for a SLAM or VIO trajectory. Averaged over a few mapping sessions, its
inverse is the factor for the baseline here:

    python3 scripts/scale_stereo_baseline.py --calibration data/office1-1/calibration.json \\
        --scale 0.986 --out data/office1-1/calibration_scaled.json

Every camera but the first moves toward (or away from) the first by
`--scale`; rotations, intrinsics and the IMU settings are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AXES = ("px", "py", "pz")


def scale_baseline(calibration: dict, scale: float) -> dict:
    """A copy of `calibration` (Basalt JSON, with or without "value0") with
    every camera after the first `scale` times as far from the first."""
    if not (scale > 0 and np.isfinite(scale)):
        raise ValueError("the scale must be a positive number")
    scaled = json.loads(json.dumps(calibration))
    body = scaled.get("value0", scaled)
    cameras = body["T_imu_cam"]
    if len(cameras) < 2:
        raise ValueError("a stereo baseline needs at least two cameras")
    origin = np.array([cameras[0][axis] for axis in AXES])
    for camera in cameras[1:]:
        position = origin + scale * (np.array([camera[axis] for axis in AXES]) - origin)
        for axis, value in zip(AXES, position):
            camera[axis] = float(value)
    return scaled


def baseline_m(calibration: dict) -> float:
    cameras = calibration.get("value0", calibration)["T_imu_cam"]
    return float(np.linalg.norm([cameras[1][a] - cameras[0][a] for a in AXES]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--calibration", type=Path, required=True, help="Basalt calibration JSON")
    parser.add_argument("--scale", type=float, required=True, help="factor for the baseline, e.g. 0.986")
    parser.add_argument("--out", type=Path, required=True, help="scaled calibration JSON")
    args = parser.parse_args(argv)
    calibration = json.loads(args.calibration.read_text())
    scaled = scale_baseline(calibration, args.scale)
    args.out.write_text(json.dumps(scaled, indent=2) + "\n")
    print(f"baseline {baseline_m(calibration) * 1000:.2f} mm -> {baseline_m(scaled) * 1000:.2f} mm: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
