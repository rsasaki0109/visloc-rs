#!/usr/bin/env python3
"""Estimate pose-free metric rig factors from accepted four-view tracks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_rig_metric_motion as motion  # noqa: E402
import diagnose_rig_stereo_track_consistency as stereo  # noqa: E402


@dataclass(frozen=True)
class Sensor:
    camera: stereo.Camera
    sensor_from_rig_rotation: np.ndarray
    sensor_from_rig_translation: np.ndarray


def load_sensors(path: Path) -> dict[int, Sensor]:
    sensors = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0].startswith("#") or fields[0] == "F":
            continue
        if fields[0] != "S" or len(fields) != 16:
            raise ValueError(f"malformed sensor row {path}:{line_number}")
        index = int(fields[1])
        camera = stereo.Camera(*(float(value) for value in fields[5:9]))
        rotation = stereo.rotation(*(float(value) for value in fields[9:13]))
        translation = np.asarray([float(value) for value in fields[13:16]], dtype=float)
        if index in sensors or not np.all(np.isfinite(translation)):
            raise ValueError(f"invalid or duplicate sensor {index} at {path}:{line_number}")
        sensors[index] = Sensor(camera, rotation, translation)
    if len(sensors) < 2:
        raise ValueError("metric factors require at least two calibrated sensors")
    return sensors


def local_ray(sensor: Sensor, point: tuple[float, float]):
    camera_ray = np.asarray(
        [
            (point[0] - sensor.camera.cx) / sensor.camera.fx,
            (point[1] - sensor.camera.cy) / sensor.camera.fy,
            1.0,
        ],
        dtype=float,
    )
    rig_from_sensor = sensor.sensor_from_rig_rotation.T
    center = -(rig_from_sensor @ sensor.sensor_from_rig_translation)
    direction = rig_from_sensor @ camera_ray
    direction /= np.linalg.norm(direction)
    return center, direction


def parse_track(row: dict[str, str], sensors: dict[int, Sensor], min_angle_deg: float):
    by_frame = defaultdict(list)
    for index in range(4):
        frame = int(row[f"frame_{index}"])
        sensor_index = int(row[f"sensor_{index}"])
        sensor = sensors.get(sensor_index)
        if sensor is None:
            raise ValueError(f"track {row['track']} names unknown sensor {sensor_index}")
        point = (float(row[f"x_{index}"]), float(row[f"y_{index}"]))
        by_frame[frame].append((sensor_index, local_ray(sensor, point)))
    if len(by_frame) != 2:
        raise ValueError(f"track {row['track']} does not span exactly two frames")
    points = []
    for frame, observations in sorted(by_frame.items()):
        if len(observations) != 2 or observations[0][0] == observations[1][0]:
            raise ValueError(f"track {row['track']} lacks two-sensor support at frame {frame}")
        point = stereo.triangulate(observations[0][1], observations[1][1], min_angle_deg)
        if point is None:
            return None
        points.append((frame, point))
    return points[0][0], points[1][0], points[0][1], points[1][1]


def summarize(values: list[float]):
    return {
        "median": motion.percentile(values, 0.5),
        "p95": motion.percentile(values, 0.95),
        "max": max(values, default=None),
    }


def estimate(
    tracks_path: Path,
    manifest_path: Path,
    *,
    min_angle_deg: float,
    min_correspondences: int,
    min_inliers: int,
    ransac_threshold_m: float,
    ransac_iterations: int,
):
    sensors = load_sensors(manifest_path)
    grouped = defaultdict(lambda: ([], []))
    input_tracks = 0
    triangulated_tracks = 0
    with tracks_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            input_tracks += 1
            parsed = parse_track(row, sensors, min_angle_deg)
            if parsed is None:
                continue
            frame_i, frame_j, point_i, point_j = parsed
            grouped[(frame_i, frame_j)][0].append(point_i)
            grouped[(frame_i, frame_j)][1].append(point_j)
            triangulated_tracks += 1

    factors = []
    rejected_support = 0
    rejected_fit = 0
    rejected_inliers = 0
    for (frame_i, frame_j), (source, target) in sorted(grouped.items()):
        if len(source) < min_correspondences:
            rejected_support += 1
            continue
        source_array = np.asarray(source, dtype=float)
        target_array = np.asarray(target, dtype=float)
        fit = motion.robust_rigid(
            source_array,
            target_array,
            ransac_threshold_m,
            ransac_iterations,
            (frame_i * 1_000_003 + frame_j) & 0xFFFFFFFF,
        )
        if fit is None:
            rejected_fit += 1
            continue
        rotation, translation, inliers, residuals = fit
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < min_inliers:
            rejected_inliers += 1
            continue
        factors.append(
            {
                "frame_i": frame_i,
                "frame_j": frame_j,
                "frame_gap": frame_j - frame_i,
                "rotation": rotation,
                "translation": translation,
                "correspondences": len(source),
                "inliers": inlier_count,
                "inlier_fraction": inlier_count / len(source),
                "median_inlier_residual_m": float(np.median(residuals[inliers])),
                "translation_m": float(np.linalg.norm(translation)),
                "rotation_deg": math.degrees(
                    math.acos(float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))
                ),
            }
        )
    return factors, {
        "schema": "visloc_rig_metric_factors_v1",
        "ground_truth_used": False,
        "descriptor_values_used": False,
        "reconstructed_poses_used": False,
        "input_tracks": input_tracks,
        "triangulated_tracks": triangulated_tracks,
        "candidate_frame_pairs": len(grouped),
        "accepted_factors": len(factors),
        "rejected_support": rejected_support,
        "rejected_fit": rejected_fit,
        "rejected_inliers": rejected_inliers,
        "gates": {
            "min_angle_deg": min_angle_deg,
            "min_correspondences": min_correspondences,
            "min_inliers": min_inliers,
            "ransac_threshold_m": ransac_threshold_m,
            "ransac_iterations": ransac_iterations,
        },
        "inlier_fraction": summarize([factor["inlier_fraction"] for factor in factors]),
        "inlier_residual_m": summarize(
            [factor["median_inlier_residual_m"] for factor in factors]
        ),
        "translation_m": summarize([factor["translation_m"] for factor in factors]),
        "rotation_deg": summarize([factor["rotation_deg"] for factor in factors]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-tsv", type=Path, required=True)
    parser.add_argument("--rig-manifest", type=Path, required=True)
    parser.add_argument("--output-factors-tsv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-angle-deg", type=float, default=0.5)
    parser.add_argument("--min-correspondences", type=int, default=8)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.15)
    parser.add_argument("--ransac-iterations", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        not math.isfinite(args.min_angle_deg)
        or args.min_angle_deg <= 0
        or not math.isfinite(args.ransac_threshold_m)
        or args.ransac_threshold_m <= 0
        or args.min_correspondences < 3
        or args.min_inliers < 3
        or args.min_inliers > args.min_correspondences
        or args.ransac_iterations < 1
    ):
        raise ValueError("invalid metric-factor gate")
    factors, result = estimate(
        args.tracks_tsv,
        args.rig_manifest,
        min_angle_deg=args.min_angle_deg,
        min_correspondences=args.min_correspondences,
        min_inliers=args.min_inliers,
        ransac_threshold_m=args.ransac_threshold_m,
        ransac_iterations=args.ransac_iterations,
    )
    args.output_factors_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_factors_tsv.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "frame_i", "frame_j", "r00", "r01", "r02", "r10", "r11", "r12",
            "r20", "r21", "r22", "tx", "ty", "tz", "correspondences", "inliers",
            "inlier_fraction", "median_inlier_residual_m",
        ]
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        for factor in factors:
            rotation = factor["rotation"]
            translation = factor["translation"]
            writer.writerow(
                {
                    "frame_i": factor["frame_i"], "frame_j": factor["frame_j"],
                    **{f"r{row}{column}": f"{rotation[row, column]:.17g}" for row in range(3) for column in range(3)},
                    "tx": f"{translation[0]:.17g}", "ty": f"{translation[1]:.17g}",
                    "tz": f"{translation[2]:.17g}", "correspondences": factor["correspondences"],
                    "inliers": factor["inliers"], "inlier_fraction": f"{factor['inlier_fraction']:.17g}",
                    "median_inlier_residual_m": f"{factor['median_inlier_residual_m']:.17g}",
                }
            )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
