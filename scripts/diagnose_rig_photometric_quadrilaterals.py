#!/usr/bin/env python3
"""Screen stereo-temporal quadrilaterals with pose-free forward/backward LK."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quadrilaterals-tsv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--accepted-tsv", type=Path)
    parser.add_argument("--target-error-px", type=float, default=2.0)
    parser.add_argument("--forward-backward-error-px", type=float, default=1.0)
    parser.add_argument("--max-lk-error", type=float, default=20.0)
    parser.add_argument("--window-size", type=int, default=21)
    parser.add_argument("--pyramid-levels", type=int, default=3)
    parser.add_argument("--frame-bin-size", type=int, default=500)
    parser.add_argument("--min-frame-gap", type=int, default=1)
    parser.add_argument("--max-frame-gap", type=int, default=1)
    return parser.parse_args()


def observation(row: dict[str, str], index: int) -> dict[str, object]:
    return {
        "image": int(row[f"image_{index}"]),
        "keypoint": int(row[f"keypoint_{index}"]),
        "frame": int(row[f"frame_{index}"]),
        "sensor": int(row[f"sensor_{index}"]),
        "name": row[f"name_{index}"],
        "point": (float(row[f"x_{index}"]), float(row[f"y_{index}"])),
    }


def main() -> int:
    args = parse_args()
    for name, value in (
        ("target error", args.target_error_px),
        ("forward/backward error", args.forward_backward_error_px),
        ("LK error", args.max_lk_error),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if args.window_size < 3 or args.window_size % 2 == 0:
        raise ValueError("window size must be odd and at least three")
    if (
        args.pyramid_levels < 0
        or args.frame_bin_size < 1
        or args.min_frame_gap < 1
        or args.max_frame_gap < args.min_frame_gap
    ):
        raise ValueError("pyramid levels and frame bin size are invalid")

    source_sha256 = hashlib.sha256(args.quadrilaterals_tsv.read_bytes()).hexdigest()
    pair_batches: dict[tuple[str, str], list[tuple[int, int, tuple[float, float], tuple[float, float]]]] = defaultdict(list)
    track_start_frame: dict[int, int] = {}
    track_edge_count: dict[int, int] = defaultdict(int)
    with args.quadrilaterals_tsv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            track = int(row["track"])
            by_sensor: dict[int, list[dict[str, object]]] = defaultdict(list)
            for index in range(4):
                obs = observation(row, index)
                by_sensor[int(obs["sensor"])].append(obs)
            frames = {int(obs["frame"]) for values in by_sensor.values() for obs in values}
            if len(by_sensor) != 2 or len(frames) != 2:
                raise ValueError(f"track {track} is not a two-sensor/two-frame quadrilateral")
            start_frame = min(frames)
            track_start_frame[track] = start_frame
            for sensor, values in by_sensor.items():
                if len(values) != 2:
                    raise ValueError(f"track {track} sensor {sensor} does not have two observations")
                values.sort(key=lambda obs: int(obs["frame"]))
                source, target = values
                frame_gap = int(target["frame"]) - int(source["frame"])
                if not args.min_frame_gap <= frame_gap <= args.max_frame_gap:
                    raise ValueError(
                        f"track {track} frame gap {frame_gap} is outside "
                        f"[{args.min_frame_gap}, {args.max_frame_gap}]"
                    )
                pair_batches[(str(source["name"]), str(target["name"]))].append(
                    (track, sensor, source["point"], target["point"])
                )
                track_edge_count[track] += 1

    edge_passes: dict[int, int] = defaultdict(int)
    target_errors: list[float] = []
    fb_errors: list[float] = []
    lk_errors: list[float] = []
    status_failures = 0
    image_cache: dict[str, np.ndarray] = {}

    def load(name: str) -> np.ndarray:
        image = image_cache.get(name)
        if image is None:
            image = cv2.imread(str(args.images_dir / name), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(args.images_dir / name)
            image_cache[name] = image
            if len(image_cache) > 8:
                oldest = next(iter(image_cache))
                if oldest != name:
                    del image_cache[oldest]
        return image

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    for (source_name, target_name), entries in sorted(pair_batches.items()):
        source_image = load(source_name)
        target_image = load(target_name)
        source_points = np.asarray([entry[2] for entry in entries], dtype=np.float32).reshape(-1, 1, 2)
        expected_points = np.asarray([entry[3] for entry in entries], dtype=np.float32).reshape(-1, 1, 2)
        predicted, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
            source_image,
            target_image,
            source_points,
            None,
            winSize=(args.window_size, args.window_size),
            maxLevel=args.pyramid_levels,
            criteria=criteria,
        )
        if predicted is None or forward_status is None or forward_error is None:
            status_failures += len(entries)
            continue
        returned, backward_status, backward_error = cv2.calcOpticalFlowPyrLK(
            target_image,
            source_image,
            predicted,
            None,
            winSize=(args.window_size, args.window_size),
            maxLevel=args.pyramid_levels,
            criteria=criteria,
        )
        if returned is None or backward_status is None or backward_error is None:
            status_failures += len(entries)
            continue
        for index, (track, _sensor, _source, _target) in enumerate(entries):
            if not forward_status[index, 0] or not backward_status[index, 0]:
                status_failures += 1
                continue
            target_error = float(np.linalg.norm(predicted[index, 0] - expected_points[index, 0]))
            fb_error = float(np.linalg.norm(returned[index, 0] - source_points[index, 0]))
            lk_error = max(float(forward_error[index, 0]), float(backward_error[index, 0]))
            target_errors.append(target_error)
            fb_errors.append(fb_error)
            lk_errors.append(lk_error)
            if (
                target_error <= args.target_error_px
                and fb_error <= args.forward_backward_error_px
                and lk_error <= args.max_lk_error
            ):
                edge_passes[track] += 1

    accepted = sorted(
        track for track, edges in track_edge_count.items() if edges == 2 and edge_passes[track] == 2
    )
    bins: dict[str, dict[str, int]] = defaultdict(lambda: {"candidates": 0, "accepted": 0})
    accepted_set = set(accepted)
    for track, frame in track_start_frame.items():
        begin = frame // args.frame_bin_size * args.frame_bin_size
        label = f"{begin}-{begin + args.frame_bin_size - 1}"
        bins[label]["candidates"] += 1
        bins[label]["accepted"] += int(track in accepted_set)
    for value in bins.values():
        value["acceptance_ppm"] = round(value["accepted"] * 1_000_000 / value["candidates"])

    result = {
        "schema": "visloc_rig_photometric_quadrilateral_diagnostic_v1",
        "ground_truth_used": False,
        "descriptor_values_used": False,
        "reconstructed_poses_used": False,
        "quadrilaterals_tsv": str(args.quadrilaterals_tsv),
        "quadrilaterals_tsv_sha256": source_sha256,
        "images_dir": str(args.images_dir),
        "gates": {
            "target_error_px": args.target_error_px,
            "forward_backward_error_px": args.forward_backward_error_px,
            "max_lk_error": args.max_lk_error,
            "window_size": args.window_size,
            "pyramid_levels": args.pyramid_levels,
            "min_frame_gap": args.min_frame_gap,
            "max_frame_gap": args.max_frame_gap,
        },
        "candidate_tracks": len(track_start_frame),
        "candidate_temporal_edges": sum(track_edge_count.values()),
        "lk_status_failures": status_failures,
        "accepted_tracks": len(accepted),
        "acceptance_fraction": len(accepted) / len(track_start_frame) if track_start_frame else 0.0,
        "target_error_px": {
            "median": percentile(target_errors, 50),
            "p95": percentile(target_errors, 95),
            "p99": percentile(target_errors, 99),
        },
        "forward_backward_error_px": {
            "median": percentile(fb_errors, 50),
            "p95": percentile(fb_errors, 95),
            "p99": percentile(fb_errors, 99),
        },
        "lk_error": {
            "median": percentile(lk_errors, 50),
            "p95": percentile(lk_errors, 95),
            "p99": percentile(lk_errors, 99),
        },
        "frame_bins": dict(sorted(bins.items())),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.accepted_tsv is not None:
        args.accepted_tsv.parent.mkdir(parents=True, exist_ok=True)
        with (
            args.quadrilaterals_tsv.open(newline="", encoding="utf-8") as source,
            args.accepted_tsv.open("w", newline="", encoding="utf-8") as destination,
        ):
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError("quadrilateral TSV has no header")
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            for row in reader:
                if int(row["track"]) in accepted_set:
                    writer.writerow(row)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
