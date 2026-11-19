#!/usr/bin/env python3
"""Evaluate RGB correspondence verification across the ROVER garden seasons."""

import argparse
from datetime import date
from functools import lru_cache
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


class Capture:
    def __init__(self, root, camera):
        self.root = root
        self.stream = root / camera
        manifest = self.stream / "rgb.txt"
        if not manifest.is_file():
            manifest = root / "rgb.txt"
        self.rows = [line.split() for line in manifest.read_text().splitlines() if line and not line.startswith("#")]
        self.times = np.array([float(row[0]) for row in self.rows])
        self.sift = cv2.SIFT_create()

    @lru_cache
    def features(self, seconds):
        index = int(np.argmin(np.abs(self.times - self.times[0] - seconds)))
        path = self.stream / self.rows[index][1]
        if not path.is_file():
            path = self.root / f"{seconds:03d}-rgb.png"
            sampled = json.loads((self.root / "selected.json").read_text())
            if sampled[str(seconds)]["rgb_time"] != self.times[index]:
                raise ValueError(f"{path}: sampled timestamp differs from official manifest")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read {path}")
        points, descriptors = self.sift.detectAndCompute(image, None)
        return float(self.times[index]), np.array([p.pt for p in points], dtype=np.float64), descriptors


def verified_matches(first, second, k, distortion):
    _, points_a, descriptors_a = first
    _, points_b, descriptors_b = second
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_matches(source, target):
        if source is None or target is None:
            return {}
        matches = {}
        for pair in matcher.knnMatch(source, target, k=2):
            if len(pair) == 2:
                nearest, alternative = pair
                if nearest.distance < 0.75 * alternative.distance:
                    matches[nearest.queryIdx] = nearest.trainIdx
        return matches

    forward = ratio_matches(descriptors_a, descriptors_b)
    reverse = ratio_matches(descriptors_b, descriptors_a)
    matches = [(a, b) for a, b in forward.items() if reverse.get(b) == a]
    if len(matches) < 8:
        return len(matches), 0
    a, b = np.asarray(matches).T
    left = cv2.undistortPoints(points_a[a], k, distortion, P=k).reshape(-1, 2)
    right = cv2.undistortPoints(points_b[b], k, distortion, P=k).reshape(-1, 2)
    _, mask = cv2.findFundamentalMat(left, right, cv2.FM_RANSAC, 2.0, 0.999, 5000)
    return len(matches), int(np.count_nonzero(mask)) if mask is not None else 0


def score_pair(first, second, k, distortion):
    matches, inliers = verified_matches(first, second, k, distortion)
    return {"mutual_matches": matches, "epipolar_inliers": inliers,
            "accepted": inliers >= 15 and inliers >= 0.35 * matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summer", type=Path, required=True)
    parser.add_argument("--winter", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.annotations.read_text())
    calibration = yaml.safe_load(args.calibration.read_text())["Cam_Intrinsics"]
    fx, fy, cx, cy = calibration["intrinsics"]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distortion = np.asarray(calibration["distortion_coeffs"])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    summer = Capture(args.summer, config["camera"])
    winter = Capture(args.winter, config["camera"])
    rows = []
    for pair in config["pairs"]:
        first, second = summer.features(pair["summer_s"]), winter.features(pair["winter_s"])
        rows.append({**pair, "summer_timestamp": first[0], "winter_timestamp": second[0],
                     **score_pair(first, second, k, distortion)})
    controls = []
    captures = {"summer": summer, "winter": winter}
    for pair in config["same_season_controls"]:
        capture = captures[pair["capture"]]
        first, second = capture.features(pair["first_s"]), capture.features(pair["second_s"])
        controls.append({**pair, "first_timestamp": first[0], "second_timestamp": second[0],
                         **score_pair(first, second, k, distortion)})
    confusion = {
        name: sum(row["overlap"] == label and row["accepted"] == prediction for row in rows)
        for name, label, prediction in (("tp", True, True), ("fp", False, True),
                                        ("fn", True, False), ("tn", False, False))
    }
    start, end = map(date.fromisoformat, config["captures"])
    result = {"revisit_days": (end - start).days, "confusion": confusion,
              "pairs": rows, "same_season_controls": controls}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
