#!/usr/bin/env python3
"""Evaluate RGB correspondence verification across the ROVER garden seasons."""

import argparse
from datetime import date
from functools import lru_cache
import hashlib
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
    def image(self, seconds, color=False):
        index = int(np.argmin(np.abs(self.times - self.times[0] - seconds)))
        path = self.stream / self.rows[index][1]
        if not path.is_file():
            path = self.root / f"{seconds:03d}-rgb.png"
            sampled = json.loads((self.root / "selected.json").read_text())
            if sampled[str(seconds)]["rgb_time"] != self.times[index]:
                raise ValueError(f"{path}: sampled timestamp differs from official manifest")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read {path}")
        return float(self.times[index]), image

    @lru_cache
    def features(self, seconds):
        timestamp, image = self.image(seconds)
        points, descriptors = self.sift.detectAndCompute(image, None)
        return timestamp, np.array([p.pt for p in points], dtype=np.float64).reshape(-1, 2), descriptors


def mutual_sift_matches(first, second):
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
    indices = np.asarray(matches, dtype=int).reshape(-1, 2)
    return points_a[indices[:, 0]], points_b[indices[:, 1]]


def epipolar_inliers(left, right, k, distortion):
    if len(left) < 8:
        return np.zeros(len(left), dtype=bool)
    left = cv2.undistortPoints(left, k, distortion, P=k).reshape(-1, 2)
    right = cv2.undistortPoints(right, k, distortion, P=k).reshape(-1, 2)
    _, mask = cv2.findFundamentalMat(left, right, cv2.FM_RANSAC, 2.0, 0.999, 5000)
    return mask.ravel().astype(bool) if mask is not None else np.zeros(len(left), dtype=bool)


class EfficientLoFTR:
    def __init__(self, path, max_side=None):
        import onnxruntime as ort

        self.max_side = max_side

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def match(self, first, second):
        images = {}
        scales = []
        for name, image in zip(("image0", "image1"), (first, second), strict=True):
            height, width = image.shape
            factor = min(1.0, self.max_side / max(height, width)) if self.max_side else 1.0
            target = (max(32, int(width * factor) // 32 * 32), max(32, int(height * factor) // 32 * 32))
            resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
            images[name] = resized[None, None].astype(np.float32) / 255.0
            scales.append(np.array([width, height]) / target)
        left, right, confidence = self.session.run(["mkpts0_f", "mkpts1_f", "mconf"], images)
        keep = (confidence >= 0.2) & np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
        return left[keep] * scales[0], right[keep] * scales[1]


def covered_cells(points, shape):
    height, width = shape
    cells = np.floor(points * (4.0 / np.array([width, height]))).astype(int).clip(0, 3)
    return len(np.unique(cells, axis=0))


def score_pair(captures, first_s, second_s, k, distortion, model):
    first, second = captures[0].features(first_s), captures[1].features(second_s)
    left, right = mutual_sift_matches(first, second)
    inliers = int(np.count_nonzero(epipolar_inliers(left, right, k, distortion)))
    result = {"mutual_matches": len(left), "epipolar_inliers": inliers,
              "accepted": inliers >= 15 and inliers >= 0.35 * len(left)}
    if model is not None:
        images = [captures[0].image(first_s)[1], captures[1].image(second_s)[1]]
        left, right = model.match(*images)
        keep = epipolar_inliers(left, right, k, distortion)
        inliers = int(np.count_nonzero(keep))
        coverage = [covered_cells(points[keep], image.shape)
                    for points, image in zip((left, right), images, strict=True)]
        result["efficient_loftr"] = {
            "confidence_filtered_matches": len(left), "epipolar_inliers": inliers, "covered_cells": coverage,
            "accepted": inliers >= 15 and inliers >= 0.35 * len(left) and min(coverage) >= 4,
        }
    return first[0], second[0], result


def confusion(rows, prediction):
    return {name: sum(row["overlap"] == label and prediction(row) == accepted for row in rows)
            for name, label, accepted in (("tp", True, True), ("fp", False, True),
                                           ("fn", True, False), ("tn", False, False))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summer", type=Path, required=True)
    parser.add_argument("--winter", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--efficient-loftr", type=Path, help="EfficientLoFTR outdoor full ONNX model")
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
    model = EfficientLoFTR(args.efficient_loftr) if args.efficient_loftr else None
    rows = []
    for pair in config["pairs"]:
        first, second, scores = score_pair((summer, winter), pair["summer_s"], pair["winter_s"],
                                           k, distortion, model)
        rows.append({**pair, "summer_timestamp": first, "winter_timestamp": second, **scores})
    controls = []
    captures = {"summer": summer, "winter": winter}
    for pair in config["same_season_controls"]:
        capture = captures[pair["capture"]]
        first, second, scores = score_pair((capture, capture), pair["first_s"], pair["second_s"],
                                           k, distortion, model)
        controls.append({**pair, "first_timestamp": first, "second_timestamp": second, **scores})
    start, end = map(date.fromisoformat, config["captures"])
    result = {"revisit_days": (end - start).days, "confusion": confusion(rows, lambda row: row["accepted"]),
              "pairs": rows, "same_season_controls": controls}
    if model is not None:
        with args.efficient_loftr.open("rb") as handle:
            result["model_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        result["efficient_loftr_confusion"] = confusion(rows, lambda row: row["efficient_loftr"]["accepted"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
