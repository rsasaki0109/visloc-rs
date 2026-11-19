#!/usr/bin/env python3
"""Measure whether held-out image pairs support a submap similarity alignment."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

from eval_rover_seasonal_overlap import Capture, EfficientLoFTR, epipolar_inliers


def pair_observations(run, models, captures, k, distortion, matcher):
    split = len(run["sample_offsets_s"][0])
    shared = defaultdict(list)
    for name, model in models.items():
        for image in model.reg_image_ids():
            shared[image].append(name)
    groups = defaultdict(list)
    for edge in run["verified_pairs"]:
        i, j = edge["images"]
        candidates = [(a, b) for a in shared[i] for b in shared[j] if a != b]
        if not candidates:
            continue
        images = []
        for image_id in (i, j):
            season = int(image_id > split)
            seconds = run["sample_offsets_s"][season][image_id - 1 - season * split]
            images.append(captures[season].image(seconds)[1])
        left, right = matcher.match(*images)
        good = epipolar_inliers(left, right, k, distortion)
        if good.sum() < 15:
            continue
        points = np.column_stack([cv2.undistortPoints(x[good], k, distortion).reshape(-1, 2)
                                  for x in (left, right)])
        for a, b in candidates:
            pose_a = models[a].image(i).cam_from_world().inverse()
            pose_b = models[b].image(j).cam_from_world().inverse()
            if a < b:
                groups[a, b].append((i, j, pose_a, pose_b, points))
            else:
                groups[b, a].append((j, i, pose_b, pose_a, points[:, [2, 3, 0, 1]]))
    return groups


def packed_pairs(rows):
    centers_a, centers_b, rotations_a, rotations_b, pixels_a, pixels_b = [], [], [], [], [], []
    for _, _, first, second, points in rows:
        centers_a.append(first.translation)
        centers_b.append(second.translation)
        rotations_a.append(first.rotation.matrix())
        rotations_b.append(second.rotation.matrix())
        sample = points[np.linspace(0, len(points) - 1, 48, dtype=int)]
        pixels_a.append(np.column_stack((sample[:, :2], np.ones(48))))
        pixels_b.append(np.column_stack((sample[:, 2:], np.ones(48))))
    return tuple(map(np.asarray, (centers_a, centers_b, rotations_a, rotations_b, pixels_a, pixels_b)))


def sampson_error(params, packed, focal):
    scale = np.exp(params[6])
    rotation = Rotation.from_rotvec(params[:3]).as_matrix()
    translation = params[3:6]
    a, b, rotation_a, rotation_b, x, y = packed
    inverse_b = np.swapaxes(rotation_b, 1, 2)
    relative_r = inverse_b @ rotation @ rotation_a
    relative_t = (inverse_b @ (scale * (a @ rotation.T) + translation - b)[..., None]).squeeze(-1)
    rotated = np.einsum("nij,nkj->nki", relative_r, x)
    ex = np.cross(relative_t[:, None], rotated)
    ey = np.einsum("nji,nkj->nki", relative_r, np.cross(y, relative_t[:, None]))
    denominator = np.maximum(1e-10, np.sqrt(np.sum(ex[:, :, :2] ** 2 + ey[:, :, :2] ** 2, axis=-1)))
    return (focal * np.sum(y * ex, axis=-1) / denominator).ravel()


def rotation_seeds(rows):
    seeds = []
    for _, _, first, second, points in rows:
        x, y = points[:, :2], points[:, 2:]
        essential, _ = cv2.findEssentialMat(x, y, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=0.0025)
        if essential is None:
            continue
        for candidate in essential.reshape(-1, 3, 3):
            inliers, rotation, direction, _ = cv2.recoverPose(candidate, x, y, np.eye(3))
            if inliers >= 10:
                target_r = second.rotation.matrix()
                common_r = target_r @ rotation @ first.rotation.matrix().T
                seeds.append((first.translation, second.translation, common_r, (target_r @ direction).ravel()))
    return seeds


def candidate_transforms(seeds):
    for _, _, rotation, _ in seeds:
        support = [seed for seed in seeds if Rotation.from_matrix(seed[2] @ rotation.T).magnitude() < 0.28]
        if len(support) < 2:
            continue
        for subset in (support, *combinations(support, 2)):
            coefficients, values = [], []
            for first, second, _, direction in subset:
                across = np.eye(3) - np.outer(direction, direction)
                coefficients.append(np.column_stack((across @ rotation @ first, across)))
                values.append(across @ second)
            solution = np.linalg.lstsq(np.concatenate(coefficients), np.concatenate(values), rcond=1e-4)[0]
            if 0.04 < solution[0] < 20:
                yield np.r_[Rotation.from_matrix(rotation).as_rotvec(), solution[1:], np.log(solution[0])]


def evaluate_split(rows, focal, fold):
    train = [row for i, row in enumerate(rows) if i % 4 != fold]
    test = [row for i, row in enumerate(rows) if i % 4 == fold]
    seeds = rotation_seeds(train)
    result = {"pair_count": len(rows), "essential_seeds": len(seeds),
              "held_out_images": [[row[0], row[1]] for row in test]}
    if len(seeds) < 3:
        return result
    training, testing = packed_pairs(train), packed_pairs(test)

    def objective(params):
        return np.mean(np.minimum(7.0, np.abs(sampson_error(params, training, focal))))

    proposals = sorted(candidate_transforms(seeds), key=objective)[:8]
    if not proposals:
        return result
    fits = [least_squares(sampson_error, params, args=(training, focal), loss="soft_l1", f_scale=1.2,
                          max_nfev=100, bounds=([-np.inf] * 6 + [-3.22], [np.inf] * 6 + [3.0])).x
            for params in proposals]
    best = min(fits, key=objective)

    def summary(pairs):
        distances = np.abs(sampson_error(best, pairs, focal)).reshape(-1, 48)
        return [{"median_px": float(np.median(row)), "fraction_within_3px": float(np.mean(row < 3))}
                for row in distances]

    return {**result, "training": summary(training), "holdout": summary(testing),
            "scale": float(np.exp(best[6]))}


def main():
    import pycolmap

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summer", "winter", "site-models", "site-results", "calibration", "efficient-loftr", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    models = {p.name: pycolmap.Reconstruction(p) for p in sorted(args.site_models.iterdir()) if p.is_dir()}
    run = json.loads(args.site_results.read_text())
    captures = [Capture(root, "realsense_D435i") for root in (args.summer, args.winter)]
    camera = yaml.safe_load(args.calibration.read_text())["Cam_Intrinsics"]
    fx, fy, cx, cy = camera["intrinsics"]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    pairs = pair_observations(run, models, captures, k, np.asarray(camera["distortion_coeffs"]),
                              EfficientLoFTR(args.efficient_loftr, 448))
    scores = {",".join(nodes): [evaluate_split(rows, fx, fold) for fold in range(4)]
              for nodes, rows in sorted(pairs.items()) if len(rows) >= 4}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
