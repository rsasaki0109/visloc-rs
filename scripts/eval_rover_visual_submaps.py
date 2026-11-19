#!/usr/bin/env python3
"""Connect ROVER reconstructions with visual landmark correspondences only."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import yaml

from eval_rover_seasonal_overlap import Capture, EfficientLoFTR, covered_cells, epipolar_inliers


def tracked_pixels(model, image_id):
    image = model.image(image_id)
    pixels, ids = [], []
    for index in image.get_observation_point2D_idxs():
        point = image.point2D(index)
        pixels.append(point.xy - 0.5)
        ids.append(point.point3D_id)
    if not pixels:
        return None
    return cKDTree(pixels), np.asarray(ids)


def localize(model, tree, ids, pixels, query, k, distortion, shape):
    distance, indices = tree.query(pixels)
    best = {}
    for index in np.argsort(distance):
        if distance[index] <= 3:
            best.setdefault(int(ids[indices[index]]), query[index])
    if len(best) < 16:
        return None
    points = np.array([model.point3D(i).xyz for i in best])
    observed = np.array(list(best.values()))
    test = np.arange(len(best)) % 4 == 0
    ok, r, t, support = cv2.solvePnPRansac(points[~test], observed[~test], k, distortion,
                                          iterationsCount=250, reprojectionError=3.0, flags=cv2.SOLVEPNP_EPNP)
    if not ok or support is None or len(support) < 10:
        return None
    x, y = points[~test][support.ravel()], observed[~test][support.ravel()]
    if covered_cells(y, shape) < 3:
        return None
    r, t = cv2.solvePnPRefineLM(x, y, k, distortion, r, t)
    projected = cv2.projectPoints(points[test], r, t, k, distortion)[0].reshape(-1, 2)
    count = np.count_nonzero(np.linalg.norm(projected - observed[test], axis=1) <= 3)
    rotation = cv2.Rodrigues(r)[0]
    if count < 4 or count < 0.5 * test.sum() or np.any((x @ rotation.T + t.ravel())[:, 2] <= 0):
        return None
    return int(count + len(support)), rotation.T, (-rotation.T @ t).ravel()


def candidate_landmarks(run, models, captures, calibration, matcher):
    split = len(run["sample_offsets_s"][0])
    fx, fy, cx, cy = calibration["intrinsics"]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distortion = np.array(calibration["distortion_coeffs"])
    images = defaultdict(list)
    for name, model in models.items():
        for image in model.reg_image_ids():
            tracks = tracked_pixels(model, image)
            if tracks is not None:
                images[image].append((name, *tracks))
    links, poses = defaultdict(dict), defaultdict(dict)
    for edge in run["verified_pairs"]:
        a, b = edge["images"]
        candidates = [(left, right) for left in images[a] for right in images[b] if left[0] != right[0]]
        if not candidates:
            continue
        gray = [captures[int(i > split)].image(run["sample_offsets_s"][int(i > split)][i - 1 - split * int(i > split)])[1]
                for i in (a, b)]
        left, right = matcher.match(*gray)
        mask = epipolar_inliers(left, right, k, distortion)
        for (m, p_tree, p_ids), (n, q_tree, q_ids) in candidates:
            for base, other, tree, ids, pixel, query, i, shape in (
                (m, n, p_tree, p_ids, left, right, b, gray[1].shape),
                (n, m, q_tree, q_ids, right, left, a, gray[0].shape),
            ):
                pose = localize(models[base], tree, ids, pixel[mask], query[mask], k, distortion, shape)
                estimates = poses[(other, base)]
                if pose is not None and (i not in estimates or pose[0] > estimates[i][0]):
                    estimates[i] = pose
            p_dist, p_idx = p_tree.query(left[mask])
            q_dist, q_idx = q_tree.query(right[mask])
            valid = (p_dist <= 3) & (q_dist <= 3)
            for x, y in zip(p_ids[p_idx[valid]], q_ids[q_idx[valid]], strict=True):
                pair, points = ((m, n), (int(x), int(y))) if m < n else ((n, m), (int(y), int(x)))
                links[pair].setdefault(points, set()).add((a, b))
    return links, poses


def similarity(source, target):
    x, y = source - source.mean(axis=0), target - target.mean(axis=0)
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        return None
    u, singular, vt = np.linalg.svd(y.T @ x)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = float(singular @ sign / np.sum(x ** 2))
    if not np.isfinite(scale) or scale <= 0:
        return None
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = target.mean(axis=0) - transform[:3, :3] @ source.mean(axis=0)
    return transform


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def fit_edge(models, nodes, pairs, random):
    first, second = [models[name] for name in nodes]
    # Prefer landmark pairs supported by multiple image pairs and avoid assigning
    # one landmark to multiple destinations in either split.
    keys = sorted(pairs, key=lambda key: (-len(pairs[key]), key))
    unique, used_a, used_b = [], set(), set()
    for a, b in keys:
        if a not in used_a and b not in used_b:
            unique.append((a, b))
            used_a.add(a)
            used_b.add(b)
    result = {"submaps": nodes, "unique_matches": len(unique),
              "image_pairs": len(set().union(*pairs.values()))}
    if len(unique) < 16:
        return result, None
    random.shuffle(unique)
    source, target = (np.asarray([model.point3D(row[i]).xyz for row in unique])
                      for i, model in enumerate((first, second)))
    centers = np.asarray([second.image(i).projection_center() for i in second.reg_image_ids()])
    extent = float(np.linalg.norm(np.percentile(centers, 95, axis=0) - np.percentile(centers, 5, axis=0)))
    threshold = 0.04 * extent
    if threshold <= 1e-6:
        return result, None
    test = np.arange(len(unique)) % 4 == 0
    train_ids = np.flatnonzero(~test)
    best = np.zeros(len(source), dtype=bool)
    for _ in range(1200):
        indices = random.choice(train_ids, size=3, replace=False)
        model = similarity(source[indices], target[indices])
        if model is None:
            continue
        support = ~test & (np.linalg.norm(transform_points(source, model) - target, axis=1) <= threshold)
        if np.count_nonzero(support) > np.count_nonzero(best):
            best = support
    model = similarity(source[best], target[best]) if np.count_nonzero(best) >= 3 else None
    if model is None:
        return result, None
    errors = np.linalg.norm(transform_points(source, model) - target, axis=1)
    holds = test & (errors <= threshold)
    train = ~test & (errors <= threshold)
    supporting_pairs = set().union(*(pairs[key] for key, keep in zip(unique, train, strict=True) if keep))
    result.update({
        "supporting_image_pairs": len(supporting_pairs), "train_inliers": int(train.sum()),
        "holdout_inliers": int(holds.sum()), "holdout_tracks": int(test.sum()),
        "holdout_median_fraction_of_camera_extent": float(np.median(errors[test]) / extent),
        "inlier_threshold_fraction_of_camera_extent": 0.04,
    })
    accept = train.sum() >= 10 and holds.sum() >= 4 and holds.sum() >= 0.5 * test.sum() and len(supporting_pairs) >= 2
    return result, model if accept else None


def fit_localized_poses(models, nodes, observations, random):
    source_model, target_model = (models[name] for name in nodes)
    indices = np.array(sorted(observations))
    result = {"submaps": nodes, "localized_views": len(indices), "method": "pnp"}
    if len(indices) < 4:
        return result, None
    random.shuffle(indices)
    frames = [source_model.image(int(i)).cam_from_world().inverse() for i in indices]
    source = np.stack([pose.translation for pose in frames])
    target = np.stack([observations[i][2] for i in indices])
    rotations = np.stack([observations[i][1] @ frame.rotation.matrix().T
                          for i, frame in zip(indices, frames, strict=True)])
    centers = np.stack([target_model.image(i).projection_center() for i in target_model.reg_image_ids()])
    extent = np.linalg.norm(np.percentile(centers, 95, axis=0) - np.percentile(centers, 5, axis=0))
    threshold = 0.05 * extent
    test = np.arange(len(indices)) % 4 == 0
    best_count, best, held = 0, None, 0
    for i in np.flatnonzero(~test):
        for j in np.flatnonzero(~test):
            dt = target[j] - target[i]
            ray = rotations[i] @ (source[j] - source[i])
            if np.linalg.norm(dt) <= threshold or np.linalg.norm(ray) <= 1e-6:
                continue
            scale = (ray @ dt) / (ray @ ray)
            if scale <= 0:
                continue
            transform = np.eye(4)
            transform[:3, :3] = scale * rotations[i]
            transform[:3, 3] = target[i] - transform[:3, :3] @ source[i]
            err = np.linalg.norm(transform_points(source, transform) - target, axis=1)
            angle = Rotation.from_matrix(rotations @ rotations[i].T).magnitude()
            support = (err <= threshold) & (angle <= np.deg2rad(8))
            count = int((support & ~test).sum())
            if count > best_count:
                best_count, best, held = count, transform, int((support & test).sum())
    result.update({"train_inliers": best_count, "holdout_inliers": held, "holdout_views": int(test.sum())})
    return result, best if best_count >= 3 and held >= 1 and held >= 0.5 * test.sum() else None


def join_components(models, edges, split):
    graph = defaultdict(list)
    for first, second, second_from_first in edges:
        graph[first].append((second, np.linalg.inv(second_from_first)))
        graph[second].append((first, second_from_first))
    visited, components = set(), []
    for anchor in sorted(models, key=lambda name: (-models[name].num_reg_images(), name)):
        if anchor in visited:
            continue
        transforms = {anchor: np.eye(4)}
        queue = [anchor]
        for head in queue:
            for other, head_from_other in graph[head]:
                if other not in transforms:
                    transforms[other] = transforms[head] @ head_from_other
                    queue.append(other)
        visited.update(transforms)
        image_ids = {i for name in transforms for i in models[name].reg_image_ids()}
        components.append({
            "submaps": list(transforms), "registered_images": [sum(i <= split for i in image_ids), sum(i > split for i in image_ids)],
            "anchor_from_submap": {name: pose.tolist() for name, pose in transforms.items()},
        })
    return components


def main():
    import pycolmap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summer", type=Path, required=True)
    parser.add_argument("--winter", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--efficient-loftr", type=Path, required=True)
    parser.add_argument("--site-results", type=Path, required=True)
    parser.add_argument("--site-models", type=Path, required=True)
    parser.add_argument("--max-side", type=int, help="limit dense matching resolution")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    run = json.loads(args.site_results.read_text())
    models = {path.name: pycolmap.Reconstruction(path) for path in sorted(args.site_models.iterdir()) if path.is_dir()}
    captures = [Capture(root, "realsense_D435i") for root in (args.summer, args.winter)]
    calibration = yaml.safe_load(args.calibration.read_text())["Cam_Intrinsics"]
    pairs, poses = candidate_landmarks(run, models, captures, calibration, EfficientLoFTR(args.efficient_loftr, args.max_side))
    split = len(run["sample_offsets_s"][0])
    reports, edges = [], []
    random = np.random.default_rng(0)
    for nodes, links in sorted(pairs.items()):
        result, model = fit_edge(models, nodes, links, random)
        reports.append({**result, "accepted": model is not None})
        if model is not None:
            edges.append((*nodes, model))
    for nodes, estimates in sorted(poses.items()):
        outcome, model = fit_localized_poses(models, nodes, estimates, random)
        reports.append({**outcome, "accepted": model is not None})
        if model is not None:
            edges.append((*nodes, model))
    components = join_components(models, edges, split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"edges": reports, "components": components}, indent=2) + "\n")


if __name__ == "__main__":
    main()
