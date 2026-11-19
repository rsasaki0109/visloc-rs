#!/usr/bin/env python3
"""Sample both ROVER traversals and evaluate their joint RGB reconstruction."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import yaml

from eval_rover_seasonal_geometry import assemble_tracks, reconstruct
from eval_rover_seasonal_overlap import Capture, EfficientLoFTR, epipolar_inliers, mutual_sift_matches


def descriptors(images, path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    net = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    rows = []
    for begin in range(0, len(images), 8):
        batch = [cv2.cvtColor(cv2.resize(im, (224, 224)), cv2.COLOR_BGR2RGB)
                 for im in images[begin:begin + 8]]
        pixels = (np.asarray(batch, dtype=np.float32).transpose(0, 3, 1, 2) / 255 - mean) / std
        rows.extend(net.run(["last_hidden_state"], {"pixel_values": pixels})[0][:, 0])
    rows = np.asarray(rows)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def query_pairs(embeddings, split):
    start_end = [(0, split), (split, len(embeddings))]
    similarities = embeddings @ embeddings.T
    cross_scores = similarities[:split, split:]
    cross = set()
    # Select two visually strongest links per 12-second sector in either traverse.
    for dim, matrix in enumerate((cross_scores, cross_scores.T)):
        for start in range(0, matrix.shape[0], 4):
            block = matrix[start:start + 4].copy()
            for _ in range(2):
                row, col = np.unravel_index(block.argmax(), block.shape)
                if not np.isfinite(block[row, col]):
                    break
                i, j = (start + row, col) if dim == 0 else (col, start + row)
                cross.add((int(i), split + int(j)))
                block[row, :] = -np.inf
                block[:, max(0, col - 1):col + 2] = -np.inf
    adjacent, loops = set(), set()
    for start, end in start_end:
        for i in range(start, end):
            adjacent.update((i, j) for j in range(i + 1, min(i + 3, end)))
            same = similarities[i, start:end].copy()
            own = i - start
            same[max(0, own - 3):own + 4] = -np.inf
            for j in np.argsort(same)[-2:]:
                if np.isfinite(same[j]):
                    loops.add(tuple(sorted((i, start + int(j)))))
    return adjacent, loops, cross


def candidate_matches(images, k, distortion, groups, dense):
    adjacent, loops, cross = groups
    detector = cv2.SIFT_create(nfeatures=1000)
    features = []
    for image in images:
        points, data = detector.detectAndCompute(image, None)
        features.append((None, np.asarray([p.pt for p in points]).reshape(-1, 2), data))

    def sparse(pair):
        a, b = pair
        left, right = mutual_sift_matches(features[a], features[b])
        inliers = np.count_nonzero(epipolar_inliers(left, right, k, distortion))
        return a, b, left, right, inliers

    with ThreadPoolExecutor(max_workers=4) as workers:
        sparse_rows = list(workers.map(sparse, sorted(adjacent | loops)))
    for a, b, left, right, inliers in sparse_rows:
        if inliers >= 25:
            yield a, b, left, right
        elif b == a + 1 and (a, b) in adjacent:
            yield a, b, *dense.match(images[a], images[b])
    for a, b in sorted(cross):
        yield a, b, *dense.match(images[a], images[b])


def reference_positions(capture, stamps):
    rows = np.loadtxt(capture.root / "groundtruth.txt")
    times, keep = np.unique(rows[:, 0], return_index=True)
    positions = np.column_stack([np.interp(stamps, times, rows[keep, i]) for i in (1, 2, 3)])
    for i, stamp in enumerate(stamps):
        if not times[0] <= stamp <= times[-1] or np.min(np.abs(times - stamp)) > 0.25:
            positions[i] = np.nan
    return positions


def evaluate_model(model, split, offsets, references, camera_from_prism):
    season_rows = [[], []]
    for i in sorted(model.reg_image_ids()):
        season = int(i > split)
        index = i - 1 - season * split
        image = model.image(i)
        rotation = image.cam_from_world().inverse().rotation
        season_rows[season].append((offsets[season][index], image.projection_center(),
                                   rotation * camera_from_prism, references[season][index]))
    result = {"registered_offsets_s": [[row[0] for row in group] for group in season_rows],
              "points": model.num_points3D(), "mean_reprojection_error_px": model.compute_mean_reprojection_error()}
    # Only the summer total station data is used to fix scale.
    groups = [[row[1:] for row in group if np.isfinite(row[3]).all()] for group in season_rows]
    if min(map(len, groups)) < 3:
        return result

    def vectors(group, scale):
        centers, offsets, reference = np.asarray(group).transpose(1, 0, 2)
        return pdist(centers * scale + offsets), pdist(reference)

    def residual(group, scale):
        estimated, reference = vectors(group, scale)
        return estimated - reference

    a, b = vectors(groups[0], 1.0)
    scale = float(least_squares(lambda x: residual(groups[0], x[0]), [np.median(b) / np.median(a)],
                                loss="soft_l1", f_scale=0.2, bounds=(1e-6, np.inf)).x[0])
    result["meters_per_model_unit"] = scale
    for label, group in zip(("summer", "winter"), groups, strict=True):
        estimated, measured = vectors(group, scale)
        error = estimated - measured
        result[label] = {
            "reference_frames": len(group), "reference_distance_pairs": len(error),
            "distance_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "distance_median_absolute_error_m": float(np.median(np.abs(error))),
            "reference_diameter_m": float(max(measured)),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summer", type=Path, required=True)
    parser.add_argument("--winter", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--efficient-loftr", type=Path, required=True)
    parser.add_argument("--dinov2", type=Path, required=True)
    parser.add_argument("--models-out", type=Path, help="save the reconstructed COLMAP models")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    captures = [Capture(root, "realsense_D435i") for root in (args.summer, args.winter)]
    offsets = [list(range(0, int(c.times[-1] - c.times[0]) + 1, 3)) for c in captures]
    frames = [[capture.image(t, color=True) for t in times] for capture, times in zip(captures, offsets, strict=True)]
    colors = [image for _, image in chain.from_iterable(frames)]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in colors]
    references = [reference_positions(c, [stamp for stamp, _ in group])
                  for c, group in zip(captures, frames, strict=True)]
    calibration = yaml.safe_load(args.calibration.read_text())
    fx, fy, cx, cy = calibration["Cam_Intrinsics"]["intrinsics"]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distortion = np.asarray(calibration["Cam_Intrinsics"]["distortion_coeffs"])
    split = len(frames[0])
    groups = query_pairs(descriptors(colors, args.dinov2), split)
    dense = EfficientLoFTR(args.efficient_loftr, max_side=384)
    matches = candidate_matches(images, k, distortion, groups, dense)
    points, pairs, edges = assemble_tracks(images, matches, k, distortion, split)
    print(f"verified {len(edges)} pairs", flush=True)
    camera_from_prism = np.linalg.inv(np.asarray(calibration["Cam-To-Prism"]))[:3, 3]
    with TemporaryDirectory(prefix="rover-site-") as temp:
        models = reconstruct(images, points, pairs, k, distortion, Path(temp))
        if args.models_out:
            args.models_out.mkdir(parents=True)
        scores = []
        for identifier, model in models.items():
            score = evaluate_model(model, split, offsets, references, camera_from_prism)
            scores.append({"id": identifier, **score})
            if args.models_out:
                destination = args.models_out / str(identifier)
                destination.mkdir()
                model.write(destination)
    result = {"sample_offsets_s": offsets, "candidate_pairs": list(map(len, groups)),
              "verified_pairs": edges, "models": scores}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
