#!/usr/bin/env python3
"""Tie existing seasonal ROVER submaps to joint short-window visual reconstructions."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
import yaml

from eval_rover_seasonal_geometry import assemble_tracks, reconstruct
from eval_rover_seasonal_overlap import Capture, EfficientLoFTR
from eval_rover_site_alignment import reference_positions
from eval_rover_visual_submaps import fit_localized_poses, join_components, transform_points


def generate_window(captures, offsets, k, distortion, matcher, out):
    images = [capture.image(seconds)[1] for capture, times in zip(captures, offsets, strict=True)
              for seconds in times]
    split = len(offsets[0])

    def matches():
        for i, first in enumerate(images):
            for j in range(i + 1, len(images)):
                cross = i < split <= j
                if abs(j - i - (split if cross else 0)) <= 2:
                    yield i, j, *matcher.match(first, images[j])

    with TemporaryDirectory(prefix="rover-local-bridge-") as scratch:
        points, pairs, _ = assemble_tracks(images, matches(), k, distortion, split)
        models = reconstruct(images, points, pairs, k, distortion, Path(scratch))
        out.mkdir(parents=True)
        for identifier, model in models.items():
            target = out / str(identifier)
            target.mkdir()
            model.write(target)


def collect_positions(component, models, split):
    poses = {}
    for name, mapping in component["anchor_from_submap"].items():
        transform = np.asarray(mapping)
        scale = np.cbrt(np.linalg.det(transform[:3, :3]))
        rotation = transform[:3, :3] / scale
        for i in models[name].reg_image_ids():
            pose = models[name].image(i).cam_from_world().inverse()
            poses.setdefault(i, (transform_points(pose.translation, transform), rotation @ pose.rotation.matrix()))
    return [sorted(i for i in poses if (i > split) == winter) for winter in (False, True)], poses


def score_reference(component, models, captures, offsets, training_windows, lever):
    split = len(offsets[0])
    ids, poses = collect_positions(component, models, split)
    rows = []
    for season, keys in enumerate(ids):
        selected = [offsets[season][i - 1 - season * split] for i in keys]
        stamps = [captures[season].image(t)[0] for t in selected]
        truth = reference_positions(captures[season], stamps)
        rows.append([(time, poses[i][0], poses[i][1] @ lever, ref)
                     for time, i, ref in zip(selected, keys, truth, strict=True) if np.isfinite(ref).all()])
    if min(map(len, rows)) < 3:
        return {}

    def error(group, scale):
        center, offset, truth = (np.stack(values) for values in zip(*(row[1:] for row in group), strict=True))
        return pdist(scale * center + offset) - pdist(truth)

    approximate = np.median(pdist(np.stack([row[3] for row in rows[0]]))) / np.median(
        pdist(np.stack([row[1] for row in rows[0]])))
    fit = least_squares(lambda p: error(rows[0], p[0]), [approximate], bounds=(1e-6, np.inf))
    scale = float(fit.x[0])
    independent = [row for row in rows[1] if row[0] not in training_windows]
    result = {"scale_m_per_unit_for_scoring": scale, "summer_reference_images": len(rows[0])}
    for key, group in (("winter_all", rows[1]), ("winter_outside_bridge", independent)):
        if len(group) >= 2:
            measured = pdist(np.stack([row[3] for row in group]))
            result[key] = {"images": len(group), "distance_rmse_m": float(np.sqrt(np.mean(error(group, scale) ** 2))),
                           "reference_diameter_m": float(measured.max())}
    return result


def bridge_site(models, windows, local_root, offsets):
    import pycolmap

    split = len(offsets[0])
    reports, edges = [], []
    acceptance_windows = set()
    for window, item in windows.items():
        local_offsets = item["offsets_s"]
        local_split = len(local_offsets[0])
        for path in sorted(p for p in (local_root / window).iterdir() if p.is_dir()):
            model = pycolmap.Reconstruction(path)
            ids = set(model.reg_image_ids())
            name = f"{window}:{path.name}"
            counts = [sum(i <= local_split for i in ids), sum(i > local_split for i in ids)]
            report = {"window": name, "registered": counts, "ties": []}
            reports.append(report)
            if min(counts) < 3:
                continue
            proposed = []
            for season, submap in enumerate(item["submaps"]):
                estimates = {}
                original = models[submap]
                for j, time in enumerate(local_offsets[season]):
                    source_id = offsets[season].index(time) + 1 + season * split
                    local_id = j + 1 + season * local_split
                    if source_id not in original.reg_image_ids() or local_id not in ids:
                        continue
                    pose = model.image(local_id).cam_from_world().inverse()
                    estimates[source_id] = (0, pose.rotation.matrix(), pose.translation)
                # Identical camera exposures tie the independently optimized reconstructions.
                outcome, transform = fit_localized_poses(
                    {submap: original, name: model}, (submap, name), estimates, np.random.default_rng(0),
                )
                report["ties"].append({**outcome, "method": "shared_camera_poses", "accepted": transform is not None})
                if transform is not None:
                    proposed.append((submap, transform))
            if len(proposed) == 2:
                (first, local_from_first), (second, local_from_second) = proposed
                edges.append((first, second, np.linalg.inv(local_from_second) @ local_from_first))
                acceptance_windows.update(local_offsets[1])
    return reports, join_components(models, edges, split), acceptance_windows


def main():
    import pycolmap

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("summer", "winter", "calibration", "site-results", "site-models", "windows", "local-models", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--efficient-loftr", type=Path, help="build local models if this option is provided")
    parser.add_argument("--map-out", type=Path, help="save the largest seasonal component in its visual gauge")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    run = json.loads(args.site_results.read_text())
    windows = json.loads(args.windows.read_text())
    captures = [Capture(root, "realsense_D435i") for root in (args.summer, args.winter)]
    calibration = yaml.safe_load(args.calibration.read_text())
    if args.efficient_loftr:
        intrinsics = calibration["Cam_Intrinsics"]
        fx, fy, cx, cy = intrinsics["intrinsics"]
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortion = np.array(intrinsics["distortion_coeffs"])
        matcher = EfficientLoFTR(args.efficient_loftr, 480)
        for name, item in windows.items():
            generate_window(captures, item["offsets_s"], k, distortion, matcher, args.local_models / name)
    models = {path.name: pycolmap.Reconstruction(path) for path in sorted(args.site_models.iterdir()) if path.is_dir()}
    offsets = run["sample_offsets_s"]
    reports, components, training = bridge_site(models, windows, args.local_models, offsets)
    mixed = [entry for entry in components if min(entry["registered_images"]) > 0]
    result = {"windows": reports, "mixed_components": mixed}
    if mixed:
        largest = max(mixed, key=lambda item: sum(item["registered_images"]))
        lever = np.linalg.inv(np.asarray(calibration["Cam-To-Prism"]))[:3, 3]
        result["reference_check"] = score_reference(largest, models, captures, offsets, training, lever)
        if args.map_out:
            points = [transform_points(np.asarray([point.xyz for point in models[name].points3D.values()]).reshape(-1, 3),
                                       np.asarray(transform))
                      for name, transform in largest["anchor_from_submap"].items()]
            args.map_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.map_out, xyz_visual_units=np.concatenate(points))
            with np.load(args.map_out, allow_pickle=False) as data:
                result["saved_points"] = len(data["xyz_visual_units"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
