#!/usr/bin/env python3
"""Align wider ROVER submaps with survey tracks and a visual seasonal bridge."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
import yaml

from eval_rover_seasonal_geometry import FACADE_WINDOW
from eval_rover_seasonal_overlap import Capture
from eval_rover_site_alignment import reference_positions


def sample_positions(capture, offsets):
    stamps = [capture.times[int(np.argmin(np.abs(capture.times - capture.times[0] - t)))] for t in offsets]
    return reference_positions(capture, stamps)


def survey_fit(model, image_ids, references, lever, *, fixed_scale=None, up=None):
    usable = [(i, p) for i, p in zip(image_ids, references, strict=True) if np.isfinite(p).all()]
    if len(usable) < 4:
        return None
    poses = [model.image(i).cam_from_world().inverse() for i, _ in usable]
    centers = np.stack([pose.translation for pose in poses])
    offsets = np.stack([pose.rotation * lever for pose in poses])
    target = np.stack([point for _, point in usable])
    if np.max(pdist(target)) < 0.5 or np.median(pdist(centers)) <= 1e-6:
        return None
    if up is None:
        up = -np.mean([pose.rotation * np.array([0., 1., 0.]) for pose in poses], axis=0)
    level = Rotation.align_vectors([[0., 0., 1.]], [up])[0].as_matrix()

    def transform(x):
        yaw = Rotation.from_rotvec([0., 0., x[0]]).as_matrix()
        return yaw @ level, x[1:4], fixed_scale if fixed_scale is not None else np.exp(x[4])

    def residual(x):
        rotation, translation, scale = transform(x)
        return ((scale * centers + offsets) @ rotation.T + translation - target).ravel()

    initial_scale = np.median(pdist(target)) / np.median(pdist(centers)) if fixed_scale is None else fixed_scale
    source = (initial_scale * centers + offsets) @ level.T
    a, b = source[:, :2] - source[:, :2].mean(axis=0), target[:, :2] - target[:, :2].mean(axis=0)
    yaw = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
    rotation = Rotation.from_rotvec([0., 0., yaw]).as_matrix()
    initial = [yaw, *(target.mean(axis=0) - source.mean(axis=0) @ rotation.T)]
    if fixed_scale is None:
        initial.append(np.log(initial_scale))
    fit = least_squares(residual, initial, loss="soft_l1", f_scale=0.15)
    rotation, translation, scale = transform(fit.x)
    distances = np.linalg.norm(residual(fit.x).reshape(-1, 3), axis=1)
    rmse = float(np.sqrt(np.mean(distances ** 2)))
    inlier_ids = {i for (i, _), error in zip(usable, distances, strict=True) if error <= 0.30}
    return rotation, translation, float(scale), rmse, len(usable), inlier_ids


def to_survey(points, fit):
    rotation, translation, scale = fit[:3]
    return scale * points @ rotation.T + translation


def nearest_summary(source, target):
    if not len(source) or not len(target):
        return None
    distance, _ = cKDTree(target).query(source)
    return {"count": len(distance), "median_m": float(np.median(distance)),
            "p90_m": float(np.percentile(distance, 90)), "within_25cm": int(np.count_nonzero(distance <= 0.25))}


def refine_survey_paths(summer, winter):
    """Refine a facade-initialized yaw transform against the first two laps."""
    summer_tree = cKDTree(summer[:, :2])
    pivot = winter.mean(axis=0)
    change = np.zeros(3)

    def transform(points, params):
        yaw = Rotation.from_rotvec([0., 0., params[0]]).as_matrix()
        return (points - pivot) @ yaw.T + pivot + np.array([params[1], params[2], 0])

    for _ in range(30):
        aligned = transform(winter, change)
        _, to_summer = summer_tree.query(aligned[:, :2])
        _, to_winter = cKDTree(aligned[:, :2]).query(summer[:, :2])
        source = np.concatenate([winter, winter[to_winter]])
        target = np.concatenate([summer[to_summer], summer])

        def residual(params):
            return (transform(source, params)[:, :2] - target[:, :2]).ravel()

        fit = least_squares(residual, change, loss="soft_l1", f_scale=0.25,
                            bounds=([-0.45, -2., -2.], [0.45, 2., 2.]))
        if np.linalg.norm(change - fit.x) < 1e-5:
            change = fit.x
            break
        change = fit.x
    yaw = Rotation.from_rotvec([0., 0., change[0]]).as_matrix()
    shift = pivot - yaw @ pivot + np.array([change[1], change[2], 0.])
    return yaw, shift


def save_points(path, points):
    np.savez_compressed(path, xyz_m=points)
    with np.load(path, allow_pickle=False) as data:
        return len(data["xyz_m"])


def evaluate(captures, facade, submaps, scale, lever, maps_out):
    # One shared vertical basis preserves a yaw-only bridge between survey frames.
    up = -np.mean([facade.image(i).cam_from_world().inverse().rotation * np.array([0., 1., 0.])
                   for i in facade.reg_image_ids()], axis=0)
    facade_offsets = json.loads(FACADE_WINDOW.read_text())
    fits = []
    for season, (capture, offsets) in enumerate(zip(captures, facade_offsets, strict=True)):
        ids = sorted(i for i in facade.reg_image_ids() if int(i > 6) == season)
        reference = sample_positions(capture, [offsets[i - 1 - season * 6] for i in ids])
        fit = survey_fit(facade, ids, reference, lever, fixed_scale=scale, up=up)
        if fit is None:
            raise ValueError("insufficient surveyed facade anchors")
        fits.append(fit)
    a, b = fits
    rotation = a[0] @ b[0].T
    translation = a[1] - rotation @ b[1]

    def winter_in_summer(points):
        return points @ rotation.T + translation

    offsets = [list(range(0, int(c.times[-1] - c.times[0]) + 1, 3)) for c in captures]
    refs = [sample_positions(c, times) for c, times in zip(captures, offsets, strict=True)]
    # The remaining surveyed trajectory samples do not enter the bridge fit.
    heldout = [np.stack([point for sec, point in zip(times, points, strict=True)
                        if not any(abs(sec - anchor) <= 7 for anchor in anchors) and np.isfinite(point).all()])
               for times, points, anchors in zip(offsets, refs, facade_offsets, strict=True)]
    path_agreement = {
        "summer_nearest_winter_before": nearest_summary(heldout[0], heldout[1]),
        "summer_nearest_winter_after": nearest_summary(heldout[0], winter_in_summer(heldout[1])),
        "winter_nearest_summer_before": nearest_summary(heldout[1], heldout[0]),
        "winter_nearest_summer_after": nearest_summary(winter_in_summer(heldout[1]), heldout[0]),
    }
    training, last_lap = [], []
    for capture, anchors in zip(captures, facade_offsets, strict=True):
        survey = np.loadtxt(capture.root / "groundtruth.txt")
        relative = survey[:, 0] - capture.times[0]
        training.append(survey[(relative >= 0) & (relative < 210), 1:4])
        independent = relative >= 225
        for anchor in anchors:
            independent &= np.abs(relative - anchor) > 7
        last_lap.append(survey[independent, 1:4])
    correction, shift = refine_survey_paths(training[0], winter_in_summer(training[1]))
    initial_last_lap = winter_in_summer(last_lap[1])
    refined_last_lap = initial_last_lap @ correction.T + shift
    survey_refinement = {
        "training_prism_positions": list(map(len, training)),
        "winter_final_lap_to_summer_before": nearest_summary(initial_last_lap, last_lap[0]),
        "winter_final_lap_to_summer_after": nearest_summary(refined_last_lap, last_lap[0]),
        "summer_final_lap_to_winter_before": nearest_summary(last_lap[0], initial_last_lap),
        "summer_final_lap_to_winter_after": nearest_summary(last_lap[0], refined_last_lap),
        "winter_to_summer_rotation": (correction @ rotation).tolist(),
        "winter_to_summer_translation_m": (correction @ translation + shift).tolist(),
    }
    points_by_season = [[], []]
    rows = {}
    split = len(offsets[0])
    for name, submap in submaps.items():
        ids = sorted(submap.reg_image_ids())
        seasons = {int(i > split) for i in ids}
        if len(seasons) != 1:
            continue
        season, = seasons
        ref = [refs[season][i - 1 - season * split] for i in ids]
        fit = survey_fit(submap, ids, ref, lever)
        if fit is None:
            continue
        inliers = fit[5]
        points = [point.xyz for point in submap.points3D.values()
                  if sum(obs.image_id in inliers for obs in point.track.elements) >= 3]
        usable = len(inliers) >= 4 and bool(points)
        rows[name] = {"season": ("summer", "winter")[season], "reference_frames": fit[4],
                      "survey_fit_rmse_m": fit[3], "inlier_frames_within_30cm": len(inliers),
                      "landmarks_with_three_inlier_observations": len(points), "used_for_update": usable,
                      "first_offset_s": offsets[season][ids[0] - 1 - season * split],
                      "last_offset_s": offsets[season][ids[-1] - 1 - season * split]}
        if usable:
            xyz = to_survey(np.asarray(points), fit)
            if season == 1:
                xyz = winter_in_summer(xyz) @ correction.T + shift
            points_by_season[season].append(xyz)
    update = None
    if all(points_by_season):
        summer, winter = [np.concatenate(points, axis=0) for points in points_by_season]
        distance, _ = cKDTree(summer).query(winter)
        # The submap fits vary by several decimeters. A one-meter margin avoids
        # promoting close duplicates when the overlap is poorly constrained.
        disjoint = winter[distance > 1.0]
        baseline_count = save_points(maps_out / "summer_landmarks.npz", summer)
        updated_count = save_points(maps_out / "updated_landmarks.npz", np.concatenate([summer, disjoint]))
        update = {
            "saved_summer_landmarks": baseline_count, "winter_landmarks": len(winter),
            "saved_updated_landmarks": updated_count, "winter_landmarks_added": updated_count - baseline_count,
            "winter_near_summer_landmarks": nearest_summary(winter, summer),
            "insertion_distance_m": 1.0,
        }
    return {
        "facade_reference_fit_rmse_m": [a[3], b[3]],
        "winter_to_summer_rotation": rotation.tolist(), "winter_to_summer_translation_m": translation.tolist(),
        "heldout_path_proximity": path_agreement, "survey_refinement": survey_refinement, "submap_fit": rows,
        "sparse_map_update": update,
    }


def main():
    import pycolmap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summer", type=Path, required=True)
    parser.add_argument("--winter", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--facade-model", type=Path, required=True)
    parser.add_argument("--facade-scale", type=float, required=True)
    parser.add_argument("--site-models", type=Path, required=True)
    parser.add_argument("--maps-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.facade_scale) or args.facade_scale <= 0:
        parser.error("expected a positive facade scale")
    captures = [Capture(p, "realsense_D435i") for p in (args.summer, args.winter)]
    lever = np.linalg.inv(np.asarray(yaml.safe_load(args.calibration.read_text())["Cam-To-Prism"]))[:3, 3]
    facade = pycolmap.Reconstruction(args.facade_model)
    models = {path.name: pycolmap.Reconstruction(path) for path in sorted(args.site_models.iterdir()) if path.is_dir()}
    args.maps_out.mkdir(parents=True)
    results = evaluate(captures, facade, models, args.facade_scale, lever, args.maps_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
