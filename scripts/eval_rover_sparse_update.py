#!/usr/bin/env python3
"""Save a sparse ROVER map update and audit the triangulated support."""

import argparse
import json
from pathlib import Path

import numpy as np


def triangulate(observations, camera, solve):
    if len(observations) < 2:
        return None
    matrices, pixels = zip(*observations, strict=True)
    matrices = np.asarray(matrices)
    rays = np.stack([camera.cam_ray_from_img(pixel) for pixel in pixels])
    world_rays = np.einsum("nki,nk->ni", matrices[:, :, :3], rays)
    if np.min(world_rays @ world_rays.T) > np.cos(np.deg2rad(1.5)):
        return None
    point = solve(matrices, rays)
    if point is None or not np.isfinite(point).all():
        return None
    for pose, pixel in observations:
        if reprojection_error(point, pose, pixel, camera) > 3:
            return None
    return point


def reprojection_error(point, pose, pixel, camera):
    camera_point = pose[:, :3] @ point + pose[:, 3]
    if camera_point[2] <= 0:
        return np.inf
    predicted = camera.img_from_cam(camera_point)
    return np.inf if predicted is None else np.linalg.norm(predicted - pixel)


def write_map(path, points):
    ids = sorted(points)
    np.savez_compressed(path, landmark_ids=ids, xyz_m=np.asarray([points[i] for i in ids]).reshape(-1, 3))
    with np.load(path, allow_pickle=False) as snapshot:
        return set(snapshot["landmark_ids"].tolist())


def evaluate(model, split, scale, map_out):
    import pycolmap

    if len(model.cameras) != 1:
        raise ValueError("expected one calibrated RGB camera")
    camera = next(iter(model.cameras.values()))
    poses = {}
    for i in model.reg_image_ids():
        poses[i] = model.image(i).cam_from_world().matrix()
    summer, winter_additions = {}, {}
    shared_gaps, heldout_errors = [], []
    winter_only = 0
    for identifier, track in model.points3D.items():
        groups = [[], []]
        for element in sorted(track.track.elements, key=lambda el: el.image_id):
            i = element.image_id
            image = model.image(i)
            pixel = image.point2D(element.point2D_idx).xy
            groups[int(i > split)].append((poses[i], pixel))
        before, after = groups
        solve = pycolmap.triangulate_multi_view_point
        initial = triangulate(before, camera, solve)
        if initial is not None:
            summer[identifier] = initial * scale
            repeated = triangulate(after, camera, solve)
            if repeated is not None:
                shared_gaps.append(float(np.linalg.norm(initial - repeated) * scale))
        elif not before and after:
            winter_only += 1
            point = triangulate(after[:-1], camera, solve)
            if point is None:
                continue
            error = reprojection_error(point, *after[-1], camera)
            if error <= 2:
                winter_additions[identifier] = point * scale
                heldout_errors.append(float(error))
    stored_before = write_map(map_out / "summer_landmarks.npz", summer)
    stored_after = write_map(map_out / "updated_landmarks.npz", summer | winter_additions)
    return {
        "summer_landmarks": len(stored_before), "winter_only_tracks": winter_only,
        "winter_landmarks_added": len(stored_after - stored_before),
        "summer_landmarks_retained": len(stored_after & stored_before),
        "winter_addition_holdout_px_median": float(np.median(heldout_errors)) if heldout_errors else None,
        "independently_triangulated_common_tracks": len(shared_gaps),
        "common_tracks_distance_median_m": float(np.median(shared_gaps)) if shared_gaps else None,
        "common_tracks_within_25cm": sum(gap <= 0.25 for gap in shared_gaps),
    }


def main():
    import pycolmap

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--summer-frames", type=int, required=True)
    parser.add_argument("--meters-per-unit", type=float, required=True)
    parser.add_argument("--maps-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.summer_frames < 1 or not np.isfinite(args.meters_per_unit) or args.meters_per_unit <= 0:
        parser.error("expected a positive summer frame count and metric scale")
    args.maps_out.mkdir(parents=True)
    result = evaluate(pycolmap.Reconstruction(args.model), args.summer_frames, args.meters_per_unit, args.maps_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
