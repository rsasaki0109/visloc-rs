#!/usr/bin/env python3
"""Reconstruct a matched seasonal RGB window and check metric scale on winter motion."""

import argparse
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import yaml

from eval_rover_seasonal_overlap import Capture, EfficientLoFTR, covered_cells, epipolar_inliers


FACADE_WINDOW = Path(__file__).resolve().parents[1] / "benchmarks" / "rover" / "facade_window.json"


def collect_tracks(images, matcher, k, distortion, summer_count):
    def matches():
        for first, second in combinations(range(len(images)), 2):
            cross_season = first < summer_count <= second
            if not cross_season and second - first > 3:
                continue
            if cross_season and abs(first - (second - summer_count)) > 1:
                continue
            yield first, second, *matcher.match(images[first], images[second])

    return assemble_tracks(images, matches(), k, distortion, summer_count)


def assemble_tracks(images, matches, k, distortion, summer_count):
    # The matcher gives slightly different subpixel coordinates for each pair.
    # Reuse tracks within a 2 px grid and keep one-to-one image correspondences.
    grids = [{} for _ in images]
    points = [[] for _ in images]
    pairs = {}
    edges = []
    for first, second, left, right in matches:
        cross_season = first < summer_count <= second
        keep = epipolar_inliers(left, right, k, distortion)
        left, right = left[keep], right[keep]
        covered = [covered_cells(uv, images[i].shape) for i, uv in ((first, left), (second, right))]
        if len(left) < 15 or len(left) < 0.35 * len(keep) or min(covered) < 4:
            continue
        correspondences = []
        for side, uv in ((first, left), (second, right)):
            ids = []
            for pixel, cell in zip(uv, np.rint(uv / 2).astype(int), strict=True):
                key = tuple(cell)
                if key not in grids[side]:
                    grids[side][key] = len(points[side])
                    points[side].append(pixel)
                ids.append(grids[side][key])
            correspondences.append(ids)
        unique = []
        used_a, used_b = set(), set()
        for a, b in zip(*correspondences, strict=True):
            if a not in used_a and b not in used_b:
                unique.append((a, b))
                used_a.add(a)
                used_b.add(b)
        pairs[first + 1, second + 1] = np.array(unique, dtype=np.uint32).reshape(-1, 2)
        edges.append({"images": [first + 1, second + 1], "matches": len(unique), "cross_season": cross_season})
    return points, pairs, edges


def reconstruct(images, points, pairs, k, distortion, work):
    import pycolmap

    height, width = images[0].shape
    if any(image.shape != (height, width) for image in images):
        raise ValueError("expected a common camera resolution")
    camera = pycolmap.Camera(
        model="OPENCV", width=width, height=height,
        params=[k[0, 0], k[1, 1], k[0, 2] + 0.5, k[1, 2] + 0.5, *distortion],
    )
    database_path = work / "matches.db"
    with pycolmap.Database.open(database_path) as db:
        camera_id = db.write_camera(camera)
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id)
        rig = pycolmap.Rig()
        rig.add_ref_sensor(sensor)
        rig_id = db.write_rig(rig)
        for i, coordinates in enumerate(points, start=1):
            frame = pycolmap.Frame(frame_id=i, rig_id=rig_id)
            frame.add_data_id(pycolmap.data_t(sensor, i))
            db.write_frame(frame, use_frame_id=True)
            image = pycolmap.Image(name=f"{i:03d}.png", camera_id=camera_id, image_id=i, frame_id=i)
            db.write_image(image, use_image_id=True)
            # COLMAP pixel centers are offset by 0.5 from OpenCV's convention.
            db.write_keypoints(i, np.asarray(coordinates, dtype=np.float32).reshape(-1, 2) + 0.5)
        for (first, second), matches in pairs.items():
            db.write_matches(first, second, matches)
            db.write_two_view_geometry(first, second, pycolmap.estimate_two_view_geometry(
                camera, np.asarray(points[first - 1]) + 0.5, camera,
                np.asarray(points[second - 1]) + 0.5, matches,
            ))
    options = pycolmap.IncrementalPipelineOptions(
        num_threads=2, min_model_size=2, extract_colors=False, ba_refine_focal_length=False,
        ba_refine_extra_params=False,
    )
    options.mapper.abs_pose_refine_focal_length = False
    options.mapper.abs_pose_refine_extra_params = False
    options.mapper.init_min_tri_angle = 3.0
    options.mapper.random_seed = 0
    models = work / "models"
    models.mkdir()
    return pycolmap.incremental_mapping(database_path, work, models, options)


def prism_positions(capture, stamps):
    rows = np.loadtxt(capture.root / "groundtruth.txt")
    time, keep = np.unique(rows[:, 0], return_index=True)
    if any(stamp < time[0] or stamp > time[-1] or np.min(np.abs(time - stamp)) > 0.25 for stamp in stamps):
        raise ValueError("missing synchronized prism positions")
    return np.column_stack([np.interp(stamps, time, rows[keep, i]) for i in (1, 2, 3)])


def check_scale(model, summer_count, references, camera_from_prism):
    groups = [[], []]
    registered = sorted(model.reg_image_ids())
    for i in registered:
        season = int(i > summer_count)
        j = i - 1 - season * summer_count
        image = model.image(i)
        world_from_camera = image.cam_from_world().inverse()
        offset = world_from_camera.rotation * camera_from_prism
        groups[season].append((image.projection_center(), offset, references[season][j]))
    shared_tracks = sum(
        len({element.image_id > summer_count for element in point.track.elements}) == 2
        for point in model.points3D.values()
    )
    result = {"registered_images": list(map(len, groups)), "registered_image_ids": registered,
              "points": model.num_points3D(), "cross_season_tracks": shared_tracks,
              "mean_reprojection_error_px": model.compute_mean_reprojection_error()}
    if min(result["registered_images"]) < 3:
        return result

    def distances(group, scale):
        centers, offsets, measured = np.asarray(group).transpose(1, 0, 2)
        return pdist(scale * centers + offsets) - pdist(measured)

    # Fit metric scale on summer distances only. Winter distances are a separate check.
    initial = np.median(pdist(np.asarray(groups[0])[:, 2])) / np.median(pdist(np.asarray(groups[0])[:, 0]))
    fit = least_squares(lambda scale: distances(groups[0], scale[0]), [initial], bounds=(1e-6, np.inf))
    scale = float(fit.x[0])
    for name, group in zip(("summer", "winter"), groups, strict=True):
        residual = distances(group, scale)
        result[name] = {"prism_distance_pairs": len(residual),
                        "prism_distance_rmse_m": float(np.sqrt(np.mean(residual ** 2)))}
    result["meters_per_model_unit"] = scale
    anchor_a, anchor_b = 4, summer_count + 4
    if anchor_a in registered and anchor_b in registered:
        winter_from_summer = model.image(anchor_b).cam_from_world() * model.image(anchor_a).cam_from_world().inverse()
        result["winter_anchor_from_summer_anchor"] = {
            "image_ids": [anchor_a, anchor_b],
            "quaternion_xyzw": winter_from_summer.rotation.quat.tolist(),
            "translation_m": (scale * winter_from_summer.translation).tolist(),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summer", type=Path, required=True)
    parser.add_argument("--winter", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--efficient-loftr", type=Path, required=True)
    parser.add_argument("--models-out", type=Path, help="save the reconstructed COLMAP models")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    capture = [Capture(root, "realsense_D435i") for root in (args.summer, args.winter)]
    # Exploratory window around the previously accepted house-facade pair.
    offsets = json.loads(FACADE_WINDOW.read_text())
    frames = [[source.image(seconds) for seconds in series]
              for source, series in zip(capture, offsets, strict=True)]
    images = [image for group in frames for _, image in group]
    calibration = yaml.safe_load(args.calibration.read_text())
    fx, fy, cx, cy = calibration["Cam_Intrinsics"]["intrinsics"]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distortion = np.array(calibration["Cam_Intrinsics"]["distortion_coeffs"])
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    refs = [prism_positions(source, [stamp for stamp, _ in group])
            for source, group in zip(capture, frames, strict=True)]
    prism = np.linalg.inv(np.array(calibration["Cam-To-Prism"]))[:3, 3]
    points, pairs, edges = collect_tracks(images, EfficientLoFTR(args.efficient_loftr), k, distortion, len(offsets[0]))
    with TemporaryDirectory(prefix="rover-sfm-") as temp:
        models = reconstruct(images, points, pairs, k, distortion, Path(temp))
        if args.models_out:
            args.models_out.mkdir(parents=True)
        scores = []
        for identifier, model in models.items():
            scores.append(check_scale(model, len(offsets[0]), refs, prism))
            if args.models_out:
                destination = args.models_out / str(identifier)
                destination.mkdir()
                model.write(destination)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"rgb_offset_seconds": offsets, "accepted_edges": edges, "models": scores}
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
