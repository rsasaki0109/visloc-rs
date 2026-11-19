#!/usr/bin/env python3
"""Persist an RGB-D map update and relocalize a third LILocBench visit."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from bonn_estimated_poses import export_visit, merge_trial, run_binary
from change_loop_gate import change_signs
from eval_bonn_map_changes import labeled_voxels, select_frames
from eval_lilocbench_changes import Visit
from lilocbench_motion import pose_errors, score_motion, wheel_fallback
from openloris_map_changes import surface_voxels, visibility
from rgbd_map_cloud import keyframe_poses

VOXEL_M = 0.07


def snapshot(path, occupied):
    keys = np.array(sorted(occupied), dtype=np.int32).reshape(-1, 3)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        np.savez_compressed(temporary, version=1, voxel_m=VOXEL_M, occupied=keys)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        keys = data["occupied"]
        if data["version"].item() != 1 or data["voxel_m"].item() != VOXEL_M or keys.ndim != 2 or keys.shape[1] != 3:
            raise ValueError("unexpected occupancy snapshot schema")
        return set(map(tuple, keys.tolist()))


def voxel_update(occupied, prior, latest, intrinsics):
    new = surface_voxels(latest, intrinsics, 8, VOXEL_M, None)
    keys = sorted(occupied | new)
    centers = (np.array(keys) + 0.5) * VOXEL_M
    signs = change_signs(centers, [prior, latest], intrinsics)
    added = {key for key, sign in zip(keys, signs, strict=True) if sign == 1 and key in new}
    removed = {key for key, sign in zip(keys, signs, strict=True) if sign == -1 and key in occupied}
    return (occupied - removed) | added, len(added - occupied), len(removed)


def query_manifest(visit, frames, root):
    path = root / "mav0" / "cam0"
    (path / "data").mkdir(parents=True)
    with (path / "data.csv").open("w") as manifest:
        for frame in frames:
            stamp = (frame + 1) * 250_000_000
            name = f"{stamp}.png"
            gray = cv2.cvtColor(visit.rectify(visit.rgb_paths[frame]), cv2.COLOR_RGB2GRAY)
            Image.fromarray(gray).save(path / "data" / name)
            manifest.write(f"{stamp},{name}\n")


def localize(map_path, tool, gauge, work):
    run_binary(tool, "--map", map_path, "--euroc-dir", work / "query", "--out-dir", work / "relocalization",
               "--min-inliers", 15, "--dump-inliers", work / "inliers.csv")
    trajectory = work / "relocalization" / "localization.tum"
    poses = {}
    if trajectory.stat().st_size:
        for row in np.loadtxt(trajectory, ndmin=2):
            frame = round(row[0] / 0.25) - 1
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
            pose[:3, 3] = row[1:4]
            poses[frame] = gauge @ pose
    return poses


def observations_at(raw, poses):
    return [(poses[int(path.stem)], path) for _, path in raw if int(path.stem) in poses]


def assess_reloaded_geometry(path, observed, intrinsics, original_box, stable):
    occupied = load_snapshot(path)
    candidate_box = original_box & occupied
    if observed:
        def free_at(keys):
            points = (np.array(sorted(keys)).reshape(-1, 3) + 0.5) * VOXEL_M
            hits, free = visibility(points, observed, intrinsics, 0.10)
            return int(np.count_nonzero((free >= 2) & (free >= 2 * hits)))
        removed_box = free_at(candidate_box)
        stable_falsely_cleared = free_at(stable & occupied)
    else:
        removed_box = stable_falsely_cleared = None
    return {
        "reloaded_surface_voxels": len(occupied),
        "known_visit_two_box_voxels_stored": len(candidate_box),
        "known_visit_two_box_voxels_now_observed_free": removed_box,
        "stored_stable_voxels": len(stable & occupied),
        "stable_voxels_falsely_observed_free": stable_falsely_cleared,
    }


def run(args):
    config = json.loads(args.annotations.read_text())["maps"][0]
    visits = [Visit(args.data, "corridor_box", name)
              for name in ("mapping", "lt_changes_0", "lt_changes_dynamics_0")]
    root = args.maps_out
    root.mkdir(parents=True)
    data, maps = [], []
    for i, (visit, selection) in enumerate(zip(visits[:2], config["visits"], strict=True)):
        frames = select_frames(selection["window"], {label["frame"] for label in selection["labels"]}, 1)
        observed = list(visit.observations(frames, root / f"depth_{i}"))
        data.append((visit, observed))
        maps.append(export_visit(visit, observed, visit.intrinsics, root / f"visit_{i}", args.map_tools / "tum_rgbd_save_map",
                                 odometry=visit.odometry(frames)))
    source_map = maps[0][0]
    merged_map = root / "updated_map"
    poses, report = merge_trial(data, maps, merged_map, args.map_tools / "basalt_map_merge")
    gauge = data[0][1][0][0] @ np.linalg.inv(keyframe_poses(source_map, 0)[0][1])
    original_poses = {frame: gauge @ pair[1] for (frame, _), pair in zip(maps[0][1], keyframe_poses(source_map, 0), strict=True)}
    first = observations_at(data[0][1], original_poses)
    second = observations_at(data[1][1], poses[1])
    intrinsics = visits[0].intrinsics
    initial = surface_voxels(first, intrinsics, 8, VOXEL_M, None)
    updated, added, removed = voxel_update(initial, first, second, intrinsics)
    before, after = source_map / "occupancy.npz", merged_map / "occupancy.npz"
    snapshot(before, initial)
    snapshot(after, updated)
    third = visits[2]
    frames = list(range(len(third.times)))
    query_manifest(third, frames, root / "query_data")
    third_depth = list(third.observations(frames, root / "third_depth"))
    box = labeled_voxels(visits[1], config["visits"][1]["labels"], VOXEL_M, intrinsics)["box"]
    stable = set()
    for i, visit in enumerate(visits[:2]):
        labels = labeled_voxels(visit, config["visits"][i]["labels"], VOXEL_M, intrinsics)
        stable |= labels["stable"] | labels["floor"]
    stable -= box
    variants = {}
    for name, map_path, occupancy in (("initial", source_map, before), ("updated", merged_map, after)):
        # A separate executable opens each persisted map for the third visit.
        output = root / f"third_{name}"
        output.mkdir()
        (output / "query").symlink_to((root / "query_data").resolve(), target_is_directory=True)
        estimates = localize(map_path, args.map_tools / "basalt_euroc_localize_demo", gauge, output)
        variants[name] = {
            "attempts": len(frames), **pose_errors(third, estimates),
            "geometry": assess_reloaded_geometry(
                occupancy, observations_at(third_depth, estimates), third.intrinsics, box, stable,
            ),
        }
    motion = None
    if args.motion_annotations:
        continued = wheel_fallback(third, estimates, frames)
        motion = score_motion(args.motion_annotations, third, first, {
            "visual_only": estimates, "visual_with_wheel_fallback": continued,
            "reference_poses": {frame: third.pose(third.times[frame]) for frame in frames},
        })
        variants["updated_with_wheel_fallback"] = assess_reloaded_geometry(
            after, observations_at(third_depth, continued), third.intrinsics, box, stable,
        )
    oracle = assess_reloaded_geometry(after, third_depth, third.intrinsics, box, stable)
    return {"visits": [v.root.name for v in visits], "capture_start_s": [float(v.times[0]) for v in visits],
            "saved_update": {"initial_voxels": len(initial), "updated_voxels": len(updated),
                             "additions": added, "removals": removed,
                             "annotated_stable_voxels_added": len(stable & (updated - initial)),
                             "annotated_stable_voxels_removed": len(stable & (initial - updated))}, "registration": report["session_alignment"],
            "third_visit": variants, "third_visit_reference_pose_geometry": oracle, "occlusion_motion": motion}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--motion-annotations", type=Path)
    parser.add_argument("--map-tools", type=Path, required=True)
    parser.add_argument("--maps-out", type=Path, required=True, help="new directory for saved maps and localization")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
