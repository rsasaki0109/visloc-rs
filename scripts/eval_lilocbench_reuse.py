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
from openloris_map_changes import protected_clearance, surface_voxels, visibility
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


def voxel_update(occupied, prior, latest, intrinsics, policy):
    new = surface_voxels(latest, intrinsics, 8, VOXEL_M, None)
    keys = sorted(occupied | new)
    centers = (np.asarray(keys).reshape(-1, 3) + 0.5) * VOXEL_M
    signs = change_signs(centers, [prior, latest], intrinsics)
    added = {key for key, sign in zip(keys, signs, strict=True) if sign == 1 and key in new}
    removed = {key for key, sign in zip(keys, signs, strict=True) if sign == -1 and key in occupied}
    if policy != "baseline":
        removed = clearance_candidates(removed, latest, intrinsics, policy)
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


def clearance_candidates(occupied, observed, intrinsics, policy):
    keys = sorted(occupied)
    points = (np.asarray(keys).reshape(-1, 3) + 0.5) * VOXEL_M
    if policy != "baseline":
        cleared = protected_clearance(
            points, observed, intrinsics, min_baseline_m=0.25 if policy == "separated" else 0.0,
        )
    else:
        hits, free = visibility(points, observed, intrinsics, 0.10)
        cleared = (free >= 2) & (free >= 2 * hits)
    return {key for key, clear in zip(keys, cleared, strict=True) if clear}


def reobserved_surfaces(occupied, earlier, latest, intrinsics):
    keys = sorted(occupied)
    points = (np.asarray(keys).reshape(-1, 3) + 0.5) * VOXEL_M
    confirmed = np.ones(len(keys), dtype=bool)
    for observations in (earlier, latest):
        hits, free = visibility(points, observations, intrinsics, 0.10)
        confirmed &= (hits >= 2) & (hits >= 2 * free)
    return {key for key, keep in zip(keys, confirmed, strict=True) if keep}


def assess_reloaded_geometry(path, observed, intrinsics, original_box, stable, policy):
    occupied = load_snapshot(path)
    candidate_box = original_box & occupied
    stable_in_map = stable & occupied
    cleared = clearance_candidates(candidate_box | stable_in_map, observed, intrinsics, policy)
    return {
        "reloaded_surface_voxels": len(occupied),
        "known_visit_two_box_voxels_stored": len(candidate_box),
        "known_visit_two_box_voxels_now_observed_free": len(cleared & candidate_box) if observed else None,
        "stored_stable_voxels": len(stable_in_map),
        "stable_voxels_falsely_observed_free": len(cleared & stable_in_map) if observed else None,
    }


def run(args):
    config = json.loads(args.annotations.read_text())["maps"][0]
    visits = [Visit(args.data, "corridor_box", name)
              for name in ("mapping", "lt_changes_0", "lt_changes_dynamics_0")]
    policy = args.clearance_policy
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
    updated, added, removed = voxel_update(initial, first, second, intrinsics, policy)
    before, after = source_map / "occupancy.npz", merged_map / "occupancy.npz"
    snapshot(before, initial)
    snapshot(after, updated)
    support_path = merged_map / "reobserved_surfaces.npz"
    if args.keep_reobserved:
        snapshot(support_path, reobserved_surfaces(updated, first, second, intrinsics))
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
                occupancy, observations_at(third_depth, estimates), third.intrinsics, box, stable, policy,
            ),
        }
    motion = None
    if args.motion_annotations or args.save_third_update:
        continued = wheel_fallback(third, estimates, frames)
    if args.motion_annotations:
        motion = score_motion(args.motion_annotations, third, first, {
            "visual_only": estimates, "visual_with_wheel_fallback": continued,
            "reference_poses": {frame: third.pose(third.times[frame]) for frame in frames},
        })
        variants["updated_with_wheel_fallback"] = assess_reloaded_geometry(
            after, observations_at(third_depth, continued), third.intrinsics, box, stable, policy,
        )
    oracle = assess_reloaded_geometry(after, third_depth, third.intrinsics, box, stable, policy)
    third_saved = None
    if args.save_third_update:
        observed = observations_at(third_depth, continued)
        previous = load_snapshot(after)
        cleared = clearance_candidates(previous, observed, third.intrinsics, policy)
        third_path = root / "occupancy_after_third_visit.npz"
        snapshot(third_path, previous - cleared)
        removed_on_reload = previous - load_snapshot(third_path)
        third_saved = {
            "total_voxels_removed": len(removed_on_reload),
            "box_voxels_removed": len(removed_on_reload & box),
            "annotated_stable_voxels_removed": len(removed_on_reload & stable),
        }
        if args.keep_reobserved:
            held = cleared & load_snapshot(support_path)
            protected_path = root / "occupancy_after_third_visit_with_history.npz"
            snapshot(protected_path, previous - (cleared - held))
            confirmed = previous - load_snapshot(protected_path)
            third_saved["with_history"] = {
                "total_voxels_removed": len(confirmed),
                "box_voxels_removed": len(confirmed & box),
                "annotated_stable_voxels_removed": len(confirmed & stable),
                "deferred_voxels": len(held),
            }
    return {"clearance_policy": policy, "third_saved_update": third_saved,
            "visits": [v.root.name for v in visits], "capture_start_s": [float(v.times[0]) for v in visits],
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
    parser.add_argument("--clearance-policy", choices=("baseline", "patch", "separated"), default="baseline",
                        help="widen the depth patch and optionally require separated camera poses")
    parser.add_argument("--save-third-update", action="store_true", help="write and reopen visit three occupancy after clearance")
    parser.add_argument("--keep-reobserved", action="store_true",
                        help="also save a snapshot that defers clearance of surfaces supported in both prior visits")
    parser.add_argument("--map-tools", type=Path, required=True)
    parser.add_argument("--maps-out", type=Path, required=True, help="new directory for saved maps and localization")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.keep_reobserved and not args.save_third_update:
        parser.error("--keep-reobserved requires --save-third-update")
    cv2.setNumThreads(1)
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
