#!/usr/bin/env python3
"""Replay two independent clearance visits from the same saved LILocBench map."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from eval_bonn_map_changes import labeled_voxels, select_frames
from eval_lilocbench_changes import Visit
from eval_lilocbench_reuse import VOXEL_M, load_snapshot, localize, query_manifest
from lilocbench_motion import pose_errors, wheel_fallback
from openloris_map_changes import visibility
from persistent_clearance import ClearanceState, load_clearance, save_clearance
from rgbd_map_cloud import keyframe_poses


def score_voxels(voxels, box, stable):
    return {"all": len(voxels), "box": len(voxels & box), "annotated_stable": len(voxels & stable)}


def evidence(occupied, observations, intrinsics):
    voxels = sorted(occupied)
    points = (np.asarray(voxels).reshape(-1, 3) + 0.5) * VOXEL_M
    hits, free = visibility(points, observations, intrinsics, 0.10, free_margin=0.15, free_patch_radius=2)
    clear_mask = (free >= 2) & (free >= 2 * hits)
    occupied_mask = (hits >= 2) & (hits >= 2 * free)
    clear = {key for key, present in zip(voxels, clear_mask, strict=True) if present}
    occupied = {key for key, present in zip(voxels, occupied_mask, strict=True) if present}
    return clear, occupied


def evaluate(args):
    directory = args.work
    directory.mkdir(parents=True)
    specs = json.loads(args.annotations.read_text())["maps"][0]["visits"]
    original, placed = (Visit(args.data, "corridor_box", spec["sequence"]) for spec in specs)
    intrinsics = original.intrinsics
    first_frame = select_frames(specs[0]["window"], {item["frame"] for item in specs[0]["labels"]}, 1)[0]
    source_map = args.saved_maps / "visit_0" / "map"
    query_map = args.saved_maps / "updated_map"
    gauge = original.pose(original.times[first_frame]) @ np.linalg.inv(keyframe_poses(source_map, 0)[0][1])
    labels = [labeled_voxels(visit, spec["labels"], VOXEL_M, intrinsics)
              for spec, visit in zip(specs, (original, placed), strict=True)]
    box = labels[1]["box"]
    stable = (labels[0]["stable"] | labels[0]["floor"] | labels[1]["stable"] | labels[1]["floor"]) - box

    initial = load_snapshot(query_map / "occupancy.npz")
    state = ClearanceState(frozenset(initial))
    immediate_removed = set()
    rows = {}
    snapshots = directory / "clearance"
    snapshots.mkdir()
    for sequence, reference in (("lt_changes_dynamics_1", False), ("lt_changes_dynamics_0", True)):
        visit = Visit(args.data, "corridor_box", sequence, reference=reference)
        if not np.allclose(visit.intrinsics, intrinsics):
            raise ValueError("revisit camera calibration differs from the saved map")
        work = directory / sequence
        frames = list(range(len(visit.times)))
        query_manifest(visit, frames, work / "query")
        visual = localize(query_map, args.map_tools / "basalt_euroc_localize_demo", gauge, work)
        poses = wheel_fallback(visit, visual, frames)
        if not poses:
            raise ValueError(f"{sequence}: the saved map provided no visual anchor")
        depth_paths = visit.depth_files(poses, work / "depth")
        observations = [(poses[frame], path) for frame, path in depth_paths]
        cleared, reoccupied = evidence(initial, observations, intrinsics)
        previous = state
        state = state.observe(sequence, round(visit.times[0] * 1e9), cleared, reoccupied)
        saved = snapshots / f"{sequence}.npz"
        save_clearance(saved, state, VOXEL_M)
        state = load_clearance(saved, VOXEL_M)
        immediate_removed.update(cleared)
        rows[sequence] = {
            "start_timestamp_s": float(visit.times[0]), "frames": len(frames),
            "visual_fixes": len(visual), "poses_with_wheel": len(poses),
            "pose_scores": pose_errors(visit, poses) if reference else None,
            "single_visit_clearance": score_voxels(cleared, box, stable),
            "pending_after_reload": score_voxels(state.pending, box, stable),
            "pending_reset_by_occupied_observation": score_voxels(previous.pending & reoccupied, box, stable),
            "removed_this_visit_after_reload": score_voxels(previous.occupied - state.occupied, box, stable),
        }
    return {
        "labeled_voxels_in_initial_map": score_voxels(initial, box, stable), "visits": rows,
        "remove_on_first_clear_visit": score_voxels(immediate_removed, box, stable),
        "remove_after_two_clear_visits": score_voxels(initial - state.occupied, box, stable),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--saved-maps", type=Path, required=True)
    parser.add_argument("--map-tools", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
