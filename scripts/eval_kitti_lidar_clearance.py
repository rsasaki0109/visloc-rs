#!/usr/bin/env python3
"""Replay occupancy clearance on public KITTI Tracking LiDAR and scored cuboids."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial import cKDTree

from kitti_tracking_lidar import TrackingSequence, transform
from refine_kitti_background import background, estimate_window_correction


def voxel_keys(points, voxel):
    return set(map(tuple, np.unique(np.floor(points / voxel).astype(np.int32), axis=0)))


def points_in_region(points, bounds):
    lower, upper = np.asarray(bounds)
    return ((points >= lower) & (points < upper)).all(axis=1)


def ray_votes(cells, scan, pose, voxel, margin):
    # Work with measured beam directions and the exact world-aligned voxel box.
    world_points = transform(scan, pose)
    origin = pose[:3, 3]
    rays = world_points - origin
    radius = np.linalg.norm(rays, axis=1)
    valid = (radius > 0.5) & np.isfinite(radius)
    if np.count_nonzero(valid) < 2:
        return np.zeros(len(cells), dtype=bool), np.zeros(len(cells), dtype=bool)
    unit = rays[valid] / radius[valid, None]
    ranges = radius[valid]
    near_corner = cells * voxel - origin
    far_corner = near_corner + voxel
    delta = near_corner + voxel / 2
    lengths = np.linalg.norm(delta, axis=1)
    eligible = lengths > voxel
    directions = delta[eligible] / lengths[eligible, None]
    _, indices = cKDTree(unit).query(directions, k=min(8, len(unit)), workers=1)
    neighbors = unit[indices]
    parallel = np.abs(neighbors) < 1e-9
    stable = np.where(parallel, 1.0, neighbors)
    lo, hi = near_corner[eligible, None, :], far_corner[eligible, None, :]
    t0, t1 = lo / stable, hi / stable
    entry = np.where(parallel, -np.inf, np.minimum(t0, t1)).max(axis=2)
    exit = np.where(parallel, np.inf, np.maximum(t0, t1)).min(axis=2)
    intersects = (exit >= np.maximum(entry, 0)) & ~np.any(parallel & ((lo > 0) | (hi < 0)), axis=2)
    endpoint = ranges[indices]
    hit = np.any(intersects & (endpoint >= entry - margin) & (endpoint <= exit + margin), axis=1)
    hidden = np.any(intersects & (endpoint < entry - margin), axis=1)
    clear = np.any(intersects & (endpoint > exit + margin), axis=1) & ~(hit | hidden)
    hits, frees = np.zeros(len(cells), dtype=bool), np.zeros(len(cells), dtype=bool)
    hits[eligible], frees[eligible] = hit, clear
    return hits, frees


def scores(predicted, positive, stable):
    tp, fp, fn = len(predicted & positive), len(predicted & stable), len(positive - predicted)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def insertion_candidates(sequence, config, correction, initial):
    frames = set(config["insertion_frames"])
    label_frames = set(config["insertion_label_frames"])
    if not frames or not label_frames or frames & label_frames:
        raise ValueError("expected separate insertion and insertion label frames")
    bounds, voxel = config["world_region_m"], config["voxel_m"]
    counts = Counter()
    observations = []
    for frame in sorted(frames):
        scan, pose = sequence.scan(frame)
        aligned = correction @ pose
        world = transform(scan, aligned)
        counts.update(voxel_keys(world[points_in_region(world, bounds)], voxel))
        observations.append((scan, aligned))

    stationary, moving = set(), set()
    for frame in sorted(label_frames):
        scan, pose = sequence.scan(frame)
        native = transform(scan, pose)
        world = transform(native, correction)
        region = points_in_region(world, bounds)
        for track in [config["moved_track"], *config["stationary_tracks"]]:
            if track not in sequence.labels[frame]:
                continue
            mask = region & sequence.in_box(native, frame, track, inset_m=0.15)
            labels = moving if track == config["moved_track"] else stationary
            labels.update(voxel_keys(world[mask], voxel))
    ambiguous = stationary & moving
    stationary -= ambiguous | initial
    moving -= ambiguous | initial
    seen = counts.keys() - initial
    persistent = {key for key in seen if counts[key] >= 2}
    ordered = sorted(persistent)
    clear_evidence = np.zeros(len(ordered), dtype=bool)
    cells = np.asarray(ordered).reshape(-1, 3)
    for scan, pose in observations:
        _, free = ray_votes(cells, scan, pose, voxel, config["endpoint_margin_m"])
        clear_evidence |= free
    consistent = {key for key, clear in zip(ordered, clear_evidence, strict=True) if not clear}
    return seen, persistent, consistent, stationary, moving


def run(config, root, out, refine):
    sequence = TrackingSequence(root, config["sequence"])
    fits = set(config["map_frames"])
    labels = set(config["label_frames"])
    replay = set(config["replay_frames"])
    if min(map(len, (fits, labels, replay))) == 0 or fits & labels or fits & replay or labels & replay:
        raise ValueError("expected disjoint map, label, and replay frames")
    voxel = config["voxel_m"]
    bounds = config["world_region_m"]
    initial = set()
    for frame in config["map_frames"]:
        scan, pose = sequence.scan(frame)
        points = transform(scan, pose)
        initial.update(voxel_keys(points[points_in_region(points, bounds)], voxel))

    group = {track: set() for track in [config["moved_track"], *config["stationary_tracks"]]}
    for frame in config["label_frames"]:
        scan, pose = sequence.scan(frame)
        points = transform(scan, pose)
        points = points[points_in_region(points, bounds)]
        for track in group:
            group[track].update(voxel_keys(points[sequence.in_box(points, frame, track, inset_m=0.15)], voxel))
    target = group.pop(config["moved_track"])
    stable = set().union(*group.values()) - target
    tracked_stable = {str(track): len(voxels & initial & stable) for track, voxels in group.items()}
    positive = target & initial
    stable &= initial
    if not positive or not stable:
        raise ValueError("map lacks moving or stable annotated voxels")

    correction, registration = np.eye(4), None
    if refine:
        correction, registration = estimate_window_correction(
            background(sequence, config["map_frames"], bounds),
            background(sequence, config["replay_frames"], bounds),
        )
    keys = sorted(initial)
    cells = np.asarray(keys)
    later = set()
    hits, frees = np.zeros(len(keys), dtype=int), np.zeros(len(keys), dtype=int)
    for frame in config["replay_frames"]:
        scan, pose = sequence.scan(frame)
        pose = correction @ pose
        points = transform(scan, pose)
        later.update(voxel_keys(points[points_in_region(points, bounds)], voxel))
        h, f = ray_votes(cells, scan, pose, voxel, config["endpoint_margin_m"])
        hits += h
        frees += f

    missing = initial - later
    cleared = {key for key, clear in zip(keys, (frees >= config["min_clear_scans"]) & (frees >= 2 * hits), strict=True)
               if clear}
    seen, persistent, consistent, new_static, moving = insertion_candidates(sequence, config, correction, initial)
    # A measured occupied voxel takes precedence over clearance from other views.
    snapshot = (initial - (cleared & missing)) | consistent
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=out.parent, suffix=".npz", delete=False) as staging:
        temporary = Path(staging.name)
    try:
        np.savez_compressed(temporary, occupied=np.asarray(sorted(snapshot), dtype=np.int32).reshape(-1, 3),
                            voxel_m=voxel, initial_frames=config["map_frames"], replay_frames=config["replay_frames"])
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    with np.load(out, allow_pickle=False) as stored:
        reloaded = set(map(tuple, stored["occupied"]))
    if reloaded != snapshot:
        raise ValueError("reloaded occupancy differs from the saved update")
    return {"registration": registration, "scene_voxels_in_initial_map": len(initial), "scene_voxels_after_update": len(reloaded),
            "label_target_voxels": len(target), "target_voxels_present_in_map": len(positive),
            "stationary_track_voxels_present_in_map": tracked_stable, "stationary_voxels_present_in_map": len(stable),
            "occupancy_difference": scores(missing, positive, stable),
            "ray_only": scores(cleared, positive, stable),
            "saved_clearance_with_occupied_veto": scores(initial - reloaded, positive, stable),
            "insertion": {"labeled_static_voxels": len(new_static), "labeled_moving_voxels": len(moving),
                          "all_new_returns": scores(seen, new_static, moving),
                          "two_scan_insertions": scores(persistent, new_static, moving),
                          "saved_insertions_without_free_evidence": scores(reloaded - initial, new_static, moving),
                          "scene_voxels_inserted": len(reloaded - initial)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "annotations", "map-out", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--refine-background", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.annotations.read_text())
    data = run(config, args.data, args.map_out, args.refine_background)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
