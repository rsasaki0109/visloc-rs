#!/usr/bin/env python3
"""Classify OpenLORIS RGB-D surface voxels from a merged map's camera poses."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

from rgbd_map_cloud import colour_intrinsics, keyframe_poses, nearest, read_stamped, yaml_matrices


def frames(map_path, session, root, every, max_dt):
    transforms = yaml_matrices(root / "trans_matrix.yaml")
    base_from_fisheye = transforms["base_link", "t265_fisheye1_optical_frame"]
    base_from_colour = transforms["base_link", "d400_color_optical_frame"]
    fisheye_from_colour = np.linalg.inv(base_from_fisheye) @ base_from_colour
    times, paths = read_stamped(root / "aligned_depth.txt")
    if not len(times):
        raise ValueError(f"{root}: no aligned depth timestamps")
    used = set()
    for time, world_from_fisheye in keyframe_poses(map_path, session=session)[::every]:
        i = nearest(times, time)
        if i in used or abs(time - times[i]) > max_dt:
            continue
        used.add(i)
        yield world_from_fisheye @ fisheye_from_colour, root / paths[i]


def load_depth(path, masks):
    with Image.open(path) as source:
        depth = np.asarray(source, dtype=np.float32) * 0.001
    if masks:
        mask_path = masks / path.name
        if mask_path.exists():
            with Image.open(mask_path) as source:
                mask = np.asarray(source)
            if mask.shape != depth.shape:
                raise ValueError(f"{mask_path}: expected a single-channel mask on the depth grid")
            depth[mask != 0] = 0.0
    return depth


def surface_voxels(observations, intrinsics, stride, voxel, masks):
    fx, fy, cx, cy = intrinsics
    occupied = set()
    for pose, path in observations:
        depth = load_depth(path, masks)
        y, x = np.mgrid[:depth.shape[0]:stride, :depth.shape[1]:stride]
        z = depth[y, x]
        valid = (z >= 0.3) & (z <= 4.0)
        z = z[valid]
        points = np.column_stack(((x[valid] - cx) * z / fx, (y[valid] - cy) * z / fy, z))
        world = points @ pose[:3, :3].T + pose[:3, 3]
        keys = np.unique(np.floor(world / voxel).astype(np.int32), axis=0)
        occupied.update(map(tuple, keys))
    return occupied


def visibility(points, observations, intrinsics, tolerance, masks=None, *, evidence=None,
               free_margin=None, free_patch_radius=1, free_baseline_m=0.0):
    fx, fy, cx, cy = intrinsics
    hits = np.zeros(len(points), dtype=np.uint32)
    free = np.zeros_like(hits)
    margin = tolerance if free_margin is None else free_margin
    offsets = range(-free_patch_radius, free_patch_radius + 1)
    free_history = []
    diverse_free = np.zeros(len(points), dtype=bool)
    if evidence is not None:
        evidence.update({key: np.zeros_like(hits) for key in ("projected", "valid_center", "center_free", "occluded")})
    for pose, path in observations:
        depth = load_depth(path, masks)
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        ids = np.flatnonzero((local[:, 2] >= 0.3) & (local[:, 2] <= 4.0))
        x, y, z = local[ids].T
        u = np.rint(fx * x / z + cx).astype(int)
        v = np.rint(fy * y / z + cy).astype(int)
        valid = (u > 0) & (v > 0) & (u < depth.shape[1] - 1) & (v < depth.shape[0] - 1)
        ids, u, v, z = ids[valid], u[valid], v[valid], z[valid]
        center = depth[v, u]
        valid_center = (center >= 0.3) & (center <= 4)
        hits[ids[valid_center & (np.abs(center - z) <= tolerance)]] += 1
        if evidence is not None:
            evidence["projected"][ids] += 1
            evidence["valid_center"][ids[valid_center]] += 1
            evidence["center_free"][ids[valid_center & (center > z + tolerance)]] += 1
            evidence["occluded"][ids[valid_center & (center < z - tolerance)]] += 1
        # Only vote for clearance when the entire patch has valid depth beyond the point.
        r = free_patch_radius
        inside = (u >= r) & (v >= r) & (u < depth.shape[1] - r) & (v < depth.shape[0] - r)
        u, v, z = u[inside], v[inside], z[inside]
        patch = np.stack([depth[v + dy, u + dx] for dy in offsets for dx in offsets])
        clear_ids = ids[inside][(patch.min(axis=0) > z + margin) & (patch.max(axis=0) <= 4)]
        free[clear_ids] += 1
        if free_baseline_m > 0:
            origin = pose[:3, 3]
            for prior_origin, prior_clear in free_history:
                if np.linalg.norm(origin - prior_origin) >= free_baseline_m:
                    diverse_free[clear_ids] |= prior_clear[clear_ids]
            clear = np.zeros(len(points), dtype=bool)
            clear[clear_ids] = True
            free_history.append((origin, clear))
    if free_baseline_m > 0:
        free[~diverse_free] = 0
    return hits, free


def protected_clearance(points, observations, intrinsics, *, min_baseline_m=0.0):
    """Require free space across a wider patch and optionally separated camera origins."""
    hits, free = visibility(points, observations, intrinsics, 0.10, free_margin=0.15,
                            free_patch_radius=2, free_baseline_m=min_baseline_m)
    return (free >= 2) & (free >= 2 * hits)


def classify(hits, free, min_views):
    occupied = (hits >= min_views) & (hits >= 2 * free)
    cleared = (free >= min_views) & (free >= 2 * hits)
    mixed = (hits >= min_views) & (free >= min_views) & ~(occupied | cleared)
    # 0 unknown, 1 stable, 2 cross-session change, 3 intermittent visibility.
    labels = np.zeros(hits.shape[1], dtype=np.uint8)
    labels[(occupied.sum(axis=0) >= 2) & ~(cleared | mixed).any(axis=0)] = 1
    labels[occupied.any(axis=0) & cleared.any(axis=0)] = 2
    labels[(labels == 0) & mixed.any(axis=0)] = 3
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--session", nargs=2, metavar=("INDEX", "RGBD_DIR"), action="append", required=True)
    parser.add_argument("--moving-masks", nargs=2, metavar=("INDEX", "MASK_DIR"), action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--voxel", type=float, default=0.07)
    parser.add_argument("--depth-margin", type=float, default=0.10)
    parser.add_argument("--max-dt-ms", type=float, default=20.0)
    parser.add_argument("--min-views", type=int, default=2)
    args = parser.parse_args()
    if not all(getattr(args, k) > 0 for k in ("every", "stride", "voxel", "depth_margin", "max_dt_ms", "min_views")):
        parser.error("sampling and threshold values must be positive")
    if not np.isfinite([args.voxel, args.depth_margin, args.max_dt_ms]).all():
        parser.error("depth and time thresholds must be finite")
    try:
        sessions = {int(i): Path(root) for i, root in args.session}
        masks = {int(i): Path(root) for i, root in args.moving_masks}
    except ValueError:
        parser.error("session indices must be integers")
    if len(sessions) < 2 or len(sessions) != len(args.session):
        parser.error("provide at least two distinct sessions")
    if len(masks) != len(args.moving_masks) or any(i not in sessions or not p.is_dir() for i, p in masks.items()):
        parser.error("moving masks must refer to distinct input sessions and existing directories")
    saved_sessions = json.loads((args.map / "map.json").read_text()).get("sessions", [])
    if any(not 0 <= i < len(saved_sessions) for i in sessions):
        parser.error("session indices must refer to the merged map's sessions")

    voxels = set()
    samples = []
    for index, root in sorted(sessions.items()):
        intrinsics = colour_intrinsics(root / "sensors.yaml")
        observations = list(frames(args.map, index, root, args.every, args.max_dt_ms * 1e-3))
        if not observations:
            raise ValueError(f"session {index}: no synchronized depth frames")
        mask = masks.get(index)
        voxels.update(surface_voxels(observations, intrinsics, args.stride, args.voxel, mask))
        samples.append((observations, intrinsics, mask))
    if not voxels:
        raise ValueError("no valid RGB-D surface voxels")

    centers = (np.asarray(sorted(voxels), dtype=float) + 0.5) * args.voxel
    hits, free = (np.stack(v) for v in zip(*(
        visibility(centers, observations, intrinsics, args.depth_margin, mask)
        for observations, intrinsics, mask in samples
    )))
    labels = classify(hits, free, args.min_views)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.out.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(
                stream, centers=centers, label=labels, occupied_views=hits, free_views=free,
                session_ids=np.asarray(sorted(sessions)),
                sampled_frames=np.asarray([len(o) for o, _, _ in samples]),
                voxel_m=args.voxel, depth_margin_m=args.depth_margin, min_views=args.min_views,
            )
            stream.close()
            temporary.replace(args.out)
        finally:
            temporary.unlink(missing_ok=True)
    print(dict(zip(("unknown", "stable", "changed", "intermittent"), np.bincount(labels, minlength=4).tolist())))


if __name__ == "__main__":
    main()
