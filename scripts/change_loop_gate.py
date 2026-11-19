"""Select a pose hypothesis by repeatability of nearby signed depth changes."""

import numpy as np

from loop_depth_gate import select_depth_variant
from openloris_map_changes import classify, surface_voxels, visibility


def observations(frame_pairs, poses, depth_paths):
    return [[(poses[i][frame], depth_paths[i][frame]) for frame in {pair[i]: None for pair in frame_pairs}]
            for i in range(2)]


def change_signs(centers, samples, intrinsics):
    hits, free = (np.stack(v) for v in zip(*[
        visibility(centers, visit, intrinsics, 0.10) for visit in samples
    ], strict=True))
    changed = classify(hits, free, 2) == 2
    return np.where(changed, np.where((free[1] >= 2) & (free[1] >= 2 * hits[1]), -1, 1), 0)


def select_by_local_change(pairs, poses, depth_paths, intrinsics, global_scores):
    """Propose regions on one split, then count sign confirmation on another."""
    both_splits = [pairs[::2], pairs[1::2]]
    fallback = select_depth_variant(global_scores, len(pairs))
    result = {"preferred": fallback, "depth_frame_pairs": [len(p) for p in both_splits],
              "reason": "insufficient_probe_pairs", "regions": 0, "scores": {}}
    if min(result["depth_frame_pairs"]) < 4:
        return result
    discovery = {name: observations(both_splits[0], pair, depth_paths) for name, pair in poses.items()}
    grid = set()
    for pair in discovery.values():
        for visit in pair:
            grid.update(surface_voxels(visit, intrinsics, 8, 0.07, None))
    if not grid:
        result["reason"] = "no_surface_candidates"
        return result
    centers = (np.asarray(sorted(grid)) + 0.5) * 0.07
    signs = np.stack([change_signs(centers, visit, intrinsics) for visit in discovery.values()])
    additions, removals = (signs == 1).any(axis=0), (signs == -1).any(axis=0)
    valid = additions ^ removals
    positive = np.where(additions[valid], 1, -1)
    result["regions"] = len(positive)
    result["reason"] = "insufficient_local_change_evidence"
    if len(positive) < 32:
        return result
    for name, estimated in poses.items():
        validation = observations(both_splits[1], estimated, depth_paths)
        confirmed = change_signs(centers[valid], validation, intrinsics)
        result["scores"][name] = float(np.mean(confirmed == positive))
    local = select_depth_variant(result["scores"], len(pairs))
    result["preferred"] = local
    result["reason"] = "independent_signed_change_confirmation"
    return result
