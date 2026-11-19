"""Estimate a bounded relative LiDAR pose correction from the surrounding scene."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from kitti_tracking_lidar import transform


def thin(points, voxel=0.35):
    if len(points) == 0:
        return points
    _, index = np.unique(np.floor(points / voxel).astype(np.int32), axis=0, return_index=True)
    return points[index]


def background(sequence, frames, scored_region):
    low, high = np.asarray(scored_region)
    pad = np.array([9, 9, 1])
    samples = []
    for frame in frames:
        scan, pose = sequence.scan(frame)
        points = transform(scan, pose)
        outside = np.any((points < low) | (points >= high), axis=1)
        nearby = np.all((points >= low - pad) & (points < high + pad), axis=1)
        samples.append(points[outside & nearby])
    return thin(np.concatenate(samples))


def estimate_window_correction(reference, query):
    if min(len(reference), len(query)) < 500:
        raise ValueError("insufficient surrounding LiDAR overlap")
    tree = cKDTree(reference)
    _, neighbors = tree.query(reference, k=12)
    samples = reference[neighbors]
    centered = samples - samples.mean(axis=1, keepdims=True)
    values, basis = np.linalg.eigh(np.swapaxes(centered, 1, 2) @ centered)
    normals = basis[:, :, 0]
    planar = values[:, 0] <= 0.15 * values[:, 1]
    pivot = np.median(query, axis=0)
    correction = np.eye(4)

    def correspondences(points):
        distance, ids = tree.query(points)
        keep = (distance < 0.85) & planar[ids]
        x, y = points[keep], reference[ids[keep]]
        n = normals[ids[keep]]
        return x, n, np.einsum("ij,ij->i", x - y, n)

    _, _, first = correspondences(query)
    for _ in range(15):
        source, normal, error = correspondences(transform(query, correction))
        if len(error) < 300:
            raise ValueError("insufficient planar registration correspondences")
        jacobian = np.column_stack((np.cross(source - pivot, normal), normal))
        weights = 1 / np.sqrt(1 + (error / 0.15) ** 2)
        step, _, rank, _ = np.linalg.lstsq(jacobian * weights[:, None], -error * weights, rcond=1e-5)
        if rank < 6:
            raise ValueError("degenerate background registration")
        rotation = Rotation.from_rotvec(step[:3]).as_matrix()
        translation = pivot + step[3:] - rotation @ pivot
        correction[:3, :3] = rotation @ correction[:3, :3]
        correction[:3, 3] = rotation @ correction[:3, 3] + translation
        if np.linalg.norm(step) < 1e-4:
            break
    _, _, final = correspondences(transform(query, correction))
    displacement = transform(pivot, correction) - pivot
    angle = np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude())
    if min(len(first), len(final)) < 300:
        raise ValueError("insufficient planar registration coverage")
    initial = float(np.median(np.abs(first)))
    after = float(np.median(np.abs(final)))
    accepted = after < initial and np.linalg.norm(displacement) <= 1 and angle <= 3
    diagnostics = {"background_matches_before": len(first), "background_matches_after": len(final),
                   "median_plane_residual_before_m": initial, "median_plane_residual_after_m": after,
                   "pivot_displacement_m": displacement.tolist(), "rotation_deg": float(angle), "accepted": bool(accepted)}
    return correction if accepted else np.eye(4), diagnostics
