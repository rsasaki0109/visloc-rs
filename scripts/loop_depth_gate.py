"""Independent RGB-D agreement for selected visual loop measurements."""

import numpy as np
from scipy.spatial.transform import Rotation

DEPTH_TOLERANCE_M = 0.07
MIN_COMPARISONS = 100
MIN_AGREEMENT = 0.70


def se3(document):
    w, x, y, z = document["quaternion_wxyz"]
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    pose[:3, 3] = document["translation"]
    return pose


def compare_depth(source, target, source_k, target_k, target_from_source):
    y, x = np.mgrid[:source.shape[0]:8, :source.shape[1]:8]
    z = source[y, x]
    valid = (z >= 0.3) & (z <= 4.0)
    z = z[valid]
    fx, fy, cx, cy = source_k
    local = np.column_stack(((x[valid] - cx) * z / fx, (y[valid] - cy) * z / fy, z))
    projected = local @ target_from_source[:3, :3].T + target_from_source[:3, 3]
    x, y, z = projected[(projected[:, 2] >= 0.3) & (projected[:, 2] <= 4.0)].T
    fx, fy, cx, cy = target_k
    u = np.rint(fx * x / z + cx).astype(int)
    v = np.rint(fy * y / z + cy).astype(int)
    inside = (0 <= u) & (u < target.shape[1]) & (0 <= v) & (v < target.shape[0])
    observed = target[v[inside], u[inside]]
    valid = (observed >= 0.3) & (observed <= 4.0)
    error = np.abs(z[inside][valid] - observed[valid])
    return {
        "count": len(error),
        "agreement": float(np.mean(error <= DEPTH_TOLERANCE_M)) if len(error) else None,
        "median_abs_error_m": float(np.median(error)) if len(error) else None,
    }


def audit_loop(base, query, base_k, query_k, query_from_base):
    forward = compare_depth(base, query, base_k, query_k, query_from_base)
    reverse = compare_depth(query, base, query_k, base_k, np.linalg.inv(query_from_base))
    if min(forward["count"], reverse["count"]) < MIN_COMPARISONS:
        decision = "unavailable"
    elif min(forward["agreement"], reverse["agreement"]) >= MIN_AGREEMENT:
        decision = "pass"
    else:
        decision = "reject"
    return {"decision": decision, "base_to_query": forward, "query_to_base": reverse}


def validation_pairs(base_poses, query_poses, maximum=16):
    """Fix co-view candidates using session-only poses and temporal subsampling."""
    queries = sorted(query_poses)
    chosen = []
    used = set()
    for image in np.linspace(0, len(queries) - 1, min(maximum, len(queries)), dtype=int):
        query = queries[image]
        pose = query_poses[query]
        ordered = []
        for base, reference in base_poses.items():
            distance = np.linalg.norm(reference[:3, 3] - pose[:3, 3])
            angle = Rotation.from_matrix(reference[:3, :3].T @ pose[:3, :3]).magnitude()
            if distance <= 0.35 and angle <= np.deg2rad(35) and base not in used:
                ordered.append((distance + 0.25 * angle, base))
        if ordered:
            _, base = min(ordered)
            used.add(base)
            chosen.append((base, query))
    return chosen


def score_validation_depth(pairs, poses, depth_rows, intrinsics):
    base_rows, query_rows = depth_rows
    agreements = []
    for a, b in pairs:
        transform = np.linalg.inv(poses[1][b] @ query_rows[b][1]) @ poses[0][a] @ base_rows[a][1]
        audit = audit_loop(base_rows[a][0], query_rows[b][0], *intrinsics, transform)
        if audit["decision"] == "unavailable":
            return None
        agreements.extend([audit["base_to_query"]["agreement"], audit["query_to_base"]["agreement"]])
    return float(np.mean(agreements)) if agreements else None


def select_depth_variant(scores, pair_count):
    """Require a small held-out geometry gain and preserve registration on ties."""
    reference = scores["session_alignment"]
    if pair_count < 5 or reference is None:
        return "session_alignment"
    improvements = [(value, name) for name, value in scores.items()
                    if value is not None and value >= reference + 0.005]
    return max(improvements)[1] if improvements else "session_alignment"
