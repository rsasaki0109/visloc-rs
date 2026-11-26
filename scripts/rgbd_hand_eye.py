#!/usr/bin/env python3
"""Calibrate the camera in a motion-capture body frame from an RGB-D sequence.

Some RGB-D datasets record the pose of the motion-capture markers, not of
the camera (Bonn RGB-D Dynamic: its published marker-to-camera matrix is not
a rotation). This estimates `T_body_cam`, the camera in the ground truth's
body frame, so that `T_world_cam = T_world_body * T_body_cam`, plus the time
offset between the ground truth and the images:

1. Frame pairs a few frames apart, with enough rotation between them.
2. Per pair, the camera's relative motion from the images alone: ORB matches
   between the two colour images, back-projected with the registered depth,
   then a RANSAC rigid fit of the 3D points (Kabsch).
3. The bodies' relative motion from the ground truth at the image stamps
   (plus a candidate time offset).
4. Hand-eye AX = XB: rotation by Park and Martin's closed form, translation
   by least squares, then the pairs whose residual is large are dropped and
   the fit repeated. The time offset is the one, on a grid, with the lowest
   residual.

Prints `T_body_cam` as `tx,ty,tz,qx,qy,qz,qw` and the residuals, and with
`--out-trajectory` writes the camera's ground truth (TUM, the image stamps
shifted by the offset) for `tum_rgbd_save_map --trajectory`.

Needs numpy and OpenCV (`opencv-python-headless`).

  python3 scripts/rgbd_hand_eye.py --seq-dir rgbd_bonn_static_close_far \\
    --fx 542.822841 --fy 542.576870 --cx 315.593520 --cy 237.756098 \\
    --dist 0.039903,-0.099343,-0.000730,-0.000144,0 --depth-scale 5000 \\
    --out-trajectory camera_groundtruth.tum
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def read_list(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            values = line.split()
            rows.append((float(values[0]), values[1]))
    return rows


def read_tum(path: Path):
    data = np.loadtxt(path, comments="#")
    return data[np.argsort(data[:, 0])]


def quat_to_rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(r):
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(d)
    return (math.sin((1 - a) * theta) * q0 + math.sin(a * theta) * q1) / math.sin(theta)


class Trajectory:
    def __init__(self, data):
        self.t = data[:, 0]
        self.p = data[:, 1:4]
        self.q = data[:, 4:8]

    def at(self, stamp, max_gap=0.1):
        i = int(np.searchsorted(self.t, stamp))
        if i == 0 or i == len(self.t) or self.t[i] - self.t[i - 1] > max_gap:
            return None
        a = (stamp - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        pose = np.eye(4)
        pose[:3, :3] = quat_to_rot(slerp(self.q[i - 1], self.q[i], a))
        pose[:3, 3] = self.p[i - 1] + a * (self.p[i] - self.p[i - 1])
        return pose


def log_rot(r):
    angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return angle / (2 * math.sin(angle)) * v


def kabsch(a, b):
    """Rigid T with b ~ R a + t."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    t = cb - r @ ca
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = r, t
    return pose


def ransac_rigid(a, b, threshold=0.03, iterations=300, rng=None):
    rng = rng or np.random.default_rng(0)
    best = None
    for _ in range(iterations):
        idx = rng.choice(len(a), 3, replace=False)
        pose = kabsch(a[idx], b[idx])
        err = np.linalg.norm((a @ pose[:3, :3].T + pose[:3, 3]) - b, axis=1)
        inliers = err < threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 20:
        return None, 0
    return kabsch(a[best], b[best]), int(best.sum())


class Frames:
    def __init__(self, args, seq_dir):
        self.seq = seq_dir
        self.rgb = read_list(self.seq / "rgb.txt")
        depth = read_list(self.seq / "depth.txt")
        self.depth_t = np.array([t for t, _ in depth])
        self.depth_names = [n for _, n in depth]
        self.k = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]])
        self.dist = np.array([float(v) for v in args.dist.split(",")])
        self.depth_scale = args.depth_scale
        self.orb = cv2.ORB_create(1500)
        self.cache = {}

    def features(self, index):
        if index in self.cache:
            return self.cache[index]
        stamp, name = self.rgb[index]
        gray = cv2.imread(str(self.seq / name), cv2.IMREAD_GRAYSCALE)
        j = int(np.argmin(np.abs(self.depth_t - stamp)))
        depth = None
        if abs(self.depth_t[j] - stamp) < 0.02:
            depth = cv2.imread(str(self.seq / self.depth_names[j]), cv2.IMREAD_UNCHANGED)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        result = None
        if depth is not None and descriptors is not None:
            pixels = np.array([kp.pt for kp in keypoints], dtype=np.float64)
            z = depth[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] / self.depth_scale
            normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), self.k, self.dist).reshape(-1, 2)
            points = np.column_stack([normalized * z[:, None], z])
            valid = (z > 0.3) & (z < 6.0)
            result = (descriptors[valid], points[valid])
        self.cache[index] = result
        return result

    def relative(self, i, j):
        """T_cam_i_cam_j from the images."""
        fi, fj = self.features(i), self.features(j)
        if fi is None or fj is None or len(fi[0]) < 30 or len(fj[0]) < 30:
            return None, 0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(fj[0], fi[0])
        if len(matches) < 30:
            return None, 0
        a = np.array([fj[1][m.queryIdx] for m in matches])
        b = np.array([fi[1][m.trainIdx] for m in matches])
        return ransac_rigid(a, b)


def solve(pairs):
    """Hand-eye AX = XB over (A, B) pairs: X = T_body_cam."""
    m = np.zeros((3, 3))
    for a, b in pairs:
        m += np.outer(log_rot(b[:3, :3]), log_rot(a[:3, :3]))
    w, v = np.linalg.eigh(m.T @ m)
    r = v @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ v.T @ m.T
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a, _ in pairs])
    rhs = np.concatenate([r @ b[:3, 3] - a[:3, 3] for a, b in pairs])
    t = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    x = np.eye(4)
    x[:3, :3], x[:3, 3] = r, t
    return x


def residuals(x, pairs):
    out = []
    xi = np.linalg.inv(x)
    for a, b in pairs:
        e = np.linalg.inv(b) @ xi @ a @ x
        out.append((math.degrees(np.linalg.norm(log_rot(e[:3, :3]))), np.linalg.norm(e[:3, 3])))
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--seq-dir",
        type=Path,
        nargs="+",
        required=True,
        help="one or more sequences of the same rig; their pairs are pooled",
    )
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--dist", default="0,0,0,0,0", help="k1,k2,p1,p2,k3")
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--gap", type=int, default=8, help="frames between the two of a pair")
    parser.add_argument("--min-rotation-deg", type=float, default=4.0)
    parser.add_argument("--max-pairs", type=int, default=250)
    parser.add_argument("--offsets-ms", default="-100:100:10")
    parser.add_argument("--out-trajectory", type=Path, default=None)
    parser.add_argument(
        "--apply",
        default=None,
        help="tx,ty,tz,qx,qy,qz,qw: skip the calibration, write --out-trajectory with this T_body_cam",
    )
    parser.add_argument("--offset-ms", type=float, default=0.0, help="with --apply")
    args = parser.parse_args()
    if args.apply:
        v = [float(x) for x in args.apply.split(",")]
        x = np.eye(4)
        x[:3, :3], x[:3, 3] = quat_to_rot(np.array(v[3:7])), v[:3]
        if len(args.seq_dir) != 1 or args.out_trajectory is None:
            raise SystemExit("--apply takes one --seq-dir and --out-trajectory")
        write_camera_trajectory(args, x, args.offset_ms / 1000)
        return

    # Visual relative motions once; the ground truth's per time offset.
    visual = []
    for seq_dir in args.seq_dir:
        frames = Frames(args, seq_dir)
        truth = Trajectory(read_tum(seq_dir / "groundtruth.txt"))
        candidates = list(range(0, len(frames.rgb) - args.gap, max(1, args.gap // 2)))
        found = 0
        for i in candidates:
            ti, tj = frames.rgb[i][0], frames.rgb[i + args.gap][0]
            gi, gj = truth.at(ti), truth.at(tj)
            if gi is None or gj is None:
                continue
            a = np.linalg.inv(gi) @ gj
            if math.degrees(np.linalg.norm(log_rot(a[:3, :3]))) < args.min_rotation_deg:
                continue
            b, inliers = frames.relative(i, i + args.gap)
            if b is not None and inliers >= 40:
                visual.append((truth, ti, tj, b))
                found += 1
            if found >= args.max_pairs:
                break
        frames.cache.clear()
        print(f"{seq_dir.name}: {found} pairs with enough rotation and a visual fit")
    start, stop, step = (float(v) for v in args.offsets_ms.split(":"))
    best = None
    for offset_ms in np.arange(start, stop + step / 2, step):
        offset = offset_ms / 1000
        pairs = []
        for truth, ti, tj, b in visual:
            gi, gj = truth.at(ti + offset), truth.at(tj + offset)
            if gi is not None and gj is not None:
                pairs.append((np.linalg.inv(gi) @ gj, b))
        if len(pairs) < 10:
            continue
        x = solve(pairs)
        for _ in range(3):
            res = residuals(x, pairs)
            keep = (res[:, 0] < max(1.0, 3 * np.median(res[:, 0]))) & (
                res[:, 1] < max(0.01, 3 * np.median(res[:, 1]))
            )
            pairs = [p for p, k in zip(pairs, keep) if k]
            x = solve(pairs)
        res = residuals(x, pairs)
        score = float(np.median(res[:, 0]))
        print(
            f"offset {offset_ms:+6.0f} ms: {len(pairs)} pairs, residual median "
            f"{score:.3f} deg / {100 * np.median(res[:, 1]):.2f} cm"
        )
        if best is None or score < best[0]:
            best = (score, offset, x, res, pairs)
    score, offset, x, res, _ = best
    q = rot_to_quat(x[:3, :3])
    print(f"best time offset {1000 * offset:+.0f} ms (ground truth = image stamp + offset)")
    print(
        f"T_body_cam (tx,ty,tz,qx,qy,qz,qw): "
        f"{x[0, 3]:.5f},{x[1, 3]:.5f},{x[2, 3]:.5f},{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}"
    )
    print(
        f"residuals: rotation median {np.median(res[:, 0]):.3f} deg, "
        f"translation median {100 * np.median(res[:, 1]):.2f} cm over {len(res)} pairs"
    )
    rotations = np.array([np.linalg.norm(log_rot(a[:3, :3])) for a, _ in best[4]])
    print(f"pair rotations: median {math.degrees(np.median(rotations)):.1f} deg")
    if args.out_trajectory:
        if len(args.seq_dir) != 1:
            raise SystemExit("--out-trajectory takes one --seq-dir; use --apply per sequence")
        write_camera_trajectory(args, x, offset)


def write_camera_trajectory(args, x, offset):
    """The camera's ground truth at the image stamps: T_world_body * x."""
    seq_dir = args.seq_dir[0]
    truth = Trajectory(read_tum(seq_dir / "groundtruth.txt"))
    lines = ["# timestamp tx ty tz qx qy qz qw (camera, image stamps)"]
    for stamp, _ in read_list(seq_dir / "rgb.txt"):
        g = truth.at(stamp + offset)
        if g is None:
            continue
        c = g @ x
        cq = rot_to_quat(c[:3, :3])
        lines.append(
            f"{stamp:.6f} {c[0, 3]:.6f} {c[1, 3]:.6f} {c[2, 3]:.6f} "
            f"{cq[0]:.7f} {cq[1]:.7f} {cq[2]:.7f} {cq[3]:.7f}"
        )
    args.out_trajectory.write_text("\n".join(lines) + "\n")
    print(f"camera ground truth -> {args.out_trajectory}")


if __name__ == "__main__":
    main()
