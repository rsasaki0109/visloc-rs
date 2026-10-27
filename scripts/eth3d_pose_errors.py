#!/usr/bin/env python3
"""Pose errors of a COLMAP text model against ETH3D ground truth.

    eth3d_pose_errors.py EST_IMAGES_TXT GT_IMAGES_TXT [IMAGE_NAMES_TXT]

Prints the registered count, the median relative-rotation error over all
image pairs, and camera-centre errors (median / p90 / max) after a
rotation-aware Sim(3): the rotation is the robust chordal mean of the
per-image rotation offsets, then scale and translation are fitted on the
centres of the images whose rotation agrees and refitted without the
centres beyond three times the median error. Image names are compared by
file name; IMAGE_NAMES_TXT
(`gsplat_photos`' `<out>/image_names.txt`, lines `frame_NNNNN.png <path>`)
maps the working copies' names back to the originals.
"""

from __future__ import annotations

import sys

import numpy as np


def rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_images(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lines = [line for line in open(path) if not line.startswith("#")]
    out = {}
    for header in lines[::2]:
        t = header.split()
        if len(t) < 10:
            continue
        r = rotation(np.array(list(map(float, t[1:5]))))
        out[t[9].split("/")[-1]] = (r, np.array(list(map(float, t[5:8]))))
    return out


def angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def mean_rotation(rs: list[np.ndarray]) -> np.ndarray:
    u, _, vt = np.linalg.svd(sum(rs))
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def main() -> None:
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    est = read_images(sys.argv[1])
    gt = read_images(sys.argv[2])
    if len(sys.argv) == 4:
        names = dict(
            (line.split()[0], line.split()[1].split("/")[-1]) for line in open(sys.argv[3])
        )
        est = {names.get(k, k): v for k, v in est.items()}
    common = sorted(set(gt) & set(est))
    if len(common) < 3:
        sys.exit(f"only {len(common)} images in common with the ground truth")

    rel = [
        angle_deg((est[j][0] @ est[i][0].T) @ (gt[j][0] @ gt[i][0].T).T)
        for a, i in enumerate(common)
        for j in common[a + 1 :]
    ]

    offsets = [gt[n][0].T @ est[n][0] for n in common]
    m = mean_rotation(offsets)
    off = np.array([angle_deg(r @ m.T) for r in offsets])
    keep = off <= 3 * np.median(off) + 1e-9
    m = mean_rotation([r for r, k in zip(offsets, keep) if k])

    a = np.array([-est[n][0].T @ est[n][1] for n in common])
    b = np.array([-gt[n][0].T @ gt[n][1] for n in common])

    def fit(keep: np.ndarray) -> tuple[float, np.ndarray]:
        ma, mb = a[keep].mean(0), b[keep].mean(0)
        ra = (a[keep] - ma) @ m.T
        s = ((b[keep] - mb) * ra).sum() / (ra**2).sum()
        return s, mb - s * m @ ma

    # Fit on the images whose rotation agrees, then refit without the
    # centres beyond three times the median error.
    s, t = fit(keep)
    e = np.linalg.norm(s * a @ m.T + t - b, axis=1)
    s, t = fit(e <= 3 * np.median(e) + 1e-12)
    e = np.linalg.norm(s * a @ m.T + t - b, axis=1)
    print(
        f"{len(common)}/{len(gt)} reg  relrot med {np.median(rel):.3f} deg  "
        f"centre med {np.median(e) * 100:.2f} p90 {np.percentile(e, 90) * 100:.2f} "
        f"max {e.max() * 100:.1f} cm"
    )


if __name__ == "__main__":
    main()
