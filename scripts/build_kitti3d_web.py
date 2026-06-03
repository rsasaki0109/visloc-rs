#!/usr/bin/env python3
"""Bundle KITTI trajectory results into a JSON the docs/kitti3d/ web viewer
renders as interactive 3D (Plotly, GitHub-Pages friendly - no build step).

Two kinds of input are understood:
  * the loop-closure demo CSVs (`id,x,y,z`) - ground truth / drifted open-VO /
    loop-corrected, the before-vs-after story; and
  * KITTI 3x4 pose files (12 numbers/row, row-major `R|t`) - the estimated and
    ground-truth trajectories of a real stereo-VO run (translation = cols 4/8/12).

KITTI camera coordinates (x right, y down, z forward) are remapped to a
natural display frame (plot_x = x, plot_y = z, plot_z = -y) so "up" is up.
Each estimate trace is rigidly aligned (Umeyama, no scale) to its ground truth
so the overlay matches the reported ATE convention, and the rigid ATE is put in
the trace label.

Usage:
    python3 scripts/build_kitti3d_web.py --output docs/kitti3d/data.json

Override the inputs with --loop-dir / --vo-run as needed. Asset tool, not CI.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loop-dir", type=Path, default=Path("target/kitti_loop_demo"),
                   help="dir with truth.csv / drifted.csv / corrected.csv")
    p.add_argument("--vo-run", type=Path, default=Path("target/kitti_sp_lg_vo_long_revisit_BA/seq00"),
                   help="dir with gt_poses.txt / vo_poses.txt (KITTI 3x4)")
    p.add_argument("--output", type=Path, default=Path("docs/kitti3d/data.json"))
    return p.parse_args()


def remap(x, y, z):
    """KITTI cam (x right, y down, z forward) -> display (x, z, -y) so Z is up."""
    return [x, z, -y]


def load_loop_csv(path: Path):
    xs, ys, zs = [], [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            px, py, pz = remap(float(row["x"]), float(row["y"]), float(row["z"]))
            xs.append(px); ys.append(py); zs.append(pz)
    return xs, ys, zs


def load_kitti_poses(path: Path):
    xs, ys, zs = [], [], []
    with path.open() as fh:
        for line in fh:
            v = line.split()
            if len(v) < 12:
                continue
            # translation = columns 4, 8, 12 (0-based 3, 7, 11)
            px, py, pz = remap(float(v[3]), float(v[7]), float(v[11]))
            xs.append(px); ys.append(py); zs.append(pz)
    return xs, ys, zs


def umeyama_align(src, dst):
    """Rigid (no-scale) align src trajectory onto dst. src/dst = (xs,ys,zs)."""
    try:
        import numpy as np
    except ImportError:
        return src  # alignment is best-effort; raw points still render
    s = np.array(src).T  # (N,3)
    d = np.array(dst).T
    n = min(len(s), len(d))
    s, d = s[:n], d[:n]
    mu_s, mu_d = s.mean(0), d.mean(0)
    sc, dc = s - mu_s, d - mu_d
    u, _, vt = np.linalg.svd(sc.T @ dc / n)
    e = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, e]) @ u.T
    aligned = (rot @ s.T).T + (mu_d - rot @ mu_s)
    return [aligned[:, 0].tolist(), aligned[:, 1].tolist(), aligned[:, 2].tolist()]


def ate(a, b):
    try:
        import numpy as np
    except ImportError:
        return None
    pa, pb = np.array(a).T, np.array(b).T
    n = min(len(pa), len(pb))
    return float(np.sqrt(((pa[:n] - pb[:n]) ** 2).sum(1).mean()))


def trace(name, xyz, color, dash=None):
    t = {"name": name, "x": xyz[0], "y": xyz[1], "z": xyz[2], "color": color}
    if dash:
        t["dash"] = dash
    return t


def main() -> int:
    args = parse_args()
    datasets = []

    # 1) loop-closure before/after (the headline story)
    lt = args.loop_dir / "truth.csv"
    ld = args.loop_dir / "drifted.csv"
    lc = args.loop_dir / "corrected.csv"
    if lt.exists() and ld.exists() and lc.exists():
        truth = load_loop_csv(lt)
        drifted = load_loop_csv(ld)
        corrected = load_loop_csv(lc)
        ad = ate(drifted, truth)
        ac = ate(corrected, truth)
        datasets.append({
            "name": f"KITTI 00 loop closure (drift {ad:.1f} m → {ac:.1f} m)" if ad else "KITTI 00 loop closure",
            "subtitle": "ground-truth-pose loop demo: open VO drifts, the SE(3) pose graph snaps it back",
            "traces": [
                trace("ground truth", truth, "#2b2b2b"),
                trace(f"drifted open VO (ATE {ad:.1f} m)" if ad else "drifted open VO", drifted, "#d23737", "dash"),
                trace(f"loop-corrected (ATE {ac:.1f} m)" if ac else "loop-corrected", corrected, "#1f9d55"),
            ],
        })

    # 2) real stereo SP/LG VO + BA vs ground truth
    gt_p = args.vo_run / "gt_poses.txt"
    vo_p = args.vo_run / "vo_poses.txt"
    if gt_p.exists() and vo_p.exists():
        gt = load_kitti_poses(gt_p)
        vo_raw = load_kitti_poses(vo_p)
        vo = umeyama_align(vo_raw, gt)
        a = ate(vo, gt)
        datasets.append({
            "name": f"KITTI 00 SuperPoint/LightGlue stereo VO + BA{f' (ATE {a:.1f} m)' if a else ''}",
            "subtitle": "real rectified-stereo VO with bundle adjustment, rigidly aligned to ground truth",
            "traces": [
                trace("ground truth", gt, "#2b2b2b"),
                trace(f"VO + BA estimate{f' (ATE {a:.1f} m)' if a else ''}", vo, "#2b6cb0", "dash"),
            ],
        })

    if not datasets:
        print("no inputs found; nothing written")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"datasets": datasets}))
    n_pts = sum(len(t["x"]) for d in datasets for t in d["traces"])
    print(f"wrote {args.output}: {len(datasets)} datasets, {n_pts} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
