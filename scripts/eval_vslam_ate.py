#!/usr/bin/env python3
"""Fair ATE comparison harness — evaluate visloc-rs and DPVO against the SAME
KITTI GT with the SAME tool (evo), each with the alignment its modality warrants:
SE(3) (rigid) for the metric stereo visloc trajectory, Sim(3) (scale-corrected)
for monocular DPVO. Prints ATE rmse/mean/max (translation) in metres.

Usage:
  eval_ate.py gt_kitti=<poses_00.txt> n=<frames>
              visloc_csv=<vo.csv|ba.csv>     (id,x,y,z ; metric -> SE3 align)
              dpvo_tum=<dpvo_traj.txt>        (TUM ; monocular -> Sim3 align)
"""
import sys
import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface


def kitti_to_traj(path, n):
    """First n lines of a KITTI poses file (3x4 row-major) -> PoseTrajectory3D."""
    rows = np.loadtxt(path)[:n]
    poses = []
    for r in rows:
        T = np.eye(4)
        T[:3, :4] = r.reshape(3, 4)
        poses.append(T)
    stamps = np.arange(len(poses), dtype=float)
    return PoseTrajectory3D(poses_se3=poses, timestamps=stamps)


def csv_to_traj(path):
    """visloc id,x,y,z (positions only) -> PoseTrajectory3D (identity rot)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    poses = []
    for row in data:
        T = np.eye(4)
        T[:3, 3] = row[1:4]
        poses.append(T)
    stamps = data[:, 0].astype(float)
    return PoseTrajectory3D(poses_se3=poses, timestamps=stamps)


def ate(ref, est, correct_scale, label):
    ref_s, est_s = sync.associate_trajectories(ref, est, max_diff=0.5)
    est_s.align(ref_s, correct_scale=correct_scale)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref_s, est_s))
    s = ape.get_all_statistics()
    align = "Sim3(scale-corrected)" if correct_scale else "SE3(rigid)"
    print(f"{label:<12} matched={len(ref_s.timestamps):>4}  ATE[{align}]  "
          f"rmse={s['rmse']:.3f}  mean={s['mean']:.3f}  max={s['max']:.3f}  (m)")


def main():
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n = int(kw["n"])
    ref = kitti_to_traj(kw["gt_kitti"], n)
    if "visloc_csv" in kw:
        ate(ref, csv_to_traj(kw["visloc_csv"]), correct_scale=False, label="visloc-rs")
    if "dpvo_tum" in kw:
        est = file_interface.read_tum_trajectory_file(kw["dpvo_tum"])
        ate(ref, est, correct_scale=False, label="DPVO(SE3)")
        est2 = file_interface.read_tum_trajectory_file(kw["dpvo_tum"])
        ate(ref, est2, correct_scale=True, label="DPVO(Sim3)")


if __name__ == "__main__":
    main()
