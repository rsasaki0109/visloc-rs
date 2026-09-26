"""EuRoC SfM head-to-head: visloc-rs (gsplat_euroc SfM, GPU) vs COLMAP (CUDA).

Both see the SAME undistorted pinhole frames (stride 4, first N) that
gsplat_euroc writes to <work>/images, with the same fixed intrinsics.
Metric: time, registered frames, Sim(3) ATE RMSE of cam0 centres vs EuRoC
ground truth (nearest GT sample within 10 ms, c = p_WB + R_WB t_BS).

usage:
  python scripts/benchmark_euroc_vs_colmap.py <seq> --euroc-root DIR       --colmap PATH --ours-exe PATH --out-root DIR       [--ours-only | --colmap-only] [--tag T] [--ours-args "..."]

Appends one JSON line per sequence to <out-root>/results<tag>.jsonl.
Configuration and results: docs/euroc_gpu_sfm_vs_colmap.md.
"""
import csv
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


EUROC = _arg('--euroc-root', '')
ROOT = _arg('--out-root', '.')
COLMAP = _arg('--colmap', 'colmap')
OURS = _arg('--ours-exe', 'gsplat_euroc')
K = (458.654, 457.296, 367.215, 248.375)
STRIDE, NFRAMES = 4, 200


def quat_to_rot(w, x, y, z):
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def gt_centres(seq):
    """Per selected frame index -> GT cam0 centre (or None)."""
    d = os.path.join(EUROC, seq, 'mav0')
    stamps = [int(r[0]) for r in csv.reader(open(os.path.join(d, 'cam0', 'data.csv'))) if r and not r[0].startswith('#')]
    stamps = stamps[::STRIDE][:NFRAMES]
    yaml = open(os.path.join(d, 'cam0', 'sensor.yaml')).read()
    nums = [float(v) for v in re.findall(r'[-+0-9.eE]+', yaml.split('T_BS')[1].split('data:')[1].split(']')[0])]
    t_bs = np.array([nums[3], nums[7], nums[11]])
    gt = np.array([[float(v) for v in r[:8]] for r in csv.reader(open(os.path.join(d, 'state_groundtruth_estimate0', 'data.csv'))) if r and not r[0].startswith('#')])
    out = []
    for ts in stamps:
        k = np.searchsorted(gt[:, 0], ts)
        best = min((j for j in (k - 1, k) if 0 <= j < len(gt)), key=lambda j: abs(gt[j, 0] - ts))
        if abs(gt[best, 0] - ts) > 10e6:
            out.append(None)
            continue
        p = gt[best, 1:4]
        r = quat_to_rot(*gt[best, 4:8])
        out.append(p + r @ t_bs)
    return out


def sim3_ate(est, gt):
    a, b = np.asarray(est), np.asarray(gt)
    ma, mb = a.mean(0), b.mean(0)
    sa, sb = a - ma, b - mb
    u, sig, vt = np.linalg.svd(sb.T @ sa / len(a))
    s = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s[2, 2] = -1
    r = u @ s @ vt
    scale = np.trace(np.diag(sig) @ s) / (sa ** 2).sum(1).mean()
    t = mb - scale * r @ ma
    err = np.linalg.norm((scale * (r @ a.T)).T + t - b, axis=1)
    return float(np.sqrt((err ** 2).mean()))


def read_colmap_images_bin(path):
    """name -> camera centre."""
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(4)
            q = struct.unpack('<4d', f.read(32))
            t = np.array(struct.unpack('<3d', f.read(24)))
            f.read(4)
            name = b''
            while (c := f.read(1)) != b'\0':
                name += c
            npts = struct.unpack('<Q', f.read(8))[0]
            f.read(24 * npts)
            out[name.decode()] = -quat_to_rot(*q).T @ t
    return out


def run_ours(seq, tag, extra=()):
    work = os.path.join(ROOT, f'ours{tag}_{seq}')
    t0 = time.time()
    log = open(work + '.log', 'w')
    rc = subprocess.call([OURS, '--euroc', os.path.join(EUROC, seq), '--work', work,
                          '--stride', str(STRIDE), '--max-frames', str(NFRAMES), '--steps', '0',
                          '--gpu-sift', '--gpu-ba', *extra], stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    text = open(work + '.log').read()
    m = re.search(r'sfm done in ([0-9.]+)s: (\d+)/(\d+) frames registered.*ATE ([0-9.]+) cm', text)
    return {'rc': rc, 'wall_s': wall, 'sfm_s': float(m[1]) if m else None,
            'registered': int(m[2]) if m else 0, 'ate_cm': float(m[4]) if m else None}


def run_colmap(seq, tag, images):
    work = os.path.join(ROOT, f'colmap{tag}_{seq}')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, 'sparse'))
    db = os.path.join(work, 'db.db')
    log = open(os.path.join(work, 'log.txt'), 'w')
    steps = [
        [COLMAP, 'feature_extractor', '--database_path', db, '--image_path', images,
         '--ImageReader.camera_model', 'PINHOLE', '--ImageReader.single_camera', '1',
         '--ImageReader.camera_params', ','.join(map(str, K)), '--FeatureExtraction.use_gpu', '1'],
        [COLMAP, 'sequential_matcher', '--database_path', db, '--FeatureMatching.use_gpu', '1'],
        [COLMAP, 'mapper', '--database_path', db, '--image_path', images,
         '--output_path', os.path.join(work, 'sparse'),
         '--Mapper.ba_refine_focal_length', '0', '--Mapper.ba_refine_principal_point', '0',
         '--Mapper.ba_refine_extra_params', '0'],
    ]
    times = []
    t0 = time.time()
    for cmd in steps:
        s = time.time()
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        times.append(time.time() - s)
        if rc != 0:
            return {'rc': rc, 'wall_s': time.time() - t0, 'registered': 0, 'ate_cm': None, 'step_s': times}
    wall = time.time() - t0
    models = [os.path.join(work, 'sparse', d) for d in os.listdir(os.path.join(work, 'sparse'))]
    best = None
    for m in models:
        cams = read_colmap_images_bin(os.path.join(m, 'images.bin'))
        if best is None or len(cams) > len(best):
            best = cams
    gt = gt_centres(seq)
    est, ref = [], []
    for name, c in (best or {}).items():
        i = int(name.split('_')[1].split('.')[0])
        if gt[i] is not None:
            est.append(c)
            ref.append(gt[i])
    return {'rc': 0, 'wall_s': wall, 'step_s': times, 'models': len(models),
            'registered': len(best or {}), 'ate_cm': 100 * sim3_ate(est, ref) if len(est) >= 3 else None}


def main():
    seq = sys.argv[1]
    tag = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else ''
    res = {'seq': seq}
    if '--colmap-only' not in sys.argv:
        extra = sys.argv[sys.argv.index('--ours-args') + 1].split() if '--ours-args' in sys.argv else []
        res['ours'] = run_ours(seq, tag, extra)
        res['ours_args'] = extra
    images = os.path.join(ROOT, f'ours{tag}_{seq}', 'images')
    if '--ours-only' not in sys.argv:
        res['colmap'] = run_colmap(seq, tag, images)
    print(json.dumps(res))
    with open(os.path.join(ROOT, f'results{tag}.jsonl'), 'a') as f:
        f.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
