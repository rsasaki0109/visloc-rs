#!/usr/bin/env python3
"""Localize real held-out photos, splat renders and another walk against a map of real photos (docs/navigation.md).

    python3 ros2/visloc-ros2/gs/real_map_test.py --run runs/walkA --out runs/walkA/realmap \
        [--query-run runs/walkB] [--map-args "--smoothing-sigma 1.5"] [--loc-args "..."]

`--run` is a `real2sim.py` output directory (`sfm/`, `level/`, `scene.ply`).
Its horizontal lens faces are split by video frame: the even frames build a
localization map from the real photos (`tum_rgbd_save_map --keep-world`,
the depth rendered from the splat at each photo's SfM pose, since the camera
has none); the odd frames are held out. Three query sets go through
`basalt_euroc_localize_demo` (one image at a time, no odometry):

- `real`: the held-out real photos, truth their SfM camera centres;
- `render`: the splat rendered at the same poses, the images a simulator
  shows the localizer (how big the render-to-real gap is);
- with `--query-run`, `other`: the horizontal faces of another walk of the
  same street (its own `gsplat_photos --insv` output). It has its own SfM
  frame, so its localized centres are fitted to its SfM centres by a
  similarity (RANSAC over the localized images), and the residuals after
  the fit are the error.

Prints per set the localized share and the centre error, and writes
OUT/summary.json.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real2sim import EX, FACE, HORIZONTAL, face_frames, run  # noqa: E402
from align_sfm_tum import qvec2rot, rot2qvec  # noqa: E402


def horizontal_faces(names, model, images_dir):
    """[(frame, face index, image path, R_cw, t_cw)] of the horizontal faces."""
    out = []
    for fr, faces in sorted(face_frames(names, model).items()):
        for face, v in faces:
            if face in HORIZONTAL:
                out.append((fr, HORIZONTAL.index(face), v['name'], qvec2rot(v['q']), np.asarray(v['t'], float)))
    return [(fr, k, os.path.join(images_dir, name), R, t) for fr, k, name, R, t in out]


def stamp(fr, k):
    return int(round((fr * 0.333677 + k * 0.01) * 1e9))


def render(server, splat, poses, depth):
    """RGB (and depth) of the splat at each (R_cw, t_cw)."""
    cmd = [server, '--ply', splat, '--camera', FACE] + (['--depth'] if depth else [])
    srv = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    n = 640 * 640
    for R, t in poses:
        srv.stdin.write((' '.join(f'{v:.7f}' for v in [*R.reshape(-1), *t]) + '\n').encode())
        srv.stdin.flush()
        buf = srv.stdout.read(n * (7 if depth else 3))
        rgb = np.frombuffer(buf[:n * 3], np.uint8).reshape(640, 640, 3)
        yield rgb, (np.frombuffer(buf[n * 3:], '<f4').reshape(640, 640) if depth else None)
    srv.stdin.close()
    srv.wait()


def write_euroc(out, items):
    """items: [(t_ns, PIL image)] -> OUT/mav0/cam0/{data/, data.csv}."""
    d = os.path.join(out, 'mav0', 'cam0', 'data')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(out, 'mav0', 'cam0', 'data.csv'), 'w', encoding='utf-8') as f:
        f.write('#timestamp [ns],filename\n')
        for ns, img in items:
            img.convert('L').save(os.path.join(d, f'{ns}.png'))
            f.write(f'{ns},{ns}.png\n')


def centres(path):
    """{t_ns: camera centre} of a TUM file."""
    rows = np.loadtxt(path, ndmin=2) if os.path.getsize(path) else np.zeros((0, 8))
    return {int(round(r[0] * 1e9)): r[1:4] for r in rows}


def umeyama(a, b):
    """s, R, t with b ~ s R a + t."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def fit_errors(src, dst, iters=500, thresh=1.0, seed=0):
    """Residuals of dst after a RANSAC similarity fit from src, and the inlier share."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(src), 3, replace=False)
        s, R, t = umeyama(src[idx], dst[idx])
        inl = np.linalg.norm(dst - (s * src @ R.T + t), axis=1) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    s, R, t = umeyama(src[best], dst[best])
    return np.linalg.norm(dst - (s * src @ R.T + t), axis=1), best.mean(), s


def stats(e):
    if len(e) == 0:
        return {}
    return {'median_cm': round(float(np.median(e)) * 100, 1), 'p90_cm': round(float(np.percentile(e, 90)) * 100, 1),
            'within_50cm': round(float((e < 0.5).mean()), 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', required=True, help='a real2sim.py output directory')
    p.add_argument('--out', required=True)
    p.add_argument('--query-run', help='another walk: a directory with gsplat_photos --insv output in sfm/')
    p.add_argument('--map-args', default='', help='extra tum_rgbd_save_map arguments')
    p.add_argument('--loc-args', default='--max-hamming 90 --reprojection-px 5 --min-inliers 20',
                   help='basalt_euroc_localize_demo arguments')
    p.add_argument('--map-tag', default='', help='suffix of the map directory (one per --map-args)')
    p.add_argument('--tag', default='', help='suffix of the result directories')
    args = p.parse_args()
    from PIL import Image  # noqa: PLC0415
    os.makedirs(args.out, exist_ok=True)
    server = os.path.join(EX, 'gsplat_render_server')
    splat = os.path.join(args.run, 'scene.ply')
    faces = horizontal_faces(os.path.join(args.run, 'sfm', 'image_names.txt'),
                             os.path.join(args.run, 'level', 'sparse', '0'), os.path.join(args.run, 'sfm', 'images'))
    mapping = [f for f in faces if f[0] % 2 == 0]
    held = [f for f in faces if f[0] % 2 == 1]

    def tum_rows(items):
        rows = []
        for fr, k, _, R, t in items:
            c = -R.T @ t
            q = rot2qvec(R.T)
            rows.append(f'{stamp(fr, k) * 1e-9:.6f} {c[0]:.5f} {c[1]:.5f} {c[2]:.5f} '
                        f'{q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {q[0]:.8f}\n')
        return rows

    seq = os.path.join(args.out, 'map_seq')
    traj = os.path.join(args.out, 'map.tum')
    if not os.path.exists(os.path.join(seq, 'depth.txt')):
        os.makedirs(os.path.join(seq, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(seq, 'depth'), exist_ok=True)
        rgb_list, depth_list = [], []
        for (fr, k, path, R, t), (_, depth) in zip(mapping, render(server, splat, [(f[3], f[4]) for f in mapping],
                                                                     True)):
            ts = f'{stamp(fr, k) * 1e-9:.6f}'
            Image.open(path).convert('RGB').save(os.path.join(seq, 'rgb', ts + '.png'))
            d16 = np.where((depth > 0) & (depth < 25), depth * 5000, 0).astype(np.uint16)
            Image.fromarray(d16).save(os.path.join(seq, 'depth', ts + '.png'))
            rgb_list.append(f'{ts} rgb/{ts}.png\n')
            depth_list.append(f'{ts} depth/{ts}.png\n')
        for name, lines in (('rgb.txt', rgb_list), ('depth.txt', depth_list)):
            with open(os.path.join(seq, name), 'w', encoding='utf-8') as f:
                f.write('# real faces, splat depth\n# timestamp filename\n# \n')
                f.writelines(lines)
        with open(traj, 'w', encoding='utf-8') as f:
            f.writelines(tum_rows(mapping))
    mapd = os.path.join(args.out, 'map' + args.map_tag)
    if not os.path.exists(os.path.join(mapd, 'map.json')):
        run([os.path.join(EX, 'tum_rgbd_save_map'), '--seq-dir', seq, '--trajectory', traj, '--out', mapd,
             '--fx', 320, '--fy', 320, '--cx', 320, '--cy', 320, '--k1', 0, '--k2', 0, '--p1', 0, '--p2', 0, '--k3', 0,
             '--match-radius-px', 12, '--max-depth', 20, '--keyframe-m', 0.3, '--keyframe-deg', 10,
             '--max-hamming', 70, '--keep-world', *shlex.split(args.map_args)],
            log=os.path.join(args.out, f'map{args.map_tag}.log'))

    truth = os.path.join(args.out, 'held.tum')
    with open(truth, 'w', encoding='utf-8') as f:
        f.writelines(tum_rows(held))
    sets = {}
    q_real = os.path.join(args.out, 'q_real')
    if not os.path.exists(os.path.join(q_real, 'mav0')):
        write_euroc(q_real, [(stamp(fr, k), Image.open(path)) for fr, k, path, _, _ in held])
    sets['real'] = q_real
    q_render = os.path.join(args.out, 'q_render')
    if not os.path.exists(os.path.join(q_render, 'mav0')):
        write_euroc(q_render, [(stamp(f[0], f[1]), Image.fromarray(rgb)) for f, (rgb, _) in
                               zip(held, render(server, splat, [(f[3], f[4]) for f in held], False))])
    sets['render'] = q_render
    other_truth = None
    if args.query_run:
        sfm = os.path.join(args.query_run, 'sfm')
        other = horizontal_faces(os.path.join(sfm, 'image_names.txt'), os.path.join(sfm, 'sparse', '0'),
                                 os.path.join(sfm, 'images'))
        other_truth = {stamp(fr, k): -R.T @ t for fr, k, _, R, t in other}
        q_other = os.path.join(args.out, 'q_other')
        if not os.path.exists(os.path.join(q_other, 'mav0')):
            write_euroc(q_other, [(stamp(fr, k), Image.open(path)) for fr, k, path, _, _ in other])
        sets['other'] = q_other

    T = centres(truth)
    summary = {}
    for name, d in sets.items():
        res = os.path.join(args.out, f'loc_{name}{args.tag}')
        run([os.path.join(EX, 'basalt_euroc_localize_demo'), '--map', mapd, '--euroc-dir', d, '--out-dir', res,
             *shlex.split(args.loc_args)], log=res + '.log')
        got = centres(os.path.join(res, 'localization.tum'))
        n = sum(1 for _ in open(os.path.join(d, 'mav0', 'cam0', 'data.csv'))) - 1
        if name == 'other':
            ks = [k for k in got if k in other_truth]
            if len(ks) >= 3:
                e, share, scale = fit_errors(np.array([other_truth[k] for k in ks]), np.array([got[k] for k in ks]))
            else:
                e, share, scale = np.zeros(0), 0.0, float('nan')
            summary[name] = {'images': n, 'localized': len(got), 'fit_inliers': round(float(share), 3),
                             'fit_scale': round(float(scale), 3), **stats(e)}
        else:
            e = np.array([np.linalg.norm(got[k] - T[k]) for k in got if k in T])
            summary[name] = {'images': n, 'localized': len(got), **stats(e)}
        print(name, summary[name], flush=True)
    json.dump(summary, open(os.path.join(args.out, f'summary{args.tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
