"""Differentiation GIF: EuRoC MH_05, COLMAP 4.1 (GPU) vs visloc-rs, then the
3DGS flythrough from visloc-rs poses.

usage:
  python scripts/make_euroc_vs_colmap_gif.py --euroc-root <EuRoC> \\
      --bench-out <benchmark out-root> --run <gsplat_euroc --work dir> \\
      --out docs/assets/euroc_mh05_vs_colmap.gif --colmap-time 764 --ours-time 102

The COLMAP model comes from scripts/benchmark_euroc_vs_colmap.py; the
visloc-rs run is `gsplat_euroc ... --steps 7000 --render-dir <run>/renders`.
"""
import csv
import importlib.util
import io
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import argparse

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--seq', default='MH_05_difficult')
ap.add_argument('--euroc-root', required=True, help='EuRoC root with <seq>/mav0')
ap.add_argument('--bench-out', required=True,
                help='--out-root of scripts/benchmark_euroc_vs_colmap.py (has colmap<tag>_<seq>/)')
ap.add_argument('--colmap-tag', default='_final')
ap.add_argument('--run', required=True,
                help='gsplat_euroc --work dir (trajectory_vs_gt.csv, images/, renders/ from --render-dir)')
ap.add_argument('--out', required=True, help='output GIF path')
ap.add_argument('--colmap-time', type=float, required=True, help='COLMAP extract+match+map seconds')
ap.add_argument('--ours-time', type=float, required=True, help='visloc-rs SfM seconds')
ap.add_argument('--fly-start', type=float, default=0.15,
                help='fraction of frames skipped before the flythrough starts')
args = ap.parse_args()
SEQ, RUN, OUT = args.seq, args.run, args.out
COLMAP_TIME_S, OURS_TIME_S = args.colmap_time, args.ours_time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('bench', os.path.join(HERE, 'benchmark_euroc_vs_colmap.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.EUROC = args.euroc_root


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    sg = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sg[2, 2] = -1
    r = u @ sg @ vt
    c = np.trace(np.diag(sig) @ sg) / (s ** 2).sum(1).mean()
    return lambda x: (c * (r @ x.T)).T + md - c * r @ ms


def aligned(frames, est, gt):
    f = umeyama(np.array(est), np.array(gt))
    a = f(np.array(est))
    err = np.linalg.norm(a - np.array(gt), axis=1)
    return np.array(frames), a, np.array(gt), float(np.sqrt((err ** 2).mean()))


# COLMAP: largest model, camera centres by frame.
gt_all = bench.gt_centres(SEQ)
col = bench.read_colmap_images_bin(os.path.join(args.bench_out, f'colmap{args.colmap_tag}_{SEQ}', 'sparse', '0', 'images.bin'))
cf, ce, cg = [], [], []
for name, c in sorted(col.items()):
    i = int(name.split('_')[1].split('.')[0])
    if gt_all[i] is not None:
        cf.append(i)
        ce.append(c)
        cg.append(gt_all[i])
cf, ca, cgt, c_ate = aligned(cf, ce, cg)

rows = list(csv.DictReader(open(os.path.join(RUN, 'trajectory_vs_gt.csv'))))
of = [int(r['frame']) for r in rows]
oe = [[float(r['est_x']), float(r['est_y']), float(r['est_z'])] for r in rows]
og = [[float(r['gt_x']), float(r['gt_y']), float(r['gt_z'])] for r in rows]
of, oa, ogt, o_ate = aligned(of, oe, og)
print(f'COLMAP ATE {100 * c_ate:.2f} cm ({len(cf)} frames), visloc-rs ATE {100 * o_ate:.2f} cm ({len(of)} frames)')

gt_path = np.array([g for g in gt_all if g is not None])
lo = gt_path[:, :2].min(0) - 1.0
hi = gt_path[:, :2].max(0) + 1.0

W, H = 800, 416
BG = (18, 20, 26)


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def trajectory_frame(k):
    """k in [0, 1]: fraction of each trajectory drawn."""
    fig, axes = plt.subplots(1, 2, figsize=(W / 100, H / 100), facecolor='#12141a')
    panels = [
        (axes[0], ca, cf, '#ff5a5f', f'COLMAP 4.1 (GPU)\n{COLMAP_TIME_S:.0f} s  ·  ATE {100 * c_ate:.1f} cm'),
        (axes[1], oa, of, '#35d0ba', f'visloc-rs (GPU)\n{OURS_TIME_S:.0f} s  ·  ATE {100 * o_ate:.2f} cm'),
    ]
    for ax, traj, frames, color, title in panels:
        ax.set_facecolor('#12141a')
        ax.plot(gt_path[:, 0], gt_path[:, 1], color='#5b6170', lw=2.5, label='ground truth')
        n = max(2, int(round(k * len(traj))))
        ax.plot(traj[:n, 0], traj[:n, 1], color=color, lw=1.8, label='estimate')
        ax.scatter(traj[n - 1, 0], traj[n - 1, 1], color=color, s=36, zorder=5)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#2a2e38')
        ax.set_title(title, color='white', fontsize=13, pad=8)
        ax.legend(loc='lower right', fontsize=8, facecolor='#1c1f27', edgecolor='#2a2e38', labelcolor='white')
    fig.suptitle(f'EuRoC {SEQ}: same 200 frames, same GPU', color='#c8ccd6', fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig_to_img(fig)


def font(size):
    for f in ('arialbd.ttf', 'arial.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def flythrough_frame(name):
    src = Image.open(os.path.join(RUN, 'images', name)).convert('RGB')
    ren = Image.open(os.path.join(RUN, 'renders', 'render_' + name)).convert('RGB')
    tw = (W - 30) // 2
    th = int(src.height * tw / src.width)
    canvas = Image.new('RGB', (W, H), BG)
    y0 = (H - th) // 2 + 14
    canvas.paste(src.resize((tw, th)), (10, y0))
    canvas.paste(ren.resize((tw, th)), (20 + tw, y0))
    d = ImageDraw.Draw(canvas)
    d.text((10, y0 - 30), 'input frame (EuRoC cam0)', fill=(200, 204, 214), font=font(15))
    d.text((20 + tw, y0 - 30), '3D Gaussian Splatting render', fill=(53, 208, 186), font=font(15))
    d.text((10, 14), 'visloc-rs: raw frames -> GPU SfM -> 3D Gaussian Splatting (pure Rust + wgpu, no COLMAP)',
           fill=(255, 255, 255), font=font(15))
    return canvas


frames, durations = [], []
steps = 40
for i in range(steps + 1):
    frames.append(trajectory_frame(i / steps))
    durations.append(70)
durations[-1] = 1800
names = sorted(os.listdir(os.path.join(RUN, 'renders')))
names = [n[len('render_'):] for n in names if n.startswith('render_')]
for n in names[int(len(names) * args.fly_start)::3]:
    frames.append(flythrough_frame(n))
    durations.append(90)
durations[-1] = 1200
frames = [f.convert('P', palette=Image.ADAPTIVE, colors=128) for f in frames]
frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
print('wrote', OUT, len(frames), 'frames', os.path.getsize(OUT) // 1024, 'KiB')
