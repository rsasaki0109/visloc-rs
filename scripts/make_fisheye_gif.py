"""Fisheye demo GIF: a splat trained on raw fisheye frames, drawn through the lens.

Two parts, three panels each (the input frame, the splat, its depth):

1. The sweep: one view drawn through a lens swept from a pinhole to the
   view's own fisheye, next to the input frame it ends at.
2. The views: every rendered view through its own lens, next to its input
   frame.

usage:
  gsplat_lens_frames_cuda --ply <run>/scene.ply --data <run> --out <frames> \\
      --every 4 --sweep <image name>
  python scripts/make_fisheye_gif.py --frames <frames> --out docs/assets/fisheye_tumvi.gif \\
      [--caption "..."]

Needs numpy and Pillow.
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--frames', required=True, help='gsplat_lens_frames_cuda --out dir')
ap.add_argument('--out', required=True)
ap.add_argument('--panel-width', type=int, default=300)
ap.add_argument('--fps', type=float, default=10.0)
ap.add_argument('--hold', type=int, default=10, help='frames held at each end of the sweep')
ap.add_argument('--every', type=int, default=1, help='use every n-th rendered view')
ap.add_argument('--greys', type=int, default=96, help='grey levels of the shared palette')
ap.add_argument('--caption', default='')
ap.add_argument('--font', default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
args = ap.parse_args()
F = args.frames


def read_depth(path):
    raw = np.fromfile(path, dtype='<u4', count=2)
    w, h = int(raw[0]), int(raw[1])
    return np.fromfile(path, dtype='<f4', offset=8).reshape(h, w)


def turbo(x):
    """Google's Turbo colormap (polynomial fit), x in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return np.clip(np.stack([r, g, b], -1), 0.0, 1.0)


views = [l.rstrip('\n').split(maxsplit=2) for l in open(f'{F}/views.txt') if l.strip()][:: args.every]
sweep, sweep_input = [], None
if os.path.exists(f'{F}/sweep.txt'):
    for l in open(f'{F}/sweep.txt'):
        if l.startswith('#'):
            sweep_input = l.rstrip('\n').split(maxsplit=2)[2]
        elif l.strip():
            sweep.append(l.split())

# One inverse-depth range for every frame, so the colours hold still.
depth_files = [f'{F}/depth_{int(k):04d}.bin' for k, _, _ in views]
depth_files += [f'{F}/sweep_depth_{int(j):04d}.bin' for j, _ in sweep]
sample = np.concatenate([read_depth(p)[::4, ::4].ravel() for p in depth_files])
sample = 1.0 / sample[sample > 0]
lo, hi = np.percentile(sample, 1), np.percentile(sample, 99)


def depth_image(path):
    d = read_depth(path)
    x = (1.0 / np.maximum(d, 1e-6) - lo) / (hi - lo)
    rgb = turbo(x) * (d > 0)[..., None]
    return Image.fromarray((rgb * 255).astype(np.uint8))


pw = args.panel_width
gap, bar = 4, 28
cap = 26 if args.caption else 0
W, H = 3 * pw + 2 * gap, bar + pw + cap
font = ImageFont.truetype(args.font, 15)
small = ImageFont.truetype(args.font, 13)
big = ImageFont.truetype(args.font.replace('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'), 20)
BG, FG, DIM, BADGE = (14, 14, 14), (235, 235, 235), (150, 150, 150), (255, 214, 90)


def canvas(panels, titles, badge=None):
    c = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(c)
    for k, (img, t) in enumerate(zip(panels, titles)):
        x = k * (pw + gap)
        c.paste(img.convert('RGB').resize((pw, pw), Image.LANCZOS), (x, bar))
        d.text((x + 6, 6), t, fill=FG, font=font)
    if badge:
        x0, y0 = pw + gap + 10, bar + pw - 40
        box = d.textbbox((x0 + 10, y0 + 4), badge, font=big)
        d.rounded_rectangle((x0, y0, box[2] + 10, box[3] + 6), 8, fill=(0, 0, 0))
        d.text((x0 + 10, y0 + 4), badge, fill=BADGE, font=big)
    if args.caption:
        d.text((6, bar + pw + 6), args.caption, fill=DIM, font=small)
    return c


frames, durations = [], []
step = round(1000 / args.fps)
if sweep:
    raw = Image.open(sweep_input)
    for idx, (j, fov) in enumerate(sweep):
        fov = float(fov)
        kind = 'pinhole' if idx == 0 else 'fisheye' if idx == len(sweep) - 1 else 'lens'
        panels = [raw, Image.open(f'{F}/sweep_splat_{int(j):04d}.png'), depth_image(f'{F}/sweep_depth_{int(j):04d}.bin')]
        titles = ['input fisheye frame', '3DGS, lens swept (width FOV)', 'its depth, same lens']
        frames.append(canvas(panels, titles, badge=f'{fov:.0f}° {kind}'))
        durations.append(step * (args.hold if idx in (0, len(sweep) - 1) else 1))
for k, name, path in views:
    panels = [Image.open(path), Image.open(f'{F}/splat_{int(k):04d}.png'), depth_image(f'{F}/depth_{int(k):04d}.bin')]
    titles = ['input fisheye frame', '3DGS through the lens', 'its depth through the lens']
    frames.append(canvas(panels, titles))
    durations.append(step)
# One palette for every frame, so each frame after the first stores only
# what changed: a grey ramp for the grey frames and text, the badge's
# colour, and the depth panels' colours (from a strip of them).
depth_x = 2 * (pw + gap)
strip = Image.new('RGB', (pw, pw * 12))
for i, f in enumerate(frames[:: max(1, len(frames) // 12)][:12]):
    strip.paste(f.crop((depth_x, bar, depth_x + pw, bar + pw)), (0, i * pw))
colours = [(v, v, v) for v in np.linspace(0, 255, args.greys).round().astype(int)]
colours.append(BADGE)
depth_pal = strip.quantize(colors=256 - len(colours), method=Image.Quantize.MEDIANCUT).getpalette()
colours += [tuple(depth_pal[3 * i:3 * i + 3]) for i in range(256 - len(colours))]
palette = Image.new('P', (1, 1))
palette.putpalette([c for rgb in colours for c in rgb])
frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
print(f'{len(frames)} frames -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)')
