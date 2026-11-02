"""Fisheye demo GIF: a splat trained on raw fisheye frames, drawn through the lens.

Two panels along the camera path, smoothed between views: the splat and its
depth (coloured by inverse depth, one range for every frame), both through
the dataset's lens, with the nearest input frame inset in the first.

usage:
  gsplat_lens_frames_cuda --ply <run>/scene.ply --data <run> --out <frames> \\
      --every 2 --interp 4
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
ap.add_argument('--panel-width', type=int, default=400)
ap.add_argument('--inset', type=float, default=0.32, help='inset width as a fraction of a panel')
ap.add_argument('--fps', type=float, default=15.0)
ap.add_argument('--every', type=int, default=1, help='use every n-th frame')
ap.add_argument('--first', type=int, default=0, help='first frame used')
ap.add_argument('--count', type=int, default=0, help='frames used (0: all from --first)')
ap.add_argument('--greys', type=int, default=64, help='grey levels of the shared palette')
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


frames_list = [l.rstrip('\n').split(maxsplit=2) for l in open(f'{F}/views.txt') if l.strip()]
frames_list = frames_list[args.first:args.first + args.count if args.count else None][:: args.every]

# One inverse-depth range for every frame, so the colours hold still.
sample = np.concatenate([read_depth(f'{F}/depth_{int(j):04d}.bin')[::4, ::4].ravel() for j, _, _ in frames_list])
sample = 1.0 / sample[sample > 0]
lo, hi = np.percentile(sample, 1), np.percentile(sample, 99)


def depth_image(path):
    d = read_depth(path)
    x = (1.0 / np.maximum(d, 1e-6) - lo) / (hi - lo)
    rgb = turbo(x) * (d > 0)[..., None]
    return Image.fromarray((rgb * 255).astype(np.uint8))


pw = args.panel_width
gap, bar = 6, 30
cap = 26 if args.caption else 0
W, H = 2 * pw + gap, bar + pw + cap
font = ImageFont.truetype(args.font, 16)
small = ImageFont.truetype(args.font, 13)
BG, FG, DIM = (14, 14, 14), (235, 235, 235), (150, 150, 150)
iw = round(pw * args.inset)
titles = ['3DGS through the 195° fisheye', 'its depth, same lens']

frames = []
for j, name, path in frames_list:
    c = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(c)
    splat = Image.open(f'{F}/splat_{int(j):04d}.png').convert('RGB').resize((pw, pw), Image.LANCZOS)
    depth = depth_image(f'{F}/depth_{int(j):04d}.bin').resize((pw, pw), Image.LANCZOS)
    c.paste(splat, (0, bar))
    c.paste(depth, (pw + gap, bar))
    # The nearest input frame, inset at the splat's lower right.
    x0, y0 = pw - iw - 8, bar + pw - iw - 8
    d.rectangle((x0 - 2, y0 - 2, x0 + iw + 1, y0 + iw + 1), fill=FG)
    c.paste(Image.open(path).convert('RGB').resize((iw, iw), Image.LANCZOS), (x0, y0))
    d.text((x0 + 4, y0 + 2), 'input', fill=FG, font=small)
    for k, t in enumerate(titles):
        d.text((k * (pw + gap) + 6, 7), t, fill=FG, font=font)
    if args.caption:
        d.text((6, bar + pw + 6), args.caption, fill=DIM, font=small)
    frames.append(c)

# One palette for every frame, so each frame after the first stores only
# what changed: a grey ramp for the grey frames and text, and the depth
# panels' colours (from a strip of them).
strip = Image.new('RGB', (pw, pw * 12))
for i, f in enumerate(frames[:: max(1, len(frames) // 12)][:12]):
    strip.paste(f.crop((pw + gap, bar, 2 * pw + gap, bar + pw)), (0, i * pw))
colours = [(v, v, v) for v in np.linspace(0, 255, args.greys).round().astype(int)]
depth_pal = strip.quantize(colors=256 - len(colours), method=Image.Quantize.MEDIANCUT).getpalette()
colours += [tuple(depth_pal[3 * i:3 * i + 3]) for i in range(256 - len(colours))]
palette = Image.new('P', (1, 1))
palette.putpalette([c for rgb in colours for c in rgb])
frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=round(1000 / args.fps), loop=0,
               optimize=True)
print(f'{len(frames)} frames -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)')
