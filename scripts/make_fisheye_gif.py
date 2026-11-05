"""Fisheye demo GIF: SfM and 3DGS on raw fisheye frames, no undistortion.

Three panels along the camera path, smoothed between views:
  1. the raw input frame with its SfM features, coloured by their angle off
     the optical axis (rings at 45° and 90°);
  2. the SfM model from above: the points, the path, and the camera drawn as
     a dome of the lens's field of view, the points it sees coloured as in 1;
  3. the splat drawn through the same lens.

usage:
  gsplat_lens_frames_cuda --ply <run>/scene.ply --data <run> --out <frames> \\
      --every 1 --interp 2
  python scripts/make_fisheye_gif.py --frames <frames> --sparse <run>/sparse/0 \\
      --out docs/assets/fisheye_tumvi.gif [--caption "..."]

The model is COLMAP text (cameras.txt, images.txt, points3D.txt). Needs numpy
and Pillow.
"""
import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--frames', required=True, help='gsplat_lens_frames_cuda --out dir')
ap.add_argument('--sparse', required=True, help='the COLMAP text model the splat was trained on')
ap.add_argument('--out', required=True)
ap.add_argument('--panel-width', type=int, default=288)
ap.add_argument('--fps', type=float, default=12.0)
ap.add_argument('--every', type=int, default=1, help='use every n-th frame')
ap.add_argument('--first', type=int, default=0, help='first frame used')
ap.add_argument('--count', type=int, default=0, help='frames used (0: all from --first)')
ap.add_argument('--fov', type=float, default=195.0, help='field of view drawn as the dome, degrees')
ap.add_argument('--elevation', type=float, default=55.0, help='view of panel 2 above the horizon, degrees')
ap.add_argument('--azimuth', type=float, default=0.0, help='view of panel 2 around the vertical, degrees')
ap.add_argument('--swing', type=float, default=20.0, help='panel 2 swings this far either way, degrees')
ap.add_argument('--caption', default='')
ap.add_argument('--font', default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
args = ap.parse_args()


def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def turbo(x):
    """Google's Turbo colormap (polynomial fit), x in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return np.clip(np.stack([r, g, b], -1), 0.0, 1.0)


# --- The SfM model.
def data_lines(path):
    return [l.split() for l in open(path) if l.strip() and not l.startswith('#')]


cam = data_lines(f'{args.sparse}/cameras.txt')[0]
model, width = cam[1], int(cam[2])
params = [float(v) for v in cam[4:]]
points = {int(v[0]): np.array([float(x) for x in v[1:4]]) for v in data_lines(f'{args.sparse}/points3D.txt')}
images = {}
rows = data_lines(f'{args.sparse}/images.txt')
for head, obs in zip(rows[0::2], rows[1::2]):
    R = quat_to_rot(*map(float, head[1:5]))
    t = np.array([float(x) for x in head[5:8]])
    xy = np.array([[float(obs[k]), float(obs[k + 1])] for k in range(0, len(obs), 3)]).reshape(-1, 2)
    ids = [int(obs[k + 2]) for k in range(0, len(obs), 3)]
    seen = [k for k, p in enumerate(ids) if p in points]
    X = np.array([points[ids[k]] for k in seen]).reshape(-1, 3) @ R.T + t
    theta = np.degrees(np.arctan2(np.linalg.norm(X[:, :2], axis=1), X[:, 2]))
    images[head[9]] = (xy[seen], theta, [ids[k] for k in seen])

frames = []
for line in open(f'{args.frames}/views.txt'):
    v = line.rstrip('\n').split(maxsplit=9)
    if len(v) == 10:
        R = quat_to_rot(*map(float, v[2:6]))
        frames.append((int(v[0]), v[1], R, np.array([float(x) for x in v[6:9]]), v[9]))
centres = np.array([-R.T @ t for _, _, R, t, _ in frames])
up = -np.mean([R[1] for _, _, R, _, _ in frames], axis=0)  # the cameras' mean up axis
up /= np.linalg.norm(up)
frames = frames[args.first:args.first + args.count if args.count else None][:: args.every]

# --- Colours: angles off the optical axis in bins, so every colour is a
# palette entry.
THETA_MAX, BINS = 100.0, 24
theta_rgb = [tuple(int(round(255 * c)) for c in turbo(0.12 + 0.88 * (b + 0.5) / BINS)) for b in range(BINS)]


def theta_colour(theta):
    return theta_rgb[min(BINS - 1, int(theta / THETA_MAX * BINS))]


BG, FG, DIM, DOME = (14, 14, 14), (235, 235, 235), (150, 150, 150), (255, 255, 255)
pw = args.panel_width
gap, bar, foot = 6, 28, 28
W, H = 3 * pw + 2 * gap, bar + pw + foot
font = ImageFont.truetype(args.font, 15)
small = ImageFont.truetype(args.font, 12)
titles = ['raw fisheye frame + SfM features', f'SfM: points, path, {args.fov:.0f}° camera',
          '3DGS through the same lens']


def lens_radius(theta):
    """Image radius of a ray `theta` off the axis, for OPENCV_FISHEYE."""
    k = params[4:8]
    d = theta * (1 + k[0] * theta ** 2 + k[1] * theta ** 4 + k[2] * theta ** 6 + k[3] * theta ** 8)
    return 0.5 * (params[0] + params[1]) * d


def dashed_ring(d, cx, cy, r, colour, segs=72):
    for s in range(0, segs, 2):
        a0, a1 = 2 * math.pi * s / segs, 2 * math.pi * (s + 1) / segs
        d.line((cx + r * math.cos(a0), cy + r * math.sin(a0), cx + r * math.cos(a1), cy + r * math.sin(a1)),
               fill=colour, width=1)


# --- Panel 2's view: orthographic, from `elevation` above the horizon, the
# model's long horizontal axis across.
cloud = np.array(list(points.values()))
lo, hi = np.percentile(cloud, 2, axis=0), np.percentile(cloud, 98, axis=0)
cloud = cloud[np.all((cloud >= lo) & (cloud <= hi), axis=1)]
centre = np.median(cloud, axis=0)
flat = (cloud - centre) - np.outer((cloud - centre) @ up, up)
east = np.linalg.eigh(np.cov(flat.T))[1][:, -1]
north = np.cross(up, east)
TRAIL = 24  # frames of the path drawn bright behind the camera


def view_axes(azimuth):
    az, el = math.radians(azimuth), math.radians(args.elevation)
    vz = math.cos(el) * (math.cos(az) * north + math.sin(az) * east) - math.sin(el) * up
    vx = np.cross(vz, up)
    vx /= np.linalg.norm(vx)
    return np.stack([vx, np.cross(vz, vx), vz])


# One scale and offset for the whole swing, so the model does not breathe.
swing = [view_axes(args.azimuth + a) for a in (-args.swing, 0.0, args.swing)]
spans = [np.percentile((cloud - centre) @ VR[:2].T, [2, 98], axis=0) for VR in swing]
scale = min(0.94 * pw / np.max(s[1] - s[0]) for s in spans)
offset = np.mean([s.mean(axis=0) for s in spans], axis=0)


def make_view(azimuth):
    """The projection at `azimuth` and the model's points drawn through it."""
    VR = view_axes(azimuth)

    def view(X):
        Xc = (np.atleast_2d(X) - centre) @ VR.T
        return (Xc[:, :2] - offset) * scale + pw / 2, Xc[:, 2]

    # The points, far to near, dimmer when farther.
    uv, z = view(cloud)
    order = np.argsort(-z)
    shade = np.interp(z[order], [np.percentile(z, 5), np.percentile(z, 95)], [190, 70]).astype(np.uint8) // 2 * 2
    base = np.full((pw, pw, 3), BG, np.uint8)
    for (u, v_), s in zip(uv[order].astype(int), shade):
        if 0 <= u < pw and 0 <= v_ < pw:
            base[v_, u] = s
    return view, Image.fromarray(base)


def dome(view, R, t, radius):
    """Wireframe of the lens's field of view: rim and meridians."""
    C = -R.T @ t
    half = math.radians(args.fov / 2)
    ring = [(half, 2 * math.pi * s / 48) for s in range(49)]
    meridians = [[(half * s / 12, math.pi / 2 * m) for s in range(13)] for m in range(4)]

    def lines(pairs):
        dirs = np.array([[math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a)] for a, b in pairs])
        return view(C + radius * dirs @ R)[0]

    return view(C)[0][0], lines(ring), [lines(m) for m in meridians]


def panel_sfm(R, t, name, at, azimuth):
    view, im = make_view(azimuth)
    d = ImageDraw.Draw(im)
    path_uv = view(centres)[0]
    d.line([tuple(p) for p in path_uv], fill=(48, 48, 48), width=1)
    for k in range(max(0, at - TRAIL), at):
        grey = int(80 + 170 * (k - at + TRAIL) / TRAIL) // 2 * 2
        d.line((tuple(path_uv[k]), tuple(path_uv[k + 1])), fill=(grey, grey, grey), width=2)
    _, theta, ids = images[name]
    puv, _ = view(np.array([points[p] for p in ids]).reshape(-1, 3))
    for (u, v_), th in zip(puv, theta):
        d.rectangle((u - 1, v_ - 1, u + 1, v_ + 1), fill=theta_colour(th))
    c, ring, meridians = dome(view, R, t, 0.11 * pw / scale)
    for m in meridians:
        d.line([tuple(p) for p in m], fill=DOME, width=1)
    d.line([tuple(p) for p in ring], fill=DOME, width=2)
    d.ellipse((c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3), fill=DOME)
    return im


def panel_input(path, name):
    im = Image.open(path).convert('RGB').resize((pw, pw), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    s = pw / width
    if model == 'OPENCV_FISHEYE':
        cx, cy = params[2] * s, params[3] * s
        for deg in (45, 90):
            r = lens_radius(math.radians(deg)) * s
            dashed_ring(d, cx, cy, r, FG)
            a = math.radians(225)
            d.text((cx + r * math.cos(a) + 3, cy + r * math.sin(a) + 3), f'{deg}°', fill=FG, font=small)
    xy, theta, _ = images[name]
    for (x, y), th in zip(xy * s, theta):
        d.rectangle((x - 1, y - 1, x + 1, y + 1), fill=theta_colour(th))
    return im


out = []
for step, (j, name, R, t, path) in enumerate(frames):
    c = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(c)
    c.paste(panel_input(path, name), (0, bar))
    azimuth = args.azimuth + args.swing * math.sin(2 * math.pi * step / len(frames))
    c.paste(panel_sfm(R, t, name, args.first + step * args.every, azimuth), (pw + gap, bar))
    splat = Image.open(f'{args.frames}/splat_{j:04d}.png').convert('RGB').resize((pw, pw), Image.LANCZOS)
    c.paste(splat, (2 * (pw + gap), bar))
    for k, title in enumerate(titles):
        d.text((k * (pw + gap) + 5, 6), title, fill=FG, font=font)
    # The colour key for the angle off the optical axis, and the caption.
    y0 = bar + pw + 8
    d.text((5, y0 - 1), '0°', fill=DIM, font=small)
    for b, rgb in enumerate(theta_rgb):
        d.rectangle((24 + 5 * b, y0, 28 + 5 * b, y0 + 10), fill=rgb)
    d.text((28 + 5 * BINS, y0 - 1), f'{THETA_MAX:.0f}° off the axis', fill=DIM, font=small)
    if args.caption:
        d.text((pw + gap + 5, y0 - 1), args.caption, fill=DIM, font=small)
    out.append(c)

# One palette for every frame, so each frame after the first stores only
# what changed: a grey ramp, the angle colours and white.
colours = [(v, v, v) for v in range(0, 256, 2)] + theta_rgb + [BG, FG, DIM]
colours += [colours[-1]] * (256 - len(colours))
palette = Image.new('P', (1, 1))
palette.putpalette([c for rgb in colours for c in rgb])
out = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in out]
out[0].save(args.out, save_all=True, append_images=out[1:], duration=round(1000 / args.fps), loop=0, optimize=True)
print(f'{len(out)} frames -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)')
