#!/usr/bin/env python3
"""Real 360-degree frames above the splat rendered at the same spots (docs/navigation.md).

    python3 ros2/visloc-ros2/gs/real_vs_sim_360.py --model runs/street/level/sparse/0 \
        --names runs/street/sfm/image_names.txt --images runs/street/sfm/images \
        --splat runs/street/scene.ply --out runs/street/real_vs_sim [--frames 0:240:2]

For a `gsplat_photos --insv` model (each video frame cut into lens faces,
`image_names.txt` mapping the faces to their frames), each chosen frame's real
faces are stitched into an equirectangular panorama by their SfM poses. The
splat is rendered at the same centre with `equirect.py`, facing the walking
direction. The two are stacked on a 1280 x 720 frame with labels. Pixels no
face covers (the camera's pole, the dataset's blurred faces) stay black in the
real one. Writes OUT/NNNNN.jpg.
"""

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from align_sfm_tum import qvec2rot, read_images
from equirect import Panorama

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PW, PH = 1280, 640


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--model', required=True, help='the levelled COLMAP text model of the faces')
    p.add_argument('--names', required=True, help="gsplat_photos' image_names.txt (face -> frame)")
    p.add_argument('--images', required=True, help='the faces (frame_NNNNN.png)')
    p.add_argument('--splat', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', default='0:240:2', help='first:last:step of the video frames to show')
    p.add_argument('--label', default='a 360° video walk\nthrough a town\n(OverMaps360, public)')
    p.add_argument('--server', default=os.path.join(REPO, 'target', 'release', 'examples', 'gsplat_render_server'))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    font = lambda n: ImageFont.load_default(size=n)  # noqa: E731
    images = {v['name']: v for v in read_images(os.path.join(args.model, 'images.txt')).values()}
    frames = {}
    for line in open(args.names, encoding='utf-8'):
        face, rest = line.split(maxsplit=1)
        if face in images:
            frames.setdefault(int(os.path.basename(rest.split()[0])[:6]), []).append(face)
    order = sorted(frames)
    centre = lambda fr: -qvec2rot(images[frames[fr][0]]['q']).T @ images[frames[fr][0]]['t']  # noqa: E731
    lon = (0.5 - (np.arange(PW) + 0.5) / PW) * 2 * math.pi
    lat = (0.5 - (np.arange(PH) + 0.5) / PH) * math.pi
    lon, lat = np.meshgrid(lon, lat)
    dirs = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    cams = {}
    for line in open(os.path.join(args.model, 'cameras.txt'), encoding='utf-8'):
        if line.strip() and not line.startswith('#'):
            c = line.split()
            cams[int(c[0])] = [float(v) for v in c[2:8]]          # w h fx fy cx cy (PINHOLE)
    pano = Panorama(args.server, args.splat, face=640, width=PW)
    first, last, step = (int(v) for v in args.frames.split(':'))
    n = 0
    for fr in order[first:last:step]:
        nxt = order[min(order.index(fr) + 3, len(order) - 1)]
        d = centre(nxt) - centre(fr)
        yaw = math.atan2(d[1], d[0])
        c, s = math.cos(yaw), math.sin(yaw)
        world = dirs @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        real = np.zeros((PH, PW, 3), np.uint8)
        best = np.full((PH, PW), -np.inf)
        for face in frames[fr]:
            v = images[face]
            w, h, fx, fy, cx, cy = cams[v['cam']]
            cam = world @ qvec2rot(v['q']).T
            z = cam[..., 2]
            u = cam[..., 0] / np.maximum(z, 1e-6) * fx + cx
            vv = cam[..., 1] / np.maximum(z, 1e-6) * fy + cy
            m = (z > best) & (z > 0.3) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
            img = np.array(Image.open(os.path.join(args.images, face)).convert('RGB'))
            real[m] = img[vv[m].astype(int), u[m].astype(int)]
            best[m] = z[m]
        sim = pano.render(*centre(fr), yaw)
        out = Image.new('RGB', (1280, 720), (14, 16, 22))
        out.paste(Image.fromarray(real).resize((720, 360)), (0, 0))
        out.paste(Image.fromarray(sim).resize((720, 360)), (0, 360))
        dr = ImageDraw.Draw(out)
        dr.text((750, 60), 'REAL', font=font(64), fill=(255, 255, 255))
        dr.text((750, 140), args.label, font=font(28), fill=(210, 210, 210))
        dr.text((750, 420), 'SIM', font=font(64), fill=(120, 220, 255))
        dr.text((750, 500), "the same spot, rebuilt by\nvisloc-rs's own SfM +\n3D Gaussian Splatting", font=font(28),
                fill=(160, 225, 255))
        out.save(os.path.join(args.out, f'{n:05d}.jpg'), quality=90)
        n += 1
    pano.close()
    print(f'{n} frames in {args.out}')


if __name__ == '__main__':
    main()
