#!/usr/bin/env python3
"""Top-down figure of Nav2 drives in the simulated apartment (docs/navigation.md).

    python3 scripts/plot_nav_sim.py --map runs/nav_sim/map/occupancy.yaml \
        --run "visloc=runs/nav_sim/visloc.csv" --run "odometry=runs/nav_sim/odom.csv" \
        --out docs/assets/nav_sim.png

One panel per run, over the occupancy grid Nav2 planned on: the true path
(solid) and the pose the robot believed (thin), the goals (circles), and
an obstacle missing from the map, per run (`--run TITLE=CSV=x0,y0,x1,y1`). The
CSVs are `visloc_nav_sim_node --record` files, or `gz/nav_goals.py --record`
files with `--world-from-map` (their poses, the goals and the boxes are in the
world, the grid in the SLAM map's frame). Needs Pillow only.
"""

import argparse
import csv
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

GOALS = [(3.0, 6.0), (8.5, 6.0), (8.0, 1.0), (4.0, 1.0), (1.5, 2.5)]
SCALE = 4  # pixels per grid cell


def read_map(yaml_path):
    meta = {}
    for line in open(yaml_path, encoding='utf-8'):
        if ':' in line:
            key, value = line.split(':', 1)
            meta[key.strip()] = value.strip()
    image = Image.open(os.path.join(os.path.dirname(yaml_path), meta['image'])).convert('RGB')
    origin = [float(v) for v in meta['origin'].strip('[]').split(',')]
    return image, float(meta['resolution']), origin[0], origin[1]


def read_csv(path):
    truth, belief = [], []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            truth.append((float(row['true_x']), float(row['true_y'])))
            if row['est_x']:
                belief.append((float(row['est_x']), float(row['est_y'])))
    return truth, belief


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--map', required=True)
    parser.add_argument('--run', action='append', required=True,
                        help='TITLE=CSV or TITLE=CSV=x0,y0,x1,y1 (an unmapped box)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--world-from-map', default='',
                        help='align_map.py JSON: the runs are in the world, the grid in the map')
    args = parser.parse_args()
    to_map = lambda x, y: (x, y)  # noqa: E731
    if args.world_from_map:
        with open(args.world_from_map, encoding='utf-8') as f:
            t = json.load(f)
        c, s = math.cos(t['yaw']), math.sin(t['yaw'])
        to_map = lambda x, y: (c * (x - t['x']) + s * (y - t['y']),  # noqa: E731
                               -s * (x - t['x']) + c * (y - t['y']))

    grid, resolution, ox, oy = read_map(args.map)
    width, height = grid.size
    base = grid.resize((width * SCALE, height * SCALE), Image.NEAREST)
    font = ImageFont.load_default(size=22) if hasattr(ImageFont, 'load_default') else None

    def px(x, y):
        x, y = to_map(x, y)
        return ((x - ox) / resolution * SCALE, (height - (y - oy) / resolution) * SCALE)

    panels = []
    for spec in args.run:
        title, path, *boxes = spec.split('=')
        truth, belief = read_csv(path)
        panel = base.copy()
        draw = ImageDraw.Draw(panel)
        for box in boxes:
            x0, y0, x1, y1 = (float(v) for v in box.split(','))
            corners = [px(x0, y0), px(x1, y0), px(x1, y1), px(x0, y1)]
            draw.polygon(corners, fill=(214, 96, 77), outline=(140, 40, 30), width=3)
        if len(belief) > 1:
            draw.line([px(*p) for p in belief], fill=(240, 170, 60), width=2)
        if len(truth) > 1:
            draw.line([px(*p) for p in truth], fill=(33, 102, 172), width=4)
        for gx, gy in GOALS:
            cx, cy = px(gx, gy)
            draw.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], outline=(27, 120, 55), width=4)
        sx, sy = px(*truth[0])
        draw.rectangle([sx - 8, sy - 8, sx + 8, sy + 8], fill=(27, 120, 55))
        draw.text((14, 10), title, fill=(0, 0, 0), font=font)
        panels.append(panel)

    gap = 16
    legend_h = 44
    out = Image.new('RGB', (sum(p.width for p in panels) + gap * (len(panels) - 1),
                            panels[0].height + legend_h), 'white')
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + gap
    draw = ImageDraw.Draw(out)
    y = panels[0].height + 10
    items = [((33, 102, 172), 'true path'), ((240, 170, 60), 'believed pose'),
             ((27, 120, 55), 'start / goals')]
    if any(len(spec.split('=')) > 2 for spec in args.run):
        items.append(((214, 96, 77), 'obstacle not in the map'))
    x = 14
    for color, label in items:
        draw.rectangle([x, y + 4, x + 28, y + 20], fill=color)
        draw.text((x + 36, y), label, fill=(0, 0, 0), font=font)
        x += 60 + int(draw.textlength(label, font=font))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.save(args.out, optimize=True)
    print(f'wrote {args.out} ({out.width} x {out.height})')


if __name__ == '__main__':
    main()
