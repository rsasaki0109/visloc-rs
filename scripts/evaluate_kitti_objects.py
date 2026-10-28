#!/usr/bin/env python3
"""Score `objects_metric_offline`'s objects on the ground against KITTI's
tracklets.

    python3 scripts/evaluate_kitti_objects.py --gt gt_objects.csv \
        --boxes boxes.csv --ground ground_objects.csv [--out matches.csv]

- **Matching:** each 2D track (`object_id` in `boxes.csv`) goes to the
  tracklet whose projected box overlaps it most (IoU >= 0.3, a compatible
  class) in most of the frames they share; a track that matches nothing in
  half its frames is left out.
- **Height:** the track's median height against the tracklet's `h`. Cyclists
  are left out: KITTI's box holds the rider, YOLOX's "bicycle" box the bike.
- **Distance:** the track's median horizontal distance to its foot, against
  the median distance to the tracklet footprint's nearest point over the
  matched frames (ground contact sees the near edge).
- **Speed:** a line fitted to the tracklet's bottom centre over the matched
  frames, against the track's speed.

`--gt` is `kitti_tracklets_to_csv.py`'s output; `--boxes` is
`yolox_track_demo --boxes-csv`'s.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# YOLOX class -> KITTI types it may match.
COMPATIBLE = {
    "person": {"Pedestrian", "Person_sitting"},
    "bicycle": {"Cyclist"},
    "motorcycle": {"Cyclist"},
    "car": {"Car", "Van"},
    "truck": {"Truck", "Van"},
    "bus": {"Tram", "Truck"},
    "train": {"Tram"},
}
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def line_speed(times, points) -> float:
    t = np.asarray(times, float) * 1e-9
    p = np.asarray(points, float)
    t -= t.mean()
    if (t * t).sum() <= 0:
        return float("nan")
    v = (t[:, None] * (p - p.mean(0))).sum(0) / (t * t).sum()
    return float(np.linalg.norm(v))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--ground", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    gt_by_time = defaultdict(list)
    gt_rows = defaultdict(dict)
    for r in csv.DictReader(open(args.gt)):
        t = int(r["timestamp_ns"])
        gt_rows[r["gt_id"]][t] = r
        if r["u_min"]:
            gt_by_time[t].append(r)
    tracks = defaultdict(list)
    for r in csv.DictReader(open(args.boxes)):
        tracks[r["object_id"]].append(r)

    matched = {}
    for track_id, rows in tracks.items():
        cls = COCO.get(int(rows[0]["class_id"]))
        if cls not in COMPATIBLE:
            continue
        votes, frames = Counter(), defaultdict(list)
        for r in rows:
            t = int(r["timestamp_ns"])
            ours = [float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
            best, best_iou = None, 0.3
            for g in gt_by_time.get(t, []):
                if g["type"] not in COMPATIBLE[cls]:
                    continue
                score = iou(ours, [float(g[k]) for k in ("u_min", "v_min", "u_max", "v_max")])
                if score >= best_iou:
                    best, best_iou = g["gt_id"], score
            if best is not None:
                votes[best] += 1
                frames[best].append(t)
        if votes:
            gt_id, n = votes.most_common(1)[0]
            if n >= 0.5 * len(rows):
                matched[track_id] = (gt_id, frames[gt_id])

    results = []
    for r in csv.DictReader(open(args.ground)):
        if r["object_id"] not in matched:
            continue
        gt_id, times = matched[r["object_id"]]
        g = [gt_rows[gt_id][t] for t in times]
        gt_type = g[0]["type"]
        gt_h = float(g[0]["h"])
        gt_near = statistics.median(float(x["near_distance"]) for x in g)
        gt_speed = line_speed(times, [[float(x["x"]), float(x["y"])] for x in g]) if len(times) >= 3 else float("nan")
        results.append({
            "object_id": r["object_id"], "class": r["class"], "gt_id": gt_id, "gt_type": gt_type,
            "frames": len(times), "height": float(r["height"]), "gt_height": gt_h,
            "distance": float(r["distance"]), "gt_near_distance": gt_near,
            "speed": float(r["speed"]), "gt_speed": gt_speed,
        })

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["object_id"])
            writer.writeheader()
            writer.writerows(results)

    print(f"{len(tracks)} 2D tracks, {len(matched)} matched to tracklets, {len(results)} placed on the ground")
    print(f"{'class':<10} {'n':>3}  {'height err (median |.|, bias)':>32}  {'distance err':>22}  {'speed err (m/s)':>18}")
    for cls in sorted({x["class"] for x in results}):
        rows = [x for x in results if x["class"] == cls]
        dist = [x["distance"] / x["gt_near_distance"] - 1 for x in rows if x["gt_near_distance"] > 0.5]
        speed = [x["speed"] - x["gt_speed"] for x in rows if x["gt_speed"] == x["gt_speed"]]
        if cls in ("bicycle", "motorcycle"):
            height = "(cyclist boxes differ)"
        else:
            rel = [x["height"] / x["gt_height"] - 1 for x in rows]
            ab = [x["height"] - x["gt_height"] for x in rows]
            height = (f"{statistics.median(map(abs, ab)):.2f} m / {100 * statistics.median(map(abs, rel)):.0f} %, "
                      f"{100 * statistics.median(rel):+.0f} %")
        dtext = f"{100 * statistics.median(map(abs, dist)):.0f} %, {100 * statistics.median(dist):+.0f} %" if dist else "-"
        stext = f"{statistics.median(map(abs, speed)):.2f}, {statistics.median(speed):+.2f}" if speed else "-"
        print(f"{cls:<10} {len(rows):>3}  {height:>32}  {dtext:>22}  {stext:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
