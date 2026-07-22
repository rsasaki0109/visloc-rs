#!/usr/bin/env python3
"""Export LightGlue matches for the acceptance-neutral DPVO submap probe."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


OFFSETS = (1, 2, 4, 8, 12, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--old-anchor", type=int, default=38)
    parser.add_argument("--new-anchor", type=int, default=462)
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--filter-threshold", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from lightglue import LightGlue
    from lightglue.utils import rbd

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((args.dump_dir / "manifest.csv").open(newline="", encoding="utf-8")))
    files = {
        int(row["arrival_index"]): (
            args.dump_dir / row["keypoints_file"],
            args.dump_dir / row["descriptors_file"],
        )
        for row in rows
    }
    cache: dict[int, dict[str, torch.Tensor]] = {}

    def features(arrival: int) -> dict[str, torch.Tensor]:
        if arrival not in cache:
            keypoint_path, descriptor_path = files[arrival]
            keypoints = torch.from_numpy(np.load(keypoint_path)).float().to(device)
            descriptors = torch.from_numpy(np.load(descriptor_path)).float().to(device)
            descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
            cache[arrival] = {
                "keypoints": keypoints[None],
                "descriptors": descriptors[None],
                "image_size": torch.tensor([[94.0, 60.0]], device=device),
            }
        return cache[arrival]

    pair_roles: dict[tuple[int, int], str] = {}
    for role, anchor in (("temporal_old", args.old_anchor), ("temporal_new", args.new_anchor)):
        arrivals = sorted(a for a in files if anchor - args.radius <= a <= anchor + args.radius)
        for index, first in enumerate(arrivals):
            for offset in OFFSETS:
                next_index = index + offset
                if next_index < len(arrivals):
                    pair_roles[(first, arrivals[next_index])] = role
    pair_roles[(args.old_anchor, args.new_anchor)] = "anchor"

    matcher = LightGlue(features="superpoint", filter_threshold=args.filter_threshold).eval().to(device)
    manifest_rows: list[tuple[str, int, int, str, int]] = []
    for pair_index, ((first, second), role) in enumerate(sorted(pair_roles.items()), start=1):
        with torch.no_grad():
            output = rbd(matcher({"image0": features(first), "image1": features(second)}))
        matches = output["matches"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        filename = f"pair_{first:06}_{second:06}.txt"
        with (args.out_dir / filename).open("w", encoding="utf-8") as handle:
            handle.write("# QUERY_IDX TRAIN_IDX CONFIDENCE\n")
            for (query_index, train_index), score in zip(matches, scores):
                handle.write(f"{int(query_index)} {int(train_index)} {float(score):.9g}\n")
        manifest_rows.append((role, first, second, filename, len(matches)))
        if pair_index % 25 == 0 or pair_index == len(pair_roles):
            print(f"lightglue_pairs={pair_index}/{len(pair_roles)}", flush=True)

    with (args.out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("role", "arrival_i", "arrival_j", "matches_file", "match_count"))
        writer.writerows(manifest_rows)
    print(f"wrote={args.out_dir / 'manifest.csv'} pairs={len(manifest_rows)} device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
