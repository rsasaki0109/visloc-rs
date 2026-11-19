#!/usr/bin/env python3
"""Evaluate saved outdoor elevation updates using independently reviewed building labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve


CATEGORIES = {"stable": 0, "addition": 1, "removal": -1}


def common_crop(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def surfaces(root, epoch, cases, tiles):
    import laspy

    data = [[] for _ in cases]
    for tile, core in tiles.items():
        crops = [common_crop(core, case["bounds"]) for case in cases]
        if not any(crops):
            continue
        with laspy.open(root / f"ahn{epoch}-{tile}.laz") as handle:
            for points in handle.chunk_iterator(1_000_000):
                x, y, z = (np.asarray(getattr(points, axis)) for axis in "xyz")
                good = ~np.isin(points.classification, [7, 18]) & (z >= -10) & (z < 400)
                for i, crop in enumerate(crops):
                    if crop is None:
                        continue
                    x0, y0, x1, y1 = crop
                    keep = good & (x0 <= x) & (x < x1) & (y0 <= y) & (y < y1)
                    width = cases[i]["bounds"][2] - cases[i]["bounds"][0]
                    origin_x, origin_y = cases[i]["bounds"][:2]
                    cells = (np.floor(y[keep]).astype(int) - origin_y) * width + np.floor(x[keep]).astype(int) - origin_x
                    data[i].append((cells, z[keep]))
    for case, chunks in zip(cases, data, strict=True):
        x0, y0, x1, y1 = case["bounds"]
        cells = np.concatenate([chunk[0] for chunk in chunks])
        z = np.concatenate([chunk[1] for chunk in chunks])
        index = np.lexsort((z, cells))
        sorted_cells, sorted_z = cells[index], z[index]
        ids, starts, counts = np.unique(sorted_cells, return_index=True, return_counts=True)
        keep = counts >= 4
        elevation = np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32)
        elevation.flat[ids[keep]] = sorted_z[starts[keep] + (0.9 * (counts[keep] - 1)).astype(int)]
        yield elevation


def masks(case, shape):
    truth = np.full(shape, 2, dtype=np.int8)
    origin_x, origin_y = case["bounds"][:2]
    labels = {}
    for item in case["labels"]:
        x0, y0, x1, y1 = item["bounds"]
        selection = np.zeros(shape, dtype=bool)
        selection[y0 - origin_y:y1 - origin_y, x0 - origin_x:x1 - origin_x] = True
        if np.any(truth[selection] != 2):
            raise ValueError("overlapping annotation rectangles")
        truth[selection] = CATEGORIES[item["class"]]
        labels[item["name"]] = (item["class"], selection)
    return truth, labels


def score(truth, prediction):
    result = {}
    for category, sign in CATEGORIES.items():
        if sign:
            valid = truth != 2
            tp = int(np.count_nonzero(valid & (truth == sign) & (prediction == sign)))
            fp = int(np.count_nonzero(valid & (truth != sign) & (prediction == sign)))
            fn = int(np.count_nonzero(valid & (truth == sign) & (prediction != sign)))
            result[category] = {"tp": tp, "fp": fp, "fn": fn,
                                "precision": tp / (tp + fp) if tp + fp else None,
                                "recall": tp / (tp + fn) if tp + fn else None}
    result["stable_false_changes"] = int(np.count_nonzero((truth == 0) & (prediction != 0)))
    result["stable_columns"] = int(np.count_nonzero(truth == 0))
    return result


def run(args, config):
    args.maps_out.mkdir(parents=True)
    cases = config["cases"]
    surfaces_by_epoch = {}
    for epoch in sorted({epoch for case in cases for epoch in case["epochs"]}):
        selected = [case for case in cases if epoch in case["epochs"]]
        data = surfaces(args.data, epoch, selected, config["tiles"])
        surfaces_by_epoch[epoch] = {case["name"]: values for case, values in zip(selected, data, strict=True)}
    rows, all_truth, all_baseline, all_updated = {}, [], [], []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for case in cases:
        name = case["name"]
        old, new = (surfaces_by_epoch[epoch][name] for epoch in case["epochs"])
        both = np.isfinite(old) & np.isfinite(new)
        difference = new - old
        baseline = np.where(both & (difference >= 3), 1, np.where(both & (difference <= -3), -1, 0)).astype(np.int8)
        changed = np.zeros_like(baseline)
        for sign in (-1, 1):
            support = convolve((baseline == sign).astype(np.uint8), kernel, mode="constant")
            changed[(baseline == sign) & (support >= 6)] = sign
        path = args.maps_out / f"{case['name']}.npz"
        updated = np.where(changed != 0, new, old)
        np.savez_compressed(path, bounds=case["bounds"], cell_m=1, elevation_m=updated, change=changed)
        with np.load(path, allow_pickle=False) as reloaded:
            if not np.array_equal(reloaded["elevation_m"], updated, equal_nan=True):
                raise ValueError("reloaded map elevation differs from the update")
            confirmed = reloaded["change"]
        truth, labels = masks(case, old.shape)
        label_rows = {}
        for label_name, (category, mask) in labels.items():
            valid = both & mask
            label_rows[label_name] = {
                "class": category, "cells": int(mask.sum()), "observed": int(valid.sum()),
                "change_median_m": float(np.median(difference[valid])) if valid.any() else None,
                "baseline_additions": int(np.count_nonzero(mask & (baseline == 1))),
                "baseline_removals": int(np.count_nonzero(mask & (baseline == -1))),
                "saved_additions": int(np.count_nonzero(mask & (confirmed == 1))),
                "saved_removals": int(np.count_nonzero(mask & (confirmed == -1))),
            }
        rows[name] = {"epochs": case["epochs"], "labels": label_rows, "baseline": score(truth, baseline),
                              "saved_update": score(truth, confirmed)}
        all_truth.append(truth.ravel())
        all_baseline.append(baseline.ravel())
        all_updated.append(confirmed.ravel())
    merged_truth = np.concatenate(all_truth)
    return {"cases": rows, "baseline": score(merged_truth, np.concatenate(all_baseline)),
            "saved_update": score(merged_truth, np.concatenate(all_updated))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--maps-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.annotations.read_text())
    result = run(args, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
