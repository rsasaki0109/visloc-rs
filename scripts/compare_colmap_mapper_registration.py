#!/usr/bin/env python3
"""Registration-order and pose comparison for the COLMAP mapper port.

This is the §4.2 items 1--3 diagnostic from
``docs/colmap_rig_mapper_port_plan.md``: it turns a COLMAP ``mapper.log`` and
a ported-mapper log (``stderr.log``/``mapper.log``) into

* the initial image pair each mapper chose (mapped from COLMAP database image
  ids through the model text + image aliases to *rig frame* ids),
* the per-frame registration order of each mapper,
* the longest common subsequence (LCS) ratio of those orders (full and with
  the initial pair removed),
* the registration index at which the orders first diverge and how many
  positions are exactly equal,
* global-BA event indices for each mapper, and
* a ground-truth-free per-frame pose difference: both final models' camera
  centres are matched by image name and one global Sim(3) (Umeyama) maps the
  ported model onto the COLMAP model; the residual is reported per frame and
  bucketed by COLMAP registration rank.

It deliberately reuses ``score_openloris_model``'s model/alias loading and
``umeyama`` by import rather than reimplementing them, and never reads ground
truth.

Example::

    python3 scripts/compare_colmap_mapper_registration.py \
      --colmap-log <colmap>/logs/mapper.log \
      --colmap-images <colmap>/models-text/0/images.txt \
      --ported-log <ported>/stderr.log \
      --ported-images <ported>/model/model/0/images.txt \
      --rig-manifest <tier>-rig-manifest-v1.txt \
      --image-aliases image_aliases.tsv \
      --out registration-diag.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import score_openloris_model as om

COLMAP_INIT_RE = re.compile(r"Registering initial image pair #(\d+) and #(\d+)")
COLMAP_REGISTER_RE = re.compile(r"Registering image #(\d+) \(num_reg_frames=(\d+)\)")
PORTED_INIT_RE = re.compile(
    r"^INIT_PAIR image1=(\d+) image2=(\d+) frame1=(\d+) frame2=(\d+)"
)
PORTED_REGISTER_RE = re.compile(r"^REGISTER path=(\w+) frame=(\d+)")


class RegistrationDiagError(RuntimeError):
    pass


def parse_rig_manifest(path: Path) -> tuple[dict[str, int], dict[int, list[str]]]:
    """Return ``image_name -> frame_id`` and ``frame_id -> [image_name]``."""
    flat_to_frame: dict[str, int] = {}
    frame_images: dict[int, list[str]] = defaultdict(list)
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] != "F":
            continue
        if len(fields) < 4:
            raise RegistrationDiagError(f"malformed manifest F row {path}:{line_number}")
        frame_id = int(fields[1])
        name = fields[2]
        if name in flat_to_frame:
            raise RegistrationDiagError(f"duplicate image {name!r} in {path}")
        flat_to_frame[name] = frame_id
        frame_images[frame_id].append(name)
    if not flat_to_frame:
        raise RegistrationDiagError(f"rig manifest has no F rows: {path}")
    return flat_to_frame, frame_images


def parse_model_image_ids(path: Path) -> dict[int, str]:
    """Return ``image_id -> name`` from a COLMAP text ``images.txt``."""
    result: dict[int, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10:
            continue
        try:
            image_id = int(fields[0])
            float(fields[1])
            int(fields[8])
        except ValueError as exc:
            raise RegistrationDiagError(f"non-numeric pose row {path}:{line_number}") from exc
        name = fields[9]
        if image_id in result:
            raise RegistrationDiagError(f"duplicate image id {image_id} in {path}")
        result[image_id] = name
    if not result:
        raise RegistrationDiagError(f"model contains no poses: {path}")
    return result


def parse_colmap_log(path: Path) -> dict[str, Any]:
    """Return init pair, registration order (image ids) and event indices."""
    init_pairs: list[tuple[int, int]] = []
    order: list[int] = []
    events: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = COLMAP_INIT_RE.search(raw)
        if match:
            init_pairs.append((int(match.group(1)), int(match.group(2))))
            continue
        match = COLMAP_REGISTER_RE.search(raw)
        if match:
            order.append(int(match.group(1)))
            continue
        if "Retriangulation and Global bundle adjustment" in raw:
            events.append({"kind": "retriangulation_global_ba", "registered": len(order)})
        elif "Global bundle adjustment" in raw:
            events.append({"kind": "initial_global_ba", "registered": len(order)})
    if not init_pairs:
        raise RegistrationDiagError(f"no initial image pair in {path}")
    if not order:
        raise RegistrationDiagError(f"no registration lines in {path}")
    return {
        "init_attempts": init_pairs,
        "init_pair": init_pairs[-1],
        "order": order,
        "events": events,
    }


def parse_ported_log(path: Path) -> dict[str, Any]:
    """Return init frames, registration frame order and event indices."""
    init: tuple[int, int, int, int] | None = None
    order: list[int] = []
    events: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = PORTED_INIT_RE.match(raw)
        if match:
            init = (int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
            continue
        match = PORTED_REGISTER_RE.match(raw)
        if match:
            order.append(int(match.group(2)))
            continue
        if raw.startswith("TIMING iterative_global_refinement"):
            events.append({"kind": "iterative_global_refinement", "registered": len(order)})
    if init is None:
        raise RegistrationDiagError(f"no INIT_PAIR line in {path}")
    if not order:
        raise RegistrationDiagError(f"no REGISTER lines in {path}")
    return {"init": init, "order": order, "events": events}


def lcs_length(a: list[Any], b: list[Any]) -> int:
    """Hirschberg length of the longest common subsequence (O(min) memory)."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for value in a:
        current = [0]
        for j, other in enumerate(b):
            if value == other:
                current.append(previous[j] + 1)
            else:
                current.append(max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


def lcs_sequence(a: list[Any], b: list[Any]) -> list[Any]:
    """Longest common subsequence (Hirschberg, O(n*m) time, O(min) memory)."""
    if not a or not b:
        return []
    if len(a) == 1:
        return [a[0]] if a[0] in b else []
    mid = len(a) // 2
    forward = _lcs_row(a[:mid], b)
    backward = _lcs_row(a[mid:][::-1], b[::-1])
    split = max(range(len(b) + 1), key=lambda k: forward[k] + backward[len(b) - k])
    return lcs_sequence(a[:mid], b[:split]) + lcs_sequence(a[mid:], b[split:])


def _lcs_row(a: list[Any], b: list[Any]) -> list[int]:
    previous = [0] * (len(b) + 1)
    for value in a:
        current = [0]
        for j, other in enumerate(b):
            if value == other:
                current.append(previous[j] + 1)
            else:
                current.append(max(previous[j + 1], current[j]))
        previous = current
    return previous


def _runs_of_equality(a: list[Any], b: list[Any], min_length: int = 8) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    i = 0
    while i < len(a):
        if a[i] == b[i]:
            j = i
            while j < len(a) and a[j] == b[j]:
                j += 1
            if j - i >= min_length:
                runs.append({"start": i, "length": j - i})
            i = j
        else:
            i += 1
    runs.sort(key=lambda run: -run["length"])
    return runs


def compare_orders(colmap_frames: list[int], ported_frames: list[int]) -> dict[str, Any]:
    common = lcs_sequence(colmap_frames, ported_frames)
    colmap_set, ported_set = set(colmap_frames), set(ported_frames)
    first_divergence = None
    for index, (left, right) in enumerate(zip(colmap_frames, ported_frames)):
        if left != right:
            first_divergence = index
            break
    equal_positions = sum(1 for left, right in zip(colmap_frames, ported_frames) if left == right)
    return {
        "colmap_count": len(colmap_frames),
        "ported_count": len(ported_frames),
        "colmap_unique": len(colmap_set),
        "ported_unique": len(ported_set),
        "registered_set_equal": colmap_set == ported_set,
        "lcs_length": len(common),
        "lcs_ratio": len(common) / len(colmap_frames) if colmap_frames else 0.0,
        "equal_positions": equal_positions,
        "first_divergence_index": first_divergence,
        "colmap_at_divergence": colmap_frames[first_divergence] if first_divergence is not None else None,
        "ported_at_divergence": ported_frames[first_divergence] if first_divergence is not None else None,
        "longest_identical_runs": _runs_of_equality(colmap_frames, ported_frames),
    }


def pose_difference(
    ported_centres: dict[str, np.ndarray],
    colmap_centres: dict[str, np.ndarray],
    flat_to_colmap: dict[str, str],
    flat_to_frame: dict[str, int],
    colmap_rank: dict[int, int],
    worst: int = 20,
) -> dict[str, Any]:
    keys: list[str] = []
    source: list[np.ndarray] = []
    destination: list[np.ndarray] = []
    for flat_name, centre in ported_centres.items():
        colmap_name = flat_to_colmap.get(flat_name)
        if colmap_name is None or colmap_name not in colmap_centres:
            continue
        keys.append(flat_name)
        source.append(centre)
        destination.append(colmap_centres[colmap_name])
    if not keys:
        raise RegistrationDiagError("no image names common to both models")
    source_array = np.asarray(source)
    destination_array = np.asarray(destination)
    scale, rotation, translation = om.umeyama(source_array, destination_array)
    aligned = (scale * (rotation @ source_array.T).T) + translation
    error = np.linalg.norm(aligned - destination_array, axis=1)

    frame_error: dict[int, float] = defaultdict(float)
    for flat_name, value in zip(keys, error):
        frame = flat_to_frame.get(flat_name)
        if frame is None:
            continue
        frame_error[frame] = max(frame_error[frame], float(value))

    buckets: dict[int, list[float]] = defaultdict(list)
    for frame, value in frame_error.items():
        rank = colmap_rank.get(frame)
        if rank is None:
            continue
        buckets[rank // 100].append(value)

    worst_frames = sorted(frame_error.items(), key=lambda item: -item[1])[:worst]
    return {
        "matched_images": len(keys),
        "sim3_scale_ported_to_colmap": scale,
        "center_error_m": {
            "mean": float(error.mean()),
            "median": float(np.median(error)),
            "p95": float(np.percentile(error, 95)),
            "max": float(error.max()),
        },
        "worst_frames": [
            {"frame": frame, "error_m": value, "colmap_rank": colmap_rank.get(frame)}
            for frame, value in worst_frames
        ],
        "error_by_colmap_rank_bucket": [
            {
                "rank_start": start * 100,
                "images": len(values),
                "mean_m": float(np.mean(values)),
                "p95_m": float(np.percentile(values, 95)),
                "max_m": float(max(values)),
            }
            for start, values in sorted(buckets.items())
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-log", type=Path, required=True)
    parser.add_argument("--colmap-images", type=Path, required=True)
    parser.add_argument("--ported-log", type=Path, required=True)
    parser.add_argument("--ported-images", type=Path, required=True)
    parser.add_argument("--rig-manifest", type=Path, required=True)
    parser.add_argument("--image-aliases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worst", type=int, default=20)
    args = parser.parse_args(argv)

    flat_to_frame, _ = parse_rig_manifest(args.rig_manifest)
    colmap_to_flat = om.load_colmap_aliases(args.image_aliases)
    flat_to_colmap = {flat: name for name, flat in colmap_to_flat.items()}

    colmap_id_to_name = parse_model_image_ids(args.colmap_images)
    colmap_centres = om.load_model_centres(args.colmap_images)
    ported_centres = om.load_model_centres(args.ported_images)

    colmap_log = parse_colmap_log(args.colmap_log)
    ported_log = parse_ported_log(args.ported_log)

    def colmap_id_to_frame(image_id: int) -> int | None:
        name = colmap_id_to_name.get(image_id)
        if name is None:
            return None
        flat = colmap_to_flat.get(name)
        if flat is None:
            return None
        return flat_to_frame.get(flat)

    def colmap_id_info(image_id: int) -> dict[str, Any]:
        name = colmap_id_to_name.get(image_id)
        flat = colmap_to_flat.get(name) if name else None
        frame = flat_to_frame.get(flat) if flat else None
        sensor = None
        if flat:
            sensor = "cam2" if "cam2" in flat else "cam1" if "cam1" in flat else None
        return {"image_id": image_id, "colmap_name": name, "flat_name": flat, "frame": frame, "sensor": sensor}

    colmap_order = [colmap_id_to_frame(image_id) for image_id in colmap_log["order"]]
    colmap_order = [frame for frame in colmap_order if frame is not None]

    colmap_init = tuple(colmap_id_to_frame(image_id) for image_id in colmap_log["init_pair"])
    ported_init = (ported_log["init"][2], ported_log["init"][3])

    colmap_frames = list(colmap_init) + colmap_order
    ported_frames = list(ported_init) + list(ported_log["order"])

    colmap_rank = {frame: index for index, frame in enumerate(colmap_frames)}
    ported_rank = {frame: index for index, frame in enumerate(ported_frames)}

    payload: dict[str, Any] = {
        "schema": "visloc_colmap_mapper_registration_diag_v1",
        "inputs": {
            "colmap_log": str(args.colmap_log),
            "colmap_images": str(args.colmap_images),
            "ported_log": str(args.ported_log),
            "ported_images": str(args.ported_images),
            "rig_manifest": str(args.rig_manifest),
            "image_aliases": str(args.image_aliases),
        },
        "initial_pair": {
            "colmap_attempts": [colmap_id_info(image_id) for pair in colmap_log["init_attempts"] for image_id in pair],
            "colmap_successful": [colmap_id_info(image_id) for image_id in colmap_log["init_pair"]],
            "colmap_successful_frames": list(colmap_init),
            "ported_image_ids": [ported_log["init"][0], ported_log["init"][1]],
            "ported_frames": list(ported_init),
            "frames_match": tuple(colmap_init) == tuple(ported_init),
        },
        "registration_order": {
            "full": compare_orders(colmap_frames, ported_frames),
            "post_init": compare_orders(colmap_order, list(ported_log["order"])),
        },
        "rank_displacement": {
            "max_abs": max(
                (abs(colmap_rank[frame] - ported_rank[frame]), frame)
                for frame in colmap_rank
                if frame in ported_rank
            ),
            "frames": [
                {
                    "frame": frame,
                    "colmap_rank": colmap_rank[frame],
                    "ported_rank": ported_rank[frame],
                }
                for frame in sorted(
                    (frame for frame in colmap_rank if frame in ported_rank),
                    key=lambda frame: -abs(colmap_rank[frame] - ported_rank[frame]),
                )[:20]
            ],
        },
        "global_ba_events": {
            "colmap_count": sum(1 for event in colmap_log["events"] if event["kind"] == "retriangulation_global_ba"),
            "ported_count": len(ported_log["events"]),
            "colmap_indices": [event["registered"] for event in colmap_log["events"]],
            "ported_indices": [event["registered"] for event in ported_log["events"]],
        },
        "pose_difference": pose_difference(
            ported_centres,
            colmap_centres,
            flat_to_colmap,
            flat_to_frame,
            colmap_rank,
            worst=args.worst,
        ),
    }
    payload["rank_displacement"]["max_abs"] = payload["rank_displacement"]["max_abs"][0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "lcs_ratio_full": payload["registration_order"]["full"]["lcs_ratio"],
        "lcs_ratio_post_init": payload["registration_order"]["post_init"]["lcs_ratio"],
        "registered_set_equal": payload["registration_order"]["full"]["registered_set_equal"],
        "init_frames_match": payload["initial_pair"]["frames_match"],
        "pose_mean_m": payload["pose_difference"]["center_error_m"]["mean"],
        "pose_max_m": payload["pose_difference"]["center_error_m"]["max"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
