#!/usr/bin/env python3
"""Select low-feature rig frames plus a bounded ordinal halo, without GT.

Print a selection manifest; do not modify features or create image links.
This implements an explicit policy, not a claim of historical reproduction.
"""

import argparse
import hashlib
import json
from pathlib import Path


def select(frames, counts, threshold, halo):
    if threshold < 0 or halo < 0:
        raise ValueError("threshold and halo must be nonnegative")
    ordered = sorted(frames)
    names = [name for frame in ordered for name in frames[frame]]
    if len(set(names)) != len(names) or set(names) != set(counts):
        raise ValueError("image/count membership must be exact and unique")
    if any(not frames[frame] for frame in ordered) or any(n < 0 for n in counts.values()):
        raise ValueError("empty frame or negative feature count")
    seeds = [i for i, frame in enumerate(ordered)
             if min(counts[name] for name in frames[frame]) < threshold]
    # Difference array avoids O(N * halo) work for large halo values.
    delta = [0] * (len(ordered) + 1)
    for index in seeds:
        delta[max(0, index - halo)] += 1
        delta[min(len(ordered), index + halo + 1)] -= 1
    selected = []
    coverage = 0
    for index, frame in enumerate(ordered):
        coverage += delta[index]
        if coverage:
            selected.append(frame)
    return {
        "schema": "visloc_adaptive_sift_selection_v1",
        "policy": {"frame_halo": halo, "min_sensor_rows_lt": threshold},
        "base_frames": len(seeds), "selected_frames": len(selected),
        "selected_images": sum(len(frames[frame]) for frame in selected),
        "image_names": sorted(name for frame in selected for name in frames[frame]),
    }


def parse_frames(raw):
    frames = {}
    sensors = set()
    declared = set()
    names = set()
    stems = set()
    for line in raw.decode().splitlines():
        fields = line.split()
        if fields and fields[0] == "S":
            if len(fields) != 16:
                raise ValueError("malformed S row")
            sensor = int(fields[1])
            if sensor < 0 or sensor in declared:
                raise ValueError("invalid or duplicate sensor declaration")
            declared.add(sensor)
            continue
        if not fields or fields[0] != "F":
            continue
        if len(fields) != 4:
            raise ValueError("malformed F row")
        frame, name, sensor = int(fields[1]), fields[2], int(fields[3])
        if frame < 0 or sensor < 0 or (frame, sensor) in sensors:
            raise ValueError("invalid or duplicate frame/sensor")
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("image names must be basenames")
        if name in names or Path(name).stem in stems:
            raise ValueError("duplicate image or colliding feature filename")
        names.add(name)
        stems.add(Path(name).stem)
        sensors.add((frame, sensor))
        frames.setdefault(frame, []).append(name)
    if not frames:
        raise ValueError("no rig frames")
    by_frame = {frame: set() for frame in frames}
    for frame, sensor in sensors:
        by_frame[frame].add(sensor)
    if not declared or any(value != declared for value in by_frame.values()):
        raise ValueError("each frame must contain exactly the declared sensors")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--min-sensor-rows-lt", type=int, required=True)
    parser.add_argument("--frame-halo", type=int, required=True)
    args = parser.parse_args()
    if args.min_sensor_rows_lt < 0 or args.frame_halo < 0:
        parser.error("threshold and halo must be nonnegative")
    raw = args.rig_manifest.read_bytes()
    frames = parse_frames(raw)
    counts = {}
    for names in frames.values():
        for name in names:
            with (args.features_dir / (Path(name).stem + "_features.txt")).open() as stream:
                counts[name] = sum(1 for line in stream
                                   if line.strip() and not line.lstrip().startswith("#"))
    result = select(frames, counts, args.min_sensor_rows_lt, args.frame_halo)
    result.update(rig_manifest=str(args.rig_manifest),
                  rig_manifest_sha256=hashlib.sha256(raw).hexdigest())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
