#!/usr/bin/env python3
"""Build bounded target-frame candidate windows; target selection is an input."""
import argparse
import json
from bisect import bisect_left, bisect_right
from pathlib import Path

from benchmark_electro import write_candidate_manifest
from select_rig_sift_supplements import parse_frames


def targeted_pairs(frames, targets, gap):
    if any(type(target) is not int for target in targets):
        raise ValueError('Target frame IDs must be integers')
    if gap < 0 or not targets or len(set(targets)) != len(targets):
        raise ValueError('Require nonnegative gap and unique nonempty targets')
    if any(target not in frames for target in targets):
        raise ValueError('Unknown target frame')
    names = sorted(name for members in frames.values() for name in members)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate image names')
    indices = {name: index for index, name in enumerate(names)}
    ordered = sorted(frames)
    pairs = set()
    for target in targets:
        begin = bisect_left(ordered, target - gap)
        end = bisect_right(ordered, target + gap)
        for frame in ordered[begin:end]:
            for left in frames[target]:
                for right in frames[frame]:
                    if left != right:
                        pairs.add(tuple(sorted((indices[left], indices[right]))))
    return names, sorted(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rig-manifest', type=Path, required=True)
    target_input = parser.add_mutually_exclusive_group(required=True)
    target_input.add_argument('--target-frames')
    target_input.add_argument('--selection-json', type=Path)
    parser.add_argument('--max-frame-gap', type=int, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    targets = (json.loads(args.selection_json.read_text())['target_frames'] if args.selection_json
               else [int(value) for value in args.target_frames.split(',')])
    if not isinstance(targets, list):
        parser.error('selection target_frames must be a list')
    frames = parse_frames(args.rig_manifest.read_bytes())
    names, pairs = targeted_pairs(frames, targets, args.max_frame_gap)
    args.output_directory.mkdir()  # Never replace an existing experiment.
    output = args.output_directory / 'candidates.txt'
    write_candidate_manifest(output, names, pairs, metadata={
        'pair_source': 'targeted-dense-window-v1',
        'max_frame_gap': str(args.max_frame_gap),
        'target_frames': ','.join(map(str, sorted(targets))),
    })
    print(f'{len(names)} images, {len(pairs)} pairs -> {output}')


if __name__ == '__main__':
    main()
