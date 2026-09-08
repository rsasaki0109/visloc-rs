#!/usr/bin/env python3
"""Read-only, bounded-memory audit of a completed SIFT shard replay.

The reference may contain other shards; the replay must contain exactly this
shard's feature/loci files. Exit status does not certify extractor completion:
the caller must independently verify the measured process exited successfully.
"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def audit(images, reference, replay, expected_images):
    inputs = sorted(path for path in images.iterdir() if path.is_file())
    stems = [path.stem for path in inputs]
    if len(inputs) != expected_images or len(set(stems)) != len(stems):
        raise ValueError("unexpected image count or colliding image stems")
    expected = {stem + suffix for stem in stems
                for suffix in ("_features.txt", "_loci.txt")}
    actual = {path.name for path in replay.iterdir()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    mismatches = []
    missing_reference = []
    total_bytes = 0
    inventory = hashlib.sha256()
    for name in sorted(expected & actual):
        candidate = replay / name
        original = reference / name
        if not original.is_file():
            missing_reference.append(name)
            continue
        if not candidate.is_file():
            mismatches.append(name)
            continue
        size = candidate.stat().st_size
        candidate_hash = digest(candidate)
        total_bytes += size
        inventory.update(f"{name}\t{size}\t{candidate_hash}\n".encode())
        if size != original.stat().st_size or candidate_hash != digest(original):
            mismatches.append(name)
    return {
        "passed": not (missing or extra or mismatches or missing_reference),
        "images": len(inputs), "expected_files": len(expected),
        "actual_entries": len(actual), "replay_bytes": total_bytes,
        "missing": missing, "extra": extra, "mismatches": mismatches,
        "missing_reference": missing_reference,
        "inventory_sha256": inventory.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    args = parser.parse_args()
    result = audit(args.images, args.reference, args.replay, args.expected_images)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
