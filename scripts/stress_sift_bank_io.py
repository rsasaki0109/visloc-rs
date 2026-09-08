#!/usr/bin/env python3
"""Synthetic bank publication/resume I/O stress, not SfM scalability evidence."""

import argparse
import json
from pathlib import Path
import resource
import time

from merge_sift_supplements import write_bank


def run(root, count):
    if count <= 0:
        raise ValueError("count must be positive")
    root.mkdir()  # Refuse to touch an existing run.
    base, supplement, output = [root / n for n in ("base", "supplement", "output")]
    base.mkdir()
    supplement.mkdir()
    row = b"0 0 1 " + b"0 " * 127 + b"1\n"
    started = time.monotonic()
    for index in range(count):
        with (base / f"image_{index:06d}_features.txt").open("xb") as stream:
            stream.write(row)
    generation_seconds = time.monotonic() - started
    started = time.monotonic()
    first = write_bank(base, supplement, {"image_names": []}, output)
    publication_seconds = time.monotonic() - started
    started = time.monotonic()
    second = write_bank(base, supplement, {"image_names": []}, output, True)
    resume_seconds = time.monotonic() - started
    if first["inventory_sha256"] != second["inventory_sha256"] or second["reused"] != count:
        raise AssertionError("resume inventory differs")
    for path in output.iterdir():
        if path.read_bytes() != row:
            raise AssertionError(f"unexpected bytes: {path.name}")
    return {"scope": "synthetic-one-feature-per-image-no-supplements",
            "count": count, "generation_seconds": generation_seconds,
            "publication_seconds": publication_seconds, "resume_seconds": resume_seconds,
            "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "publication": first, "resume": second, "independent_all_rows_equal": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.count), indent=2, sort_keys=True))
