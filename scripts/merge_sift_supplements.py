#!/usr/bin/env python3
"""Prefix-preserving, per-image SIFT supplement merge primitives.

Novelty is measured against the original bank only. Supplement orientation
variants remain distinct. No all-image descriptor bank or pair matrix is built.
"""

import math
import os
from pathlib import Path
import tempfile

HEADER = b"# visloc adaptive feature bank: legacy prefix + spatially novel compatible supplement\n"


def publish_file(destination, chunks):
    """Publish complete bytes without overwriting an existing destination.

    A same-directory temporary file is fsynced then hard-linked atomically to
    the destination (link fails if it exists). A crash can leave an unreferenced
    temporary file, never a partially written destination. This primitive does
    not implement bank-level completion or restart validation.
    """
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix=".sift-merge-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for chunk in chunks:
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)


def feature_rows(lines):
    for line in lines:
        if line.strip() and not line.lstrip().startswith(b"#"):
            yield line


def position(row):
    fields = row.split()
    if len(fields) < 2:
        raise ValueError("feature row lacks coordinates")
    x, y = map(float, fields[:2])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("nonfinite feature coordinates")
    return x, y


def merged_rows(base_lines, supplement_lines):
    """Yield original row bytes then novel supplement rows, with a 1px radius.

    Retains only base coordinates for one image; it does not retain descriptors.
    Callers must finish consuming this generator before publishing output.
    """
    grid = {}
    for row in feature_rows(base_lines):
        x, y = position(row)
        grid.setdefault((math.floor(x), math.floor(y)), []).append((x, y))
        yield row
    for row in feature_rows(supplement_lines):
        x, y = position(row)
        ix, iy = math.floor(x), math.floor(y)
        if not any((x - a) ** 2 + (y - b) ** 2 <= 1.0
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for a, b in grid.get((ix + dx, iy + dy), ())):
            yield row
