#!/usr/bin/env python3
"""A LaMAria submission zip from an earlier one, with some sequences' estimates replaced.

    python scripts/lamaria_assemble_submission.py BASE.zip NEW_SLAM_DIR OUT.zip SEQ [SEQ ...]

Only the named sequences are taken from NEW_SLAM_DIR (SEQ.txt); every other entry of BASE.zip is copied as it is.
Each replacement must have a pose at every timestamp of the base estimate and only finite values, so a submission
never loses a pose or carries a NaN. It may add poses, as a filled late start does (`lamaria_reversed_start.py`).
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np


def check(new, old):
    """Why an estimate cannot replace another (a pose missing, timestamps out of order, non-finite values), or
    None."""
    a, b = np.loadtxt(new), np.loadtxt(old)
    ta, tb = a[:, 0].astype(np.int64), b[:, 0].astype(np.int64)
    if not np.isin(tb, ta).all():
        return "poses missing"
    if (np.diff(ta) <= 0).any():
        return "timestamps not increasing"
    if not np.isfinite(a).all():
        return "non-finite values"
    return None


def assemble(base, new_dir, out, seqs):
    """Writes OUT; returns the number of estimates in it."""
    replace = {f"slam/{s}.txt": Path(new_dir) / f"{s}.txt" for s in seqs}
    missing = [str(p) for p in replace.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"missing estimates: {missing}")
    with zipfile.ZipFile(base) as src:
        names = src.namelist()
        absent = [n for n in replace if n not in names]
        if absent:
            raise SystemExit(f"not in {base}: {absent}")
        bad = [(n, why) for n in replace if (why := check(replace[n], io.BytesIO(src.read(n))))]
        if bad:
            raise SystemExit(f"cannot replace: {bad}")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
            for name in names:
                if name in replace:
                    dst.write(replace[name], name)
                else:
                    dst.writestr(src.getinfo(name), src.read(name))
    with zipfile.ZipFile(out) as z:
        return sum(1 for n in z.namelist() if n.endswith(".txt"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("base")
    p.add_argument("new_dir")
    p.add_argument("out")
    p.add_argument("seqs", nargs="+")
    a = p.parse_args()
    n = assemble(a.base, a.new_dir, a.out, a.seqs)
    print(f"{a.out}: {n} estimates, {len(a.seqs)} replaced: {', '.join(a.seqs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
