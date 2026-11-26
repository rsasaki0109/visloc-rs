#!/usr/bin/env python3
"""Convert a TUM RGB-D sequence's colour images to the EuRoC/ASL cam0 layout.

`basalt_euroc_localize_demo` reads queries in the EuRoC layout. This writes

  OUT/mav0/cam0/data.csv   `#timestamp [ns],filename`, nanosecond stamps
  OUT/mav0/cam0/data/      the colour images as 8-bit grey PNGs
  OUT/groundtruth.tum      the sequence's ground truth, unchanged (seconds)

The grey value is PIL's luma (ITU-R 601-2), the same conversion as
`image::DynamicImage::to_luma8`, which `tum_rgbd_save_map` uses for the map's
keyframes, so query and map descriptors see the same pixels. (The EuRoC
reader keeps only the red channel of a colour PNG, hence the conversion.)

Freiburg 3's colour images are already undistorted: localize against a map
saved with zero distortion (`tum_rgbd_save_map ... --k1 0 --k2 0 --p1 0
--p2 0 --k3 0`). Dataset images stay outside the repository and remain
subject to the dataset's CC BY 4.0 terms.

  python3 scripts/tum_rgbd_to_euroc.py \\
    --seq-dir /data/tum/rgbd_dataset_freiburg3_walking_xyz --out runs/tum/walking_xyz
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


def read_list(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        stamp, name = line.split()[:2]
        rows.append((float(stamp), name))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seq-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    rows = read_list(args.seq_dir / "rgb.txt")
    if args.max_frames is not None:
        rows = rows[: args.max_frames]
    data = args.out / "mav0" / "cam0" / "data"
    data.mkdir(parents=True, exist_ok=True)
    lines = ["#timestamp [ns],filename"]
    for stamp, name in rows:
        # TUM stamps carry 4-6 decimals; round to the microsecond, then ns.
        stamp_ns = int(round(stamp * 1e6)) * 1000
        file_name = f"{stamp_ns}.png"
        target = data / file_name
        if not target.exists():
            Image.open(args.seq_dir / name).convert("L").save(target)
        lines.append(f"{stamp_ns},{file_name}")
    (args.out / "mav0" / "cam0" / "data.csv").write_text("\n".join(lines) + "\n")
    ground_truth = args.seq_dir / "groundtruth.txt"
    if ground_truth.exists():
        shutil.copyfile(ground_truth, args.out / "groundtruth.tum")
    print(f"{len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
