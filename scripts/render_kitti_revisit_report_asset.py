#!/usr/bin/env python3
"""Render the README KITTI revisit loop-candidate asset.

Consumes the output directory produced by `kitti_revisit_scanner_demo`
(`summary.txt`, `candidates.csv`, copied strongest-pair images, and the
optional verified-inlier overlay SVG under `assets/`) and writes a compact JPEG
for the README.

Example:
    python scripts/render_kitti_revisit_report_asset.py \
        target/kitti_revisit_report_50x30_deep200_strict \
        --out docs/assets/kitti_revisit_loop_candidate.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

from kitti_revisit_report_asset_render import (
    load_report_asset_inputs,
    render_asset_image,
    save_asset_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path, help="kitti_revisit_scanner_demo output dir")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/assets/kitti_revisit_loop_candidate.jpg"),
        help="Output JPEG path",
    )
    parser.add_argument("--width", type=int, default=1400, help="Output image width")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality")
    return parser.parse_args()


def render(report_dir: Path, out_path: Path, width: int, quality: int) -> None:
    image = render_asset_image(load_report_asset_inputs(report_dir), width)
    save_asset_image(image, out_path, quality)
    print(f"wrote {out_path}")


def main() -> int:
    args = parse_args()
    render(args.report_dir, args.out, args.width, args.quality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
