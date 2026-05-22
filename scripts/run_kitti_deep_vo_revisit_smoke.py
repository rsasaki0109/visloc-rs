#!/usr/bin/env python3
"""Run the public-data KITTI 00 revisit scanner smoke test.

This is the cross-platform counterpart to
`scripts/run_kitti_deep_vo_revisit_smoke.sh`: it fetches or reuses the KITTI 00
start/revisit grayscale slices, runs `kitti_revisit_scanner_demo`, and writes a
small summary plus the demo's HTML report.

Default quick run:

    python scripts/run_kitti_deep_vo_revisit_smoke.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from kitti_revisit_smoke_cli import parse_args
from kitti_revisit_smoke_io import (
    expand_run_paths,
    read_summary,
    remove_output_dir,
    require_path,
    run_command,
)
from kitti_revisit_smoke_lib import read_candidates_csv, validate_expectations
from kitti_revisit_smoke_runner import (
    cargo_demo_command,
    config_from_args,
    fetch_command,
    render_asset_command,
    write_smoke_summary,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = parse_args(__doc__)

    expand_run_paths(args)
    config = config_from_args(args)

    if not args.skip_fetch:
        fetch_script = repo_root / "scripts" / "fetch_kitti_seq00_images.py"
        run_command(
            fetch_command(
                fetch_script,
                out_dir=config.start_dir,
                max_frames=config.start_frames,
                workers=config.workers,
            ),
            repo_root,
        )
        run_command(
            fetch_command(
                fetch_script,
                out_dir=config.revisit_dir,
                max_frames=config.revisit_frames,
                workers=config.workers,
                start_frame=config.revisit_start_frame,
            ),
            repo_root,
        )

    require_path(config.start_dir / "image_0", "start image_0 directory")
    require_path(config.revisit_dir / "image_0", "revisit image_0 directory")
    require_path(config.start_dir / "calib.txt", "start calib.txt")
    require_path(config.revisit_dir / "calib.txt", "revisit calib.txt")

    remove_output_dir(config.out_dir, repo_root)
    run_command(cargo_demo_command(config), repo_root)

    summary_path = config.out_dir / "summary.txt"
    candidates_path = config.out_dir / "candidates.csv"
    report_path = config.out_dir / "index.html"
    require_path(summary_path, "summary.txt")
    require_path(candidates_path, "candidates.csv")
    require_path(report_path, "index.html")

    summary_text = read_summary(summary_path)
    if "strongest_from=" not in summary_text:
        raise RuntimeError(f"no strongest loop pair found in {summary_path}")
    validate_expectations(read_candidates_csv(candidates_path), config.expectations)

    if config.readme_asset_out is not None:
        render_script = repo_root / "scripts" / "render_kitti_revisit_report_asset.py"
        run_command(render_asset_command(render_script, config), repo_root)
        require_path(config.readme_asset_out, "README asset JPEG")

    smoke_summary = write_smoke_summary(config, summary_text)
    print(f"# Wrote {smoke_summary}")
    for line in smoke_summary.read_text(encoding="utf-8").splitlines()[:120]:
        print(line)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
