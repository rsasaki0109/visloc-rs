#!/usr/bin/env python3
"""Translate frozen Windows held-out inputs and launch GLUEMAP inside WSL2."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distro", required=True)
    parser.add_argument("--wsl-python", required=True)
    parser.add_argument("--wsl-source-dir", required=True)
    parser.add_argument("--inner-adapter", type=Path, required=True)
    parser.add_argument("--images-path", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--timestamps-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    return parser.parse_args()


def wsl_path(path: Path, distro: str) -> str:
    completed = subprocess.run(
        ["wsl.exe", "-d", distro, "--", "wslpath", "-a", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = completed.stdout.strip()
    if not rendered.startswith("/"):
        raise ValueError(f"wslpath returned an invalid path: {rendered!r}")
    return rendered


def parse_peak_rss_bytes(path: Path) -> int | None:
    if not path.is_file():
        return None
    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*([0-9]+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    return int(match.group(1)) * 1024 if match else None


def main() -> int:
    args = parse_args()
    for path in (
        args.inner_adapter,
        args.images_path,
        args.calibration_path,
        args.timestamps_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_path / "wsl_time_verbose.txt"
    command = [
        "wsl.exe",
        "-d",
        args.distro,
        "--",
        "/usr/bin/time",
        "-v",
        "-o",
        wsl_path(metrics_path, args.distro),
        args.wsl_python,
        wsl_path(args.inner_adapter, args.distro),
        "--source-dir",
        args.wsl_source_dir,
        "--images-path",
        wsl_path(args.images_path, args.distro),
        "--calibration-path",
        wsl_path(args.calibration_path, args.distro),
        "--timestamps-path",
        wsl_path(args.timestamps_path, args.distro),
        "--output-path",
        wsl_path(args.output_path, args.distro),
        "--expected-frames",
        str(args.expected_frames),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    result_path = args.output_path / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["wsl_command"] = command
    result["peak_process_tree_rss_bytes"] = parse_peak_rss_bytes(metrics_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
