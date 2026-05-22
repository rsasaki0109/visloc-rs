"""Filesystem and process helpers for the KITTI revisit smoke runner."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def expand_run_paths(args: object) -> None:
    args.start_dir = args.start_dir.expanduser()
    args.revisit_dir = args.revisit_dir.expanduser()
    args.out_dir = args.out_dir.expanduser()
    if args.readme_asset_out is not None:
        args.readme_asset_out = args.readme_asset_out.expanduser()


def run_command(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def require_path(path: Path, description: str) -> None:
    if path.is_dir() or path.is_file():
        return
    raise FileNotFoundError(f"missing {description}: {path}")


def remove_output_dir(path: Path, repo_root: Path) -> None:
    resolved = path.resolve()
    root = repo_root.resolve()
    if resolved == root or resolved == Path(resolved.anchor):
        raise ValueError(f"refusing to remove unsafe output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def read_summary(summary_path: Path) -> str:
    return summary_path.read_text(encoding="utf-8")
