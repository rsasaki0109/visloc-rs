"""Command builders and summary helpers for the KITTI revisit smoke runner."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from kitti_revisit_smoke_lib import RevisitExpectations


@dataclass(frozen=True)
class SmokeRunConfig:
    start_dir: Path
    revisit_dir: Path
    out_dir: Path
    start_frames: int
    revisit_start_frame: int
    revisit_frames: int
    workers: int
    frontend: str
    max_features: int
    min_matches: int
    min_inliers: int
    min_inlier_ratio: float
    max_mean_sampson_error: float
    cargo_profile: str
    readme_asset_out: Path | None
    readme_headline_gate: bool
    expectations: RevisitExpectations


def config_from_args(args: object) -> SmokeRunConfig:
    return SmokeRunConfig(
        start_dir=args.start_dir,
        revisit_dir=args.revisit_dir,
        out_dir=args.out_dir,
        start_frames=args.start_frames,
        revisit_start_frame=args.revisit_start_frame,
        revisit_frames=args.revisit_frames,
        workers=args.workers,
        frontend=args.frontend,
        max_features=args.max_features,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        max_mean_sampson_error=args.max_mean_sampson_error,
        cargo_profile=args.cargo_profile,
        readme_asset_out=args.readme_asset_out,
        readme_headline_gate=args.readme_headline_gate,
        expectations=RevisitExpectations(
            min_candidates=args.expect_min_candidates,
            strongest_from=args.expect_strongest_from,
            strongest_to=args.expect_strongest_to,
            min_strongest_inliers=args.expect_min_strongest_inliers,
            min_strongest_ratio=args.expect_min_strongest_ratio,
        ),
    )


def fetch_command(
    fetch_script: Path,
    *,
    out_dir: Path,
    max_frames: int,
    workers: int,
    start_frame: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(fetch_script),
        "--stride",
        "1",
    ]
    if start_frame is not None:
        cmd.extend(["--start-frame", str(start_frame)])
    cmd.extend(
        [
            "--max-frames",
            str(max_frames),
            "--workers",
            str(workers),
            "--skip-existing",
            "--cameras",
            "image_0",
            "--out-dir",
            str(out_dir),
        ]
    )
    return cmd


def cargo_demo_command(config: SmokeRunConfig) -> list[str]:
    cmd = ["cargo", "run"]
    if config.cargo_profile == "release":
        cmd.append("--release")
    cmd.extend(
        [
            "--features",
            "image-io",
            "--example",
            "kitti_revisit_scanner_demo",
            "--",
            "--segment-a",
            str(config.start_dir / "image_0"),
            "--calib-a",
            str(config.start_dir / "calib.txt"),
            "--segment-b",
            str(config.revisit_dir / "image_0"),
            "--calib-b",
            str(config.revisit_dir / "calib.txt"),
            "--projection",
            "P0",
            "--frontend",
            config.frontend,
            "--max-features",
            str(config.max_features),
            "--min-matches",
            str(config.min_matches),
            "--min-inliers",
            str(config.min_inliers),
            "--min-inlier-ratio",
            str(config.min_inlier_ratio),
            "--max-mean-sampson-error",
            str(config.max_mean_sampson_error),
            "--out-dir",
            str(config.out_dir),
        ]
    )
    return cmd


def render_asset_command(render_script: Path, config: SmokeRunConfig) -> list[str]:
    if config.readme_asset_out is None:
        raise ValueError("readme_asset_out is required to render the README asset")
    return [
        sys.executable,
        str(render_script),
        str(config.out_dir),
        "--out",
        str(config.readme_asset_out),
    ]


def smoke_summary_text(config: SmokeRunConfig, summary_text: str) -> str:
    lines = [
        "# KITTI deep revisit scanner smoke summary",
        f"start_dir={config.start_dir}",
        f"revisit_dir={config.revisit_dir}",
        f"out_dir={config.out_dir}",
        f"start_frames={config.start_frames}",
        f"revisit_start_frame={config.revisit_start_frame}",
        f"revisit_frames={config.revisit_frames}",
        f"frontend={config.frontend}",
        f"max_features={config.max_features}",
        f"min_matches={config.min_matches}",
        f"min_inliers={config.min_inliers}",
        f"min_inlier_ratio={config.min_inlier_ratio}",
        f"max_mean_sampson_error={config.max_mean_sampson_error}",
        f"report={config.out_dir / 'index.html'}",
        f"candidates_csv={config.out_dir / 'candidates.csv'}",
    ]
    if config.readme_asset_out is not None:
        lines.append(f"readme_asset={config.readme_asset_out}")
    if config.readme_headline_gate:
        lines.append("readme_headline_gate=true")
    if config.expectations.min_candidates is not None:
        lines.append(f"expect_min_candidates={config.expectations.min_candidates}")
    if config.expectations.strongest_from is not None:
        lines.append(f"expect_strongest_from={config.expectations.strongest_from}")
    if config.expectations.strongest_to is not None:
        lines.append(f"expect_strongest_to={config.expectations.strongest_to}")
    if config.expectations.min_strongest_inliers is not None:
        lines.append(
            f"expect_min_strongest_inliers={config.expectations.min_strongest_inliers}"
        )
    if config.expectations.min_strongest_ratio is not None:
        lines.append(
            f"expect_min_strongest_ratio={config.expectations.min_strongest_ratio}"
        )
    lines.extend(["", summary_text.rstrip(), ""])
    return "\n".join(lines)


def write_smoke_summary(
    config: SmokeRunConfig,
    summary_text: str,
    *,
    filename: str = "deep_revisit_smoke_summary.txt",
) -> Path:
    summary_file = config.out_dir / filename
    summary_file.write_text(smoke_summary_text(config, summary_text), encoding="utf-8")
    return summary_file
