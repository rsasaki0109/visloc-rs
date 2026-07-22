#!/usr/bin/env python3
"""Evaluate all completed held-out SSfM engines and emit one normalized cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ssfm_external_baseline_evidence import validate_external_baseline_manifest


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--hierarchical-dir", type=Path, required=True)
    parser.add_argument("--colmap-dir", type=Path, required=True)
    parser.add_argument("--external-protocol", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-csv", type=Path, required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reported_mean_reprojection(log: Path) -> float | None:
    if not log.is_file():
        return None
    match = re.search(
        r"mean reproj\s+([0-9]+(?:\.[0-9]+)?)\s+px",
        log.read_text(encoding="utf-8", errors="replace"),
    )
    return float(match.group(1)) if match else None


def evaluate(
    ground_truth: Path, trajectories: dict[str, Path], output: Path, common: bool
) -> dict[str, dict]:
    command = [
        sys.executable,
        str(REPO / "scripts" / "evaluate_euroc_trajectory.py"),
        "--ground-truth-csv",
        str(ground_truth),
    ]
    for trajectory in trajectories.values():
        command.extend(["--trajectory", str(trajectory)])
    command.extend(["--tum-time-unit", "s"])
    if common:
        command.append("--common-estimate-timestamps")
    command.extend(["--out-json", str(output)])
    subprocess.run(command, cwd=REPO, check=True)
    runs = read_json(output)["runs"]
    if len(runs) != len(trajectories):
        raise RuntimeError("evaluator result count mismatch")
    return dict(zip(trajectories, runs))


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    external_protocol = read_json(args.external_protocol)
    external_protocol_sha256 = sha256(args.external_protocol)
    if external_protocol["heldout_protocol"]["sha256"] != protocol_sha256:
        raise ValueError("external protocol does not bind held-out protocol")
    if args.sequence not in protocol["selection"]["held_out_sequences"]:
        raise ValueError(f"not a frozen held-out sequence: {args.sequence}")
    if not args.ground_truth_csv.is_file():
        raise FileNotFoundError(args.ground_truth_csv)

    prepared_path = args.prepared_dir / "manifest.json"
    hierarchical_path = args.hierarchical_dir / "manifest.json"
    colmap_path = args.colmap_dir / "manifest.json"
    external_path = args.external_dir / "manifest.json"
    ground_truth_manifest_path = args.ground_truth_manifest
    prepared = read_json(prepared_path)
    hierarchical = read_json(hierarchical_path)
    colmap = read_json(colmap_path)
    external = read_json(external_path)
    ground_truth_manifest = read_json(ground_truth_manifest_path)
    if prepared["protocol_sha256"] != protocol_sha256:
        raise ValueError("prepared input protocol mismatch")
    if colmap["protocol_sha256"] != protocol_sha256:
        raise ValueError("COLMAP protocol mismatch")
    external_results = validate_external_baseline_manifest(
        external,
        sequence=args.sequence,
        heldout_protocol_sha256=protocol_sha256,
        external_protocol_sha256=external_protocol_sha256,
        manifest_dir=args.external_dir,
    )
    if ground_truth_manifest["protocol_sha256"] != protocol_sha256:
        raise ValueError("ground-truth materializer protocol mismatch")
    if any(
        manifest.get("sequence", args.sequence) != args.sequence
        for manifest in (prepared, colmap, ground_truth_manifest)
    ):
        raise ValueError("sequence mismatch")
    if not ground_truth_manifest.get(
        "ground_truth_first_read_after_all_timed_engines_exited"
    ):
        raise ValueError("ground truth was not materialized after engine exit")
    ground_truth_evidence = ground_truth_manifest["ground_truth"]
    if Path(ground_truth_evidence["path"]).resolve() != args.ground_truth_csv.resolve():
        raise ValueError("ground-truth materializer path mismatch")
    if ground_truth_evidence["sha256"] != sha256(args.ground_truth_csv):
        raise ValueError("ground-truth materializer hash mismatch")
    engine_evidence = ground_truth_manifest["engine_exit_evidence"]
    if engine_evidence["hierarchical"]["sha256"] != sha256(hierarchical_path):
        raise ValueError("hierarchical exit evidence changed after GT materialization")
    if engine_evidence["colmap"]["sha256"] != sha256(colmap_path):
        raise ValueError("COLMAP exit evidence changed after GT materialization")
    if engine_evidence["external"]["sha256"] != sha256(external_path):
        raise ValueError("external exit evidence changed after GT materialization")
    policy = protocol["policy"]
    if hierarchical["config_id"] != policy["config_id"]:
        raise ValueError("hierarchical policy mismatch")
    if hierarchical["build_git_revision"] != policy["source_revision"]:
        raise ValueError("hierarchical build revision mismatch")
    if Path(hierarchical["protocol"]["features_dir"]).resolve() != (
        args.prepared_dir / "features"
    ).resolve():
        raise ValueError("hierarchical feature input mismatch")
    if Path(hierarchical["protocol"]["timestamps"]).resolve() != (
        args.prepared_dir / "rect" / "timestamps.txt"
    ).resolve():
        raise ValueError("hierarchical timestamp input mismatch")
    if hierarchical["protocol"].get("ground_truth_read") is not False:
        raise ValueError("hierarchical runner read GT before suite finalization")
    if colmap.get("ground_truth_read") is not False:
        raise ValueError("COLMAP runner read GT before suite finalization")

    expected_frames = int(prepared["expected_frames"])
    if hierarchical["protocol"]["expected_frames"] != expected_frames:
        raise ValueError("hierarchical expected frame mismatch")
    if colmap["expected_frames"] != expected_frames:
        raise ValueError("COLMAP expected frame mismatch")

    trajectories = {}
    results = {}
    hierarchical_trajectory = args.hierarchical_dir / "trajectory.tum"
    hierarchical_success = (
        hierarchical["mapper"]["returncode"] == 0
        and hierarchical_trajectory.is_file()
    )
    if hierarchical_success:
        trajectories["visloc_hierarchical"] = hierarchical_trajectory
        results["visloc_hierarchical"] = {
            "status": "success",
            "registered_images": hierarchical["mapper"].get("registered_images"),
            "registration_rate": hierarchical["mapper"].get("registered_images", 0)
            / expected_frames,
            "points3d": hierarchical["mapper"].get("points3d"),
            "mean_reprojection_px": reported_mean_reprojection(
                args.hierarchical_dir / "mapping.log"
            ),
            "mapping_wall_seconds": hierarchical["mapper"]["wall_seconds"],
            "frontend_wall_seconds": prepared["stage_seconds"]["superpoint"],
            "total_wall_seconds": hierarchical["mapper"]["wall_seconds"]
            + prepared["stage_seconds"]["superpoint"],
            "peak_mapper_rss_bytes": hierarchical["mapper"][
                "peak_process_tree_rss_bytes"
            ],
            "peak_process_tree_rss_bytes": max(
                hierarchical["mapper"]["peak_process_tree_rss_bytes"],
                prepared["stages"]["superpoint"]["peak_process_tree_rss_bytes"],
            ),
            "peak_global_gpu_memory_mib": prepared["stages"]["superpoint"][
                "peak_global_gpu_memory_mib"
            ],
            "resource_poll_seconds": {
                "frontend": prepared["stages"]["superpoint"][
                    "resource_poll_seconds"
                ],
                "mapper": hierarchical["mapper"]["resource_poll_seconds"],
            },
        }
    else:
        results["visloc_hierarchical"] = {
            "status": "dnf",
            "reason": "mapper failed, completeness gate failed, or trajectory missing",
            "registered_images": hierarchical["mapper"].get("registered_images", 0),
            "registration_rate": hierarchical["mapper"].get("registered_images", 0)
            / expected_frames,
        }

    for engine in ("incremental", "global"):
        source = colmap["results"][engine]
        trajectory = args.colmap_dir / f"{engine}.tum"
        if source["status"] == "success" and trajectory.is_file():
            name = f"colmap_{engine}"
            trajectories[name] = trajectory
            results[name] = {
                **source,
                "registration_rate": source["registered_images"] / expected_frames,
            }
        else:
            results[f"colmap_{engine}"] = {
                **source,
                "status": "dnf",
                "registration_rate": source.get("registered_images", 0)
                / expected_frames,
            }

    for engine, source in external_results.items():
        if source["status"] == "success":
            trajectory = Path(source["trajectory"]["path"])
            trajectories[engine] = trajectory
            results[engine] = source
        else:
            results[engine] = source

    args.out_dir.mkdir(parents=True)
    if trajectories:
        individual = evaluate(
            args.ground_truth_csv,
            trajectories,
            args.out_dir / "evaluation.json",
            common=False,
        )
        common = evaluate(
            args.ground_truth_csv,
            trajectories,
            args.out_dir / "evaluation_common_frames.json",
            common=True,
        )
        for engine in trajectories:
            results[engine]["evaluation"] = individual[engine]
            results[engine]["common_frame_evaluation"] = common[engine]

    output = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "sequence": args.sequence,
        "finalized_utc": timestamp(),
        "ground_truth_first_read_after_all_timed_engines_exited": True,
        "ground_truth_read": True,
        "expected_frames": expected_frames,
        "input_manifests": {
            "prepared": {"path": str(prepared_path.resolve()), "sha256": sha256(prepared_path)},
            "hierarchical": {
                "path": str(hierarchical_path.resolve()),
                "sha256": sha256(hierarchical_path),
            },
            "colmap": {"path": str(colmap_path.resolve()), "sha256": sha256(colmap_path)},
            "external": {
                "path": str(external_path.resolve()),
                "sha256": sha256(external_path),
            },
            "ground_truth_materializer": {
                "path": str(ground_truth_manifest_path.resolve()),
                "sha256": sha256(ground_truth_manifest_path),
            },
        },
        "ground_truth": {
            "path": str(args.ground_truth_csv.resolve()),
            "sha256": sha256(args.ground_truth_csv),
        },
        "results": results,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
