#!/usr/bin/env python3
"""Run the R1e independent MASt3R-SLAM submap scale gate transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PINNED_REVISION = "6717231a2daf55d501a5824bbec43314d4fb77d9"
REPO = Path(__file__).resolve().parents[1]
PROBE_SOURCE = REPO / "examples" / "dpvo_independent_submap_probe.rs"
MIN_INLIER_RATIO = 0.60
PASS_PATTERN = re.compile(
    r"same_side_scale_transfer_status=pass "
    r"old_matches=(?P<old_matches>\d+) old_inliers=(?P<old_inliers>\d+) "
    r"old_metric_per_local=(?P<old_scale>\S+) "
    r"new_matches=(?P<new_matches>\d+) new_inliers=(?P<new_inliers>\d+) "
    r"new_metric_per_local=(?P<new_scale>\S+) "
    r"new_from_old_scale=(?P<transfer_scale>\S+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--lightglue-dir", type=Path, required=True)
    parser.add_argument("--probe-exe", type=Path, required=True)
    parser.add_argument("--probe-build-revision", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verify_evidence(item: dict, label: str) -> Path:
    path = Path(item["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256(path)
    if actual != item["sha256"]:
        raise ValueError(f"{label} hash mismatch: {actual}")
    return path


def snapshot_directory(path: Path) -> list[dict]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"evidence directory is empty: {path}")
    return [evidence(item) for item in files]


def changed_evidence(items: list[dict]) -> list[str]:
    changed = []
    for item in items:
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            changed.append(str(path))
    return changed


def normalized_source(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def validate_probe_source_revision(revision: str) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("probe build revision must be a full lowercase Git SHA-1")
    committed = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{revision}:examples/dpvo_independent_submap_probe.rs"]
    )
    current = PROBE_SOURCE.read_bytes()
    if hashlib.sha256(normalized_source(committed)).digest() != hashlib.sha256(
        normalized_source(current)
    ).digest():
        raise ValueError("probe source does not match the declared build revision")
    return {
        "build_revision": revision,
        "source": str(PROBE_SOURCE.resolve()),
        "source_sha256": hashlib.sha256(current).hexdigest(),
    }


def path_for_probe(path: Path, probe_exe: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt" and probe_exe.suffix.lower() == ".exe":
        return subprocess.check_output(
            ["wslpath", "-w", str(resolved)], text=True
        ).strip()
    return str(resolved)


def validate_export_manifest(path: Path) -> tuple[dict, Path, Path, Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported R1e export manifest schema")
    if manifest.get("official_revision") != PINNED_REVISION:
        raise ValueError("R1e export is not from the pinned MASt3R-SLAM revision")
    if manifest.get("execution_mode") != "independent_runs":
        raise ValueError("R1e export did not execute independent submap runs")
    if manifest.get("independent_process_per_side") is not True:
        raise ValueError("R1e export lacks independent-process evidence")
    radius = manifest.get("radius")
    if not isinstance(radius, int) or isinstance(radius, bool) or radius <= 0:
        raise ValueError("R1e export has invalid radius")
    frozen_inputs = manifest.get("frozen_inputs")
    if not isinstance(frozen_inputs, list) or not frozen_inputs:
        raise ValueError("R1e export lacks frozen input evidence")
    for index, item in enumerate(frozen_inputs):
        verify_evidence(item, f"frozen R1e input {index}")

    sides = manifest.get("sides")
    if not isinstance(sides, dict) or set(sides) != {"old", "new"}:
        raise ValueError("R1e export must contain exactly old/new sides")
    point_paths = []
    anchors = []
    for side in ("old", "new"):
        record = sides[side]
        if record.get("local_anchor_index") != radius:
            raise ValueError(f"{side} local anchor does not match radius")
        anchor = record.get("anchor_arrival")
        arrivals = record.get("arrivals")
        if not isinstance(anchor, int) or not isinstance(arrivals, list):
            raise ValueError(f"{side} lacks anchor/arrival evidence")
        expected_arrivals = list(range(anchor - radius, anchor + radius + 1))
        if arrivals != expected_arrivals:
            raise ValueError(f"{side} arrivals are not the frozen contiguous window")
        if record.get("command") is None or record.get("run_log") is None:
            raise ValueError(f"{side} lacks independent run command/log")
        verify_evidence(record["run_log"], f"{side} run log")
        verify_evidence(
            {"path": record["state"], "sha256": record["state_sha256"]},
            f"{side} optimized state",
        )
        point_paths.append(
            verify_evidence(
                {
                    "path": record["anchor_points"],
                    "sha256": record["anchor_points_sha256"],
                },
                f"{side} anchor points",
            )
        )
        anchors.append(anchor)
    if anchors[0] >= anchors[1]:
        raise ValueError("R1e old anchor must precede new anchor")
    descriptor_manifest = verify_evidence(
        {
            "path": manifest["descriptor_manifest"],
            "sha256": manifest["descriptor_manifest_sha256"],
        },
        "descriptor manifest",
    )
    return manifest, descriptor_manifest.parent, point_paths[0], point_paths[1]


def parse_pass_metrics(log_text: str) -> dict | None:
    matches = list(PASS_PATTERN.finditer(log_text))
    if len(matches) != 1:
        return None
    fields = matches[0].groupdict()
    metrics = {
        key: int(value) if key.endswith(("matches", "inliers")) else float(value)
        for key, value in fields.items()
    }
    if metrics["old_matches"] <= 0 or metrics["new_matches"] <= 0:
        return None
    metrics["old_inlier_ratio"] = metrics["old_inliers"] / metrics["old_matches"]
    metrics["new_inlier_ratio"] = metrics["new_inliers"] / metrics["new_matches"]
    return metrics


def scale_gate_passes(metrics: dict | None) -> bool:
    if metrics is None:
        return False
    return (
        metrics["old_matches"] >= 12
        and metrics["new_matches"] >= 12
        and metrics["old_inliers"] >= 10
        and metrics["new_inliers"] >= 10
        and metrics["old_inliers"] <= metrics["old_matches"]
        and metrics["new_inliers"] <= metrics["new_matches"]
        and metrics["old_inlier_ratio"] >= MIN_INLIER_RATIO
        and metrics["new_inlier_ratio"] >= MIN_INLIER_RATIO
        and all(
            math.isfinite(metrics[key]) and metrics[key] > 0.0
            for key in ("old_scale", "new_scale", "transfer_scale")
        )
        and math.isclose(
            metrics["transfer_scale"],
            metrics["old_scale"] / metrics["new_scale"],
            rel_tol=1.0e-6,
            abs_tol=1.0e-9,
        )
    )


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    if not args.probe_exe.is_file():
        raise FileNotFoundError(args.probe_exe)
    probe_source = validate_probe_source_revision(args.probe_build_revision)
    export, descriptor_dir, old_points, new_points = validate_export_manifest(
        args.export_manifest
    )
    frozen = [
        evidence(args.export_manifest),
        evidence(args.probe_exe),
        evidence(PROBE_SOURCE),
    ]
    frozen.extend(export["frozen_inputs"])
    frozen.extend(snapshot_directory(args.lightglue_dir))
    for side in ("old", "new"):
        record = export["sides"][side]
        frozen.extend(
            [
                record["run_log"],
                {"path": record["state"], "sha256": record["state_sha256"]},
                {
                    "path": record["anchor_points"],
                    "sha256": record["anchor_points_sha256"],
                },
            ]
        )

    args.out_dir.mkdir(parents=True)
    log_path = args.out_dir / "probe.log"
    command = [
        str(args.probe_exe.resolve()),
        "--dump-dir",
        path_for_probe(descriptor_dir, args.probe_exe),
        "--lightglue-dir",
        path_for_probe(args.lightglue_dir, args.probe_exe),
        "--learned-old-points",
        path_for_probe(old_points, args.probe_exe),
        "--learned-new-points",
        path_for_probe(new_points, args.probe_exe),
        "--old-anchor",
        str(export["sides"]["old"]["anchor_arrival"]),
        "--new-anchor",
        str(export["sides"]["new"]["anchor_arrival"]),
        "--radius",
        str(export["radius"]),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    changed = changed_evidence(frozen)
    log_text = log_path.read_text(encoding="utf-8")
    metrics = parse_pass_metrics(log_text) if completed.returncode == 0 else None
    gate_passed = scale_gate_passes(metrics) and not changed
    result = {
        "schema_version": 1,
        "status": "success" if completed.returncode == 0 and not changed else "failed",
        "generated_utc": timestamp(),
        "ground_truth_read": False,
        "backend_writeback": False,
        "export_manifest": evidence(args.export_manifest),
        "probe_executable": evidence(args.probe_exe),
        "probe_source": probe_source,
        "lightglue_directory": str(args.lightglue_dir.resolve()),
        "frozen_inputs": frozen,
        "command": command,
        "returncode": completed.returncode,
        "probe_log": evidence(log_path),
        "changed_inputs": changed,
        "same_side_scale_transfer_gate": {
            "passed": gate_passed,
            "minimum_inlier_ratio": MIN_INLIER_RATIO,
            "metrics": metrics,
            "reason": None if gate_passed else "probe rejected or evidence changed",
        },
    }
    write_json_atomic(args.out_dir / "manifest.json", result)
    print(args.out_dir / "manifest.json")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
