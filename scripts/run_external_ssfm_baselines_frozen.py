#!/usr/bin/env python3
"""Run or explicitly DNF frozen external SSfM adapters before GT is available."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from benchmark_process_metrics import run_monitored
from ssfm_external_baseline_evidence import (
    EXTERNAL_ENGINES,
    sha256,
    validate_external_baseline_manifest,
)


ALLOWED_DNF_SETUP_STATUSES = {
    "source_unavailable",
    "install_failed",
    "unsupported",
    "resource_incompatible",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--external-protocol", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--setup-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_hash(path: Path) -> tuple[dict, str]:
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def render_command(template: list[str], replacements: dict[str, str]) -> list[str]:
    command = []
    for token in template:
        rendered = token
        for key, value in replacements.items():
            rendered = rendered.replace("{" + key + "}", value)
        command.append(rendered)
    return command


def max_optional(*values: float | int | None) -> float | int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def dnf_cell(engine: str, setup: dict) -> dict:
    status = setup.get("status")
    if status not in ALLOWED_DNF_SETUP_STATUSES:
        raise ValueError(
            f"{engine} setup status {status!r} is not an evidence-backed DNF"
        )
    attempt = setup.get("attempt")
    if not isinstance(attempt, dict):
        raise ValueError(f"{engine} unavailable setup lacks attempt evidence")
    return {
        "status": "dnf",
        "reason": setup.get("reason") or f"{engine} setup is not ready",
        "source_revision": setup.get("source_revision")
        or setup.get("source_identity"),
        "source_tree_sha256": setup.get("source_tree_sha256"),
        "attempt": attempt,
        "trajectory": None,
    }


def run_ready_engine(
    engine: str,
    setup: dict,
    prepared_dir: Path,
    out_dir: Path,
    expected_frames: int,
    poll_seconds: float,
) -> dict:
    adapter = setup.get("adapter", {})
    adapter_path = Path(adapter.get("path", ""))
    if not adapter_path.is_file():
        raise FileNotFoundError(f"{engine} adapter: {adapter_path}")
    if sha256(adapter_path) != adapter.get("sha256"):
        raise ValueError(f"{engine} adapter hash mismatch")
    for dependency in adapter.get("dependencies", []):
        dependency_path = Path(dependency.get("path", ""))
        if not dependency_path.is_file():
            raise FileNotFoundError(f"{engine} adapter dependency: {dependency_path}")
        if sha256(dependency_path) != dependency.get("sha256"):
            raise ValueError(f"{engine} adapter dependency hash mismatch")
    template = adapter.get("command_template")
    if not isinstance(template, list) or not template:
        raise ValueError(f"{engine} adapter command template is empty")

    engine_out = out_dir / engine
    engine_out.mkdir()
    command = render_command(
        template,
        {
            "adapter": str(adapter_path.resolve()),
            "images_path": str((prepared_dir / "rect" / "image_0").resolve()),
            "calibration_path": str((prepared_dir / "rect" / "calib.txt").resolve()),
            "timestamps_path": str((prepared_dir / "rect" / "timestamps.txt").resolve()),
            "output_path": str(engine_out.resolve()),
            "expected_frames": str(expected_frames),
        },
    )
    stage = run_monitored(
        command,
        out_dir / "logs" / f"{engine}.log",
        cwd=adapter_path.parent,
        poll_seconds=poll_seconds,
    )
    attempt = {"command": command, **stage}
    result_path = engine_out / "result.json"
    trajectory_path = engine_out / "trajectory.tum"
    if stage["returncode"] != 0:
        return {
            "status": "dnf",
            "reason": f"official {engine} adapter returned {stage['returncode']}",
            "source_revision": setup["source_revision"],
            "source_tree_sha256": setup.get("source_tree_sha256"),
            "attempt": attempt,
            "trajectory": None,
        }
    if not result_path.is_file() or not trajectory_path.is_file():
        return {
            "status": "dnf",
            "reason": "adapter exited successfully without standardized result/trajectory",
            "source_revision": setup["source_revision"],
            "source_tree_sha256": setup.get("source_tree_sha256"),
            "attempt": attempt,
            "trajectory": None,
        }

    result = read_json(result_path)
    registered = int(result["registered_images"])
    if registered <= 0 or registered > expected_frames:
        raise ValueError(f"invalid {engine} registered image count: {registered}")
    adapter_peak_rss_value = result.get("peak_process_tree_rss_bytes")
    adapter_peak_rss = (
        int(adapter_peak_rss_value) if adapter_peak_rss_value is not None else 0
    )
    adapter_peak_gpu = result.get("peak_global_gpu_memory_mib")
    return {
        "status": "success",
        "source_revision": setup["source_revision"],
        "source_tree_sha256": setup.get("source_tree_sha256"),
        "attempt": attempt,
        "trajectory": {
            "path": str(trajectory_path.resolve()),
            "sha256": sha256(trajectory_path),
        },
        "registered_images": registered,
        "registration_rate": registered / expected_frames,
        "points3d": result.get("points3d"),
        "mean_reprojection_px": result.get("mean_reprojection_px"),
        "total_wall_seconds": stage["wall_seconds"],
        "peak_process_tree_rss_bytes": max(
            int(stage["peak_process_tree_rss_bytes"]), adapter_peak_rss
        ),
        "peak_global_gpu_memory_mib": max_optional(
            stage.get("peak_global_gpu_memory_mib"),
            adapter_peak_gpu,
        ),
        "adapter_result": {
            "path": str(result_path.resolve()),
            "sha256": sha256(result_path),
        },
    }


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    heldout, heldout_sha256 = protocol_hash(args.protocol)
    external, external_sha256 = protocol_hash(args.external_protocol)
    if args.sequence not in heldout["selection"]["held_out_sequences"]:
        raise ValueError(f"not a frozen held-out sequence: {args.sequence}")
    if external["heldout_protocol"]["sha256"] != heldout_sha256:
        raise ValueError("external protocol does not bind this held-out protocol")
    prepared_path = args.prepared_dir / "manifest.json"
    prepared = read_json(prepared_path)
    if prepared["protocol_sha256"] != heldout_sha256:
        raise ValueError("prepared input protocol mismatch")
    if prepared.get("ground_truth_read") is not False:
        raise ValueError("prepared input does not prove GT isolation")
    if prepared.get("sequence") != args.sequence:
        raise ValueError("prepared input sequence mismatch")
    expected_frames = int(prepared["expected_frames"])

    setup = read_json(args.setup_manifest)
    if setup.get("external_protocol_sha256") != external_sha256:
        raise ValueError("setup/external protocol hash mismatch")
    if set(setup.get("engines", {})) != set(EXTERNAL_ENGINES):
        raise ValueError("setup manifest engine set mismatch")

    args.out_dir.mkdir(parents=True)
    (args.out_dir / "logs").mkdir()
    results = {}
    started_utc = timestamp()
    for engine in EXTERNAL_ENGINES:
        engine_setup = setup["engines"][engine]
        if engine_setup.get("status") == "ready":
            results[engine] = run_ready_engine(
                engine,
                engine_setup,
                args.prepared_dir,
                args.out_dir,
                expected_frames,
                args.poll_seconds,
            )
        else:
            results[engine] = dnf_cell(engine, engine_setup)

    output = {
        "schema_version": 1,
        "status": "complete",
        "sequence": args.sequence,
        "heldout_protocol_sha256": heldout_sha256,
        "external_protocol_sha256": external_sha256,
        "setup_manifest": {
            "path": str(args.setup_manifest.resolve()),
            "sha256": sha256(args.setup_manifest),
        },
        "prepared_manifest": {
            "path": str(prepared_path.resolve()),
            "sha256": sha256(prepared_path),
        },
        "ground_truth_read": False,
        "all_engine_processes_exited": True,
        "started_utc": started_utc,
        "finished_utc": timestamp(),
        "host": platform.platform(),
        "results": results,
    }
    validate_external_baseline_manifest(
        output,
        sequence=args.sequence,
        heldout_protocol_sha256=heldout_sha256,
        external_protocol_sha256=external_sha256,
        manifest_dir=args.out_dir,
    )
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
