#!/usr/bin/env python3
"""Benchmark registry helper.

The registry has two related but deliberately separate records:

* run manifests: machine-readable evidence for one actual command execution.
* README claims: concise public benchmark rows rendered from JSON.

The script is dependency-free so benchmark runners can call it from shell
scripts without adding a Python package install step.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from summarize_euroc_active_observation_sweep import (
    load_latest_runs as load_active_observation_runs,
    missing_expected_runs as missing_active_observation_runs,
    render as render_active_observation_sweep,
)
from summarize_euroc_tight_vio_gate_smoke import (
    load_latest_runs as load_tight_vio_gate_smoke_runs,
    render as render_tight_vio_gate_smoke,
)
from summarize_euroc_covisibility_ab import (
    load_latest_runs as load_covisibility_ab_runs,
    missing_expected_runs as missing_covisibility_ab_runs,
    render as render_covisibility_ab,
)
from summarize_euroc_covisibility_mh05_mitigation import (
    load_latest_runs as load_covisibility_mh05_mitigation_runs,
    missing_expected_runs as missing_covisibility_mh05_mitigation_runs,
    parse_config as parse_covisibility_mh05_mitigation_config,
    render as render_covisibility_mh05_mitigation,
)
from summarize_euroc_covisibility_mh05_boundary_support_gate import (
    load_latest_runs as load_covisibility_mh05_boundary_support_gate_runs,
    missing_expected_runs as missing_covisibility_mh05_boundary_support_gate_runs,
    parse_gate as parse_covisibility_mh05_boundary_support_gate,
    render as render_covisibility_mh05_boundary_support_gate,
)
from summarize_euroc_covisibility_runtime_sweep import (
    load_latest_runs as load_covisibility_runtime_runs,
    missing_expected_runs as missing_covisibility_runtime_runs,
    render as render_covisibility_runtime_sweep,
)
from summarize_euroc_covisibility_window_sweep import (
    load_latest_runs as load_covisibility_window_runs,
    missing_expected_runs as missing_covisibility_window_runs,
    parse_window_cap,
    render as render_covisibility_window_sweep,
)
from summarize_kitti_adaptive_depth_gate_smoke import (
    load_failure_runs as load_kitti_adaptive_depth_gate_smoke_failures,
    load_latest_runs as load_kitti_adaptive_depth_gate_smoke_runs,
    missing_expected_runs as missing_kitti_adaptive_depth_gate_smoke_runs,
    render as render_kitti_adaptive_depth_gate_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_STATUSES = {"success", "dnf", "failure"}
RESULT_KINDS = {"visloc_run", "external_published", "external_rerun", "exploratory"}
CLAIM_SCOPES = {"headline", "supporting", "exploratory", "negative"}
CLAIM_KINDS = {"registered_run", "documented_historical", "external_published", "mixed"}
COMPARISON_VERDICTS = {"win", "near", "behind", "negative", "unsupported"}
SECONDARY_METRIC_PRIORITIES = [
    "sim3",
    "similarity",
    "verified_loops",
    "tracking_success_rate",
    "frames",
    "successes",
    "failures",
    "boundary_support_failures",
    "quality_gate_failures",
    "elapsed_ms_mean",
    "mean_reprojection_after",
]


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run(cmd: list[str], cwd: Path = ROOT) -> str | None:
    cmd = resolve_tool_command(cmd)
    try:
        proc = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def resolve_tool_command(cmd: list[str]) -> list[str]:
    if not cmd:
        return cmd
    if cmd[0] not in {"cargo", "rustc"}:
        return cmd
    tool = shutil.which(cmd[0])
    if tool:
        return [tool, *cmd[1:]]
    suffix = ".exe" if os.name == "nt" else ""
    fallback = Path.home() / ".cargo" / "bin" / f"{cmd[0]}{suffix}"
    if fallback.exists():
        return [str(fallback), *cmd[1:]]
    return cmd


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: Path) -> str | None:
    if not path.exists() or not path.is_dir():
        return None
    h = hashlib.sha256()
    files = sorted(
        (child for child in path.rglob("*") if child.is_file()),
        key=lambda child: child.relative_to(path).as_posix(),
    )
    for child in files:
        rel = child.relative_to(path).as_posix().encode("utf-8")
        file_hash = sha256_file(child)
        if file_hash is None:
            return None
        h.update(rel)
        h.update(b"\0")
        h.update(str(child.stat().st_size).encode("ascii"))
        h.update(b"\0")
        h.update(file_hash.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def sha256_path(path: Path) -> str | None:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return sha256_tree(path)
    return None


def dataset_record(args: argparse.Namespace) -> dict[str, Any]:
    checksum = args.dataset_checksum
    checksum_method = args.dataset_checksum_method
    if args.dataset_path:
        path = Path(args.dataset_path)
        resolved = path if path.is_absolute() else ROOT / path
        if checksum is None:
            checksum = sha256_path(resolved)
            if checksum is not None and checksum_method is None:
                checksum_method = "sha256_tree_v1" if resolved.is_dir() else "sha256_file"
    if checksum is not None and checksum_method is None:
        checksum_method = "user-provided"
    return {
        "name": args.dataset_name,
        "sequence": args.dataset_sequence,
        "version": args.dataset_version,
        "path": args.dataset_path,
        "checksum": checksum,
        "checksum_method": checksum_method,
    }


def parse_jsonish(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_key_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"expected KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"empty key in {raw!r}")
    return key, parse_jsonish(value.strip())


def parse_metric(raw: str, primary_name: str | None) -> dict[str, Any]:
    key, value = parse_key_value(raw)
    unit = None
    if isinstance(value, str) and ":" in value:
        value, unit = value.rsplit(":", 1)
    parsed_value = parse_jsonish(str(value))
    metric = {
        "name": key,
        "value": parsed_value,
        "unit": unit,
        "primary": key == primary_name,
        "implementation": None,
        "source_artifact": None,
    }
    return metric


def parse_named_path(raw: str) -> tuple[str, Path]:
    key, value = parse_key_value(raw)
    return key, Path(str(value))


def command_record(raw: str | None) -> dict[str, Any]:
    if raw:
        try:
            # Windows paths use backslashes heavily. POSIX shlex treats
            # those as escapes and corrupts paths like `C:\Users\...`,
            # so switch to non-POSIX mode for Windows-looking commands.
            argv = shlex.split(raw, posix="\\" not in raw)
            argv = [
                part[1:-1]
                if len(part) >= 2 and part[0] == part[-1] and part[0] in {'"', "'"}
                else part
                for part in argv
            ]
        except ValueError:
            argv = []
        return {"raw": raw, "argv": argv, "cwd": str(ROOT)}
    return {"raw": None, "argv": [], "cwd": str(ROOT)}


def git_record() -> dict[str, Any]:
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


def build_record(features: list[str], profile: str | None, target: str | None) -> dict[str, Any]:
    lock = ROOT / "Cargo.lock"
    return {
        "cargo_lock_sha256": sha256_file(lock),
        "rustc": run(["rustc", "--version"]),
        "cargo": run(["cargo", "--version"]),
        "features": features,
        "profile": profile,
        "target": target,
    }


def hardware_record(note: str | None) -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "gpu": os.environ.get("VISLOC_BENCH_GPU"),
        "notes": note,
    }


def artifact_record(kind: str, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    return {
        "kind": kind,
        "path": str(path),
        "sha256": sha256_file(resolved),
        "exists": resolved.exists(),
    }


def model_record(name: str, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else ROOT / path
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256_file(resolved),
        "exists": resolved.exists(),
        "source": None,
        "license": None,
    }


def capture(args: argparse.Namespace) -> int:
    config_params = dict(parse_key_value(item) for item in args.config)
    metrics = [parse_metric(item, args.primary_metric) for item in args.metric]
    artifacts = [artifact_record(*parse_named_path(item)) for item in args.artifact]
    models = [model_record(*parse_named_path(item)) for item in args.model]
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id or default_run_id(args),
        "created_utc": utc_now(),
        "result_kind": args.result_kind,
        "claim_scope": args.claim_scope,
        "status": args.status,
        "failure_reason": args.failure_reason,
        "benchmark": {
            "id": args.benchmark_id,
            "name": args.benchmark_name or args.benchmark_id,
            "script": args.script,
            "protocol": args.protocol,
            "docs": args.docs,
        },
        "dataset": dataset_record(args),
        "git": git_record(),
        "build": build_record(args.feature, args.profile, args.target),
        "command": command_record(args.command),
        "hardware": hardware_record(args.hardware_note),
        "config": {
            "seed": args.seed,
            "params": config_params,
            "config_files": [artifact_record("config", Path(p)) for p in args.config_file],
        },
        "models": models,
        "metrics": metrics,
        "artifacts": artifacts,
        "notes": args.notes,
    }
    validate_run(manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def default_run_id(args: argparse.Namespace) -> str:
    seq = f"-{args.dataset_sequence}" if args.dataset_sequence else ""
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{args.benchmark_id}{seq}-{stamp}"


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_run(obj: dict[str, Any]) -> None:
    errors: list[str] = []
    require(obj.get("schema_version") == 1, "schema_version must be 1", errors)
    require(bool(obj.get("run_id")), "run_id is required", errors)
    require(obj.get("result_kind") in RESULT_KINDS, "result_kind is invalid", errors)
    require(obj.get("claim_scope") in CLAIM_SCOPES, "claim_scope is invalid", errors)
    require(obj.get("status") in RUN_STATUSES, "status is invalid", errors)
    if obj.get("status") in {"dnf", "failure"}:
        require(bool(obj.get("failure_reason")), "failure_reason is required for DNF/failure", errors)
    for parent, keys in {
        "benchmark": ["id", "name"],
        "dataset": ["name"],
        "git": ["commit", "dirty"],
        "build": ["cargo_lock_sha256", "features"],
        "command": ["raw", "argv", "cwd"],
        "hardware": ["os", "machine"],
        "config": ["seed", "params", "config_files"],
    }.items():
        require(isinstance(obj.get(parent), dict), f"{parent} object is required", errors)
        record = obj.get(parent) if isinstance(obj.get(parent), dict) else {}
        for key in keys:
            require(key in record, f"{parent}.{key} is required", errors)
    require(isinstance(obj.get("metrics"), list), "metrics must be a list", errors)
    require(isinstance(obj.get("artifacts"), list), "artifacts must be a list", errors)
    if obj.get("result_kind") == "visloc_run":
        require(bool(obj.get("git", {}).get("commit")), "visloc_run requires git.commit", errors)
        require(
            bool(obj.get("build", {}).get("cargo_lock_sha256")),
            "visloc_run requires build.cargo_lock_sha256",
            errors,
        )
    if errors:
        raise ValueError("; ".join(errors))


def validate_claims(obj: dict[str, Any]) -> None:
    errors: list[str] = []
    require(obj.get("schema_version") == 1, "claims schema_version must be 1", errors)
    claims = obj.get("claims")
    require(isinstance(claims, list), "claims must be a list", errors)
    if isinstance(claims, list):
        for i, claim in enumerate(claims):
            prefix = f"claims[{i}]"
            require(bool(claim.get("benchmark")), f"{prefix}.benchmark is required", errors)
            require(bool(claim.get("result_markdown")), f"{prefix}.result_markdown is required", errors)
            require(claim.get("claim_kind") in CLAIM_KINDS, f"{prefix}.claim_kind is invalid", errors)
            require(isinstance(claim.get("source_docs"), list), f"{prefix}.source_docs must be a list", errors)
    if errors:
        raise ValueError("; ".join(errors))


def validate_claim_matrix(obj: dict[str, Any]) -> None:
    errors: list[str] = []
    require(obj.get("schema_version") == 1, "claim matrix schema_version must be 1", errors)
    comparisons = obj.get("comparisons")
    require(isinstance(comparisons, list), "comparisons must be a list", errors)
    seen_ids: set[str] = set()
    required = [
        "comparison_id",
        "benchmark",
        "dataset",
        "sequence",
        "sensor_mode",
        "metric",
        "protocol",
        "visloc_result",
        "reference_system",
        "reference_result",
        "verdict",
        "claim_kind",
        "claim_scope",
        "source_docs",
        "evidence_run_ids",
    ]
    if isinstance(comparisons, list):
        for i, comparison in enumerate(comparisons):
            prefix = f"comparisons[{i}]"
            require(isinstance(comparison, dict), f"{prefix} must be an object", errors)
            if not isinstance(comparison, dict):
                continue
            for key in required:
                require(key in comparison, f"{prefix}.{key} is required", errors)
            comparison_id = comparison.get("comparison_id")
            require(bool(comparison_id), f"{prefix}.comparison_id is required", errors)
            if comparison_id:
                require(comparison_id not in seen_ids, f"{prefix}.comparison_id is duplicated", errors)
                seen_ids.add(str(comparison_id))
            require(comparison.get("verdict") in COMPARISON_VERDICTS, f"{prefix}.verdict is invalid", errors)
            require(comparison.get("claim_kind") in CLAIM_KINDS, f"{prefix}.claim_kind is invalid", errors)
            require(comparison.get("claim_scope") in CLAIM_SCOPES, f"{prefix}.claim_scope is invalid", errors)
            require(isinstance(comparison.get("source_docs"), list), f"{prefix}.source_docs must be a list", errors)
            require(
                isinstance(comparison.get("evidence_run_ids"), list),
                f"{prefix}.evidence_run_ids must be a list",
                errors,
            )
    if errors:
        raise ValueError("; ".join(errors))


def iter_json_paths(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(p for p in path.rglob("*.json") if "templates" not in p.parts)
        else:
            out.append(path)
    return sorted(out)


def validate_cmd(args: argparse.Namespace) -> int:
    failed = 0
    for path in iter_json_paths(args.paths):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and "claims" in obj:
                validate_claims(obj)
            elif isinstance(obj, dict) and "comparisons" in obj:
                validate_claim_matrix(obj)
            elif isinstance(obj, dict):
                validate_run(obj)
            else:
                raise ValueError("top-level JSON must be an object")
        except Exception as exc:  # noqa: BLE001 - CLI should report all files.
            failed += 1
            print(f"{path}: {exc}", file=sys.stderr)
    if failed:
        return 1
    print(f"validated {len(iter_json_paths(args.paths))} registry JSON file(s)")
    return 0


def render_claim_table(claims_obj: dict[str, Any]) -> str:
    validate_claims(claims_obj)
    lines = [
        "| Benchmark | Result |",
        "| --- | ---: |",
    ]
    for claim in claims_obj["claims"]:
        benchmark = one_line(claim["benchmark"])
        result = one_line(claim["result_markdown"])
        lines.append(f"| {benchmark} | {result} |")
    return "\n".join(lines) + "\n"


def render_claim_snapshot(claims_obj: dict[str, Any], with_heading: bool) -> str:
    table = render_claim_table(claims_obj)
    if not with_heading:
        return table
    return (
        "# Benchmark Snapshot\n\n"
        "Generated from `benchmarks/registry/readme_claims_v1.json`. "
        "This is the public headline table; registered run evidence, "
        "including exploratory and negative runs, is rendered separately "
        "in `docs/generated/registered_runs.md`.\n\n"
        + table
    )


def render_claim_matrix_table(matrix_obj: dict[str, Any], with_heading: bool) -> str:
    validate_claim_matrix(matrix_obj)
    rows = sorted(
        matrix_obj["comparisons"],
        key=lambda item: (
            str(item.get("dataset", "")),
            str(item.get("sequence", "")),
            str(item.get("reference_system", "")),
            str(item.get("comparison_id", "")),
        ),
    )
    lines = [
        "| comparison | mode / protocol | visloc | reference | verdict | evidence |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        evidence = []
        docs = row.get("source_docs") or []
        if docs:
            evidence.append("docs: " + ", ".join(str(doc) for doc in docs))
        run_ids = row.get("evidence_run_ids") or []
        if run_ids:
            evidence.append("runs: " + ", ".join(str(run_id) for run_id in run_ids))
        if not evidence:
            evidence.append(str(row.get("claim_kind", "")))
        comparison = f"{row['benchmark']}<br>{row['reference_system']}"
        mode = f"{row['sensor_mode']}<br>{row['metric']}<br>{row['protocol']}"
        reference = f"{row['reference_system']} {row['reference_result']}"
        notes = row.get("notes")
        verdict = str(row["verdict"])
        if notes:
            verdict = f"{verdict}<br>{notes}"
        lines.append(
            "| "
            + " | ".join(
                one_line(str(cell))
                for cell in [
                    comparison,
                    mode,
                    row["visloc_result"],
                    reference,
                    verdict,
                    "<br>".join(evidence),
                ]
            )
            + " |"
        )
    text = "\n".join(lines) + "\n"
    if with_heading:
        text = (
            "# Benchmark Claim Matrix\n\n"
            "Generated from `benchmarks/registry/claim_matrix_v1.json`. "
            "This matrix keeps comparison claims scoped by sequence, sensor mode, "
            "metric, protocol, and evidence class. A `behind` verdict is an "
            "explicit non-win, not a marketing claim.\n\n"
            + text
        )
    return text


def one_line(value: str) -> str:
    return " ".join(str(value).replace("\n", " ").split())


def format_metric(metric: dict[str, Any]) -> str:
    value = metric.get("value")
    if isinstance(value, float):
        value_text = f"{value:.6g}"
    else:
        value_text = str(value)
    unit = metric.get("unit") or ""
    unit_text = f" {unit}" if unit else ""
    return f"{metric.get('name')}={value_text}{unit_text}"


def secondary_metric_priority(name: str) -> int:
    for index, needle in enumerate(SECONDARY_METRIC_PRIORITIES):
        if needle == "frames":
            if name in {"frames", "frames_recorded"}:
                return index
            continue
        if needle == "successes":
            if name.endswith("_successes") and "boundary_fallback" not in name:
                return index
            continue
        if needle == "failures":
            if name == "failures" or name.endswith("_ba_failures"):
                return index
            continue
        if needle in name:
            return index
    return len(SECONDARY_METRIC_PRIORITIES) + 1


def render_secondary_metrics(metrics: list[dict[str, Any]], limit: int = 6) -> str:
    secondary = [metric for metric in metrics if not metric.get("primary")]
    selected = [
        metric
        for metric in secondary
        if secondary_metric_priority(str(metric.get("name", ""))) < len(SECONDARY_METRIC_PRIORITIES)
    ]
    if not selected:
        selected = secondary[:limit]
    selected = sorted(
        selected,
        key=lambda metric: secondary_metric_priority(str(metric.get("name", ""))),
    )[:limit]
    return "<br>".join(format_metric(metric) for metric in selected)


def replace_marked_block(readme: Path, table: str) -> None:
    readme.write_text(replace_marked_block_text(readme.read_text(encoding="utf-8"), table), encoding="utf-8")


def replace_marked_block_text(text: str, table: str) -> str:
    start = "<!-- benchmark-registry:start -->"
    end = "<!-- benchmark-registry:end -->"
    if start not in text or end not in text:
        raise ValueError("README is missing benchmark registry markers")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return f"{before}{start}\n{table}{end}{after}"


def render_readme(args: argparse.Namespace) -> int:
    claims = json.loads(Path(args.claims).read_text(encoding="utf-8"))
    table = render_claim_snapshot(claims, with_heading=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_claim_snapshot(claims, with_heading=args.with_heading), encoding="utf-8")
    if args.readme:
        replace_marked_block(Path(args.readme), table)
    if not args.out and not args.readme:
        print(table, end="")
    return 0


def render_runs_table(registry_dir: str, with_heading: bool) -> str:
    rows = []
    for path in iter_json_paths([registry_dir]):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if "claims" in obj:
            continue
        validate_run(obj)
        primary = next((m for m in obj.get("metrics", []) if m.get("primary")), None)
        metric = ""
        if primary:
            metric = format_metric(primary)
        rows.append(
            [
                obj["run_id"],
                obj["benchmark"]["id"],
                obj["dataset"]["name"],
                obj["dataset"].get("sequence") or "",
                obj["result_kind"],
                obj["claim_scope"],
                obj["status"],
                metric,
                render_secondary_metrics(obj.get("metrics", [])),
                obj.get("notes") or "",
            ]
        )
    rows.sort(key=lambda row: (row[1], row[3], row[0]))
    lines = [
        "| run_id | benchmark | dataset | seq | kind | scope | status | primary metric | secondary metrics | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(one_line(str(cell)) for cell in row) + " |")
    text = "\n".join(lines) + "\n"
    if with_heading:
        text = (
            "# Registered Benchmark Runs\n\n"
            "Generated from `benchmarks/registry/runs`. This table is evidence, "
            "not the public headline claim table; use `scope` to distinguish "
            "supporting, exploratory, and negative runs.\n\n"
            + text
        )
    return text


def render_runs(args: argparse.Namespace) -> int:
    text = render_runs_table(args.registry_dir, with_heading=args.with_heading)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def render_claim_matrix(args: argparse.Namespace) -> int:
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    text = render_claim_matrix_table(matrix, with_heading=args.with_heading)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def report_stale(path: Path, command: str) -> None:
    print(f"stale generated benchmark file: {path}", file=sys.stderr)
    print(f"  regenerate with: {command}", file=sys.stderr)


def check_generated(args: argparse.Namespace) -> int:
    claims_path = Path(args.claims)
    readme_path = Path(args.readme)
    snapshot_path = Path(args.benchmark_snapshot)
    registered_runs_path = Path(args.registered_runs)
    claim_matrix_path = Path(args.claim_matrix)
    claim_matrix_out_path = Path(args.claim_matrix_out)
    kitti_adaptive_depth_gate_smoke_path = Path(
        getattr(
            args,
            "kitti_adaptive_depth_gate_smoke",
            "docs/generated/kitti_adaptive_depth_gate_smoke.md",
        )
    )
    active_observation_sweep_path = Path(args.active_observation_sweep)
    tight_vio_gate_smoke_path = Path(
        getattr(
            args,
            "tight_vio_gate_smoke",
            "docs/generated/euroc_tight_vio_gate_smoke.md",
        )
    )
    tight_vio_gate_mh03_1500_path = Path(
        getattr(
            args,
            "tight_vio_gate_mh03_1500",
            "docs/generated/euroc_tight_vio_gate_mh03_1500.md",
        )
    )
    covisibility_runtime_sweep_path = Path(args.covisibility_runtime_sweep)
    covisibility_window_sweep_path = Path(args.covisibility_window_sweep)
    covisibility_window_validation_path = Path(args.covisibility_window_validation)
    covisibility_ab_path = Path(args.covisibility_ab)
    covisibility_mh05_mitigation_path = Path(args.covisibility_mh05_mitigation)
    covisibility_mh05_quality_gate_path = Path(args.covisibility_mh05_quality_gate)
    covisibility_mh05_boundary_support_gate_path = Path(
        args.covisibility_mh05_boundary_support_gate
    )
    covisibility_mh05_boundary_support_gate_sweep_path = Path(
        args.covisibility_mh05_boundary_support_gate_sweep
    )
    registry_dir = args.registry_dir

    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    claim_matrix = json.loads(claim_matrix_path.read_text(encoding="utf-8"))
    table = render_claim_snapshot(claims, with_heading=False)
    expected_readme = replace_marked_block_text(readme_path.read_text(encoding="utf-8"), table)
    expected_snapshot = render_claim_snapshot(claims, with_heading=True)
    expected_registered = render_runs_table(registry_dir, with_heading=True)
    expected_claim_matrix = render_claim_matrix_table(claim_matrix, with_heading=True)
    kitti_adaptive_depth_gate_smoke_args = argparse.Namespace(
        registry_dir=Path(
            getattr(
                args,
                "kitti_adaptive_depth_gate_smoke_registry_dir",
                "benchmarks/registry/runs/kitti",
            )
        ),
        out=kitti_adaptive_depth_gate_smoke_path,
        sequence=getattr(args, "kitti_adaptive_depth_gate_smoke_sequence", "00"),
        max_frames=getattr(args, "kitti_adaptive_depth_gate_smoke_max_frames", 2),
        variant=getattr(args, "kitti_adaptive_depth_gate_smoke_variant", None)
        or ["adaptive", "fixed"],
    )
    kitti_adaptive_depth_gate_smoke_runs = load_kitti_adaptive_depth_gate_smoke_runs(
        kitti_adaptive_depth_gate_smoke_args
    )
    kitti_adaptive_depth_gate_smoke_failures = load_kitti_adaptive_depth_gate_smoke_failures(
        kitti_adaptive_depth_gate_smoke_args
    )
    expected_kitti_adaptive_depth_gate_smoke = render_kitti_adaptive_depth_gate_smoke(
        kitti_adaptive_depth_gate_smoke_args,
        kitti_adaptive_depth_gate_smoke_runs,
        kitti_adaptive_depth_gate_smoke_failures,
    )
    missing_kitti_adaptive_depth_gate_smoke = missing_kitti_adaptive_depth_gate_smoke_runs(
        kitti_adaptive_depth_gate_smoke_args,
        kitti_adaptive_depth_gate_smoke_runs,
    )
    active_args = argparse.Namespace(
        registry_dir=Path(args.active_observation_registry_dir),
        out=active_observation_sweep_path,
        max_frames=args.active_observation_max_frames,
        sequence=args.active_observation_sequence
        or ["MH_01_easy", "MH_03_medium", "MH_05_difficult"],
        active_floor=args.active_observation_floor or [20, 50],
        fallback=args.active_observation_fallback,
    )
    active_observation_runs = load_active_observation_runs(active_args)
    expected_active_observation_sweep = render_active_observation_sweep(
        active_args,
        active_observation_runs,
    )
    missing_active_observation = missing_active_observation_runs(
        active_args,
        active_observation_runs,
    )
    tight_vio_gate_smoke_args = argparse.Namespace(
        registry_dir=Path(
            getattr(
                args,
                "tight_vio_gate_registry_dir",
                "benchmarks/registry/runs/euroc",
            )
        ),
        out=tight_vio_gate_smoke_path,
        max_frames=getattr(args, "tight_vio_gate_max_frames", 400),
        sequence=getattr(args, "tight_vio_gate_sequence", None)
        or ["MH_01_easy", "MH_03_medium", "MH_05_difficult"],
        variant=getattr(args, "tight_vio_gate_variant", None)
        or [
            "baseline",
            "adaptive_velocity",
            "gated_10mps",
            "gated_20mps",
            "velocity_tripwire_1mps",
        ],
    )
    tight_vio_gate_smoke_runs = load_tight_vio_gate_smoke_runs(
        tight_vio_gate_smoke_args,
    )
    expected_tight_vio_gate_smoke = render_tight_vio_gate_smoke(
        tight_vio_gate_smoke_args,
        tight_vio_gate_smoke_runs,
    )
    tight_vio_gate_mh03_1500_args = argparse.Namespace(
        registry_dir=Path(
            getattr(
                args,
                "tight_vio_gate_mh03_1500_registry_dir",
                "benchmarks/registry/runs/euroc",
            )
        ),
        out=tight_vio_gate_mh03_1500_path,
        max_frames=getattr(args, "tight_vio_gate_mh03_1500_max_frames", 1500),
        sequence=getattr(args, "tight_vio_gate_mh03_1500_sequence", None)
        or ["MH_03_medium"],
        variant=getattr(args, "tight_vio_gate_mh03_1500_variant", None)
        or ["baseline", "adaptive_velocity"],
    )
    tight_vio_gate_mh03_1500_runs = load_tight_vio_gate_smoke_runs(
        tight_vio_gate_mh03_1500_args,
    )
    expected_tight_vio_gate_mh03_1500 = render_tight_vio_gate_smoke(
        tight_vio_gate_mh03_1500_args,
        tight_vio_gate_mh03_1500_runs,
    )
    runtime_args = argparse.Namespace(
        registry_dir=Path(args.covisibility_runtime_registry_dir),
        out=covisibility_runtime_sweep_path,
        max_frames=args.covisibility_runtime_max_frames,
        sequence=args.covisibility_runtime_sequence or ["MH_03_medium"],
        landmark_cap=args.covisibility_runtime_landmark_cap or [100, 200, 400],
        neighbor_keyframes=args.covisibility_runtime_neighbor_keyframes,
        boundary_keyframes=args.covisibility_runtime_boundary_keyframes,
        min_active_observations=args.covisibility_runtime_min_active_observations,
        fallback=args.covisibility_runtime_fallback,
        remove_outliers=args.covisibility_runtime_remove_outliers,
        max_outlier_observation_ratio=args.covisibility_runtime_max_outlier_observation_ratio,
        boundary_support_min_optimized_keyframes=(
            args.covisibility_runtime_boundary_support_min_optimized_keyframes
        ),
        boundary_support_min_fixed_keyframes=(
            args.covisibility_runtime_boundary_support_min_fixed_keyframes
        ),
    )
    covisibility_runtime_runs = load_covisibility_runtime_runs(runtime_args)
    expected_covisibility_runtime_sweep = render_covisibility_runtime_sweep(
        runtime_args,
        covisibility_runtime_runs,
    )
    missing_covisibility_runtime = missing_covisibility_runtime_runs(
        runtime_args,
        covisibility_runtime_runs,
    )
    window_args = argparse.Namespace(
        registry_dir=Path(args.covisibility_window_registry_dir),
        out=covisibility_window_sweep_path,
        max_frames=args.covisibility_window_max_frames,
        sequence=args.covisibility_window_sequence
        or ["MH_01_easy", "MH_03_medium", "MH_05_difficult"],
        window_cap=args.covisibility_window_cap or [(5, 5), (10, 10), (15, 15)],
        landmark_cap=args.covisibility_window_landmark_cap,
        min_keyframes=args.covisibility_window_min_keyframes,
        trigger_every=args.covisibility_window_trigger_every,
        min_active_observations=args.covisibility_window_min_active_observations,
        fallback=args.covisibility_window_fallback,
        remove_outliers=args.covisibility_window_remove_outliers,
        max_outlier_observation_ratio=args.covisibility_window_max_outlier_observation_ratio,
        boundary_support_min_optimized_keyframes=(
            args.covisibility_window_boundary_support_min_optimized_keyframes
        ),
        boundary_support_min_fixed_keyframes=(
            args.covisibility_window_boundary_support_min_fixed_keyframes
        ),
    )
    covisibility_window_runs = load_covisibility_window_runs(window_args)
    expected_covisibility_window_sweep = render_covisibility_window_sweep(
        window_args,
        covisibility_window_runs,
    )
    missing_covisibility_window = missing_covisibility_window_runs(
        window_args,
        covisibility_window_runs,
    )
    window_validation_args = argparse.Namespace(
        registry_dir=Path(args.covisibility_window_validation_registry_dir),
        out=covisibility_window_validation_path,
        max_frames=args.covisibility_window_validation_max_frames,
        sequence=args.covisibility_window_validation_sequence
        or ["MH_01_easy", "MH_03_medium", "MH_05_difficult"],
        window_cap=args.covisibility_window_validation_cap or [(5, 5), (10, 10)],
        landmark_cap=args.covisibility_window_validation_landmark_cap,
        min_keyframes=args.covisibility_window_validation_min_keyframes,
        trigger_every=args.covisibility_window_validation_trigger_every,
        min_active_observations=args.covisibility_window_validation_min_active_observations,
        fallback=args.covisibility_window_validation_fallback,
        remove_outliers=args.covisibility_window_validation_remove_outliers,
        max_outlier_observation_ratio=(
            args.covisibility_window_validation_max_outlier_observation_ratio
        ),
        boundary_support_min_optimized_keyframes=(
            args.covisibility_window_validation_boundary_support_min_optimized_keyframes
        ),
        boundary_support_min_fixed_keyframes=(
            args.covisibility_window_validation_boundary_support_min_fixed_keyframes
        ),
    )
    covisibility_window_validation_runs = load_covisibility_window_runs(
        window_validation_args,
    )
    expected_covisibility_window_validation = render_covisibility_window_sweep(
        window_validation_args,
        covisibility_window_validation_runs,
    )
    missing_covisibility_window_validation = missing_covisibility_window_runs(
        window_validation_args,
        covisibility_window_validation_runs,
    )
    ab_args = argparse.Namespace(
        registry_dir=Path(args.covisibility_ab_registry_dir),
        out=covisibility_ab_path,
        max_frames=args.covisibility_ab_max_frames,
        sequence=args.covisibility_ab_sequence
        or ["MH_01_easy", "MH_03_medium", "MH_05_difficult"],
        enabled_neighbor_keyframes=args.covisibility_ab_enabled_neighbor_keyframes,
        enabled_boundary_keyframes=args.covisibility_ab_enabled_boundary_keyframes,
        enabled_min_keyframes=args.covisibility_ab_enabled_min_keyframes,
        enabled_trigger_every=args.covisibility_ab_enabled_trigger_every,
        enabled_landmark_cap=args.covisibility_ab_enabled_landmark_cap,
        enabled_min_active_observations=args.covisibility_ab_enabled_min_active_observations,
        enabled_fallback=args.covisibility_ab_enabled_fallback,
        enabled_remove_outliers=args.covisibility_ab_enabled_remove_outliers,
        enabled_max_outlier_observation_ratio=(
            args.covisibility_ab_enabled_max_outlier_observation_ratio
        ),
        enabled_boundary_support_min_optimized_keyframes=(
            args.covisibility_ab_enabled_boundary_support_min_optimized_keyframes
        ),
        enabled_boundary_support_min_fixed_keyframes=(
            args.covisibility_ab_enabled_boundary_support_min_fixed_keyframes
        ),
    )
    covisibility_ab_runs = load_covisibility_ab_runs(ab_args)
    expected_covisibility_ab = render_covisibility_ab(
        ab_args,
        covisibility_ab_runs,
    )
    missing_covisibility_ab = missing_covisibility_ab_runs(
        ab_args,
        covisibility_ab_runs,
    )
    mh05_mitigation_args = argparse.Namespace(
        registry_dir=Path(args.covisibility_mh05_mitigation_registry_dir),
        out=covisibility_mh05_mitigation_path,
        sequence=args.covisibility_mh05_mitigation_sequence,
        max_frames=args.covisibility_mh05_mitigation_max_frames,
        neighbor_keyframes=args.covisibility_mh05_mitigation_neighbor_keyframes,
        boundary_keyframes=args.covisibility_mh05_mitigation_boundary_keyframes,
        landmark_cap=args.covisibility_mh05_mitigation_landmark_cap,
        min_active_observations=args.covisibility_mh05_mitigation_min_active_observations,
        fallback=args.covisibility_mh05_mitigation_fallback,
        remove_outliers=args.covisibility_mh05_mitigation_remove_outliers,
        max_outlier_observation_ratio=(
            args.covisibility_mh05_mitigation_max_outlier_observation_ratio
        ),
        boundary_support_min_optimized_keyframes=(
            args.covisibility_mh05_mitigation_boundary_support_min_optimized_keyframes
        ),
        boundary_support_min_fixed_keyframes=(
            args.covisibility_mh05_mitigation_boundary_support_min_fixed_keyframes
        ),
        config=args.covisibility_mh05_mitigation_config
        or [
            ("enabled min3/every1", 3, 1),
            ("enabled min6/every3", 6, 3),
            ("enabled min10/every5", 10, 5),
        ],
    )
    covisibility_mh05_mitigation_runs = load_covisibility_mh05_mitigation_runs(
        mh05_mitigation_args,
    )
    expected_covisibility_mh05_mitigation = render_covisibility_mh05_mitigation(
        mh05_mitigation_args,
        covisibility_mh05_mitigation_runs,
    )
    missing_covisibility_mh05_mitigation = missing_covisibility_mh05_mitigation_runs(
        mh05_mitigation_args,
        covisibility_mh05_mitigation_runs,
    )
    mh05_quality_gate_args = argparse.Namespace(
        registry_dir=Path(args.covisibility_mh05_mitigation_registry_dir),
        out=covisibility_mh05_quality_gate_path,
        sequence=args.covisibility_mh05_mitigation_sequence,
        max_frames=args.covisibility_mh05_mitigation_max_frames,
        neighbor_keyframes=args.covisibility_mh05_mitigation_neighbor_keyframes,
        boundary_keyframes=args.covisibility_mh05_mitigation_boundary_keyframes,
        landmark_cap=args.covisibility_mh05_mitigation_landmark_cap,
        min_active_observations=args.covisibility_mh05_mitigation_min_active_observations,
        fallback=args.covisibility_mh05_mitigation_fallback,
        remove_outliers=args.covisibility_mh05_mitigation_remove_outliers,
        max_outlier_observation_ratio=(
            args.covisibility_mh05_quality_gate_max_outlier_observation_ratio
        ),
        boundary_support_min_optimized_keyframes=(
            args.covisibility_mh05_quality_gate_boundary_support_min_optimized_keyframes
        ),
        boundary_support_min_fixed_keyframes=(
            args.covisibility_mh05_quality_gate_boundary_support_min_fixed_keyframes
        ),
        config=args.covisibility_mh05_mitigation_config
        or [
            ("enabled min3/every1", 3, 1),
            ("enabled min6/every3", 6, 3),
            ("enabled min10/every5", 10, 5),
        ],
    )
    covisibility_mh05_quality_gate_runs = load_covisibility_mh05_mitigation_runs(
        mh05_quality_gate_args,
    )
    expected_covisibility_mh05_quality_gate = render_covisibility_mh05_mitigation(
        mh05_quality_gate_args,
        covisibility_mh05_quality_gate_runs,
    )
    missing_covisibility_mh05_quality_gate = missing_covisibility_mh05_mitigation_runs(
        mh05_quality_gate_args,
        covisibility_mh05_quality_gate_runs,
    )
    mh05_boundary_support_gate_args = argparse.Namespace(
        registry_dir=Path(args.covisibility_mh05_mitigation_registry_dir),
        out=covisibility_mh05_boundary_support_gate_path,
        sequence=args.covisibility_mh05_mitigation_sequence,
        max_frames=args.covisibility_mh05_mitigation_max_frames,
        neighbor_keyframes=args.covisibility_mh05_mitigation_neighbor_keyframes,
        boundary_keyframes=args.covisibility_mh05_mitigation_boundary_keyframes,
        landmark_cap=args.covisibility_mh05_mitigation_landmark_cap,
        min_active_observations=args.covisibility_mh05_mitigation_min_active_observations,
        fallback=args.covisibility_mh05_mitigation_fallback,
        remove_outliers=args.covisibility_mh05_mitigation_remove_outliers,
        max_outlier_observation_ratio=(
            args.covisibility_mh05_boundary_support_gate_max_outlier_observation_ratio
        ),
        boundary_support_min_optimized_keyframes=(
            args.covisibility_mh05_boundary_support_gate_min_optimized_keyframes
        ),
        boundary_support_min_fixed_keyframes=(
            args.covisibility_mh05_boundary_support_gate_min_fixed_keyframes
        ),
        config=args.covisibility_mh05_boundary_support_gate_config
        or [
            ("enabled min3/every1 boundary10", 3, 1),
        ],
    )
    covisibility_mh05_boundary_support_gate_runs = (
        load_covisibility_mh05_mitigation_runs(mh05_boundary_support_gate_args)
    )
    expected_covisibility_mh05_boundary_support_gate = (
        render_covisibility_mh05_mitigation(
            mh05_boundary_support_gate_args,
            covisibility_mh05_boundary_support_gate_runs,
        )
    )
    missing_covisibility_mh05_boundary_support_gate = (
        missing_covisibility_mh05_mitigation_runs(
            mh05_boundary_support_gate_args,
            covisibility_mh05_boundary_support_gate_runs,
        )
    )
    mh05_boundary_support_gate_sweep_args = argparse.Namespace(
        registry_dir=Path(args.covisibility_mh05_mitigation_registry_dir),
        out=covisibility_mh05_boundary_support_gate_sweep_path,
        sequence=args.covisibility_mh05_mitigation_sequence,
        max_frames=args.covisibility_mh05_mitigation_max_frames,
        neighbor_keyframes=args.covisibility_mh05_mitigation_neighbor_keyframes,
        boundary_keyframes=args.covisibility_mh05_mitigation_boundary_keyframes,
        landmark_cap=args.covisibility_mh05_mitigation_landmark_cap,
        min_keyframes=args.covisibility_mh05_mitigation_min_keyframes,
        trigger_every=args.covisibility_mh05_mitigation_trigger_every,
        min_active_observations=args.covisibility_mh05_mitigation_min_active_observations,
        fallback=args.covisibility_mh05_mitigation_fallback,
        remove_outliers=args.covisibility_mh05_mitigation_remove_outliers,
        max_outlier_observation_ratio=(
            args.covisibility_mh05_boundary_support_gate_max_outlier_observation_ratio
        ),
        gate=args.covisibility_mh05_boundary_support_gate_sweep_gate
        or [
            ("quality-gate only", "none", 0),
            ("boundary7/2", "7", 2),
            ("boundary10/2", "10", 2),
        ],
    )
    covisibility_mh05_boundary_support_gate_sweep_runs = (
        load_covisibility_mh05_boundary_support_gate_runs(
            mh05_boundary_support_gate_sweep_args
        )
    )
    expected_covisibility_mh05_boundary_support_gate_sweep = (
        render_covisibility_mh05_boundary_support_gate(
            mh05_boundary_support_gate_sweep_args,
            covisibility_mh05_boundary_support_gate_sweep_runs,
        )
    )
    missing_covisibility_mh05_boundary_support_gate_sweep = (
        missing_covisibility_mh05_boundary_support_gate_runs(
            mh05_boundary_support_gate_sweep_args,
            covisibility_mh05_boundary_support_gate_sweep_runs,
        )
    )

    stale = 0
    if readme_path.read_text(encoding="utf-8") != expected_readme:
        report_stale(
            readme_path,
            f"python scripts/benchmark_registry.py render-readme --claims {claims_path} --readme {readme_path}",
        )
        stale = 1
    if not snapshot_path.exists() or snapshot_path.read_text(encoding="utf-8") != expected_snapshot:
        report_stale(
            snapshot_path,
            f"python scripts/benchmark_registry.py render-readme --claims {claims_path} --out {snapshot_path} --with-heading",
        )
        stale = 1
    if not registered_runs_path.exists() or registered_runs_path.read_text(encoding="utf-8") != expected_registered:
        report_stale(
            registered_runs_path,
            f"python scripts/benchmark_registry.py render-runs --registry-dir {registry_dir} --with-heading --out {registered_runs_path}",
        )
        stale = 1
    if not claim_matrix_out_path.exists() or claim_matrix_out_path.read_text(encoding="utf-8") != expected_claim_matrix:
        report_stale(
            claim_matrix_out_path,
            f"python scripts/benchmark_registry.py render-claim-matrix --matrix {claim_matrix_path} --with-heading --out {claim_matrix_out_path}",
        )
        stale = 1
    if (
        not kitti_adaptive_depth_gate_smoke_path.exists()
        or kitti_adaptive_depth_gate_smoke_path.read_text(encoding="utf-8")
        != expected_kitti_adaptive_depth_gate_smoke
    ):
        report_stale(
            kitti_adaptive_depth_gate_smoke_path,
            "python scripts/summarize_kitti_adaptive_depth_gate_smoke.py "
            f"--registry-dir {args.kitti_adaptive_depth_gate_smoke_registry_dir} "
            f"--out {kitti_adaptive_depth_gate_smoke_path}",
        )
        stale = 1
    if missing_kitti_adaptive_depth_gate_smoke:
        print("missing KITTI adaptive depth-gate smoke registry run(s):", file=sys.stderr)
        for variant in missing_kitti_adaptive_depth_gate_smoke:
            print(f"  variant={variant}", file=sys.stderr)
        print(
            "  regenerate with the adaptive and fixed KITTI depth-gate smoke runs",
            file=sys.stderr,
        )
        stale = 1
    if (
        not active_observation_sweep_path.exists()
        or active_observation_sweep_path.read_text(encoding="utf-8")
        != expected_active_observation_sweep
    ):
        report_stale(
            active_observation_sweep_path,
            "python scripts/summarize_euroc_active_observation_sweep.py "
            f"--registry-dir {args.active_observation_registry_dir} "
            f"--out {active_observation_sweep_path}",
        )
        stale = 1
    if missing_active_observation:
        print("missing active-observation sweep registry run(s):", file=sys.stderr)
        for floor, sequence, variant in missing_active_observation:
            print(
                f"  floor={floor} sequence={sequence} variant={variant}",
                file=sys.stderr,
            )
        print(
            "  regenerate with: python scripts/run_euroc_active_observation_sweep.py",
            file=sys.stderr,
        )
        stale = 1
    if (
        not tight_vio_gate_smoke_path.exists()
        or tight_vio_gate_smoke_path.read_text(encoding="utf-8")
        != expected_tight_vio_gate_smoke
    ):
        report_stale(
            tight_vio_gate_smoke_path,
            "python scripts/summarize_euroc_tight_vio_gate_smoke.py "
            f"--registry-dir {tight_vio_gate_smoke_args.registry_dir} "
            f"--out {tight_vio_gate_smoke_path}",
        )
        stale = 1
    if (
        not tight_vio_gate_mh03_1500_path.exists()
        or tight_vio_gate_mh03_1500_path.read_text(encoding="utf-8")
        != expected_tight_vio_gate_mh03_1500
    ):
        report_stale(
            tight_vio_gate_mh03_1500_path,
            "python scripts/summarize_euroc_tight_vio_gate_smoke.py "
            f"--registry-dir {tight_vio_gate_mh03_1500_args.registry_dir} "
            f"--max-frames {tight_vio_gate_mh03_1500_args.max_frames} "
            "--sequence MH_03_medium --variant baseline --variant adaptive_velocity "
            f"--out {tight_vio_gate_mh03_1500_path}",
        )
        stale = 1
    if (
        not covisibility_runtime_sweep_path.exists()
        or covisibility_runtime_sweep_path.read_text(encoding="utf-8")
        != expected_covisibility_runtime_sweep
    ):
        report_stale(
            covisibility_runtime_sweep_path,
            "python scripts/summarize_euroc_covisibility_runtime_sweep.py "
            f"--registry-dir {args.covisibility_runtime_registry_dir} "
            f"--out {covisibility_runtime_sweep_path}",
        )
        stale = 1
    if missing_covisibility_runtime:
        print("missing covisibility-runtime sweep registry run(s):", file=sys.stderr)
        for sequence, cap in missing_covisibility_runtime:
            print(f"  sequence={sequence} landmark_cap={cap}", file=sys.stderr)
        print(
            "  regenerate with: python scripts/run_euroc_covisibility_runtime_sweep.py",
            file=sys.stderr,
        )
        stale = 1
    if (
        not covisibility_window_sweep_path.exists()
        or covisibility_window_sweep_path.read_text(encoding="utf-8")
        != expected_covisibility_window_sweep
    ):
        report_stale(
            covisibility_window_sweep_path,
            "python scripts/summarize_euroc_covisibility_window_sweep.py "
            f"--registry-dir {args.covisibility_window_registry_dir} "
            f"--out {covisibility_window_sweep_path}",
        )
        stale = 1
    if missing_covisibility_window:
        print("missing covisibility-window sweep registry run(s):", file=sys.stderr)
        for sequence, neighbor, boundary in missing_covisibility_window:
            print(
                f"  sequence={sequence} neighbor={neighbor} boundary={boundary}",
                file=sys.stderr,
            )
        print(
            "  regenerate with: python scripts/run_euroc_covisibility_window_sweep.py",
            file=sys.stderr,
        )
        stale = 1
    if (
        not covisibility_window_validation_path.exists()
        or covisibility_window_validation_path.read_text(encoding="utf-8")
        != expected_covisibility_window_validation
    ):
        report_stale(
            covisibility_window_validation_path,
            "python scripts/summarize_euroc_covisibility_window_sweep.py "
            f"--registry-dir {args.covisibility_window_validation_registry_dir} "
            f"--out {covisibility_window_validation_path}",
        )
        stale = 1
    if missing_covisibility_window_validation:
        print("missing covisibility-window validation registry run(s):", file=sys.stderr)
        for sequence, neighbor, boundary in missing_covisibility_window_validation:
            print(
                f"  sequence={sequence} neighbor={neighbor} boundary={boundary}",
                file=sys.stderr,
            )
        print(
            "  regenerate with: python scripts/run_euroc_covisibility_window_sweep.py --max-frames 400",
            file=sys.stderr,
        )
        stale = 1
    if (
        not covisibility_ab_path.exists()
        or covisibility_ab_path.read_text(encoding="utf-8") != expected_covisibility_ab
    ):
        report_stale(
            covisibility_ab_path,
            "python scripts/summarize_euroc_covisibility_ab.py "
            f"--registry-dir {args.covisibility_ab_registry_dir} "
            f"--out {covisibility_ab_path}",
        )
        stale = 1
    if missing_covisibility_ab:
        print("missing covisibility A/B registry run(s):", file=sys.stderr)
        for sequence, variant in missing_covisibility_ab:
            print(f"  sequence={sequence} variant={variant}", file=sys.stderr)
        print(
            "  regenerate with: python scripts/run_euroc_covisibility_local_ba_ab.py --max-frames 400",
            file=sys.stderr,
        )
        stale = 1
    if (
        not covisibility_mh05_mitigation_path.exists()
        or covisibility_mh05_mitigation_path.read_text(encoding="utf-8")
        != expected_covisibility_mh05_mitigation
    ):
        report_stale(
            covisibility_mh05_mitigation_path,
            "python scripts/summarize_euroc_covisibility_mh05_mitigation.py "
            f"--registry-dir {args.covisibility_mh05_mitigation_registry_dir} "
            f"--out {covisibility_mh05_mitigation_path}",
        )
        stale = 1
    if missing_covisibility_mh05_mitigation:
        print("missing covisibility MH_05 mitigation registry run(s):", file=sys.stderr)
        for label in missing_covisibility_mh05_mitigation:
            print(f"  config={label}", file=sys.stderr)
        print(
            "  regenerate with: python scripts/run_euroc_covisibility_local_ba_ab.py --sequence MH_05_difficult --max-frames 400",
            file=sys.stderr,
        )
        stale = 1
    if (
        not covisibility_mh05_quality_gate_path.exists()
        or covisibility_mh05_quality_gate_path.read_text(encoding="utf-8")
        != expected_covisibility_mh05_quality_gate
    ):
        report_stale(
            covisibility_mh05_quality_gate_path,
            "python scripts/summarize_euroc_covisibility_mh05_mitigation.py "
            f"--registry-dir {args.covisibility_mh05_mitigation_registry_dir} "
            f"--out {covisibility_mh05_quality_gate_path} "
            "--max-outlier-observation-ratio "
            f"{args.covisibility_mh05_quality_gate_max_outlier_observation_ratio}",
        )
        stale = 1
    if missing_covisibility_mh05_quality_gate:
        print("missing covisibility MH_05 quality-gate registry run(s):", file=sys.stderr)
        for label in missing_covisibility_mh05_quality_gate:
            print(f"  config={label}", file=sys.stderr)
        print(
            "  regenerate with: python scripts/run_euroc_covisibility_local_ba_ab.py --sequence MH_05_difficult --max-frames 400 --max-outlier-observation-ratio 0.3",
            file=sys.stderr,
        )
        stale = 1
    if (
        not covisibility_mh05_boundary_support_gate_path.exists()
        or covisibility_mh05_boundary_support_gate_path.read_text(encoding="utf-8")
        != expected_covisibility_mh05_boundary_support_gate
    ):
        report_stale(
            covisibility_mh05_boundary_support_gate_path,
            "python scripts/summarize_euroc_covisibility_mh05_mitigation.py "
            f"--registry-dir {args.covisibility_mh05_mitigation_registry_dir} "
            f"--out {covisibility_mh05_boundary_support_gate_path} "
            "--max-outlier-observation-ratio "
            f"{args.covisibility_mh05_boundary_support_gate_max_outlier_observation_ratio} "
            "--boundary-support-min-optimized-keyframes "
            f"{args.covisibility_mh05_boundary_support_gate_min_optimized_keyframes} "
            "--boundary-support-min-fixed-keyframes "
            f"{args.covisibility_mh05_boundary_support_gate_min_fixed_keyframes}",
        )
        stale = 1
    if missing_covisibility_mh05_boundary_support_gate:
        print("missing covisibility MH_05 boundary-support registry run(s):", file=sys.stderr)
        for label in missing_covisibility_mh05_boundary_support_gate:
            print(f"  config={label}", file=sys.stderr)
        print(
            "  regenerate with: python scripts/run_euroc_covisibility_local_ba_ab.py --sequence MH_05_difficult --max-frames 400 --max-outlier-observation-ratio 0.3 --boundary-support-min-optimized-keyframes 10 --boundary-support-min-fixed-keyframes 2",
            file=sys.stderr,
        )
        stale = 1
    if (
        not covisibility_mh05_boundary_support_gate_sweep_path.exists()
        or covisibility_mh05_boundary_support_gate_sweep_path.read_text(encoding="utf-8")
        != expected_covisibility_mh05_boundary_support_gate_sweep
    ):
        report_stale(
            covisibility_mh05_boundary_support_gate_sweep_path,
            "python scripts/summarize_euroc_covisibility_mh05_boundary_support_gate.py "
            f"--registry-dir {args.covisibility_mh05_mitigation_registry_dir} "
            f"--out {covisibility_mh05_boundary_support_gate_sweep_path}",
        )
        stale = 1
    if missing_covisibility_mh05_boundary_support_gate_sweep:
        print("missing covisibility MH_05 boundary-support sweep registry run(s):", file=sys.stderr)
        for label in missing_covisibility_mh05_boundary_support_gate_sweep:
            print(f"  config={label}", file=sys.stderr)
        print(
            "  regenerate with the MH_05 quality-gate-only, boundary7/2, and boundary10/2 covisibility runs",
            file=sys.stderr,
        )
        stale = 1
    if stale:
        return 1
    print("generated benchmark docs are up to date")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="write one run manifest")
    cap.add_argument("--out", required=True)
    cap.add_argument("--run-id")
    cap.add_argument("--benchmark-id", required=True)
    cap.add_argument("--benchmark-name")
    cap.add_argument("--script")
    cap.add_argument("--protocol")
    cap.add_argument("--docs", action="append", default=[])
    cap.add_argument("--dataset-name", required=True)
    cap.add_argument("--dataset-sequence")
    cap.add_argument("--dataset-version")
    cap.add_argument("--dataset-path")
    cap.add_argument("--dataset-checksum")
    cap.add_argument("--dataset-checksum-method")
    cap.add_argument("--result-kind", choices=sorted(RESULT_KINDS), default="visloc_run")
    cap.add_argument("--claim-scope", choices=sorted(CLAIM_SCOPES), default="exploratory")
    cap.add_argument("--status", choices=sorted(RUN_STATUSES), required=True)
    cap.add_argument("--failure-reason")
    cap.add_argument("--command")
    cap.add_argument("--feature", action="append", default=[])
    cap.add_argument("--profile")
    cap.add_argument("--target")
    cap.add_argument("--config", action="append", default=[], help="KEY=VALUE, JSON values accepted")
    cap.add_argument("--config-file", action="append", default=[])
    cap.add_argument("--seed")
    cap.add_argument("--metric", action="append", default=[], help="NAME=VALUE or NAME=VALUE:UNIT")
    cap.add_argument("--primary-metric")
    cap.add_argument("--artifact", action="append", default=[], help="KIND=PATH")
    cap.add_argument("--model", action="append", default=[], help="NAME=PATH")
    cap.add_argument("--hardware-note")
    cap.add_argument("--notes")
    cap.set_defaults(func=capture)

    val = sub.add_parser("validate", help="validate run manifests or claim files")
    val.add_argument("paths", nargs="+")
    val.set_defaults(func=validate_cmd)

    rr = sub.add_parser("render-readme", help="render README benchmark table from claims JSON")
    rr.add_argument("--claims", required=True)
    rr.add_argument("--out")
    rr.add_argument("--readme")
    rr.add_argument("--with-heading", action="store_true")
    rr.set_defaults(func=render_readme)

    runs = sub.add_parser("render-runs", help="render a table of registered run manifests")
    runs.add_argument("--registry-dir", default="benchmarks/registry/runs")
    runs.add_argument("--out")
    runs.add_argument("--with-heading", action="store_true")
    runs.set_defaults(func=render_runs)

    matrix = sub.add_parser("render-claim-matrix", help="render the benchmark comparison claim matrix")
    matrix.add_argument("--matrix", default="benchmarks/registry/claim_matrix_v1.json")
    matrix.add_argument("--out")
    matrix.add_argument("--with-heading", action="store_true")
    matrix.set_defaults(func=render_claim_matrix)

    gen = sub.add_parser("check-generated", help="check generated benchmark docs are up to date")
    gen.add_argument("--claims", default="benchmarks/registry/readme_claims_v1.json")
    gen.add_argument("--claim-matrix", default="benchmarks/registry/claim_matrix_v1.json")
    gen.add_argument("--registry-dir", default="benchmarks/registry/runs")
    gen.add_argument("--readme", default="README.md")
    gen.add_argument("--benchmark-snapshot", default="docs/generated/benchmark_snapshot.md")
    gen.add_argument("--registered-runs", default="docs/generated/registered_runs.md")
    gen.add_argument("--claim-matrix-out", default="docs/generated/benchmark_claim_matrix.md")
    gen.add_argument(
        "--kitti-adaptive-depth-gate-smoke",
        default="docs/generated/kitti_adaptive_depth_gate_smoke.md",
    )
    gen.add_argument(
        "--kitti-adaptive-depth-gate-smoke-registry-dir",
        default="benchmarks/registry/runs/kitti",
    )
    gen.add_argument("--kitti-adaptive-depth-gate-smoke-sequence", default="00")
    gen.add_argument("--kitti-adaptive-depth-gate-smoke-max-frames", type=int, default=2)
    gen.add_argument(
        "--kitti-adaptive-depth-gate-smoke-variant",
        action="append",
        choices=["adaptive", "fixed"],
        default=None,
    )
    gen.add_argument(
        "--active-observation-sweep",
        default="docs/generated/euroc_active_observation_sweep.md",
    )
    gen.add_argument(
        "--active-observation-registry-dir",
        default="benchmarks/registry/runs/euroc",
    )
    gen.add_argument("--active-observation-max-frames", type=int, default=400)
    gen.add_argument(
        "--active-observation-sequence",
        action="append",
        default=None,
    )
    gen.add_argument(
        "--active-observation-floor",
        action="append",
        type=int,
        default=None,
    )
    gen.add_argument("--active-observation-fallback", default="none")
    gen.add_argument(
        "--tight-vio-gate-smoke",
        default="docs/generated/euroc_tight_vio_gate_smoke.md",
    )
    gen.add_argument(
        "--tight-vio-gate-registry-dir",
        default="benchmarks/registry/runs/euroc",
    )
    gen.add_argument("--tight-vio-gate-max-frames", type=int, default=400)
    gen.add_argument(
        "--tight-vio-gate-sequence",
        action="append",
        default=None,
    )
    gen.add_argument(
        "--tight-vio-gate-variant",
        action="append",
        default=None,
    )
    gen.add_argument(
        "--tight-vio-gate-mh03-1500",
        default="docs/generated/euroc_tight_vio_gate_mh03_1500.md",
    )
    gen.add_argument(
        "--tight-vio-gate-mh03-1500-registry-dir",
        default="benchmarks/registry/runs/euroc",
    )
    gen.add_argument("--tight-vio-gate-mh03-1500-max-frames", type=int, default=1500)
    gen.add_argument(
        "--tight-vio-gate-mh03-1500-sequence",
        action="append",
        default=None,
    )
    gen.add_argument(
        "--tight-vio-gate-mh03-1500-variant",
        action="append",
        default=None,
    )
    gen.add_argument(
        "--covisibility-runtime-sweep",
        default="docs/generated/euroc_covisibility_runtime_sweep.md",
    )
    gen.add_argument(
        "--covisibility-runtime-registry-dir",
        default="benchmarks/registry/runs/euroc",
    )
    gen.add_argument("--covisibility-runtime-max-frames", type=int, default=80)
    gen.add_argument(
        "--covisibility-runtime-sequence",
        action="append",
        default=None,
    )
    gen.add_argument(
        "--covisibility-runtime-landmark-cap",
        action="append",
        type=int,
        default=None,
    )
    gen.add_argument("--covisibility-runtime-neighbor-keyframes", type=int, default=10)
    gen.add_argument("--covisibility-runtime-boundary-keyframes", type=int, default=10)
    gen.add_argument("--covisibility-runtime-min-active-observations", type=int, default=20)
    gen.add_argument("--covisibility-runtime-fallback", default="none")
    gen.add_argument("--covisibility-runtime-remove-outliers", action="store_true")
    gen.add_argument("--covisibility-runtime-max-outlier-observation-ratio", default="none")
    gen.add_argument(
        "--covisibility-runtime-boundary-support-min-optimized-keyframes",
        default="none",
    )
    gen.add_argument("--covisibility-runtime-boundary-support-min-fixed-keyframes", type=int, default=0)
    gen.add_argument(
        "--covisibility-window-sweep",
        default="docs/generated/euroc_covisibility_window_sweep.md",
    )
    gen.add_argument(
        "--covisibility-window-registry-dir",
        default="benchmarks/registry/runs/euroc",
    )
    gen.add_argument("--covisibility-window-max-frames", type=int, default=80)
    gen.add_argument(
        "--covisibility-window-sequence",
        action="append",
        default=None,
    )
    gen.add_argument(
        "--covisibility-window-cap",
        action="append",
        type=parse_window_cap,
        default=None,
    )
    gen.add_argument("--covisibility-window-landmark-cap", type=int, default=200)
    gen.add_argument("--covisibility-window-min-keyframes", type=int, default=3)
    gen.add_argument("--covisibility-window-trigger-every", type=int, default=1)
    gen.add_argument("--covisibility-window-min-active-observations", type=int, default=20)
    gen.add_argument("--covisibility-window-fallback", default="none")
    gen.add_argument("--covisibility-window-remove-outliers", action="store_true")
    gen.add_argument("--covisibility-window-max-outlier-observation-ratio", default="none")
    gen.add_argument(
        "--covisibility-window-boundary-support-min-optimized-keyframes",
        default="none",
    )
    gen.add_argument("--covisibility-window-boundary-support-min-fixed-keyframes", type=int, default=0)
    gen.add_argument(
        "--covisibility-window-validation",
        default="docs/generated/euroc_covisibility_window_validation.md",
    )
    gen.add_argument(
        "--covisibility-window-validation-registry-dir",
        default="benchmarks/registry/runs/euroc",
    )
    gen.add_argument("--covisibility-window-validation-max-frames", type=int, default=400)
    gen.add_argument(
        "--covisibility-window-validation-sequence",
        action="append",
        default=None,
    )
    gen.add_argument(
        "--covisibility-window-validation-cap",
        action="append",
        type=parse_window_cap,
        default=None,
    )
    gen.add_argument("--covisibility-window-validation-landmark-cap", type=int, default=200)
    gen.add_argument("--covisibility-window-validation-min-keyframes", type=int, default=3)
    gen.add_argument("--covisibility-window-validation-trigger-every", type=int, default=1)
    gen.add_argument("--covisibility-window-validation-min-active-observations", type=int, default=20)
    gen.add_argument("--covisibility-window-validation-fallback", default="none")
    gen.add_argument("--covisibility-window-validation-remove-outliers", action="store_true")
    gen.add_argument(
        "--covisibility-window-validation-max-outlier-observation-ratio",
        default="none",
    )
    gen.add_argument(
        "--covisibility-window-validation-boundary-support-min-optimized-keyframes",
        default="none",
    )
    gen.add_argument(
        "--covisibility-window-validation-boundary-support-min-fixed-keyframes",
        type=int,
        default=0,
    )
    gen.add_argument(
        "--covisibility-ab",
        default="docs/generated/euroc_covisibility_ab_400.md",
    )
    gen.add_argument(
        "--covisibility-ab-registry-dir",
        default="benchmarks/registry/runs/euroc",
    )
    gen.add_argument("--covisibility-ab-max-frames", type=int, default=400)
    gen.add_argument(
        "--covisibility-ab-sequence",
        action="append",
        default=None,
    )
    gen.add_argument("--covisibility-ab-enabled-neighbor-keyframes", type=int, default=10)
    gen.add_argument("--covisibility-ab-enabled-boundary-keyframes", type=int, default=10)
    gen.add_argument("--covisibility-ab-enabled-min-keyframes", type=int, default=3)
    gen.add_argument("--covisibility-ab-enabled-trigger-every", type=int, default=1)
    gen.add_argument("--covisibility-ab-enabled-landmark-cap", type=int, default=200)
    gen.add_argument("--covisibility-ab-enabled-min-active-observations", type=int, default=20)
    gen.add_argument("--covisibility-ab-enabled-fallback", default="none")
    gen.add_argument("--covisibility-ab-enabled-remove-outliers", action="store_true")
    gen.add_argument("--covisibility-ab-enabled-max-outlier-observation-ratio", default="none")
    gen.add_argument(
        "--covisibility-ab-enabled-boundary-support-min-optimized-keyframes",
        default="none",
    )
    gen.add_argument("--covisibility-ab-enabled-boundary-support-min-fixed-keyframes", type=int, default=0)
    gen.add_argument(
        "--covisibility-mh05-mitigation",
        default="docs/generated/euroc_covisibility_mh05_mitigation.md",
    )
    gen.add_argument(
        "--covisibility-mh05-quality-gate",
        default="docs/generated/euroc_covisibility_mh05_quality_gate_0p3.md",
    )
    gen.add_argument(
        "--covisibility-mh05-boundary-support-gate",
        default="docs/generated/euroc_covisibility_mh05_boundary_support_gate_10_2.md",
    )
    gen.add_argument(
        "--covisibility-mh05-boundary-support-gate-sweep",
        default="docs/generated/euroc_covisibility_mh05_boundary_support_gate_sweep.md",
    )
    gen.add_argument(
        "--covisibility-mh05-mitigation-registry-dir",
        default="benchmarks/registry/runs/euroc",
    )
    gen.add_argument("--covisibility-mh05-mitigation-sequence", default="MH_05_difficult")
    gen.add_argument("--covisibility-mh05-mitigation-max-frames", type=int, default=400)
    gen.add_argument("--covisibility-mh05-mitigation-neighbor-keyframes", type=int, default=10)
    gen.add_argument("--covisibility-mh05-mitigation-boundary-keyframes", type=int, default=10)
    gen.add_argument("--covisibility-mh05-mitigation-landmark-cap", type=int, default=200)
    gen.add_argument("--covisibility-mh05-mitigation-min-keyframes", type=int, default=3)
    gen.add_argument("--covisibility-mh05-mitigation-trigger-every", type=int, default=1)
    gen.add_argument("--covisibility-mh05-mitigation-min-active-observations", type=int, default=20)
    gen.add_argument("--covisibility-mh05-mitigation-fallback", default="none")
    gen.add_argument("--covisibility-mh05-mitigation-remove-outliers", action="store_true")
    gen.add_argument("--covisibility-mh05-mitigation-max-outlier-observation-ratio", default="none")
    gen.add_argument("--covisibility-mh05-quality-gate-max-outlier-observation-ratio", default="0.3")
    gen.add_argument(
        "--covisibility-mh05-boundary-support-gate-max-outlier-observation-ratio",
        default="0.3",
    )
    gen.add_argument(
        "--covisibility-mh05-mitigation-boundary-support-min-optimized-keyframes",
        default="none",
    )
    gen.add_argument(
        "--covisibility-mh05-mitigation-boundary-support-min-fixed-keyframes",
        type=int,
        default=0,
    )
    gen.add_argument(
        "--covisibility-mh05-quality-gate-boundary-support-min-optimized-keyframes",
        default="none",
    )
    gen.add_argument(
        "--covisibility-mh05-quality-gate-boundary-support-min-fixed-keyframes",
        type=int,
        default=0,
    )
    gen.add_argument(
        "--covisibility-mh05-boundary-support-gate-min-optimized-keyframes",
        default="10",
    )
    gen.add_argument(
        "--covisibility-mh05-boundary-support-gate-min-fixed-keyframes",
        type=int,
        default=2,
    )
    gen.add_argument(
        "--covisibility-mh05-mitigation-config",
        action="append",
        type=parse_covisibility_mh05_mitigation_config,
        default=None,
    )
    gen.add_argument(
        "--covisibility-mh05-boundary-support-gate-config",
        action="append",
        type=parse_covisibility_mh05_mitigation_config,
        default=None,
    )
    gen.add_argument(
        "--covisibility-mh05-boundary-support-gate-sweep-gate",
        action="append",
        type=parse_covisibility_mh05_boundary_support_gate,
        default=None,
    )
    gen.set_defaults(func=check_generated)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
