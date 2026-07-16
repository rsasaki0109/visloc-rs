#!/usr/bin/env python3
"""Run provenance-complete official ORB-SLAM3 and COLMAP global baselines.

The runner deliberately does not score trajectories against ground truth.  It
only runs the competing engine and captures immutable inputs plus raw outputs;
evaluation is a separate process so ground truth cannot leak into SLAM/SfM.

Examples:
  python scripts/run_official_baselines.py orb-slam3 \
    --executable /opt/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc \
    --vocabulary /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt \
    --settings /opt/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC.yaml \
    --sequence-dir /datasets/euroc/MH01 \
    --timestamps /opt/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC_TimeStamps/MH01.txt \
    --sequence MH_01_easy --source-revision v1.0-release \
    --out-root /mnt/e/visloc_archive/official/orb_mh01 --repetitions 5

  python scripts/run_official_baselines.py colmap-global \
    --executable colmap --database /data/south-building/database.db \
    --images /data/south-building/images --sequence south-building \
    --source-revision 43dd3bb2 --out-root /mnt/e/visloc_archive/official/glomap_sb \
    --repetitions 5
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
POLL_SECONDS = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_data_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return sum(1 for line in stream if line.strip() and not line.lstrip().startswith("#"))


def git_metadata() -> Dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.autocrlf=true",
                "-c",
                "core.filemode=false",
                "-C",
                str(ROOT),
                *args,
            ],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain=v1")
    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def executable_fingerprint(raw: str) -> Dict[str, Any]:
    resolved = shutil.which(raw)
    path = Path(resolved or raw).expanduser()
    if not path.is_file():
        raise ValueError(f"executable not found: {raw}")
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def file_fingerprint(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"required file not found: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def directory_identity(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"required directory not found: {path}")
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        size = candidate.stat().st_size
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "little"))
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files += 1
        total_bytes += size
    return {
        "path": str(path),
        "file_count": files,
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def rss_bytes(pid: int) -> Optional[int]:
    """Best-effort resident memory for the direct child, without dependencies."""
    if sys.platform.startswith("linux"):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    elif os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            query_information, vm_read = 0x0400, 0x0010
            handle = ctypes.windll.kernel32.OpenProcess(query_information | vm_read, False, pid)
            if not handle:
                return None
            try:
                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                )
                return int(counters.WorkingSetSize) if ok else None
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return None
    return None


def run_logged(
    argv: Sequence[str], cwd: Path, stdout_path: Path, stderr_path: Path, timeout: Optional[float]
) -> Dict[str, Any]:
    started = time.perf_counter()
    peak_rss = 0
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(list(argv), cwd=cwd, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            sample = rss_bytes(process.pid)
            if sample is not None:
                peak_rss = max(peak_rss, sample)
            if timeout is not None and time.perf_counter() - started > timeout:
                timed_out = True
                process.terminate()
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(POLL_SECONDS)
        return_code = process.wait()
    return {
        "argv": list(argv),
        "cwd": str(cwd),
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss or None,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def option_argv(options: Sequence[str]) -> List[str]:
    argv: List[str] = []
    for option in options:
        if "=" not in option:
            raise ValueError(f"option must be NAME=VALUE, got: {option}")
        name, value = option.split("=", 1)
        if not name or name.startswith("-") or not value:
            raise ValueError(f"invalid NAME=VALUE option: {option}")
        argv.extend([f"--{name}", value])
    return argv


def orb_command(args: argparse.Namespace, label: str) -> List[str]:
    return [
        str(Path(args.executable).expanduser().resolve()),
        *args.executable_arg,
        str(args.vocabulary.expanduser().resolve()),
        str(args.settings.expanduser().resolve()),
        str(args.sequence_dir.expanduser().resolve()),
        str(args.timestamps.expanduser().resolve()),
        label,
        *args.engine_arg,
    ]


def colmap_commands(args: argparse.Namespace, run_dir: Path, database: Path) -> List[List[str]]:
    executable = executable_fingerprint(args.executable)["path"]
    prefix = [executable, *args.executable_arg]
    commands: List[List[str]] = []
    if not args.skip_view_graph_calibrator:
        commands.append(
            prefix + ["view_graph_calibrator", "--database_path", str(database)]
            + option_argv(args.calibrator_option)
        )
    commands.append(
        prefix
        + [
            "global_mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(args.images.expanduser().resolve()),
            "--output_path",
            str(run_dir / "sparse"),
        ]
        + option_argv(args.mapper_option)
    )
    return commands


def parse_model_analyzer(text: str) -> Dict[str, Union[float, int]]:
    patterns: Dict[str, Tuple[str, type]] = {
        "registered_images": (r"Registered images\s*:\s*(\d+)", int),
        "points3d": (r"Points\s*:\s*(\d+)", int),
        "observations": (r"Observations\s*:\s*(\d+)", int),
        "mean_track_length": (r"Mean track length\s*:\s*([0-9.eE+-]+)", float),
        "mean_observations_per_image": (
            r"Mean observations per image\s*:\s*([0-9.eE+-]+)",
            float,
        ),
        "mean_reprojection_error_px": (
            r"Mean reprojection error\s*:\s*([0-9.eE+-]+)px",
            float,
        ),
    }
    metrics: Dict[str, Union[float, int]] = {}
    for name, (pattern, cast) in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            metrics[name] = cast(match.group(1))
    return metrics


def median_or_none(values: Sequence[Optional[Union[float, int]]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.median(finite) if finite else None


def base_manifest(args: argparse.Namespace, engine: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "official-orbslam3-colmap-global-baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "source_revision": args.source_revision,
        "runner_git": git_metadata(),
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "protocol": {
            "repetitions": args.repetitions,
            "ground_truth_available_to_engine": False,
            "fresh_output_per_repetition": True,
        },
        "inputs": inputs,
        "runs": [],
    }


def run_orb(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    executable = executable_fingerprint(args.executable)
    args.executable = executable["path"]
    inputs = {
        "executable": executable,
        "vocabulary": file_fingerprint(args.vocabulary),
        "settings": file_fingerprint(args.settings),
        "sequence": directory_identity(args.sequence_dir),
        "timestamps": file_fingerprint(args.timestamps),
        "sequence_name": args.sequence,
    }
    manifest = base_manifest(args, "orb-slam3-stereo-inertial", inputs)
    manifest["protocol"]["sensor_mode"] = "stereo-inertial"
    manifest["protocol"]["trajectory_label_contract"] = "f_<label>.txt and kf_<label>.txt"
    failed = False
    for index in range(1, args.repetitions + 1):
        run_dir = args.out_root / f"run_{index:02d}"
        label = f"{args.sequence}_stereoi_r{index:02d}"
        command = orb_command(args, label)
        if args.dry_run:
            manifest["runs"].append(
                {"run": index, "status": "dry_run", "argv": command, "cwd": str(run_dir)}
            )
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        process = run_logged(
            command, run_dir, run_dir / "stdout.log", run_dir / "stderr.log", args.timeout
        )
        frame = run_dir / f"f_{label}.txt"
        keyframe = run_dir / f"kf_{label}.txt"
        artifacts = {
            "frame_trajectory": str(frame) if frame.is_file() else None,
            "keyframe_trajectory": str(keyframe) if keyframe.is_file() else None,
        }
        status = "success" if process["return_code"] == 0 and frame.is_file() else "failure"
        failed |= status != "success"
        manifest["runs"].append(
            {
                "run": index,
                "status": status,
                "process": process,
                "metrics": {
                    "frame_trajectory_poses": count_data_lines(frame),
                    "keyframe_trajectory_poses": count_data_lines(keyframe),
                },
                "artifacts": artifacts,
            }
        )
    return manifest, int(failed)


def find_colmap_model(sparse: Path) -> Optional[Path]:
    candidates = sorted(path for path in sparse.iterdir() if path.is_dir()) if sparse.is_dir() else []
    return candidates[0] if candidates else None


def run_colmap(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    executable = executable_fingerprint(args.executable)
    inputs = {
        "executable": executable,
        "database": file_fingerprint(args.database),
        "images": directory_identity(args.images),
        "sequence_name": args.sequence,
    }
    manifest = base_manifest(args, "colmap-global-mapper", inputs)
    manifest["protocol"].update(
        {
            "same_database_copied_per_repetition": True,
            "view_graph_calibrator": not args.skip_view_graph_calibrator,
            "mapper_options": args.mapper_option,
            "calibrator_options": args.calibrator_option,
        }
    )
    failed = False
    for index in range(1, args.repetitions + 1):
        run_dir = args.out_root / f"run_{index:02d}"
        database = run_dir / "database.db"
        commands = colmap_commands(args, run_dir, database)
        if args.dry_run:
            manifest["runs"].append(
                {"run": index, "status": "dry_run", "commands": commands, "cwd": str(run_dir)}
            )
            continue
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.database, database)
        (run_dir / "sparse").mkdir()
        processes = []
        run_failed = False
        for command_index, command in enumerate(commands, 1):
            process = run_logged(
                command,
                run_dir,
                run_dir / f"command_{command_index:02d}.stdout.log",
                run_dir / f"command_{command_index:02d}.stderr.log",
                args.timeout,
            )
            processes.append(process)
            if process["return_code"] != 0:
                run_failed = True
                break
        model = find_colmap_model(run_dir / "sparse")
        metrics: Dict[str, Any] = {}
        analyzer_process = None
        if not run_failed and model is not None:
            analyzer_process = run_logged(
                [executable["path"], *args.executable_arg, "model_analyzer", "--path", str(model)],
                run_dir,
                run_dir / "model_analyzer.stdout.log",
                run_dir / "model_analyzer.stderr.log",
                args.timeout,
            )
            # COLMAP uses glog and may emit model statistics on stderr depending
            # on the build/runtime logging configuration. Parse both streams so
            # the manifest is independent of that packaging detail.
            analyzer_text = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in (
                    run_dir / "model_analyzer.stdout.log",
                    run_dir / "model_analyzer.stderr.log",
                )
            )
            metrics = parse_model_analyzer(analyzer_text)
        analyzer_ok = (
            analyzer_process is not None
            and analyzer_process["return_code"] == 0
            and "registered_images" in metrics
            and "points3d" in metrics
        )
        status = (
            "success"
            if not run_failed and model is not None and analyzer_ok
            else "failure"
        )
        failed |= status != "success"
        manifest["runs"].append(
            {
                "run": index,
                "status": status,
                "processes": processes,
                "model_analyzer_process": analyzer_process,
                "metrics": metrics,
                "artifacts": {"sparse_model": str(model) if model is not None else None},
            }
        )
    return manifest, int(failed)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--executable", required=True)
    parser.add_argument(
        "--executable-arg",
        action="append",
        default=[],
        help="argument inserted immediately after the executable (mainly for test/wrapper shims)",
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=None, help="seconds per command")
    parser.add_argument("--dry-run", action="store_true")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # argparse only gained required subparsers in Python 3.7. Keep this runner
    # usable in the Ubuntu 18.04/Python 3.6 environment supported by ORB-SLAM3.
    subparsers = parser.add_subparsers(dest="engine")
    orb = subparsers.add_parser("orb-slam3", help="official EuRoC stereo-inertial example")
    add_common(orb)
    orb.add_argument("--vocabulary", type=Path, required=True)
    orb.add_argument("--settings", type=Path, required=True)
    orb.add_argument("--sequence-dir", type=Path, required=True)
    orb.add_argument("--timestamps", type=Path, required=True)
    orb.add_argument("--engine-arg", action="append", default=[])

    colmap = subparsers.add_parser("colmap-global", help="current GLOMAP successor")
    add_common(colmap)
    colmap.add_argument("--database", type=Path, required=True)
    colmap.add_argument("--images", type=Path, required=True)
    colmap.add_argument("--skip-view-graph-calibrator", action="store_true")
    colmap.add_argument("--calibrator-option", action="append", default=[], metavar="NAME=VALUE")
    colmap.add_argument("--mapper-option", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args(argv)
    if args.engine is None:
        parser.error("an engine subcommand is required")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.out_root = args.out_root.expanduser().resolve()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.out_root.exists() and any(args.out_root.iterdir()):
            raise ValueError(f"out-root must be absent or empty: {args.out_root}")
        args.out_root.mkdir(parents=True, exist_ok=True)
        manifest, return_code = run_orb(args) if args.engine == "orb-slam3" else run_colmap(args)
        manifest["summary"] = {
            "status": "dry_run" if args.dry_run else ("failure" if return_code else "success"),
            "successful_runs": sum(run["status"] == "success" for run in manifest["runs"]),
            "median_wall_seconds": median_or_none(
                [
                    run.get("process", {}).get("wall_seconds")
                    if args.engine == "orb-slam3"
                    else sum(p.get("wall_seconds", 0.0) for p in run.get("processes", []))
                    for run in manifest["runs"]
                ]
            ),
            "median_peak_rss_bytes": median_or_none(
                [
                    run.get("process", {}).get("peak_rss_bytes")
                    if args.engine == "orb-slam3"
                    else max(
                        [p.get("peak_rss_bytes") or 0 for p in run.get("processes", [])],
                        default=0,
                    )
                    or None
                    for run in manifest["runs"]
                ]
            ),
        }
        manifest_path = args.out_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(manifest_path)
        return return_code
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
