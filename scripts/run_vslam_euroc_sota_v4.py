#!/usr/bin/env python3
"""Run the frozen EuRoC V4 matrix serially with immutable evidence.

The algorithm configuration is a separate input because it must be frozen only
after V3 development. Its byte-level SHA-256, the executable, model bundle,
ORT DLL, protocol, and bounded-queue settings are rechecked before every run.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_bundle_sha256(root: Path, extra_files: list[Path] | None = None) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"model bundle has no files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_digest = bytes.fromhex(sha256(path))
        digest.update(file_digest)
    for index, path in enumerate(extra_files or []):
        label = f"@external/{index}/{path.name}".encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def resolve_sequence(dataset_root: Path, sequence: str) -> Path:
    direct = dataset_root / sequence
    if (direct / "mav0" / "cam0" / "data.csv").is_file():
        return direct
    matches = [
        path.parent.parent.parent
        for path in dataset_root.rglob("mav0/cam0/data.csv")
        if path.parent.parent.parent.name == sequence
    ]
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ValueError(f"expected one dataset directory for {sequence}, found {unique}")
    return unique[0]


def camera_rows(sequence_dir: Path) -> int:
    path = sequence_dir / "mav0" / "cam0" / "data.csv"
    return sum(
        1
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def unique_option(arguments: list[str], option: str) -> str:
    indices = [index for index, value in enumerate(arguments) if value == option]
    if len(indices) != 1 or indices[0] + 1 >= len(arguments):
        raise ValueError(f"configuration must contain one {option} value")
    return arguments[indices[0] + 1]


def validate_configuration(configuration: dict[str, Any], gates: dict[str, Any]) -> None:
    if configuration.get("schema_version") != 1:
        raise ValueError("configuration must have schema_version=1")
    arguments = configuration.get("arguments")
    if not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments):
        raise ValueError("configuration arguments must be a string list")
    forbidden = {
        "--euroc-dir",
        "--out-dir",
        "--max-frames",
        "--stride",
        "--seed",
        "--model-dir",
        "--ll-superpoint-model",
        "--native-cuda-correlation-dll",
    }
    if forbidden.intersection(arguments):
        raise ValueError("configuration contains runner-owned arguments")
    required_switches = {
        "--onnx-cuda",
        "--imu",
        "--loop-closure",
        "--global-ba",
        "--gba-widen-t0",
        "--sim3-backend",
        "--long-loop",
        "--pipeline-prefetch",
    }
    missing = required_switches.difference(arguments)
    if missing:
        raise ValueError("configuration is missing " + ", ".join(sorted(missing)))
    if "--onnx-cpu" in arguments:
        raise ValueError("configuration cannot combine strict CUDA with --onnx-cpu")
    if "--onnx-correlation" in arguments:
        raise ValueError("configuration cannot use the measured-negative grouped correlation graph")
    if gates.get("required_onnx_backend") != "cuda":
        raise ValueError("frozen V4 protocol must require the strict CUDA backend")
    if gates.get("required_onnx_full_update_graph") is not True:
        raise ValueError("frozen V4 protocol must require the fused update graph")
    if gates.get("forbid_grouped_onnx_correlation") is not True:
        raise ValueError("frozen V4 protocol must forbid grouped ONNX correlation")
    if gates.get("required_native_cuda_correlation") is not True:
        raise ValueError("frozen V4 protocol must require native CUDA correlation")
    if gates.get("required_native_cuda_correlation_abi") != 3:
        raise ValueError("frozen V4 protocol must require native CUDA correlation ABI 3")
    if gates.get("required_final_refinement_iterations") != 12:
        raise ValueError("frozen V4 protocol must require 12 final refinement iterations")
    if gates.get("required_pipeline_prefetch") is not True:
        raise ValueError("frozen V4 protocol must require bounded pipeline prefetch")
    queue_bounds = gates["queue_bounds"]
    if configuration.get("queue_bounds") != queue_bounds:
        raise ValueError("configuration queue bounds differ from the frozen V4 protocol")
    for option, key in (
        ("--gba-inactive-edge-cap", "inactive_edge_cap"),
        ("--gba-max-free-poses", "max_free_poses"),
        ("--ll-max-indexed-frames", "long_loop_max_indexed_frames"),
    ):
        try:
            actual = int(unique_option(arguments, option))
        except ValueError as error:
            raise ValueError(f"configuration has invalid {option}") from error
        if actual < 1 or actual > queue_bounds[key]:
            raise ValueError(f"configuration {option} exceeds the frozen V4 queue bound")
    try:
        actual_scale_gate = float(
            unique_option(arguments, "--s3b-max-abs-log-scale-correction")
        )
    except ValueError as error:
        raise ValueError("configuration has invalid Sim3 scale gate") from error
    if not math.isclose(
        actual_scale_gate,
        gates["max_committed_abs_log_scale"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("configuration Sim3 scale gate differs from the frozen V4 protocol")
    superpoint = configuration.get("long_loop_superpoint_model")
    if not isinstance(superpoint, str) or not superpoint:
        raise ValueError("configuration has no long_loop_superpoint_model")


def process_working_set_bytes(process_id: int) -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCountersEx(ctypes.Structure):
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
            ("PrivateUsage", ctypes.c_size_t),
        ]

    query_information = 0x0400
    handle = ctypes.windll.kernel32.OpenProcess(query_information, False, process_id)
    if not handle:
        return None
    try:
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else None
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def parse_gpu_peak(path: Path, process_id: int) -> int | None:
    if not path.is_file():
        return None
    peak_mib: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            candidate_pid, used_mib = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if candidate_pid == process_id:
            peak_mib = used_mib if peak_mib is None else max(peak_mib, used_mib)
    return None if peak_mib is None else peak_mib * 1024 * 1024


def run_one(
    executable: Path,
    arguments: list[str],
    run_dir: Path,
    manifest: dict[str, Any],
    sample_interval: float,
) -> dict[str, Any]:
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    gpu_log = run_dir / "nvidia_smi_compute_apps.log"
    manifest_path = run_dir / "run_manifest.json"
    atomic_json(manifest_path, manifest)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen([str(executable), *arguments], stdout=stdout, stderr=stderr)
        gpu_stream = gpu_log.open("wb")
        gpu_monitor: subprocess.Popen[bytes] | None = None
        try:
            gpu_monitor = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=500",
                ],
                stdout=gpu_stream,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            gpu_stream.close()
        peak_working_set = 0
        while process.poll() is None:
            sampled = process_working_set_bytes(process.pid)
            if sampled is not None:
                peak_working_set = max(peak_working_set, sampled)
            time.sleep(sample_interval)
        exit_code = process.wait()
        sampled = process_working_set_bytes(process.pid)
        if sampled is not None:
            peak_working_set = max(peak_working_set, sampled)
        if gpu_monitor is not None:
            gpu_monitor.terminate()
            try:
                gpu_monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gpu_monitor.kill()
                gpu_monitor.wait()
            gpu_stream.close()

    summary_path = run_dir / "summary.txt"
    manifest["finished_at"] = utc_now()
    manifest["exit_code"] = exit_code
    manifest["sampled_peak_working_set_bytes"] = peak_working_set or None
    manifest["sampled_peak_gpu_memory_bytes"] = parse_gpu_peak(gpu_log, process.pid)
    manifest["summary_sha256"] = sha256(summary_path) if summary_path.is_file() else None
    atomic_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--native-cuda-correlation-dll", type=Path, required=True)
    parser.add_argument("--ort-provider-shared-dll", type=Path, required=True)
    parser.add_argument("--ort-provider-cuda-dll", type=Path, required=True)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_interval <= 0 or not math.isfinite(args.sample_interval):
        raise ValueError("sample interval must be finite and positive")
    for path in (
        args.dataset_root,
        args.protocol,
        args.configuration,
        args.executable,
        args.model_dir,
        args.native_cuda_correlation_dll,
        args.ort_provider_shared_dll,
        args.ort_provider_cuda_dll,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    ort_path_value = os.environ.get("ORT_DYLIB_PATH")
    if not ort_path_value or not (ort_path := Path(ort_path_value)).is_file():
        raise ValueError("ORT_DYLIB_PATH must name the frozen ONNX Runtime DLL")
    executable_dir = args.executable.resolve().parent
    required_runtime_files = {
        "onnxruntime.dll": ort_path,
        "onnxruntime_providers_shared.dll": args.ort_provider_shared_dll,
        "onnxruntime_providers_cuda.dll": args.ort_provider_cuda_dll,
    }
    for name, path in required_runtime_files.items():
        expected = executable_dir / name
        if path.resolve() != expected.resolve():
            raise ValueError(f"{name} must be frozen beside the executable: {expected}")
    native_library = (
        ctypes.WinDLL(str(args.native_cuda_correlation_dll))
        if os.name == "nt"
        else ctypes.CDLL(str(args.native_cuda_correlation_dll))
    )
    abi_version = native_library.visloc_dpvo_corr_abi_version
    abi_version.restype = ctypes.c_uint32
    if abi_version() != 3:
        raise ValueError("native CUDA correlation DLL must expose ABI version 3")

    protocol = load_json(args.protocol)
    if protocol.get("schema_version") != 1 or protocol.get("repetitions") != 3:
        raise ValueError("unsupported V4 protocol")
    sequences = protocol.get("sequences")
    frame_counts = protocol.get("full_sequence_frame_counts")
    gates = protocol.get("gates")
    queue_bounds = gates.get("queue_bounds") if isinstance(gates, dict) else None
    if not isinstance(sequences, list) or len(sequences) != 11:
        raise ValueError("V4 protocol must specify 11 sequences")
    if not isinstance(frame_counts, dict) or set(frame_counts) != set(sequences):
        raise ValueError("V4 protocol frame counts do not match sequences")
    if not isinstance(queue_bounds, dict):
        raise ValueError("V4 protocol has no queue bounds")
    configuration = load_json(args.configuration)
    validate_configuration(configuration, gates)
    superpoint_path = Path(configuration["long_loop_superpoint_model"])
    if not superpoint_path.is_file():
        raise FileNotFoundError(superpoint_path)
    sequence_dirs = {sequence: resolve_sequence(args.dataset_root, sequence) for sequence in sequences}
    for sequence, sequence_dir in sequence_dirs.items():
        observed = camera_rows(sequence_dir)
        if observed != frame_counts[sequence]:
            raise ValueError(
                f"{sequence}: cam0 CSV has {observed} frames, expected {frame_counts[sequence]}"
            )

    args.out_root.mkdir(parents=True, exist_ok=True)
    experiment_path = args.out_root / "experiment_manifest.json"
    frozen = {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol_sha256": sha256(args.protocol),
        "configuration_sha256": sha256(args.configuration),
        "executable_sha256": sha256(args.executable),
        "model_bundle_sha256": model_bundle_sha256(args.model_dir, [superpoint_path]),
        "ort_dylib_sha256": sha256(ort_path),
        "ort_provider_shared_sha256": sha256(args.ort_provider_shared_dll),
        "ort_provider_cuda_sha256": sha256(args.ort_provider_cuda_dll),
        "native_cuda_correlation_sha256": sha256(args.native_cuda_correlation_dll),
        "sequences": sequences,
        "repetitions": 3,
        "host": {"platform": platform.platform(), "python": sys.version},
    }
    if experiment_path.exists():
        if not args.resume:
            raise FileExistsError(f"experiment already exists: {experiment_path}")
        existing = load_json(experiment_path)
        for key in (
            "protocol_sha256",
            "configuration_sha256",
            "executable_sha256",
            "model_bundle_sha256",
            "ort_dylib_sha256",
            "ort_provider_shared_sha256",
            "ort_provider_cuda_sha256",
            "native_cuda_correlation_sha256",
            "sequences",
            "repetitions",
        ):
            if existing.get(key) != frozen[key]:
                raise ValueError(f"resume evidence differs in {key}")
        frozen = existing
    else:
        atomic_json(experiment_path, frozen)

    base_arguments = [
        "--model-dir",
        str(args.model_dir),
        "--max-frames",
        "0",
        "--stride",
        "1",
        "--ll-superpoint-model",
        str(superpoint_path),
        "--native-cuda-correlation-dll",
        str(args.native_cuda_correlation_dll),
        *configuration["arguments"],
    ]
    failures = 0
    for sequence in sequences:
        for repetition in range(1, 4):
            current_evidence = {
                "protocol_sha256": sha256(args.protocol),
                "configuration_sha256": sha256(args.configuration),
                "executable_sha256": sha256(args.executable),
                "model_bundle_sha256": model_bundle_sha256(
                    args.model_dir, [superpoint_path]
                ),
                "ort_dylib_sha256": sha256(ort_path),
                "ort_provider_shared_sha256": sha256(args.ort_provider_shared_dll),
                "ort_provider_cuda_sha256": sha256(args.ort_provider_cuda_dll),
                "native_cuda_correlation_sha256": sha256(
                    args.native_cuda_correlation_dll
                ),
            }
            for key, value in current_evidence.items():
                if frozen[key] != value:
                    raise ValueError(f"frozen evidence changed before run: {key}")
            run_dir = args.out_root / f"{sequence}_r{repetition}"
            manifest_path = run_dir / "run_manifest.json"
            if run_dir.exists():
                if not args.resume or not manifest_path.is_file():
                    raise FileExistsError(f"refusing to overwrite {run_dir}")
                previous = load_json(manifest_path)
                if previous.get("exit_code") == 0 and (run_dir / "summary.txt").is_file():
                    if sha256(run_dir / "summary.txt") != str(previous.get("summary_sha256", "")):
                        raise ValueError(f"completed summary changed: {run_dir}")
                    continue
                failures += 1
                continue
            run_dir.mkdir()
            arguments = [
                "--euroc-dir",
                str(sequence_dirs[sequence]),
                "--out-dir",
                str(run_dir),
                "--seed",
                str(repetition - 1),
                *base_arguments,
            ]
            manifest = {
                "schema_version": 1,
                "sequence": sequence,
                "repetition": repetition,
                "seed": repetition - 1,
                "started_at": utc_now(),
                "finished_at": None,
                "arguments": arguments,
                "exit_code": None,
                "summary_sha256": None,
                "sampled_peak_working_set_bytes": None,
                "sampled_peak_gpu_memory_bytes": None,
                "queue_bounds": queue_bounds,
                "protocol_sha256": frozen["protocol_sha256"],
                "configuration_sha256": frozen["configuration_sha256"],
                "executable_sha256": frozen["executable_sha256"],
                "model_bundle_sha256": frozen["model_bundle_sha256"],
                "ort_dylib_sha256": frozen["ort_dylib_sha256"],
                "ort_provider_shared_sha256": frozen["ort_provider_shared_sha256"],
                "ort_provider_cuda_sha256": frozen["ort_provider_cuda_sha256"],
                "native_cuda_correlation_sha256": frozen[
                    "native_cuda_correlation_sha256"
                ],
            }
            result = run_one(
                args.executable,
                arguments,
                run_dir,
                manifest,
                args.sample_interval,
            )
            if result["exit_code"] != 0 or result["summary_sha256"] is None:
                failures += 1
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
