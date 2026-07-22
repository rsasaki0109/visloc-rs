#!/usr/bin/env python3
"""Run frozen COLMAP incremental/global baselines on prepared SSfM inputs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASELINE_ID = "colmap-4.1-sift-sequential-incremental-global-v1"


class ProcessEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--ground-truth-csv", type=Path, required=True)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def windows_process_table() -> dict[int, tuple[int, int]]:
    if os.name != "nt":
        raise RuntimeError("frozen process-tree monitoring requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    table = {}
    entry = ProcessEntry32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        present = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while present:
            pid = int(entry.th32ProcessID)
            rss = 0
            handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
            if handle:
                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    rss = int(counters.WorkingSetSize)
                kernel32.CloseHandle(handle)
            table[pid] = (int(entry.th32ParentProcessID), rss)
            present = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return table


def process_tree_rss(root_pid: int) -> int:
    table = windows_process_table()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in table.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(table.get(pid, (0, 0))[1] for pid in descendants)


def gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return sum(int(line.strip()) for line in output.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def run_monitored(command: list[str], log: Path, poll_seconds: float) -> dict:
    idle_gpu = gpu_memory_mib()
    peak_gpu = idle_gpu
    peak_rss = 0
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            peak_rss = max(peak_rss, process_tree_rss(process.pid))
            used = gpu_memory_mib()
            if used is not None:
                peak_gpu = used if peak_gpu is None else max(peak_gpu, used)
            time.sleep(max(poll_seconds, 0.1))
        returncode = process.returncode
    result = {
        "command": command,
        "returncode": returncode,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_tree_rss_bytes": peak_rss,
        "idle_gpu_memory_mib": idle_gpu,
        "peak_global_gpu_memory_mib": peak_gpu,
    }
    if idle_gpu is not None and peak_gpu is not None:
        result["peak_global_gpu_memory_delta_mib"] = max(peak_gpu - idle_gpu, 0)
    return result


def parse_pinhole(calib: Path) -> tuple[float, float, float, float]:
    line = next(line for line in calib.read_text(encoding="utf-8").splitlines() if line.startswith("P0:"))
    values = [float(value) for value in line.split()[1:]]
    return values[0], values[5], values[2], values[6]


def registered_images(images_txt: Path) -> int:
    count = 0
    for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 10:
            continue
        try:
            int(fields[0])
            [float(value) for value in fields[1:8]]
            int(fields[8])
        except ValueError:
            continue
        count += 1
    return count


def point_statistics(points_txt: Path) -> tuple[int, float | None]:
    errors = []
    for line in points_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 8:
            errors.append(float(fields[7]))
    return len(errors), sum(errors) / len(errors) if errors else None


def model_candidates(root: Path) -> list[Path]:
    if (root / "images.bin").is_file() or (root / "images.txt").is_file():
        return [root]
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and ((path / "images.bin").is_file() or (path / "images.txt").is_file())
    )


def convert_best_model(colmap: Path, model_root: Path, text_root: Path, logs: Path) -> Path:
    candidates = model_candidates(model_root)
    if not candidates:
        raise RuntimeError(f"no COLMAP model below {model_root}")
    scored = []
    for index, candidate in enumerate(candidates):
        output = text_root / str(index)
        output.mkdir(parents=True)
        command = [
            str(colmap),
            "model_converter",
            "--input_path",
            str(candidate),
            "--output_path",
            str(output),
            "--output_type",
            "TXT",
        ]
        with (logs / f"convert_{model_root.name}_{index}.log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        scored.append((registered_images(output / "images.txt"), output))
    return max(scored, key=lambda item: item[0])[1]


def main() -> int:
    args = parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if args.sequence not in protocol["selection"]["held_out_sequences"]:
        raise ValueError(f"sequence is not frozen held-out data: {args.sequence}")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    prepared_manifest_path = args.prepared_dir / "manifest.json"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    if prepared_manifest["sequence"] != args.sequence:
        raise ValueError("prepared sequence mismatch")
    if prepared_manifest["protocol_sha256"] != hashlib.sha256(protocol_bytes).hexdigest():
        raise ValueError("prepared inputs use a different frozen protocol")
    if prepared_manifest["expected_frames"] != args.expected_frames:
        raise ValueError("prepared frame count mismatch")
    for path in (args.colmap, args.ground_truth_csv):
        if not path.is_file():
            raise FileNotFoundError(path)

    images = args.prepared_dir / "rect" / "image_0"
    timestamps = args.prepared_dir / "rect" / "timestamps.txt"
    fx, fy, cx, cy = parse_pinhole(args.prepared_dir / "rect" / "calib.txt")
    args.out_dir.mkdir(parents=True)
    logs = args.out_dir / "logs"
    logs.mkdir()
    database = args.out_dir / "database.db"
    started_utc = timestamp()
    stages = {}
    stages["feature_extraction"] = run_monitored(
        [
            str(args.colmap),
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.camera_params",
            f"{fx},{fy},{cx},{cy}",
            "--FeatureExtraction.type",
            "SIFT",
            "--FeatureExtraction.use_gpu",
            "1",
        ],
        logs / "feature_extraction.log",
        args.poll_seconds,
    )
    stages["sequential_matching"] = run_monitored(
        [
            str(args.colmap),
            "sequential_matcher",
            "--database_path",
            str(database),
            "--FeatureMatching.type",
            "SIFT_BRUTEFORCE",
            "--FeatureMatching.use_gpu",
            "1",
        ],
        logs / "sequential_matching.log",
        args.poll_seconds,
    )

    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    colmap_info = {
        "path": str(args.colmap.resolve()),
        "sha256": sha256(args.colmap),
        "version": subprocess.check_output([str(args.colmap), "version"], text=True).strip(),
    }

    def finish(status: str, results: dict, ground_truth_read: bool = False) -> int:
        manifest = {
            "schema_version": 1,
            "baseline_id": BASELINE_ID,
            "status": status,
            "sequence": args.sequence,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha256,
            "prepared_manifest_sha256": sha256(prepared_manifest_path),
            "ground_truth_used_after_all_engines_exited": True,
            "ground_truth_read": ground_truth_read,
            "expected_frames": args.expected_frames,
            "started_utc": started_utc,
            "finished_utc": timestamp(),
            "colmap": colmap_info,
            "camera": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "host": platform.platform(),
            "stages": stages,
            "results": results,
        }
        manifest_path = args.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(manifest_path)
        return 0

    failed_upstream = next(
        (
            name
            for name in ("feature_extraction", "sequential_matching")
            if stages[name]["returncode"] != 0
        ),
        None,
    )
    if failed_upstream is not None:
        return finish(
            "dnf",
            {
                engine: {
                    "status": "dnf",
                    "reason": f"shared {failed_upstream} returned nonzero",
                }
                for engine in ("incremental", "global")
            },
        )

    mapper_roots = {
        "incremental": args.out_dir / "incremental_models",
        "global": args.out_dir / "global_models",
    }
    for root in mapper_roots.values():
        root.mkdir()
    stages["incremental_mapping"] = run_monitored(
        [
            str(args.colmap),
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(mapper_roots["incremental"]),
            "--Mapper.random_seed",
            "0",
            "--Mapper.ba_local_backend",
            "CERES",
            "--Mapper.ba_global_backend",
            "CERES",
            "--Mapper.ba_use_gpu",
            "0",
        ],
        logs / "incremental_mapping.log",
        args.poll_seconds,
    )
    stages["global_mapping"] = run_monitored(
        [
            str(args.colmap),
            "global_mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images),
            "--output_path",
            str(mapper_roots["global"]),
            "--GlobalMapper.random_seed",
            "0",
        ],
        logs / "global_mapping.log",
        args.poll_seconds,
    )

    models = {}
    trajectories = {}
    results = {}
    for engine, root in mapper_roots.items():
        mapping_stage = stages[f"{engine}_mapping"]
        if mapping_stage["returncode"] != 0:
            results[engine] = {
                "status": "dnf",
                "reason": f"{engine}_mapping returned {mapping_stage['returncode']}",
            }
            continue
        try:
            text_model = convert_best_model(
                args.colmap, root, args.out_dir / f"{engine}_text", logs
            )
        except Exception as error:
            results[engine] = {
                "status": "dnf",
                "reason": f"model conversion failed: {type(error).__name__}: {error}",
            }
            continue
        trajectory = args.out_dir / f"{engine}.tum"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "colmap_images_to_tum.py"),
                str(text_model / "images.txt"),
                str(timestamps),
                str(trajectory),
            ],
            cwd=REPO,
            check=True,
        )
        models[engine] = text_model
        trajectories[engine] = trajectory

    evaluations = {}
    for engine, trajectory in trajectories.items():
        output = args.out_dir / f"evaluation_{engine}.json"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "evaluate_euroc_trajectory.py"),
                "--ground-truth-csv",
                str(args.ground_truth_csv),
                "--trajectory",
                str(trajectory),
                "--tum-time-unit",
                "s",
                "--out-json",
                str(output),
            ],
            cwd=REPO,
            check=True,
        )
        evaluations[engine] = json.loads(output.read_text(encoding="utf-8"))["runs"][0]

    common_evaluations = {}
    if len(trajectories) == 2:
        output = args.out_dir / "evaluation_common_frames.json"
        command = [
            sys.executable,
            str(REPO / "scripts" / "evaluate_euroc_trajectory.py"),
            "--ground-truth-csv",
            str(args.ground_truth_csv),
        ]
        for trajectory in trajectories.values():
            command.extend(["--trajectory", str(trajectory)])
        command.extend(
            [
                "--tum-time-unit",
                "s",
                "--common-estimate-timestamps",
                "--out-json",
                str(output),
            ]
        )
        subprocess.run(command, cwd=REPO, check=True)
        common_runs = json.loads(output.read_text(encoding="utf-8"))["runs"]
        common_evaluations = dict(zip(trajectories, common_runs))

    shared_wall = stages["feature_extraction"]["wall_seconds"] + stages["sequential_matching"]["wall_seconds"]
    for engine in trajectories:
        points, reprojection = point_statistics(models[engine] / "points3D.txt")
        mapping_stage = stages[f"{engine}_mapping"]
        engine_stages = [
            stages["feature_extraction"],
            stages["sequential_matching"],
            mapping_stage,
        ]
        gpu_peaks = [
            stage["peak_global_gpu_memory_mib"]
            for stage in engine_stages
            if stage["peak_global_gpu_memory_mib"] is not None
        ]
        results[engine] = {
            "status": "success",
            "registered_images": registered_images(models[engine] / "images.txt"),
            "points3d": points,
            "mean_reprojection_px": reprojection,
            "total_wall_seconds": shared_wall + mapping_stage["wall_seconds"],
            "peak_process_tree_rss_bytes": max(
                stage["peak_process_tree_rss_bytes"] for stage in engine_stages
            ),
            "peak_global_gpu_memory_mib": max(gpu_peaks) if gpu_peaks else None,
            "evaluation": evaluations[engine],
            "common_frame_evaluation": common_evaluations.get(engine),
            "model": str(models[engine].resolve()),
        }
    suite_status = "success" if all(
        result["status"] == "success" for result in results.values()
    ) else "partial_or_dnf"
    return finish(suite_status, results, ground_truth_read=bool(trajectories))


if __name__ == "__main__":
    raise SystemExit(main())
