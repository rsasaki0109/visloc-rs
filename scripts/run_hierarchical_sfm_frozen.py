#!/usr/bin/env python3
"""Run the frozen S2 hierarchical SfM configuration and capture its manifest.

Ground truth is intentionally not opened until the mapper process has exited.
The runner can wait for another benchmark PID so wall/RAM measurements are not
contaminated by a concurrently running dense control.
"""

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
FROZEN_CONFIG_ID = "s2-mh03-smoke-w88-104-o72-shared4-workers2-seamba5-v1"


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


def windows_process_table() -> dict[int, tuple[int, int]]:
    """Return PID -> (parent PID, working-set bytes), using only Win32 APIs."""
    if os.name != "nt":
        raise RuntimeError("this frozen benchmark runner currently requires Windows")
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
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    table: dict[int, tuple[int, int]] = {}
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
                if psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    rss = int(counters.WorkingSetSize)
                kernel32.CloseHandle(handle)
            table[pid] = (int(entry.th32ParentProcessID), rss)
            present = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--ground-truth-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--build-git-revision", required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--fx", type=float, default=436.2442956471)
    parser.add_argument("--fy", type=float, default=436.2442956471)
    parser.add_argument("--cx", type=float, default=364.4412345886)
    parser.add_argument("--cy", type=float, default=256.951675415)
    return parser.parse_args()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def wait_for_pid(pid: int) -> None:
    print(f"waiting for PID {pid} to exit before frozen S2 timing", flush=True)
    while pid in windows_process_table():
        time.sleep(15.0)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_file_count(directory: Path) -> int:
    return sum(1 for path in directory.iterdir() if path.name.endswith("_features.txt"))


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


def point_count(points_txt: Path) -> int:
    return sum(
        1
        for line in points_txt.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    )


def main() -> int:
    args = parse_args()
    for path in [args.exe, args.features_dir, args.timestamps, args.ground_truth_csv]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.expected_frames <= 0:
        raise ValueError("--expected-frames must be positive")
    input_frames = feature_file_count(args.features_dir)
    if input_frames != args.expected_frames:
        raise ValueError(
            f"feature input has {input_frames} frames, expected {args.expected_frames}"
        )
    timestamp_rows = sum(
        1
        for line in args.timestamps.read_text(encoding="utf-8").splitlines()
        if len(line.split()) >= 2
    )
    if timestamp_rows < args.expected_frames:
        raise ValueError(
            f"timestamp input has {timestamp_rows} rows, expected at least {args.expected_frames}"
        )
    executable_sha256 = sha256(args.exe)
    runner_git_revision = git_revision()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = args.out_dir / "model"
    log = args.out_dir / "mapping.log"
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() or model.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing frozen run: {args.out_dir}"
        )

    if args.wait_pid is not None:
        wait_for_pid(args.wait_pid)

    command = [
        str(args.exe),
        "--features-dir",
        str(args.features_dir),
        "--out-colmap",
        str(model),
        "--width",
        "752",
        "--height",
        "480",
        "--fx",
        str(args.fx),
        "--fy",
        str(args.fy),
        "--cx",
        str(args.cx),
        "--cy",
        str(args.cy),
        "--window",
        "5",
        "--min-matches",
        "30",
        "--colmap-style",
        "--next-image-policy",
        "visibility",
        "--skip-offsets",
        "8,12",
        "--skip-stride",
        "2",
        "--post-refinement-registration",
        "--structureless-registration",
        "--hierarchical",
        "--submap-min-images",
        "80",
        "--submap-target-images",
        "88",
        "--submap-max-images",
        "104",
        "--submap-overlap-images",
        "72",
        "--submap-boundary-search-radius",
        "0",
        "--submap-min-shared-observations",
        "4",
        "--submap-build-threads",
        "2",
        "--submap-seam-ba",
    ]
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    peak_rss = 0
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
            time.sleep(max(args.poll_seconds, 0.1))
        peak_rss = max(peak_rss, process_tree_rss(process.pid))
        returncode = process.returncode
    wall_seconds = time.perf_counter() - started

    base_manifest = {
        "schema_version": 1,
        "config_id": FROZEN_CONFIG_ID,
        "runner_git_revision": runner_git_revision,
        "build_git_revision": args.build_git_revision,
        "executable": {
            "path": str(args.exe.resolve()),
            "sha256": executable_sha256,
        },
        "started_utc": started_utc,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "protocol": {
            "ground_truth_used_after_engine_exit": True,
            "features_dir": str(args.features_dir.resolve()),
            "input_feature_frames": input_frames,
            "timestamp_rows": timestamp_rows,
            "expected_frames": args.expected_frames,
            "wait_pid": args.wait_pid,
        },
        "command": command,
        "mapper": {
            "returncode": returncode,
            "wall_seconds": wall_seconds,
            "peak_process_tree_rss_bytes": peak_rss,
        },
    }
    if returncode != 0:
        manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
        raise RuntimeError(f"mapper failed ({returncode}); see {log}")

    images_txt = model / "images.txt"
    points_txt = model / "points3D.txt"
    registered = registered_images(images_txt)
    if registered != args.expected_frames:
        base_manifest["mapper"]["registered_images"] = registered
        manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"registered {registered}/{args.expected_frames}; frozen completeness gate failed"
        )

    trajectory = args.out_dir / "trajectory.tum"
    evaluation = args.out_dir / "evaluation.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "colmap_images_to_tum.py"),
            str(images_txt),
            str(args.timestamps),
            str(trajectory),
        ],
        cwd=REPO,
        check=True,
    )
    # First ground-truth read occurs inside this subprocess, after mapper exit.
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
            str(evaluation),
        ],
        cwd=REPO,
        check=True,
    )
    evaluated = json.loads(evaluation.read_text(encoding="utf-8"))
    base_manifest["mapper"].update(
        {
            "registered_images": registered,
            "points3d": point_count(points_txt),
        }
    )
    base_manifest["evaluation"] = evaluated["runs"][0]
    manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
