#!/usr/bin/env python3
"""Download and verify the official archives bound by an SSfM protocol."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--curl", type=Path, default=Path("curl.exe"))
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm.lower())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pid_exists(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def download(curl: Path, archive: dict, out_dir: Path) -> dict:
    final = out_dir / archive["name"]
    partial = out_dir / f"{archive['name']}.partial"
    expected_size = int(archive["size_bytes"])
    expected_hash = archive["checksum"].lower()
    algorithm = archive["checksum_algorithm"]

    if final.exists():
        if final.stat().st_size != expected_size:
            raise ValueError(f"existing {final} has the wrong size")
        actual_hash = file_hash(final, algorithm)
        if actual_hash != expected_hash:
            raise ValueError(f"existing {final} has the wrong {algorithm}")
        return {
            "name": archive["name"],
            "status": "verified_existing",
            "path": str(final.resolve()),
            "size_bytes": expected_size,
            "checksum_algorithm": algorithm,
            "checksum": actual_hash,
        }

    command = [
        str(curl),
        "--location",
        "--fail",
        "--retry",
        "10",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        archive["content_url"],
    ]
    print("COMMAND:", subprocess.list2cmdline(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    wall_seconds = time.perf_counter() - started
    if partial.stat().st_size != expected_size:
        raise ValueError(
            f"{partial} is {partial.stat().st_size} bytes, expected {expected_size}"
        )
    actual_hash = file_hash(partial, algorithm)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{partial} {algorithm}={actual_hash}, expected {expected_hash}"
        )
    partial.replace(final)
    return {
        "name": archive["name"],
        "status": "downloaded_and_verified",
        "path": str(final.resolve()),
        "size_bytes": expected_size,
        "checksum_algorithm": algorithm,
        "checksum": actual_hash,
        "wall_seconds": wall_seconds,
    }


def main() -> int:
    args = parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    archives = protocol["inputs"]["official_archives"]
    if not archives:
        raise ValueError("protocol has no official archives")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "download_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

    if args.wait_pid is not None:
        print(f"waiting for PID {args.wait_pid} before dataset download", flush=True)
        while pid_exists(args.wait_pid):
            time.sleep(15.0)

    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "started_utc": timestamp(),
        "wait_pid": args.wait_pid,
        "host": platform.platform(),
        "archives": [],
    }
    try:
        for archive in archives:
            manifest["archives"].append(download(args.curl, archive, args.out_dir))
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["finished_utc"] = timestamp()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise
    manifest["status"] = "success"
    manifest["finished_utc"] = timestamp()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
