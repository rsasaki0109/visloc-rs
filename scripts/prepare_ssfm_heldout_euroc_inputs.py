#!/usr/bin/env python3
"""Prepare frozen rectified images and SuperPoint features without reading GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark_process_metrics import run_monitored


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--mav0", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument(
        "--superpoint-checkpoint",
        type=Path,
        default=Path.home() / ".cache/torch/hub/checkpoints/superpoint_v1.pth",
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronized_image_names(mav0: Path) -> list[str]:
    left = {path.name for path in (mav0 / "cam0" / "data").glob("*.png")}
    right = {path.name for path in (mav0 / "cam1" / "data").glob("*.png")}
    names = sorted(left & right)
    if not names:
        raise FileNotFoundError(f"no synchronized stereo image names below {mav0}")
    return names


def feature_set_hash(features: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(features.glob("frame_*_features.txt"))
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest(), len(files)


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def python_environment(python: Path) -> dict:
    probe = (
        "import cv2,json,lightglue,torch;"
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'cv2':cv2.__version__,'lightglue_file':lightglue.__file__}))"
    )
    return json.loads(command_output([str(python), "-c", probe]))


def main() -> int:
    args = parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    allowed = protocol["selection"]["held_out_sequences"]
    if args.sequence not in allowed:
        raise ValueError(f"{args.sequence} is not frozen held-out data: {allowed}")
    if args.mav0.parent.name != args.sequence:
        raise ValueError(
            f"--mav0 parent is {args.mav0.parent.name}, expected {args.sequence}"
        )
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    for path in (
        args.python,
        args.superpoint_checkpoint,
        args.mav0 / "cam0" / "sensor.yaml",
        args.mav0 / "cam1" / "sensor.yaml",
        args.mav0 / "cam0" / "data.csv",
        args.mav0 / "cam1" / "data.csv",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    expected_frames = len(synchronized_image_names(args.mav0))
    args.out_dir.mkdir(parents=True)
    logs = args.out_dir / "logs"
    logs.mkdir()
    rect = args.out_dir / "rect"
    features = args.out_dir / "features"
    rectify_script = REPO / "scripts" / "rectify_euroc_stereo.py"
    feature_script = REPO / "scripts" / "export_superpoint_lightglue.py"

    started_utc = timestamp()
    stages = {}
    stages["rectification"] = run_monitored(
        [
            str(args.python),
            str(rectify_script),
            "--mav0",
            str(args.mav0),
            "--out-dir",
            str(rect),
            "--frames",
            str(expected_frames),
            "--left-only",
        ],
        logs / "rectification.log",
        cwd=REPO,
    )
    if stages["rectification"]["returncode"] != 0:
        raise RuntimeError("rectification failed")
    stages["superpoint"] = run_monitored(
        [
            str(args.python),
            str(feature_script),
            "--mono-dir",
            str(rect / "image_0"),
            "--out-dir",
            str(features),
            "--frames",
            str(expected_frames),
            "--device",
            args.device,
            "--max-keypoints",
            str(args.max_keypoints),
        ],
        logs / "superpoint.log",
        cwd=REPO,
    )
    if stages["superpoint"]["returncode"] != 0:
        raise RuntimeError("SuperPoint extraction failed")

    rectified_frames = len(list((rect / "image_0").glob("*.png")))
    timestamp_rows = len((rect / "timestamps.txt").read_text(encoding="utf-8").splitlines())
    validation_started = time.perf_counter()
    features_sha256, feature_frames = feature_set_hash(features)
    validation_seconds = time.perf_counter() - validation_started
    if (rectified_frames, timestamp_rows, feature_frames) != (
        expected_frames,
        expected_frames,
        expected_frames,
    ):
        raise RuntimeError(
            "full-sequence preparation failed: "
            f"expected={expected_frames} rectified={rectified_frames} "
            f"timestamps={timestamp_rows} features={feature_frames}"
        )

    manifest = {
        "schema_version": 1,
        "status": "success",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "sequence": args.sequence,
        "started_utc": started_utc,
        "finished_utc": timestamp(),
        "ground_truth_read": False,
        "full_sequence": True,
        "expected_frames": expected_frames,
        "rectified_frames": rectified_frames,
        "feature_frames": feature_frames,
        "timestamp_rows": timestamp_rows,
        "stage_seconds": {
            name: stage["wall_seconds"] for name, stage in stages.items()
        },
        "stages": stages,
        "post_timing_validation_seconds": validation_seconds,
        "configuration": {
            "device": args.device,
            "max_keypoints": args.max_keypoints,
            "camera": "cam0",
            "rectification_alpha": 0.0,
        },
        "inputs": {
            "mav0": str(args.mav0.resolve()),
            "cam0_sensor_sha256": sha256(args.mav0 / "cam0" / "sensor.yaml"),
            "cam1_sensor_sha256": sha256(args.mav0 / "cam1" / "sensor.yaml"),
            "cam0_index_sha256": sha256(args.mav0 / "cam0" / "data.csv"),
            "cam1_index_sha256": sha256(args.mav0 / "cam1" / "data.csv"),
            "superpoint_checkpoint": str(args.superpoint_checkpoint.resolve()),
            "superpoint_checkpoint_sha256": sha256(args.superpoint_checkpoint),
            "rectify_script_sha256": sha256(rectify_script),
            "feature_script_sha256": sha256(feature_script),
        },
        "outputs": {
            "rect": str(rect.resolve()),
            "features": str(features.resolve()),
            "features_sha256": features_sha256,
        },
        "host": {
            "platform": platform.platform(),
            "python": command_output([str(args.python), "--version"]),
            "python_environment": python_environment(args.python),
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
