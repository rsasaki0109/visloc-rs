#!/usr/bin/env python3
"""Audit WSL GLUEMAP and recheck InstantSfM before frozen held-out runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
GLUEMAP_CHECKPOINTS = (
    "checkpoints/pi3.safetensors",
    "checkpoints/dino_salad.ckpt",
    "checkpoints/vggsfm_v2_0_0_track_predictor.bin",
    "checkpoints/checkpoint-dg+visym.pth",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_capture(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def wsl(distro: str, *command: str) -> list[str]:
    return ["wsl.exe", "-d", distro, "--", *command]


def source_identity(
    revision: str, submodule_status: str, checkpoint_hashes: dict[str, str]
) -> str:
    value = json.dumps(
        {
            "revision": revision,
            "submodule_status": submodule_status.splitlines(),
            "checkpoint_sha256": checkpoint_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def parse_sha256sum(output: str, expected_path: str) -> str:
    fields = output.strip().split(maxsplit=1)
    if len(fields) != 2 or fields[1].lstrip("*") != expected_path:
        raise ValueError(f"unexpected sha256sum output for {expected_path!r}")
    digest = fields[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"invalid SHA-256 for {expected_path!r}")
    return digest


def failed_setup(reason: str, revision: str, attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "install_failed",
        "reason": reason,
        "source_revision": revision,
        "attempt": attempt,
    }


def audit_gluemap(
    *,
    protocol: dict[str, Any],
    distro: str,
    source_dir: str,
    python: str,
    demo: str,
    wrapper: Path,
    inner_adapter: Path,
) -> dict[str, Any]:
    expected_revision = protocol["engines"]["gluemap"]["revision"]
    revision_attempt = run_capture(wsl(distro, "git", "-C", source_dir, "rev-parse", "HEAD"))
    revision = revision_attempt["stdout"].strip()
    if revision_attempt["returncode"] != 0 or revision != expected_revision:
        return failed_setup(
            f"GLUEMAP revision {revision!r} does not equal frozen {expected_revision}",
            revision or expected_revision,
            revision_attempt,
        )
    status_attempt = run_capture(
        wsl(distro, "git", "-C", source_dir, "status", "--porcelain", "--untracked-files=no")
    )
    if status_attempt["returncode"] != 0 or status_attempt["stdout"].strip():
        return failed_setup("GLUEMAP tracked worktree is not clean", revision, status_attempt)
    submodule_attempt = run_capture(
        wsl(distro, "git", "-C", source_dir, "submodule", "status", "--recursive")
    )
    submodule_lines = [line for line in submodule_attempt["stdout"].splitlines() if line.strip()]
    if (
        submodule_attempt["returncode"] != 0
        or not submodule_lines
        or any(line[0] in "-+U" for line in submodule_lines)
    ):
        return failed_setup(
            "GLUEMAP recursive submodules are missing or differ from the frozen commit",
            revision,
            submodule_attempt,
        )

    checkpoint_hashes: dict[str, str] = {}
    checkpoint_attempts = []
    for relative in GLUEMAP_CHECKPOINTS:
        full_path = source_dir.rstrip("/") + "/" + relative
        attempt = run_capture(wsl(distro, "sha256sum", full_path))
        checkpoint_attempts.append(attempt)
        if attempt["returncode"] != 0:
            return failed_setup(f"missing GLUEMAP checkpoint {relative}", revision, attempt)
        try:
            checkpoint_hashes[relative] = parse_sha256sum(attempt["stdout"], full_path)
        except ValueError as error:
            return failed_setup(str(error), revision, attempt)

    import_attempt = run_capture(
        wsl(
            distro,
            python,
            "-c",
            "import gluemap,pygluemap,torch; print(pygluemap.__file__); print(torch.cuda.is_available())",
        )
    )
    import_lines = import_attempt["stdout"].splitlines()
    if (
        import_attempt["returncode"] != 0
        or len(import_lines) < 2
        or import_lines[-1].strip() != "True"
    ):
        return failed_setup("GLUEMAP import or CUDA availability check failed", revision, import_attempt)
    help_attempt = run_capture(wsl(distro, demo, "--help"))
    if help_attempt["returncode"] != 0:
        return failed_setup("gluemap-demo --help failed", revision, help_attempt)

    return {
        "status": "ready",
        "source_revision": revision,
        "source_tree_sha256": source_identity(
            revision, submodule_attempt["stdout"], checkpoint_hashes
        ),
        "source_dir_wsl": source_dir,
        "recursive_submodules": submodule_lines,
        "checkpoint_sha256": checkpoint_hashes,
        "setup_attempts": {
            "revision": revision_attempt,
            "worktree": status_attempt,
            "submodules": submodule_attempt,
            "checkpoints": checkpoint_attempts,
            "import": import_attempt,
            "help": help_attempt,
        },
        "adapter": {
            "path": str(wrapper.resolve()),
            "sha256": sha256(wrapper),
            "dependencies": [
                {"path": str(inner_adapter.resolve()), "sha256": sha256(inner_adapter)}
            ],
            "command_template": [
                sys.executable,
                "{adapter}",
                "--distro",
                distro,
                "--wsl-python",
                python,
                "--wsl-source-dir",
                source_dir,
                "--inner-adapter",
                str(inner_adapter.resolve()),
                "--images-path",
                "{images_path}",
                "--calibration-path",
                "{calibration_path}",
                "--timestamps-path",
                "{timestamps_path}",
                "--output-path",
                "{output_path}",
                "--expected-frames",
                "{expected_frames}",
            ],
        },
    }


def audit_instantsfm(protocol: dict[str, Any]) -> dict[str, Any]:
    engine = protocol["engines"]["instantsfm"]
    urls = [engine["published_repository"], engine["observed_browser_redirect"] + ".git"]
    attempts = [run_capture(["git", "ls-remote", url, "HEAD"]) for url in urls]
    available = [attempt for attempt in attempts if attempt["returncode"] == 0 and attempt["stdout"].strip()]
    if available:
        raise RuntimeError(
            "InstantSfM source is available now; bind its exact revision and implement the official adapter"
        )
    return {
        "status": "source_unavailable",
        "reason": "published and redirected InstantSfM Git repositories remain unavailable at setup time",
        "source_identity": "instant-sfm-published-source-outage",
        "source_revision": None,
        "retrieval_attempts": attempts,
        "attempt": attempts[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-protocol", type=Path, required=True)
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument("--gluemap-source-wsl", required=True)
    parser.add_argument("--gluemap-python-wsl", required=True)
    parser.add_argument("--gluemap-demo-wsl", required=True)
    parser.add_argument(
        "--wrapper", type=Path, default=REPO / "scripts" / "run_gluemap_ssfm_wsl.py"
    )
    parser.add_argument(
        "--inner-adapter",
        type=Path,
        default=REPO / "scripts" / "run_gluemap_ssfm_adapter.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.external_protocol, args.wrapper, args.inner_adapter):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol_bytes = args.external_protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    engines = {
        "gluemap": audit_gluemap(
            protocol=protocol,
            distro=args.distro,
            source_dir=args.gluemap_source_wsl,
            python=args.gluemap_python_wsl,
            demo=args.gluemap_demo_wsl,
            wrapper=args.wrapper,
            inner_adapter=args.inner_adapter,
        ),
        "instantsfm": audit_instantsfm(protocol),
    }
    output = {
        "schema_version": 1,
        "created_utc": timestamp(),
        "external_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "host_execution": f"WSL2 {args.distro}",
        "engines": engines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
