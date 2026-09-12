#!/usr/bin/env python3
"""Export an ALIKED+LightGlue oracle for a frozen, bounded pair manifest.

Only images incident to the declared pairs are passed through ALIKED. Features
are persisted one image at a time and reloaded one pair at a time for
LightGlue, so the script never retains an image-count-sized descriptor bank.
The remaining image files contain a single inert descriptor solely to preserve
the manifest's physical image indices for the Rust verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


MAGIC = "visloc_candidate_manifest_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def parse_candidates(path: Path) -> tuple[list[str], list[tuple[int, int]]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines or lines[0] != MAGIC:
        raise ValueError(f"{path} has no {MAGIC} header")
    if len(lines) < 3 or not lines[1].startswith("images "):
        raise ValueError("candidate manifest is truncated")
    count = int(lines[1].split()[1])
    names: list[str] = []
    cursor = 2
    for expected in range(count):
        fields = lines[cursor].split(maxsplit=2)
        cursor += 1
        if len(fields) != 3 or fields[:2] != ["image", str(expected)]:
            raise ValueError(f"candidate image row {expected} is malformed")
        names.append(fields[2])
    while cursor < len(lines) and lines[cursor].startswith("metadata "):
        fields = lines[cursor].split(maxsplit=2)
        if len(fields) != 3:
            raise ValueError("candidate metadata row is malformed")
        cursor += 1
    if cursor >= len(lines) or not lines[cursor].startswith("pairs "):
        raise ValueError("candidate pair count is missing")
    pair_count = int(lines[cursor].split()[1])
    cursor += 1
    pairs: list[tuple[int, int]] = []
    for _ in range(pair_count):
        fields = lines[cursor].split()
        cursor += 1
        if len(fields) != 3 or fields[0] != "pair":
            raise ValueError("candidate pair row is malformed")
        pair = (int(fields[1]), int(fields[2]))
        if not 0 <= pair[0] < pair[1] < count or pair in pairs:
            raise ValueError(f"candidate pair is invalid or repeated: {pair}")
        pairs.append(pair)
    if cursor != len(lines):
        raise ValueError("candidate manifest has trailing rows")
    return names, pairs


def feature_text(keypoints, scores, descriptors) -> str:
    rows = ["# X Y SCORE D0 D1 ..."]
    for xy, score, descriptor in zip(keypoints, scores, descriptors):
        values = [xy[0], xy[1], score, *descriptor]
        rows.append(" ".join(f"{float(value):.9g}" for value in values))
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--lightglue-repo", type=Path, required=True)
    parser.add_argument("--torch-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to replace output directory: {args.out_dir}")
    if args.max_keypoints <= 0 or args.threads <= 0:
        parser.error("--max-keypoints and --threads must be positive")

    names, pairs = parse_candidates(args.candidate_manifest)
    incident = sorted({index for pair in pairs for index in pair})
    missing = [name for name in names if not (args.images_dir / name).is_file()]
    if missing:
        parser.error(f"missing {len(missing)} images; first is {missing[0]}")
    commit = subprocess.run(
        ["git", "-C", str(args.lightglue_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    sys.path.insert(0, str(args.lightglue_repo))
    os.environ["TORCH_HOME"] = str(args.torch_cache)
    import numpy as np
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image, rbd

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    extractor = ALIKED(max_num_keypoints=args.max_keypoints).eval()
    matcher = LightGlue(features="aliked").eval()

    args.out_dir.mkdir(parents=True)
    features_dir = args.out_dir / "features"
    cache_dir = args.out_dir / "feature-cache"
    features_dir.mkdir()
    cache_dir.mkdir()
    extraction_started = time.monotonic()
    for position, index in enumerate(incident, 1):
        image = load_image(args.images_dir / names[index])
        with torch.inference_mode():
            features = rbd(extractor.extract(image))
        keypoints = features["keypoints"].detach().cpu().numpy()
        descriptors = features["descriptors"].detach().cpu().numpy()
        scores = features["keypoint_scores"].detach().cpu().numpy()
        image_size = features["image_size"].detach().cpu().numpy()
        stem = Path(names[index]).stem
        atomic_write(
            features_dir / f"{stem}_features.txt",
            feature_text(keypoints, scores, descriptors),
        )
        np.savez(
            cache_dir / f"{index:06}.npz",
            keypoints=keypoints,
            descriptors=descriptors,
            keypoint_scores=scores,
            image_size=image_size,
        )
        print(f"extract {position}/{len(incident)} image={index} keypoints={len(keypoints)}", flush=True)
        del image, features, keypoints, descriptors, scores, image_size
    extraction_seconds = time.monotonic() - extraction_started

    zero_descriptor = " ".join(["0"] * 128)
    inert = f"# inert non-candidate image\n0 0 0 {zero_descriptor}\n"
    incident_set = set(incident)
    for index, name in enumerate(names):
        if index not in incident_set:
            atomic_write(features_dir / f"{Path(name).stem}_features.txt", inert)

    match_rows: list[tuple[int, int, list[tuple[int, int]]]] = []
    matching_started = time.monotonic()
    for position, (left, right) in enumerate(pairs, 1):
        loaded = []
        for index in (left, right):
            with np.load(cache_dir / f"{index:06}.npz") as data:
                loaded.append({
                    key: torch.from_numpy(data[key]).unsqueeze(0)
                    for key in ("keypoints", "descriptors", "keypoint_scores", "image_size")
                })
        with torch.inference_mode():
            result = rbd(matcher({"image0": loaded[0], "image1": loaded[1]}))
        matches = [(int(row[0]), int(row[1])) for row in result["matches"].cpu().tolist()]
        match_rows.append((left, right, matches))
        print(f"match {position}/{len(pairs)} pair={left},{right} matches={len(matches)}", flush=True)
        del loaded, result
    matching_seconds = time.monotonic() - matching_started

    imported = [str(len(names)), *names, str(len(match_rows))]
    for left, right, matches in match_rows:
        imported.append(f"{left} {right} {len(matches)}")
        imported.extend(f"{query} {train}" for query, train in matches)
    matches_path = args.out_dir / "matches-import.txt"
    atomic_write(matches_path, "\n".join(imported) + "\n")

    checkpoints = sorted((args.torch_cache / "hub" / "checkpoints").glob("*.pth"))
    aggregate = hashlib.sha256()
    for path in sorted(features_dir.iterdir()):
        aggregate.update(path.name.encode())
        aggregate.update(bytes.fromhex(sha256(path)))
    report = {
        "schema": "visloc-bounded-aliked-lightglue-v1",
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "lightglue_commit": commit,
        "model_sha256": {path.name: sha256(path) for path in checkpoints},
        "images": len(names),
        "incident_images": len(incident),
        "pairs": len(pairs),
        "max_keypoints": args.max_keypoints,
        "threads": args.threads,
        "extraction_seconds": extraction_seconds,
        "matching_seconds": matching_seconds,
        "raw_matches": sum(len(matches) for _, _, matches in match_rows),
        "features_aggregate_sha256": aggregate.hexdigest(),
        "matches_import_sha256": sha256(matches_path),
        "nonincident_rows_are_inert": True,
        "ground_truth_used": False,
    }
    atomic_write(args.out_dir / "manifest.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
