#!/usr/bin/env python3
"""Reconstruct a matches-import.txt file from a frozen verified-pair snapshot.

`unordered_sfm_demo --import-matches-file` consumes a plain-text dump of raw
matches: an image-name preamble bound to a candidate manifest's global image
indices, followed by one block per pair listing its correspondence indices.
No exporter for this format existed in the repo, so re-verifying a looser-ratio
match snapshot under a stricter ratio had no reproducible input path. This
script reconstructs the file deterministically from the snapshot.

The image-name preamble comes from a `visloc_candidate_manifest_v1` file (its
image order must match the snapshot's own image order -- both are bound to
the same feature-bank tier); the pairs and their correspondences come from
the snapshot itself via the `inspect_verified_pair_snapshot` example binary.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

from benchmark_electro import parse_candidate_manifest_with_metadata

PAIR_MATCHES_HEADER = "pair_match_index_i\tpair_match_index_j"


def parse_pairs_tsv(text):
    """Return the ordered (image_i, image_j) pairs from `--pairs-tsv` output.

    The row order is the snapshot's own internal pair order (its
    `pair_order_hash`), which is also the order `matches-import.txt` must
    list pair blocks in.
    """

    lines = text.splitlines()
    if not lines or not lines[0].startswith("image_i\timage_j"):
        raise ValueError(f"unexpected pairs.tsv header: {lines[:1]!r}")
    pairs = []
    for line in lines[1:]:
        if not line:
            continue
        fields = line.split("\t")
        pairs.append((int(fields[0]), int(fields[1])))
    return pairs


def parse_images_tsv(text):
    """Return the snapshot's own image names, indexed by their global index."""

    lines = text.splitlines()
    if not lines or not lines[0].startswith("image_index\timage_name"):
        raise ValueError(f"unexpected images.tsv header: {lines[:1]!r}")
    names = {}
    for line in lines[1:]:
        if not line:
            continue
        index, name, _ = line.split("\t")
        names[int(index)] = name
    if set(names) != set(range(len(names))):
        raise ValueError("images.tsv does not cover a dense 0..N index range")
    return [names[index] for index in range(len(names))]


def parse_pair_matches(text):
    """Parse `--pair-matches I,J` stdout into `[(query_index, train_index), ...]`."""

    lines = text.splitlines()
    if not lines or lines[0] != PAIR_MATCHES_HEADER:
        raise ValueError(f"unexpected pair-matches header: {lines[:1]!r}")
    correspondences = []
    for line in lines[1:]:
        if line.startswith("images=") or line.startswith("verifier_config_hash"):
            break
        if not line:
            continue
        query, train = line.split("\t")
        correspondences.append((int(query), int(train)))
    return correspondences


def build_matches_import_lines(names, pairs, correspondences_by_pair):
    """Render `matches-import.txt` contents (the `--import-matches-file` format)."""

    lines = [str(len(names))]
    lines.extend(names)
    lines.append(str(len(pairs)))
    for pair in pairs:
        correspondences = correspondences_by_pair[pair]
        lines.append(f"{pair[0]} {pair[1]} {len(correspondences)}")
        lines.extend(f"{query} {train}" for (query, train) in correspondences)
    return lines


def run_inspect(binary, snapshot, *extra_args):
    result = subprocess.run(
        [str(binary), "--snapshot", str(snapshot), *extra_args],
        check=True, capture_output=True, text=True,
    )
    return result.stdout


def export_matches_import(candidate_manifest, snapshot, inspect_binary):
    """Return `(names, pairs, correspondences_by_pair)` reconstructed from `snapshot`."""

    names, _, _ = parse_candidate_manifest_with_metadata(candidate_manifest)
    with tempfile.TemporaryDirectory() as tmp_dir:
        images_tsv = Path(tmp_dir) / "images.tsv"
        pairs_tsv = Path(tmp_dir) / "pairs.tsv"
        run_inspect(
            inspect_binary, snapshot,
            "--images-tsv", str(images_tsv), "--pairs-tsv", str(pairs_tsv),
        )
        snapshot_names = parse_images_tsv(images_tsv.read_text(encoding="utf-8"))
        pairs = parse_pairs_tsv(pairs_tsv.read_text(encoding="utf-8"))
    if snapshot_names != names:
        raise ValueError(
            f"candidate manifest {candidate_manifest} image order does not match "
            f"snapshot {snapshot} (they must share the same feature-bank tier)"
        )
    correspondences_by_pair = {}
    for pair in pairs:
        stdout = run_inspect(inspect_binary, snapshot, "--pair-matches", f"{pair[0]},{pair[1]}")
        correspondences_by_pair[pair] = parse_pair_matches(stdout)
    return names, pairs, correspondences_by_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-manifest", type=Path, required=True,
        help="visloc_candidate_manifest_v1 file supplying the image-name preamble",
    )
    parser.add_argument(
        "--snapshot", type=Path, required=True,
        help="verified-pair snapshot (.vps) to reconstruct raw matches from",
    )
    parser.add_argument(
        "--inspect-binary", type=Path,
        default=Path("target/release/examples/inspect_verified_pair_snapshot"),
        help="path to the inspect_verified_pair_snapshot example binary",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate_manifest.is_file():
        parser.error(f"candidate manifest not found: {args.candidate_manifest}")
    if not args.snapshot.is_file():
        parser.error(f"snapshot not found: {args.snapshot}")
    if not args.inspect_binary.is_file():
        parser.error(f"inspect binary not found: {args.inspect_binary}")
    names, pairs, correspondences_by_pair = export_matches_import(
        args.candidate_manifest, args.snapshot, args.inspect_binary,
    )
    lines = build_matches_import_lines(names, pairs, correspondences_by_pair)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total_correspondences = sum(len(c) for c in correspondences_by_pair.values())
    print(f"{len(names)} images, {len(pairs)} pairs, {total_correspondences} correspondences -> {args.output}")


if __name__ == "__main__":
    main()
