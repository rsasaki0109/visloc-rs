#!/usr/bin/env python3
"""Export COLMAP keypoints and verified correspondences for mapper diagnostics.

The output contains no COLMAP poses or 3D points.  It transfers only the
frontend state immediately before mapping: image keypoints and rows from
``two_view_geometries``.  A diagnostic-only mode can instead export chain
edges that preserve final COLMAP track membership while discarding every 3D
coordinate, pose, color, and error. Keypoints are shifted by -0.5 px to undo
the OpenCV-to-COLMAP pixel-centre shift used by the OpenLORIS control runner.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
from pathlib import Path

MAGIC = b"VISLOC-COLMAP-1\0"
MAX_IMAGE_ID = 2_147_483_647


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_image_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("image_names")
    if names is None:
        images = payload.get("images")
        if isinstance(images, list) and all(isinstance(row, dict) for row in images):
            names = [row.get("name") for row in images]
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError(
            f"{path} does not contain a non-empty image_names list or images[].name records"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{path} repeats image names")
    return names


def decode_pair_id(pair_id: int) -> tuple[int, int]:
    image_j = pair_id % MAX_IMAGE_ID
    image_i = (pair_id - image_j) // MAX_IMAGE_ID
    return image_i, image_j


def load_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = path.read_text(encoding="utf-8").splitlines()
    aliases: dict[str, str] = {}
    for line_number, line in enumerate(rows, 1):
        if not line.strip() or line.startswith("flat_name\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number} must contain flat_name and colmap_name")
        flat_name, colmap_name = fields
        if colmap_name in aliases:
            raise ValueError(f"{path}:{line_number} repeats {colmap_name!r}")
        aliases[colmap_name] = flat_name
    return aliases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--image-manifest-json", type=Path, required=True)
    parser.add_argument("--image-aliases", type=Path)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--pairs-bin", type=Path, required=True)
    parser.add_argument("--feature-suffix", default="_features.txt")
    parser.add_argument("--keypoint-shift", type=float, default=-0.5)
    parser.add_argument(
        "--model-points3d",
        type=Path,
        action="append",
        help="diagnostic-only points3D.txt membership; repeat for disconnected models",
    )
    parser.add_argument(
        "--membership-seed-image-ids",
        help="optional COLMAP IMAGE_ID_I,IMAGE_ID_J made adjacent in every imported track",
    )
    args = parser.parse_args()

    names = load_image_names(args.image_manifest_json)
    aliases = load_aliases(args.image_aliases)
    by_basename = {Path(name).name: index for index, name in enumerate(names)}
    if len(by_basename) != len(names):
        raise ValueError("image manifest basenames are not unique")

    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    database_images = connection.execute("SELECT image_id, name FROM images").fetchall()
    id_to_index: dict[int, int] = {}
    for row in database_images:
        database_name = str(row["name"])
        basename = Path(aliases.get(database_name, database_name)).name
        if basename not in by_basename:
            raise ValueError(f"database image {row['name']!r} is absent from the manifest")
        id_to_index[int(row["image_id"])] = by_basename[basename]
    if len(id_to_index) != len(names):
        raise ValueError(f"database has {len(id_to_index)} mapped images, expected {len(names)}")

    feature_counts = [0] * len(names)
    args.features_dir.mkdir(parents=True, exist_ok=True)
    for row in connection.execute("SELECT image_id, rows, cols, data FROM keypoints"):
        image_id, count, columns = int(row[0]), int(row[1]), int(row[2])
        if image_id not in id_to_index or columns < 2 or row[3] is None:
            raise ValueError(f"invalid keypoint row for image_id={image_id}")
        values = struct.unpack(f"<{count * columns}f", row[3])
        index = id_to_index[image_id]
        feature_counts[index] = count
        lines = [
            f"{values[offset] + args.keypoint_shift:.9g} "
            f"{values[offset + 1] + args.keypoint_shift:.9g}\n"
            for offset in range(0, len(values), columns)
        ]
        stem = Path(names[index]).stem
        atomic_write_text(args.features_dir / f"{stem}{args.feature_suffix}", "".join(lines))
    if any(count == 0 for count in feature_counts):
        raise ValueError("one or more manifest images have no exported keypoints")

    track_pairs: dict[tuple[int, int], list[tuple[int, int]]] | None = None
    source_tracks = 0
    repaired_conflicting_tracks = 0
    removed_duplicate_image_observations = 0
    if args.model_points3d:
        seed_images: tuple[int, int] | None = None
        if args.membership_seed_image_ids:
            left_raw, right_raw = args.membership_seed_image_ids.split(",", 1)
            left_id, right_id = int(left_raw), int(right_raw)
            if left_id not in id_to_index or right_id not in id_to_index:
                raise ValueError("--membership-seed-image-ids names an unknown database image")
            seed_images = (id_to_index[left_id], id_to_index[right_id])
        track_pairs = {}
        owned_observations: set[tuple[int, int]] = set()
        for points_path in args.model_points3d:
            for line_number, line in enumerate(points_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 12 or (len(fields) - 8) % 2:
                    raise ValueError(f"{points_path}:{line_number} is not a COLMAP point row")
                observations: list[tuple[int, int]] = []
                for offset in range(8, len(fields), 2):
                    image_id = int(fields[offset])
                    keypoint = int(fields[offset + 1])
                    if image_id not in id_to_index:
                        raise ValueError(f"{points_path}:{line_number} uses unknown image {image_id}")
                    image = id_to_index[image_id]
                    if keypoint < 0 or keypoint >= feature_counts[image]:
                        raise ValueError(
                            f"{points_path}:{line_number} keypoint ({image},{keypoint}) is out of range"
                        )
                    observation = (image, keypoint)
                    observations.append(observation)
                observations.sort()
                if len({image for image, _ in observations}) != len(observations):
                    repaired_conflicting_tracks += 1
                    deduplicated: list[tuple[int, int]] = []
                    for observation in observations:
                        if deduplicated and deduplicated[-1][0] == observation[0]:
                            removed_duplicate_image_observations += 1
                            continue
                        deduplicated.append(observation)
                    observations = deduplicated
                if len(observations) < 2:
                    continue
                for observation in observations:
                    if observation in owned_observations:
                        raise ValueError(
                            f"{points_path}:{line_number} repeats owned observation {observation}"
                        )
                    owned_observations.add(observation)
                source_tracks += 1
                if seed_images is not None and all(image in {x[0] for x in observations} for image in seed_images):
                    by_image = {image: (image, keypoint) for image, keypoint in observations}
                    observations = [by_image[seed_images[0]], by_image[seed_images[1]]] + [
                        observation
                        for observation in observations
                        if observation[0] not in seed_images
                    ]
                for left, right in zip(observations, observations[1:]):
                    (image_i, keypoint_i), (image_j, keypoint_j) = left, right
                    if image_i > image_j:
                        image_i, image_j = image_j, image_i
                        keypoint_i, keypoint_j = keypoint_j, keypoint_i
                    track_pairs.setdefault((image_i, image_j), []).append(
                        (keypoint_i, keypoint_j)
                    )
        pair_count = len(track_pairs)
    else:
        pair_count = int(
            connection.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0").fetchone()[0]
        )
    args.pairs_bin.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.pairs_bin.with_name(f".{args.pairs_bin.name}.tmp")
    accepted = 0
    with temporary.open("wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<Q", len(names)))
        for name, count in zip(names, feature_counts):
            encoded = name.encode("utf-8")
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            output.write(struct.pack("<Q", count))
        output.write(struct.pack("<Q", pair_count))
        if track_pairs is not None:
            for (image_i, image_j), matches in sorted(track_pairs.items()):
                output.write(struct.pack("<QQQ", image_i, image_j, len(matches)))
                for keypoint_i, keypoint_j in matches:
                    output.write(struct.pack("<II", keypoint_i, keypoint_j))
                accepted += len(matches)
        else:
            for row in connection.execute(
                "SELECT pair_id, rows, cols, data FROM two_view_geometries "
                "WHERE rows > 0 ORDER BY pair_id"
            ):
                pair_id, count, columns, blob = int(row[0]), int(row[1]), int(row[2]), row[3]
                image_i, image_j = decode_pair_id(pair_id)
                if image_i not in id_to_index or image_j not in id_to_index:
                    raise ValueError(f"pair_id={pair_id} names an unknown image")
                if columns != 2 or blob is None or len(blob) != count * columns * 4:
                    raise ValueError(f"pair_id={pair_id} has malformed geometry data")
                output.write(struct.pack("<QQQ", id_to_index[image_i], id_to_index[image_j], count))
                output.write(blob)
                accepted += count
    temporary.replace(args.pairs_bin)
    connection.close()
    print(
        f"exported {len(names)} images / {sum(feature_counts)} keypoints / "
        f"{pair_count} verified pairs / {accepted} correspondences"
        + (f" / {source_tracks} membership tracks" if track_pairs is not None else "")
        + (
            f" / {repaired_conflicting_tracks} repaired conflicting tracks"
            f" / {removed_duplicate_image_observations} removed duplicate-image observations"
            if track_pairs is not None
            else ""
        )
        + f" -> {args.pairs_bin}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
