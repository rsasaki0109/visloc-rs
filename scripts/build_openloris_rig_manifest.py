#!/usr/bin/env python3
"""Build the explicit generalized-rig replay manifest for an OpenLORIS tier.

The tier manifest is the authoritative image-to-camera/timestamp assignment.
The rig JSON is the exact frozen calibration consumed by the formal COLMAP
control, including COLMAP's +0.5 pixel-centre principal-point convention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


class ManifestError(ValueError):
    pass


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON {path}: {exc}") from exc


def finite_vector(value: object, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ManifestError(f"{label} must contain {length} values")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{label} contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in result):
        raise ManifestError(f"{label} contains a non-finite value")
    return result


def build(tier_path: Path, rig_path: Path, width: int, height: int) -> str:
    tier = load_json(tier_path)
    if not isinstance(tier, dict) or tier.get("schema") != "visloc_openloris_corridor_manifest_v1":
        raise ManifestError("unsupported OpenLORIS tier manifest schema")
    rows = tier.get("images")
    if not isinstance(rows, list) or not rows:
        raise ManifestError("tier manifest contains no images")

    rig_payload = load_json(rig_path)
    if not isinstance(rig_payload, list) or len(rig_payload) != 1:
        raise ManifestError("rig JSON must contain exactly one rig")
    rig = rig_payload[0]
    cameras = rig.get("cameras") if isinstance(rig, dict) else None
    if not isinstance(cameras, list) or len(cameras) != 2:
        raise ManifestError("OpenLORIS replay requires exactly two rig cameras")

    output = [
        "# generalized-rig-manifest-v1",
        "# S index camera_id width height fx fy cx cy qw qx qy qz tx ty tz",
    ]
    camera_number_to_sensor: dict[int, int] = {}
    for sensor_index, camera in enumerate(cameras):
        if not isinstance(camera, dict) or camera.get("camera_model_name") != "PINHOLE":
            raise ManifestError(f"sensor {sensor_index} is not PINHOLE")
        params = finite_vector(camera.get("camera_params"), 4, f"sensor {sensor_index} params")
        prefix = camera.get("image_prefix")
        if not isinstance(prefix, str) or not prefix.startswith("rig/camera"):
            raise ManifestError(f"sensor {sensor_index} has an invalid image_prefix")
        try:
            camera_number = int(prefix.removeprefix("rig/camera").split("/", 1)[0])
        except ValueError as exc:
            raise ManifestError(f"sensor {sensor_index} image_prefix has no camera number") from exc
        camera_number_to_sensor[camera_number] = sensor_index
        if camera.get("ref_sensor") is True:
            quaternion = [1.0, 0.0, 0.0, 0.0]
            translation = [0.0, 0.0, 0.0]
        else:
            quaternion = finite_vector(
                camera.get("cam_from_rig_rotation"), 4, f"sensor {sensor_index} quaternion"
            )
            translation = finite_vector(
                camera.get("cam_from_rig_translation"), 3, f"sensor {sensor_index} translation"
            )
        values = [
            "S",
            str(sensor_index),
            str(sensor_index + 1),
            str(width),
            str(height),
            *(format(value, ".17g") for value in params),
            *(format(value, ".17g") for value in quaternion),
            *(format(value, ".17g") for value in translation),
        ]
        output.append(" ".join(values))
    if set(camera_number_to_sensor) != {1, 2}:
        raise ManifestError("rig camera prefixes must identify camera1 and camera2")

    output.append("# F frame_id image_name sensor_index")
    timestamp_ids: dict[str, int] = {}
    frame_sensors: dict[int, set[int]] = {}
    seen_names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ManifestError("tier image row is not an object")
        try:
            name = str(row["name"])
            camera_number = int(row["camera"])
            timestamp = str(row["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError("tier image row is malformed") from exc
        if not name or any(character.isspace() for character in name) or name in seen_names:
            raise ManifestError(f"invalid or duplicate image name {name!r}")
        seen_names.add(name)
        if camera_number not in camera_number_to_sensor:
            raise ManifestError(f"image {name!r} uses unknown camera {camera_number}")
        frame_id = timestamp_ids.setdefault(timestamp, len(timestamp_ids))
        sensor_index = camera_number_to_sensor[camera_number]
        frame_sensors.setdefault(frame_id, set()).add(sensor_index)
        output.append(f"F {frame_id} {name} {sensor_index}")
    incomplete = [frame for frame, sensors in frame_sensors.items() if sensors != {0, 1}]
    if incomplete:
        raise ManifestError(f"incomplete stereo frame {min(incomplete)}")
    if len(seen_names) != 2 * len(frame_sensors):
        raise ManifestError("tier does not contain exactly two images per frame")
    return "\n".join(output) + "\n"


def write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier-manifest", type=Path, required=True)
    parser.add_argument("--rig-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    try:
        payload = build(args.tier_manifest, args.rig_config, args.width, args.height)
        write_atomic(args.output, payload)
    except ManifestError as exc:
        parser.error(str(exc))
    image_rows = sum(line.startswith("F ") for line in payload.splitlines())
    print(f"wrote {args.output} ({image_rows} image rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
