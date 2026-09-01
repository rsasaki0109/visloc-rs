#!/usr/bin/env python3
"""Stage a bounded 10k-frame OpenLORIS corridor SfM validation set.

The official ``corridor1-1`` archive contains 8,520 timestamped images from
each T265 fisheye.  This helper freezes the first 5,000 frames from both
cameras (10,000 images), globally orders them by timestamp, extracts only those
members through HTTP Range reads, undistorts the official Kannala-Brandt
calibration into PINHOLE images while preserving the official focal lengths
and principal points, and writes hash-bound 1k/2.5k/5k/10k prefix manifests.
It never stores the 13.85 GB archive locally.

``libarchive-c``, ``py7zr``, ``fsspec``, OpenCV, and numpy are optional staging
dependencies. Dataset images stay outside this repository and remain subject
to CC BY-ND 4.0; do not redistribute the staged derivative dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path


SOURCE_COMMIT = "cbc03108723d08322b23d0338680bffa9404cce9"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/"
    f"{SOURCE_COMMIT}/package/corridor1-1.7z"
)
TERMS_URL = (
    "https://huggingface.co/datasets/shixuesong/openloris-scene/raw/"
    f"{SOURCE_COMMIT}/README.md"
)
ARCHIVE_BYTES = 13_853_763_765
ARCHIVE_SHA256 = "c7ff1a472ca54da82198521eda8c18f2065691075a05e706880f7fb58fda8415"
TIER_COUNTS = (1000, 2500, 5000, 10000)

# OpenLORIS corridor1-1 sensors.yaml.  The official serialization orders the
# four intrinsics as fx, cx, fy, cy.  The fifth KB coefficient is zero and
# OpenCV's fisheye model consumes the first four.
CAMERAS = {
    1: {
        "directory": "fisheye1",
        "intrinsics": (
            284.98089599609375,
            425.244384765625,
            286.1023864746094,
            398.46759033203125,
        ),
        "distortion": (
            -0.007304710801690817,
            0.043499931693077087,
            -0.04128304123878479,
            0.007652460131794214,
        ),
    },
    2: {
        "directory": "fisheye2",
        "intrinsics": (
            284.8125915527344,
            427.6615905761719,
            285.97601318359375,
            397.1234130859375,
        ),
        "distortion": (
            -0.006379498168826103,
            0.04145561158657074,
            -0.03946448862552643,
            0.0069808149710297585,
        ),
    },
}
WIDTH = 848
HEIGHT = 800


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timestamps", type=int, default=5000)
    parser.add_argument("--archive-url", default=ARCHIVE_URL)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument(
        "--reuse-complete-raw",
        action="store_true",
        help="reuse an already completed target extraction after validating every target exists and is non-empty",
    )
    parser.add_argument("--http-block-mib", type=int, default=16)
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=0,
        help="target-count fallback batching; 0 streams each solid folder exactly once",
    )
    return parser.parse_args()


def official_members(url: str, block_bytes: int):
    import fsspec
    import py7zr

    patch_py7zr_backpressure()
    remote = fsspec.open(url, "rb", block_size=block_bytes, cache_type="readahead").open()
    archive = py7zr.SevenZipFile(remote, "r")
    return remote, archive, archive.getnames()


def patch_py7zr_backpressure() -> None:
    """Prevent py7zr from feeding LZMA while it still has buffered input.

    py7zr 1.1.3 reads another compressed block at every output-file boundary.
    For large solid folders this can build a multi-GiB native input backlog.
    Python's LZMADecompressor exposes ``needs_input`` specifically to avoid
    that: drain its buffered input before issuing the next HTTP Range read.
    """

    from py7zr.compressor import SevenZipDecompressor

    if getattr(SevenZipDecompressor, "_visloc_bounded_input", False):
        return
    original = SevenZipDecompressor._read_data

    def bounded_read_data(self, source):
        if (
            len(self.chain) == 1
            and hasattr(self.chain[0], "needs_input")
            and not self.chain[0].needs_input
        ):
            return b""
        return original(self, source)

    SevenZipDecompressor._read_data = bounded_read_data
    SevenZipDecompressor._visloc_bounded_input = True


def releasing_extract_callback(archive):
    """Release py7zr's per-folder decompressor as each folder completes.

    py7zr caches a native decompressor on every Folder object. OpenLORIS has
    thousands of small folders, so a long extraction otherwise grows by about
    two MiB per folder even though each image has already been written.
    """

    import ctypes
    import gc

    from py7zr.callbacks import ExtractCallback

    folders = archive.header.main_streams.unpackinfo.folders
    completed = {}
    for folder in folders:
        folder_files = list(folder.files)
        if folder_files:
            completed[str(folder_files[-1].filename)] = folder

    class ReleasingExtractCallback(ExtractCallback):
        def __init__(self) -> None:
            self.released = 0
            try:
                self.libc = ctypes.CDLL(None)
                self.malloc_trim = self.libc.malloc_trim
                self.malloc_trim.argtypes = [ctypes.c_size_t]
                self.malloc_trim.restype = ctypes.c_int
            except (AttributeError, OSError):
                self.malloc_trim = None

        def report_start_preparation(self) -> None:
            pass

        def report_start(self, processing_file_path: str, processing_bytes: str) -> None:
            pass

        def report_update(self, decompressed_bytes: str) -> None:
            pass

        def report_end(self, processing_file_path: str, wrote_bytes: str) -> None:
            folder = completed.pop(processing_file_path, None)
            if folder is not None:
                folder.decompressor = None
                self.released += 1
                if self.released % 64 == 0:
                    gc.collect()
                    if self.malloc_trim is not None:
                        self.malloc_trim(0)

        def report_warning(self, message: str) -> None:
            print(f"py7zr warning: {message}", file=sys.stderr)

        def report_postprocess(self) -> None:
            pass

    return ReleasingExtractCallback()


def trim_allocator() -> None:
    import ctypes
    import gc

    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass


def extract_target_batches(
    url: str, raw_root: Path, targets: list[str], block_bytes: int, batch_size: int
) -> None:
    """Bound py7zr state by closing the remote archive after each target batch."""

    total_batches = (len(targets) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(targets), batch_size), 1):
        batch = targets[start : start + batch_size]
        remote, archive, _ = official_members(url, block_bytes)
        try:
            archive.extract(
                path=raw_root,
                targets=batch,
                callback=releasing_extract_callback(archive),
            )
        finally:
            archive.close()
            remote.close()
        del archive
        del remote
        trim_allocator()
        print(
            f"extract batch {batch_index}/{total_batches}: {min(start + len(batch), len(targets))}/{len(targets)} targets",
            flush=True,
        )


def extract_targets_once(
    url: str, raw_root: Path, targets: list[str], block_bytes: int
) -> None:
    remote, archive, _ = official_members(url, block_bytes)
    try:
        archive.extract(
            path=raw_root,
            targets=targets,
            callback=releasing_extract_callback(archive),
        )
    finally:
        archive.close()
        remote.close()
    trim_allocator()


def extract_targets_libarchive(
    remote, raw_root: Path, targets: list[str], block_bytes: int = 1024 * 1024
) -> None:
    """Stream one bounded block at a time through the system libarchive."""

    try:
        import libarchive
    except ImportError as exc:
        raise RuntimeError(
            "bounded OpenLORIS extraction requires libarchive-c and the system libarchive"
        ) from exc

    wanted = set(targets)
    found = set()
    with libarchive.stream_reader(
        remote, format_name="7zip", block_size=block_bytes
    ) as entries:
        for entry in entries:
            name = entry.pathname
            if name not in wanted:
                continue
            destination = raw_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            with temporary.open("wb") as output:
                for block in entry.get_blocks(block_bytes):
                    output.write(block)
            os.replace(temporary, destination)
            found.add(name)
            if len(found) % 500 == 0 or len(found) == len(wanted):
                print(f"libarchive extract: {len(found)}/{len(wanted)} targets", flush=True)
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"libarchive did not find {len(missing)} targets; first={missing[0]}")


def validate_complete_raw(raw_root: Path, targets: list[str]) -> None:
    """Reject partial extraction before an explicitly requested raw reuse."""

    missing = [target for target in targets if not (raw_root / target).is_file()]
    if missing:
        raise ValueError(
            f"cannot reuse incomplete raw extraction: {len(missing)} targets missing; first={missing[0]}"
        )
    empty = [target for target in targets if (raw_root / target).stat().st_size == 0]
    if empty:
        raise ValueError(
            f"cannot reuse incomplete raw extraction: {len(empty)} targets empty; first={empty[0]}"
        )


def selected_members(names: list[str], timestamps: int) -> tuple[list[tuple[int, str, str]], list[str]]:
    by_camera = {}
    for camera, config in CAMERAS.items():
        prefix = f"corridor1-1/{config['directory']}/"
        members = sorted(
            name for name in names if name.startswith(prefix) and name.endswith(".png")
        )
        by_camera[camera] = members
    available = min(len(members) for members in by_camera.values())
    if timestamps <= 0 or timestamps > available:
        raise ValueError(f"--timestamps must be within 1..{available}, got {timestamps}")
    selected = []
    targets = [
        "corridor1-1/sensors.yaml",
        "corridor1-1/trans_matrix.yaml",
        "corridor1-1/groundtruth.txt",
        "corridor1-1/fisheye1.txt",
        "corridor1-1/fisheye2.txt",
    ]
    for camera in sorted(by_camera):
        for member in by_camera[camera][:timestamps]:
            selected.append((camera, member, Path(member).stem))
            targets.append(member)
    selected.sort(key=lambda item: (float(item[2]), item[0]))
    return selected, targets


def undistort(
    raw_root: Path, output: Path, selected: list[tuple[int, str, str]]
) -> tuple[dict[int, tuple[float, ...]], list[dict]]:
    import cv2
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    state_dir = output.parent / "image-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    new_intrinsics = {}
    maps = {}
    config_hashes = {}
    sensors_path = raw_root / "corridor1-1" / "sensors.yaml"
    storage = cv2.FileStorage(str(sensors_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"cannot read official OpenLORIS calibration: {sensors_path}")
    for camera, config in CAMERAS.items():
        sensor = storage.getNode(f"t265_fisheye{camera}_optical_frame")
        official_intrinsics = tuple(float(value) for value in sensor.getNode("intrinsics").mat().reshape(-1))
        official_distortion = tuple(
            float(value)
            for value in sensor.getNode("distortion_coefficients").mat().reshape(-1)[:4]
        )
        if not np.allclose(official_intrinsics, config["intrinsics"], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"camera {camera} constants disagree with official sensors.yaml intrinsics")
        if not np.allclose(official_distortion, config["distortion"], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"camera {camera} constants disagree with official sensors.yaml distortion")
        fx, cx, fy, cy = config["intrinsics"]
        matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        distortion = np.asarray(config["distortion"], dtype=np.float64)
        # OpenCV's automatic new-K estimator is unstable at the extreme image
        # boundary for this >160-degree KB lens (it can emit a negative cy).
        # Keeping the calibrated central focal/principal point produces a
        # deterministic ~112-degree pinhole view with no fitted intrinsics.
        rectified = matrix.copy()
        if not (
            0.1 * WIDTH <= rectified[0, 0] <= 2.0 * WIDTH
            and 0.1 * HEIGHT <= rectified[1, 1] <= 2.0 * HEIGHT
            and 0.0 <= rectified[0, 2] < WIDTH
            and 0.0 <= rectified[1, 2] < HEIGHT
        ):
            raise ValueError(f"camera {camera} produced non-physical rectified intrinsics")
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            matrix, distortion, np.eye(3), rectified, (WIDTH, HEIGHT), cv2.CV_32FC1
        )
        new_intrinsics[camera] = (
            float(rectified[0, 0]),
            float(rectified[1, 1]),
            float(rectified[0, 2]),
            float(rectified[1, 2]),
        )
        maps[camera] = (map_x, map_y)
        config_hashes[camera] = sha256_bytes(
            json.dumps(
                {
                    "camera": camera,
                    "cv2": cv2.__version__,
                    "distortion": config["distortion"],
                    "input_intrinsics": config["intrinsics"],
                    "output_intrinsics": new_intrinsics[camera],
                    "output_size": [WIDTH, HEIGHT],
                    "rectified_intrinsics_policy": "preserve official focal lengths and principal point",
                    "interpolation": "linear",
                },
                sort_keys=True,
            ).encode()
        )
    storage.release()

    records = []
    for sequence_index, (camera, member, timestamp) in enumerate(selected):
        source = raw_root / member
        # A globally unique integer suffix gives the bounded temporal candidate
        # scheduler an unambiguous, chronological order across both cameras.
        destination = output / f"cam{camera}_{sequence_index:06d}.png"
        state_path = state_dir / f"{destination.name}.json"
        source_hash = sha256(source)
        valid_existing = False
        output_hash = None
        if destination.is_file() and state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                output_hash = sha256(destination)
                valid_existing = (
                    state.get("schema") == "visloc_openloris_undistort_state_v1"
                    and state.get("source_member") == member
                    and state.get("source_sha256") == source_hash
                    and state.get("config_sha256") == config_hashes[camera]
                    and state.get("output_sha256") == output_hash
                )
            except (OSError, json.JSONDecodeError):
                valid_existing = False
        if not valid_existing:
            image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (HEIGHT, WIDTH):
                raise ValueError(f"invalid OpenLORIS image: {source}")
            map_x, map_y = maps[camera]
            rectified = cv2.remap(
                image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}.png")
            if not cv2.imwrite(str(temporary), rectified, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(f"cannot write {temporary}")
            os.replace(temporary, destination)
            output_hash = sha256(destination)
            write_atomic(
                state_path,
                (
                    json.dumps(
                        {
                            "schema": "visloc_openloris_undistort_state_v1",
                            "source_member": member,
                            "source_sha256": source_hash,
                            "config_sha256": config_hashes[camera],
                            "output_sha256": output_hash,
                        },
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                ).encode(),
            )
        assert output_hash is not None
        records.append(
            {
                "camera": camera,
                "sequence_index": sequence_index,
                "timestamp": timestamp,
                "name": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": output_hash,
                "source_sha256": source_hash,
                "source_member": member,
            }
        )
    return new_intrinsics, records


def write_calibration(root: Path, intrinsics: dict[int, tuple[float, ...]], records: list[dict]) -> None:
    calibration = root / "calibration"
    camera_lines = ["# Camera list with one line of data per camera:"]
    for camera in sorted(intrinsics):
        fx, fy, cx, cy = intrinsics[camera]
        camera_lines.append(f"{camera} PINHOLE {WIDTH} {HEIGHT} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}")
    image_lines = ["# Identity poses are intrinsics-only staging values; never use as GT."]
    for image_id, record in enumerate(records, 1):
        image_lines.extend(
            (f"{image_id} 1 0 0 0 0 0 0 {record['camera']} {record['name']}", "")
        )
    write_atomic(calibration / "cameras.txt", ("\n".join(camera_lines) + "\n").encode())
    write_atomic(calibration / "images.txt", ("\n".join(image_lines) + "\n").encode())
    write_atomic(calibration / "points3D.txt", b"# Empty intrinsics-only model.\n")


def write_tier_views(
    root: Path, intrinsics: dict[int, tuple[float, ...]], records: list[dict]
) -> dict[str, dict[str, str | int]]:
    """Create deterministic prefix views without duplicating staged images."""

    tiers = {}
    for count in TIER_COUNTS:
        if count > len(records):
            continue
        tier_root = root / "tiers" / f"tier-{count}"
        images = tier_root / "images"
        images.mkdir(parents=True, exist_ok=True)
        for record in records[:count]:
            link = images / record["name"]
            target = Path("../../..") / "images" / record["name"]
            if link.is_symlink():
                if Path(os.readlink(link)) != target:
                    raise ValueError(f"unexpected tier image link target: {link}")
            elif link.exists():
                raise ValueError(f"tier image path is not a symlink: {link}")
            else:
                link.symlink_to(target)
        write_calibration(tier_root, intrinsics, records[:count])
        tiers[str(count)] = {
            "images": str(images),
            "calibration": str(tier_root / "calibration"),
            "image_count": count,
            "storage": "relative symlinks to the full staged image set",
        }
    return tiers


def main() -> int:
    args = parse_args()
    if args.http_block_mib <= 0 or args.extract_batch_size < 0:
        print("--http-block-mib must be positive and --extract-batch-size non-negative", file=sys.stderr)
        return 2
    try:
        remote, archive, names = official_members(
            args.archive_url, args.http_block_mib * 1024 * 1024
        )
        try:
            selected, targets = selected_members(names, args.timestamps)
            if args.list_only:
                print(json.dumps({"archive_members": len(names), "selected_images": len(selected), "first": selected[0][2], "last": selected[-1][2]}, indent=2))
                return 0
        finally:
            archive.close()
            remote.close()

        raw_root = args.output_dir / "raw"
        raw_root.mkdir(parents=True, exist_ok=True)
        if args.reuse_complete_raw:
            validate_complete_raw(raw_root, targets)
            extraction_backend = "validated completed raw extraction"
        elif args.extract_batch_size > 0:
            extract_target_batches(
                args.archive_url,
                raw_root,
                targets,
                args.http_block_mib * 1024 * 1024,
                args.extract_batch_size,
            )
            extraction_backend = "py7zr target batches"
        else:
            import fsspec

            remote = fsspec.open(
                args.archive_url,
                "rb",
                block_size=args.http_block_mib * 1024 * 1024,
                cache_type="readahead",
            ).open()
            try:
                extract_targets_libarchive(remote, raw_root, targets)
            finally:
                remote.close()
            extraction_backend = "libarchive streaming"

        intrinsics, records = undistort(raw_root, args.output_dir / "images", selected)
        write_calibration(args.output_dir, intrinsics, records)
        tier_views = write_tier_views(args.output_dir, intrinsics, records)
        with urllib.request.urlopen(TERMS_URL, timeout=30) as response:
            terms = response.read()
        write_atomic(args.output_dir / "source-terms.md", terms)
        manifests = {}
        for count in TIER_COUNTS:
            if count > len(records):
                continue
            payload = {
                "schema": "visloc_openloris_corridor_manifest_v1",
                "scene": "corridor1-1",
                "images": records[:count],
            }
            encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
            path = args.output_dir / "manifests" / f"tier-{count}.json"
            write_atomic(path, encoded)
            manifests[str(count)] = {"path": str(path), "sha256": sha256_bytes(encoded)}
        audit = {
            "schema": "visloc_openloris_corridor_source_audit_v1",
            "scene": "corridor1-1",
            "connected_rig": True,
            "archive_url": args.archive_url,
            "archive_bytes": ARCHIVE_BYTES,
            "archive_sha256": ARCHIVE_SHA256,
            "archive_sha256_source": "Hugging Face LFS oid at source_commit; Range extraction does not rehash the complete remote archive",
            "source_commit": SOURCE_COMMIT,
            "terms_url": TERMS_URL,
            "terms_sha256": sha256_bytes(terms),
            "license": "CC BY-ND 4.0",
            "redistribution_of_staged_derivatives": False,
            "remote_range_extraction": True,
            "archive_stored_locally": False,
            "extract_batch_size": args.extract_batch_size,
            "extraction_backend": extraction_backend,
            "extract_memory_policy": "LZMA needs_input backpressure; release each solid-folder decompressor; optional target batching fallback",
            "image_resume_validation": "source SHA-256 + config SHA-256 + output SHA-256",
            "selected_frames_per_camera": args.timestamps,
            "selected_images": len(records),
            "sampling": "first frames per camera, globally timestamp-sorted, both T265 fisheyes",
            "calibration": "official Kannala-Brandt -> OpenCV fisheye undistortion; official focal lengths/principal points preserved in PINHOLE export",
            "intrinsics": {str(camera): values for camera, values in intrinsics.items()},
            "tier_manifests": manifests,
            "tier_views": tier_views,
        }
        write_atomic(
            args.output_dir / "source-audit.json",
            (json.dumps(audit, sort_keys=True, indent=2) + "\n").encode(),
        )
        if not args.keep_raw:
            shutil.rmtree(raw_root)
        print(json.dumps(audit, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
