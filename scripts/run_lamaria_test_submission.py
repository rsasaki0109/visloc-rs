#!/usr/bin/env python3
"""Download, run and package a LaMAria test-set submission.

LaMAria (https://lamaria.ethz.ch) scores a zip of `slam/<sequence>.txt`
trajectories in TUM-nanosecond form (`world_from_imu`).  This driver does the
whole per-sequence loop for a track:

    download ASL zip + pinhole calibration
      -> verify + extract -> rename `aria/` to `mav0/`
      -> apply Aria's factory IMU calibration to `imu0/data.csv` (read from
         the first 4 MiB of the raw `.vrs`; `--no-imu-rectify` skips it)
      -> pinhole -> Basalt Double-Sphere calibration (variant A noise,
         gyro bias random walk tightened to --gyro-bias-std)
      -> run the Basalt VIO
      -> convert the trajectory to the submission estimate
      -> append `slam/<sequence>.txt`, then delete the sequence data

It is resumable: a sequence whose estimate already exists in `--slam-dir` is
skipped, so an interrupted run can simply be started again.  The track layout
is one sequence family per leaderboard track (sequence_1_* Short, sequence_2_*
Medium, sequence_3_* Long, sequence_4_* Low light, sequence_5_* Moving).

Test data is large (~862 GB total); each sequence is deleted after its estimate
is written.  Use `--tracks` to process a subset.

Example:

    python scripts/run_lamaria_test_submission.py \
        --tracks 1 \
        --vio-exe E:/visloc-rs-runs/onlinefix_target/release/examples/basalt_euroc_vio_demo.exe \
        --config configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json \
        --gyro-bias-std 1e-7 \
        --work-dir E:/visloc-rs-runs/lamaria_submission/work \
        --slam-dir E:/visloc-rs-runs/lamaria_submission/slam \
        --threads 12

Defaults are the best measured LaMAria setup: the multi-camera big-window
config (docs/lamaria_multicam.md), the rectified IMU and `gyro_bias_std` 1e-7
(docs/lamaria_imu_rectification.md). With the rectified IMU, 1e-7 beats the
earlier 1e-6 by +2.53 Score on average over one training sequence per track
(sequence_1_19 52.26, sequence_2_11 38.36, sequence_3_17 35.76, sequence_4_10
32.42, sequence_5_11 32.27). Pass `--gyro-bias-std 1e-6 --no-imu-rectify` for
the previous setup, or `--config .../euroc_config_big_window.json
--gyro-bias-std 1e-4 --no-imu-rectify` for the original one.

IMU rectification (docs/lamaria_imu_rectification.md) is on by default: on one
training sequence per non-Short track it raises the score by +4.35 on average
(Medium 37.72 -> 39.47, Long 24.14 -> 28.89, Low light 20.42 -> 26.42, Moving
27.64 -> 32.56). If the `.vrs` header cannot be fetched or parsed the sequence
falls back to the raw IMU with a WARN line rather than being dropped.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aria_factory_imu_rectify import VRS_HEAD_BYTES, extract_record, rectify_csv  # noqa: E402

TRACK_COUNTS = {1: 18, 2: 10, 3: 16, 4: 9, 5: 10}
BASE_URL = "https://cvg-data.inf.ethz.ch/lamaria"
DEFAULT_VARIANT_A_NOISE = {
    "accel_noise_std": [0.016] * 3,
    "gyro_noise_std": [0.000282] * 3,
    "accel_bias_std": [0.001] * 3,
    "gyro_bias_std": [0.0001] * 3,
}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", flush=True)


def track_sequences(tracks):
    return [f"sequence_{t}_{i}" for t in tracks for i in range(1, TRACK_COUNTS[t] + 1)]


def run(cmd, **kwargs):
    log("run: " + " ".join(str(c) for c in cmd))
    kwargs.setdefault("check", True)
    return subprocess.run([str(c) for c in cmd], **kwargs)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    aria2c = shutil.which("aria2c")
    if aria2c:
        result = subprocess.run(
            [aria2c, "-x16", "-s16", "-q", "--console-log-level=warn",
             "-d", str(dest.parent), "-o", dest.name, url],
            check=False,
        )
        if result.returncode == 0 and dest.exists():
            return
    log(f"aria2c unavailable/failed; falling back to urllib for {url}")
    urllib.request.urlretrieve(url, dest)


def download_head(url: str, dest: Path, nbytes: int) -> None:
    """Fetch only the first `nbytes` of `url` (HTTP range request)."""
    if dest.is_file() and dest.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        # read(nbytes) also bounds a server that ignores Range and answers 200.
        dest.write_bytes(response.read(nbytes))


def rectify_imu(seq: str, inner: Path, work: Path, calib_dir: Path) -> None:
    """Replace inner/mav0/imu0/data.csv with the factory-rectified IMU."""
    head = work / "vrs_head.bin"
    download_head(f"{BASE_URL}/raw_data/test/{seq}.vrs", head, VRS_HEAD_BYTES)
    factory = extract_record(head.read_bytes())
    (calib_dir / f"{seq}_factory_calib.json").write_text(json.dumps(factory, indent=1))
    imu_csv = inner / "mav0" / "imu0" / "data.csv"
    rectified = imu_csv.with_name("data_rectified.csv")
    samples, gyro_bias, accel_bias = rectify_csv(factory, imu_csv, rectified)
    rectified.replace(imu_csv)
    log(f"{seq}: rectified {samples} IMU samples (serial {factory.get('Serial')}, "
        f"gyro bias {gyro_bias.round(5).tolist()}, accel bias {accel_bias.round(4).tolist()})")


def verify_and_extract(zip_path: Path, extract_dir: Path) -> Path:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.testzip()
            archive.extractall(extract_dir)
    except zipfile.BadZipFile:
        log(f"corrupt zip {zip_path}; deleting for a fresh retry")
        zip_path.unlink(missing_ok=True)
        raise
    inner = next((p for p in extract_dir.iterdir() if p.is_dir()), None)
    if inner is None:
        raise RuntimeError(f"no extracted directory under {extract_dir}")
    aria = inner / "aria"
    mav0 = inner / "mav0"
    if aria.is_dir() and not mav0.is_dir():
        aria.rename(mav0)
    if not (mav0 / "cam0" / "data.csv").is_file():
        raise RuntimeError(f"extracted {inner} has no mav0/cam0/data.csv")
    return inner


def write_variant_a_calib(base_calib: Path, out: Path,
                          gyro_bias_std: float | None = None) -> None:
    data = json.loads(base_calib.read_text())
    data["value0"].update(DEFAULT_VARIANT_A_NOISE)
    if gyro_bias_std is not None:
        data["value0"]["gyro_bias_std"] = [gyro_bias_std] * 3
    out.write_text(json.dumps(data, indent=4))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", type=int, nargs="+", default=list(TRACK_COUNTS),
                        choices=sorted(TRACK_COUNTS))
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="explicit sequence names (overrides --tracks)")
    parser.add_argument("--vio-exe", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json"))
    parser.add_argument("--gyro-bias-std", type=float, default=1e-7,
                        help="gyro bias random walk written into the calibration "
                             "(Basalt default 1e-4; 1e-7 measured best on LaMAria "
                             "with the rectified IMU)")
    parser.add_argument("--imu-rectify", action=argparse.BooleanOptionalAction, default=True,
                        help="apply Aria's factory IMU calibration from the .vrs header "
                             "(default on; docs/lamaria_imu_rectification.md)")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--work-dir", type=Path, default=Path("lamaria_submission/work"))
    parser.add_argument("--slam-dir", type=Path, default=Path("lamaria_submission/slam"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--keep-data", action="store_true",
                        help="keep downloaded/extracted sequence data after each run")
    args = parser.parse_args()

    sequences = args.sequences or track_sequences(args.tracks)
    args.slam_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = args.work_dir / "calibrations"
    calib_dir.mkdir(parents=True, exist_ok=True)
    converter = args.repo / "scripts" / "lamaria_to_basalt_calib.py"
    estimate_converter = args.repo / "scripts" / "basalt_tum_to_lamaria_estimate.py"
    python = sys.executable

    log(f"{len(sequences)} sequences: {sequences[0]} .. {sequences[-1]}")
    for seq in sequences:
        estimate = args.slam_dir / f"{seq}.txt"
        if estimate.is_file():
            log(f"{seq}: estimate exists, skip")
            continue
        work = args.work_dir / seq
        work.mkdir(parents=True, exist_ok=True)
        try:
            log(f"{seq}: download")
            pinhole = work / "pinhole.json"
            download(f"{BASE_URL}/pinhole_calibrations/test/{seq}.json", pinhole)
            zip_path = work / f"{seq}.zip"
            download(f"{BASE_URL}/asl_folder/test/{seq}.zip", zip_path)

            log(f"{seq}: extract")
            extract_dir = work / "extracted"
            extract_dir.mkdir(exist_ok=True)
            try:
                inner = verify_and_extract(zip_path, extract_dir)
            except zipfile.BadZipFile:
                download(f"{BASE_URL}/asl_folder/test/{seq}.zip", zip_path)
                inner = verify_and_extract(zip_path, extract_dir)

            if args.imu_rectify:
                log(f"{seq}: IMU rectification")
                try:
                    rectify_imu(seq, inner, work, calib_dir)
                except Exception as error:
                    log(f"{seq}: WARN IMU rectification failed ({error!r}); using raw IMU")

            log(f"{seq}: calibration")
            base_calib = calib_dir / f"{seq}_calib.json"
            run([python, converter, "--pinhole-calib", pinhole, "--out", base_calib],
                check=True)
            variant_a = calib_dir / f"{seq}_calib_variantA_default_noise.json"
            write_variant_a_calib(base_calib, variant_a, args.gyro_bias_std)

            log(f"{seq}: VIO")
            vio_out = work / "vio"
            vio_out.mkdir(exist_ok=True)
            run([args.vio_exe, "--euroc-dir", inner, "--calibration", variant_a,
                 "--config", args.config, "--out-dir", vio_out, "--pipeline",
                 "--threads", args.threads, "--no-trace", "--no-marg-data"],
                check=True)
            trajectory = vio_out / "trajectory.tum"
            if not trajectory.is_file():
                log(f"{seq}: FAIL no trajectory")
                continue

            run([python, estimate_converter, "--in-tum", trajectory,
                 "--out-estimate", estimate], check=True)
            if estimate.is_file():
                log(f"{seq}: DONE ({sum(1 for _ in estimate.open())} poses)")
            else:
                log(f"{seq}: FAIL estimate conversion")
        except Exception as error:  # keep the loop going across sequences
            log(f"{seq}: EXCEPTION {error!r}")
        finally:
            # Always reclaim the sequence's download/extract work so a failure
            # cannot accumulate disk across the whole track.
            if not args.keep_data:
                shutil.rmtree(work, ignore_errors=True)

    submission = args.slam_dir.parent / "submission.zip"
    with zipfile.ZipFile(submission, "w", zipfile.ZIP_DEFLATED) as archive:
        for estimate in sorted(args.slam_dir.glob("*.txt")):
            archive.write(estimate, f"slam/{estimate.name}")
    log(f"packaged {submission} with {len(list(args.slam_dir.glob('*.txt')))} estimates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
