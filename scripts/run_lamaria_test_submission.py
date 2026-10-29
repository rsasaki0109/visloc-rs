#!/usr/bin/env python3
"""Download, run and package a LaMAria test-set submission.

LaMAria (https://lamaria.ethz.ch) scores a zip of `slam/<sequence>.txt`
trajectories in TUM-nanosecond form (`world_from_imu`).  This driver does the
whole per-sequence loop for a track:

    download ASL zip + pinhole calibration
      -> verify + extract -> rename `aria/` to `mav0/`
      -> apply Aria's factory IMU calibration to `imu0/data.csv` (read from
         the first 4 MiB of the raw `.vrs`; `--no-imu-rectify` skips it),
         plus the per-device time alignment (`--no-device-refinement` skips it)
      -> pinhole -> Basalt Double-Sphere calibration (variant A noise,
         gyro bias random walk tightened to --gyro-bias-std; camera rig
         rotated by the per-device refinement)
      -> run the Basalt VIO, or with `--slam-exe` the VIO plus the online
         mapper (`basalt_euroc_online_slam_demo`)
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
        --slam-exe E:/visloc-rs-runs/onlinefix_target/release/examples/basalt_euroc_online_slam_demo.exe \
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

The per-device IMU-camera refinement (scripts/aria_device_refinement.py,
docs/lamaria_drift_analysis.md) is on by default. LaMAria was recorded with two
Aria devices; for each, training pseudo-GT shows the gyroscope lagging the
images by ~4.6 ms and, on 1WM093701G1276, a 0.35 deg IMU-to-camera rotation
error. Correcting them raises the VIO-only Score on sequence_2_11 38.36 ->
49.68 and on the held-out sequence_5_11 32.27 -> 43.14 and sequence_1_20
33.18 -> 38.10. An unknown device serial falls back to the factory calibration
with a WARN line.

`--slam-exe` (the online-mapper build, docs/lamaria_online_mapper.md) is the
recommended estimator. It writes the full-frame mapper trajectory
`trajectory_online.tum` and the VIO's own `trajectory_vio.tum`, and the driver
picks one per track. With the per-device refinement the VIO beats the mapper
on Short, Long and Low light, while the mapper still wins on Medium and Moving
platform (one training sequence per track; VIO / mapper: Short 71.96 / 68.54,
Medium 49.68 / 55.51, Long 50.06 / 46.68, Low light 34.49 / 31.62, Moving
43.14 / 49.84; docs/lamaria_drift_analysis.md#with-the-online-mapper). So by
default the VIO trajectory is submitted for tracks 1, 3 and 4 (`--vio-tracks`;
pass it without values to submit the mapper everywhere). With `--vio-exe` as
well, those tracks run the faster VIO-only build. If a mapper run produces only
`trajectory_vio.tum`, that is used with a WARN line. `--vio-exe` alone runs the
VIO only on every track.

Low light (track 4) sequences use `--low-light-config`, the multicam config
with a 0.5 px instead of 0.2 px KLT round-trip threshold: it raises the VIO on
sequence_4_10 from 34.49 to 39.00 but costs 11 points on the Medium sequence,
so it is limited to that track (docs/lamaria_klt_threshold.md). Pass
`--low-light-config none` to use `--config` there too.

`--ba-chain` runs the visual-inertial bundle adjustment of
scripts/lamaria_ba_chain.py behind the VIO and submits its trajectory: SIFT
keyframes, matching and revisits guided by the VIO, the online windows and, by
day, one batch pass (docs/lamaria_batch_ba.md#the-first-minutes-online). On
the ten training sequences it averages 64.35 per track. The VIO is then
Basalt's default window, `euroc_config.json` (`euroc_config_lowlight.json` in
low light), unless `--config` / `--low-light-config` say otherwise, and the
BA gets the calibration with the factory camera rig, since it estimates the
camera rotations itself. It needs numpy, scipy, opencv-python, gtsam and
numba, and about 2 GB of disk per sequence on top of the images.

A VIO that diverges (faster than 30 m/s over a twentieth of its seconds) runs
again with both cameras' features (`MULTICAM`): on test sequence_1_9 the
default config's VIO runs away from the first seconds (3 km away after a
minute) while the same config with both cameras' features stays on a walk.
If it still diverges, it starts again 10, 30 or 60 s into the recording
(`LATE_STARTS_S`): on test sequence_1_15 the head shakes at up to 2 rad/s
over the first seconds and both configs run away from there, while either
one started 10 s later holds. The poses before that start are then missing.
On a walking track (not Moving platform) averaging more than 4 m/s over
10 s counts as diverging too (`runaway`; test sequence_4_2 ran away for
60 s at up to 12 m/s from 10 s), and the last starts follow the end of the
runaway; if every start fails, the attempt that ran away least is used.
If the BA chain still fails, the VIO trajectory is submitted.
By day the chain then matches once more after its pass, guided by the pass's
trajectory, and solves again (lamaria_ba_chain.py --rematch; `--no-ba-rematch`
skips it): +0.26 on the eight day training sequences, +3.21 on R_12_10cp,
whose VIO heading drifts most.
`--ba-pass-features aliked` gives that pass learned keypoints (LightGlue's, on
the GPU; lamaria_ba_chain.py --pass-features) on `--ba-pass-features-tracks`
(default Short) from the devices in `--ba-pass-features-serials` (default
1WM093701G1276) only: the two Short training sequences of that device gain 6.4
and 9.3, where sequence_1_19 and 1_20 (the other device), 2_11 and 3_18 lose
1.2 to 12.7 (docs/lamaria_batch_ba.md#learned-features). Off by default.
`--ba-pass-match online` matches those keypoints within the online result's
epipolar band instead of the VIO's (lamaria_ba_chain.py --pass-match): the two
R sequences then reach 97.02 and 79.31 instead of 92.20 and 73.96 (submission 6).

`--stream` extracts the ASL zip over HTTP range requests instead of
downloading it first (scripts/http_zip.py), so a sequence needs its images'
size on disk, not twice that: what a GitHub runner can hold
(.github/workflows/lamaria_submission.yml). `--log-dir` keeps the BA chain's
step logs of each sequence before its work directory is deleted.
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
from aria_device_refinement import lookup as device_refinement, rotate_rig  # noqa: E402
from lamaria_ba_chain import PASS_MATCH, run_chain  # noqa: E402
import http_zip  # noqa: E402

LOOP_MATCH_MAX_ROT_ERROR_DEG = 5
TRACK_COUNTS = {1: 18, 2: 10, 3: 16, 4: 9, 5: 10}
# Tracks whose VIO trajectory is submitted even with --slam-exe (Short, Long, Low light).
DEFAULT_VIO_TRACKS = (1, 3, 4)
LOW_LIGHT_TRACK = 4
# Mean cam0 grey level from which a Low light sequence runs the day options: the eight day training
# sequences average 74-88, the two low-light ones 26-29 (docs/lamaria_submission.md)
DAYLIT_BRIGHTNESS = 70.0
DEFAULT_PLATFORM_TRACKS = (5,)  # Moving platform: docs/lamaria_submission.md
# Where --ba-pass-features applies by default: the Short training sequences of this device gain with it
DEFAULT_PASS_FEATURES_TRACKS = (1,)
DEFAULT_PASS_FEATURES_SERIALS = ("1WM093701G1276",)
DEFAULT_CONFIG = Path("configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json")
DEFAULT_LOW_LIGHT_CONFIG = Path(
    "configs/basalt/variants/lamaria/euroc_config_big_window_multicam_lowlight.json")
# the VIO of the --ba-chain (the real-time chain's): Basalt's default window
BA_CHAIN_CONFIG = Path("configs/basalt/variants/lamaria/euroc_config.json")
BA_CHAIN_LOW_LIGHT_CONFIG = Path("configs/basalt/variants/lamaria/euroc_config_lowlight.json")
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


def rectify_imu(seq: str, inner: Path, work: Path, calib_dir: Path,
                refine: bool = False, split: str = "test"):
    """Replace inner/mav0/imu0/data.csv with the factory-rectified IMU.

    With `refine`, a known device's IMU is also re-aligned in time, and its
    refinement (see aria_device_refinement.py) is returned for the camera rig.
    """
    head = work / "vrs_head.bin"
    download_head(f"{BASE_URL}/raw_data/{split}/{seq}.vrs", head, VRS_HEAD_BYTES)
    factory = extract_record(head.read_bytes())
    (calib_dir / f"{seq}_factory_calib.json").write_text(json.dumps(factory, indent=1))
    refinement = device_refinement(factory) if refine else None
    if refine and refinement is None:
        log(f"{seq}: WARN no device refinement for serial {factory.get('Serial')}")
    delays = ({"gyro_delay_ns": refinement["gyro_delay_ns"],
               "accel_delay_ns": refinement["accel_delay_ns"]} if refinement else {})
    imu_csv = inner / "mav0" / "imu0" / "data.csv"
    rectified = imu_csv.with_name("data_rectified.csv")
    samples, gyro_bias, accel_bias = rectify_csv(factory, imu_csv, rectified, **delays)
    rectified.replace(imu_csv)
    log(f"{seq}: rectified {samples} IMU samples (serial {factory.get('Serial')}, "
        f"gyro bias {gyro_bias.round(5).tolist()}, accel bias {accel_bias.round(4).tolist()}"
        + (f", delays gyro {delays['gyro_delay_ns']} ns accel {delays['accel_delay_ns']} ns"
           if refinement else "") + ")")
    return refinement


def sequence_track(seq: str) -> int:
    """The leaderboard track of `sequence_<track>_<n>`."""
    return int(seq.split("_")[1])


def device_serial(calib_dir: Path, seq: str) -> str | None:
    """The recording device's serial, from the factory calibration that the IMU rectification wrote."""
    factory = calib_dir / f"{seq}_factory_calib.json"
    return json.loads(factory.read_text()).get("Serial") if factory.is_file() else None


def pass_features(args, track: int, serial: str | None) -> str | None:
    """The chain pass's learned keypoints for this sequence (--ba-pass-features), or None for SIFT."""
    if args.ba_pass_features and track in args.ba_pass_features_tracks \
            and serial in args.ba_pass_features_serials:
        return args.ba_pass_features
    return None


def estimator_command(args, inner: Path, calibration: Path, out_dir: Path,
                      config: Path | None = None, submit_vio: bool = False):
    """Return (command, trajectories to try in order) for the configured estimator.

    `submit_vio` submits the VIO trajectory: through the VIO-only build when
    there is one, otherwise from the online demo's `trajectory_vio.tum`.
    """
    common = ["--euroc-dir", inner, "--calibration", calibration,
              "--config", config or args.config, "--out-dir", out_dir,
              "--threads", args.threads]
    if args.slam_exe and not (submit_vio and args.vio_exe):
        # Same VIO as basalt_euroc_vio_demo (bit-identical trajectory_vio.tum)
        # plus the online mapper; trajectory_online.tum has every frame.
        # 5 deg loop rotation gate: with the camera-frame comparison, genuine
        # loops disagree with the map by ~1 deg (median on sequence_1_19), so
        # 5 deg drops only broken two-view estimates
        # (docs/lamaria_online_mapper.md#loop-rotation-gate).
        trajectories = [out_dir / "trajectory_online.tum", out_dir / "trajectory_vio.tum"]
        return ([args.slam_exe, *common, "--no-urgent-keyframes",
                 "--loop-match-max-rot-error", LOOP_MATCH_MAX_ROT_ERROR_DEG],
                trajectories[1:] if submit_vio else trajectories)
    return ([args.vio_exe, *common, "--pipeline", "--no-trace", "--no-marg-data"],
            [out_dir / "trajectory.tum"])


# Both cameras' features in the VIO (docs/lamaria_multicam.md): the fallback for a VIO that diverged
MULTICAM = {
    "config.optical_flow_detect_all_cameras": True,
    "config.optical_flow_matching_default_depth": 2.0,
    "config.optical_flow_matching_guess_type": "REPROJ_FIX_DEPTH",
    "config.vio_kf_connectivity_all_cameras": True,
    "config.vio_landmarks_all_cameras": True,
}
DIVERGED_SPEED = 30.0  # m/s; a tram or a funicular stays well below


def diverged(tum: Path, speed: float = DIVERGED_SPEED) -> bool:
    """Whether the trajectory moves faster than `speed` over a twentieth or more of its seconds."""
    rows = [[float(v) for v in line.split()[:4]] for line in tum.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    fast = total = 0
    j = 0
    for i, (t, x, y, z) in enumerate(rows):
        while j < len(rows) and rows[j][0] < t + 1.0:
            j += 1
        if j == len(rows):
            break
        u, a, b, c = rows[j]
        total += 1
        fast += ((a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2) ** 0.5 / (u - t) > speed
    return total > 0 and fast > total / 20


LATE_STARTS_S = (10, 30, 60)  # where a VIO that diverges from the first seconds starts again
WALK_SPEED = 4.0  # m/s held over RUNAWAY_WINDOW_S: nobody walks that fast for 10 s
RUNAWAY_WINDOW_S = 10.0
MOVING_TRACK = 5


def runaway(tum: Path, speed: float = WALK_SPEED, window: float = RUNAWAY_WINDOW_S):
    """(seconds, end): how many seconds start a `window` over which the trajectory averages more
    than `speed`, and the trajectory time at which the last such window ends (None without)."""
    rows = [[float(v) for v in line.split()[:4]] for line in tum.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    fast, end, j = 0, None, 0
    for i, (t, x, y, z) in enumerate(rows):
        while j < len(rows) and rows[j][0] < t + window:
            j += 1
        if j == len(rows):
            break
        u, a, b, c = rows[j]
        if ((a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2) ** 0.5 / (u - t) > speed:
            fast += 1
            end = u
    seconds = fast / ((len(rows) - 1) / (rows[-1][0] - rows[0][0])) if fast else 0.0
    return seconds, end


def vio_failure(tum: Path, track: int) -> float:
    """0 for a usable VIO trajectory, else how badly it failed: diverged (inf), or the seconds of a
    walking track's trajectory that run away at more than WALK_SPEED (test sequence_4_2: 60 s at up
    to 12 m/s, under the divergence check's twentieth)."""
    if diverged(tum):
        return float("inf")
    return 0.0 if track == MOVING_TRACK else runaway(tum)[0]


def mean_brightness(cam_dir: Path, samples: int = 60) -> float:
    """The mean grey level of `samples` images spread over the recording."""
    import cv2  # the BA's dependency; only the Low light track needs it here
    rows = [r for r in (cam_dir / "data.csv").read_text().splitlines() if r.strip() and not r.startswith("#")]
    picked = rows[::max(1, len(rows) // samples)][:samples]
    levels = [cv2.imread(str(cam_dir / "data" / r.split(",")[1].strip()), cv2.IMREAD_GRAYSCALE).mean()
              for r in picked]
    return sum(levels) / len(levels)


def recording_start(inner: Path) -> float:
    """The first cam0 frame's time in seconds."""
    for row in (inner / "mav0" / "cam0" / "data.csv").read_text().splitlines():
        if row.strip() and not row.startswith("#"):
            return int(row.split(",")[0]) * 1e-9
    raise RuntimeError(f"no frames in {inner}")


def late_start(inner: Path, out: Path, seconds: float) -> Path:
    """An EuRoC folder of `inner` that starts `seconds` into the recording: trimmed manifests,
    the images linked."""
    t0 = None
    for cam in ("cam0", "cam1"):
        (out / "mav0" / cam).mkdir(parents=True, exist_ok=True)
        rows = [r for r in (inner / "mav0" / cam / "data.csv").read_text().splitlines() if r.strip()]
        head = [r for r in rows if r.startswith("#")]
        frames = [r for r in rows if not r.startswith("#")]
        if t0 is None:
            t0 = int(frames[0].split(",")[0]) + int(seconds * 1e9)
        (out / "mav0" / cam / "data.csv").write_text(
            "\n".join(head + [r for r in frames if int(r.split(",")[0]) >= t0]) + "\n")
        link = out / "mav0" / cam / "data"
        if not link.exists():
            link.symlink_to((inner / "mav0" / cam / "data").resolve(), target_is_directory=True)
    (out / "mav0" / "imu0").mkdir(parents=True, exist_ok=True)
    rows = (inner / "mav0" / "imu0" / "data.csv").read_text().splitlines()
    (out / "mav0" / "imu0" / "data.csv").write_text("\n".join(
        r for r in rows if r.startswith("#") or (r.strip() and int(r.split(",")[0]) >= t0 - 100_000_000)) + "\n")
    return out


def keep_attempt(trajectory: Path, out: Path) -> Path:
    """A copy of a VIO attempt's trajectory, which the next attempt overwrites."""
    shutil.copyfile(trajectory, out)
    return out


def multicam_config(config: Path, out: Path) -> Path:
    """`config` with both cameras' features."""
    return config_with(config, out, MULTICAM)


def config_with(config: Path, out: Path, values: dict) -> Path:
    """`config` with `values` set."""
    data = json.loads(config.read_text())
    data["value0"].update(values)
    out.write_text(json.dumps(data, indent=4))
    return out


def verify_and_extract(zip_path: Path, extract_dir: Path) -> Path:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.testzip()
            archive.extractall(extract_dir)
    except zipfile.BadZipFile:
        log(f"corrupt zip {zip_path}; deleting for a fresh retry")
        zip_path.unlink(missing_ok=True)
        raise
    return sequence_dir(extract_dir)


def sequence_dir(extract_dir: Path) -> Path:
    """The extracted sequence, its `aria/` renamed to `mav0/`."""
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
    parser.add_argument("--split", choices=("test", "training"), default="test",
                        help="dataset split to download from; 'training' runs the same pipeline on "
                             "training sequences (with --sequences), to score it with the evaluator")
    parser.add_argument("--vio-exe", type=Path,
                        help="basalt_euroc_vio_demo: VIO only")
    parser.add_argument("--slam-exe", type=Path,
                        help="basalt_euroc_online_slam_demo: VIO + online mapper (recommended; "
                             "takes precedence over --vio-exe except on --vio-tracks)")
    parser.add_argument("--config", type=Path,
                        help=f"VIO config (default {DEFAULT_CONFIG}; with --ba-chain {BA_CHAIN_CONFIG})")
    parser.add_argument("--vio-tracks", type=int, nargs="*", default=list(DEFAULT_VIO_TRACKS),
                        choices=sorted(TRACK_COUNTS),
                        help="tracks whose VIO trajectory is submitted instead of the online "
                             "mapper's (default: 1 3 4; no values: the mapper everywhere)")
    parser.add_argument("--platform-tracks", type=int, nargs="*", default=list(DEFAULT_PLATFORM_TRACKS),
                        choices=sorted(TRACK_COUNTS),
                        help="tracks whose VIO keeps a vehicle's cabin out during rides "
                             "(config.vio_platform_filter; default 5; no values: none)")
    parser.add_argument("--platform-multicam", action=argparse.BooleanOptionalAction, default=True,
                        help="on --platform-tracks, the VIO tracks both cameras' features (MULTICAM), so the "
                             "filter sees the street through the windows on either side (default on)")
    parser.add_argument("--low-light-config",
                        help=f"config for Low light (track 4) sequences (default {DEFAULT_LOW_LIGHT_CONFIG}; "
                             f"with --ba-chain {BA_CHAIN_LOW_LIGHT_CONFIG}); 'none' uses --config")
    parser.add_argument("--ba-chain", action="store_true",
                        help="run the visual-inertial BA (scripts/lamaria_ba_chain.py) behind the VIO and "
                             "submit its trajectory")
    parser.add_argument("--ba-rematch", action=argparse.BooleanOptionalAction, default=True,
                        help="--ba-chain: by day, match again guided by the pass and solve again (lamaria_ba_chain.py "
                             "--rematch; default on); with --log-dir the pass's estimate before it is kept as "
                             "<log-dir>/<sequence>/ba2_estimate.txt")
    parser.add_argument("--ba-pass-features", choices=("aliked", "disk", "superpoint"), default=None,
                        help="--ba-chain: learned keypoints for the pass after the online windows (GPU, LightGlue; "
                             "lamaria_ba_chain.py --pass-features) on the tracks and devices below (default off)")
    parser.add_argument("--ba-pass-features-tracks", type=int, nargs="*",
                        default=list(DEFAULT_PASS_FEATURES_TRACKS), choices=sorted(TRACK_COUNTS),
                        help="--ba-pass-features: the tracks it applies to (default 1)")
    parser.add_argument("--ba-pass-features-serials", nargs="*", default=list(DEFAULT_PASS_FEATURES_SERIALS),
                        help="--ba-pass-features: the device serials it applies to (default 1WM093701G1276)")
    parser.add_argument("--ba-pass-match", choices=PASS_MATCH, default="vio",
                        help="--ba-pass-features: keep the matches within the VIO's epipolar band (default), the online "
                             "result's, or those an essential matrix explains (lamaria_ba_chain.py --pass-match)")
    parser.add_argument("--ba-jobs", type=int, default=4, help="--ba-chain: processes for features, match, loops")
    parser.add_argument("--ba-threads", type=int, default=2, help="--ba-chain: threads for its solves")
    parser.add_argument("--gyro-bias-std", type=float, default=1e-7,
                        help="gyro bias random walk written into the calibration "
                             "(Basalt default 1e-4; 1e-7 measured best on LaMAria "
                             "with the rectified IMU)")
    parser.add_argument("--imu-rectify", action=argparse.BooleanOptionalAction, default=True,
                        help="apply Aria's factory IMU calibration from the .vrs header "
                             "(default on; docs/lamaria_imu_rectification.md)")
    parser.add_argument("--device-refinement", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="re-align the IMU in time and rotate the camera rig by the "
                             "per-device refinement learned on training data (default on; "
                             "needs --imu-rectify; docs/lamaria_drift_analysis.md)")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--work-dir", type=Path, default=Path("lamaria_submission/work"))
    parser.add_argument("--slam-dir", type=Path, default=Path("lamaria_submission/slam"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--stream", action="store_true",
                        help="extract the ASL zip over HTTP range requests instead of downloading it")
    parser.add_argument("--log-dir", type=Path,
                        help="keep each sequence's BA chain step logs under <log-dir>/<sequence>")
    parser.add_argument("--keep-data", action="store_true",
                        help="keep downloaded/extracted sequence data after each run")
    args = parser.parse_args()
    if not (args.slam_exe or args.vio_exe):
        parser.error("one of --slam-exe or --vio-exe is required")

    args.config = args.config or (BA_CHAIN_CONFIG if args.ba_chain else DEFAULT_CONFIG)
    low_light_arg = args.low_light_config or str(BA_CHAIN_LOW_LIGHT_CONFIG if args.ba_chain else DEFAULT_LOW_LIGHT_CONFIG)
    low_light_config = None if low_light_arg.lower() == "none" else Path(low_light_arg)
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
            download(f"{BASE_URL}/pinhole_calibrations/{args.split}/{seq}.json", pinhole)
            zip_url = f"{BASE_URL}/asl_folder/{args.split}/{seq}.zip"
            extract_dir = work / "extracted"
            extract_dir.mkdir(exist_ok=True)
            if args.stream:
                log(f"{seq}: extract over HTTP")
                http_zip.extract(zip_url, extract_dir, log=lambda m, seq=seq: log(f"{seq}: {m}"))
                inner = sequence_dir(extract_dir)
            else:
                zip_path = work / f"{seq}.zip"
                download(zip_url, zip_path)
                log(f"{seq}: extract")
                try:
                    inner = verify_and_extract(zip_path, extract_dir)
                except zipfile.BadZipFile:
                    download(zip_url, zip_path)
                    inner = verify_and_extract(zip_path, extract_dir)

            refinement = None
            if args.imu_rectify:
                log(f"{seq}: IMU rectification")
                try:
                    refinement = rectify_imu(seq, inner, work, calib_dir,
                                             refine=args.device_refinement, split=args.split)
                except Exception as error:
                    log(f"{seq}: WARN IMU rectification failed ({error!r}); using raw IMU")

            log(f"{seq}: calibration")
            base_calib = calib_dir / f"{seq}_calib.json"
            run([python, converter, "--pinhole-calib", pinhole, "--out", base_calib],
                check=True)
            variant_a = calib_dir / f"{seq}_calib_variantA_default_noise.json"
            write_variant_a_calib(base_calib, variant_a, args.gyro_bias_std)
            ba_calib = calib_dir / f"{seq}_calib_ba.json"  # the factory rig: the BA estimates the rotations
            write_variant_a_calib(base_calib, ba_calib, args.gyro_bias_std)
            if refinement:
                variant_a.write_text(json.dumps(
                    rotate_rig(json.loads(variant_a.read_text()),
                               refinement["rig_rotation_deg"]), indent=4))
                log(f"{seq}: camera rig rotated by {refinement['rig_rotation_deg']} deg")

            track = sequence_track(seq)
            low_light = track == LOW_LIGHT_TRACK
            if low_light:  # test sequence_4_1, 4_4, 4_8 and 4_9 are shot in dusk or overcast daylight
                brightness = mean_brightness(inner / "mav0" / "cam0")
                low_light = brightness < DAYLIT_BRIGHTNESS
                log(f"{seq}: mean grey level {brightness:.1f}: {'low-light' if low_light else 'day'} options")
            config = low_light_config if low_light and low_light_config else args.config
            if track in args.platform_tracks:
                config = config_with(config, work / f"{config.stem}_platform.json",
                                     {"config.vio_platform_filter": True})
                if args.platform_multicam:  # test sequence_5_9: the tram's street is in cam1's door window
                    config = multicam_config(config, work / f"{config.stem}_multicam.json")
            submit_vio = track in args.vio_tracks or args.ba_chain  # the BA chain starts from the VIO
            log(f"{seq}: {'VIO' if submit_vio or not args.slam_exe else 'VIO + online mapper'}"
                f", config {config.name}")
            vio_out = work / "vio"
            vio_out.mkdir(exist_ok=True)
            command, candidates = estimator_command(args, inner, variant_a, vio_out,
                                                    config, submit_vio)
            run(command, check=True)
            trajectory = next((t for t in candidates if t.is_file()), None)
            if trajectory is None:
                log(f"{seq}: FAIL no trajectory")
                continue
            if trajectory != candidates[0]:
                log(f"{seq}: WARN {candidates[0].name} missing; submitting {trajectory.name}")
            if vio_failure(trajectory, track):  # test sequence_1_9: the default config's VIO runs away from the start
                attempts = [(vio_failure(trajectory, track), keep_attempt(trajectory, work / "vio_default.tum"))]
                fallback = multicam_config(config, work / f"{config.stem}_multicam.json")
                log(f"{seq}: WARN the VIO diverged; again with both cameras' features ({fallback.name})")
                command, candidates = estimator_command(args, inner, variant_a, vio_out, fallback, submit_vio)
                run(command, check=True)
                trajectory = next(t for t in candidates if t.is_file())
                starts, tried, after_runaway = list(LATE_STARTS_S), 0, False
                while vio_failure(trajectory, track) and starts:  # test sequence_1_15: the head shakes over the first seconds
                    attempts.append((vio_failure(trajectory, track),
                                     keep_attempt(trajectory, work / f"vio_from_{tried}.tum")))
                    tried = starts.pop(0)
                    log(f"{seq}: WARN the VIO diverged again; starting it {tried} s into the recording")
                    late = late_start(inner, work / f"late_{tried}", tried)
                    command, candidates = estimator_command(args, late, variant_a, vio_out, fallback, submit_vio)
                    run(command, check=True)
                    trajectory = next(t for t in candidates if t.is_file())
                    end = runaway(trajectory)[1]
                    if (not starts and not after_runaway and track != MOVING_TRACK and end is not None
                            and vio_failure(trajectory, track)):
                        after_runaway = True  # test sequence_4_2: once more, 5 s after the runaway ends
                        starts.append(round(end - recording_start(inner) + 5))
                failure = vio_failure(trajectory, track)
                if failure:
                    attempts.append((failure, keep_attempt(trajectory, work / f"vio_from_{tried:.0f}.tum")))
                    failure, trajectory = min(attempts, key=lambda a: a[0])
                    log(f"{seq}: WARN the VIO diverged at every start; submitting {trajectory.name} "
                        f"({failure:.0f} s run away)")
            if args.ba_chain:
                learned = pass_features(args, track, device_serial(calib_dir, seq))
                log(f"{seq}: BA chain (track {track}" + (f", {learned} in the pass)" if learned else ")"))
                try:
                    trajectory = run_chain(inner / "mav0", trajectory, inner / "mav0" / "imu0" / "data.csv",
                                           ba_calib, track, work / "ba", work / "ba_chain.tum", args.ba_jobs,
                                           args.ba_threads, python, log=lambda m, seq=seq: log(f"{seq}: {m}"),
                                           low_light=low_light, rematch=args.ba_rematch, pass_features=learned,
                                           pass_match=args.ba_pass_match)
                except subprocess.CalledProcessError as error:  # a VIO estimate scores, none does not
                    log(f"{seq}: WARN BA chain failed ({error.cmd[3]}); submitting the VIO trajectory")
                before = work / "ba" / "ba2.tum"
                if args.ba_rematch and args.log_dir and (work / "ba" / "ba3.tum").is_file() and before.is_file():
                    (args.log_dir / seq).mkdir(parents=True, exist_ok=True)  # to score the rematch against
                    run([python, estimate_converter, "--in-tum", before,
                         "--out-estimate", args.log_dir / seq / "ba2_estimate.txt"], check=True)

            run([python, estimate_converter, "--in-tum", trajectory,
                 "--out-estimate", estimate], check=True)
            if estimate.is_file():
                log(f"{seq}: DONE ({sum(1 for _ in estimate.open())} poses)")
            else:
                log(f"{seq}: FAIL estimate conversion")
        except Exception as error:  # keep the loop going across sequences
            log(f"{seq}: EXCEPTION {error!r}")
        finally:
            if args.log_dir:
                for step_log in sorted((work / "ba").glob("*.log")):
                    (args.log_dir / seq).mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(step_log, args.log_dir / seq / step_log.name)
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
