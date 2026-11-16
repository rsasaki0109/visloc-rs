#!/usr/bin/env python3
"""The real-time LaMAria chain behind a VIO trajectory: lamaria_batch_ba.py's features, match, loops and online,
then one batch pass after the recording (docs/lamaria_batch_ba.md#the-first-minutes-online).

    python scripts/lamaria_ba_chain.py --mav0 seq/mav0 --vio vio/trajectory.tum --imu seq/mav0/imu0/data.csv \\
        --calib seq_calib_ba.json --track 2 --work work/seq --out seq.tum

--vio is the VIO on the rectified, time-aligned IMU with the per-device camera rig (Basalt's `euroc_config`,
`euroc_config_lowlight` in low light); --imu is that IMU; --calib is the Basalt calibration of the pinhole images
with the factory camera rig and `gyro_bias_std` 1e-7 (the BA estimates the camera rotations itself); --track is
the leaderboard track (1 Short, 2 Medium, 3 Long, 4 Low light, 5 Moving platform), which picks the options:

- low light: CLAHE and non-local means before SIFT, no warm-up online, 30 iterations per solve in the pass and
  the gyroscope's response matrix estimated there (sequence_4_10 and 4_11 over three IMU seeds: 63.87 at the
  online result, 68.77 after the pass, 71.19 with the matrix; --low-light-online stops at the online result,
  docs/lamaria_batch_ba.md#low-light-the-gyroscopes-response-matrix);
- moving platform: no focal-length estimation;
- by day: the gyro bias random walk ten times looser over the first minutes (online and pass) and ten times
  looser throughout in the pass.

--rematch (the submission driver's default) matches the keyframes and finds the revisits once more after
the pass, guided by the pass's trajectory instead of the VIO, and solves again from there with those matches:
training R_12_10cp, whose VIO heading drifts 0.78 deg/min, goes from 61.09 to 64.30 (the same solve with the
VIO's matches: 60.48), the eight day training sequences from 64.50 to 64.76 on average
(docs/lamaria_batch_ba.md#matching-again-after-the-pass).

--features and --pass-features (the pass only, by day) try LightGlue's learned keypoints on the GPU; neither beats
SIFT across the training sequences (docs/lamaria_batch_ba.md#learned-features).

A step whose output exists is skipped, so an interrupted run resumes; the online result and the pass are only
taken as done once their step finished.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

LOW_LIGHT, MOVING = 4, 5
BA = Path(__file__).resolve().parent / "lamaria_batch_ba.py"
WARMUP = ["--gyro-rw-warmup", "10", "--gyro-rw-warmup-s", "300"]
LEARNED_KEYPOINTS = 1024


def feature_options(features, num_features=None):
    """The features step's options: SIFT, or a learned extractor keeping its `num_features` best keypoints (1,024
    by default) whatever their score, so that LightGlue matches the same number in every image, in batches."""
    if features in ("sift", "rootsift"):
        return ([] if features == "sift" else ["--rootsift"]) + \
            ([] if num_features is None else ["--num-features", str(num_features)])
    return ["--extractor", features, "--num-features", str(num_features or LEARNED_KEYPOINTS),
            "--detection-threshold", "0"]


PASS_MATCH = ("vio", "online", "essential")


def keypoints(mav0, vio, calib, work, jobs, feats, name="", match=(), match_args=()):
    """The features, match and loops steps into `work`, guided by the VIO; `match` replaces the match step's
    guidance (--vio TUM, or --verify essential)."""
    return [
        (f"{name}features", ["features", "--mav0", mav0, "--vio", vio, "--work", work, "--jobs", jobs, *feats],
         work / "features.npz", False),
        (f"{name}match", ["match", "--work", work, *(match or ["--vio", vio]), "--calib", calib, "--jobs", jobs,
                          "--tau-slope", "0.3", "--tau-max", "20", *match_args], work / "match.done", False),
        (f"{name}loops", ["loops", "--work", work, "--vio", vio, "--calib", calib, "--jobs", jobs, "--step", "4"],
         work / "loops.npz", False),
    ]


def pass_work(work, pass_features=None):
    """Where the pass reads its keypoints, matches and revisits: work itself, or work/<extractor>."""
    return Path(work) / pass_features if pass_features else Path(work)


def steps(mav0, vio, imu, calib, track, work, jobs, low_light=None, rematch=False, features="sift",
          num_features=None, pass_features=None, pass_match="vio", pass_args=(), low_light_pass=True, match_args=(), flow=False):
    """(name, lamaria_batch_ba.py arguments, output that marks it done, threaded) per step, in order.

    `low_light` overrides the track's choice of the low-light options (a Low light sequence shot in daylight);
    `rematch` adds the matching guided by the pass and a solve with its matches (in work/rematch);
    `features` picks the keypoints: "sift", or a learned extractor that match and loops pair with LightGlue;
    `pass_features` gives the pass by day keypoints of its own: a learned extractor's, in work/<extractor>, matched
    and searched for revisits guided by the VIO like the online windows' (the rematch then goes there too);
    `pass_match` keeps those matches within the VIO's epipolar band ("vio"), the online result's ("online"), or
    those an essential matrix explains ("essential"). In low light the pass runs on the low-light keypoints, with
    30 iterations per solve and the gyroscope's response matrix; `low_light_pass=False` stops there at the online windows instead. `match_args` are
    extra options of the match steps (the first matching and the rematch); `flow` links consecutive keyframes
    by Lucas-Kanade flow through the frames between them (the flow step, after match) for every solve."""
    work = Path(work)
    low = track == LOW_LIGHT if low_light is None else low_light
    solve = ["--imu", imu, "--calib", calib, "--sigma-size", "2.7", "--sigma-size-pow", "0.35"]
    solve += [] if track == MOVING else ["--est-focal"]
    feats = ["--clahe", "--denoise", "7"] if low else []
    out = keypoints(mav0, vio, calib, work, jobs, [*feats, *feature_options(features, num_features)],
                    match_args=match_args)
    if flow and (features not in ("sift", "rootsift") or pass_features):
        raise ValueError("the flow adds SIFT keypoints: not with learned features")
    if flow:  # after match, which it reads, and before anything that reads its keypoints
        out.insert(2, ("flow", ["flow", "--mav0", mav0, "--work", work, "--jobs", jobs], work / "flow.npz", False))
        solve += ["--flow", work / "flow.npz"]
    out.append(("online", ["online", "--work", work, "--vio", vio, "--out", work / "online.tum", "--loops", *solve,
                           *([] if low else WARMUP)], work / "online.done", True))
    if not low or low_light_pass:  # the batch pass after the recording, its random walk ten times looser
        pass_features = None if low else pass_features
        pw = pass_work(work, pass_features)
        if pass_features:
            guide = {"vio": (), "online": ("--vio", work / "online.tum"),
                     "essential": ("--vio", vio, "--verify", "essential")}[pass_match]
            out += keypoints(mav0, vio, calib, pw, jobs, feature_options(pass_features), name="pass ", match=guide)
        iters = ["--iterations", "30", "--rel-tol", "1e-5", "--est-gyro-m"] if low else ["--iterations", "60"]
        pas = [*solve, *iters, "--gyro-rw-scale", "10", *WARMUP, *pass_args]
        out += [
            ("pass solve", ["solve", "--work", pw, "--vio", work / "online.tum", "--out", work / "ba1.tum",
                            "--save-states", *pas], work / "ba1.done", True),
            ("loop-correct", ["loop-correct", "--work", pw, "--calib", calib, "--states",
                              work / "ba1_states.npz", "--tum", work / "ba1.tum", "--out", work / "lc.tum"],
             work / "lc.tum", False),
            ("pass solve with the revisits", ["solve", "--work", pw, "--vio", work / "lc.tum", "--out",
                                              work / "ba2.tum", "--loops", *pas], work / "ba2.done", True),
        ]
        if rematch:  # the pass's trajectory guides the matching: its heading drifts far less than the VIO's
            again = pw / "rematch"
            out += [
                ("rematch", ["match", "--work", again, "--vio", work / "ba2.tum", "--calib", calib, "--jobs", jobs,
                             "--tau-slope", "0.3", "--tau-max", "20", *match_args], again / "match.done", False),
                ("rematch loops", ["loops", "--work", again, "--vio", work / "ba2.tum", "--calib", calib, "--jobs",
                                   jobs, "--step", "4"], again / "loops.npz", False),
                ("solve with the new matches", ["solve", "--work", again, "--vio", work / "ba2.tum", "--out",
                                                work / "ba3.tum", "--loops", *pas], work / "ba3.done", True),
            ]
    return out


def result(work, track, low_light=None, rematch=False, low_light_pass=True):
    """The trajectory the chain ends with."""
    low = (track == LOW_LIGHT if low_light is None else low_light) and not low_light_pass
    return Path(work) / ("online.tum" if low else "ba3.tum" if rematch else "ba2.tum")


def run_chain(mav0, vio, imu, calib, track, work, out, jobs=4, threads=2, python=sys.executable, log=print, ba=BA,
              low_light=None, rematch=False, features="sift", num_features=None, pass_features=None,
              pass_match="vio", pass_args=(), low_light_pass=True, match_args=(), flow=False):
    work = Path(work)
    pw = pass_work(work, pass_features)
    pw.mkdir(parents=True, exist_ok=True)
    if rematch:  # the second matching reads the same keypoints
        (pw / "rematch").mkdir(exist_ok=True)
        if not (pw / "rematch" / "features.npz").is_symlink():
            (pw / "rematch" / "features.npz").symlink_to(Path("..") / "features.npz")
    for name, cmd, done, threaded in steps(mav0, vio, imu, calib, track, work, jobs, low_light, rematch, features,
                                           num_features, pass_features, pass_match, pass_args, low_light_pass,
                                           match_args, flow):
        if done.exists():
            log(f"{name}: done before, skipped")
            continue
        n = threads if threaded else 1  # match, loops and features run --jobs processes of one thread each
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(n), OMP_NUM_THREADS=str(n), NUMBA_NUM_THREADS=str(n))
        stem = name.replace(" ", "_")
        t0 = time.time()
        with open(work / f"{stem}.log", "w") as f:
            subprocess.run([python, "-I", ba, *[str(c) for c in cmd]], check=True, stdout=f,
                           stderr=subprocess.STDOUT, env=env)
        if done.suffix == ".done":
            done.touch()
        log(f"{name}: {time.time() - t0:.0f} s")
    for matches in {work / "matches.npz", pw / "matches.npz", pw / "rematch" / "matches.npz"}:
        matches.unlink(missing_ok=True)  # 0.2 GB, not needed once the chain has run
    shutil.copyfile(result(work, track, low_light, rematch, low_light_pass), out)
    return Path(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mav0", required=True, help="ASL mav0 directory (cam0/data, cam1/data)")
    p.add_argument("--vio", required=True, help="the VIO trajectory (TUM)")
    p.add_argument("--imu", required=True, help="the IMU csv the VIO read")
    p.add_argument("--calib", required=True, help="Basalt calibration with the factory camera rig")
    p.add_argument("--track", type=int, required=True, choices=range(1, 6))
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True, help="the chain's trajectory (TUM)")
    p.add_argument("--jobs", type=int, default=4, help="processes for features, match and loops")
    p.add_argument("--threads", type=int, default=2, help="threads for the online windows and the pass")
    p.add_argument("--daylit", action="store_true", help="day options on a Low light sequence shot in daylight")
    p.add_argument("--low-light-online", action="store_true",
                   help="in low light, stop at the online windows (by default the pass and --rematch follow, "
                        "with 30 iterations per solve and the gyroscope's response matrix)")
    p.add_argument("--rematch", action="store_true", help="match again guided by the pass and solve again")
    p.add_argument("--features", choices=("sift", "rootsift", "aliked", "disk", "superpoint"), default="sift",
                   help="the keypoints: SIFT (or with RootSIFT descriptors), or learned ones on the GPU, "
                        "matched by LightGlue")
    p.add_argument("--num-features", type=int, default=None,
                   help="keypoints per image (default: the features step's for SIFT, 1,024 for a learned extractor)")
    p.add_argument("--pass-features", choices=("aliked", "disk", "superpoint"), default=None,
                   help="by day, learned keypoints for the pass after the online windows (in work/<extractor>)")
    p.add_argument("--pass-match", choices=PASS_MATCH, default="vio",
                   help="--pass-features: keep the matches within the VIO's epipolar band, the online result's, or "
                        "those an essential matrix explains")
    p.add_argument("--pass-args", default="", help="extra options of the pass's solves (one string, e.g. "
                   "\"--est-gyro-m\")")
    p.add_argument("--match-args", default="", help="extra options of the match steps (one string, e.g. "
                   "\"--ratio 0.9\")")
    p.add_argument("--flow", action="store_true", help="link consecutive keyframes by Lucas-Kanade flow through "
                   "the frames between them, for every solve")
    a = p.parse_args()
    run_chain(a.mav0, a.vio, a.imu, a.calib, a.track, a.work, a.out, a.jobs, a.threads,
              log=lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True),
              low_light=False if a.daylit else None, rematch=a.rematch, features=a.features,
              num_features=a.num_features, pass_features=a.pass_features, pass_match=a.pass_match,
              pass_args=a.pass_args.split(), low_light_pass=not a.low_light_online,
              match_args=a.match_args.split(), flow=a.flow)


if __name__ == "__main__":
    main()
