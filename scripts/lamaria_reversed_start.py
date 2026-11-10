#!/usr/bin/env python3
"""Fill the poses a late VIO start leaves out, from a VIO run backwards in time.

    python scripts/lamaria_reversed_start.py reverse --inner work/extracted/sequence_4_11 --out work/reversed --until 80
    python scripts/lamaria_reversed_start.py join --forward work/vio/trajectory.tum \
        --backward work/vio_reversed/trajectory.tum --reversed work/reversed \
        --mav0 work/extracted/sequence_4_11/mav0 --out work/vio_filled.tum

A VIO that only holds when it starts some seconds into the recording (run_lamaria_test_submission.py, LATE_STARTS_S)
has no poses before that start. `reverse` writes an EuRoC folder of the recording's first --until seconds with time
reversed: the frames in reverse order under mirrored timestamps (t' = C - t), the gyroscope negated, the
accelerometer as it is (the specific force does not change under time reversal). The same VIO runs on it as on any
recording. `join` takes that run's poses back to the frames' own times and moves them onto the late VIO's by the
rigid transform that fits the two over --window seconds around --join-after seconds after the late start, where
both have settled. Before the join every frame gets the reversed run's pose, from it on the late VIO's stay as
they are.

The join's position residual says whether the two runs agree: 0.01-0.15 m on seven training sequences started 60 s
late, 24 m on one whose late VIO diverges (docs/lamaria_batch_ba.md#the-poses-before-a-late-start). No ground
truth is used.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Seconds after the late VIO's first pose. 3-10 s join alike; later, the reversed run has not settled from its own start
JOIN_AFTER_S = 5.0
JOIN_WINDOW_S = 5.0


def manifest(path):
    """An EuRoC data.csv's comment rows and data rows."""
    rows = [r for r in Path(path).read_text().splitlines() if r.strip()]
    return [r for r in rows if r.startswith("#")], [r for r in rows if not r.startswith("#")]


def reversed_start(inner, out, seconds):
    """An EuRoC folder of `inner`'s first `seconds` with time reversed (manifests under mirrored times, the images
    linked); returns C, the sum of a time and its mirror, also written to <out>/mirror_ns.txt."""
    inner, out = Path(inner), Path(out)
    _, frames = manifest(inner / "mav0" / "cam0" / "data.csv")
    times = [int(r.split(",")[0]) for r in frames]
    first = times[0]
    last = max(t for t in times if t <= first + int(seconds * 1e9))
    # no mirrored time may be a frame's own time: an image looked up by its time instead of its name would be another's
    taken = set(times)
    c = first + last + 1_000_003
    while any(c - t in taken for t in times if t <= last):
        c += 1
    for cam in ("cam0", "cam1"):
        (out / "mav0" / cam).mkdir(parents=True, exist_ok=True)
        head, rows = manifest(inner / "mav0" / cam / "data.csv")
        rows = [r.split(",") for r in rows if first <= int(r.split(",")[0]) <= last]
        (out / "mav0" / cam / "data.csv").write_text(
            "\n".join(head + [f"{c - int(t)},{name.strip()}" for t, name in reversed(rows)]) + "\n")
        link = out / "mav0" / cam / "data"
        if not link.exists():
            link.symlink_to((inner / "mav0" / cam / "data").resolve(), target_is_directory=True)
    (out / "mav0" / "imu0").mkdir(parents=True, exist_ok=True)
    head, rows = manifest(inner / "mav0" / "imu0" / "data.csv")
    samples = []
    for row in rows:
        v = [s.strip() for s in row.split(",")]
        if first - 100_000_000 <= int(v[0]) <= last + 100_000_000:
            gyro = [s[1:] if s.startswith("-") else "-" + s for s in v[1:4]]
            samples.append(",".join([str(c - int(v[0]))] + gyro + v[4:7]))
    (out / "mav0" / "imu0" / "data.csv").write_text("\n".join(head + samples[::-1]) + "\n")
    (out / "mirror_ns.txt").write_text(f"{c}\n")
    return c


def read_tum(path):
    """A TUM trajectory's timestamps in ns (exact, from the decimal string) and its position and quaternion rows."""
    ns, poses = [], []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            v = line.split()
            whole, _, frac = v[0].partition(".")
            ns.append(int(whole) * 10 ** 9 + int((frac + "000000000")[:9]))
            poses.append([float(x) for x in v[1:8]])
    return np.array(ns, np.int64), np.array(poses)


def write_tum(path, ns, poses):
    with open(path, "w") as f:
        for n, v in zip(ns, poses):
            f.write(f"{n // 10 ** 9}.{n % 10 ** 9:09d} {v[0]:.9f} {v[1]:.9f} {v[2]:.9f} "
                    f"{v[3]:.12f} {v[4]:.12f} {v[5]:.12f} {v[6]:.12f}\n")


def forward_times(tum, c, out):
    """The reversed run's trajectory `tum` under the frames' own times (t = C - t'), in their order; returns `out`."""
    ns, poses = read_tum(tum)
    order = np.argsort(c - ns)
    write_tum(out, (c - ns)[order], poses[order])
    return Path(out)


def fit(p_to, q_to, p_from, q_from):
    """The rigid transform taking the `from` poses onto the `to` ones: the chordal mean of the relative rotations,
    then the mean translation."""
    m = sum(Rotation.from_quat(a).as_matrix() @ Rotation.from_quat(b).as_matrix().T for a, b in zip(q_to, q_from))
    u, _, vt = np.linalg.svd(m)
    r = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    return r, (p_to - p_from @ r.T).mean(0)


def join(forward, backward, mav0, out, after=JOIN_AFTER_S, window=JOIN_WINDOW_S, log=print):
    """Write `out`: the late VIO's trajectory `forward` with the poses of `backward` (the reversed run's, under the
    frames' own times) at every cam0 frame before the join, `after` seconds after `forward`'s first pose. Returns
    the position rms in metres between the two over the `window` seconds around the join, after the fit."""
    f_ns, f_poses = read_tum(forward)
    b_ns, b_poses = read_tum(backward)
    t_join = f_ns[0] + int(after * 1e9)
    common, fi, bi = np.intersect1d(f_ns, b_ns, return_indices=True)
    near = np.abs(common - t_join) <= int(window * 0.5e9)
    if near.sum() < 10:
        raise ValueError(f"only {near.sum()} poses of both runs around the join")
    fi, bi = fi[near], bi[near]
    r, t = fit(f_poses[fi, :3], f_poses[fi, 3:], b_poses[bi, :3], b_poses[bi, 3:])
    b_poses = np.c_[b_poses[:, :3] @ r.T + t, (Rotation.from_matrix(r) * Rotation.from_quat(b_poses[:, 3:])).as_quat()]
    residual = float(np.sqrt((np.linalg.norm(b_poses[bi, :3] - f_poses[fi, :3], axis=1) ** 2).mean()))
    turn = (Rotation.from_quat(f_poses[fi, 3:]) * Rotation.from_quat(b_poses[bi, 3:]).inv()).magnitude()
    # every frame before the join gets a pose, the nearest one's where the reversed run has none: the chain reads
    # the trajectory at its keyframes
    _, frames = manifest(Path(mav0) / "cam0" / "data.csv")
    want = np.array([int(row.split(",")[0]) for row in frames], np.int64)
    want = want[want < t_join]
    right = np.clip(np.searchsorted(b_ns, want), 0, len(b_ns) - 1)
    left = np.clip(right - 1, 0, len(b_ns) - 1)
    pick = np.where(np.abs(b_ns[left] - want) < np.abs(b_ns[right] - want), left, right)
    kept = f_ns >= t_join
    write_tum(out, np.concatenate([want, f_ns[kept]]), np.concatenate([b_poses[pick], f_poses[kept]]))
    log(f"joined {after:.0f} s after the late start over {near.sum()} poses: {residual:.3f} m rms, "
        f"{np.degrees(turn).mean():.2f} deg; {len(want)} frames before it "
        f"({int((b_ns[pick] != want).sum())} with the nearest pose), {kept.sum()} poses after")
    return residual


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("reverse", help="the time-reversed EuRoC folder of a recording's first seconds")
    p.add_argument("--inner", type=Path, required=True, help="the recording's EuRoC folder (holds mav0)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--until", type=float, required=True, help="seconds into the recording the reversed run starts at")
    p = sub.add_parser("join", help="the late VIO's trajectory with the reversed run's poses before the join")
    p.add_argument("--forward", type=Path, required=True, help="the late VIO's trajectory (TUM)")
    p.add_argument("--backward", type=Path, required=True, help="the reversed run's trajectory (TUM, mirrored times)")
    p.add_argument("--reversed", type=Path, required=True, help="the folder `reverse` wrote (its mirror_ns.txt)")
    p.add_argument("--mav0", type=Path, required=True, help="the recording's mav0 (its frames' times)")
    p.add_argument("--join-after", type=float, default=JOIN_AFTER_S, help="seconds after the late VIO's first pose")
    p.add_argument("--window", type=float, default=JOIN_WINDOW_S, help="seconds of poses the transform is fitted over")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reverse":
        c = reversed_start(args.inner, args.out, args.until)
        print(f"{args.out}: the first {args.until:.0f} s reversed, C = {c}")
    else:
        c = int((args.reversed / "mirror_ns.txt").read_text())
        backward = forward_times(args.backward, c, args.out.with_name(args.out.stem + "_reversed.tum"))
        join(args.forward, backward, args.mav0, args.out, args.join_after, args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
