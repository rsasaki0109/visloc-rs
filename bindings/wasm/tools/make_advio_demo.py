#!/usr/bin/env python3
"""Builds the demo page's sensor-free demo from an ADVIO iPhone sequence:
`<out>.mp4` (the portrait video at 30 fps, 360 px wide) and `<out>.json`
(frame times, IMU samples resampled to one 60 Hz stream like a browser's
motion events, and the rig).

    python3 tools/make_advio_demo.py ~/datasets/advio/advio-02 web/demo/advio-02 \
        --start 0 --seconds 60 --td-ms -300

ADVIO (Cortés et al., ECCV 2018, CC BY-NC 4.0): sequences 1-12 use the
`iphone-02` calibration below. Its frame timestamps run ~300 ms behind its
IMU clock (see examples/advio.rs), hence --td-ms.
"""
import argparse
import bisect
import json
import math
import subprocess
from pathlib import Path

INTRINSICS_720 = [1077.2, 1079.3, 362.145, 636.3873]
DISTORTION = [0.0478, 0.0339, -0.00033, -0.00091]
# Kalibr T_cam_imu; the rig wants T_imu_cam.
T_CAM_IMU = [
    [0.9999763379093255, -0.004079205042965442, -0.005539287650170447, -0.008977668364731128],
    [-0.004066386342107199, -0.9999890330121858, 0.0023234365646622014, 0.07557012320238939],
    [-0.00554870467502187, -0.0023008567036498766, -0.9999819588046867, -0.005545773942541918],
]


def read_csv(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append([float(v) for v in line.split(",")])
    return rows


def interp(rows, times, t):
    k = bisect.bisect_left(times, t)
    if k == 0 or k >= len(rows):
        return None
    a, b = rows[k - 1], rows[k]
    w = (t - a[0]) / (b[0] - a[0])
    return [a[i] + w * (b[i] - a[i]) for i in range(1, len(a))]


def quat_from_matrix(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]) * 2
    q = [0.0] * 4
    q[0] = (m[k][j] - m[j][k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j][i] + m[i][j]) / s
    q[1 + k] = (m[k][i] + m[i][k]) / s
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--td-ms", type=float, default=-300.0)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--imu-hz", type=float, default=60.0)
    args = ap.parse_args()
    seq = Path(args.sequence)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = read_csv(seq / "iphone/frames.csv")
    acc = read_csv(seq / "iphone/accelerometer.csv")
    gyr = read_csv(seq / "iphone/gyro.csv")
    t0 = max(acc[0][0], gyr[0][0])
    first = next(i for i, r in enumerate(frames) if r[0] >= max(args.start, t0 + 0.4))
    picked = [i for i in range(first, len(frames), 2) if frames[i][0] < frames[first][0] + args.seconds]
    td = args.td_ms / 1000.0
    # Frame n of the demo video is ADVIO frame picked[n]; times on the IMU clock.
    frame_times = [frames[i][0] + td for i in picked]
    acc_t = [r[0] for r in acc]
    gyr_t = [r[0] for r in gyr]
    imu = []
    t = frame_times[0] - 0.5
    while t < frame_times[-1] + 0.5:
        a = interp(acc, acc_t, t)
        g = interp(gyr, gyr_t, t)
        if a and g:
            imu.append([round(t, 6)] + [round(v, 6) for v in g + a])
        t += 1.0 / args.imu_hz
    width = args.width
    height = width * 16 // 9
    s = width / 720.0
    fx, fy, cx, cy = INTRINSICS_720
    R = [row[:3] for row in T_CAM_IMU]
    tvec = [row[3] for row in T_CAM_IMU]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]  # R_imu_cam
    t_imu_cam = [-sum(Rt[i][k] * tvec[k] for k in range(3)) for i in range(3)]
    rig = {
        "width": width, "height": height,
        "fx": fx * s, "fy": fy * s, "cx": (cx + 0.5) * s - 0.5, "cy": (cy + 0.5) * s - 0.5,
        "distortion": DISTORTION,
        "imuFromCamera": quat_from_matrix(Rt) + t_imu_cam,
        "imuRateHz": args.imu_hz,
        "accelNoise": 0.0144, "gyroNoise": 0.0072, "accelBiasWalk": 6.3e-4, "gyroBiasWalk": 1.53e-4,
    }
    # Every picked frame, in order, at a constant 30 fps (frame n at n / 30 s).
    start_s = frames[picked[0]][0] - frames[0][0]
    vf = f"select='gte(n\\,{picked[0]})*not(mod(n-{picked[0]}\\,2))',setpts=N/30/TB,scale={width}:{height}:flags=area"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(seq / "iphone/frames.mov"),
        "-vf", vf, "-frames:v", str(len(picked)), "-r", "30", "-an",
        "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p", "-crf", "23",
        "-g", "30", "-movflags", "+faststart", str(out.with_suffix(".mp4")),
    ], check=True)
    meta = {
        "source": f"ADVIO {seq.name} (Cortés et al., ECCV 2018, CC BY-NC 4.0), from {start_s:.1f} s",
        "fps": 30,
        "frameTimes": [round(v, 6) for v in frame_times],
        "imu": imu,
        "rig": rig,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"{len(picked)} frames, {len(imu)} IMU samples -> {out.with_suffix('.mp4')}, {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
