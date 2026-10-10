#!/usr/bin/env python3
"""Apply Project Aria's factory IMU calibration to a LaMAria ASL `imu0/data.csv`.

LaMAria's ASL export ships the *raw* Aria IMU samples. Aria's factory
calibration models each IMU as

    raw = R * real + b

per sensor (gyroscope and accelerometer separately), where R is a 3x3
"rectification" matrix (scale, misalignment) and b a constant bias. The ASL
release does not apply it and its pinhole calibration JSON does not carry it;
it lives in the JSON calibration record embedded in the header of the raw
`.vrs` recording. This script has two subcommands:

* `extract`: pull that JSON record out of the first bytes of a `.vrs` file
  (the record sits within the first 4 MiB, so an HTTP range request of
  `0-4194303` is enough -- the multi-GB recording is never downloaded):

      curl -fsS -r 0-4194303 -o seq.head \\
          https://cvg-data.inf.ethz.ch/lamaria/raw_data/training/<seq>.vrs
      aria_factory_imu_rectify.py extract seq.head seq_factory_calib.json

* `rectify`: write `real = R^-1 (raw - b)` for every sample to a new CSV
  (same ASL column layout), using the `imu-right` IMU -- the one LaMAria's
  ASL `imu0` stream and `T_b_s` frames refer to:

      aria_factory_imu_rectify.py rectify seq_factory_calib.json \\
          in/mav0/imu0/data.csv out/mav0/imu0/data.csv

The camera images are untouched, so a rectified sequence can share them
(e.g. hardlinks) with the raw one. Results: docs/lamaria_imu_rectification.md.
"""

import argparse
import json
import sys

import numpy as np

ASL_IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
)


VRS_HEAD_BYTES = 4 * 1024 * 1024


def extract_record(blob: bytes) -> dict:
    """Return the factory calibration JSON record embedded in a .vrs header."""
    key = blob.find(b'"ImuCalibrations"')
    if key < 0:
        raise ValueError("no ImuCalibrations record in the .vrs header")
    # Walk back over enclosing '{' until one parses as a complete JSON object
    # that contains the IMU calibrations.
    decoder = json.JSONDecoder()
    start = blob.rfind(b"{", 0, key)
    while start >= 0:
        try:
            obj, _ = decoder.raw_decode(blob[start:].decode("latin1"))
            if "ImuCalibrations" in obj:
                return obj
        except ValueError:
            pass
        start = blob.rfind(b"{", 0, start)
    raise ValueError("could not parse the calibration record in the .vrs header")


def sensor_model(sensor):
    bias = np.array(sensor["Bias"]["Offset"])
    rect_inv = np.linalg.inv(np.array(sensor["Model"]["RectificationMatrix"]))
    return bias, rect_inv


def rectify_csv(calib: dict, inp, out, label="imu-right", part="both", shift_ns=0):
    """Write `real = R^-1 (raw - b)` for every sample of an ASL IMU CSV.

    Returns (samples, gyro bias, accel bias).
    """
    imus = [x for x in calib["ImuCalibrations"] if x["Label"] == label]
    if not imus:
        raise ValueError(f"no IMU labelled {label!r} in the calibration")
    bg, rg_inv = sensor_model(imus[0]["Gyroscope"])
    ba, ra_inv = sensor_model(imus[0]["Accelerometer"])

    with open(inp) as f:
        header = f.readline()
    data = np.loadtxt(inp, delimiter=",", comments="#", ndmin=2)
    ts = data[:, 0].astype(np.int64) + shift_ns
    gyro, acc = data[:, 1:4], data[:, 4:7]
    if part in ("both", "gyro"):
        gyro = (rg_inv @ (gyro - bg).T).T
    if part in ("both", "accel"):
        acc = (ra_inv @ (acc - ba).T).T

    with open(out, "w") as f:
        f.write(header if header.startswith("#") else ASL_IMU_HEADER)
        for t, g, a in zip(ts, gyro, acc):
            f.write(f"{t},{g[0]:.9g},{g[1]:.9g},{g[2]:.9g},{a[0]:.9g},{a[1]:.9g},{a[2]:.9g}\n")
    return len(ts), bg, ba


def extract(args):
    try:
        obj = extract_record(open(args.vrs_head, "rb").read())
    except ValueError as error:
        sys.exit(f"{args.vrs_head}: {error}")
    with open(args.out, "w") as f:
        json.dump(obj, f, indent=1)
    print(obj.get("Serial"), [c["Label"] for c in obj["ImuCalibrations"]])


def rectify(args):
    try:
        n, bg, ba = rectify_csv(json.load(open(args.calib)), args.inp, args.out,
                                args.label, args.part, args.shift_ns)
    except ValueError as error:
        sys.exit(f"{args.calib}: {error}")
    print(f"wrote {args.out}: {n} samples, gyro bias {bg}, accel bias {ba}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="pull the factory calibration JSON out of a .vrs header")
    e.add_argument("vrs_head", help="the first 4 MiB (or all) of a .vrs file")
    e.add_argument("out", help="output JSON")
    e.set_defaults(func=extract)

    r = sub.add_parser("rectify", help="apply the factory IMU model to an ASL imu0/data.csv")
    r.add_argument("calib", help="JSON written by `extract`")
    r.add_argument("inp", help="input ASL imu0/data.csv (raw)")
    r.add_argument("out", help="output ASL imu0/data.csv (rectified)")
    r.add_argument("--label", default="imu-right", help="IMU to use (default: imu-right)")
    r.add_argument("--part", choices=["both", "gyro", "accel"], default="both",
                   help="rectify only one sensor (ablation)")
    r.add_argument("--shift-ns", type=int, default=0, help="added to every IMU timestamp")
    r.set_defaults(func=rectify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
