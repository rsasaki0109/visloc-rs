#!/usr/bin/env python3
"""Write walking pedestrians as Gaussian-splat PLYs, one per gait phase.

For simulators that move splat actors inside a splat scene (Robot Native
Engine's `load_splat_actor`): each walker of `people.Figure` (seeded, varied
clothes) at each of `people.PHASES` gait phases, as flat gaussians
(`people.mesh_splats`), in the walker's own frame: x forward, z up, feet at
z = 0, metres. Cycling the phases while moving the actor at its walking speed
animates the gait (one cycle is `STRIDE_M` of travel).

Every phase of a walker has the same number of gaussians (padded with
transparent ones), so a simulator can swap them in place.

The PLY layout is the one 3DGS viewers read without spherical harmonics:
x y z f_dc_0..2 opacity scale_0..2 rot_0..3 (rot as w x y z), binary little
endian; colour as the degree-0 coefficient, opacity as a logit, scales as
logs.

  python3 walker_splat_plys.py --out walkers --count 4 --seed 0

writes walkers/walker_<i>_phase_<k>.ply and walkers/walkers.json (the files
per walker, the phase count, the stride).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import people

SH_C0 = 0.28209479177387814
# Travel per gait cycle (two steps), for driving the phase from the distance walked.
STRIDE_M = 1.4


def write_ply(path, rows):
    """rows: N x 14 as people.mesh_splats (x y z, log scales, quat w x y z,
    opacity logit, r g b in 0..1)."""
    n = len(rows)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(
            f"property float {name}\n"
            for name in [
                "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
            ]
        )
        + "end_header\n"
    )
    out = np.zeros((n, 14), np.float32)
    out[:, 0:3] = rows[:, 0:3]
    out[:, 3:6] = (rows[:, 11:14] - 0.5) / SH_C0
    out[:, 6] = rows[:, 10]
    out[:, 7:10] = rows[:, 3:6]
    out[:, 10:14] = rows[:, 6:10]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(out.astype("<f4").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    walkers = []
    for i in range(args.count):
        figure = people.Figure(rng)
        phases = [people.mesh_splats(mesh) for mesh in figure.frames]
        # Every phase the same count (a simulator swaps them in place): pad
        # with transparent gaussians at the feet.
        count = max(len(rows) for rows in phases)
        files = []
        for k, rows in enumerate(phases):
            pad = np.zeros((count - len(rows), 14), np.float32)
            pad[:, 3:6] = np.log(0.001)
            pad[:, 6] = 1.0
            pad[:, 10] = -20.0
            name = f"walker_{i}_phase_{k}.ply"
            write_ply(args.out / name, np.concatenate([rows, pad]))
            files.append(name)
        walkers.append(files)
    (args.out / "walkers.json").write_text(
        json.dumps({"phases": people.PHASES, "stride_m": STRIDE_M, "walkers": walkers}, indent=1)
    )
    print(f"{args.count} walkers x {people.PHASES} phases -> {args.out}")


if __name__ == "__main__":
    main()
