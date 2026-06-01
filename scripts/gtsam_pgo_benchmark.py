#!/usr/bin/env python3
"""GTSAM baseline for the SE(3) pose-graph benchmark — the OSS opponent.

Loads the same SE-Sync `.g2o` datasets the Rust `pgo_g2o_benchmark` example
solves, fixes the gauge with a tight prior on the first pose (matching the Rust
solver's anchor on pose 0), runs GTSAM's Levenberg-Marquardt, and reports the
final chi^2 in the *g2o convention* (Sum e^T Omega e) so it is directly
comparable to the Rust `final chi^2`.

GTSAM's `graph.error(values)` returns 0.5 * Sum ||e||^2_Sigma = 0.5 * chi^2_g2o,
so chi^2_g2o = 2 * graph.error(values). Only the optimize() call is timed (file
parsing excluded), matching how the Rust benchmark times its solve.

Usage:
    python3 scripts/gtsam_pgo_benchmark.py datasets/pgo_g2o/*.g2o
"""
import sys
import time

import gtsam
import numpy as np


def solve(path: str, chordal_init: bool) -> None:
    graph, initial = gtsam.readG2o(path, is3D=True)

    # Optional chordal rotation initialization (Carlone et al.) — the apples-to-
    # apples seed for comparing against the Rust solver's `--chordal-init`. GTSAM
    # exposes it as InitializePose3.initialize; time it separately like the Rust
    # benchmark does.
    init_ms = 0.0
    if chordal_init:
        t0 = time.perf_counter()
        initial = gtsam.InitializePose3.initialize(graph)
        init_ms = (time.perf_counter() - t0) * 1e3

    # Fix the gauge: a tight prior on key 0 at its initial value (the Rust solver
    # anchors pose 0). chi^2 is gauge-invariant, but GTSAM's LM needs a
    # non-singular system. Sigma 1e-3 on all 6 DOF -> Omega 1e6, negligible cost
    # since pose 0 stays at its prior mean.
    first_key = min(initial.keys())
    prior_model = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-3))
    graph.add(gtsam.PriorFactorPose3(first_key, initial.atPose3(first_key), prior_model))

    chi2_init = 2.0 * graph.error(initial)

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

    t0 = time.perf_counter()
    result = optimizer.optimize()
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    chi2_final = 2.0 * graph.error(result)
    name = path.rsplit("/", 1)[-1]
    seed = "chordal" if chordal_init else "raw    "
    total_ms = init_ms + elapsed_ms
    print(
        f"{name:<20} seed={seed} poses={initial.size():>6} factors={graph.size():>6} "
        f"iters={optimizer.iterations():>3} "
        f"chi2_init={chi2_init:.6e} chi2_final={chi2_final:.6e} "
        f"init={init_ms:.1f}ms solve={elapsed_ms:.1f}ms total={total_ms:.1f}ms"
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--chordal-init"]
    chordal_init = "--chordal-init" in sys.argv[1:]
    if not args:
        print("usage: gtsam_pgo_benchmark.py [--chordal-init] <dataset.g2o> [more.g2o ...]")
        sys.exit(2)
    print(f"GTSAM {gtsam.__version__ if hasattr(gtsam, '__version__') else '4.x'} "
          f"LevenbergMarquardt, chi^2 in g2o convention (Sum e^T Omega e)")
    for path in args:
        try:
            solve(path, chordal_init)
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            print(f"{path}: FAILED ({exc})")


if __name__ == "__main__":
    main()
