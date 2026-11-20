# Reproducibility contract

What visloc-rs promises about getting the same answer twice, what it does not
promise, and the rules new code follows to keep those promises. The idea of
writing determinism down as a contract, and several rules below, come from
[OpenMVG-rs's reproducibility contract](https://github.com/openMVG/openmvg-rs/blob/main/docs/REPRODUCIBILITY.md).

## What is guaranteed

Same binary, same input, same configuration, same seeds: the same output, to
the bit. This covers:

- **Model choices and inlier sets.** Every RANSAC in the library (PnP, essential,
  fundamental, homography, generalized relative pose) draws from a `SmallRng`
  seeded from its config's `seed` field. No library code seeds from entropy
  (`thread_rng`, `from_entropy`); only tests and benchmarks may.
- **Thread count.** Where a loop runs on rayon, the parallel path gives the
  same bits as the serial one at any thread count:
  - the parallel bundle-adjustment assembly computes each observation's
    contribution on the pool, then scatters them serially in ascending
    observation order (`pipelines/slam/src/bundle/normal_equations.rs`,
    checked by `parallel_ba_tests.rs`);
  - the `parallel` feature of `visloc-vision` scores a fixed-budget PnP RANSAC's
    hypotheses on rayon, but draws samples in order from the seeded RNG and
    picks the best hypothesis in iteration order.
- **Build-level speed levers.** The `release-rt` profile, `mimalloc-global` and
  `basalt-rt-priority` are adopted only after a VIO trajectory hash matches the
  `release` build bit for bit.

## What is not guaranteed

- **Bit identity across rebuilds or toolchains.** Two `cargo build --release`
  runs from the same source have shifted the EuRoC V2_01 rigid ATE at the
  1e-3 m level: codegen differences (FMA fusion, instruction selection) flip a
  borderline inlier test, and RANSAC amplifies it. `rust-toolchain.toml` pins
  the compiler to remove one source of this, and
  `scripts/verify_binary_determinism.sh` measures what remains. See
  [binary_determinism_findings.md](binary_determinism_findings.md).
- **Bit identity across targets, math libraries or GPU backends.** Native vs
  wasm32, CPU vs wgpu, and two GPU vendors each need their own stated tolerance.
- **Timing and progress output.** Wall-clock, memory and log lines are telemetry
  and may differ freely.

## Rules for new code

1. **Never let telemetry decide.** A timer, a progress counter or a memory
   reading must not change a numerical decision: no "stop when the budget runs
   out" that depends on wall-clock in a path whose result is reported as a
   benchmark number. A real-time mode that drops work on a deadline is fine,
   but it is a separate, documented mode, and its outputs are not compared
   bit for bit.
2. **Seed from the problem, not from the schedule.** An RNG stream must not
   depend on worker index, task completion order or `HashMap` iteration order.
   When several independent problems each need a stream (one RANSAC per image
   pair, say), derive each stream's seed from the config seed and a stable
   identifier of the problem (for example the image-ID pair), not from a
   counter of how many problems ran before.
3. **Reduce in a fixed order.** Parallel results are collected and then
   combined in a canonical order (ascending ID or index). Do not sum
   floating-point values in completion order; a reassociated sum changes bits.
   Iterate `BTreeMap`/sorted keys, not `HashMap`, wherever the order reaches a
   result.
4. **A protocol change is a new baseline, not a regression or a win.**
   Changing how seeds are derived, the RNG, the sampler or the iteration budget
   changes which samples are drawn, so the old and new runs are not paired. Do
   not compare one favourable seed against the old number. Compare
   distributions over a fixed set of seeds (at least five), with the acceptance
   margin decided before running, and record the new baseline under its own
   name. Do not tune thresholds after seeing the new stream's results.
5. **Record what produced a number.** A benchmark result committed to `docs/`
   or `benchmarks/` states the commit, the toolchain, the build profile and
   features, the seeds, and a hash of the input data or its official name and
   version, so that someone else can reproduce it.

## Checking a change

- A change that should not move any output (refactor, speed-up, a new
  parallel path): show that an output hash (trajectory, model, inlier set) is
  identical before and after, on the same binary-producing toolchain.
- A change meant to move the output: follow rule 4.
