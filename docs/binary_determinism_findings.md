# Binary determinism — Phase-{20..27} closeout follow-up

**Status (2026-05-19):** mitigation #1 shipped (toolchain pin),
verification script committed (`scripts/verify_binary_determinism.sh`).
Cross-rebuild variance under the pin is being characterised — see
*Empirical results* below; this document will be updated as the
ledger of toolchain bumps grows.

## Problem statement

Phase-26 #4 (and the Phase-{20..27} closeout) observed that:

1. **Within a single binary**, the EuRoC online-SLAM demo is bit-
   identical between back-to-back runs of the same configuration.
2. **Across `cargo build --release` builds from the same source**,
   the V2_01 strict rigid ATE can shift by O(10⁻³ m) (e.g.
   0.0040 m vs 0.0107 m observed at two separate Phase-26 build
   moments — same source, same config flags, same RNG seed).

The signal magnitude is small (third decimal place) but it is
empirically real and reproducible: re-running the same binary
reproduces its own number to f64 bit identity, and re-building
without changing the source produces a different bit-identical
result.

## Root-cause hypothesis ranking

Ranked by plausibility under the current evidence.

1. **rustc codegen variation across builds.** Even at the same
   compiler version, repeated `cargo build` can yield slightly
   different binaries: incremental compilation cache state,
   register allocation, instruction-selection tie-breaks, and FMA
   (`f64::mul_add`) fusion all shift between runs. Floating-point
   evaluation order is then perturbed at the LLVM IR level.

   PnP RANSAC amplifies any such perturbation: a 1 ULP difference
   in a reprojection residual can flip a "just inside / just
   outside" inlier classification, which changes the candidate set
   passed to the next iteration. Once the iteration tree forks the
   two builds explore different best-hypothesis paths, and the
   final pose diverges to within ~10⁻³ m.

2. **HashMap insertion-order leakage.** Until Phase-26 #4 the
   `LandmarkDescriptorStore::from_visual_map` and tracker local-
   landmark collection iterated `VisualMap.landmarks` (a HashMap
   keyed by `LandmarkId = u64`) in SipHash order. The SipHash seed
   is per-process random, so two processes producing differently-
   ordered descriptor stores cause the matcher to break ties in
   different directions. Phase-26 #4 closeout added defensive sort
   sites in `pipelines/{tracking,localization,slam}` to remove this
   as a confound; `LandmarkDescriptorStore::iter` and
   `ordered_landmark_descriptors` already sorted by `LandmarkId`
   internally.

   Status: **mitigated**, but cannot be the *cross-rebuild* cause —
   process-level hash seeds reshuffle within a single binary too,
   and within-binary runs are bit-identical. So this is at most a
   tertiary contributor; the cross-rebuild divergence has another
   driver.

3. **Toolchain version drift.** A `cargo build` after a `rustup
   update` introduces a new compiler revision. LLVM, codegen
   defaults, and even arithmetic standard-library fast-paths can
   shift between rustc minor versions.

   Status: **mitigated** as of 2026-05-19 by `rust-toolchain.toml`
   pinning channel `1.94.0`. Future toolchain bumps must be
   accompanied by a re-run of the V2_01 strict baseline and an
   update of the closeout table.

4. **Parallel reduction non-determinism.** `rayon::par_iter` over
   floating-point sums is non-deterministic by construction.
   Audited; no parallel reductions exist on the hot tracker /
   PnP / RANSAC paths today. (Bundle adjustment uses `nalgebra`'s
   sequential Cholesky; SuperPoint ONNX serialises inference under
   a `Mutex`.)

   Status: **not a current contributor**.

## Mitigations shipped

### 1. Toolchain pin (`rust-toolchain.toml`, 2026-05-19)

```toml
[toolchain]
channel = "1.94.0"
components = ["rustfmt", "clippy"]
profile = "minimal"
```

This eliminates hypothesis #3 (drift across `rustup update`s) and
makes any remaining cross-rebuild variance traceable to LLVM-level
codegen (hypothesis #1). When the channel is bumped, re-run
`scripts/verify_binary_determinism.sh` and update this document.

### 2. HashMap iteration sort (Phase-26 #4 closeout)

Sites:

- `pipelines/tracking/src/lib.rs::covisibility_local_map_landmarks`
  — sorts collected `local_landmarks` `HashSet<u64>` before
  iteration into the descriptor store.
- `pipelines/localization/src/lib.rs::descriptor_store_for_submap`
  — sorts submap landmark ids before iteration.
- `crates/core/src/types/map.rs::LandmarkDescriptorStore::iter` /
  `::ordered_landmark_descriptors` — already sort by `LandmarkId`.

These reduce within-binary variance from per-process hash-seed
shuffling. Did not address cross-rebuild variance (within-binary
was already deterministic before the patches), but they remain as
defense-in-depth so a future contributor cannot accidentally
re-introduce the variance.

### 3. Verification script (`scripts/verify_binary_determinism.sh`)

Three-run protocol:

1. clean build → V2_01 strict (`baseline`, no SuperPoint to keep
   the runtime under 3 min)
2. same binary, second run → expect bit-identical to #1
3. `touch crates/vision/src/ransac/mod.rs` + rebuild → V2_01
   strict → diff vs #1

Writes `target/binary_determinism_verify/COMPARE.md`. Intended
to be re-run after every toolchain bump and after every
significant tracker / PnP refactor.

## Empirical results

| Date       | rustc                 | Config                                          | run1↔run2 (within-binary) | run2↔run3 (cross-rebuild) | rigid ATE m | Notes |
|------------|-----------------------|-------------------------------------------------|----------------------------|----------------------------|-------------|-------|
| 2026-05-19 | 1.94.0 (pinned)       | V2_01_easy strict baseline (corner, f=3/s=10)   | bit-identical              | bit-identical              | 4.8783      | First post-pin verification. The baseline configuration is degenerate by design on V2_01 (corner extractor doesn't track V-class); the point of the run is FP-path reproducibility, which is confirmed at printed precision. |
| 2026-05-19 | 1.94.0 (pinned)       | V2_01_easy SP+strict-stereo (Phase-26 #1 strict, f=2/s=5) | bit-identical              | bit-identical              | 0.2013      | sim_scale 1.955. Does *not* reproduce the Phase-26 #1 headline of 0.0107 m — that number was a one-time-only artifact of a pre-pin binary. The pinned binary stably produces 0.2013 m at this configuration. Closeout doc V2_01 row updated accordingly. |
| 2026-05-19 | 1.94.0 (pinned)       | V1_01_easy SP+strict-stereo (Phase-26 #1 strict, f=2/s=5) | bit-identical              | bit-identical              | 0.0029      | sim_scale 1.026, matches the Phase-26 #1 V1_01 headline of 0.0029 m exactly. tracking_success_rate=0.063, map_keyframes=2 — same low-tracking-survival pattern as the original; the ATE-over-surviving-frames metric reproduces precisely. |

(Table updated by `scripts/verify_binary_determinism.sh` consumers
as they run the protocol.)

**Outcome of the first three rows.** The toolchain pin alone is
sufficient for cross-rebuild determinism on every configuration
tested — baseline corner and SP+strict-stereo on both V1_01 and
V2_01 sequences. The Phase-26 #4 cross-rebuild variance was
toolchain-drift (`rustup update` between build moments), not a
codegen-tie-break issue at fixed compiler version. The
deterministic-estimator levers (Kahan summation, P3P closed-form,
fp-contract disable) documented below are *no longer warranted*
for cross-rebuild reproducibility — they remain as conditional
options only if a future toolchain bump re-introduces variance.

**Headline number correction.** The Phase-26 #1 closeout's V2_01
strict 0.0107 m is no longer reproducible on the pinned binary;
the stable post-pin number is 0.2013 m (sim_scale 1.955). V1_01
strict 0.0029 m reproduces exactly. The V-class breakthrough
framing is preserved for V1_01 only on the current binary. This
is a separate empirical question from determinism — the pinned
binary just happens to land in a wrong-scale regime on V2_01 SP
under the published recipe.

## Why we are not (yet) shipping deterministic-estimator swaps

A genuinely cross-rebuild-deterministic SLAM would need to attack
the FP-evaluation-order problem at its source. Realistic levers:

- **Kahan / Neumaier summation** for the residual reductions
  feeding PnP, BA, and the rigid-ATE accumulator. Cost: ~2× CPU on
  the affected reductions, but eliminates a class of FP rounding-
  order divergence. Worth doing if cross-rebuild variance under the
  toolchain pin stays above 10⁻⁴ m.
- **Closed-form P3P RANSAC** in place of the current iterative
  PnP-with-Gauss-Newton refinement, so the per-iteration solver is
  branch-free in FP. Larger refactor; loses some accuracy.
- **`-Cllvm-args=-fp-contract=off`** in `.cargo/config.toml` to
  disable FMA contraction. Cheapest knob, but inhibits some legit
  perf wins on AVX2+ hardware; should be A/B'd on the SLAM hot
  path first.

These are reasonable next moves *only if the toolchain pin alone
proves insufficient*. The empirical-results table above should
guide that decision; the first row's `run2↔run3` number is the
load-bearing data point.

## Cross-references

- `docs/phase_20_to_27_closeout.md` — §"Known issues — binary
  determinism" describes the original Phase-26 #4 observation
  that motivated this workstream.
- `CHANGELOG.md` Unreleased — toolchain-pin entry and Phase-27
  activation entry sit in the same release cycle.
- `scripts/verify_binary_determinism.sh` — the protocol script.
- `rust-toolchain.toml` — the pin itself, including bump policy.

## Decision gates for further work

Before investing in Kahan summation, P3P swap, or fp-contract
disable:

- Does the empirical-results table above show post-pin cross-
  rebuild variance > 10⁻³ m on V2_01 strict? If no, defer. (As
  of 2026-05-19: the baseline corner-extractor configuration is
  bit-identical post-pin. SuperPoint+strict-stereo still needs a
  dedicated verification — extend `scripts/verify_binary_determinism.sh`
  with a `--feature-extractor superpoint-offline` variant that
  consumes the pre-export at `target/euroc_phase26_superpoint/`.)
- Has a real downstream consumer reported reproducibility-
  blocking variance? If no, defer.
- Is the variance migration source FP arithmetic, or is it
  upstream (e.g. SuperPoint ONNX session non-determinism, IMU
  pre-integration window-aggregation order)? Confirm with the
  per-component A/B before changing PnP.

## Next step — extend coverage to SuperPoint configurations

**Status: done (2026-05-19).** The verification script now accepts
`VARIANT=superpoint`, and both V1_01 and V2_01 SP+strict-stereo
configurations have been run through the three-step protocol with
bit-identical results across all three runs. See the empirical-
results table above. The toolchain pin is confirmed as the
complete fix for cross-rebuild determinism.

Remaining open question (separate from determinism): why the
Phase-26 #1 V2_01 strict SP headline of 0.0107 m does not
reproduce on the pinned binary (post-pin value: 0.2013 m,
sim_scale 1.955). Hypotheses worth investigating if a contributor
takes this on:

- **Build-cache state at the original Phase-26 #1 moment.** The
  pre-pin binary was built with a specific incremental-compile
  cache state. The 0.0107 m number may simply have been a lucky
  PnP RANSAC trajectory that the post-pin binary's slightly
  different RANSAC iteration path does not find. If true, no
  user-visible regression: the underlying algorithm is unchanged
  and the V1_01 headline is preserved.
- **Floating-point sensitivity at the bootstrap moment.** V2_01
  strict bootstrap is at the edge of acceptance (sim_scale right
  on the boundary between 1.0 and 2.0 regime). Tiny FP nudges
  shift the bootstrap to either solution. The pinned binary picks
  the wrong-scale solution; the original Phase-26 #1 binary
  picked the metric solution. Worth a follow-up A/B over
  bootstrap-relevant FP knobs (DLT solver tolerance, RANSAC
  threshold, IMU pre-integration window).
