# A pure-CPU trainer, and 3DGS²'s local Newton step on it

`crates/gsplat-cpu` (`visloc-gsplat-cpu`) trains a 3DGS scene with no GPU,
wgpu or lavapipe: projection, tile binning, both rasterizer passes and the
optimizer run on the host (rayon over 16x16 tiles, 8-lane rows that
vectorize, as `cpu_raster.rs`). It has two optimizers on the same
rasterizer:

- **Adam**, the Inria recipe, one view per step;
- **a per-gaussian local Gauss-Newton step** after 3DGS² (Lan et al., arXiv
  2501.13975), the "bet" of the training-speed survey: the backward pass also
  sums each splat's Gauss-Newton matrix in screen space, lifted through the
  projection Jacobian (forward-mode duals) to its 11 geometry parameters;
  each gaussian solves its own damped 11x11 system, all at once.

```text
cargo run --release -p visloc-gsplat-cpu --example gsplat_cpu_train -- \
    --data <colmap root> --optimizer adam|newton --steps N [--loss l2|l1] \
    [--overlap 1] [--decay 0.9] [--neighbours 3] [--groups mean,shape,opacity,color]
```

With Adam its defaults are `gsplat_train`'s (L1 + 0.2 D-SSIM, Inria
densification, SH degree up every 1000 steps); the Newton step runs on a
fixed population (`--loss l2 --no-densify`).

## Correctness

`tests/oracle.rs`: the image and every parameter gradient against the f64
oracle (`visloc_gsplat_core::backward`); the Gauss-Newton matrices against
fourth-order finite differences of the oracle's image (step 3e-2: the
`alpha >= 1/255` edge is a jump, which inflates smaller steps' squares by
~1/eps).

## Setup

Mip-NeRF 360 bonsai at `images_8` (390x260, 255 train / 37 eval views,
every 8th held out), 206,613 SfM points, black background. 4 vCPUs (Xeon
2.1 GHz, AVX-512), 15 GB. Eval PSNR over the 37 held-out views.

## Adam: the same as `gsplat_train`, without lavapipe

L1 loss, no densification, 500 steps, against `gsplat_train --no-densify
--ssim-weight 0` (lavapipe, host rasterizer):

| Steps | `gsplat_train` | `gsplat-cpu` Adam |
| ---: | ---: | ---: |
| 100 | 18.30 dB | 18.34 dB |
| 200 | 20.59 dB | 20.56 dB |
| 300 | 21.60 dB | 21.62 dB |
| 500 | 23.02 dB (97 s) | 23.02 dB (88 s, 176 ms/step) |

## Adam, 7,000 steps with densification: 1.41x faster than `gsplat_train`

Both with their defaults and `--max-gaussians 400000`, run one after the
other on the idle machine:

| Trainer | Eval PSNR at 1k / 2k / 4k / 5k / 7k | Gaussians | Training time |
| --- | --- | ---: | ---: |
| `gsplat_train` (lavapipe + host rasterizer) | 24.81 / 28.62 / 29.63 / 30.30 / **29.89** | 418,731 | 1,578 s |
| `gsplat-cpu` | 24.96 / 28.68 / 29.51 / 30.06 / **29.87** | 416,642 | **1,119 s** |

(Steps 3,000 and 6,000 end with an opacity reset, so their evals, ~10 dB in
both, are left out.) A step at 400k gaussians, ms: Adam 43.6, backward
51.0, forward 27.0, projection and binning 26.1, loss 5.2.

What it took besides the host pipeline:

- **The projection's reverse pass by hand** (`project::screen_vjp`, tested
  against the dual Jacobian), fused with the Adam update per gaussian.
- **10 floats per intersection** in the gradient-only backward walk (the
  Gauss-Newton walk needs 32).
- **No prune at the cap**, as `gsplat_train`'s classify pass: pruning past
  the cap (large splats once `prune_large` is on) cost 3 dB by step 5,000.
- **Subnormals flushed** in the Adam moments: gaussians culled after an
  opacity reset get no gradient, their first moment decays by 0.9 a step
  into subnormals within ~650 steps, and x86 computes those on a slow path
  (the Adam stage went from 44 to 120 ms a step; lavapipe flushes them).

## The local Newton step

Each iteration renders the primary view at full resolution and its 3
nearest training views at half resolution, as 3DGS² does. ms per
iteration ~800-900 against Adam's ~180: 4.8x.

| Variant | Eval PSNR by iteration | |
| --- | --- | --- |
| Adam (L2), per step | 17.88 dB at 100 steps (18.7 s) | |
| 3DGS² as described (plain GN, step 1, overlap 0) | 15.9 (10), 16.7 (20), 17.0 (40), 16.0 (50) | rises, then falls |
| + overlap majorizer (`--overlap 1`) | 15.9 (10), 16.9 (30), 17.0 (40), 16.0 (50), 16.6 (70) | same |
| + incremental sums (`--decay 0.9`, `g += H dx`) | 15.1 (10), 16.3 (30), 17.5 (50), **17.8 (60, 53 s)** | rises steadily |
| incremental, `--overlap 0.5` or `0` | 7.0 / 6.7 (10) | diverges |

What each change addresses:

- **Simultaneous steps overshoot.** Every gaussian's Newton step assumes the
  others fixed; splats that overlap all correct the same residual, and the
  image moves by about their number times too much. `--overlap 1` weighs
  each pixel's term by `W / (alpha T)` (`W = 1 - T_final`), so the
  per-gaussian matrices majorize the joint one (Cauchy-Schwarz) and the
  quadratic model descends with every gaussian stepping at once.
- **One view's step undoes the last.** A full step on one view (plus three
  neighbours) moves each gaussian to that view's optimum; colour alone
  (`--groups color`) stalls at ~12 dB with a rising train loss from the
  start. `--decay 0.9` keeps the earlier views' sums, decayed, and moves
  their gradient with the parameters (`g += H dx`), so each step solves the
  sum of the recent views' quadratic models.

## Verdict so far

The best stable variant reaches in 60 iterations (53 s) what Adam reaches
in ~100 steps (18.7 s): **1.7x fewer iterations at 4.8x the cost, 2.8x
slower in wall time.** 3DGS²'s "10x fewer iterations" did not appear: as
published (step 1, no line search, 3 neighbours) the step is unstable here,
and the changes that stabilize it cost most of its per-iteration gain.

Where an iteration goes (incremental variant, ms): backward 534 (the 21
extra sums per splat and pixel), lift 140 (duals for the Jacobian, 11x11
products), forward 97, projection 85.

Open, if the bet is pursued further:

- A cheaper majorizer weight than `W / (alpha T)`, which over-damps splats
  of low weight (it made the opacity step all but vanish).
- The paper's reduced parameterizations (2-DOF position across the ray,
  one rotation angle, 2D scales) instead of damping a rank-deficient 11x11.
- Fewer pixels per Gauss-Newton view (strided, as LM-RS samples 32 per
  tile), which would cut the 534 ms backward.
