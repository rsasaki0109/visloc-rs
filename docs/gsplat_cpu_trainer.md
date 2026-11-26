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

The techniques tried here come from a survey of 2024-2026 fast-training
papers ranked for a CPU trainer
([`docs/research/cpu_3dgs_training_survey_ja.md`](research/cpu_3dgs_training_survey_ja.md),
in Japanese, with its notes).

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

## brush's refine strategy (`--strategy brush`)

`gsplat_train --strategy brush` (and `gsplat_photos`'s training) refine
with brush 0.3's strategy instead of Inria's rule. `--strategy brush` ports
it, after `Trainer::brush_refine_now` and the `mean_noise`, `adam` (aux
losses) and `densify_scatter` (action 4) shaders:

- **The statistic**: per view, each splat's refine weight
  `sum_px |dL/du| w + |dL/dv| h` (the backward walk's tenth accumulator, on
  in this mode only), the largest over the views that blended it.
- **Every 200 steps**: prune the transparent (< 2/255), tiny (a log-scale
  below -15) and far-off (10x the central 75% box) gaussians; replace them
  with live ones sampled by opacity; grow by 10% of those whose weight per
  view passes 4e-5, sampled by weight (both without replacement). A sampled
  gaussian becomes two at `mean -+ R (N(0, 1) s / 2)`, scales / sqrt 2,
  opacity `1 - sqrt(1 - a)`; the first keeps its Adam moments.
- **brush's rates** (mean 2e-5 -> 1e-6 times the bound size, log-scale
  1e-2 -> 6e-3, opacity 1e-2, SH DC 2e-3), **aux losses** (opacity 1e-9,
  scale 1e-8, faded out by 90% of the run; they reach unseen gaussians, so
  Adam is dense in this mode) and **mean noise** on nearly transparent
  gaussians (`(1 - a)^100` times the mean rate times 40).

bonsai `images_8`, 7,000 steps, cap 400k:

| Trainer | Eval PSNR at 1k / 3k / 5k / 7k | Training time |
| --- | --- | ---: |
| `gsplat_train --strategy brush` | 25.28 / 28.51 / 29.82 / **31.40** | 1,775 s |
| `gsplat-cpu --strategy brush` | 25.37 / 28.48 / 29.92 / **31.04** | **1,425 s (1.25x)** |
| `gsplat-cpu --strategy brush --seed 7` | 25.35 / 28.34 / 29.44 / **31.10** | 1,479 s |

- **The populations agree** (228,789 against 228,481 at step 1,000; both
  at the cap by 6,000), and so do the curves up to step 5,000.
- **0.3 dB lower at 7,000, in both seeds.** `gsplat_train` has no seed
  flag, so its spread is unknown. Every piece above reads the same as
  `gsplat_train`'s on inspection, and the cause is not found yet.
- **A step costs more than with Inria's rule** (204 against 151 ms): Adam
  is dense and carries the aux losses and the noise (54 ms a step).
- **brush trains 1.1 dB higher than Inria's rule here** (31.04 against
  29.95 dB at 7,000 steps), at 1.35x its time.

## Speedups from the survey, measured

Same protocol (bonsai `images_8`, 7,000 steps, cap 400k):

| Change | Eval PSNR at 7k | Training time | Adopted |
| --- | ---: | ---: | --- |
| None (dense Adam) | 29.87 dB | 1,119 s | |
| Deferred sparse Adam (GS-Scale: only projected gaussians are updated, the others caught up in closed form when next projected, and before evaluating, saving and densifying) | 29.95 dB | 1,054 s (-6%) | yes, the default (`--dense-adam` turns it off) |
| + 2x2 strided pixels, cycling offset, Turbo-GS's low-pass (`--stride 2`) | 24.52 dB at 2k (28.72 without) | -10% a step | no |
| + exact ellipse-tile test (AccuTile) | | binning 21 -> 30 ms | no |

- **Sparse Adam** moves the Adam stage from 44 to 32 ms a step: a view of
  bonsai projects about two thirds of the gaussians, so most are updated
  anyway.
- **Strided pixels** lose 4 dB by step 2,000 for 10%: at 390x260 the step
  is not bound by pixels but by tile intersections (293k a view; the
  stride leaves 215k, since most splats are a few pixels across), and each
  costs ~0.7 µs in the backward walk.
- **The exact tile test** removes 2% of the intersections (293k -> 287k):
  the splats are small, so their boxes waste little, and the test costs
  more than it saves.

## At 780x520 (bonsai `images_4`)

3,000 steps, the same settings otherwise (cap 400k), one run each:

| Trainer | Eval PSNR at 1k / 2k / 3k | Training time | ms/step |
| --- | --- | ---: | ---: |
| `gsplat_train` | 24.62 / 28.14 / **28.86** | 972 s | 324 |
| `gsplat-cpu` | 24.47 / 28.10 / **28.93** | **706 s (1.38x)** | 236 |
| `gsplat-cpu --stride 2` | 24.13 / 26.37 / 26.69 | 509 s | 170 |

- **Four times the pixels cost 1.6x the time** (236 against 150 ms a step at
  390x260): intersections went from 293k to 564k a view. Backward 95 ms,
  forward 50, Adam 34, projection and binning 31, loss 18.
- **Strided pixels still lose at equal time**: 26.69 dB at 509 s, where the
  full-pixel run had 28.10 dB at 458 s (step 2,000).

### Coarse to fine (`--res-schedule`)

DashGaussian's idea with fixed boundaries: the images box-averaged 4x for
the first part of the run, 2x for the next, then full. Same protocol (780x520,
3,000 steps):

| Schedule | Eval PSNR at 1k / 2k / 3k | Training time |
| --- | --- | ---: |
| full resolution throughout | 24.47 / 28.10 / **28.93** | 706 s |
| `--res-schedule 0.1,0.3` | 24.63 / 28.07 / **28.94** | 618 s (-12%) |
| `--res-schedule 0.2,0.5` | 23.78 / 28.01 / **28.92** | **597 s (-15%)** |
| `--res-schedule 0.3,0.7` | 23.52 / 25.36 / 28.78 | 536 s (-24%) |

- **0.2,0.5 keeps the PSNR for 15% less time**: 1.63x faster than
  `gsplat_train` (972 s, 28.86 dB) at this resolution. It stays opt-in: at
  390x260 a 4x reduction leaves 97x65 images, and it has not been measured
  there or on other scenes.
- **0.3,0.7 is 0.15 dB lower**: full resolution for the last 30% of the
  run is too little.

The same schedule on two more scenes (3,000 steps, cap 400k):

| Scene | Full resolution | `--res-schedule 0.2,0.5` |
| --- | --- | --- |
| garden `images_8` (648x420, outdoors) | 26.43 dB, 521 s | 26.15 dB, 432 s (-17%) |
| bonsai `images_8` (390x260) | 29.47 dB, 511 s | 29.16 dB, 459 s (-10%) |

- **0.3 dB lower on both.** Only bonsai at 780x520 kept its PSNR, so the
  schedule stays opt-in: a trade of ~0.3 dB for 10-17%, not a free gain.

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
