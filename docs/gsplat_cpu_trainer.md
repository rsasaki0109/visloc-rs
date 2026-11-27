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
| `gsplat_train --strategy brush --seed 7` | 25.41 / 28.47 / 29.70 / **31.43** | 1,709 s |
| `gsplat-cpu --strategy brush` | 24.91 / 28.71 / 30.04 / **31.39** | **1,168 s (1.52x)** |

- **Times are after the tight raster bounds** (below). Before them the same
  run, bit-identical, took 1,416 s.
- **garden holds up too** (`images_8`, 648x420, outdoors, same settings):
  `gsplat_train` 23.80 / 26.16 / 27.46 / **27.75** dB in 1,983 s,
  `gsplat-cpu` 23.77 / 25.92 / 27.47 / **27.82** dB in **1,568 s (1.26x)**.
  The populations track each other (174,603 against 174,685 at step 1,000,
  303,082 against 303,120 at 3,000, both at the cap by 5,000).
- **A lower cap is not a speed lever.** Same run with `--max-gaussians
  300000`: 31.12 dB (-0.27) in 1,196 s. Through step 3,000 it was identical
  to the 400k run, yet 9% slower there, so this run's box was slower and the
  cap bought no measurable time. With 200,000 the cap sits below the initial
  206,613 points and nothing grows: 29.57 dB (-1.8) in 985 s (-16%). The
  time is mostly per-pixel blending, and fewer, larger splats cover about
  as many pixels.
- **The populations agree** (228,751 against 228,481 at step 1,000; both
  at the cap by 6,000), and so do the curves.
- **Same quality, within 0.05 dB** of `gsplat_train`'s two seeds (31.40 /
  31.43 dB, `--seed` added for this).
- **A 0.35 dB gap was a dense-Adam bug.** Before the fix `gsplat-cpu`
  landed at 31.04 / 31.10 dB. The dense Adam path never advanced each
  gaussian's last-updated step, so every `sync` (each refine, every 200
  steps, and each evaluation) replayed the sparse path's closed-form
  catch-up over all the steps since the start or the last refine: about
  ten phantom momentum steps on every gaussian, and m wiped to zero.
  Found by elimination: turning off the mean noise (31.01 against 31.45 dB)
  or the aux losses left the gap; a fixed population with no refine
  matched (29.39 against 29.42 dB); refine at a cap equal to the initial
  count, so only prune and replace, opened it again (29.29 against 29.54).
- **A step costs more than with Inria's rule** (204 against 151 ms): Adam
  is dense and carries the aux losses and the noise (54 ms a step).
- **brush trains 1.4 dB higher than Inria's rule here** (31.39 against
  29.95 dB at 7,000 steps), at 1.34x its time.

## Speedups from the survey, measured

Same protocol (bonsai `images_8`, 7,000 steps, cap 400k):

| Change | Eval PSNR at 7k | Training time | Adopted |
| --- | ---: | ---: | --- |
| None (dense Adam; measured with the bug above, so its every densify and evaluation added phantom steps) | 29.87 dB | 1,119 s | |
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
- **Re-measured after the tight raster bounds**, back to back on one machine:
  `gsplat_train` 999 s (28.86 dB), `gsplat-cpu` before the bounds 774 s,
  after 765 s (both 29.03 dB), so 1.31x. At this resolution the bounds buy
  only ~1% (forward and backward -6%): splats cover more pixels, so the
  one-pixel pad was a smaller share of the walk than at 390x260.
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

## Tight raster bounds

A splat's rows and 8-lane half rows were padded by a full pixel. On bonsai
`images_8` (brush), 28% of the half rows the rasterizer walked had no pixel
at or above `alpha = 1/255`. With the bounds on the exact level, computed in
f64 and widened only for the f32 `sigma` and polynomial `exp` of the pixel
loops, the result is bit-identical. Time over 7,000 steps:

| Strategy | Before | After |
| --- | ---: | ---: |
| brush | 1,413 s | 1,184 s (-16%) |
| Inria | 1,156 s | 1,043 s (-10%) |

The bounds must be conservative. An f32 version was 20% faster on Inria,
but on brush it cut off one real hit of a large splat ~300 px off the tile,
where the bound quadratic's terms nearly cancel, and the run drifted from
step 3,000. A unit test draws random splats, far ones included, and checks
that every hit stays inside the bounds. A regression case for that splat
fails on the f32 bounds.

## From photos, with no GPU

This repository's SfM can make the model the trainer starts from, all on
the CPU. Every 2nd photo of bonsai `images_4` (146 images, 780x520), with a
rough focal length that SfM refines:

```bash
cargo build --release --example unordered_sfm_demo --features image-io
unordered_sfm_demo --feature-extractor sift --images-dir images \
  --image-suffix .JPG --width 780 --height 520 \
  --fx 936 --fy 936 --cx 390 --cy 260 --refine-intrinsics \
  --sift-max-keypoints 4096 --retrieval-topk 12 --min-matches 30 \
  --match-ratio 0.8 --verification-mode full --mapper incremental \
  --next-image-policy auto --post-refinement-registration \
  --final-iterative-refinement --out-colmap sfm
ln -s sfm sparse   # gsplat_cpu_train reads images/ and sparse/
gsplat_cpu_train --data . --strategy brush --steps 7000 --max-gaussians 400000
```

SfM takes 322 s on 4 threads: 146 / 146 images registered, 72,288 points,
mean reprojection error 0.337 px, `fx` 936 -> 810.2 (COLMAP's: 806.2).

The SfM writes every point white. `gsplat_cpu_train` now colours such a
model's points itself: the per-channel median of the pixels each point
projects to in the training photos. Without it the start is a white image
(eval PSNR 3.2 dB at step 0, 23.67 dB at step 7,000); with it, 13.8 dB at
step 0.

Eval PSNR at step 7,000, the same photos and train / eval split:

| Poses and points | Points | PSNR |
| --- | ---: | ---: |
| this repo's SfM | 72,288 | 25.55 dB |
| the dataset's COLMAP model | 206,613 | 30.03 dB |

The run works end to end, but 4.5 dB short of COLMAP's poses. The SfM
model starts from a third of the points, and its principal point moved
from 260 to 252.3 in `cy`. Which of the two costs more is not measured yet.
