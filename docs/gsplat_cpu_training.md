# 3DGS training on a CPU (lavapipe)

`gsplat_train` runs its wgpu trainer on any Vulkan device. Without a GPU,
that device is Mesa's lavapipe (llvmpipe): the compute shaders are compiled
for the CPU. This page measures where a training step goes there, which
settings help, and what moved the rasterizer and its backward pass from
compute shaders to rayon loops on the host. A 7,000-step run went from
**2 hours 11 minutes to 46 minutes** at the same PSNR.

## Setup

- **Data.** TUM RGB-D fr2/pioneer_360, pioneer_slam and pioneer_slam3, as
  in the *Real to sim* recipe of `docs/navigation.md` (not on main yet):
  every 5th frame undistorted, `gsplat_photos --steps 0 --cpu` (1,336 of 1,339
  registered, 61,238 points), `align_sfm_tum.py` (camera centres 3.7 cm
  RMSE from the motion capture), 20 px cropped: 600 x 440, 1,169 training
  and 167 held-out views (every 8th).
- **Training.** `gsplat_train --max-gaussians 400000`, default Inria
  schedule, seed 42.
- **Machine.** 4 vCPUs (Xeon, 2.1 GHz, AVX-512), 15 GB, lavapipe
  (LLVM 20.1.2, 256 bits), Mesa 25.2.
- **PSNR** is `gsplat_eval` on the 167 held-out views at 600 x 440.

## Where a step goes

`GSPLAT_TRAIN_PROFILE=1 GSPLAT_PROFILE=1`, 200 steps from the SfM points
(61k gaussians, 11k visible and 308k tile intersections per frame). Every
stage waits for the device, so the stages add up. ms per step:

| Stage | Shaders | Host rasterizer |
| --- | ---: | ---: |
| Projection (project, compaction counts, scan, map, tile offsets) | 6.6 | 6.3 |
| Sorting (depth 6.9, tile 32.8) | 39.7 | 34.8 |
| Rasterize | 160.4 | **72.7** |
| Loss: L1 + SSIM | 62.6 | 55.9 |
| Backward: rasterize | **994.5** | **113.0** |
| Backward: projection | 4.7 | 4.0 |
| Adam, densify statistics | 9.2 | 7.1 |
| **Step** | **~1,300** | **~300** |

Densification runs every 100 steps from step 500 on. Profiled from the
step-1,000 scene (140-160k gaussians), it takes ~570 ms each time, ~6 ms
a step.

**The backward rasterizer was 77% of a step.** lavapipe runs a workgroup on
one CPU thread as 8-wide vectors. `rasterize_backward.wgsl` sums each
splat's gradient over a tile's 256 pixels with subgroup reductions, then
ten compare-and-swap float adds per subgroup into shared memory. On a GPU
that is cheap; on lavapipe it is a scalar loop of atomics per splat and
per subgroup. The forward rasterizer has no atomics, but it stops at a
workgroup barrier every 256 splats.

## Settings

Each setting trained 2,000 steps (new code). Step times are 100 steps
from the 140k-gaussian scene of step 1,000, densification off, ±10 ms:

| Setting | ms/step, shaders | ms/step, host | 2,000 steps, host | PSNR (dB) | Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default (600 x 440, SSIM every step, 4 threads) | 860 | 260 | 600 s | 21.60, 21.71 | 358k |
| Half resolution (300 x 220), scored at 600 x 440 | 350 | 110 | 305 s | 21.44 | 391k |
| SSIM every 4th step (`--ssim-every 4`) | 850 | 220 | 507 s | 21.20 | 219k |
| No SSIM (`--ssim-weight 0`) | 900 | 200 | 466 s | 20.93 | 176k |
| `--max-gaussians 200000` | | | 586 s | 22.96 * | 210k |
| `--max-gaussians 100000` | | | 535 s | 23.12 * | 116k |
| `LP_NUM_THREADS` = `RAYON_NUM_THREADS` = 1 | 2,550 | 860 | | same | |
| ... = 2 | 1,480 | 440 | | same | |
| ... = 8 (4 vCPUs) | 740 | 290 | | same | |

\* Not a like-for-like comparison: see below.

- **Resolution** is the strongest setting: a quarter of the pixels gives
  2.4x the speed per step, for 0.2 dB at 2,000 steps and 0.5 dB at 7,000
  (below).
- **SSIM** costs 50-60 ms a step. Computing it less often, or not at all,
  costs 0.4-0.7 dB. Fewer gaussians are densified without it, which is
  part of the speedup.
- **Threads.** The thread count changes only summation order, not the
  result. 4 threads on 4 vCPUs is the right default for the host
  rasterizer. The shaders were faster with 8 lavapipe threads.
- **`--max-gaussians`** saves little time per step: most of a step is per
  pixel, not per gaussian. Its PSNR at 2,000 steps is not comparable,
  because a training step that is a multiple of 100 ends with
  densification. A run still below its cap at its last step is scored right
  after a fresh split and clone. In the 7,000-step run below, the model
  at step 1,999 scores 22.21 dB and at step 2,000 21.74 dB. A run at
  its cap only prunes there (24.663 dB at step 6,999, 24.659 dB at
  7,000). Over 7,000 steps the 200k cap trains in 37 instead of 46 minutes
  and scores 0.23 dB lower (24.43 against 24.66 dB).

## The change

`Renderer::set_cpu_raster` (on by default when the wgpu adapter is a CPU;
`GSPLAT_CPU_RASTER=0` or `=1` overrides) runs the two rasterizers on the
host with rayon (`crates/gsplat-render/src/cpu_raster.rs`). Projection,
both sorts, the loss and Adam stay on the device, unchanged.

- **Backward.** The forward's residuals (final transmittance, last
  blended index), the tile lists and dL/dC are read back in one map
  (~3 ms). Each tile walks its list back to front as the shader does, and
  sums each splat's gradient over its pixels with plain float adds into
  the splat's slot of the intersection list. The slots are then added
  into the per-gaussian records in list order. That order does not depend
  on the thread count; the shader's atomics did. The records go back
  with one `write_buffer`, and `project_backward` runs on the device.
- **Forward.** The same per-tile walk, front to back, writes the image
  and the residuals, which are uploaded for the loss and the backward.
- **Per splat**, only the tile rows that can reach alpha >= 1/255 (the
  conic's level set at ln(255 opacity), one pixel wider) are visited.
  A row is a fixed 16-lane loop with masks instead of branches, and
  `exp` is a polynomial (Cephes `expf`), so LLVM vectorizes it with AVX2.

The backward went from 995 to 113 ms and the forward from 160 to 73 ms.

## Same result

- **Unit tests.** The renderer's tests check the forward image and the
  screen-space and parameter gradients against the f64 CPU reference, on
  several lenses. They now run both rasterizers. Both are within 2e-5 of
  the reference (worst gradient error 4.8e-6 of each component's maximum).
- **The real scene.** `raster_parity` renders 20 training views of the
  358k-gaussian scene of a 2,000-step run with both rasterizers. Forward:
  mean absolute difference 3e-8. Backward, on the same frame: relative L2
  difference of every gradient component 1-3e-6. Running the shader
  twice differs by 0.4-3e-7 (its atomics).
- **Training.** Same seed, same steps:

  | Steps | Shaders | Host rasterizer |
  | ---: | --- | --- |
  | 1,000 (training log) | 20.47, 20.44 dB | 20.32, 20.38 dB |
  | 2,000 | 21.49 dB, 2,079 s | 21.60, 21.71 dB, 600 s |
  | 7,000 | 24.61 dB, 7,858 s | 24.66 dB, 2,768 s |

  Neither is bit-reproducible run to run: `project_forward` assigns
  compact slots with an atomic counter, and a cloned gaussian has exactly
  its parent's depth, so the order of the two after the (stable) depth
  sort varies. Within a run's spread, the two rasterizers give the same
  PSNR.

## 7,000 steps

| Run | Time | PSNR (dB) | SSIM | Gaussians |
| --- | ---: | ---: | ---: | ---: |
| Shaders (`docs/navigation.md`, an earlier SfM of the same frames) | 2.3 h | 25.0 | | 403k |
| Shaders | 2 h 11 min | 24.61 | 0.844 | 404k |
| Host rasterizer | **46 min** | 24.66 | 0.844 | 406k |
| Host rasterizer, half resolution (scored at 600 x 440) | 27 min | 24.20 | 0.836 | 417k |
| Host rasterizer, `--max-gaussians 200000` | 37 min | 24.43 | 0.842 | 215k |

## What is left

A step averages 0.4 s over the 7,000-step run (400k gaussians from step
2,500). In the 200-step profile, the largest stages left beside the two
host rasterizers (~190 ms) are SSIM (~50 ms) and the tile sort (~30 ms). The tile sort
could move to the host as well, and the forward could keep its tile lists
for the backward instead of reading them back.
