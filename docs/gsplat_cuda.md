# CUDA 3DGS rasterizer and trainer (`visloc-gsplat-cuda`)

An opt-in CUDA backend for 3D Gaussian Splatting, next to the wgpu renderer
in `visloc-gsplat-render` and the wgpu trainer in `visloc-gsplat-train`. It
exists for two reasons:

- **Machines where wgpu cannot reach the GPU.** Under WSL2 the NVIDIA driver
  exposes CUDA but no Vulkan driver, so wgpu falls back to the CPU's
  lavapipe. Windows Smart App Control blocks locally built executables. On
  such a laptop, CUDA under WSL2 is the only way to the GPU.
- **A reference for CUDA-level optimizations.** Write-ups such as George
  Kopanas's [five CUDA optimizations for 3DGS](https://grgkopanas.github.io/3dgs_cuda_blog/)
  rely on warp intrinsics, float atomics and Nsight Compute. They can be
  measured here first and ported to WGSL where they pay off.

The packaging follows the DPVO CUDA kernels:

- `native/gsplat_cuda/gsplat_cuda.cu` is built into a shared library by
  `scripts/build_gsplat_cuda.sh`.
- The crate loads it at runtime with `libloading` after checking its ABI
  version (currently 3).
- Cargo never compiles CUDA, and nothing in the default build depends on the
  crate. It is `publish = false`.

## Status

| Part | State |
|---|---|
| Rasterizer forward (projection, SH colour, tile binning, sort, compositing) | Done |
| Backward to every gaussian parameter (mean, quaternion, log-scale, opacity logit, SH) | Done |
| Training step: L1 + D-SSIM, Adam with brush's aux losses, refine statistics, mean noise; opt-in sparse SH Adam | Done |
| Trainer loop (`trainer::CudaTrainer`, example `gsplat_train_cuda`): brush's refine strategy and learning rates, or a fixed population | Done |
| Inria densification rule, per-image appearance, depth-normal loss | Not supported (the trainer refuses these configurations) |
| Geometry channels (normal, depth) for the normal loss and the mesh | Not started |

The maths is the wgpu renderer's (`compute_projected`, `tile_row_span`,
`rasterize`, `rasterize_backward`, `project_backward`), so both backends and
the CPU f64 oracle (`visloc_gsplat_core::backward`) agree. Two things are
organised differently:

- Per-gaussian state stays indexed by gaussian id (no visible-set compaction).
- Tile-then-depth order comes from one 64-bit CUB radix sort of
  `tile << 32 | depth bits`, stable in gaussian order (the oracle's tie
  order).

The training step is the wgpu trainer's: the `loss_l1`, `ssim`,
`densify_stats`, `adam`, `mean_noise` and `densify_scatter` shaders, with
the same hashes for the random draws.

- **On the device**: the scene, its Adam moments and the training images
  (packed RGBA8) stay there.
- **One step**: one `train_step` call renders, computes the loss and its
  gradient, runs the backward pass, accumulates refine statistics, takes
  the Adam step and adds mean noise.
- **Refine**: split as in the wgpu trainer. The host reads the statistics
  and picks an action per gaussian; the device scatters the parameters and
  moments.
- **Shared code**: the schedule (view order, learning-rate decay, SH degree,
  aux-loss fade) and the refine decisions are copies of `Trainer`'s. The
  configuration, image packing and bounds come from
  `visloc_gsplat_train::config`, which needs no GPU feature.

## Build and test

Under Linux or WSL2 with a CUDA toolkit (12.8 or newer for the newest GPUs):

```bash
scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda            # ARCH defaults to `native`
export VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so
cargo test --release -p visloc-gsplat-cuda               # skipped when the variable is unset
cargo run --release -p visloc-gsplat-cuda --example gsplat_cuda_bench -- \
    --scene docs/euroc_splat/euroc_v101.splat --width 752 --height 480 --iters 30
cargo run --release -p visloc-gsplat-cuda --example gsplat_train_cuda -- \
    --data <colmap_root> --steps 30000 --eval-every 5000 --max-gaussians 2500000 \
    --out scene.ply [--profile]
```

On Ubuntu 26.04, CUDA 13.1's headers clash with glibc's C23 `rsqrt`
declaration ("exception specification is incompatible"). CUDA 13.3 compiles
cleanly, and its binaries run on a driver that reports CUDA 13.1 (minor-version
compatibility).

## Against the CPU oracle

`tests/oracle.rs` uses the same scenes and tolerances as the wgpu renderer's
tests. The worst error is shown relative to each component's largest
magnitude. The tolerances are 2e-3 (image, screen gradients) and 3e-3
(parameters).

| Test | Image | Screen gradients | Parameter gradients |
|---|---|---|---|
| 3 gaussians, 40x32 | | 2.1e-5 | 4.2e-6 |
| 64 gaussians, 96x80 | 4.9e-7 (absolute) | 3.2e-6 | 1.5e-6 |
| 2,500 faint gaussians, 64x48 (about 500 per tile: several batches) | 4.5e-7 (absolute) | 6.7e-6 | 1.1e-6 |
| Active SH degree 1 and 0 of a degree-3 scene | | | 1.8e-6; bands above the active degree get exactly 0 |

`tests/training.rs` checks the training step:

| Test | Result |
|---|---|
| Loss gradient against the CPU L1 and `metrics::ssim_with_grad` (weight 0.2) | 4.5e-5 of its largest value; L1 alone 2.0e-8 |
| Masked pixels (ground-truth alpha below 128) | Exactly zero gradient |
| Refine scatter (prune, keep, clone, brush split) | Row order, split means symmetric about the parent, scales / sqrt 2, opacity 1 - sqrt(1 - a), statistics reset |
| 300 steps on a scene rendered from known gaussians, colours and opacities perturbed, dense and sparse SH Adam | Mean L1 0.0175 → 0.0009 in both; a gaussian behind every camera keeps its SH bit for bit |

## Training on south-building

The dataset is south-building's 128 raw JPGs through `gsplat_photos
--steps 0 --cpu --cpu-sift --max-size 1024`:

- All 128 images registered, with 101,450 points.
- The focal length was refined to 850.2 px (COLMAP: 847.2).
- The run took 945 s on the CPU.

Then `gsplat_train_cuda --steps 30000 --eval-every 5000 --max-gaussians
2500000`, with brush's preset, every 8th view held out (112 train / 16
eval), 1024x768. The first run, on the unoptimized rasterizer:

| Step | Eval PSNR | Eval SSIM | Gaussians | Training time (evaluations left out) |
|---|---|---|---|---|
| 5,000 | 22.37 | 0.738 | 304k | 76 s |
| 10,000 | 22.87 | 0.787 | 750k | 202 s |
| 15,000 | **23.30** | 0.809 | 1.35M | 387 s |
| 20,000 | 23.09 | 0.814 | 1.35M | 600 s |
| 25,000 | 22.96 | 0.814 | 1.35M | 805 s |
| 30,000 | 22.99 | 0.815 | 1.35M | **978 s** |

- **Peak and overfitting.** Eval PSNR peaks when growth stops (15k) and
  then drifts down while the training L1 keeps falling (0.033 → 0.026), so
  the model overfits the 112 views.
- **Against the wgpu run.** The README's wgpu run of the same photos
  reached 22.77 dB at 30k. That run used GPU SIFT for the SfM and
  `--normal-weight 0.005`, which the photos page puts at about -0.2 dB, so
  the two runs agree within their differences.
- **Where the time went.** At 1.35M gaussians a step took 28.9 GPU ms. The
  largest stages were raster_bwd (9.6 ms), the optimizer (8.7 ms) and
  raster (3.8 ms), which the next section works on.

## Optimizations

Each change was checked against the oracle and training tests and timed
with `gsplat_cuda_bench --data`. That bench runs training steps with zero
learning rates on the first run's trained scene (1.35M gaussians), from all
128 cameras of the dataset. These are the kernels and gradients of
training, and back-to-back runs agree to a few percent. The figures are GPU
ms per step, the median over 128 steps, then the range over 3 or 4 bench
runs:

| Change | raster | raster_bwd | project_bwd | optimizer | Total |
|---|---|---|---|---|---|
| Baseline (rasterizer of the first run) | 4.3–4.9 | 11.0–11.8 | 2.8–2.9 | 8.1 | 30.3–31.6 |
| Sub-tile masks, `project_bwd` gating | 2.6–2.8 | 7.9–8.7 | 1.1–1.2 | 6.7 | 22.7–23.7 |
| Gradients added to global memory directly | 2.4–2.7 | 6.5–7.3 | 1.05–1.07 | 6.7 | 20.7–21.3 |
| `--sparse-sh-adam` (opt-in) | 2.5–2.7 | 6.8–7.2 | 1.09–1.16 | 3.4–3.5 | 17.8–18.5 |

- **Sub-tile masks.** The duplicate pass records, for each (tile, splat)
  entry, which 8x4 sub-tiles of the 16x16 tile the footprint reaches: 8
  bits in the top byte of the sorted value, so the sort carries them.
  - **Threads.** Each warp of a tile block covers one 8x4 sub-tile,
    instead of two 16-pixel rows. Forward and backward skip the splats
    whose bit for their warp is clear, without evaluating them.
  - **Cost.** The bits come from the footprint's columns over 4-row bands.
    Each tile row's bands are computed once, which keeps duplicate at
    0.8 ms. Computed per tile, it took 4.8 ms.
  - **Limit.** Gaussian ids are limited to 24 bits (16.7M). `set_scene` and
    refine refuse more.
  - **Rejected first try.** A forward pass that voted per warp for the
    contributor mask made raster 37% slower and was dropped.
- **project_bwd gating.** Gaussians that blended into no pixel this frame
  (all screen gradients zero) are skipped. Their outputs are zero anyway.
- **Gradients added directly.** A float `atomicAdd` on shared memory
  compiles to a compare-and-swap loop on this GPU (`ATOMS.CAST.SPIN` in the
  SASS). The global one is a native reduction (`REDG.E.ADD.F32`). Each warp
  now adds its butterfly-reduced sums straight to the gaussian's record,
  with no shared accumulator and no flush.
- **Sparse SH Adam (opt-in, `--sparse-sh-adam`).** The SH group, 48 of the
  59 floats per gaussian, is updated only for the gaussians blended this
  step. Their SH gradient rows need no zeroing either.
  - **What stays dense.** The transform and opacity groups stay dense, so
    brush's aux losses still fade the gaussians no view sees.
  - **First try.** Using "projected" instead of "blended" as the criterion
    saved only 1.6 ms of the optimizer and cost 0.8 ms in `project_bwd`.

The forward image is bit-identical through all of these. The gradients
change only in summation order.

South-building trained again with the optimized kernels:

| Run | Adam | PSNR at 15k | PSNR at 30k | Gaussians | Training time |
|---|---|---|---|---|---|
| First run (baseline kernels) | dense | 23.30 | 22.99 | 1.35M | 978 s |
| Sub-tile masks, shared-memory accumulation | dense | 23.04 | 22.71 | 1.33M | 674 s |
| Final kernels | dense | **23.32** | **23.22** | 1.41M | **691 s** |
| Final kernels | sparse SH | 23.18 | 22.87 | 1.41M | 659 s |

- **Run-to-run spread.** Float atomics make every run diverge, and the
  three dense runs end 0.5 dB apart (22.71–23.22 dB). The final kernels
  train 29% faster than the first run without losing quality.
- **Sparse SH Adam.** It falls inside that spread at 0.13–0.35 dB below the
  dense run of the same kernels. In wall-clock time it saved only 5%
  (659 s against 691 s), against 18% in the bench, because GPU clocks vary
  between runs on this laptop. It stays opt-in.
- **Final breakdown.** At steps 29,001 to 30,000 of the final dense run
  (1.41M gaussians), a step takes 23.0 GPU ms:

| Stage | ms/step |
|---|---|
| project | 1.05 |
| scan | 0.26 |
| duplicate | 0.79 |
| sort | 1.47 |
| ranges | 0.07 |
| raster | 2.59 |
| loss | 0.73 |
| raster_bwd | 6.85 |
| project_bwd | 1.13 |
| optimizer | 8.02 |

## Rasterizer timing on synthetic views

Without `--data`, `gsplat_cuda_bench` renders and runs the backward pass
from 6 views that stand at the median of the means and turn about the y
axis. It reads the image and every gradient back each iteration, so the GPU
idles between iterations. The repository's EuRoC splats are degree 0,
rendered at 752x480 over 30 iterations, with an 8 GB laptop GPU under
WSL2, CUDA 13.3:

| Stage | V1_01 (423k gaussians, 2.10M intersections), before / after | MH_01 (346k, 0.18M), before / after |
|---|---|---|
| project | 0.236 / 0.235 ms | 0.171 / 0.186 ms |
| scan (includes the intersection-count readback) | 0.124 / 0.144 ms | 0.122 / 0.139 ms |
| duplicate | 0.237 / 0.659 ms | 0.105 / 0.198 ms |
| sort | 0.745 / 0.753 ms | 0.426 / 0.167 ms |
| ranges | 0.035 / 0.038 ms | 0.033 / 0.035 ms |
| raster | 0.253 / 0.296 ms | 0.171 / 0.164 ms |
| raster_bwd | 0.844 / 0.872 ms | 0.497 / 0.451 ms |
| project_bwd | 0.114 / 0.040 ms | 0.090 / 0.088 ms |
| **forward / backward** | **1.63 / 0.96 → 2.12 / 0.91 ms** | **1.03 / 0.59 → 0.89 / 0.54 ms** |

- **The masks do not help here.** The views stand inside the scene, where
  large nearby splats cover whole tiles, so the masks are nearly all ones.
  Their cost in duplicate shows on V1_01.
- **Which numbers count.** Training views are what the optimizations
  target, and they are measured in the previous section.
- **Earlier figures.** ABI 1 timed the gaps between consecutive events, and
  so counted host gaps between stages. It reported 2.44 / 0.95 ms and 1.80 /
  0.58 ms for the baseline kernels.

## Against the write-up's optimizations

| Write-up step | Here |
|---|---|
| Exact conic-tile intersection | In (`tile_row_span`, as in the wgpu renderer) |
| Two-stage sort (visible by depth, then a narrow tile sort) | Not yet: one 64-bit sort (1.5 ms of a training step) |
| Warp butterfly reduce-scatter of the gradients (step 2) | In: a transposed butterfly (8+4+2+1+1 shuffles for 10 sums), skipped when no lane blended the splat; the sums go to global memory directly |
| Sub-tile bitmasks to skip splats before evaluating them | In: 8x4 sub-tiles, 8 bits in the sorted value |
| SH register streaming, cooperative zeroing (step 3) | Not yet |
| Contributor compaction and pruning the backward pass (step 4) | `splat_touched` gating in (`project_bwd`); compaction not yet (the backward pass starts at each tile's last blended entry) |
| Branchless predication (step 5) | Not yet |
| Two pixels per thread, per-warp staging without block barriers (step 6) | Not yet |

## Next

1. **Speed.** The remaining large stages are the dense optimizer (8.0 ms;
   sparse SH Adam halves it), raster_bwd (6.9 ms: the write-up's steps 5
   and 6), raster (2.6 ms) and the sort (1.5 ms: the two-stage sort).
2. **Geometry.** Add the geometry channels, then score `gsplat_mesh`'s mesh
   on ETH3D.
