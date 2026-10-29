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
  version (currently 2).
- Cargo never compiles CUDA, and nothing in the default build depends on the
  crate. It is `publish = false`.

## Status

| Part | State |
|---|---|
| Rasterizer forward (projection, SH colour, tile binning, sort, compositing) | Done |
| Backward to every gaussian parameter (mean, quaternion, log-scale, opacity logit, SH) | Done |
| Training step: L1 + D-SSIM, Adam with brush's aux losses, refine statistics, mean noise | Done |
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
| 300 steps on a scene rendered from known gaussians, colours and opacities perturbed | Mean L1 0.0175 → 0.0009 |

## Training on south-building

The dataset is south-building's 128 raw JPGs through `gsplat_photos
--steps 0 --cpu --cpu-sift --max-size 1024`:

- All 128 images registered, with 101,450 points.
- The focal length was refined to 850.2 px (COLMAP: 847.2).
- The run took 945 s on the CPU.

Then `gsplat_train_cuda --steps 30000 --eval-every 5000 --max-gaussians
2500000`, with brush's preset, every 8th view held out (112 train / 16
eval), 1024x768:

| Step | Eval PSNR | Eval SSIM | Gaussians | Training time (evaluations left out) |
|---|---|---|---|---|
| 5,000 | 22.37 | 0.738 | 304k | 76 s |
| 10,000 | 22.87 | 0.787 | 750k | 202 s |
| 15,000 | **23.30** | 0.809 | 1.35M | 387 s |
| 20,000 | 23.09 | 0.814 | 1.35M | 600 s |
| 25,000 | 22.96 | 0.814 | 1.35M | 805 s |
| 30,000 | 22.99 | 0.815 | 1.35M | **978 s** |

- **Peak and overfitting.** Eval PSNR peaks when growth stops (15k) and
  then drifts down 0.3 dB while the training L1 keeps falling
  (0.033 → 0.026), so the model overfits the 112 views.
- **Against the wgpu run.** The README's wgpu run of the same photos
  reached 22.77 dB at 30k. That run used GPU SIFT for the SfM and
  `--normal-weight 0.005`, which the photos page puts at about -0.2 dB, so
  the two runs agree within their differences.
- **Speed.** 978 s is 32.6 ms per step on average. The breakdown for steps
  29,001 to 30,000, with 1.35M gaussians, in GPU ms per step:

| Stage | ms/step |
|---|---|
| project | 1.03 |
| scan | 0.21 |
| duplicate | 0.61 |
| sort | 1.53 |
| ranges | 0.11 |
| raster | 3.77 |
| loss (L1 + D-SSIM) | 0.83 |
| raster_bwd | 9.64 |
| project_bwd | 2.45 |
| optimizer (statistics, Adam on every group, mean noise) | 8.69 |
| **total** | **28.9** |

The two largest stages are the next targets:

- **raster_bwd.** The write-up's steps 4 to 6 address it.
- **The optimizer.** It is a dense Adam over all 1.35M x 59 parameters, as
  in the wgpu trainer and Inria's code. It is bandwidth-bound, and the 45
  SH-rest coefficients per gaussian dominate it. A sparse Adam that skips
  gaussians no view saw (as in Taming 3DGS) would cut most of it, but it
  changes the training, so it needs its own PSNR check.

## Rasterizer timing

`gsplat_cuda_bench` runs a render and a backward pass for each iteration,
cycling through 6 views. The scene has no cameras, so the views stand at the
median of the means and turn about the y axis. The figures are median GPU
times of each stage (a CUDA event pair around it, so host gaps between
stages and host copies are left out). Hardware: an 8 GB laptop GPU under
WSL2, CUDA 13.3.

The repository's EuRoC splats are degree 0, rendered at 752x480 over 30
iterations:

| Stage | V1_01 (423k gaussians, 2.10M intersections) | MH_01 (346k, 0.18M) |
|---|---|---|
| project | 0.236 ms | 0.171 ms |
| scan (includes the intersection-count readback) | 0.124 ms | 0.122 ms |
| duplicate | 0.237 ms | 0.105 ms |
| sort | 0.745 ms | 0.426 ms |
| ranges | 0.035 ms | 0.033 ms |
| raster | 0.253 ms | 0.171 ms |
| raster_bwd | 0.844 ms | 0.497 ms |
| project_bwd | 0.114 ms | 0.090 ms |
| **forward / backward** | **1.63 / 0.96 ms** | **1.03 / 0.59 ms** |

- **Earlier figures.** ABI 1 timed the gaps between consecutive events, and
  so counted host gaps between stages. It reported 2.44 / 0.95 ms and 1.80 /
  0.58 ms for the same kernels.
- **What these are not.** These are not training numbers: the views are
  synthetic, there is no SH, and the loss gradient is a fixed pattern.

## Against the write-up's optimizations

| Write-up step | Here |
|---|---|
| Exact conic-tile intersection | In (`tile_row_span`, as in the wgpu renderer) |
| Two-stage sort (visible by depth, then a narrow tile sort) | Not yet: one 64-bit sort |
| Warp butterfly reduce-scatter of the gradients (step 2) | In: a transposed butterfly (8+4+2+1+1 shuffles for 10 sums), skipped when no lane blended the splat |
| Sub-tile bitmasks to skip splats before evaluating them | Not yet |
| SH register streaming, cooperative zeroing (step 3) | Not yet |
| Contributor compaction and pruning the backward pass (step 4) | Not yet (the backward pass starts at each tile's last blended entry) |
| Branchless predication (step 5) | Not yet |
| Two pixels per thread, per-warp staging without block barriers (step 6) | Not yet |

## Next

1. **Speed.** Work on raster_bwd (the write-up's steps 4 to 6) and on a
   sparse Adam. Check each change against the oracle tests and the
   south-building PSNR.
2. **Geometry.** Add the geometry channels, then score `gsplat_mesh`'s mesh
   on ETH3D.
