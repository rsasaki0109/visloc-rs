# CUDA 3DGS rasterizer (`visloc-gsplat-cuda`)

An opt-in CUDA backend for the 3D Gaussian Splatting rasterizer, next to the
wgpu one in `visloc-gsplat-render`. It exists for two reasons:

- **Machines where wgpu cannot reach the GPU.** Under WSL2 the NVIDIA driver
  exposes CUDA but no Vulkan driver (wgpu falls back to the CPU's lavapipe),
  and Windows Smart App Control blocks locally built executables, so on such
  a laptop CUDA under WSL2 is the only way to the GPU.
- **A reference for CUDA-level optimizations.** Write-ups such as George
  Kopanas's [five CUDA optimizations for 3DGS](https://grgkopanas.github.io/3dgs_cuda_blog/)
  rely on warp intrinsics, float atomics and Nsight Compute. They can be
  measured here first and ported to WGSL where they pay off.

The packaging follows the DPVO CUDA kernels: `native/gsplat_cuda/gsplat_cuda.cu`
is built into a shared library by `scripts/build_gsplat_cuda.sh`, and the
crate loads it at runtime with `libloading` after checking its ABI version.
Cargo never compiles CUDA, nothing in the default build depends on the crate,
and it is `publish = false` until the trainer lands.

## Status

| Stage | State |
|---|---|
| Rasterizer forward (projection, SH colour, tile binning, sort, compositing) | Done |
| Backward to every gaussian parameter (mean, quaternion, log-scale, opacity logit, SH) | Done |
| Trainer (losses, Adam, densification) | Not started |
| Geometry channels (normal, depth) for the normal loss and the mesh | Not started |

The maths is the wgpu renderer's (`compute_projected`, `tile_row_span`,
`rasterize`, `rasterize_backward`, `project_backward`), so both backends and
the CPU f64 oracle (`visloc_gsplat_core::backward`) agree. Two things are
organised differently:

- Per-gaussian state stays indexed by gaussian id (no visible-set compaction).
- Tile-then-depth order comes from one 64-bit CUB radix sort of
  `tile << 32 | depth bits`, stable in gaussian order (the oracle's tie
  order).

## Build and test

Under Linux or WSL2 with a CUDA toolkit (12.8 or newer for the newest GPUs):

```bash
scripts/build_gsplat_cuda.sh /tmp/gsplat_cuda            # ARCH defaults to `native`
export VISLOC_GSPLAT_CUDA_LIB=/tmp/gsplat_cuda/libvisloc_gsplat_cuda.so
cargo test --release -p visloc-gsplat-cuda               # skipped when the variable is unset
cargo run --release -p visloc-gsplat-cuda --example gsplat_cuda_bench -- \
    --scene docs/euroc_splat/euroc_v101.splat --width 752 --height 480 --iters 30
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
| 64 gaussians, 96x80 | 4.9e-7 (absolute) | 2.8e-6 | 1.6e-6 |
| 2,500 faint gaussians, 64x48 (about 500 per tile: several batches) | 4.5e-7 (absolute) | 6.8e-6 | 1.1e-6 |
| Active SH degree 1 and 0 of a degree-3 scene | | | 1.8e-6; bands above the active degree get exactly 0 |

## Timing

`gsplat_cuda_bench` runs a render and a backward pass for each iteration,
cycling through 6 views. The scene has no cameras, so the views stand at the
median of the means and turn about the y axis. The figures are median GPU
times between CUDA events, with host copies excluded. Hardware: an 8 GB
laptop GPU under WSL2, CUDA 13.3.

The repository's EuRoC splats are degree 0, rendered at 752x480 over 30
iterations:

| Stage | V1_01 (423k gaussians, 2.10M intersections) | MH_01 (346k, 0.18M) |
|---|---|---|
| project | 0.449 ms | 0.431 ms |
| scan (includes the intersection-count readback) | 0.301 ms | 0.526 ms |
| duplicate | 0.321 ms | 0.230 ms |
| sort | 1.082 ms | 0.386 ms |
| ranges | 0.039 ms | 0.052 ms |
| raster | 0.250 ms | 0.172 ms |
| raster_bwd | 0.835 ms | 0.493 ms |
| project_bwd | 0.117 ms | 0.088 ms |
| **forward / backward** | **2.44 / 0.95 ms** | **1.80 / 0.58 ms** |

These are not training numbers. The views are synthetic, there is no SH,
and the loss gradient is a fixed pattern. They set the baseline for the
optimizations below.

## Against the write-up's optimizations

| Write-up step | Here |
|---|---|
| Exact conic-tile intersection | In (`tile_row_span`, as in the wgpu renderer) |
| Two-stage sort (visible by depth, then a narrow tile sort) | Not yet: one 64-bit sort, the largest forward stage above |
| Warp butterfly reduce-scatter of the gradients (step 2) | In: a transposed butterfly (8+4+2+1+1 shuffles for 10 sums), skipped when no lane blended the splat |
| Sub-tile bitmasks to skip splats before evaluating them | Not yet |
| SH register streaming, cooperative zeroing (step 3) | Not yet |
| Contributor compaction and pruning the backward pass (step 4) | Not yet (the backward pass starts at each tile's last blended entry) |
| Branchless predication (step 5) | Not yet |
| Two pixels per thread, per-warp staging without block barriers (step 6) | Not yet |

## Next

1. A CUDA trainer on these kernels: L1 + D-SSIM, Adam, and brush's refine
   strategy. Measure PSNR and time on south-building against the wgpu
   trainer's recorded numbers.
2. The geometry channels, then scoring `gsplat_mesh`'s mesh on ETH3D.
3. The write-up's optimizations, one at a time, each against the oracle
   tests and the timing table.
