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
  version (currently 9).
- Cargo never compiles CUDA, and nothing in the default build depends on the
  crate. It is `publish = false`.

## Status

| Part | State |
|---|---|
| Rasterizer forward (projection, SH colour, tile binning, sort, compositing) | Done |
| Backward to every gaussian parameter (mean, quaternion, log-scale, opacity logit, SH) | Done |
| Training step: L1 + D-SSIM, Adam with brush's aux losses, refine statistics, mean noise; opt-in sparse SH Adam | Done |
| Trainer loop (`trainer::CudaTrainer`, example `gsplat_train_cuda`): brush's refine strategy and learning rates, or a fixed population | Done |
| Inria densification rule, per-image appearance | Not supported (the trainer refuses these configurations) |
| Median depth (splat centre, or where the ray meets the splat plane) and mesh extraction (`gsplat_mesh_cuda`, `gsplat_train_cuda --mesh`) | Done |
| Depth prior: an expected-depth channel and a relative L1 against visloc-mvs's fused depth (`--depth-prior`, opt-in) | Done; ETH3D mesh F1 33.1% → 35.0% at w = 0.1 |
| Depth-normal loss (2DGS / PGSR style, the wgpu trainer's `normal_weight` / `normal_start`): a normal channel next to the expected depth (`--normal-weight`, opt-in) | Done; no gain on ETH3D (mean F1 32.0% at w = 0.005 against 33.1%) |
| TSDF fusion and surface nets on the device, in slabs of bounded memory (`gsplat_mesh_cuda --gpu-tsdf`, `mesh::extract_mesh_gpu`) | Done; same ETH3D F1 as the CPU, 4–5x faster, 1/6 the memory |
| Meshes from the expected depth the depth losses train (`gsplat_mesh_cuda --expected-depth`, `mesh::MeshDepth`) | Done; with the depth prior at 30, ETH3D mesh F1 33.3% → 63.6% (the prior's MVS clouds: 67.6%) |
| Exhaustive SIFT matching for `build_photo_dataset` (`gsplat_photos_cuda`, feature `photos`) | Done; the CPU matcher's matches bit for bit, south-building's front end 780.7 s → 18.4 s |

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
- **Depth losses**: with a depth prior or the depth-normal loss on, the
  step also blends the expected depth, and for the depth-normal loss the
  normal `sum_i T_i a_i n_i` (n_i the axis of the smallest scale, turned to
  face the camera). `normal_loss.wgsl`'s two passes become two kernels.
  The backward pass carries both channels in free slots of its butterfly:
  through alpha to opacity, conic and 2D mean, the depth to the centre's z,
  and the normal to the rotation.

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

More tests check the depth and normal outputs:

- **Median depth.** A faint disc gives no median, with opacity 0.4. An
  opaque disc behind it gives its depth, with opacity 1 - 0.6 x 0.1.
- **Planar depth.** For a disc turned 30 degrees, the depth matches the
  ray-plane intersection to 1e-3.
- **Expected-depth channel** (the depth prior's path). Each splat's red is
  set to its view depth / 10 on black, so red is exactly D / 10. A red
  gradient and a depth gradient / 10 then give the same opacity, conic and
  2D-mean gradients, and mean gradients that differ by exactly W^T e_z
  times the red gradient / 10. Worst error 1.8e-7.
  - Finite differences cannot check this: a pixel crossing the alpha gate
    moves D by z / 255.
- **Normal channel** (the depth-normal loss's path), by the same trick.
  Each splat's colour is set to its camera-space normal + 1 on black, so
  the colour is N + A (A = 1 - T), and a colour-1 render gives A. A colour
  gradient then gives the normal channel's opacity, conic, mean and scale
  gradients as the difference of the two renders' gradients. Its
  quaternion gradients differ by J_n^T g_n, the rotation of the normal,
  which the test takes by central differences of the smooth n(q) on the
  CPU. Worst error 9.0e-7.
- **Depth-normal loss.** Its value and its gradients with respect to the
  normal and expected-depth channels against central differences of an f64
  CPU copy (opacity held constant, as in the kernel): 1.9e-6 and 2.5e-6 of
  their largest values.

`tests/training.rs` checks the training step:

| Test | Result |
|---|---|
| Loss gradient against the CPU L1 and `metrics::ssim_with_grad` (weight 0.2) | 4.5e-5 of its largest value; L1 alone 2.0e-8 |
| Masked pixels (ground-truth alpha below 128) | Exactly zero gradient |
| Refine scatter (prune, keep, clone, brush split) | Row order, split means symmetric about the parent, scales / sqrt 2, opacity 1 - sqrt(1 - a), statistics reset |
| 300 steps on a scene rendered from known gaussians, colours and opacities perturbed, dense and sparse SH Adam | Mean L1 0.0175 → 0.0009 in both; a gaussian behind every camera keeps its SH bit for bit |
| 300 steps of the depth-normal loss (weight 1) on a dense scene fitted to its own renders, only rotations learning | Depth-normal loss 0.286 → 0.099 |

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

### Tried and not adopted

Three more changes were timed with `gsplat_cuda_bench --data`, three rounds
each, alternating with the current kernels. The bench now also reports the
host wall time of a step, which counts the gradient memsets and launches
between the timed stages. The current kernels took 21.0–21.3 ms of wall
time per step (stages 19.7–20.0 ms).

| Change | Effect per step | Why |
|---|---|---|
| Adam on 4 elements per thread (16-byte loads and stores, no 64-bit index division) | optimizer 6.66 → 6.69 ms | The kernel was already bound by memory: 28 bytes per element. |
| No gradient memset: `project_bwd` writes zeros for the gaussians that blended into nothing | project_bwd 1.0 → 2.5 ms; wall unchanged or worse | Each thread writes its own 59-float rows, which do not coalesce; one memset is faster. |
| Two-stage sort (gaussians by depth, then a stable 32-bit sort by tile alone) | sort 1.35 → 0.29 ms, but scan +0.27 ms (the depth sort) and duplicate +0.35 ms; wall −0.1 ms (under 1%) | Duplicating in depth order reads the gaussians out of order. |

- **Order.** The two-stage sort keeps the order of the one 64-bit sort
  (equal depths in gaussian order), and all three passed the oracle and
  training tests.
- **Where the time is.** A step stays at about 21 ms. The optimizer
  (6.7 ms) moves 28 bytes per element; only fewer bytes would help, such as
  moments in bf16 or the opt-in sparse SH Adam, and both change the
  numerics. raster_bwd (6.1 ms) is the write-up's steps 5 and 6.

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
| Two-stage sort (visible by depth, then a narrow tile sort) | Tried, under 1% per step ([above](#tried-and-not-adopted)); one 64-bit sort (1.4 ms of a training step) |
| Warp butterfly reduce-scatter of the gradients (step 2) | In: a transposed butterfly (8+4+2+1+1 shuffles for 10 sums), skipped when no lane blended the splat; the sums go to global memory directly |
| Sub-tile bitmasks to skip splats before evaluating them | In: 8x4 sub-tiles, 8 bits in the sorted value |
| SH register streaming, cooperative zeroing (step 3) | Not yet |
| Contributor compaction and pruning the backward pass (step 4) | `splat_touched` gating in (`project_bwd`); compaction not yet (the backward pass starts at each tile's last blended entry) |
| Branchless predication (step 5) | Not yet |
| Two pixels per thread, per-warp staging without block barriers (step 6) | Not yet |

## Geometry on ETH3D

How good is the geometry of a trained splat? Three ETH3D high-res training
scenes were trained on their ground-truth poses and the meshes scored with
the official evaluator against the laser scans
(`scripts/eth3d_3dgs_mesh.sh`, evaluator built by
`scripts/build_eth3d_evaluator.sh`). The setup:

- **Images.** `colmap_resize --size 1600x1067` scales the undistorted DSLR
  images and their cameras. terrains has two cameras of slightly different
  sizes, so `--size` gives every camera the same size and scales each
  camera's intrinsics per axis.
- **Training.** 30k steps with brush's preset on every view (`--holdout 0`),
  up to 2.5M gaussians.
- **Mesh.** Median depth from every view, fused into a TSDF
  (`visloc_gsplat_train::mesh::fuse_views`, shared with wgpu's
  `extract_from_splat`). The evaluator scores the mesh vertices.

F1 at 2 cm, with accuracy / completeness in parentheses:

| Scene | Images | Gaussians | Training | Default mesh | ETH3D-oriented mesh | visloc-mvs cloud |
|---|---|---|---|---|---|---|
| terrains | 42 | 616k | 628 s | 18.6% (72.7 / 10.7) | 34.1% (46.5 / 27.0) | 91.2% |
| playground | 38 | 1.93M | 1231 s | 34.6% (57.2 / 24.8) | 43.9% (50.3 / 39.0) | 69.8% |
| office | 26 | 201k | 460 s | 9.2% (84.2 / 4.9) | 21.4% (39.1 / 14.8) | 79.0% |
| **mean** | | | | **20.8%** | **33.1%** | 80.0% |

- **Default mesh.** `gsplat_photos`' options, built for photos-to-mesh:
  - Voxel scene scale / 256: 1.0 cm on terrains, 2.7 cm on playground.
  - Depths up to 2 x the scene scale.
  - Triangles within 10 voxels of an SfM point. ETH3D's points are sparse,
    18,553 on terrains.
  - Voxels seen by at least 3 views.
  - Accuracy is fair, but most of the scan goes missing.
- **ETH3D-oriented mesh** (the script's default):
  - 1 cm voxels.
  - No support filter.
  - Depths up to 20 m.
  - Voxels seen once are kept.
  - Planar median depth.
  - Exception: playground uses 1.5 cm voxels and depths up to 14 m. At
    1 cm the CPU TSDF ran out of WSL2's 11 GB, and even this setting peaks
    at 9.7 GB. The GPU TSDF ([below](#tsdf-on-the-gpu)) meshes it at 1 cm
    and 20 m in 4.2 GB: 44.5%.
- **visloc-mvs cloud.** From `docs/mvs.md` (3200 px, all 13 scenes on an
  L4). For meshes from MVS depth maps, the 13-scene mean is 79.3%.

Each option on terrains, at 2 cm:

| Mesh options | Accuracy | Completeness | F1 |
|---|---|---|---|
| Default | 72.7% | 10.7% | 18.6% |
| No support filter | 54.9% | 15.0% | 23.6% |
| No support filter, depths up to 20 m | 42.3% | 17.6% | 24.9% |
| … and planar depth | 52.7% | 19.0% | 27.9% |
| No support filter, depths up to 20 m, min weight 1 | 36.6% | 22.4% | 27.8% |
| … and planar depth | 46.2% | 27.0% | 34.1% |

- **Planar depth** (`--planar-depth`). The median depth is the camera z of
  the splat that takes the transmittance to 0.5. Splats lie along the
  surface, so on surfaces seen at an angle their centre is off the pixel's
  ray. Planar depth takes the depth where the ray meets that splat's plane
  instead: the normal is the axis of its smallest scale, as in PGSR's
  unbiased depth. If that point is more than twice the centre's depth, the
  centre's depth is kept. On terrains it adds 3–6 points of F1 at 2 cm.
- **Where 3DGS stands.** A plain 3DGS (no geometric regularization) reaches
  about 2/5 of MVS's F1 at 2 cm. The gap is at cm scale: on terrains,
  completeness reaches 92% at 50 cm. office, with 26 views and weak
  texture, is the worst; the splat fits its appearance without its
  geometry.

### Depth prior from visloc-mvs

`gsplat_train_cuda --depth-prior <dir> --depth-weight w` adds
`w * mean |D - P| / P` to the loss:

- **P** is the prior. visloc-mvs (`--save-fused-depth`) writes, per view,
  the depths that fusion's source views agree with. Pixels without one are
  left out of the mean's sum but not its count.
- **D** is the splat's expected depth `sum_i T_i a_i z_i`, a fourth blended
  channel whose gradient reaches opacity, conic and mean as the colour's
  does.

visloc-mvs ran on the CPU on the same 1600x1067 images: at 1600 px on
terrains (4483 s), and at 800 px on playground (804 s) and office (468 s),
whose maps are resampled to 1600 px by nearest neighbour. Training and
meshing are otherwise as above. F1 at 2 cm, with accuracy / completeness:

| Scene | No prior | w = 0.1 | w = 0.5 | w = 1.0 | The prior's own cloud |
|---|---|---|---|---|---|
| terrains | 34.1% (46.5 / 27.0) | 34.6% (62.2 / 24.0) | 37.3% (74.5 / 24.8) | **38.6%** (78.8 / 25.5) | 84.7% |
| playground | 43.9% (50.3 / 39.0) | **44.8%** (59.5 / 35.9) | 42.3% (64.5 / 31.4) | 42.7% (66.4 / 31.5) | 61.3% |
| office | 21.4% (39.1 / 14.8) | **25.5%** (42.7 / 18.2) | 22.5% (40.3 / 15.6) | 24.7% (46.3 / 16.9) | 56.8% |
| **mean** | 33.1% | 35.0% | 34.0% | **35.3%** | 67.6% |

- **Accuracy rises, completeness does not.** The prior moves the surfaces
  onto the MVS depths. On terrains and playground accuracy climbs with the
  weight, by up to 32 and 16 points. Completeness falls there by 2 to 8
  points, so F1 moves little.
- **The weight matters little.** Every weight gains 1 to 2 points of mean
  F1. Their order is not monotonic (0.5 is lowest), so the differences are
  likely run-to-run variation, which float atomics cause and which was not
  measured on these scenes.
- **The mesh stays far below its prior.** Even with the prior, the 3DGS
  mesh scores about half of the cloud that supervised it.
- **Cost.** Training time is unchanged (608, 843 and 370 s at w = 0.1).
  The prior stays opt-in.

### Depth-normal loss

`gsplat_train_cuda --normal-weight w` adds `w * mean (1 - N . n_d)` from
step 7000 (`--normal-start`), the wgpu trainer's 2DGS / PGSR-style loss:

- **N** is the blended normal channel.
- **n_d** is the normal of the expected-depth map, from the cross product of
  its central differences.
- **Pixels.** Only those whose own and 4-neighbour opacities exceed 0.5
  count.

Training and meshing are as above. F1 at 2 cm, with accuracy /
completeness:

| Scene | No loss | w = 0.005 | w = 0.005 and depth prior 0.1 | Depth prior 0.1 alone |
|---|---|---|---|---|
| terrains | 34.1% (46.5 / 27.0) | 31.6% (45.8 / 24.1) | 35.6% (70.9 / 23.8) | 34.6% (62.2 / 24.0) |
| playground | 43.9% (50.3 / 39.0) | 43.1% (48.7 / 38.6) | 43.0% (58.5 / 33.9) | 44.8% (59.5 / 35.9) |
| office | 21.4% (39.1 / 14.8) | 21.5% (36.9 / 15.2) | 23.3% (41.0 / 16.3) | 25.5% (42.7 / 18.2) |
| **mean** | 33.1% | 32.0% | 34.0% | 35.0% |

- **No gain on these scenes.** Alone, w = 0.005 lowers the mean by 1.1
  points: terrains loses 2.5, the other two move by under a point. With the
  depth prior it scores a point below the prior alone. Both differences are
  about the spread of the depth-prior weights above (34.0–35.3%), so the loss
  helps no more than run-to-run variation.
- **What it acts on.** The loss shapes the expected depth and the blended
  normals, while the mesh is fused from the planar median depth. A loss on
  the median depth itself, as PGSR's, was not tried.
- **Cost.** Training time stays within the runs' spread (653, 915 and 427 s
  alone). The loss stays opt-in.
- **w = 0.05 is far too strong.** It took terrains to 22.9% (40.3 / 16.0).
  The image L1 at 30k was 0.0152 against 0.0083, and the loss itself did
  not fall during training. The wgpu trainer saw the same on bonsai (-3.3 dB
  at 7k with 0.05) and uses 0.005.

### Meshing the depth the losses train

The depth prior and the depth-normal loss act on the expected depth
`D / (1 - T)`, but the meshes above were fused from the planar median depth.
`gsplat_mesh_cuda --expected-depth` (`mesh::MeshDepth::Expected`) fuses the
expected depth where the opacity exceeds 0.5, and 0 elsewhere. The splats
are the ones trained above, plus stronger priors. All meshes use the GPU TSDF ([below](#tsdf-on-the-gpu))
at 1 cm and depths up to 20 m on every scene. F1 at 2 cm:

| Trained with | Mesh depth | terrains | playground | office | mean |
|---|---|---|---|---|---|
| no loss | planar median | 34.1% | 44.5% | 21.2% | 33.3% |
| no loss | expected | 4.8% | 8.9% | 5.4% | 6.4% |
| depth prior 0.1 | planar median | 34.7% | 45.1% | 25.3% | 35.0% |
| depth prior 0.1 | expected | 43.9% | 29.7% | 25.8% | 33.1% |
| depth prior 1.0 | planar median | 38.6% | 43.3% | 24.6% | 35.5% |
| depth prior 1.0 | expected | 56.8% (82.9 / 43.2) | 40.3% | 38.4% (59.1 / 28.4) | 45.2% |
| depth prior 0.1, normal 0.005 | planar median | 35.6% | 43.4% | 23.2% | 34.1% |
| depth prior 0.1, normal 0.005 | expected | 51.8% | 32.4% | 24.2% | 36.1% |
| depth prior 1.0, normal 0.005 | planar median | 42.5% | 40.7% | 23.0% | 35.4% |
| depth prior 1.0, normal 0.005 | expected | 60.5% | 41.0% | 36.2% | 45.9% |
| depth prior 3.0 | planar median | 52.5% | 43.4% | 26.8% | 40.9% |
| depth prior 3.0 | expected | 69.2% | 49.6% | 42.5% | 53.8% |
| depth prior 10 | planar median | 64.8% | 46.3% | 30.7% | 47.3% |
| depth prior 10 | expected | 76.0% | 54.2% | 45.7% | 58.6% |
| depth prior 30 | planar median | 70.7% | 50.2% | 39.4% | 53.4% |
| depth prior 30 | expected | **78.1%** (88.1 / 70.2) | **60.4%** (71.4 / 52.4) | **52.3%** (60.8 / 45.9) | **63.6%** |
| *the prior's own cloud* | | *84.7%* | *61.3%* | *56.8%* | *67.6%* |

- **Mesh the depth the losses train.** The expected-depth mesh beats the
  median mesh at every prior of 1.0 or more, and the gap grows with the
  weight. Completeness, which no loss raised in the median mesh, rises
  with it: terrains 26.9% (no loss, median) to 70.2%, office 14.5% to
  45.9%.
- **The weight matters now.** The median meshes moved by a point or two
  between weights 0.1 and 1.0 (see the depth-prior section). The
  expected-depth meshes climb from 33.1% at 0.1 to 63.6% at 30. At 30 they
  come within 5–7 points of the MVS cloud that supervised them on terrains
  and office, and within 1 on playground.
- **Not without a prior.** A plain splat's expected depth blends the splats
  along each ray, and its mesh scores 6.4%. With the prior at 0.1,
  playground still does better from the median.
- **The depth-normal loss** helps the expected-depth mesh where the prior is
  weak (terrains at 0.1: 43.9% → 51.8%). At 1.0 it changes the mean by under
  a point (45.9% against 45.2%).
- **The cost is the image.** A strong prior pulls the splats onto the MVS
  depths against the photometric loss. At 30 the final training L1 is about
  twice the plain splat's: terrains 0.0083 → 0.0174, playground 0.0200 →
  0.0372, office 0.0085 → 0.0147. terrains also trains a third longer (850k
  gaussians against 616k). Held-out image quality was not measured here
  (`--holdout 0`).
- **What this is.** At 30 the splat largely carries the MVS depth into a
  coloured mesh. It does not see more than MVS does. For geometry, train with
  a strong prior and mesh the expected depth; for novel views, keep the
  prior low or off.

**Held-out image quality, and a prior that fades.** These runs hold out every
8th view (`--holdout 8`), so the training views drop to 36, 33 and 22. They
report PSNR and SSIM on the held-out views at 30k, and mesh from all views.
`--depth-weight-final W` moves the prior's weight log-linearly from
`--depth-weight` to W over the run, as the mean's learning rate does.

| Prior | Held-out PSNR (terrains / playground / office) | SSIM | vs no prior | Mesh F1 at 2 cm, expected depth (terrains / playground / office) | mean |
|---|---|---|---|---|---|
| none | 25.79 / 19.27 / 22.31 dB | 0.897 / 0.698 / 0.917 | | 5.1 / 8.3 / 8.8% (median: 34.3 / 43.5 / 22.6%) | 6.1% (median 33.5%) |
| 30, constant | 24.15 / 18.13 / 19.65 dB | 0.878 / 0.645 / 0.880 | −1.81 dB | 79.0 / 60.4 / 50.1% | 63.2% |
| 30 → 1 | 24.77 / 18.51 / 21.75 dB | 0.885 / 0.670 / 0.907 | −0.79 dB | 70.6 / 51.1 / 42.0% | 54.6% |

- **The fade halves the image's cost.** It keeps 21 of the constant prior's
  30 points over the plain median mesh. office loses only 0.56 dB.
- **Holding views out costs the geometry nothing.** With the prior constant
  at 30 the meshes match the all-view runs (63.2% against 63.6%).
- **The choice.** Constant for the best mesh, a fade for a mesh and images
  together; the prior off for the best images.

## TSDF on the GPU

`gsplat_mesh_cuda --gpu-tsdf` (`mesh::extract_mesh_gpu`) fuses and extracts
the mesh on the device (`native/gsplat_cuda/tsdf.cuh`). It uses the same
maths as the CPU `Tsdf` and the same options and filters: 8^3-voxel
blocks over the truncation band, running means, free-space carving and
surface nets.

- **Allocation first.** Every view is rendered once and the blocks its
  truncation band crosses go into a hash set on the device. So every block
  sees every view. The CPU allocates as it fuses, and a block first touched
  by view k misses views 0 to k-1.
- **Slabs.** The sorted blocks are fused in slabs along x of at most
  `--slab-blocks` blocks (default 65536, 0.67 GB of voxels). Each slab also
  holds the next layer up for the cells on its top face. Each slab yields
  its vertices (one per cell) and its edges with a sign change. The host
  assembles the quads across slabs by sorted lookup, in parallel.
- **View cache.** The views' depth and colour stay on the device when they
  fit in 2 GiB (16 bytes per pixel); otherwise every slab renders the views
  again.
- **Memory.** The CPU `Tsdf` keeps 20 bytes per voxel for the whole
  volume. The device keeps them for one slab, and the host keeps only the
  output.

`tests/tsdf.rs`:

- **One view.** Both allocate before fusing, so the meshes match:
  - every vertex within 4e-4 voxels;
  - colours within 1 of 255;
  - the same triangles.
- **Four views.** The surface is bit-identical whether it is fused in 1 slab
  or 30, with or without the view cache. 99.8% of its vertices lie within
  0.01 voxels of the CPU's.

On the ETH3D scenes above (baseline splats, the ETH3D-oriented options,
1 cm voxels and depths up to 20 m everywhere), while another training run
shared the GPU:

| Scene | Blocks | CPU TSDF: F1 at 2 cm, wall, peak RSS | GPU TSDF: F1 at 2 cm, wall, peak RSS |
|---|---|---|---|
| terrains | 834k | 34.1%, 24.0 s, 10.3 GB | 34.1%, 5.4 s, 1.6 GB |
| playground | 2.35M | out of memory (11 GB) | 44.5%, 12.0 s, 4.2 GB |
| office | 786k | 21.4%, 20.3 s, 9.8 GB | 21.2%, 4.5 s, 1.5 GB |

- **Quality.** F1 is the CPU's to 0.2 points. The GPU carves a little more:
  every block sees every view, so it has slightly fewer triangles.
- **Time.** Wall times include loading the splat and writing the mesh. On
  playground the fusion takes 6.5 s, the quads 1.9 s and the filters 1.0 s
  (`remove_small_components` now counts in a vector instead of a hash map).
- **Without the cache.** Each of playground's 38 slabs rendered the 38 views
  again, and fusion took 31.6 s.

## Matching on the GPU

`gsplat_photos_cuda` (feature `photos`) runs `build_photo_dataset` on the
CPU path of `gsplat_photos`: SIFT, two-view verification and mapping on the
CPU, but the descriptor matching on the device. It plugs
`matching::CudaPairMatcher` into the new `PhotoSfmConfig::matcher` hook.
Without wgpu (WSL2) the CPU matched before.

- **On the device.** Per image pair, every descriptor's best and second
  score against the other image, in both directions
  (`CudaGsplat::match_top2`). The dot products are 8-bit (`__dp4a`), over
  tiles of 128 queries x 128 trains.
- **Exact.** The descriptors are RootSIFT bytes / 512, so every score is an
  exact integer. The host then repeats the CPU matcher's final filter (ratio
  0.8 both ways, then mutual) on the same values, so the matches are
  `BruteForceMatcher::match_descriptors_cross_checked`'s, bit for bit.
- **Ties.** The best is the lowest score, ties to the lower index; the second
  is the lowest of the others. That is the CPU's first-wins scan.
- **Test.** `tests/matching.rs` covers 30 pairs, with noisy copies, duplicate
  rows (ties), an empty image and a one-row image, at scales 512 and 1. The
  matches are identical.

south-building (128 images at 1024x768, mean 10.5k SIFT, 8128 pairs):

| | CPU matching (`gsplat_photos --cpu --cpu-sift`) | `gsplat_photos_cuda` |
|---|---|---|
| Matching and verification | 780.7 s (overlapped: match 780.2 s, verify 578.2 s) | **18.4 s** (match 15.0 s, verify 5.3 s) |
| Verified pairs | 2461 | 2461 |
| SfM in all | 945.0 s | **148.4 s** |
| Registered, points, refined focal | 128/128, 101,450, 850.2 px | the same |

- **Verification stays on the CPU.** It took 578 s only because the CPU
  matcher held every core. Alone it takes 5.3 s, so moving it to the GPU
  (the plan's next step) is not needed here.

## Next

1. **Geometry.** Meshing the expected depth with a strong depth prior
   brings the mesh within a few points of its MVS prior, at 1.8 dB of
   held-out PSNR; fading the prior halves that cost and keeps most of the
   gain. Open: a prior that leaves the image alone (on separate geometry
   channels, or splats the photometric loss cannot move off the surface),
   and why the median depth lags.
2. **Speed.** A step takes about 21 ms. The dense optimizer (6.7 ms) is
   bound by memory, so it only gets faster with fewer bytes per element,
   such as bf16 moments or the opt-in sparse SH Adam. raster_bwd (6.1 ms)
   needs the write-up's steps 5 and 6. The two-stage sort, a vectorized Adam
   and dropping the gradient memset each gained under 1%
   ([tried](#tried-and-not-adopted)).
