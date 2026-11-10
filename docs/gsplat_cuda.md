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
  version (currently 14).
- Cargo never compiles CUDA, and nothing in the default build depends on the
  crate. It is `publish = false`.

## Status

| Part | State |
|---|---|
| Rasterizer forward (projection, SH colour, tile binning, sort, compositing) | Done |
| Backward to every gaussian parameter (mean, quaternion, log-scale, opacity logit, SH) | Done |
| Training step: L1 + D-SSIM, Adam with brush's aux losses, refine statistics, mean noise; opt-in sparse SH Adam and bf16 moments | Done |
| Trainer loop (`trainer::CudaTrainer`, example `gsplat_train_cuda`): brush's refine strategy and learning rates, or a fixed population | Done |
| Inria densification rule, per-image appearance | Not supported (the trainer refuses these configurations) |
| Median depth (splat centre, or where the ray meets the splat plane) and mesh extraction (`gsplat_mesh_cuda`, `gsplat_train_cuda --mesh`) | Done |
| Depth prior: an expected-depth channel and a relative L1 against visloc-mvs's fused depth (`--depth-prior`, opt-in) | Done; ETH3D mesh F1 33.1% → 35.0% at w = 0.1 |
| Depth-normal loss (2DGS / PGSR style, the wgpu trainer's `normal_weight` / `normal_start`): a normal channel next to the expected depth (`--normal-weight`, opt-in) | Done; no gain on ETH3D (mean F1 32.0% at w = 0.005 against 33.1%) |
| TSDF fusion and surface nets on the device, in slabs of bounded memory (`gsplat_mesh_cuda --gpu-tsdf`, `mesh::extract_mesh_gpu`) | Done; same ETH3D F1 as the CPU, 4–5x faster, 1/6 the memory |
| MVS depth maps fused in place of the splat's depth, the splat filling their holes (`gsplat_mesh_cuda --gpu-tsdf --depth-maps <dir>`, `mesh::DepthMaps`) | Done; with a plain splat (image untouched) ETH3D mesh F1 62.7%, with the depth prior at 30 65.7%; with 1600 px MVS on CUDA 64.2% and 70.5% (the prior trained on it too); over six scenes, with map holes unknown and no component filter, 71.4% and 76.5% (MVS depth alone 70.2%); the splat's depth at half a map's weight (`--fill-weight 0.5`, the default) adds 0.3–2.0 points over three scenes |
| Meshes from the expected depth the depth losses train (`gsplat_mesh_cuda --expected-depth`, `mesh::MeshDepth`) | Done; with the depth prior at 30, ETH3D mesh F1 33.3% → 63.6% (the prior's MVS clouds: 67.6%) |
| Exhaustive SIFT matching for `build_photo_dataset` (`gsplat_photos_cuda`, feature `photos`) | Done; the CPU matcher's matches bit for bit, south-building's front end 780.7 s → 18.4 s |
| Label votes: each splat's blending weight per pixel class, summed over views (`CudaGsplat::vote_labels`, example `gsplat_semantic_cuda`) | Done; SAM 3 classes on FIORD's MeetingRoom_In, 82.6% of held-out pixels ([semantic_splat.md](semantic_splat.md)) |
| Semantic training: a logit per class and gaussian, its softmax blended like colour, the cross-entropy against per-pixel labels into the logits and, through alpha, the opacities and centres (the shapes too with `--semantic-shape`; `CudaTrainer::set_labels`, `gsplat_train_cuda --labels`) | Done; held-out pixels at the same or a higher PSNR: MeetingRoom_In 79.4–84.5% → 88.1–88.3%, Building_In 87.2–87.3% → 91.2–91.3%, Bridge_Out (outdoors) 93.7–93.8% → 95.0% (89.8% and 92.4% with the shapes, which grow floaters in some runs); training 12% slower, the labels carried by the colour kernels (2x as separate passes) ([semantic_splat.md](semantic_splat.md#training-with-the-labels)) |

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
- **Label votes.** With one splat white and the rest black on black, the
  CPU render is that splat's blending weight per pixel, so its votes are
  those weights summed by label. Over two views, through a pinhole and
  through fisheyes (one past 90°), and after a reset: worst error 1.7e-5.
- **Class channels** (`render_classes`, `classes_backward`).
  - **Setup.** Each splat's colour c is made three of its k class
    probabilities: logits ln c, then ln(1 - sum c) on one more channel and
    -40 on the rest.
  - **Forward.** On black, those three channels must render the image.
  - **Backward.** A gradient on them alone must give the colour's screen,
    transform and opacity gradients, and the logits' gradients must be the
    colour's through the softmax.
  - **Result.** For k = 4, 9 and 16 (the most), with the colour in
    channels 0–2, 4–6 and 12–14: worst error 3.0e-7.
- **Labels' cross-entropy** (training's path: the label channel inside the
  colour kernels, `label_backward`).
  - **Reference.** The same loss through every class channel:
    `render_classes`, the cross-entropy's gradient on the CPU, then
    `classes_backward`.
  - **Labels.** Warp-sized blocks on the left half, so a warp sees one or
    two labels, and narrow stripes on the right, three or more. Some pixels
    have none.
  - **Result.** For k = 5 and 16, the loss and the screen, transform,
    opacity and logit gradients with the shapes moving (`LabelGeometry::Full`):
    worst error 2.5e-6 (screen), 7.3e-7 elsewhere.
  - **The default** (`LabelGeometry::OpacityPosition`) gives the conic,
    rotations and scales exactly zero, and the centres, opacities, refine
    weight and logits the full gradient's within 1e-6. With the logits
    only, only the logits get a gradient, the same one.

`tests/training.rs` checks the training step:

| Test | Result |
|---|---|
| Loss gradient against the CPU L1 and `metrics::ssim_with_grad` (weight 0.2) | 4.5e-5 of its largest value; L1 alone 2.0e-8 |
| Masked pixels (ground-truth alpha below 128) | Exactly zero gradient |
| Refine scatter (prune, keep, clone, brush split) | Row order, split means symmetric about the parent, scales / sqrt 2, opacity 1 - sqrt(1 - a), statistics reset |
| 300 steps on a scene rendered from known gaussians, colours and opacities perturbed, dense and sparse SH Adam | Mean L1 0.0175 → 0.0009 in both; a gaussian behind every camera keeps its SH bit for bit |
| 300 steps of the depth-normal loss (weight 1) on a dense scene fitted to its own renders, only rotations learning | Depth-normal loss 0.286 → 0.099 |
| Refine scatter with semantic logits | Each row's logits follow it (a split's and a clone's two rows keep the parent's); `set_scene` zeroes them |
| 300 steps of the labels' cross-entropy (weight 1) from zero logits, against labels rendered from one-hot logits: geometry held still, opacities and centres learning, or shapes too | Cross-entropy 0.76 → 0.32, 0.75 → 0.30 and 0.74 → 0.17; 95.2%, 92.6% and 96.5% of the labelled pixels agree |

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
  numerics. raster_bwd (6.1 ms) is the write-up's steps 5 and 6. Both are
  done below.

### Two pixels per thread, and bf16 moments

Two more changes, timed the same way (128 steps on the first run's scene,
three rounds alternating with the kernels before them). GPU ms per step,
the sum of the stages, and the host wall time of a step:

| Backward | Adam moments | raster_bwd | optimizer | Stages | Wall |
|---|---|---|---|---|---|
| One pixel per thread | float | 6.05–6.42 | 6.66 | 19.9–20.5 | 21.2–22.0 |
| One pixel per thread | bf16 | 6.28–6.80 | 4.79 | 18.2–19.2 | 19.4–20.5 |
| Two pixels per thread | float | 5.60–5.83 | 6.66 | 19.5–19.8 | 20.8–21.2 |
| Two pixels per thread | bf16 | 5.53–5.54 | 4.79 | 17.4–17.5 | **18.6** |

- **Two pixels per thread** (the write-up's step 6, without its per-warp
  staging; the default).
  - **Layout.** raster_bwd runs 128 threads per tile, and each warp covers
    an 8x8 block: two of the 8x4 sub-tiles.
  - **Fewer reductions.** For each splat a lane adds its two pixels'
    contributions before the warp's butterfly, so the shuffles and global
    reductions per pixel halve. A warp skips a splat that misses both its
    sub-tiles, and a sub-tile's pixels skip one that misses theirs.
  - **Result.** raster_bwd takes 9–10% less time. The maths per pixel is
    unchanged, and the gradients change only in summation order.
- **bf16 moments** (opt-in, `--bf16-moments`).
  - **Format.** The Adam moments of the three groups are kept as bf16, the
    top half of a float.
  - **Result.** The dense optimizer moves 20 bytes per element instead of
    28 and takes 28% less time. The moments take half the memory: 236
    instead of 472 bytes per gaussian at degree 3.
  - **Stochastic rounding.** With beta2 0.999 the second moment changes by
    0.1% a step, which is under half a bf16 step (0.2%). Rounded to nearest
    it would stop moving, so each store adds 16 random bits before it
    truncates.
  - **Switching.** A step that asks for the other precision converts the
    moments, and refines scatter them in either.
- **Both together** take a step to 18.6 ms of wall time, 15% less than
  before.

south-building trained again with each kind of moments, two runs each,
before the two-pixel backward:

| Adam moments | PSNR at 30k | SSIM | Gaussians | Training time |
|---|---|---|---|---|
| float | 23.02 / 22.88 | 0.814 / 0.813 | 1.39M / 1.36M | 617 / 601 s |
| bf16 | 23.00 / 22.86 | 0.814 / 0.810 | 1.35M / 1.40M | 593 / 567 s |

- **bf16 is within the spread.** It lands within the float runs' spread,
  and within that of the three earlier dense runs (22.71–23.22 dB).
- **Time saved.** It saves 4–6% of the training time. Refines and the
  other stages make up the rest of a step.

Four ETH3D scenes at 1600 px, with the two-pixel backward, 30k steps,
every 8th view held out, one run each. PSNR / SSIM of the held-out views:

| Scene | float | bf16 | Training time (float / bf16) |
|---|---|---|---|
| office | 21.82 / 0.909 | 21.19 / 0.912 | 340 / 345 s |
| terrains | 26.20 / 0.897 | 25.77 / 0.896 | 476 / 462 s |
| playground | 19.36 / 0.701 | 19.54 / 0.701 | 743 / 696 s |
| kicker | 23.45 / 0.802 | 22.96 / 0.802 | 414 / 380 s |

- **bf16 is 0.34 dB lower on average.** It is lower on three of the four
  scenes; the SSIM is the same.
- **The spread is as large.** The same float training before the
  two-pixel backward gave 22.31 dB on office and 25.79 on terrains: one
  run moves by about 0.5 dB. The gap lies within that, but three of the
  four scenes point the same way.
- **It stays opt-in.** It saves 5% of the training time on these scenes.

### Per-warp staging in the backward pass

raster_bwd staged each batch of 128 list entries for the whole tile,
between two block barriers. Every warp then walked all 128 and skipped
those whose footprint missed its sub-tiles. Now each warp stages its own
splats (the write-up's step 6, complete):

- **Staging.** The warp's lanes read 32 entries. A ballot keeps those whose
  mask reaches either of the warp's sub-tiles, and their splats go, in
  order, to 32 shared-memory slots of the warp's own.
- **No block barrier.** A warp with little to do does not wait for the
  others.
- **Its own start.** A warp starts from the last entry its own pixels
  blended, not the tile's.

Timed as above (128 steps on the first run's scene, rounds alternating with
the kernels before). A CPU-heavy SfM ran alongside, so these times are a
little higher than in the table above. One round of the old kernels ran at
25.2 ms of stages and is left out. GPU ms per step:

| Backward staging | Adam moments | raster_bwd | Stages | Wall |
|---|---|---|---|---|
| Per tile | float | 5.95–6.30 | 20.6–20.9 | 21.9–22.6 |
| Per warp | float | 4.99–5.40 | 19.4–20.1 | 21.1–22.1 |
| Per tile | bf16 | 6.24–6.38 | 18.9–19.1 | 20.4–20.8 |
| Per warp | bf16 | 5.29–5.44 | 18.2–18.5 | 19.9–20.3 |

- **raster_bwd takes 15% less time**, about 0.9 ms a step. The wall time of
  a step falls by 0.5–1 ms.
- **The same maths.** It is unchanged per pixel, and the gradients change
  only in summation order. A splat that reaches several warps' sub-tiles is
  read by each of them, from the cache.
- **south-building, 30k steps, float moments.** 22.92 dB held out with the
  new kernels, against 23.02 dB with the old, within one run's spread
  (above). Training took 698 and 711 s.

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
| Two pixels per thread, per-warp staging without block barriers (step 6) | In: two pixels per thread in raster_bwd, 128 threads per tile, each warp an 8x8 block ([above](#two-pixels-per-thread-and-bf16-moments)); each warp stages the splats of its own sub-tiles, with no block barrier ([above](#per-warp-staging-in-the-backward-pass)) |

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

### MVS depth, the splat filling its holes

A depth prior moves the splats, and the image pays for it. The MVS depth
can go into the mesh instead.
`gsplat_mesh_cuda --gpu-tsdf --depth-maps mvs/fused_depth` fuses
visloc-mvs's depth (`--save-fused-depth`, resampled to each view)
wherever it has one, and the splat's depth elsewhere.
`--depth-maps-only` fuses the maps alone. The meshes are built from all
views, on the GPU TSDF at 1 cm with depths up to 20 m. The hybrids in this
section weigh the splat's depth as a map's; it now weighs half
([below](#the-splats-depth-at-half-the-weight)). F1 at 2 cm, with accuracy
/ completeness:

| Mesh | Splat | terrains | playground | office | mean | Splat's held-out PSNR |
|---|---|---|---|---|---|---|
| MVS depth only | | 74.4% (93.5 / 61.8) | 50.5% (84.3 / 36.1) | 38.9% (96.4 / 24.3) | 54.6% | |
| MVS + splat (planar median) | plain, every view | 78.5% | **64.7%** (69.7 / 60.3) | 45.0% | 62.7% | not held out |
| MVS + splat (planar median) | plain, every 8th view held out | 79.1% | 64.0% | 43.7% | 62.3% | as the plain splat |
| MVS + splat (expected depth) | prior 30 → 1, held out | 79.5% | 62.9% | 50.6% | 64.3% | −0.79 dB |
| MVS + splat (expected depth) | prior 30, every view | **79.7%** | 62.8% | **54.6%** (62.5 / 48.5) | **65.7%** | (−1.81 dB held out) |
| splat alone (expected depth) | prior 30, every view | 78.1% | 60.4% | 52.3% | 63.6% | (−1.81 dB held out) |
| splat alone (planar median) | plain, every view | 34.1% | 44.5% | 21.2% | 33.3% | |

- **The image need not pay.** With a plain splat, which is trained for
  images only, the hybrid mesh scores 62.7%: 29 points above the plain
  splat's own mesh and 8 above the MVS depth alone. On playground its 64.7%
  beats every mesh trained with a prior.
- **What each side brings.** The MVS depth is accurate but full of holes:
  office's mesh from it alone is 96.4% accurate and 24.3% complete. The splat
  fills the holes, and completeness rises by 9 to 24 points.
- **Where a prior still helps.** office's weak texture leaves MVS few
  depths. There the splat does the most filling, and a splat trained with a
  strong prior fills better (54.6% against 45.0%). Across the three scenes,
  a prior adds 3 points to the hybrid at its price in held-out PSNR.
- **The choice.** For geometry with images untouched, use the hybrid with a
  plain splat. For the best geometry, use the hybrid with a prior-trained
  splat.

#### With 1600 px MVS depth

visloc-mvs on CUDA ([docs/mvs.md](mvs.md#on-cuda)) takes playground and
office to 1600 px as well, in 2 to 3 minutes, where the CPU had taken them
at 800 px in 8 to 13. The same splats, meshed the same way with the CUDA
runs' 1600 px maps (terrains' maps were 1600 px already), score as
follows. The earlier mean is in brackets:

| Mesh | Splat | terrains | playground | office | mean |
|---|---|---|---|---|---|
| MVS depth only | | 74.7% | 48.6% (85.7 / 33.9) | 42.4% (98.9 / 27.0) | 55.2% (54.6%) |
| MVS + splat (planar median) | plain, every view | 79.0% | 64.7% (70.5 / 59.8) | 48.8% | 64.2% (62.7%) |
| MVS + splat (planar median) | plain, every 8th view held out | 79.4% | 64.4% | 48.4% | 64.1% (62.3%) |
| MVS + splat (expected depth) | prior 30 → 1, held out | 79.8% | 64.7% | 55.4% | 66.6% (64.3%) |
| MVS + splat (expected depth) | prior 30, every view | **79.9%** | **65.3%** | **59.0%** (66.6 / 52.9) | **68.1%** (65.7%) |

- **office gains the most.** Its maps alone gain 3.5 points, and its
  hybrid meshes gain 3.8 to 4.8.
- **playground's maps alone lose 1.9 points** (completeness 36.1% →
  33.9%), although its cloud gains 6.4 points and its maps cover about as
  many pixels (32.7% of each view, against 33.7%). With the splat filling
  the holes it gains up to 2.5 points.
- **terrains moves by 0.5 points or less.** Only the rounding of its maps
  changed.
- The prior-trained splats above still learned from the 800 px maps.

Retrained on the 1600 px maps (playground and office; terrains' prior was
1600 px already), with the same settings and meshed from the expected
depth, the prior-trained splats score as follows. The first number in each
cell is from the splats trained on the 800 px maps, meshed the same way:

| Splat | Mesh | playground | office | mean of three |
|---|---|---|---|---|
| prior 30, every view | splat alone | 60.4% → 62.0% | 52.3% → **62.9%** (81.0 / 51.5) | 63.6% → 67.7% |
| prior 30, every view | MVS + splat | 65.3% → 64.5% | 59.0% → **66.9%** (82.8 / 56.1) | 68.1% → **70.5%** |
| prior 30 → 1, held out | splat alone | 51.1% → 52.0% | 42.0% → 54.9% | 54.6% → 59.2% |
| prior 30 → 1, held out | MVS + splat | 64.7% → 64.2% | 55.4% → 64.6% | 66.6% → 69.5% |
| prior 30 → 1, held out | held-out PSNR | 18.51 → 18.51 dB | 21.75 → 20.91 dB | −1.06 dB against the plain splat |

- **office gains about 10 points.** Its 1600 px cloud scores 72.8%
  against 56.8% at 800 px, and the splat follows its prior. The splat
  alone comes within 10 points of that cloud. The hybrid reaches 66.9%.
- **playground does not gain.** The splat alone gains 0.9 and 1.6 points;
  the hybrid moves by 0.8 points or less. Each number is one training run.
- **The fade costs office more image.** Against the plain splat, office's
  held-out PSNR falls by 1.40 dB instead of 0.56. Over the three scenes the
  fade now costs 1.06 dB.

#### Four more scenes

pipes, terrace, kicker and courtyard (14, 23, 31 and 38 images), end to end
the same way: the ground-truth poses and the images at 1600x1067, visloc-mvs
on CUDA at 1600 px (53 to 182 s), a plain splat and one with the prior at 30
(30k steps on every view, 6.5 to 17.5 minutes each), and the same meshes.
F1 at 2 cm; the means also include terrains, playground and office:

| Mesh | pipes | terrace | kicker | mean of three | mean of six |
|---|---|---|---|---|---|
| MVS cloud (`fused.ply`, not a mesh) | 79.2% | 88.5% | 83.6% | 83.8% | 79.5% |
| plain splat alone (planar median) | 16.7% | 36.8% | 36.5% | 30.0% | 31.6% |
| prior-30 splat alone (expected depth) | 69.2% | 84.0% | 75.8% | 76.3% | 72.0% |
| MVS depth only | 68.9% (96.6 / 53.5) | 85.5% | 47.8% (97.0 / 31.7) | 67.4% | 61.3% |
| MVS + plain splat | 67.8% (83.0 / 57.4) | **87.6%** | 72.5% | 76.0% | 70.1% |
| MVS + prior-30 splat | **76.8%** (92.7 / 65.6) | 87.3% | **79.5%** (85.1 / 74.7) | **81.2%** | **75.8%** |

- **The hybrid with the prior-30 splat is the best mesh** in four of the
  six scenes, and within 0.4 points of the plain splat's hybrid on terrace
  and playground.
- **A plain splat can hurt.** On pipes (14 images) its own mesh scores
  16.7%, and filling the MVS depth's holes with it costs more accuracy
  (96.6% → 83.0%) than it adds completeness: 67.8% against 68.9% for the
  MVS depth alone.
- **The MVS depth's mesh keeps much less than its cloud.** On kicker the
  cloud is 79.7% complete, the TSDF mesh of the same depth 31.7%
  (playground: 56.6% and 33.9%). The maps' holes carved the other views'
  surfaces away ([below](#map-holes-are-unknown)).
- **courtyard:** the MVS cloud scores 84.3% (90.6 / 78.8, 182 s). Its 1 cm
  meshes are 1.0 to 1.8 GB, and the evaluator ran out of memory on them in
  this 11 GB WSL2 VM, so courtyard has no mesh scores.

#### Map holes are unknown

kicker's maps hold the information: back-projected as points, its fused
depth scores 82.7% (82.1 / 83.3), more complete than its cloud. The TSDF
lost it in two places.

- **The maps' holes carved free space.** A pixel without a surface carves
  its whole ray as free space, which is right for the splat's depth: 0
  means nothing opaque there. A map's 0 means that the pixel's depth was
  not confirmed by other views, and two thirds to three quarters of each
  view's pixels are 0 (playground, office). Fused alone
  (`--depth-maps-only`), each hole erased the surfaces that other views
  had seen along its ray. A map's hole is now unknown: it adds no surface
  and carves nothing.
- **The component filter dropped real surface.** It removes connected
  pieces under 500 triangles, which are floaters in a splat's mesh. In
  the MVS depth's mesh they are mostly pieces of surface. With
  `--depth-maps`, `gsplat_mesh_cuda`'s `--min-component` now defaults to
  0.

F1 at 2 cm, before → after. The hybrids fill the holes from the splat, so
only the filter changes them:

| Mesh | terrains | playground | office | pipes | terrace | kicker | mean |
|---|---|---|---|---|---|---|---|
| MVS depth only | 74.7 → **81.6%** | 48.6 → **57.9%** | 42.4 → 49.2% | 68.9 → 70.6% | 85.5 → 86.8% | 47.8 → **75.0%** | 61.3 → **70.2%** |
| MVS + plain splat | 79.0 → 79.7% | 64.7 → 65.6% | 48.8 → 52.1% | 67.8 → 68.5% | 87.6 → 87.6% | 72.5 → 75.0% | 70.1 → 71.4% |
| MVS + prior-30 splat | 79.9 → 80.2% | 64.5 → 65.3% | 66.9 → 67.2% | 76.8 → 78.6% | 87.3 → 87.4% | 79.5 → 80.4% | 75.8 → **76.5%** |

- **Each change on its own** (MVS depth only): the filter off with holes
  still carving scores 67.0% on average, and holes unknown with the filter
  on 66.8%. Together they score 70.2%.
- **Holes unknown is not better everywhere.** With the filter off, it
  gains 17.8 points on kicker and 4.3 on terrains, but loses 1.6 on office
  and 2.7 on pipes: there the carving also removed wrong depths.
- **terrains' MVS depth alone now beats both hybrids** (81.6% against
  80.2%). On the other scenes the better hybrid still adds 0.8 (terrace)
  to 18.0 points (office).

#### A geometry copy of the plain splat

A prior moves the splats and costs the images; a plain splat keeps the
images but meshes worse. Both can come from one training run: the plain
splat for the images, and a copy of it fine-tuned briefly on the prior for
the mesh.

```text
gsplat_train_cuda --data data --init scene.ply --steps 3000 --no-densify --holdout 8 \
    --depth-prior mvs_cuda/fused_depth --depth-weight 30 --out geometry.ply
gsplat_mesh_cuda --ply geometry.ply --data data --expected-depth \
    --depth-maps mvs_cuda/fused_depth --gpu-tsdf --voxel 0.01 --max-depth 20 ...
```

- **The copy.** `--init` starts from the plain splat (here the one that
  held out every 8th view). The gaussians are kept as they are, with no
  refine, and brush's learning rates start again from their first values.
- **The meshes.** Every mesh below is made by the same binary from all
  views: the GPU TSDF at 1 cm, depths up to 20 m, the 1600 px maps for the
  hybrids. F1 at 2 cm:

| Mesh | Extra training (office / terrains / playground) | office | terrains | playground | mean |
|---|---|---|---|---|---|
| plain splat alone (planar median) | | 22.6% | 34.3% | 43.5% | 33.4% |
| MVS + plain splat (planar median) | | 52.3% | 80.3% | 65.3% | 66.0% |
| copy, 3k steps, alone | 37 / 58 / 87 s | 46.2% | 71.9% | 53.3% | 57.1% |
| MVS + copy, 3k steps | | 65.2% | **81.0%** | **66.6%** | **71.0%** |
| copy, 10k steps, alone | 112 / 173 / 265 s | 54.6% | 75.9% | 61.2% | 63.9% |
| MVS + copy, 10k steps | | 63.7% | 80.5% | 65.8% | 70.0% |
| prior-30 splat alone (30k steps, every view) | a run of its own | 62.9% | 78.1% | 62.0% | 67.7% |
| MVS + prior-30 splat | | **67.2%** | 80.2% | 65.3% | 70.9% |
| prior 30 → 1 splat alone (held out) | a run of its own | 54.9% | 70.6% | 52.0% | 59.2% |
| MVS + prior 30 → 1 splat | | 65.7% | 80.3% | 66.1% | 70.7% |

- **The copy's hybrid matches the prior-trained splats', and the images
  stay the plain splat's.**
  - **Geometry.** With 3k steps the hybrid scores 71.0% on average,
    against 70.9% for the prior-30 splat and 70.7% for the prior that
    fades.
  - **Images.** They come from the plain splat: 22.31 / 25.79 / 19.27 dB
    held out. The fading prior costs 1.06 dB of that, and the constant one
    more.
  - **Cost.** The copy takes 37 to 87 s, a tenth of the plain splat's
    training (377 / 508 / 799 s). It renders worse than the plain splat,
    20.05 / 23.53 / 18.45 dB held out, but it is only meshed.
- **Where it pays.**
  - office gains 13.0 points over the plain splat's hybrid. Its weak
    texture leaves the MVS depth with holes, and there the copy fills them
    better.
  - On terrains and playground it adds 0.8 and 1.3 points: the plain
    splat already fills their holes about as well.
- **Longer fine-tuning helps the copy alone, not the hybrid.** At 10k steps
  the copy alone scores 63.9% (the prior-30 splat's: 67.7%), and the
  hybrid 70.0%.

**With APD's maps.** visloc-mvs with APD's patches, the strong median and
20 sources ([mvs.md](mvs.md#apds-patches-with-the-strong-median-on-cuda);
`--deformable --deformable-strong-median --sources 20`, 1600 px) fuses
better depth. The meshes above, made again from those maps
(`mvs_apd/fused_depth`):

| Mesh | Maps | office | terrains | playground | mean |
|---|---|---|---|---|---|
| MVS cloud (`fused.ply`, not a mesh) | defaults | 72.8% | 85.1% | 67.7% | 75.2% |
| MVS cloud (`fused.ply`, not a mesh) | APD | 78.0% | 87.9% | 70.7% | 78.9% |
| MVS depth only | defaults | 49.2% | 81.6% | 57.9% | 62.9% |
| MVS depth only | APD | 67.1% | **84.6%** | 64.6% | 72.1% |
| MVS + plain splat | defaults | 52.3% | 80.3% | 65.3% | 66.0% |
| MVS + plain splat | APD | 68.9% | 82.9% | 68.0% | 73.3% |
| MVS + copy (fine-tuned on the default maps) | defaults | 65.2% | 81.0% | 66.6% | 71.0% |
| MVS + copy (fine-tuned on the default maps) | APD | **75.3%** | 83.7% | **70.4%** | **76.5%** |
| MVS + copy fine-tuned on the APD maps | APD | 74.9% | 83.4% | 69.6% | 75.9% |
| copy fine-tuned on the APD maps, alone | | 52.8% | 73.0% | 56.2% | 60.7% |

- **Every mesh gains.** The MVS depth alone gains 9.2 points on average,
  office 17.9: APD's completeness fills the maps' holes. The best hybrid
  reaches 76.5%, 5.5 points above the default maps, with the images still
  the plain splat's.
- **The copy fine-tuned on the default maps does as well.** Fine-tuning it
  on the APD maps instead scores 0.5 points lower in the hybrid (75.9%). It
  matters which maps go into the mesh, not which ones the copy learned.
- **terrains' MVS depth alone beats every hybrid** (84.6% against 83.7%),
  as with the default maps.
- **Cost.** The APD maps take about four times as long: 397, 905 and
  824 s, against 108, 209 and 194 s.

**One splat for both?** The copy renders worse than the plain splat. These
runs tried to keep one splat good for both. Each is 3k steps from the plain
splat on the default maps' prior, without refine. The table gives the
held-out PSNR, then F1 of the hybrid with the APD maps (from the expected
depth; the plain splat's from the planar median):

| Splat | office | terrains | playground | mean |
|---|---|---|---|---|
| plain | 22.31 dB, 68.9% | 25.79 dB, 82.9% | 19.27 dB, 68.0% | 22.46 dB, 73.3% |
| copy, prior 30 (above) | 20.05 dB, 75.3% | 23.53 dB, 83.7% | 18.45 dB, 70.4% | 20.68 dB, 76.5% |
| prior 30, every SH band from the start | 20.38 dB, 74.9% | 23.93 dB, 83.5% | 18.34 dB, 70.5% | 20.88 dB, 76.3% |
| prior 10, every SH band | 20.13 dB, 74.6% | 24.57 dB, 83.5% | 18.62 dB, 70.0% | 21.10 dB, 76.0% |
| prior 3, every SH band | 20.81 dB, 73.1% | 25.04 dB, 83.4% | 18.97 dB, 70.5% | 21.61 dB, 75.7% |

And on office alone:

| Splat | PSNR | Hybrid |
|---|---|---|
| no prior, every SH band | 22.22 dB | 63.5% |
| the copy's colours alone trained 3k / 10k more steps (shapes frozen) | 19.94 / 19.83 dB | 75.3% (the copy's mesh) |
| prior 30 at the final learning rates, no aux losses, 3k / 10k steps | 20.41 / 20.49 dB | 73.1 / 75.6% |

- **The prior costs the images.** Without it, 3k steps keep office at
  22.22 dB. At weight 3 it already loses 1.5 dB.
- **The colours cannot win them back.** The copy's shapes are what the
  mesh needs, and its images suffer from them. Training its colours alone
  lowers the held-out PSNR further (19.9 dB).
- **Neither does the schedule.** The final learning rates do not help,
  and raising the SH degree again from 0 costs 0.36 dB without a prior.
- **So it stays two splats.** Weight 3 is the middle ground: 0.85 dB below
  the plain splat, and 0.8 points below the copy's hybrid.

#### The splat's depth at half the weight

terrains' MVS depth alone beat every hybrid. In the maps' holes the
splat's depth went in with a map's weight. Where it is wrong it adds wrong
surface, and its free space carves away real surface that other views'
maps saw. `gsplat_mesh_cuda --fill-weight` (`DepthMaps::fill_weight`) now
weighs it, for its surface, colour and carving alike. The default is 0.5;
it was 1. Below 1, a voxel that only the splat's depth reaches needs more
views to pass `--min-weight`: two at 0.5. Where a map has a depth too, the
map counts twice as much as the splat.

F1 at 2 cm, weight 1 → 0.5, the meshes of the tables above:

| Mesh | Maps | office | terrains | playground | mean |
|---|---|---|---|---|---|
| MVS + plain splat (planar median) | defaults | 52.3 → 55.4% | 80.3 → 81.7% | 65.3 → 67.1% | 66.0 → 68.0% |
| MVS + copy | defaults | 65.2 → 67.0% | 81.0 → 81.7% | 66.6 → 66.6% | 71.0 → 71.8% |
| MVS + plain splat (planar median) | APD | 68.9 → 71.4% | 82.9 → 84.0% | 68.0 → 68.8% | 73.3 → 74.8% |
| MVS + copy | APD | 75.3 → **75.6%** | 83.7 → 84.3% | 70.4 → 70.2% | 76.5 → **76.7%** |

- **Every mean gains**, by 0.3 to 2.0 points, and so do 10 of the 12
  meshes. The copy's hybrids on playground lose 0.1 and 0.2 points.
- **The weight mostly sets how many views the splat's depth needs.**
  Every weight from 0.5 up to 1 needs two. The copy's hybrid with the APD
  maps scores 76.5% at 0.7, 76.0% at 0.35 (three views) and 73.3% at 0.2
  (five). At 0.2 office and playground lose the holes the splat filled:
  office's completeness falls from 69.4% to 55.8%.
- **terrains** reaches 84.3%, against 84.6% for its MVS depth alone.
- **Tried and not kept:**
  - **The splat's depth only where it has a surface**, so that its empty
    pixels carve nothing. It moved no score by more than 0.04 points: the
    splat covers these views.
  - **Its truncation band only**: no carving and no free space in front
    of it. It raised terrains' completeness from 81.9% to 85.2%, but cost
    office 6.6 points (68.7%) and playground 9.0 (61.4%), with their
    floaters left in.

#### Where terrains' hybrid loses, and why the median depth lags

Two questions left open above. The answers come from the meshes and splats
of the tables above.

**terrains: the splat's depth moves surface the maps already have.** Its
MVS depth alone (84.6%) still beats its best hybrid (84.3%). Both meshes
use the APD maps; the hybrid fills their holes with the geometry copy's
expected depth at weight 0.5. The evaluator wrote each mesh's per-point
results at 2 cm (`--accuracy_cloud_output_path`,
`--completeness_cloud_output_path`). They were compared ground-truth point
by point, and per 1 cm cell, which is how the evaluator averages
completeness:

- **Gains and losses.** The hybrid gains completeness in some cells and
  loses it in others: +1.7 and −2.1 points of the cell mean (83.5% →
  83.1%).
- **Moved, not gone.** Net, the middling cells lose: those with 3 to 10
  ground-truth points go from 85.6% to 84.2%, those with 10 to 30 from
  89.1% to 88.3%. The densest and the sparsest gain. In the cells that
  lose more than half their share, the hybrid has a surface within 4–8 cm
  of 75% of the points.
- **The mechanism.** Where other views' maps already see a surface, the
  splat's depth from one view's hole goes into the same voxels and pulls
  the surface by a few centimetres. Elsewhere it fills real holes.
- **Accuracy.** The surface only the splat adds is 8% of the hybrid's
  points. It is 42% accurate where the scan sees it, but the scan sees only
  5% of it, so accuracy drops little (86.4% → 86.1%).
- **Tried since.** A fill that leaves alone the voxels some map has seen
  does not keep them
  ([below](#tried-and-not-adopted-keeping-the-splat-out-of-the-maps-voxels)).

**Why the median depth lags in a splat trained on a prior.** Each view's
three depths of the plain splat and of its geometry copy were rendered and
compared with the MVS maps the copy was fine-tuned on (`mvs_cuda`), over
the pixels both have. The relative error, its median and the share within
1%:

| Splat | Depth | office | terrains |
|---|---|---|---|
| plain | median | 4.31%, 21.8% | 3.95%, 26.9% |
| plain | planar | 5.64%, 21.9% | 4.45%, 25.9% |
| plain | expected | 14.75%, 4.7% | 36.57%, 2.7% |
| geometry copy | median | 2.09%, 32.1% | 1.13%, 47.1% |
| geometry copy | planar | 3.23%, 25.9% | 2.07%, 35.2% |
| geometry copy | expected | **0.93%, 51.9%** | **0.37%, 76.2%** |

- **The prior trains the expected depth.** The copy's expected depth comes
  within 1% of the maps on 52% and 76% of the pixels.
- **The median depth is one splat's.** It is the camera depth of the splat
  that takes the transmittance to 0.5. Its error is two to three times the
  expected depth's, with no offset: the copy's signed median is within
  ±0.4% on both scenes. The splats lie scattered about the surface, and
  only their blend sits on it.
- **The planar depth does worse still.** It meets the plane of that same
  splat, whose normal is looser than its centre.
- **Without a prior it is the other way round.** The plain splat's expected
  depth sits behind the surface (signed median +2.5% on office, +35% on
  terrains), pulled back by what the surface lets through. There the median
  depth is the better one, as the meshes above found.

#### Tried and not adopted: keeping the splat out of the maps' voxels

The analysis above suggested a fill that keeps the splat's depth out of
the voxels a map has seen. A prototype fused every view's map first,
marked the voxels the maps updated, then fused every view's splat depth
(weight 0.5) into the unmarked ones only. Three variants:

1. **Every voxel a map updated**: its truncation band and the free space
   in front of it.
2. **Only the maps' truncation bands.** The splat's depth may still go
   into a map's free space.
3. **The splat's surface only.** Its truncation band stays out of the
   maps' bands; its free space goes everywhere, as before.

F1 at 2 cm with the APD maps (GPU TSDF at 1 cm, as above; variant 0 is
the default, which gave the same scores as the tables above):

| Splat | Variant | office | terrains | playground | mean |
|---|---|---|---|---|---|
| geometry copy (expected) | 0 | 75.6% | 84.3% | 70.2% | **76.7%** |
| | 1 | 75.2% | 84.3% | 68.5% | 76.0% |
| | 2 | 75.0% | 84.3% | 68.4% | 75.9% |
| | 3 | 75.7% | 84.3% | 70.2% | **76.7%** |
| plain (planar median) | 0 | 71.4% | 84.0% | 68.8% | **74.8%** |
| | 1 | 70.7% | 84.5% | 67.8% | 74.4% |
| | 2 | 69.9% | 84.4% | 66.9% | 73.7% |
| | 3 | 71.4% | 84.0% | 68.8% | **74.8%** |

- **The splat's free space keeps the hybrids accurate.** Variants 1 and 2
  also keep it out of the maps' voxels. Completeness rises by 2.4 to 3.2
  points on the geometry copy, but accuracy falls by 2.4 to 6.2: office
  85.4% → 79.9%, playground 71.4% → 65.6% (variant 1). The splat's free
  space carved away the maps' wrong surface there.
- **Its surface in the maps' voxels costs nothing.** Variant 3 keeps the
  free space and drops only that surface. No score moves by more than
  0.05 points.
- **terrains stays below its MVS depth alone** (84.6%). Only the plain
  splat with variant 1 comes close (84.5%), and it loses on the other two
  scenes. So the moved surface found above does not set terrains' F1.
- The prototype (`gsplat_mesh_cuda --fill-unseen`) is not merged.

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
- **Lenses.** Voxels project, and pixels back-project, through each
  view's lens, fisheyes included, as on the CPU (see
  [fisheye.md](fisheye.md)).

`tests/tsdf.rs`:

- **One view.** Both allocate before fusing, so the meshes match:
  - every vertex within 4e-4 voxels;
  - colours within 1 of 255;
  - the same triangles.
- **Four views.** The surface is bit-identical whether it is fused in 1 slab
  or 30, with or without the view cache. 99.8% of its vertices lie within
  0.01 voxels of the CPU's.
- **Through lenses.** One view through a distorted pinhole and a fisheye
  gives the CPU's mesh, 99.97% or more of the vertices within 1e-3 voxels.

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

1. **Geometry.** Fusing the MVS depth and letting a plain splat fill its
   holes reaches 64.2% with the image untouched, from 1600 px MVS on CUDA,
   and 70.5% with a splat trained on that depth as its prior. Meshing the
   expected depth with a strong depth prior
   brings the mesh within a few points of its MVS prior, at 1.8 dB of
   held-out PSNR; fading the prior halves that cost and keeps most of the
   gain.
   - **Image and mesh together.** A copy of the plain splat, fine-tuned for
     3k steps on the prior (a tenth of the training), gives the hybrid mesh
     of the prior-trained splats (71.0% over three scenes), while the
     images stay the plain splat's
     ([a geometry copy](#a-geometry-copy-of-the-plain-splat)).
   - **Answered.** One splat for both costs the images (the prior itself
     does). The median depth lags because it is one splat's depth, while
     the prior trains their blend
     ([where terrains' hybrid loses](#where-terrains-hybrid-loses-and-why-the-median-depth-lags)).
   - **Tried and not adopted.** A fill that leaves alone the voxels a map
     has seen: the splat's free space is what keeps the hybrids accurate
     ([keeping the splat out of the maps' voxels](#tried-and-not-adopted-keeping-the-splat-out-of-the-maps-voxels)).
2. **Speed.** Per-warp staging takes raster_bwd from about 6 ms to 5 ms
   ([per-warp staging](#per-warp-staging-in-the-backward-pass)); the opt-in
   bf16 moments save 1.9 ms more ([two pixels per thread, and bf16
   moments](#two-pixels-per-thread-and-bf16-moments)), and stay opt-in
   after four ETH3D scenes. Open: the write-up's step 5 (branchless
   predication) for raster_bwd.

   The two-stage sort, a vectorized Adam and dropping the gradient memset
   each gained under 1% ([tried](#tried-and-not-adopted)).
