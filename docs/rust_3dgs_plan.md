# Rust 3D Gaussian Splatting: Renderer + Trainer Plan

Goal: take a visloc-rs SfM reconstruction (COLMAP model) and produce a 3D
Gaussian Splatting scene that a **pure-Rust real-time renderer** can display,
with the **training implemented in Rust** rather than the current
Python/torch/gsplat path. Browser (WebGPU/wasm) is a desired secondary target,
but must not impose constraints that slow the native path.

This document records the decision, the concrete architecture, and the staged
plan. It is a plan, not a claim.

## Where we start from

visloc-rs already produces the *input* to a 3DGS optimizer:

- `write_colmap_reconstruction_for_3dgs` (`crates/io/src/colmap/mod.rs:523`)
  writes a COLMAP model with genuine multi-view `TRACK[]` tails.
- EuRoC V2_03 (Vicon room orbit, frames 0-150) already renders **crisp** from
  visloc-rs's own poses: reprojection 0.53 px, downstream l1 ~= 0.006
  (`docs/euroc_sfm_benchmark.md:53`).
- A real `.splat` (antimatter15 32-byte) and a WebGL viewer already exist under
  `docs/euroc_splat/`.

What does not exist is a Rust trainer or a Rust renderer. Today those are
`scripts/gsplat_mcmc_train.py` (Python + torch + gsplat, CUDA) and
`scripts/render_splat_flythrough.py` / the WebGL viewer.

## Survey of the ecosystem (2026)

| Project | Lang | Render | Train | GPU backend | Browser | License | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [brush](https://github.com/ArthurBrussee/brush) | Rust | yes | **yes** | wgpu + CUDA (Burn/CubeCL) | yes (wasm) | Apache-2.0 | 5.0k star, v0.3.0, active |
| bevy_gaussian_splatting | Rust | yes | no | wgpu | yes | MIT/Apache | 276 star, active |
| web-splat | Rust | yes | no | wgpu | yes | Apache-2.0 | 297 star |
| wgpu-3dgs-viewer | Rust | yes | no | wgpu 30 | yes | see repo | v0.8.0 |
| gsplat (nerfstudio) | CUDA/Python | yes | yes | CUDA | no | Apache-2.0 | reference |
| candle | Rust | - | custom op | CUDA only | CPU wasm only | MIT/Apache | maintenance-only |
| burn + cubecl | Rust | - | yes (custom op) | wgpu / CUDA | yes | MIT/Apache | active |

Key findings that shape the decision:

1. **brush is the only mature pure-Rust 3DGS trainer**, and it uses **Burn +
   CubeCL custom kernels** — one kernel language that compiles to both wgpu and
   CUDA. This is the proven architecture, not speculative.
2. **brush pins Burn to git `main`** (not a released version) and uses a
   **forked wgpu** for WebGPU subgroup support. Depending on released
   crates.io Burn would not reproduce the browser path.
3. **WebGPU reached Baseline in January 2026** (Chrome 113+ since 2023, Safari
   26, Firefox 141+ on Windows). But core WebGPU has **no subgroup ops and no
   reliable global atomics**, so any GPU sort must be a hierarchical radix sort.
   In-browser *training* is proven but niche and Chrome-only.
4. A full forward + **analytic backward** differentiable rasterizer is the hard
   part. brush's renderer alone is ~8k lines (forward + backward kernels, sort,
   scan, camera models), plus ~1.7k lines of training loop.

## Decision

**Self-implement on wgpu + Burn + CubeCL, in a separate workspace crate.**
Do not depend on brush (it would forfeit control of the training loop, the
visloc-rs integration, and the benchmark surface, and it drags in a forked
wgpu). Read brush as the reference implementation.

Constraints inherited from visloc-rs architecture (`docs/decisions.md`):

- **`visloc-core` must not depend on CUDA, torch, wgpu, or Burn.** The 3DGS
  crates are separate workspace members, exactly like `visloc-basalt`.
- The SfM side stays the existing `write_colmap_reconstruction_for_3dgs`; the
  new crate consumes a COLMAP model directory. No SfM changes are required for
  stage 0.
- Default builds stay lightweight; the 3DGS crates are opt-in workspace members
  and are not pulled in by `visloc-rs`'s default features.

### Library vs. brush vs. our own (why not just use brush)

- Control: we want to drive training from visloc-rs SfM output and measure it
  in our own benchmark registry.
- Dependency weight: brush's fork of wgpu is a hard fork; adopting it couples us
  to their WebGPU workarounds.
- Learning: reimplementing the backward pass is the point of "学習もRust移植".

## Architecture

```
visloc SfM  --(write_colmap_reconstruction_for_3dgs)-->  COLMAP model dir
                                                              |
                        +-------------------------------------+
                        v
   crates/gsplat-core   (plain Rust: COLMAP model parse, SH math, camera,
                         gaussian parameter types, .ply/.splat IO, losses)
                        |
        +---------------+----------------+
        v                                v
 crates/gsplat-render              crates/gsplat-train
 (wgpu compute rasterizer,         (Burn/CubeCL autodiff trainer:
  forward; then backward)           init from SfM points, optimize,
                                    densify/prune, export)
        |                                |
        +---------------+----------------+
                        v
                apps/gsplat-cli / examples/*
                (train, render flythrough, export .ply/.splat)
```

Staged so each stage is useful on its own:

### Stage 0 — IO + preview (no GPU training yet) — IMPLEMENTED
- `crates/gsplat-core`: COLMAP model → scene seed (via `visloc-io`), gaussian
  parameter types, SH degree-3 evaluation, `.splat` and Inria `.ply` read/write,
  camera projection.
- A **CPU reference renderer** (tiny, slow, plane-scan rasterizer) used as the
  ground truth for the GPU kernels.
- Deliverable: `gsplat_cpu_render` example loads a real `.splat` (422,860
  gaussians from `docs/euroc_splat/`) and renders a frame; a COLMAP model
  (`--colmap`) renders its sparse cloud from a registered camera. Round-trip
  `.splat`/`.ply` pinned by tests.

### Stage 1 — wgpu forward renderer — IMPLEMENTED
- `crates/gsplat-render`: wgpu 30 compute rasterizer. Five dispatches run the
  whole forward pass on the GPU — `project_forward` (project + compact visible
  gaussians + tile counts), `project_visible` (SH colour + projected splats),
  `map_gaussians` (tile expansion), `get_tile_offsets` (per-tile ranges), and
  `rasterize` (one workgroup per 16x16 tile, front-to-back transmittance).
- All shaders use **only baseline WebGPU**: `u32` storage atomics, workgroup
  barriers, no subgroups. This keeps the wasm path open.
- The CPU reference in `gsplat-core` was corrected to the same physically
  correct front-to-back transmittance compositing (it previously applied an
  "over" operator in scene order), so GPU and CPU agree by construction.
- Deliverable met: `gsplat_gpu_render` renders a real 422,860-gaussian
  `.splat` on the GPU and writes a PNG.

**Measured.** Synthetic 3-gaussian scene at 64x64: GPU vs CPU **max abs error
0.006, mean 0.000014, no pixel over 0.1** (pinned by
`gpu_matches_cpu_reference`). Real `euroc_v101.splat` (418k gaussians after
outlier prune) at 640x480: mean abs error 0.003 on a 1/104 subsample; the
residual is faint boundary floaters. Native forward timing on a GTX 1660 Ti:
~79 ms/frame at 640x480, ~119 ms/frame at 1920x1080 for 419k gaussians.

**Device-side sort + scan (implemented).** The two per-frame sorts and the
compact-order tile-count prefix sum now run on the GPU:

- `shaders/radix.wgsl` — subgroup-free LSD radix sort (4 bits/pass, 8 passes),
  with a **stable** in-block per-digit exclusive scan so LSD correctness holds.
  Per-pass: `radix_histogram` → `radix_scan` (O(BINS·blocks)) → `radix_scatter`.
  A dedicated test (`gpu_radix_sort_is_correct_and_stable`) checks 5,000 keys
  with duplicates against a stable host sort.
- `shaders/scan.wgsl` — subgroup-free blocked inclusive scan, used for the
  compact-order tile counts that `map_gaussians` consumes.

This removes every large GPU→CPU readback from the frame (depth array, id
array, counts array, both isect arrays); only the 8-byte `num_visible` /
`num_intersections` counters and the output image still come back.
`gpu_matches_cpu_reference` still passes on the fully device-side pipeline.

**Honest performance.** The end-to-end PNG path is ~80 ms at 640x480 and
~114 ms at 1920x1080 for 419k gaussians — but almost all of that is the
**output-image readback** (~90 ms at 1920x1080), not the render. Timing the
GPU-only path (no image readback, which is what a viewer pays) gives:

| resolution | GPU-only | full PNG path |
| --- | --- | --- |
| 640x480 | 49 ms | 80 ms |
| 1920x1080 | 26 ms | 114 ms |

So the renderer is already near-real-time (~40 fps at 1080p for 419k
gaussians); the counter readback is only ~0.4 ms. Two costs remain:

1. The **output-image readback** (~90 ms at 1080p), which a viewer avoids by
   presenting the output buffer directly.
2. ~~CPU submit overhead~~ — **falsified**, see below.

`Renderer::set_skip_readback(true)` exposes the GPU-only path for measurement.

**Per-stage profile and two correctness fixes.** `GSPLAT_PROFILE=1` now makes
`render` block after every stage and print per-stage wall time, the raw
counters, and per-tile list-length statistics. What it showed (GTX 1660 Ti,
419k gaussians):

- *Submit overhead is not the cost.* Folding each radix sort into one submit
  (24 → 2 submits/frame) left GPU-only time unchanged (640x480: 48.6 → 47.7
  ms; 1080p within noise).
- *The auto-framed benchmark view is unrepresentative.* It puts the camera far
  outside the scene, so ~57% of visible gaussians land in a single tile and
  `rasterize` (one workgroup per tile) is bound by that one tile. The example
  now takes `--pose-cw qw,qx,qy,qz,tx,ty,tz` (world-to-camera); a V1_01
  ground-truth cam0 pose (`T_WC = T_WB · T_BS`) renders the room correctly,
  confirming `euroc_v101.splat` is in the EuRoC GT frame.
- *Bug: prefix-scan race.* `scan_apply` had every workgroup rewrite
  `block_sums` in place, racing across workgroups, so `cum_tiles_hit` was wrong
  whenever the scan spanned more than one 2048-element block and isects landed
  in other gaussians' slots (blocky tiles). The block-sum scan is now its own
  single-workgroup kernel; `gpu_prefix_scan_matches_host_across_many_blocks`
  fails on the old code at element 2048.
- *Bug: footprint cap.* CPU and GPU both capped the projected radius at
  `2 * max(w, h)`, cutting large near-camera gaussians off at a rectangle (hard
  tile-aligned edges). Both now use the Inria 1.3x-frustum Jacobian clamp and
  no cap. Real-pose parity (1/104 subsample): max error 0.63 → **0.0077**,
  pixels over 0.1: 5798 → **0**.

From the real pose (640x480, 5.3M intersections) the frame is sort-bound:

| stage | ms |
| --- | --- |
| tile_sort | 17.2 |
| tile_offsets | 6.4 |
| map_gaussians | 4.3 |
| rasterize | 3.6 |
| depth_sort | 2.8 |
| rest | 2.5 |
| **total** | **36.8** |

The tile sort now runs only `ceil(log2(num_tiles) / 4)` radix passes (tile ids
are small; the LSD sort is stable so depth order survives): 3 passes at
640x480, 4 at 1080p, versus 8. `rasterize` also stops loading batches once
every pixel in its tile is saturated. Next targets: the tile sort and
`tile_offsets` on real views.

**Sort, offsets and capacity (after the DXC switch).** Re-profiled with DXC
(which made the old tile sort *slower*, 17 → 24.6 ms), then:

- `radix_scan` was 16 threads each walking every block serially; it is now a
  chunked parallel scan of the digit-major histogram (tile_sort 24.6 → 17.7).
- 16 keys per radix thread instead of 4 amortises the in-block digit scan
  (17.7 → 10.8; 32 spills registers and doubles it).
- `get_tile_offsets` did an atomicMin/atomicMax per isect on the tile's two
  counters; the list is sorted, so only run boundaries now write, with plain
  stores (6.4 → 0.6 ms), dispatched in 2D past the 65535-workgroup limit.
- Isect buffers were sized once at `64 * n` (26.8M for this scene) and
  silently truncated beyond it; a 1080p view from inside the room needs 31M,
  and its `tile_offsets` dispatch then exceeded 65535 workgroups and panicked
  (also before this branch). The renderer now grows the isect buffers, the tile
  sort scratch and the affected bind groups on demand after reading the
  counters. `projected_splats` is per visible gaussian and now sized by `n`.

V1_01 GT pose, GTX 1660 Ti, GPU-only per frame:

| resolution | isects | before (FXC build) | now |
| --- | --- | --- | --- |
| 640x480 | 5.3M | 41 ms | **21 ms** |
| 1920x1080 | 31M | crash | 117 ms |

At 1080p the frame was dominated by the tile sort (72 ms), `map_gaussians`
(22 ms) and `rasterize` (19 ms) over 31M intersections: large near-camera
gaussians each cover hundreds of tiles.

**Tight tile extents.** A splat's tiles came from a 3-sigma circle of its major
axis. `rasterize` only blends pixels where `opacity * exp(-sigma) >= 1/255`, so
the footprint that matters is the ellipse `d^T C^-1 d <= 2 ln(255 * opacity)`
(3.33 sigma at full opacity, smaller for faint splats, empty below 1/255), and
its per-axis bounding box `sqrt(k * C00) x sqrt(k * C11)` is much tighter than
the circle for elongated splats. CPU reference and GPU use the same extent.

| resolution | isects | before | after |
| --- | --- | --- | --- |
| 640x480 | 5.3M → 2.8M | 21 ms | **12 ms** |
| 1920x1080 | 31M → 14.8M | 117 ms | **54 ms** |

Output vs. the 3-sigma circle: max 4/255, mean 0.1/255 (8-bit); the only
differences are tails the circle used to cut above the 1/255 cutoff. Real-pose
CPU/GPU parity is now max 1e-5.

**Exact tile coverage.** The bounding box still lists corner tiles of large
or diagonal splats that hold no blendable pixel. The footprint ellipse is
convex, so its tiles in each tile row are contiguous with an analytic
x-extent (the concave right boundary peaks at the ellipse's rightmost point
clamped to the row's band; mirror for the left), snapped to pixel centres.
`project_forward` counts and `map_gaussians` writes per-row spans with the same
function, so they agree, and only tiles without a pixel above the cutoff are
dropped: renders are **bit-identical** to the bounding-box version. (A per-tile
test gave the same list but cost 6.7 ms in `project_forward` at 1080p.)

| resolution | isects | bbox | exact rows |
| --- | --- | --- | --- |
| 640x480 | 2.8M → 1.9M | 12 ms | **9.0 ms** |
| 1920x1080 | 14.8M → 8.8M | 54 ms | **33.5 ms** |

At 1080p the tile sort is now ~22 of 36 ms (4 passes over 8.8M pairs), then
`map_gaussians` (6 ms, per-thread imbalance for frame-sized splats). Skipping
one radix kernel at a time attributes the tile sort as: `radix_scatter` ~16 ms
(4 ms/pass), `radix_histogram` ~6.4, `radix_scan` ~2.6, `radix_clear` ~1.3,
copies ~1.7. Staging the scatter through shared memory so the global writes
coalesce gave **no gain** (33.5 → 35 ms, reverted): the cost is the in-block
ranking (16-digit Hillis-Steele over 256 threads, runtime-indexed register
arrays that spill), not the stores.

**Cheaper radix kernels.**
- `radix_scatter` now packs `(digit << 12 | index)` per element in shared
  memory and ranks with four stable 1-bit splits (one 256-wide scan of zero
  counts each), then writes each digit run contiguously, gathering key/value
  from the original index.
- `radix_histogram` packs per-thread counts 8 bits per digit in a `vec4`,
  combines them with shared (not global) atomics and *stores* the block's row,
  so the `radix_clear` pass is gone.

| resolution | tile_sort | frame (GPU-only) |
| --- | --- | --- |
| 1920x1080 | ~22 → 16.4 (scatter) → **11.5 ms** | 33.5 → 28 → **22.9 ms** |
| 640x480 | ~4.5 → 2.7 ms | 9.0 → 8.0 → **6.7 ms** |

Renders stay bit-identical. From `main` before this series the 1080p real view
went 117 → 22.9 ms (~5x) and 640x480 21 → 6.7 ms. Remaining at 1080p: tile sort
11.5, `map_gaussians` 5.9 (one thread writes every tile of a frame-sized splat),
`rasterize` 3.9.

**DX12 startup.** `Renderer::new` used to take ~10 minutes on Windows: wgpu's
`Auto` shader-compiler choice falls back to FXC when `dxcompiler.dll` is not
on the DLL search path, and FXC is extremely slow on these compute shaders.
`GpuContext` now finds `dxcompiler.dll` on `PATH` or in the newest installed
Windows SDK and loads DXC explicitly (`WGPU_DX12_COMPILER` still overrides).
GPU test suite 145 s → 3.7 s; `gsplat_gpu_render` end to end ~590 s → 11 s.

## Trainer goal (decided 2026-09-23)

Compete with brush as a Rust 3DGS **trainer**, differentiated by:

1. **COLMAP-free, all-Rust pipeline**: images (+IMU) → visloc-rs SfM/VIO poses →
   trained splat in one command. brush expects a COLMAP/nerfstudio dataset.
2. **Released crates only**: stock wgpu from crates.io, no git-pinned Burn, no
   wgpu fork (brush needs both).

Success bar: eval PSNR within 0.5 dB of brush at equal steps, training time
<= brush on the same GPU (GTX 1660 Ti), and an EuRoC sequence → splat with no
Python or COLMAP.

Milestones: **M0** baselines (brush on the COLMAP sample scenes, scored by our
own evaluator) → **M1** differentiable renderer → **M2** trainer → **M3**
end-to-end CLI.

**Evaluation protocol.** Undistort to PINHOLE with `colmap image_undistorter
--max_image_size 1024`, sort views by image name, hold out every 8th (index
0, 8, ...: brush's and the Inria split), train on the rest, and score every
method's exported Inria `.ply` with `visloc-gsplat-train`'s `gsplat_eval`
(one renderer, black background, PSNR on display-space RGB). brush prints no
metrics without its viewer, and scoring all methods with one evaluator keeps
the comparison fair anyway.

### Stage 2 (M1) — differentiable rasterizer on wgpu

Decision: hand-written WGSL backward next to the existing wgpu forward, **not**
Burn + CubeCL (that would bring back the git-pinned Burn this project is
differentiating against). Structure:

- **CPU reference backward** in `gsplat-core`, checked against central finite
  differences for every parameter group (mean, log-scale, quaternion, opacity
  logit, SH). It is the oracle for the GPU kernels.
- **Forward residuals**: the forward pass also stores each pixel's final
  transmittance and last contributing list index, so the backward walk can
  start there and recover `T_i = T_{i+1} / (1 - alpha_i)` back to front.
- **`rasterize_backward`** (one workgroup per tile, one thread per pixel):
  per-pixel gradients of each splat's `(mean2d, conic, opacity, colour)` are
  summed over the tile with `subgroupAdd` (native `SUBGROUP`; DX12 + DXC has it)
  and a shared-memory CAS float add across the 8 subgroups (no float atomics in
  wgpu on this GPU). Each (tile, splat) total goes to its own slot — the tile
  sort carries the original isect index — so there are **no global atomics**
  and a gaussian's gradient is the deterministic sum of its contiguous isect
  range. A baseline-WebGPU fallback (no subgroups) can come later.
- **`project_backward`**: per gaussian, chain `(mean2d, conic, colour, opacity)`
  gradients through the EWA projection, the covariance and the SH basis to the
  parameters.
- Deliverable: GPU gradients match the CPU reference (and FD) on synthetic
  scenes and on a real-scene subsample.

### Stage 3 (M2) — Trainer
- Init from the SfM points; L1 + 0.2 (1 - SSIM) loss with its image gradient on
  the GPU; Adam on the GPU; densify/prune (brush-style growth) or MCMC.
- Deliverable: train on the M0 scenes and land within 0.5 dB of brush at equal
  steps.

### M3 — End-to-end CLI
- `gsplat-cli train` from a COLMAP model, and from raw EuRoC via visloc-rs
  SfM/VIO poses (no COLMAP, no Python), exporting `.ply` / `.splat`.

### Status (2026-09-23)

- **M0**: `visloc-gsplat-train` with the brush-compatible evaluator
  (`gsplat_eval`: PSNR + SSIM, black background, every 8th view held out).
  brush v0.3.0 baselines on the undistorted COLMAP sample scenes (1024 px,
  30k steps, GTX 1660 Ti): south-building 21.68 dB (7k: 21.57) in 2379 s,
  1.05M splats; gerrard-hall in 2147 s. Getting there fixed three bugs:
  a gamma-2.2 encode in `Image::to_rgb8` (all PNGs were washed out), a PLY
  reader that assumed contiguous properties (brush writes them sorted), and
  `sh_rest_coeffs_per_channel` returning 3x the per-channel count (every
  standard degree 1-3 PLY was rejected and the GPU read wrong SH for
  degree > 0).
- **M1**: done. CPU f64 oracle (`gsplat_core::backward`, FD-checked); GPU
  backward (`Renderer::backward`) matches it to < 1e-6 relative on degree 2
  and 3 scenes.
- **M2**: on-device trainer (`trainer::Trainer`): L1 + 0.2 D-SSIM (GPU SSIM
  gradient matches the CPU one to 1e-6), Adam with Inria learning rates,
  Inria densification on the host with the Adam state carried across.
- **M2 result** (same evaluator, split, 1024 px, GTX 1660 Ti; our run used
  DXC and atomic backward accumulation):

  | scene | method | steps | PSNR | SSIM | gaussians | train time |
  | --- | --- | --- | --- | --- | --- | --- |
  | south-building | brush 0.3.0 | 30k | 21.684 | 0.7875 | 1.05M | 2379 s |
  | south-building | ours | 30k | 21.672 | 0.7872 | 1.20M | 5319 s |
  | gerrard-hall | brush 0.3.0 | 30k | 19.022 | 0.6934 | 0.83M | 2147 s |
  | gerrard-hall | ours | 30k | 19.686 | 0.7070 | 0.64M | 3533 s |
  | south-building | brush / ours | 7k | 21.574 / 20.696 | 0.726 / 0.685 | | |
  | gerrard-hall | brush / ours | 7k | 19.191 / 19.555 | 0.671 / 0.659 | | |

  Quality bar met (parity on south-building, +0.66 dB on gerrard-hall at
  30k). Speed bar not met: 1.6-2.2x brush's wall time (~190 ms/step at 1.2M
  gaussians after densification stops, plus a host round-trip and renderer
  rebuild every 100 steps while densifying).
- **M2 re-benchmark (2026-09-24)** after on-device densification, the
  backward/SSIM speedups and halving the trainer's GPU memory (the 1.2M
  gaussian run had overflowed the 6 GB card and was being paged by WDDM):
  same protocol, GPU exclusive but thermal-throttling (88 C) and shared
  with a desktop app.

  | scene | method | steps | PSNR | SSIM | gaussians | train time |
  | --- | --- | --- | --- | --- | --- | --- |
  | south-building | brush 0.3.0 | 30k | 21.684 | 0.7875 | 1.05M | 2379 s |
  | south-building | ours | 30k | 22.102 | 0.8009 | 1.20M | 3054 s |
  | south-building | ours (main @ PR #206) | 30k | **22.296** | **0.8001** | 1.19M | **1906 s** |
  | gerrard-hall | brush 0.3.0 | 30k | 19.022 | 0.6934 | 0.83M | 2147 s |
  | gerrard-hall | ours | 30k | 19.579 | 0.7046 | 0.64M | 2344 s |
  | gerrard-hall | ours (main @ PR #206) | 30k | **19.766** | **0.7053** | 0.64M | **1356 s** |

  **Correction (2026-09-25): the brush scores above are wrong** and so is
  the "beats brush" reading of this table. brush's PLYs were scored by
  rendering them with our renderer, whose degree-2/3 SH constants were
  swapped (fixed in the SH-normalisation commit); that mis-coloured every
  view-dependent term of *other* trainers' PLYs (our own training was
  self-consistent) and made brush look 1-6 dB worse. The speed numbers are
  unaffected.
- **Fair benchmark (2026-09-25)**, after that fix: our renderer now renders
  brush's PLY to within 0.02 dB of brush's own eval images (Mip-NeRF 360
  bonsai: 33.018 vs 33.004 via `gsplat_eval --renders-dir`). Same split
  (every 8th image by name), black background, 30k steps, GTX 1660 Ti
  (thermal-throttling); Mip-NeRF 360 at `images_4` for both trainers
  (brush `--max-resolution 1600`, i.e. no downscale), our population
  capped at 2.5M. brush times are from its first runs (a duplicate run
  landed within 6%).

  | scene | ours PSNR / SSIM | brush PSNR / SSIM | ours time | brush time |
  | --- | --- | --- | --- | --- |
  | Mip-NeRF 360 bonsai | 32.43 / 0.955 | **33.02 / 0.960** | 1583 s | **1333 s** |
  | Mip-NeRF 360 room | 32.12 / 0.942 | **32.95 / 0.951** | 1747 s | **1184 s** |
  | Mip-NeRF 360 garden | 27.20 / 0.855 | **27.62 / 0.868** | 3401 s | **3134 s** |
  | south-building | 22.44 / 0.804 | **22.69 / 0.805** | **2205 s** | 2379 s |
  | gerrard-hall | 19.93 / 0.713 | **19.98 / 0.719** | **1444 s** | 2147 s |

  brush has the better quality on all five scenes (by 0.05-0.83 dB; within
  the 0.5 dB bar on three). We are faster on the two COLMAP sample scenes
  and slower on Mip-NeRF 360, so the trainer goal is **not** met yet: the
  open work is quality on the indoor 360 scenes and speed at 2.5M
  gaussians / 1297x840.
- **brush strategy + vec4 Adam (2026-09-26/27)**: `gsplat_train --strategy
  brush` ports brush's refine schedule (prune + multinomial refill + split
  of the top 10% by the max-weight statistic, mean noise). `GSPLAT_TRAIN_PROFILE`
  showed Adam at ~34% of a step at ~1M gaussians, about 3x its bandwidth
  floor. Updating four elements per thread through `vec4` bindings halves it
  (18.5 -> 9.2 ms/step). The protocol is the same as the table above, on
  the same GPU:

  | scene | ours PSNR / SSIM | brush PSNR / SSIM | ours time | brush time |
  | --- | --- | --- | --- | --- |
  | Mip-NeRF 360 bonsai | 32.88 / 0.959 | **33.02 / 0.960** | 1355 s | **1333 s** |
  | Mip-NeRF 360 room | **33.15 / 0.950** | 32.95 / 0.951 | 1278 s | **1184 s** |
  | Mip-NeRF 360 garden | 27.55 / 0.865 | **27.62 / 0.868** | **3130 s** | 3134 s |
  | south-building (2 runs) | 21.89, 22.30 / 0.790, 0.791 | **22.69 / 0.805** | **2156, 2137 s** | 2379 s |
  | gerrard-hall | 19.93 / 0.711 | **19.98 / 0.719** | **1859 s** | 2147 s |

  - **Quality:** every scene is within 0.14 dB of brush except
    south-building (0.39-0.80 dB), and room is 0.20 dB above brush.
  - **Speed:** faster than brush on garden and on both COLMAP scenes;
    within 2% on bonsai and 8% on room.
  - **south-building varies from run to run.** The same settings give
    21.9-22.4 dB: two pre-vec4 runs scored 22.44 and 22.38, and a
    padded-buffer run with the scalar Adam scored 21.98. The gradient
    accumulation uses unordered atomics and the refill sampling depends on
    it, so its single-run numbers are noisy. The vec4 change is not what
    moves them.
  - **Tried and reverted:** per-subgroup shared-memory slots instead of the
    CAS float add in `rasterize_backward`. The batch had to shrink from 128
    to 64 splats to fit, and backward went from 20.8 to 26-28 ms. CAS
    contention was not the bottleneck.
- **Butterfly backward reduction (2026-09-27)**: timing-only ablations of
  `rasterize_backward` put ~43% of the kernel in the per-splat subgroup
  reductions (10 components, each a full `subgroupAdd`, then 10 serial
  shared CAS adds on lane 0). On 32-wide subgroups a transposed xor
  butterfly (`subgroupShuffleXor`, each step halving the values a lane
  carries) needs 16 shuffles and leaves component c on lane 2c, so the
  shared adds also run on 10 lanes at once. rasterize_backward on bonsai at
  ~1M gaussians: 18.49 -> 8.32 ms. Same protocol and GPU as above:

  | scene | ours PSNR / SSIM | brush PSNR / SSIM | ours time | brush time |
  | --- | --- | --- | --- | --- |
  | Mip-NeRF 360 bonsai | 32.88 / 0.959 | **33.02 / 0.960** | **970 s** (was 1355) | 1333 s |
  | Mip-NeRF 360 room | **33.08 / 0.951** | 32.95 / 0.951 | **887 s** (was 1278) | 1184 s |
  | Mip-NeRF 360 garden | 27.57 / 0.865 | **27.62 / 0.868** | **2718 s** (was 3130) | 3134 s |
  | south-building | 22.37 / 0.791 | **22.69 / 0.805** | **1449 s** (was 2137-2156) | 2379 s |
  | gerrard-hall | **20.00** / 0.712 | 19.98 / **0.719** | **1266 s** (was 1859) | 2147 s |

  - **Speed:** faster than brush on all five scenes: 13% on garden, 25-27%
    on bonsai and room, and 39-41% on the COLMAP scenes.
  - **Quality:** unchanged from the vec4 table within run-to-run noise.
    Brush stays ahead by 0.05-0.14 dB on bonsai and garden and by
    0.32 dB on south-building. We are ahead by 0.13 dB on room and
    0.02 dB on gerrard-hall.
  - **Tried and reverted:**
    - Packing the ten components into three `subgroupAdd` calls: no gain
      (18.57 ms), because the cost scales with the component count.
    - Per-lane shared CAS when only a few lanes hit a splat, falling back
      to the reduction above K lanes: 19.7 / 32.8 / 76.7 ms at K = 4 / 8 / 16.
- **brush refine fidelity (2026-09-27)**: splat counts on south-building
  trailed brush's (0.71M vs 0.94M at 14k steps). Reading brush 0.3's
  source turned up three differences:
  - brush counts a view only when the splat was blended into a pixel. The
    port counted every projected splat, which deflated
    max-weight / view-count, so fewer splats crossed the growth threshold.
    **This was the cause:** south-building now grows 0.32M / 0.92M at
    7k / 14k (brush 0.32M / 0.94M).
  - brush samples growth indices without replacement. Porting that
    changed little.
  - brush's auxiliary opacity / scale losses (1e-9 / 1e-8) are now in the
    brush preset. They made no measurable difference.
  - An alpha cap of 0.999 (brush) instead of 0.99 was tried and dropped:
    no effect.

  Same protocol and GPU as above:

  | scene | ours PSNR / SSIM | brush PSNR / SSIM | ours time | brush time |
  | --- | --- | --- | --- | --- |
  | Mip-NeRF 360 bonsai | 32.97 / 0.959 | **33.02 / 0.960** | **1085 s** | 1333 s |
  | Mip-NeRF 360 room | 32.93 / 0.950 | **32.95 / 0.951** | **934 s** | 1184 s |
  | Mip-NeRF 360 garden | 27.60 / 0.866 | **27.62 / 0.868** | **2608 s** | 3134 s |
  | south-building (3 runs) | 21.92, 22.18, 22.33 / 0.800, 0.799, 0.803 | **22.69 / 0.805** | **~1750 s** | 2379 s |
  | gerrard-hall | 19.95 / 0.717 | **19.98 / 0.719** | **1509 s** | 2147 s |

  - **Quality:** within 0.05 dB of brush on four scenes. south-building's
    SSIM gap shrank from 0.014 to 0.005; its PSNR stays noisy
    (21.9-22.3) against a single brush run.
  - **Speed:** faster than brush everywhere (14-30%). We are about 10%
    slower than before the fix because there are more splats.

- **Per-photo appearance (2026-09-27)**: `--appearance` learns a 3x4
  affine colour transform per training photo, applied to the render before
  the loss. It starts at the identity with a weak pull back towards it, and
  held-out views render without it.
  - Why try it: on south-building, fitting such a transform to each
    held-out render at evaluation time gains 0.5 dB, so exposure does vary.
  - Results at 30k steps, also scored with that per-image fit (CC-PSNR):

    | south-building | PSNR | CC-PSNR | SSIM |
    | --- | --- | --- | --- |
    | off (3 runs) | 21.91-22.33 | 22.71-23.11 (mean 22.88) | 0.799-0.803 |
    | `--appearance` (2 runs) | 21.55, 21.77 | 23.01, 23.29 (mean 23.15) | 0.802, 0.801 |
    | brush 0.3 | 22.69 | 23.32 | 0.805 |

  - CC-PSNR gains 0.27 dB and nearly reaches brush. Plain PSNR loses
    0.4 dB, because the splat now matches a canonical exposure rather than
    each held-out photo's.
  - brush still leads after colour correction, so the rest of the
    south-building gap is not exposure.
  - The flag stays opt-in. It is meant for photo sets with visibly
    varying exposure.
- **Aux losses fused into Adam (2026-09-28)**: at about 1M gaussians
  (bonsai), `GSPLAT_TRAIN_PROFILE` showed brush's opacity / scale aux losses
  costing 1.7 ms per step. They ran as two extra passes over every gaussian.
  - They now run inside the Adam kernel, which reads visibility from the
    renderer's per-gaussian tile counts.
  - Step time: 33.8 → 29.3 ms. Bonsai 30k scores 33.05 dB (brush 33.02).
  - Updating the SH block (81% of Adam's traffic) only for gaussians visible
    in the frame took Adam from 7.6 to 5.3 ms. It also cost about 0.1 dB at
    30k (bonsai 32.93, garden 27.49), so it was removed.
- **Mesh extraction (2026-09-27)**: `gsplat_mesh` (feature `gpu`) turns a
  trained `.ply` plus its COLMAP dataset into a vertex-coloured triangle
  mesh without retraining.
  1. `Renderer::render_depth` renders every view's median depth: the camera
     z where transmittance first drops to 0.5. It is a second pass over the
     same tile lists.
  2. The depths are fused into a sparse TSDF of 8^3-voxel blocks.
  3. Surface nets extract the surface, with no lookup tables and quads
     facing free space.

  Pixels with no opaque surface carve free space, and voxels seen from
  fewer than 3 views are dropped. Defaults scale with the camera rig
  (voxel = rig radius / 256, max depth = 2x rig radius).

  | scene | views | render + fuse + extract | triangles |
  | --- | --- | --- | --- |
  | Mip-NeRF 360 bonsai | 292 | 22 s | 0.43M |
  | Mip-NeRF 360 garden | 185 | ~35 s | 3.5M |
  | south-building | 128 | ~30 s | 1.2M (2.6M before carving) |

  - The object scenes (bonsai: tree, pot and table; garden: table legs,
    vase and ball) come out clean.
  - Surfaces carry the usual 3DGS depth noise.
  - On south-building the sky stays as geometry: it is modelled by opaque
    splats that every view agrees on, so carving cannot remove it. The SfM
    support filter below removes it.
  - **SfM support filter (2026-09-27)**: `--support-voxels 10` (default)
    drops triangles farther than 10 voxels from every SfM point. Sky has no
    features, so no triangulated points.
    - south-building: 1.17M -> 0.77M triangles; the sky sheets go, while
      the building, trees and bushes stay.
    - garden: the floating fragments go.
    - bonsai: unchanged.
    - Before the filter, half of south-building's mesh vertices were
      within 4.9 voxels of an SfM point, but 10% were beyond 40.
  - Smoother surfaces need depth / normal regularisation during training;
    see the next entry.
- **Depth-normal consistency loss (2026-09-27)**: `gsplat_train
  --normal-weight W [--normal-start 7000]`, 2DGS / PGSR style.
  - `Renderer::set_geometry(true)` adds two alpha-blended channels: each
    splat's camera-space normal (the axis of its smallest scale, facing the
    camera) and its depth. They go through the geometry variants of
    rasterize / rasterize_backward / project_backward.
  - The variants are selected with `// #if GEO` blocks, so the plain kernels
    compile exactly as before. rasterize_backward still runs in 8.4 ms.
  - The extra 4 components fill the spare slots of the 16-slot butterfly.
  - The loss is `1 - N . n_d`: the rendered normal against the normal of
    the rendered depth map (cross product of central differences).
    Gradients flow to both terms, through the depth side via a gather pass.
  - Geometry gradients match central finite differences wherever the same
    harness on the colour path (CPU-validated) does.

  | bonsai | PSNR | train | mesh dihedral median / >30deg | SfM pts to mesh (median) |
  | --- | --- | --- | --- | --- |
  | 7k, off | 30.13 | 157 s | 8.7 deg / 14.3% | 0.26 vox |
  | 7k, W 0.005 (from 3k) | 29.94 | 167 s | 6.9 deg / 10.5% | 0.23 vox |
  | 7k, W 0.015 | 27.74 | 169 s | 5.9 deg / 8.5% | 0.22 vox |
  | 7k, W 0.05 | 26.81 | 162 s | 5.5 deg / 8.9% | 0.21 vox |
  | 30k, off | 32.88 | 970 s | 10.8 deg / 20.1% | 0.33 vox |
  | 30k, W 0.005 (from 7k) | 32.68 | 1028 s | 7.2 deg / 12.0% | 0.27 vox |

  - W = 0.005 buys a visibly smoother mesh and slightly more accurate
    geometry for -0.2 dB.
  - Stronger weights trade too much PSNR.
  - **Tried and dropped:** a flattening regulariser alone (push the
    smallest scale to 0). It cost 1.3-1.5 dB at 7k and made the mesh
    rougher (10.5 deg): thin splats with unaligned normals.
- **M3**: `gsplat_euroc` example (feature `euroc`): raw EuRoC -> undistort
  -> SIFT -> verified temporal matches -> visloc-rs incremental SfM ->
  trainer, no COLMAP or Python. With `--gpu-sift --gpu-ba` (crates
  `visloc-sift-gpu`, `visloc-ba-gpu`) V1_01 (200 frames, stride 4) goes from
  images to poses in 239 s (SIFT 28 s, batched GPU matching 38 s, parallel
  verification 32 s, SfM 141 s; the CPU path spends 838 s in SfM alone) and
  to a 7k-step splat in 766 s total: 183/200 frames registered, 0.638 px,
  held-out PSNR 29.26 / SSIM 0.946 (CPU-SfM run: 29.03 / 0.945).
- **SfM accuracy (2026-09-24)**: `gsplat_euroc` now reports the Sim(3) ATE of
  the registered cam0 centres against the EuRoC ground truth. The original
  4000-keypoint / window-5 matching broke the monocular scale on V1_01
  (73.6 cm ATE, a 2.5x scale jump). New defaults, cheap with GPU matching and
  BA: 8000 keypoints, window 10, skips 15..120, and a keyframe gate that
  leaves out frames with < 2 px accumulated motion (parallax-free frames
  before take-off were registered up to 1.5 m off). 200 frames, stride 4:

  | sequence | old defaults: registered / ATE | new defaults: registered / ATE |
  | --- | --- | --- |
  | V1_01 | 183 / 73.6 cm | 179 / 3.81 cm |
  | MH_01 | 75 / 0.75 cm | 104 / 0.46 cm |
  | V1_02 | 91 / 8.71 cm | 182 / 8.13 cm |
  | V2_01 | 129 / 2.64 cm | 184 / 5.48 cm |

  (Old-default ATE covers only the frames that registered.) More BA
  iterations (50) were not a consistent win. V1_01 end to end with the new
  defaults: 514 s, ATE 3.81 cm, PSNR 28.87 / SSIM 0.955 (23 views); with
  stride 2 / 400 frames: 726 s, ATE 5.55 cm, PSNR 29.98 / SSIM 0.955 (45
  views) -- more views help the splat, shorter baselines hurt the ATE.
- **CPU training (2026-11-21)**: without a GPU the trainer runs on lavapipe,
  where `rasterize_backward`'s subgroup reductions and atomics were 77% of
  a step. On a CPU adapter the renderer now runs both rasterizers on the
  host with rayon: TUM RGB-D pioneer (600 x 440, 7k steps) in 46 minutes
  instead of 2 h 11 min, at the same PSNR (24.66 / 24.61 dB); with the
  sorts and SSIM also on the host, 31.5 minutes (24.58 dB). Profile, settings and checks in
  [gsplat_cpu_training.md](gsplat_cpu_training.md).

### Stage 4 — Browser (optional, non-blocking)
- wasm + WebGPU build of the renderer; training only if stage 2/3 land cleanly.
- Explicitly deferred so it cannot slow the native path.

## Benchmarking

Every claim recorded in `benchmarks/registry` with: commit, cargo lock hash,
cargo/rustc, model hash, dataset identity, command, and metric implementation.
Report against the existing Python gsplat baseline on EuRoC V2_03:
train wall-clock, PSNR/l1, gaussian count, render ms/frame at 752x480, and
peak GPU memory.

## Risks

- **Analytic backward correctness** — highest risk. Mitigation: CPU reference +
  finite-difference gradient checks before any training run.
- **WebGPU sort without global atomics** — mitigation: hierarchical radix sort
  (brush-sort reference), native-only for v1.
- **Burn is pre-1.0 and brush tracks git main** — pin an exact Burn/CubeCL
  revision; isolate behind our own crate so a Burn bump is contained.
- **Scope** — ~8k lines of renderer + ~1.7k of training in the reference. Staged
  so stage 0/1 deliver value even if stage 2 stalls.

## Non-goals for v1

- Feed-forward (inference) splatting.
- 2DGS / 4DGS.
- Real-time performance on wasm.
- Replacing the Python trainer in CI.
