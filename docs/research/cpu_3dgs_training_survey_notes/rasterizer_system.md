# Rasterizer- and system-level speedups for 3DGS training (2024-2026), and how each transfers to a CPU trainer

CPU context used for the transfer judgements: a pure Rust trainer using rayon over 16x16 tiles on 4 vCPUs with AVX2 8-lane half-rows. Per step, forward raster takes 37 ms, backward raster 61 ms (walked back to front with per-splat float adds), L1+SSIM 13 ms and Adam 35 ms, out of about 175 ms. PennyGS (arXiv 2611.27525) is already known and is excluded here.

Unless noted otherwise, all numbers below come from the primary arXiv HTML, PDF or project pages, fetched on 2026-11-26.

## Q1. Faster-GS (arXiv 2602.09999, Hahlbohm, Franke, Eisemann, Magnor, Feb 2026): the listed optimizations and their individual effects

### Takeaway
Faster-GS reaches about 3-5x end-to-end over 3DGS (RTX 4090) at equal PSNR.

When each optimization is added to the same baseline on its own:
- Per-Gaussian backward is by far the largest win: 1.20-1.84x on its own.
- Fused Adam gives 1.12-1.33x.
- Each culling, AABB or SH-layout tweak gives only 1.00-1.09x.
- Z-ordering (spatial reordering of the primitive array) adds about 1-6% on top of everything else.

Fusing the optimizer into backward and skipping invisible Gaussians gives a further 1.8x on outdoor scenes, at -0.13 dB.

### Cited Findings
- **Setup.** Ablations and the main table use an RTX 4090. The GPU comparison also covers RTX 3090 and RTX 5090. The benchmark is 13 scenes from Mip-NeRF360 (split into outdoor and indoor), Tanks and Temples (T&T) and Deep Blending. The baseline is official 3DGS. — [Faster-GS HTML](https://arxiv.org/html/2602.09999v1)
- **Table 2 ablation** (training time on outdoor / indoor / T&T / DB, with the speedup factor against the "Basis" row). Each row's time is not cumulative: the per-Gaussian-backward row is 7m52s indoor, while the fused-Adam row after it is 13m40s. Each row therefore appears to be a single addition to Basis. This is my reading of the table, not a statement in the paper. — [Faster-GS HTML](https://arxiv.org/html/2602.09999v1)

  | Row | Outdoor | Indoor | T&T | DB |
  |---|---|---|---|---|
  | Basis | 17m07s | 14m29s | 9m39s | 17m15s |
  | + fused activations | 1.05x | 1.03x | 1.04x | 1.05x |
  | + separate SH buffers (SH band 0 split from higher bands) | 1.09x | 1.02x | 1.05x | 1.04x |
  | + rectangular AABBs | 1.02x | 1.04x | 1.04x | 1.03x |
  | + rectangular AABBs with opacity-aware extent | 1.02x | 1.05x | 1.05x | 1.03x |
  | + load-balanced instancing | about 1.00x | 1.01x | 1.01x | 1.01x |
  | + full tile-based culling | 1.01x | 1.06x | 1.05x | 1.03x |
  | + separate sorting | 1.01x | 1.03x | 1.01x | 1.02x |
  | **+ per-Gaussian backward** | **14m14s (1.20x)** | **7m52s (1.84x)** | **6m56s (1.39x)** | **10m27s (1.65x)** |
  | + fused Adam (PyTorch) | 1.22x | 1.06x | 1.11x | 1.12x |
  | + fused Adam (Apex) | 1.30x | 1.12x | 1.20x | 1.18x |
  | + fused Adam (Faster-GS's own) | 1.33x | 1.12x | 1.22x | 1.19x |
  | **Full, without z-ordering** | **5m52s (2.92x)** | 3m17s (4.41x) | 3m11s (3.02x) | 3m50s (4.50x) |
  | **Full** | **5m31s (3.10x)** | 3m14s (4.47x) | 3m04s (3.14x) | 3m46s (4.58x) |
- **VRAM.** Per-Gaussian backward raises VRAM (outdoor: 6.39 GiB to 7.69 GiB). Separate SH buffers lower it (outdoor: 6.39 GiB to 5.72 GiB). — [Faster-GS HTML](https://arxiv.org/html/2602.09999v1)
- **Table 3, optimizer fused into backward** (outdoor, PSNR / time / VRAM / #Gaussians). — [Faster-GS HTML](https://arxiv.org/html/2602.09999v1)
  - Full: 24.72 / 5m31s / 6.0 GiB / 3.87M
  - + fused updates: 24.73 / 5m04s / 5.6 GiB / 3.89M
  - + fused updates, skipping invisible Gaussians: 24.59 / 3m03s / 5.1 GiB / 3.34M
  - + fused updates with SH degree 0: 24.38 / 2m24s / 3.2 GiB / 3.85M
- **Table 1, end-to-end against 3DGS.** — [Faster-GS HTML](https://arxiv.org/html/2602.09999v1)
  - Mip-NeRF360: PSNR 27.53 to 27.56; time 18m44s to 4m31s; VRAM 8.8 to 6.1 GiB
  - T&T: PSNR 23.77 to 23.75; time 11m26s to 3m04s
  - DB: PSNR 29.81 to 29.78; time 19m43s to 3m46s
  - The paper claims up to 5.2x over 3DGS and 2.4x over Taming-3DGS.
- **Table 4, across GPUs, all 13 scenes** (3DGS against Faster-GS). — [Faster-GS HTML](https://arxiv.org/html/2602.09999v1)
  - RTX 3090: 23m46s against 6m03s (3.9x)
  - RTX 4090: 17m46s against 4m10s (4.3x)
  - RTX 5090: 13m05s against 2m43s (4.8x)
- **Truncation and stability.**
  - Gaussians are truncated at about 3.33 sigma, from tau_alpha = 1/255. This is not ablated.
  - Numerical stability is handled with front-to-back gradients and handling of degenerate Gaussians, described qualitatively. The "about 15%" gain refers to the basis implementation as a whole.
  - Lower-precision optimization is explicitly out of scope.
  - The appendix reports "up to 2x" for a shared-memory per-Gaussian backward kernel, with no table.
  - Fused SSIM is used by every method and is not ablated.
  — [Faster-GS HTML](https://arxiv.org/html/2602.09999v1)

### Inferences
- **Per-Gaussian backward.** The CPU trainer already walks each tile's list and accumulates gradients per splat, so most of this structural win is already in hand. What remains:
  - Make sure no per-pixel atomic or scatter happens inside the inner loop.
  - Keep per-splat accumulators in registers across a whole tile, or a half-row strip.
- **Fusing Adam into backward with skip-invisible** is the most transferable large win for the 35 ms Adam stage. On CPU, Adam is bandwidth-bound, so not touching Gaussians that were invisible this step, together with a single pass that reads the gradient and writes the parameter and both moments, directly cuts memory traffic. Expect a quality cost of about -0.1 dB (Faster-GS outdoor: -0.13 dB) and a lower Gaussian count. This is lazy or sparse Adam semantics.
- **Z-ordering (Morton sort of the parameter arrays)** gives only a few percent on GPU. On CPU it should help the cache locality of projection and Adam gathers, with an uncertain magnitude.

### Gaps
- No per-stage millisecond numbers were found. Figure 2 shows per-step runtime for four scenes, but no numeric values are given in the text.
- No per-component PSNR is reported, except the optimizer variants in Table 3.
- I could not see whether further rows sit between "fused Adam (Ours)" (12m50s outdoor) and "Full without z-ordering" (5m52s). The jump suggests that the combination of all items, or other items, accounts for it. I read only the first 100k of 122k characters.

## Q2. LiteGS (arXiv 2503.01199): system/algorithm co-design and measured effects

### Takeaway
LiteGS works on two levels:
- **Primitive level:** Morton-sorted primitives, 128-primitive clusters with AABB frustum culling and compaction, and cluster-granular sparse gradients.
- **Pixel level:** a warp-per-tile rasterizer (4x32 tiles, 4 pixels per thread, a register-local gradient sum, then one warp shuffle reduction and one atomic per splat per tile), a scanline-incremental quadratic exponent, and FP16 blending.

On an RTX 3090, the raster kernels are 4-15x faster than original 3DGS. End-to-end training is 8.4-13.4x faster than 3DGS in the LiteGS-turbo setting.

### Cited Findings
- **Hardware.** A single RTX 3090. — [LiteGS HTML](https://arxiv.org/html/2503.01199)
- **Warp-based raster.**
  - One warp per tile. The tile is 4x32 = 128 pixels, with 4 pixels per thread.
  - Each thread sums the gradient over its pixels locally. The warp then does one shuffle reduction per tile, and each splat gradient needs one atomic per tile.
  — [LiteGS HTML](https://arxiv.org/html/2503.01199)
- **Scanline algorithm.**
  - Along a row, the exponent is written as Basic + Linear·j + Quad·j², with the coefficients computed once per scanline.
  - Per-pixel variables drop from 7 to 3, and the multiplications for the Gaussian drop from 9 to 2 per pixel.
  — [LiteGS HTML](https://arxiv.org/html/2503.01199)
- **Mixed precision.**
  - FP16 is used for blending temporaries (Gaussian value, transmittance, colour). FP16 was preferred over BF16 for transmittance accuracy.
  - Colour and opacity gradients are packed as half2.
  - Covariance gradients use an integer warp-reduce: take the max exponent, align the mantissas, sum as integers, then reconstruct.
  - The paper claims "nearly 2x" blending throughput but gives no isolated measurement.
  — [LiteGS HTML](https://arxiv.org/html/2503.01199)
- **Morton ordering and Cluster-Cull-Compact.**
  - Centres are quantized to 21 bits per axis into a 64-bit Morton code.
  - The array is re-sorted every 5 epochs and after each densification.
  - Clusters of 128 contiguous primitives each get an AABB, are frustum-culled per cluster, and the visible ones are compacted.
  — [LiteGS HTML](https://arxiv.org/html/2503.01199)
- **Table 2, with against without culling** (LiteGS-quality). Disabling culling costs +21% to +62% time; keeping it costs -0.03 dB. — [LiteGS HTML](https://arxiv.org/html/2503.01199)
  - Mip-NeRF360: 515 s against 701 s (+36% without)
  - T&T: 398 s against 483 s (+21%)
  - DB: 330 s against 533 s (+62%)
  - PSNR difference: -0.03 dB in each case
- **Table 3, raster kernel speedups** (garden scene, at 500k / 2000k / 5728k primitives). The ratios are my computations from the table. The text claims "7-13x" and "about 4x", which does not match the table exactly. — [LiteGS HTML](https://arxiv.org/html/2503.01199)

  | LiteGS warp-based raster against | Forward | Backward |
  |---|---|---|
  | Original 3DGS atomic backward | 15.1x / 7.3x / 5.9x | 13.5x / 10.2x / 7.5x |
  | Taming per-splat backward | 14x / 5.2x / 4.2x | 4.0x / 3.7x / 3.9x |
  | Tensor-core GS | 5.0x / 2.5x / 2.3x | not reported |
- **Sparse gradient.** Applied at cluster granularity, building on Taming, so updates touch contiguous memory. No standalone number is given. — [LiteGS HTML](https://arxiv.org/html/2503.01199)
- **End-to-end.** — [LiteGS HTML](https://arxiv.org/html/2503.01199)
  - LiteGS-turbo against 3DGS: 11.2x (Mip360), 8.4x (T&T), 13.4x (DB), with PSNR +0.23, +0.18 and +0.88 dB.
  - LiteGS-quality against 3DGS-MCMC: about 6.1x, 2.8x and 7.0x, with PSNR +0.17, +0.62 and +0.67 dB.
- **Densification.**
  - The criterion is the variance of each Gaussian's opacity gradient, computed as C·Var = S - M²/C.
  - Opacity decay halves opacity every 10 epochs instead of a hard reset.
  - Ablation on Mip-NeRF360 PSNR: full model 28.12 dB, without decay 27.68 dB, without the variance criterion 27.40 dB.
  — [LiteGS HTML](https://arxiv.org/html/2503.01199)

### Inferences
- **The scanline quadratic exponent is the most directly transferable per-pixel trick.**
  - The CPU half-row of 8 lanes can be evaluated as base + j·lin + j²·quad, with j a constant lane-index vector.
  - Evaluated incrementally across rows, this is 2 FMAs per pixel instead of computing dx, dy and the conic quadratic form.
  - The trainer already uses exp2-prescaled conics, so this should be combined with those.
  - Expected effect: fewer FMAs per half-row in both forward and backward. The backward also needs the per-pixel derivative terms dx and dy, which can likewise be generated incrementally.
- **The warp-reduce-then-one-atomic pattern equals the CPU's "sum over the half-row's lanes in a register, then one horizontal add per splat per tile".** The trainer already does a variant of this.
- **The integer-mantissa reduction trick is not needed on CPU.** Plain f32 accumulation is fine.
- **Cluster-Cull-Compact.** With 400k Gaussians and a single camera, cluster AABB culling (128 per cluster) would skip projection, SH evaluation and Adam work for off-screen clusters.
  - On a 4-vCPU box, this mainly reduces per-splat work and memory traffic in preprocessing and Adam, not per-pixel raster.
  - Measured GPU end-to-end effect: 21-62% saved. This gain partly reflects GPU launch and occupancy effects.
- **FP16 blending is not attractive on AVX-512 Xeon** without AVX512-FP16, which Sapphire Rapids has but older Xeons lack. Its value is register pressure, which is not the CPU bottleneck. I mark this as low transfer.

### Gaps
- No standalone timing was found for the scanline trick, FP16, or sparse gradients.

## Q3. Other GPU systems: what each does and measures

### Takeaway
- **Big wins.** These come from:
  - per-splat or per-tile register-reduced backward (Taming 2x, LiteGS, VkSplat)
  - tighter splat-to-tile sets (Speedy-Splat SnugBox/AccuTile, VkSplat scanline culling, HiGS)
  - fused projection-backward plus Adam (VkSplat, Faster-GS)
  - pixel subsampling (Turbo-GS dilated or checkerboard rendering: about -0.04 dB)
  - fused SSIM (5-8x on the SSIM op)
  - 32-bit sort keys (VkSplat, RoofGS)
- **Rendering only.** HiGS and RoofGS do not implement training.
- **Sort-free.** Methods change the image-formation model and need view-dependent opacity.

### Cited Findings

**Taming 3DGS** (Mallick et al., SIGGRAPH Asia 2024, arXiv 2406.15643)
- **Per-splat backward.**
  - Threads are parallel over a tile's splats rather than its pixels.
  - The forward pass stores per-pixel state (T and accumulated RGB) every N = 32 splats.
  - In backward, buckets of 32 splats are assigned to a warp. The warp rebuilds intermediate pixel states via shuffle by re-applying alpha blending from the checkpoint.
  - The last contributor per tile is tracked to skip redundant pairs, and the tighter culling from Radl et al. is used.
  — [Taming HTML](https://arxiv.org/html/2406.15643)
- **Other changes.**
  - Higher SH bands take an Adam step only every 16 iterations. The paper notes this is the only change that is not numerically equivalent to 3DGS.
  - SH band 0 is stored separately.
  - SSIM uses separable 1D convolutions in a fused kernel.
  — [Taming HTML](https://arxiv.org/html/2406.15643)
- **Ablation (T&T, RTX A4500).** Removing per-splat backward doubles training time: Truck 7 to 14 min, Train 9 to 17 min, at identical PSNR. Removing the reduced SH frequency costs 2-3 min (7 to 9 and 9 to 12 min) and slightly raises PSNR (+0.19, +0.06 dB). — [Taming HTML](https://arxiv.org/html/2406.15643)
- **End-to-end (budgeted).** — [Taming HTML](https://arxiv.org/html/2406.15643)
  - T&T: 7 against 28 min, PSNR 23.95 against 23.65
  - Mip360: 11 against 43 min, PSNR 27.31 against 27.46
  - DB: 7 against 39 min, PSNR 29.82 against 29.64

**Speedy-Splat** (Hanson et al., CVPR 2025, arXiv 2412.00578)
- **SnugBox.** A tight AABB of the ellipse at the threshold t = 2·ln(255·sigma_i), where sigma_i is the opacity, so it is opacity-aware. The extremes come from the derivative of the ellipse equation. — [Speedy-Splat HTML](https://arxiv.org/html/2412.00578v3)
- **AccuTile.** Walks tile rows (or columns, whichever is shorter). For each row it computes the two ellipse intersection points on the row boundaries, then marks every tile between the min and max as touched. — [Speedy-Splat HTML](https://arxiv.org/html/2412.00578v3)
- **Pruning.**
  - Soft pruning removes 80% right before the opacity resets at 6k, 9k and 12k iterations.
  - Hard pruning removes 30% every 3k iterations after 15k.
  — [Speedy-Splat HTML](https://arxiv.org/html/2412.00578v3)
- **Ablation (RTX A5000, average over Mip360, T&T and DB).** — [Speedy-Splat HTML](https://arxiv.org/html/2412.00578v3)

  | Row | Gaussians | Training time (speedup) | FPS (speedup) | Mip360 PSNR delta |
  |---|---|---|---|---|
  | Baseline | 2.93M | 23.2 min | 134 | — |
  | + SnugBox | — | 21.2 min (1.09x) | 244 (1.82x) | 0.00 |
  | + AccuTile | — | 21.0 min (1.10x) | 267 (1.99x) | +0.02 |
  | + Soft pruning | 1.64M | 17.5 min (1.32x) | 420 (3.14x) | -0.23 dB, SSIM -0.007, LPIPS +0.024 |
  | + Hard pruning | 0.28M | 15.7 min (1.47x) | 898 (6.71x) | -0.61 dB, SSIM -0.032, LPIPS +0.074 |

**VkSplat** (Chen, Ibrahim, Liu, Huawei Canada; Eurographics 2026 short paper; arXiv 2605.00219; code at github.com/harry7557558/vksplat)
- **Culling.** Scanline tile culling with closed-form interval arithmetic gives exact Gaussian-to-tile overlaps, with no false negatives. Counting and traversal run in fused kernels. — [VkSplat project page](https://harry7557558.github.io/vksplat/)
- **Backward.** Two raster-backward kernels exist: per-Gaussian, and a shared-memory forward+backward kernel. A Thompson-sampling scheduler picks the faster one per dataset and hardware. The page says the design minimizes atomic contention, while the abstract says it eliminates pixel-level atomics. — [VkSplat project page](https://harry7557558.github.io/vksplat/)
- **Fusion and keys.**
  - Projection-backward and Adam are fused into one kernel.
  - Sort keys are 32-bit: 14-bit tile ID and 18-bit depth. The page says quality is preserved up to 1080p.
  — [VkSplat project page](https://harry7557558.github.io/vksplat/)
- **Results (RTX 3090, 7 Mip360 scenes).** — [VkSplat project page](https://harry7557558.github.io/vksplat/)
  - Default densification: gsplat 1384 s against VkSplat 412 s (3.35x); VRAM 4.56 to 3.01 GiB
  - MCMC: 995 s against 285 s (3.49x)
  - PSNR about 29.2 for both; SSIM 0.878 for both
- **Per-stage totals, gsplat to VkSplat** (seconds over the full run). — [VkSplat project page](https://harry7557558.github.io/vksplat/)

  | Stage | gsplat | VkSplat |
  |---|---|---|
  | Projection forward | 94 | 19 |
  | Tiling and sort | 42 | 25 |
  | Raster forward | 69 | 25 |
  | Loss | 103 | 32 |
  | Raster backward | 246 | 130 |
  | Projection backward + optimizer | 459 | 131 |
  | Densification | 31 | 5 |
  | Unaccounted | 341 | 43 |
- **Cross-vendor.** On bicycle, RTX 3090 takes 575 s and AMD RX 7800 XT takes 1201 s. — [VkSplat project page](https://harry7557558.github.io/vksplat/)

**HiGS** (Pająk, Bisson, Lima, NVIDIA; arXiv 2606.00352; submitted 29 May 2026). Rendering only; a backward pass is left as future work.
- **Tiling.**
  - Render tiles are 8x8. Macro-tiles of 64x32 pixels are the unit for binning and sorting.
  - Each (macro-tile, batch of 1024 Gaussians) pair is one work unit.
  - A count, scan and fill partition replaces the global 64-bit composite-key sort.
  - Each macro-tile is then sorted as its own segment with 32-bit depth keys.
  — [HiGS HTML](https://arxiv.org/html/2606.00352)
- **Raster.**
  - Each unit loads its Gaussians into shared memory and tests each against the 32 member tiles with a conservative per-row ellipse scan (AccuTile, one level coarser).
  - Per-Gaussian masks are transposed into per-tile masks, then active tiles are compacted into a queue. This is the "hit-list compaction" in the original question.
  - A drain batch of 64 separates the variable-rate compaction from fixed-rate blending.
  - Partial RGBT per batch is composited front to back afterwards.
  — [HiGS HTML](https://arxiv.org/html/2606.00352)
- **Numerics.**
  - FP16 (half2) throughout.
  - The inverse covariance is stored as Cholesky factors, so the Mahalanobis distance is a sum of squares, which avoids cancellation.
  - Positions are relative to the macro-tile centre, and factors are prescaled for a single exp2.
  — [HiGS HTML](https://arxiv.org/html/2606.00352)
- **Results (RTX PRO 6000 Blackwell).** — [HiGS HTML](https://arxiv.org/html/2606.00352)
  - Pair count against gsplat: -85% at 1080p (10.1M to 1.51M), -92% at 4K
  - Frame time at matched 8x8 tiles: 0.52 against 2.49 ms at 1080p (4.8x); 0.82 against 6.92 ms at 4K (8.4x)
  - Sort at 4K: global radix 2.5-4.2 ms against under 0.08 ms
  - PSNR against fp32 gsplat: 67.03 dB, so there is effectively no ground-truth quality change (27.68 against GT)

**RoofGS** (Luo, Gong, Gao, Zhao; arXiv 2608.15785; Aug 2026). Rendering only; training is listed as future work.
- **Front end.**
  - SH stored as INT8 with per-degree, per-channel scales; rotation, scale and opacity in FP16; the mean in FP32. This cuts the record from 236 to about 96 bytes.
  - A 32-bit key (tile in the high bits, quantized depth in the low bits) halves radix passes from 8 to 4.
  - Preprocessing, duplication and key generation are fused, cutting global-memory transactions from 5 to 2 per primitive.
  — [RoofGS HTML](https://arxiv.org/html/2608.15785)
- **Raster.**
  - Constants are folded into the coefficients.
  - Each thread evaluates two vertically adjacent pixels, deriving the second exponent incrementally (dual-pixel ILP).
  - exp uses a range-aware Schraudolph bit trick, with a stated colour error of about 0.03 per pixel.
  — [RoofGS HTML](https://arxiv.org/html/2608.15785)
- **Result.** 10.1x end-to-end rendering over 3DGS at 4K on an RTX 4090 (61 to 616 FPS), at -0.028 dB. No per-stage numbers are given. — [RoofGS HTML](https://arxiv.org/html/2608.15785)

**Turbo-GS** (Dhiman, Lu et al.; arXiv 2412.13547 v2; CVPR 2026)
- **Dilated rendering.**
  - Each view renders a chessboard-like subset of pixels, set by a dilation size and two offsets.
  - Threads per block match the active pixel count, and backward skips the inactive pixels.
  - A screen-space low-pass filter adds 0.3 + 0.5(p-1) to the covariance diagonal for pattern size p.
  - Dilated rendering is used exclusively during densification, then randomly afterwards.
  — [Turbo-GS HTML](https://arxiv.org/html/2412.13547v2)
- **Ablation (bicycle 4K, PSNR / SSIM / LPIPS).** Dilated loses -0.04 dB; multi-scale loses -0.22 dB. Figure 9 shows reduced forward and backward time, with no number given. — [Turbo-GS HTML](https://arxiv.org/html/2412.13547v2)
  - Full resolution: 25.51 / 0.770 / 0.216
  - Dilated: 25.47 / 0.770 / 0.216
  - Multi-scale: 25.29 / 0.755 / 0.237
- **Base and results.** Turbo-GS is built on Scaffold-GS plus Taming's backward kernel, on an RTX A6000. Mip360 trains in about 5 min, against about 16 min for Taming and about 30 min for 3DGS. — [Turbo-GS HTML](https://arxiv.org/html/2412.13547v2)
- **Not to be confused with** TurboGS (Dong et al., ICML 2026, arXiv 2606.15924), which does error-guided sparse pixel sampling. I did not read it. — [arXiv 2606.15924](https://arxiv.org/abs/2606.15924)

**gsplat** (Ye et al., arXiv 2409.06765, JMLR 2025)
- The abstract claims "up to 10% less training time and 4x less memory than the original implementation". The detailed techniques (packed mode, absgrad) were not verified from the abstract page. — [arXiv abs](https://arxiv.org/abs/2409.06765)

**fused-ssim** (github.com/rahul-goel/fused-ssim)
- A separable, symmetric 11x11 Gaussian.
- Several statistics are computed in one convolution pass, with intermediates kept on-chip.
- The backward pass of the convolution is another Gaussian convolution.
- About 5-8x faster than pytorch-msssim for 2D, and about 11x for 3D. Charts cover RTX 4090, M1 Pro and A5000, with no exact timings.
- Backends: CUDA, ROCm, Metal and SYCL. No CPU backend.
- [fused-ssim README](https://github.com/rahul-goel/fused-ssim)

**Sort-free, weighted-sum rendering** (Hou et al., ICLR 2025, arXiv 2410.18931)
- Replaces alpha blending with a depth-weighted sum using learned weight functions, so no sort is needed.
- Requires view-dependent opacity for quality.
- Reports on average 1.23x faster rendering on a mobile GPU.
- [Sort-free HTML](https://arxiv.org/html/2410.18931v2)

### Inferences (CPU transfer)
- **AccuTile / scanline exact tile culling (Speedy-Splat, VkSplat, HiGS) transfers fully.** The trainer uses 16x16 tiles and already visits only half-rows reaching alpha >= 1/255, which is a per-half-row mask.
  - Exact per-tile culling with an opacity-aware threshold (SnugBox: t = 2·ln(255·opacity)) reduces tile-list length. That saves per-splat setup per tile, sort keys, and empty half-row checks.
  - Training gain on GPU was only 1.10x, so expect a modest gain.
  - A finer variant: compute, for each (splat, tile) pair, the exact row interval [y0, y1] and the per-row x-interval. Then only the 8-lane half-rows that intersect are visited, with no test needed. This is HiGS's per-row ellipse scan applied at half-row granularity.
- **Checkerboard or dilated pixel subsampling (Turbo-GS) reduces per-pixel work in forward, backward and loss by about 2x, at about -0.04 dB.**
  - It is close in spirit to PennyGS's 2x2 strided sampling, which is already known, so it is not new.
  - Turbo-GS's covariance dilation of +0.3 + 0.5(p-1) is a needed detail to avoid aliasing.
- **Fused SSIM.** The trainer's 13 ms L1+SSIM already uses separable blurs.
  - The fused-ssim idea is to compute all five blurred statistics (mu_x, mu_y, sigma_xx, sigma_yy, sigma_xy) in one pass with SIMD over a tile.
  - The backward pass of the blur is another blur of the three per-pixel derivative maps.
  - On CPU this helps mainly through fewer full-image passes, so less memory traffic.
- **The fused projection-backward + Adam pass (VkSplat, Faster-GS) is the biggest per-splat/memory win.** In VkSplat it cut "projection backward + optimizer" from 459 s to 131 s.
- **32-bit keys** are already known through PennyGS's 4-byte keys.
- **Sort-free weighted sum** would remove sorting and back-to-front traversal dependencies, so the backward needs no T reconstruction. However, it changes the model and needs view-dependent opacity. Quality on standard benchmarks was not verified here. It is a risky, model-level change.

### Gaps
- **StopThePop** (Radl et al., SIGGRAPH 2024, arXiv 2402.00525) and **Kheradmand 2025** (sort-free / stochastic methods) were not fetched within budget, so their numbers are unverified. Taming cites "tighter culling from Radl et al.", which is StopThePop's tile-based culling.
- **LichtFeld Studio:** no source was fetched; nothing is verified.
- **gsplat:** detailed per-technique numbers were not verified.
- **VkSplat:** the exact backward design and its ablation could not be read; the PDF was compressed.
- **The specific "hit-list compaction" wording for HiGS** was not found as a term. The closest match is the per-tile mask compaction into a work queue.

## Q4. CPU-SIMD, mobile/embedded and WebGPU training of Gaussian splats

### Takeaway
I found no paper or open-source project describing a SIMD-vectorized (AVX or NEON) CPU 3DGS trainer.
- OpenSplat runs CPU-only via libtorch at "~100x slower" than GPU.
- PocketGS trains on an iPhone 15 GPU through Metal, using a replay-cache backward.
- Brush trains via WebGPU (Burn) on browsers and Android, with no CPU backend.
- CPU appears in the literature only as an offload host (GS-Scale, CLM).

### Cited Findings
- **PocketGS** (arXiv 2601.17354): Swift on Apple Metal, iPhone 15 (A16, 6 GB). — [PocketGS HTML](https://arxiv.org/html/2601.17354)
  - **Forward.** Stores a per-pixel replay cache S = {id, C_in, alpha} with a per-pixel counter, up to capacity K_max.
  - **Backward.** Replays the stored C_in and alpha to compute closed-form compositing gradients, traversing only entries 0..count-1.
  - **Reset.** Only the counters are cleared each iteration.
  - **Effect of counter-only reset against a full-cache clear:** backward 21.8 to 15.2 ms; peak memory 2.41 to 2.21 GB.
  - **Gradient scatter.** Gradients go back to the canonical layout through the sort permutation (grad[pi(i)] += g_i).
  - **Timing.** 500-step training plus pipeline takes 255.2 s on MobileScan, 105.4 s on LLFF and 145.5 s on Mip360.
  - **Quality.** PSNR / SSIM / LPIPS of 23.54 / 0.791 / 0.222 on LLFF and 22.25 / 0.702 / 0.302 on Mip360. Both are better than DashGaussian.
- **Brush** (github.com/ArthurBrussee/brush) uses WebGPU-compatible technology and the Burn framework. — [Brush README](https://github.com/ArthurBrussee/brush)
  - Targets: macOS, Windows, Linux, Android (arm64) and Chrome 134+.
  - Reference kernels come from gsplat; the radix sort comes from Raph Levien.
  - It claims "generally faster than gsplat" with no numbers, and has no CPU backend.
- **OpenSplat** (C++/libtorch) "can also run entirely on the CPU (~100x slower)". I did not check how the CPU path is threaded or vectorized. — [OpenSplat](https://github.com/pierotofy/opensplat)
- **GS-Scale** (arXiv 2509.15645) keeps Gaussians in host memory. It notes that frustum culling and optimizer updates on the CPU cause slowdowns because of the CPU's limited compute and memory bandwidth. — [GS-Scale HTML](https://arxiv.org/html/2509.15645v1)
- **CLM** (ASPLOS 2026) offloads Gaussians to CPU memory. — [CLM HTML](https://arxiv.org/html/2511.04951v1)
- **Hardware rasterization.** Differentiable hardware rasterization (arXiv 2505.18764) reports 3.07x end-to-end over a tile-based rasterizer on an RTX 4080. GPU-specific; no CPU transfer. — [arXiv 2505.18764](https://arxiv.org/html/2505.18764v1)
- **Dedicated hardware.** Vorion (a RISC-V GPU with 3DGS hardware) reports 38.6 iterations/s training. Specialized silicon; no CPU transfer. — [Vorion](https://arxiv.org/html/2511.16831v1)
- **rlx-cpu** (a Rust crate) contains a CPU "splat" op with SIMD dispatch. Its contents were not verified. — [docs.rs rlx-cpu](https://docs.rs/crate/rlx-cpu/0.2.3/source/src/splat.rs)

### Inferences
- **The CPU trainer appears to be beyond the published state of the art for CPU 3DGS training.** OpenSplat's libtorch CPU path is "~100x slower" than GPU.
- **PocketGS's replay cache** (store C_in and alpha per pixel per splat, then replay without re-evaluating the Gaussian) is the opposite trade-off to the CPU trainer, which recomputes. On CPU, storing alpha per (pixel, splat) pair costs memory bandwidth roughly equal to recomputing exp with a polynomial. It is probably not a win, except that it removes the second evaluation of the Gaussian in backward. Worth a micro-benchmark only for tiles with short lists.
- **Counter-only reset of per-pixel buffers** is a cheap, transferable hygiene trick.

### Gaps
- No SIMD-CPU (AVX/NEON) 3DGS training paper was found within the search budget. The rlx-cpu splat kernel is unverified.

## Q5. Which techniques cut per-pixel work, per-splat work, or memory traffic

### Takeaway
On the CPU trainer, raster forward, raster backward and loss take about 110 of the 175 ms per step, so the per-pixel techniques matter most. They fall into four kinds:
1. Visit fewer (pixel, splat) pairs: exact tile and row culling, an opacity-aware extent, pixel subsampling, and pruning.
2. Spend fewer FLOPs per pair: the scanline quadratic exponent, folded or prescaled coefficients, fast exp, and dual-pixel incremental exponents.
3. Avoid a second evaluation in backward: replay caches or per-chunk checkpoints.
4. Reduce passes in the loss: fused SSIM.

Per-splat and memory-traffic wins (fused Adam, skip-invisible, cluster culling, compact attributes, Morton order) target the 35 ms Adam stage and preprocessing.

### Cited Findings
- **Fewer pairs.**
  - AccuTile / SnugBox: about 1.99x rendering FPS and 1.10x training, lossless. — [Speedy-Splat](https://arxiv.org/html/2412.00578v3)
  - HiGS: -85% to -92% pairs. — [HiGS](https://arxiv.org/html/2606.00352)
  - Turbo-GS dilated rendering: -0.04 dB. — [Turbo-GS](https://arxiv.org/html/2412.13547v2)
  - Speedy-Splat pruning: 1.47x training, -0.61 dB. — [Speedy-Splat](https://arxiv.org/html/2412.00578v3)
- **Fewer FLOPs per pair.**
  - LiteGS scanline: 9 to 2 multiplications per pixel. — [LiteGS](https://arxiv.org/html/2503.01199)
  - RoofGS: coefficient folding, dual-pixel ILP, and a Schraudolph exp with error about 0.03. — [RoofGS](https://arxiv.org/html/2608.15785)
  - HiGS: Cholesky-factor sum of squares with prescaled exp2. — [HiGS](https://arxiv.org/html/2606.00352)
- **Backward structure.**
  - Taming per-splat backward: 2x training. — [Taming](https://arxiv.org/html/2406.15643)
  - Faster-GS per-Gaussian backward: 1.20-1.84x. — [Faster-GS](https://arxiv.org/html/2602.09999v1)
  - LiteGS warp reduction: 3.7-4.0x backward over Taming. — [LiteGS](https://arxiv.org/html/2503.01199)
  - PocketGS replay cache: backward 21.8 to 15.2 ms. — [PocketGS](https://arxiv.org/html/2601.17354)
- **Memory traffic and per-splat work.**
  - Faster-GS fused Adam: 1.12-1.33x; fused into backward with skip-invisible: 5m31s to 3m03s at -0.13 dB. — [Faster-GS](https://arxiv.org/html/2602.09999v1)
  - VkSplat fused projection-backward + Adam: 459 to 131 s. — [VkSplat](https://harry7557558.github.io/vksplat/)
  - LiteGS cluster culling: 21-62%. — [LiteGS](https://arxiv.org/html/2503.01199)
  - RoofGS compact attributes: 236 to 96 bytes. — [RoofGS](https://arxiv.org/html/2608.15785)
  - Taming: SH higher bands updated every 16 iterations. — [Taming](https://arxiv.org/html/2406.15643)

### Inferences: ranked candidates for the CPU trainer (my judgement)

**Per-pixel work (forward + backward + loss, about 110 ms)**
1. **Scanline-incremental exponent across 8-lane half-rows, with folded and exp2-prescaled coefficients** (LiteGS, RoofGS, HiGS). Expect fewer arithmetic ops per lane in both forward and backward. This is likely the cheapest change for a measurable gain.
2. **Exact per-row x-intervals per (splat, tile)**, the AccuTile/HiGS row scan carried down to half-rows. It removes the visits to half-rows that fail the alpha >= 1/255 test, and the threshold test itself.
3. **Checkerboard or dilated sampling with covariance dilation** (Turbo-GS). Overlaps with the already-known PennyGS 2x2 stride.
4. **Fused single-pass SSIM statistics plus a fused backward blur.** Cuts memory passes in the 13 ms loss stage.
5. **Schraudolph-style exp.** The trainer already uses a polynomial exp; the marginal gain is uncertain.

**Per-splat work and memory traffic (Adam 35 ms, preprocessing)**
1. **Fuse Adam into the end of backward, or into one streaming pass, and skip Gaussians that were invisible this step.** This mirrors Faster-GS and VkSplat. GPU evidence shows about 1.8x on top of an already-fused optimizer, at about -0.1 dB.
2. **Update SH bands 1-3 every N steps** (Taming: every 16). The degree-3 SH is 45 of 59 floats per Gaussian, so this cuts most of the Adam bandwidth on those steps. Numerically non-equivalent, with near-zero or slightly positive PSNR effect in Taming's ablation.
3. **Morton-order the parameter arrays plus cluster AABB culling.** Locality and skipping off-screen clusters (LiteGS, Faster-GS z-order).
4. **Store moments or SH in BF16 or FP16** (RoofGS-style compaction). This is an assumption on my part: it is not measured for training in these papers.

**Not transferable or low value on CPU**
- FP16 blending and integer warp reductions (LiteGS)
- Load-balanced instancing and Thompson-sampled kernel selection (VkSplat)
- Segmented GPU radix cascades (HiGS), beyond the idea of a per-macro-tile depth sort with 32-bit keys
- Hardware rasterization

### Gaps
- Per-stage millisecond breakdowns for Faster-GS were not available numerically.
- No CPU measurements exist in any source, so all CPU transfer estimates above are my inference, not measured.
