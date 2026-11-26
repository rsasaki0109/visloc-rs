# Densification, primitive-budget and training-schedule techniques for faster 3DGS training (2024-2026), and how they transfer to a CPU-only trainer

The CPU trainer this assesses: Rust, 4 vCPU with AVX2, 600x440 images, 7k steps in about 25 min. Each step takes about 175 ms: about 110 ms of per-pixel work (raster forward/backward plus loss), about 35 ms of bandwidth-bound Adam, and about 30 ms of other work. The cap is 400k Gaussians at SH degree 3.

Caveat on timings: every paper times on a different GPU. Absolute minutes are not comparable across papers. Compare the ratios within each paper.

## Q1. FastGS (arXiv 2511.04283): multi-view consistent densification (VCD) and pruning (VCP)

### Takeaway
FastGS densifies a Gaussian only when its footprint covers many high-error pixels, averaged over K=10 random training views. It prunes by an error-weighted score of the same kind. On an RTX 4090 this gives 0.40M Gaussians on Mip-NeRF 360 and about 2 min of training, with PSNR equal to 3DGS. Almost all of the speedup comes from a much smaller model. It does not reduce the number of pixels rendered per step.

### Cited Findings
- **VCD rule:**
  - Per-pixel error is the mean absolute RGB difference between render and ground truth. The error map is min-max normalized, then thresholded at τ (value not given in the text) to make a high-error mask.
  - Score: s_d^i = (1/K) Σ_j Σ_{p∈footprint_i} 1[mask_j(p)=1]. This is the number of high-error pixels in the Gaussian's 2D footprint, averaged over the K views.
  - A Gaussian is densified when s_d > τ_d = 5.
  - Clone decisions use the vanilla 3DGS gradient. Split decisions use the AbsGS absolute gradient.
  - K = 10 randomly sampled training views.
  - Source: [FastGS arXiv HTML](https://arxiv.org/html/2511.04283)
- **VCP rule:**
  - Per-view photometric error: E_photo^j = 0.8·L1 + 0.2·(1−SSIM).
  - Score: s_p^i = minmax-normalize(Σ_j (high-error footprint count)·E_photo^j).
  - Before 15K iterations: on top of the vanilla 3DGS pruning, half of the candidate Gaussians are sampled for removal by pruning score.
  - After 15K iterations, every 3K iterations: remove Gaussians with opacity < 0.1 or s_p > 0.9.
  - Source: [FastGS](https://arxiv.org/html/2511.04283)
- **Schedule:**
  - Densify every 500 iterations until 15K. Prune every 500 iterations before 15K and every 3K after. 30K iterations in total.
  - FastGS-Big densifies every 100 iterations.
  - Optimizer schedule: from 15K to 20K the optimizer updates every 32 iterations, then every 64.
  - Also used: per-splat backward and accelerated SH optimization from Taming, and "Compact Box", a Mahalanobis-based pruning of Gaussian–tile pairs adapted from Speedy-Splat.
  - Source: [FastGS](https://arxiv.org/html/2511.04283)
- **Results on one RTX 4090** (time in min / PSNR / SSIM / LPIPS / #Gaussians). Source: [FastGS](https://arxiv.org/html/2511.04283)

  | Dataset | FastGS | FastGS-Big | DashGaussian | 3DGS | Taming |
  |---|---|---|---|---|---|
  | Mip-NeRF 360 | 1.93 / 27.56 / 0.797 / 0.261 / 0.40M | 3.58 / 27.93 / 0.820 / 0.216 / 1.15M | 6.35 / 27.73 / – / – / 2.40M | 20.93 / 27.53 / 0.812 / 0.221 / 2.63M | 5.36 / 27.48 / – / – / 0.68M |
  | Deep Blending | 1.28 / 30.03 / 0.901 / 0.270 / 0.22M | – | – | 19.77 / 29.71 / – / – / – | – |
  | Tanks&Temples | 1.32 / 24.15 / 0.839 / 0.210 / 0.24M | – | – | 11.34 / 23.71 / – / – / – | – |

  - Note that FastGS's SSIM and LPIPS are worse than 3DGS's (0.797 vs 0.812 SSIM, 0.261 vs 0.221 LPIPS on Mip-NeRF 360).
- **Ablation** (Mip-NeRF 360, on the 3DGS-accel baseline). Source: [FastGS](https://arxiv.org/html/2511.04283)

  | Config | Time (min) | PSNR | #Gaussians |
  |---|---|---|---|
  | Baseline | 7.10 | 27.46 | 2.64M |
  | +VCD | 3.53 | 27.69 | 0.53M |
  | +VCP | 5.32 | 27.70 | 1.96M |
  | +Compact Box | 6.13 | 27.44 | 2.78M |
  | Full | 1.93 | 27.56 | 0.40M |

- **Number of views K:** K=5 takes 1.91 min, K=10 takes 1.93, K=20 takes 2.03, all views take 2.29.
- **τ_d on T&T:** τ_d=1 takes 1.44 min, τ_d=5 takes 1.32 min, τ_d=100 takes 1.12 min with lower PSNR.
- **Sparse Adam vs the optimizer schedule:** both take 1.93 min, with PSNR 27.37 vs 27.56. Source: [FastGS](https://arxiv.org/html/2511.04283)

### Inferences
- **CPU transfer:**
  - VCD/VCP mainly shrinks N. The CPU trainer is already capped at 400k, which is about FastGS's final count, so the direct time gain will be much smaller than on GPU.
  - The real CPU value is (a) better placement of a fixed budget and (b) fewer Gaussians per pixel, which shortens the per-pixel blend lists.
  - Scoring with 10 separate full-resolution renders is expensive on CPU (10 × ~110 ms ≈ 1.1 s per densify event). PennyGS's approach, which accumulates the score from training renders that are already computed, is the right adaptation.
  - The optimizer schedule (Adam updates every 32 to 64 steps late in training) directly attacks the 35 ms Adam cost, but it only applies after 15K steps. A 7K-step run would need to compress it.

### Gaps
- The value of the error-mask threshold τ is not stated in the text.
- No per-pixel versus per-splat time breakdown is given.

## Q2. DashGaussian (Chen et al., CVPR 2025, arXiv 2503.18402): frequency-based resolution schedule and primitive schedule

### Takeaway
DashGaussian trains at low resolution first and reaches full resolution at the end. It switches resolution at iterations proportional to the log of the Fourier energy captured at each level. The primitive count grows in step with resolution. On the Taming backbone, training time drops by 41% with PSNR +0.3 dB. The resolution schedule alone saves 20% with PSNR +0.38 dB. This is the technique in this survey that most directly cuts total per-pixel work across the run.

### Cited Findings
- **Frequency significance:** X(F) = (1/N) Σ_n Σ_{i,j} ‖F^n(i,j)‖₂, where F^n is the DFT of training view n at a given resolution.
- **Switch iteration for level r_i:** s_{r_i} = S · ln(X(F_{r_i})/X(F_{r_m})) / ln(X(F)/X(F_{r_m})), where S is the total number of steps.
  - The coarsest level is chosen so that X(F_{r_m}) = X(F)/a, with a = 4 by default.
  - Intermediate levels are spaced linearly in X.
  - Resolution is interpolated using the inverse square of the downsampling factors and floored to integers.
  - Source: [DashGaussian arXiv HTML](https://arxiv.org/html/2503.18402)
- **Sensitivity to a:** quality is largely insensitive to a, while time falls as a grows (a=4: 3.23 min; a=10: 2.79 min). Source: [DashGaussian](https://arxiv.org/html/2503.18402)
- **Primitive schedule:**
  - Budget: P_i = P_init + (P_fin − P_init)/(r^(i))^(2 − i/S), where r^(i) is the current downsampling factor.
  - Momentum-based final count: P_fin^(i) = max(P_fin^(i−1), γ·P_fin^(i−1) + η·P_add^(i)), with γ=0.98, η=1, and P_fin^(0) = 5·P_init.
  - Source: [DashGaussian](https://arxiv.org/html/2503.18402)
- **Results** (PSNR / SSIM / LPIPS / minutes). Source: [DashGaussian](https://arxiv.org/html/2503.18402)

  | Method | Mip-NeRF 360 | Deep Blending | T&T |
  |---|---|---|---|
  | 3DGS | 27.72 / 0.826 / 0.201 / 18.31 | 29.50 / 0.904 / 0.244 / 17.27 | 23.62 / 0.847 / 0.177 / 10.59 |
  | 3DGS + Dash | 27.81 / 0.828 / 0.203 / 10.16 | 29.29 / 0.901 / 0.256 / 8.16 | 23.83 / 0.844 / 0.189 / 6.13 |
  | Taming | 27.61 / 0.821 / 0.210 / 5.51 | 29.69 / 0.906 / 0.245 / 4.52 | 23.62 / 0.849 / 0.174 / 4.02 |
  | Taming + Dash | 27.92 / 0.826 / 0.208 / 3.23 | 30.02 / 0.907 / 0.248 / 2.20 | 23.97 / 0.851 / 0.180 / 2.62 |

  - Gaussian counts for Taming + Dash: 2.04M (Mip-NeRF 360), 1.94M (DB), 1.19M (T&T).
  - Average time reduction over 12 backbone/dataset pairs: 45.7%.
  - With the 3DGS backbone, Deep Blending loses 0.21 dB and LPIPS rises 0.012.
- **Ablation** (Taming backbone, Mip-NeRF 360). Source: [DashGaussian](https://arxiv.org/html/2503.18402)

  | Variant | Primitives | PSNR | LPIPS | Time (min) |
  |---|---|---|---|---|
  | Taming | 2.28M | 27.61 | 0.210 | 5.51 |
  | Resolution schedule only | 2.46M | 27.99 | 0.196 | 4.43 |
  | Primitive schedule only | 2.23M | 27.85 | 0.208 | 4.60 |
  | Full | 2.04M | 27.92 | 0.208 | 3.23 |

- **Hardware:** one GPU of about 80 TFLOPS FP32. The model is not named; the paper says "consumer-grade GPU" (probably RTX 4090-class, but this is not verifiable). Source: [DashGaussian](https://arxiv.org/html/2503.18402)

### Inferences
- **The schedule ends at full resolution.** Only the late iterations run at full resolution, unlike the trainer's earlier constant half-resolution test (2.4x per step, −0.5 dB at 7k). DashGaussian's results suggest the quality loss largely goes away when the run ends at full resolution.
- **Expected CPU saving: roughly 30–40%.** Per-pixel work is about 63% of a CPU step, against a smaller share on GPU. With a=4 and a log schedule, a large part of the run happens at 1/4 to 1/2 of the full pixel count.
- **The primitive schedule fits naturally with the 400k cap.** Low-resolution phases need fewer Gaussians, which also trims the 35 ms Adam cost early in the run.
- **Caveat on run length:** the paper tests 30K-step runs. Whether the log schedule holds up when compressed into 7K steps is untested.

### Gaps
- The exact GPU model is not named.
- The number of resolution levels actually used per scene is not stated.
- No result at a short schedule such as 7K steps.

## Q3. Taming 3DGS (Mallick et al., SIGGRAPH Asia 2024, arXiv 2406.15643): score-based densification and budget mode

### Takeaway
Taming replaces gradient thresholding with weighted-score sampling under a parabolic growth curve toward a fixed budget. Budget mode is 4–6x faster with 4–6x fewer Gaussians at about the same PSNR, but LPIPS is clearly worse. The higher-order SH trick (Adam only every 16 iterations) is directly useful against the CPU Adam cost.

### Cited Findings
- **Score:**
  - Per-view saliency: S_v = 1_ROI ⊙ (0.5·L1 + 0.5·Laplacian).
  - The per-Gaussian score sums over N=10 sampled views, with weights: position-gradient 50, pixel-count 0.1, distance-to-center 50, saliency 10, blending-weight 50, depth 5, opacity 100, scale 25.
  - Densification runs every 500 iterations. New Gaussians are sampled with the score as probability weight.
  - Source: [Taming arXiv HTML](https://arxiv.org/html/2406.15643)
- **Budget curve:** Gaussians added per step follow A(x) = ((B − S − 2N)/N²)·x² + 2x + B. Here B is the budget, S the SfM point count and N the number of densification steps. Source: [Taming](https://arxiv.org/html/2406.15643)
- **Budgets:**
  - Budget mode: 2x the SfM points for Mip-NeRF 360 and T&T, 5x for DB, 15x for unbounded outdoor scenes.
  - Big mode: the 3DGS final size.
  - Source: [Taming](https://arxiv.org/html/2406.15643)
- **Results on RTX A4500** (PSNR / SSIM / LPIPS / time / #Gaussians). Source: [Taming](https://arxiv.org/html/2406.15643)

  | Dataset | 3DGS | Taming budget | Taming big |
  |---|---|---|---|
  | Mip-NeRF 360 | 27.46 / 0.815 / 0.215 / 43 min / 3.31M | 27.31 / 0.801 / 0.252 / 11 min / 0.63M | 27.79 / 0.822 / 0.205 / 32 min / – |
  | T&T | 23.65 / – / – / 28 min / – | 23.95 / 0.837 / 0.201 / 7 min / 0.29M | – |
  | DB | 29.64 / – / – / 39 min / – | 29.82 / 0.904 / 0.260 / 7 min / 0.27M | – |

- **Higher-order SH:** SH bands above the first are updated with Adam once every 16 iterations, with a 4x SH learning rate and a halved opacity learning rate.
- **Other speedups:** per-splat backward (about 2x faster backward; Truck total 7 vs 14 min), and SSIM with separable 1D convolutions.
- **Score-sampling ablation:** removing score-based sampling drops Truck/Train PSNR from 25.20/22.69 to 24.92/22.24.
- Source: [Taming](https://arxiv.org/html/2406.15643)

### Inferences
- **The SH trick suits the CPU trainer best.** At SH degree 3, 45 of the 59 floats per Gaussian are higher-order SH. Updating them every 16 steps cuts the per-step Adam traffic over all Gaussians by roughly 70%: about 35 ms → about 10–12 ms per step, or about 13–14% of a step.
- Score densification plus the budget curve is a cleaner way to hold 400k than gradient thresholding with a hard cap. On its own it does not reduce per-pixel work.

### Gaps
- No pixel-count breakdown is given.

## Q4. Mini-Splatting (ECCV 2024) and Mini-Splatting2 / "Fang & Wang 2026" (arXiv 2411.12788)

### Takeaway
Mini-Splatting2 grows the model early and aggressively: it clones critical Gaussians every 250 iterations from 500 to 3K and reinitializes from depth at 2K. It then culls occluded Gaussians per view during rasterization from iteration 500 to 13K, and trains for 18K steps. On an RTX 3090 Mip-NeRF 360 trains in 3 min 34 s with 0.62M Gaussians, and SSIM/LPIPS are better than Taming-S. The v2 revision of arXiv 2411.12788 is retitled "Efficient Scene Modeling via Structure-Aware and Region-Prioritized 3D Gaussians", and is very likely the "Fang and Wang 2026" journal version.

### Cited Findings
- **Densification and culling rules:**
  - Blur split: split a Gaussian when its maximum-contribution pixel area exceeds θ = 2e-4·H·W.
  - Depth reinitialization: points come from the ray–ellipsoid midpoint of the maximum-contribution Gaussian.
  - Critical Gaussians (0.99 quantile of maximum blending weight) are each cloned into 2, every 250 iterations from 500 to 3K. Depth reinitialization happens at 2K.
  - Visibility culling: a Gaussian is visible if its summed blending weight exceeds a 0.99-quantile threshold. Non-visible Gaussians are skipped during rasterization from iteration 500 to 13K.
  - Simplification keeps Gaussians that are the main contributor to at least one pixel, plus importance-weighted sampling.
  - Densification starts at 0.5x resolution.
  - Source: [Mini-Splatting2 arXiv HTML](https://arxiv.org/html/2411.12788)
- **Results on Xeon 8383C + RTX 3090** (PSNR / SSIM / LPIPS / time / #Gaussians). Source: [MSv2](https://arxiv.org/html/2411.12788)

  | Dataset | 3DGS | 3DGS-accel | Taming-S | MSv2 |
  |---|---|---|---|---|
  | Mip-NeRF 360 | 27.47 / 0.815 / 0.216 / 27m06s / 3.35M | 10m02s | 27.25 / 0.795 / 0.259 / 6m28s / 0.67M | 27.33 / 0.821 / 0.215 / 3m34s / 0.62M |
  | T&T | 23.66 / – / – / 15m06s / – | – | – | 23.14 / 0.841 / 0.186 / 2m22s / 0.35M |
  | DB | 29.54 / – / – / 24m51s / – | – | – | 30.08 / 0.912 / 0.240 / 2m45s / 0.65M |

  - The authors blame the T&T PSNR loss of 0.5 dB on sky and textureless regions, where depth reinitialization is unreliable.
- [Search results](https://arxiv.org/html/2411.12788v2) show the v2 title "Efficient Scene Modeling via Structure-Aware and Region-Prioritized 3D Gaussians". No separate 2026 paper by Fang & Wang was found.

### Inferences
- **CPU transfer:**
  - The schedule that stops densification at 3K suits 7K-step CPU runs.
  - Visibility culling removes occluded Gaussians from each view's blend lists. This is a real per-pixel saving on CPU, because occluded Gaussians still cost sorting and tile-list entries.
  - Depth reinitialization needs only one depth render.
  - Given the trainer's poor outdoor history, the T&T sky regression is a risk.

### Gaps
- The venue and year of "Fang & Wang 2026" are not confirmed.
- I did not re-fetch the original Mini-Splatting (ECCV 2024) numbers this session.

## Q5. Turbo-GS (Dhiman et al., arXiv 2412.13547, CVPR 2026) and TurboGS (Dong et al., arXiv 2606.15924, ICML 2026): pixel subsampling

### Takeaway
These are two different papers.
- **Turbo-GS** uses chessboard "dilated rendering" with width/height offsets, plus a low-pass filter. On Bicycle it costs only 0.04 dB against full resolution, while a multi-scale (coarse-to-fine) schedule costs 0.22 dB.
- **TurboGS** samples a fraction of the pixels per 16x16 tile: 40% hard pixels drawn by error, 30% stalest pixels, 30% mixed. It reaches 3DGS PSNR on Mip-NeRF 360 in 77 s on an RTX 5090. Error-guided sampling beats uniform random sampling by 0.47 dB at the same time.

Both methods cut per-pixel work at every step.

### Cited Findings
- **Turbo-GS densification and budget:**
  - Position-gradient threshold plus color-gradient threshold (τ_color = 0.01·τ_pos). The color criterion fires with probability 0.2. Only Gaussians visited more than τ_v times may spawn.
  - Budget: B(t) = N + (t^α − 1)/(100^α − 1)·(M − N), with α adapted by a tanh of the recent-vs-history loss decline (λ=0.5).
  - M = 300K or 500K for low-resolution data, 700K for 4K.
  - Run: 10K iterations, densify every 20 steps during the first 3K after a 300-step warm-up, batch of 4 for the last 50 iterations.
  - Source: [Turbo-GS arXiv HTML](https://arxiv.org/html/2412.13547v2)
- **Dilated rendering:**
  - Renders a chessboard subset of pixels with width/height offsets.
  - Used throughout densification, then applied randomly.
  - Screen-space low-pass filter: add 0.3 + 0.5(p−1) to the covariance diagonal, where p is the pattern size.
  - Source: [Turbo-GS](https://arxiv.org/html/2412.13547v2)
- **Bicycle ablation** (PSNR / SSIM / LPIPS):
  - Full resolution: 25.51 / 0.770 / 0.216
  - Multi-scale: 25.29 / 0.755 / 0.237
  - Dilated: 25.47 / 0.770 / 0.216
  - No time is reported for dilated rendering (Fig. 9 shows only qualitative bars).
  - Source: [Turbo-GS](https://arxiv.org/html/2412.13547v2)
- **Turbo-GS results on RTX A6000:**
  - Mip-NeRF 360: 27.38 / 0.812 / 0.210 in 5.28 min with a peak of 0.49M Gaussians, against 3DGS 27.45 / 0.814 / 0.217 in 30.08 min (2.71M).
  - DB: 30.41 / 0.910 / 0.239 in 3.24 min (0.39M).
  - 4K Mip-NeRF 360: 12 min against 113 min for 3DGS and 28 min for Taming.
  - T&T Train scene: −0.8 dB against 3DGS.
  - Fig. 1 and Table 2 disagree (13 vs 12 min; Taming 40 vs 28 min).
  - Source: [Turbo-GS](https://arxiv.org/html/2412.13547v2)
- **TurboGS pixel sampling:**
  - Each S×S tile (e.g. 16×16) gets Npix = max(1, ⌊r_t·S²⌋) pixels: 40% drawn by error (r_hard = 0.4), 30% the oldest unsampled pixels (r_stable = 0.3), and 30% mixed.
  - The error map is updated by EMA (β=0.4) only at sampled pixels.
  - Each step uses K views (K_max=10, K_min=3): an anchor view with the highest error, plus geometry-weighted neighbors.
  - r_t and K are scheduled by error ratio and training progress, with sinusoidal modulation.
  - The loss adds a sparse tile NCC term. The optimizer uses Levenberg–Marquardt for geometry, DC color and opacity, and Adam for higher-order SH.
  - Run: 5000 iterations (8000 for Big), densify from 200 to 3500.
  - Hardware: RTX 5090.
  - Source: [TurboGS arXiv HTML](https://arxiv.org/html/2606.15924)
- **TurboGS results:**
  - Mip-NeRF 360: 77 s, 27.57 / 0.794 / 0.256, 0.64M Gaussians, against 3DGS at 891 s and 27.55.
  - T&T: 73 s, 23.79 / 0.832 / 0.200 (3DGS 23.84).
  - DB: 62 s, 30.00 / 0.900 / 0.277.
  - Big: 27.99 on Mip-NeRF 360 in 168 s.
  - Source: [TurboGS](https://arxiv.org/html/2606.15924)
- **TurboGS ablations (Mip-NeRF 360):**
  - Random pixel sampling: 75 s, 27.10 PSNR. Error-guided sampling: 77 s, 27.57.
  - Sparse pixel sampling alone: 85 s, 26.58.
  - r_hard sweep: 0.2 → 27.29, 0.4 → 27.57, 0.6 → 27.33, 0.8 → 27.41.
  - Sparse Adam only versus Moment-LM: 27.52 versus 27.57.
  - Source: [TurboGS](https://arxiv.org/html/2606.15924)

### Inferences
- **Dilated rendering has the best measured quality per pixel saved of anything found here.** Its measured cost was −0.04 dB versus −0.22 dB for multi-scale. Together with PennyGS's 2x2 stride with a cycling offset, this backs strided subsampling with rotating offsets as the main CPU lever.
- **Use the low-pass filter.** The 0.3 + 0.5(p−1) covariance dilation is cheap to add and should be used with stride subsampling.
- **Error-guided per-tile sampling costs extra on CPU.** It needs an error EMA map and gather/scatter, but it gave +0.47 dB over random sampling at the same pixel budget.
- **Sparse rendering forces scattered pixel evaluation.** A CPU rasterizer that rasterizes by tile must evaluate only sampled pixels. A strided pattern keeps SIMD lanes regular; a random scatter does not.

### Gaps
- Neither paper gives the time saved by subsampling alone.
- The range of TurboGS's sampling rate r_t is not given in the text I read.

## Q6. Structure-aware densification (Lyu et al., arXiv 2604.28016, SIGGRAPH 2026)

### Takeaway
This method computes multiscale structure tensors of the input images once. It then splits Gaussians anisotropically into grids of children sized to the local image wavelength, which allows short runs: 3K steps for Mip-NeRF 360 and DB. It is about 18x faster than 3DGS on an H100 and has the best SSIM and LPIPS, but it ends with many Gaussians (4.05M on Mip-NeRF 360).

### Cited Findings
- **Image analysis:** structure tensors at L=4 blur levels (σ_l = 1.5^l), aggregated with energy weights (γ=3). Local wavelength: Λ_min = 1/(√λ₁ + ε). Source: [arXiv 2604.28016 HTML](https://arxiv.org/html/2604.28016)
- **Split rule:**
  - Per axis, η_k = ‖v_k‖/Λ_min.
  - Split if η > 1 in more than 80% of views. Each axis gets ⌈√η⌉ subdivisions, with scales divided by that number.
  - Runs every 500 iterations, plus AbsGS densification every 100 iterations.
  - Source: [2604.28016](https://arxiv.org/html/2604.28016)
- **Run length:** 3K iterations for Mip-NeRF 360 and DB, 7K for T&T. Batch size 2 indoors, 1 outdoors. Source: [2604.28016](https://arxiv.org/html/2604.28016)
- **Results on one H100:**
  - Mip-NeRF 360: 52.96 s, 27.25 / 0.821 / 0.197, 4.05M Gaussians.
  - DB: 41.22 s, 29.89 / 0.910 / 0.238, 1.81M.
  - T&T: 84.44 s, 23.60 / 0.857 / 0.147, 1.86M.
  - Speedup: about 18x, 23x and 6.8x over 3DGS; about 2.7x, 2.8x and 1.5x over FastGS-Base.
  - Source: [2604.28016](https://arxiv.org/html/2604.28016)

### Inferences
- **The value for CPU is the shorter run.** The split rule is computed once from the images, so it adds no per-step pixel cost. Finishing in 3K steps instead of 7K cuts total per-pixel work by about 2.3x.
- **But it needs millions of Gaussians,** far beyond the 400k cap. Its PSNR is also below FastGS.
- **One part transfers well:** using the wavelength test as the split gate inside a capped budget.

### Gaps
- No result under a primitive cap.

## Q7. Other densification papers: Improved ADC, AbsGS, 3DGS-MCMC, EfficientGS, group training, Grubert 2025, Elrawy & Mohammed 2025, Gao 2025

### Takeaway
Most of these mainly change where Gaussians go, not how many pixels are rendered. Group training (Wang et al., ICCV 2025) renders only 60% of the Gaussians per step and gains 27–30% on an RTX 3090 with slightly better PSNR, but that is a per-splat saving. Grubert 2025 claims more than 2x faster training through an exponentially rising gradient threshold.

### Cited Findings
- **Group training (Wang et al., arXiv 2412.07608, ICCV 2025):**
  - Rule:
    - An Under-training ratio of 0.6 of Gaussians is sampled by opacity (p_i = α_i/Σα).
    - The rest are cached and excluded from rendering and optimization, and groups are reshuffled every 500 iterations.
    - Merges happen at 14.5K and 29K.
  - Results with 3DGS on RTX 3090:
    - Mip-NeRF 360: 26.72 → 19.56 min, 27.445 → 27.564 PSNR.
    - T&T: 15.00 → 10.95 min, +0.156 dB.
    - DB: 23.90 → 16.85 min, +0.16 dB.
  - Per-pixel savings are argued in theory (higher-opacity Gaussians saturate transmittance sooner) but not measured.
  - Source: [Group Training arXiv HTML](https://arxiv.org/html/2412.07608v3)
- **Grubert, Barthel, Hilsmann, Eisert, "Improving Adaptive Density Control for 3D Gaussian Splatting" (VISAPP 2025):**
  - Corrected scene-extent computation, an exponentially rising gradient threshold, and significance-aware pruning.
  - The repository claims "more than twice as fast" training and better quality than 3DGS.
  - I did not verify the numbers.
  - Source: [GitHub fraunhoferhhi/Improving-ADC-3DGS](https://github.com/fraunhoferhhi/Improving-ADC-3DGS)
- **Elrawy & Mohammed 2025 (arXiv 2510.10257):** opacity-gradient density control for few-shot 3DGS. It combines a delayed, conservative pruning schedule with a hard primitive budget. It targets compactness in few-shot scenes, not training speed for dense captures. Source: [ResearchGate listing](https://www.researchgate.net/publication/396460537_Opacity-Gradient_Driven_Density_Control_for_Compact_and_Efficient_Few-Shot_3D_Gaussian_Splatting)
- **Related papers found but not read:**
  - Faster-GS (arXiv 2602.09999), claiming "up to 5x faster training"
  - AdpSplit (arXiv 2605.06876), error-driven adaptive splitting
  - Decomposing Densification (arXiv 2507.20239)
  - Source: [search results](https://arxiv.org/pdf/2602.09999), [AdpSplit](https://arxiv.org/pdf/2605.06876), [2507.20239](https://arxiv.org/pdf/2507.20239)
- **Improved ADC, AbsGS, 3DGS-MCMC, EfficientGS:** listed among the related densification works in the search results ([Revising Densification, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8041_ECCV_2024_paper.php)).
  - DashGaussian reports Revising-3DGS* + Dash: Mip-NeRF 360 5.73 → 3.46 min, DB 4.06 → 2.21, T&T 4.50 → 3.16 ([DashGaussian](https://arxiv.org/html/2503.18402)).
  - AbsGS's absolute gradient is used for splitting in FastGS (6.85 vs 7.10 min on Mip-NeRF 360) and in 2604.28016 ([FastGS](https://arxiv.org/html/2511.04283)).

### Inferences
- **Group training mostly saves per-splat work,** so it will gain little on CPU. One exception: the CPU's 35 ms Adam would also shrink by 40% if cached Gaussians are skipped in the optimizer.
- **3DGS-MCMC:** given its earlier outdoor losses on the GPU trainer, there is no reason to revisit it.

### Gaps
- I did not fetch the full texts this session for Improved ADC (Rota Bulò 2024, arXiv 2404.06109), AbsGS (arXiv 2404.10484), 3DGS-MCMC (arXiv 2404.09591) or EfficientGS. Their exact rules and numbers need checking from the primary sources.
- **Gao 2025:** I could not identify which paper is meant.
- **Grubert 2025:** the exact numbers are unverified.

## Q8. Synthesis: coarse-to-fine and pixel subsampling, per-pixel versus per-splat savings, CPU ranking

### Takeaway
The techniques that cut total per-pixel work over the whole run are:
- Resolution scheduling (DashGaussian)
- Strided or dilated pixel subsampling with rotating offsets (Turbo-GS, PennyGS)
- Error-guided sparse pixel sampling (TurboGS)
- Shorter runs enabled by better early densification (2604.28016 at 3K steps, MSv2 stopping densification at 3K)

Densify and prune scoring (FastGS, Taming, MSv2 culling, group training) mainly cuts per-splat work and the Gaussian count, which matters less on a CPU where pixels dominate. Most of their benefit is already captured by the 400k cap.

### Cited Findings
- **Measured quality costs of resolution and pixel schemes:**
  - Turbo-GS on Bicycle: multi-scale −0.22 dB, dilated −0.04 dB ([Turbo-GS](https://arxiv.org/html/2412.13547v2)).
  - DashGaussian, resolution schedule only: +0.38 dB and −20% time on the Taming backbone ([DashGaussian](https://arxiv.org/html/2503.18402)).
  - TurboGS: random sparse sampling costs 0.47 dB against error-guided sampling ([TurboGS](https://arxiv.org/html/2606.15924)).
  - The two coarse-to-fine results disagree in sign (Turbo-GS −0.22 dB; DashGaussian +0.38 dB). Their implementations and backbones differ.
- **Per-splat techniques:**
  - FastGS VCD: 2x faster with 80% fewer Gaussians ([FastGS](https://arxiv.org/html/2511.04283)).
  - Group training: 27–30% faster with 60% of the Gaussians active ([Group Training](https://arxiv.org/html/2412.07608v3)).

### Inferences
- **Recommended CPU ranking:**
  1. **Strided 2x2 subsampling with rotating offset, plus the low-pass covariance dilation.** This is already in PennyGS. Expected about 2–3x on per-pixel time at about −0.04 dB (Turbo-GS Bicycle).
  2. **DashGaussian log-frequency resolution schedule ending at full resolution,** combined with (1) in the late phase. Estimated additional total saving of about 30%. It needs re-validation at 7K steps.
  3. **Taming's higher-order SH Adam every 16 iterations:** about 20+ ms per step.
  4. **Densification that finishes early** (MSv2 or 2604.28016 style), allowing 3–5K-step runs.
  5. **FastGS-style footprint-error scoring computed from training renders,** for better placement under the 400k cap.
  6. **Error-guided per-tile sampling (TurboGS)** if the error-map overhead stays small on CPU.

- **Uncertainty:** these estimates scale GPU ratios by the CPU's cost split. None has been measured on CPU.

### Gaps
- No paper measures any of these on CPU.
- No paper isolates the wall-clock saving of dilated or strided rendering alone.
