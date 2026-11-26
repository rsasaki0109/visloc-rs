# Optimizer-side and selective-update techniques for faster 3DGS training (2024-2026), and transfer to a CPU-only trainer

Scope note: about 15 sources were fetched (arXiv abstract and HTML full text, plus the gsplat source). Numbers are quoted from the papers with their hardware. Anything I could not check against a primary source is marked **[UNVERIFIED]**. CPU-trainer reference figures, as given in the brief: 4 vCPU, ~22 GB/s, 400k gaussians x 59 floats, dense Adam ~35 ms of a ~175 ms step (~770 MB/step), fwd+bwd raster ~100 ms, 7,000 steps.

Arithmetic check used throughout (my own inference): 770 MB / 22 GB/s = 35 ms, so dense Adam already runs at the memory-bandwidth limit. The only ways to make it cheaper are touching fewer bytes (sparsity, fewer or narrower moments) or taking fewer steps. Per gaussian and per step, fp32 Adam moves about 59 x (param r+w 8 B + m r+w 8 B + v r+w 8 B + grad r 4 B) = 1,652 B, which is about 661 MB for 400k gaussians. The remainder of the ~770 MB is presumably grad zeroing and other traffic.

## Q1. Second-order and non-Adam optimizers: step-count reductions, wall-clock, extra forward/backward passes

### Takeaway
Only Lan et al.'s 3DGS² (per-gaussian local Newton) gives a large end-to-end wall-clock gain: 5.1-6.4x versus vanilla 3DGS on an RTX 3090, from about 10x fewer iterations at 20-25% more cost per iteration. That is the kind of saving that matters on a CPU, because it cuts the number of raster passes. The LM-family methods (3DGS-LM, LM-RS) give only 1.2-1.3x and need 15-53 GB of Jacobian/gradient caches. 3DGS²-TR (diagonal Gauss-Newton via Hutchinson) halves the iteration count at about 10% per-iteration overhead, but its published wall-clock is slower than Adam.

### Cited Findings
**3DGS-LM (Höllein, Božič, Zollhöfer, Nießner; ICCV 2025; arXiv 2409.12892)**
- Method: replaces Adam with Levenberg-Marquardt for the final phase only. Adam/3DGS runs for 20K iterations (densification ends at 15K), then "5 LM iterations with 8 PCG iterations each". A gradient cache built in a per-pixel rasterizer pass lets custom CUDA kernels compute J·v and Jᵀ·v without re-rendering. Each LM iteration solves several image batches separately and combines the update directions as a weighted mean: 25 images in 4 batches (MipNeRF360), 25 in 3 (Deep Blending), 70 in 3 (T&T). The line search uses a 30% image subset — [arXiv HTML](https://arxiv.org/html/2409.12892v2)
- Wall-clock (A100), base vs +LM: MipNeRF360 3DGS 1271 -> 972 s, DISTWAR 966 -> 764 s, gsplat 1064 -> 818 s, Taming 566 -> 453 s. T&T 3DGS 736 -> 663 s, Taming 366 -> 310 s. DB 3DGS 1222 -> 951 s, Taming 447 -> 347 s. The abstract states "20% faster than the original 3DGS while obtaining the same reconstruction quality" — [arXiv HTML](https://arxiv.org/html/2409.12892v2), [abs](https://arxiv.org/abs/2409.12892)
- Quality deltas are within about ±0.2 dB PSNR. Examples: MipNeRF360/3DGS PSNR −0.01, LPIPS +0.003; DB/3DGS PSNR +0.21 — [arXiv HTML](https://arxiv.org/html/2409.12892v2)
- Memory: "On average our method consumes 53 GB of GPU memory", against 6-11 GB for the baselines — [arXiv HTML](https://arxiv.org/html/2409.12892v2)
- Pass count: each LM iteration needs a cache-building render per batch plus line-search renders. PCG reuses the cache. The paper gives no explicit pass count — [arXiv HTML](https://arxiv.org/html/2409.12892v2)

**3DGS² — "Near Second-order Converging 3D Gaussian Splatting" (Lei Lan et al.; arXiv 2501.13975, Jan 2025)**
- Method: a "stochastic local Newton" step per gaussian, run Jacobi-style in parallel across kernels, with analytic small Hessians per attribute group. Solve order: position (2-DOF, in the plane perpendicular to the view ray), then rotation (a single angle about the view direction) and scale (in the eigen-space of the projected 2D covariance), then opacity (scalar Newton with a log barrier), then SH colour (linear). Step size 1.0 with no line search. To prevent overshoot, the K=3 nearest-neighbour views, downsampled, are added to the local objective as secondary targets — [arXiv HTML](https://arxiv.org/html/2501.13975v2)
- Steps: "over 10× fewer iterations" / "one-tenth iterations". Per-iteration cost: "only 20%–25% slower compared with one gradient descent" step — [arXiv HTML](https://arxiv.org/html/2501.13975v2)
- Wall-clock (i7-12700 + RTX 3090), ours vs vanilla 3DGS: MipNeRF360 256 vs 1307 s (5.1x), T&T 131 vs 695 s (5.3x), DB 189 vs 1210 s (6.4x) — [arXiv HTML](https://arxiv.org/html/2501.13975v2)
- Quality (SSIM/PSNR/LPIPS): Mip360 0.876/29.42/0.168 vs 0.871/29.18/0.183. T&T 0.871/24.43/0.154 vs 0.853/23.71/0.169. DB 0.927/30.43/0.228 vs 0.907/29.90/0.238. Taming is still slightly better on Mip360 (29.48) — [arXiv HTML](https://arxiv.org/html/2501.13975v2)
- The paper reports no memory figures and no public absolute iteration counts. The speed claims are internally inconsistent ("constantly 10×" in one place, "5× to 10×" in another) — [arXiv HTML](https://arxiv.org/html/2501.13975v2)

**LM-RS — "Matrix-free Second-order Optimization of Gaussian Splats with Residual Sampling" (Pehlivan et al., MPI-INF; arXiv 2504.12905, v3 Jan 2026; no venue shown)**
- Method: LM with a matrix-free Jacobi-preconditioned PCG. JVPs use forward-mode dual numbers. Views are sampled by K-means on pose (batch 16, rising to 32 after iteration 50). Residuals are sampled with stratified N=32 pixels per tile. PCG runs 5 iterations, rising to 8. Damping λ=0.1. Densification is disabled, and SSIM uses a diagonal approximation — [arXiv HTML](https://arxiv.org/html/2504.12905)
- Steps: 200 LM iterations against 10,000 Adam iterations — [arXiv HTML](https://arxiv.org/html/2504.12905)
- Mip-NeRF360 (A100): Adam 252 s / 5.81 GB. 3DGS-LM 398 s / ~50 GB. LM-RS 190 s / 14.75 GB (≈1.3x vs Adam). Full LM without residual sampling takes 810 s. NeRF-synthetic (A40): Adam 58 s vs LM-RS 13 s — [arXiv HTML](https://arxiv.org/html/2504.12905)
- Quality: Mip360 PSNR 28.59 vs 28.21 (Adam) — [arXiv HTML](https://arxiv.org/html/2504.12905)

**3DGS²-TR (Hsiao, Fang, Huang, Lavaei, Demmel, Shao — UC Berkeley; Li, Rabeti, Gojcic — NVIDIA; arXiv 2602.00395, 2026)**
- Method: Sophia-style diagonal Gauss-Newton. diag(JᵀJ) is estimated with Hutchinson (Rademacher vectors) every l=10 iterations. The update is Δx = −D̂⁻¹ĝ, with EMAs (θ₁=0.9, θ₂=0.999), and is clipped to a per-parameter trust region derived from the Hellinger distance. It is O(n), like Adam. Each Hessian-vector product costs "exactly one forward and backward pass", for about 10% overhead versus Adam. No densification — [arXiv HTML](https://arxiv.org/html/2602.00395)
- Steps: better quality "using 50% fewer training iterations compared to ADAM" — [arXiv HTML](https://arxiv.org/html/2602.00395)
- Wall-clock (A100-80GB; naive PyTorch Sophia), Mip360: ADAM 212 s, 3DGS²-TR 652 s, 3DGS-LM 2602 s. PSNR @30k: Mip360 25.08 -> 25.39, DB 27.36 -> 27.74, T&T 21.66 -> 22.85. Peak memory 5.60 vs 6.51 GB. The 7k-iteration T&T gain quoted in the text (0.56 dB) does not match its own table — [arXiv HTML](https://arxiv.org/html/2602.00395)

### Inferences
- For a CPU trainer whose time is about 57% raster (100/175 ms), only methods that cut the number of fwd/bwd passes matter. 3DGS² cuts passes about 10x at +20-25% per pass. If that transferred, 7,000 steps would become roughly 700-1,000 steps. However, its per-gaussian Newton needs per-gaussian accumulation of Hessian blocks over pixels (more backward work per pixel) plus extra renders of 3 downsampled neighbour views. On a CPU, the +20-25% GPU figure may grow, because the extra per-pixel accumulation hits the same scatter/atomic bottleneck as the backward pass. Implementation effort is high (custom analytic Hessians for every attribute), and no reference code was verified.
- LM methods (3DGS-LM, LM-RS) are poor fits. They need 15-53 GB of caches, are memory-bandwidth heavy (bad at 22 GB/s), and give only 1.2-1.3x on GPU. LM-RS's residual sampling idea (32 px per tile) is interesting on its own as a way to cut per-pixel work.
- 3DGS²-TR is cheap to implement (Adam-shaped: one EMA plus a diagonal), but it adds a fwd+bwd pass every 10 steps (+10%). Its "50% fewer iterations" claim is measured without densification and against a baseline that took 212 s, so it is not proven in a 7k-step, densifying regime.

### Gaps
- "Zhang 2026" second-order/optimizer paper: not located within the call budget. **[UNVERIFIED — not found]**
- No public code for 3DGS² was confirmed. Its per-iteration breakdown and memory are unreported.
- None of these papers report results at a 7k-step budget against a tuned Adam with an identical densification schedule.

## Q2. Sparse/selective Adam, fewer moments, lr tricks

### Takeaway
"Sparse Adam" (Taming 3DGS, gsplat `SelectiveAdam`) freezes both moments and params of invisible gaussians (β' = β·V + (1−V)). This is faster but measurably lossy on vanilla 3DGS: Mip360 PSNR 27.507 -> 27.285 in one 2026 study. AdamW-GS (2601.16736) repairs it by periodically rescaling moments and decoupling the opacity/scale L1 regularizers from the moments (AdamW-style). That directly addresses the known "regularizers don't reach unseen gaussians" problem, and the paper also reports 40%+ total runtime reductions.

### Cited Findings
- Sparse Adam formulation: β' = β×V + (1−V), with V the visibility filter. Only visible primitives' moments and params are updated, so there are no "implicit updates" from momentum carry-over — [Ding et al., arXiv 2601.16736](https://arxiv.org/html/2601.16736v2)
- Vanilla 3DGS, Mip360, A6000: Adam PSNR 27.507 / SSIM 0.815 / LPIPS 0.216, 3.331M prims, 0.232M dead. Sparse Adam 27.285 / 0.809 / 0.228, 2.532M prims, 0.039M dead. Sparse Adam is "less explorative" — [arXiv 2601.16736](https://arxiv.org/html/2601.16736v2)
- AdamW-GS = Sparse Adam + Re-State Regularization (at milestone intervals, sample primitives and scale m -> α₁m, v -> α₂v with α<1) + Decoupled Attribute Regularization: θ ← θ − η[m̂'/(√v̂'+ε) + min(λ(∇R/N_I)/(√v̂'+ε), C_t)], with R = opacity/scale L1 kept out of the moments — [arXiv 2601.16736](https://arxiv.org/html/2601.16736v2)
- AdamW-GS results (A6000): vanilla 3DGS Mip360 27.506 -> 27.678 PSNR, 30.58 -> 18.53 min, ~49.3% fewer primitives. 3DGS-MCMC 27.948 -> 28.219, 46.81 -> 39.77 min. Taming 20.30 -> 10.46 min. Sparse Adam alone cuts step time by "approximately 50%" in MCMC and gives ">40% reduction in total runtime" in vanilla 3DGS. Part of this comes from fewer primitives, not only from cheaper optimizer steps — [arXiv 2601.16736](https://arxiv.org/html/2601.16736v2)
- gsplat `SelectiveAdam`: `step(visibility)` passes an external per-gaussian visibility mask into a single fused CUDA `adam` kernel ("fused into a single kernel", "optimized sparse updates"). The docstring says it is "one of the two optimizers mentioned in the Taming3DGS paper". The Python file has no documented speedup and does not show how the moments of masked rows or the bias-correction step are handled. That logic lives in the CUDA kernel — [gsplat selective_adam.py](https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/gsplat/optimizers/selective_adam.py)
- **[UNVERIFIED, from prior knowledge]** gsplat's kernel skips masked rows entirely (moments untouched, no decay), and bias correction uses the global step count rather than a per-gaussian count.
- Taming 3DGS (Mallick et al., 2024, arXiv 2406.15643) introduced sparse Adam. I did not extract an isolated speedup figure for sparse Adam alone — [Taming HTML](https://arxiv.org/html/2406.15643v1)

### Inferences
- CPU byte budget: with ~33% visibility (PennyGS), sparse Adam touches about 1/3 of the 770 MB, so about 12 ms instead of 35 ms if visible rows are gathered contiguously (tile-ordered or sorted indices). That saves about 23 ms/step (~13% of the step). Random-gather overhead on 4 vCPUs could eat part of this.
- Fewer or narrower moments (dense): bf16 m and v cut per-gaussian traffic from ~1,652 to ~1,180 B (−29%, about −10 ms). Dropping m for the 45 SH-rest floats (RMSprop-style, or SGD with a per-tensor lr) removes a further 45 x 8 B. These options stack with sparsity. PennyGS already found bf16 moments cost ~0.34 dB on ETH3D, so momentum-free SH is the safer byte cut to test next. **[inference; no 3DGS paper with a measured Adafactor/momentum-free result was found]**
- For the regularizer problem, use the DAR pattern: apply opacity/scale L1 as a decoupled, moment-free step, every k steps over all gaussians. Only the opacity and 3 scale floats are touched (4/59 of the data), so it is cheap. A closed-form decay for L1 on log/logit parameters is another option. Periodic moment re-state (RSR) is also cheap.

### Gaps
- No paper isolates sparse-Adam optimizer-kernel time from the effect of a smaller primitive count.
- No 3DGS study of Adafactor or factored second moments was found. **[gap]**

## Q3. Freezing and computation caching (ElsaGS, arXiv 2611.22059) and related

### Takeaway
ElsaGS progressively freezes up to 90% of gaussians, chosen by the lowest densification-gradient statistic, from 6k iterations onward. Frozen gaussians are excluded from all optimization. Contiguous runs of frozen gaussians are cached per tile as group colour and transmittance, so both forward and backward skip them. The algorithm alone (ElsaGS-sw, on GPU) gives 1.9x at about −0.2 dB. The 18.7-20.6x figures need the custom accelerator.

### Cited Findings
- Authors: He Zhu, …, Jingwen Leng, Yu Feng, Minyi Guo. cs.AR. Algorithm-hardware co-design — [arXiv abs](https://arxiv.org/abs/2611.22059)
- Freezing criterion: rank by the same gradient statistic used for densification (after removing low-opacity gaussians) and freeze the smallest. Freezing starts after 6k iterations, at each densification event. Target fraction ρ_t = min(ρ, κ·max((N_curr−N_init)/N_curr, 0)), with ρ=0.9 and κ=2.0. A gaussian passes through a "transitional" state for one interval before it is frozen. At finetune start the total is pushed to ρ, and the set is then fixed. New densified gaussians start active — [arXiv HTML](https://arxiv.org/html/2611.22059)
- Frozen: "All attributes of a frozen Gaussian are excluded from further optimization", so no gradients and no Adam. Cached: per tile, contiguous groups of frozen gaussians keyed by (image ID, tile ID, leading gaussian ID, group length). Each entry holds a 16x16 tile of accumulated colour and transparency, with T_group = ∏(1−α_i) and C_group used in the backward pass — [arXiv HTML](https://arxiv.org/html/2611.22059)
- Speed and quality: ElsaGS-sw (GPU-only) is 1.9x faster than the GPU baseline. ElsaGS-hw averages 18.7x (up to 20.6x) over an Orin mobile Ampere GPU, with 99.4% energy reduction. PSNR is 0.2 dB below canonical 3DGS. Random or abrupt freezing loses an extra 0.5-0.6 dB. On Taming3DGS: 19.0x with −0.4 dB. Datasets: Mip-NeRF360, T&T, DeepBlending. Fwd+bwd rasterization is ">80% of execution time" — [arXiv HTML](https://arxiv.org/html/2611.22059)

### Inferences
- On a CPU trainer: freezing removes frozen rows from Adam (up to 90% of rows late in training), which compounds with sparse Adam. Caching per-(image, tile) group contributions requires every training image's tiles to be cached, at 16x16 x (RGB+T) per group per tile per image. Memory grows with the number of images and gets stale whenever an active gaussian in front changes the sort order. On CPU, the practical part is simpler: skip frozen gaussians in the backward pass and in Adam, and keep forward compositing over them. That gives an optimizer saving of up to ~90% late in training and part of the backward saving. A plausible gain is 1.3-1.9x over the late phase, by analogy with ElsaGS-sw. **[inference]**
- Freezing starts at 6k of their 30k schedule. Under a 7k-step budget, the schedule must be compressed (for example, starting freezing around 1.5-2k). How quality behaves under that compression is untested.

### Gaps
- Per-stage GPU breakdown (Fig. 4) and per-dataset PSNR (Fig. 13) are figures only and were not extracted. The "1.6× over Arc" claim is ambiguous in the text.
- Other "freeze converged gaussians" papers were not surveyed within the budget.

## Q4. CPU-side Adam in offloading systems (CLM, GS-Scale, TideGS, Lee 2026)

### Takeaway
GS-Scale's deferred optimizer update is the most directly transferable technique. It skips gaussians with zero gradient and catches up their Adam state in closed form when they reappear, using a per-gaussian 8-bit skip counter (MAX 15). That cuts CPU memory traffic to roughly the visible fraction while staying numerically close to dense Adam, at negligible quality cost. CLM uses ZeRO-Offload's CPU Adam (AVX-vectorized in DeepSpeed) on a dedicated thread that updates gaussians as soon as their last micro-batch finishes. Neither paper publishes CPU Adam throughput in GB/s.

### Cited Findings
**GS-Scale (Donghyun Lee, Dawoon Jeong, Jae W. Lee, Hongil Yoon; arXiv 2509.15645, Sep 2025). This is likely the "Lee" offload paper.**
- Offloading design: all gaussians live in host memory. Frustum culling and the optimizer run on the CPU, and only the visible subset is moved to the GPU. Optimizations: (1) selective offloading of geometric params for fast culling, (2) parameter forwarding to pipeline the CPU optimizer with GPU compute, (3) deferred optimizer updates for zero-gradient gaussians — [abs](https://arxiv.org/abs/2509.15645)
- Deferred update math: if a gaussian has had zero gradient for d iterations, then at step t: m_t = β1^(d+1)·m_{t−d−1} + (1−β1)·g_t, v_t = β2^(d+1)·v_{t−d−1} + (1−β2)·g_t². The skipped weight updates are replayed approximately as w_t ≈ w_{t−d} − m_{t−d−1}/(√v_{t−d−1}+ε)·w_scale, with ε dropped so that w_scale is a precomputable constant. The counter is `char counter[N]` with MAX = 15; the text says "4-bit". Saturation causes "6.7% (1/15) unnecessary updates". Counter fetch is 1 byte per gaussian, against 7D 32-bit accesses for a full update (D = 59) — [arXiv HTML](https://arxiv.org/html/2509.15645)
- Implementation: a custom C++ PyTorch extension with OpenMP. CPU-GPU pipelining uses Python `threading`. No SIMD details are given — [arXiv HTML](https://arxiv.org/html/2509.15645)
- Visibility: "each training iteration utilizes only 8.28% of total Gaussians on average in large-scale scenes". The naive baseline is "around 4×" slower than GPU-only, with the CPU optimizer limited by host bandwidth that is "3×" slower than the GPU's — [arXiv HTML](https://arxiv.org/html/2509.15645)
- Hardware: laptop i7-13620H with 83.2 GB/s host BW and RTX 4070M 8 GB. Desktop i9-13900K with 89.6 GB/s and RTX 4080S. Server 2x Xeon Gold 6530 with 614.4 GB/s and H100 — [arXiv HTML](https://arxiv.org/html/2509.15645)
- Results: all optimizations combined are 4.47x (laptop) and 4.57x (desktop) geomean over baseline GS-Scale. Against GPU-only: 1.22x (laptop) and 0.84x (desktop). GPU memory is 3.3-5.6x lower. On the RTX 4070M, gaussian capacity goes from 4M to 18M, giving 23-35% better LPIPS. Quality vs original: Rubble 26.63 -> 26.62, Building 22.74 -> 22.78, Aerial 27.69 -> 27.66 PSNR (negligible). Per-optimization speedups appear only in Fig. 11 — [arXiv HTML](https://arxiv.org/html/2509.15645), [abs](https://arxiv.org/abs/2509.15645)

**CLM (arXiv 2511.04951)**
- CPU Adam "extends the Zero-offload … implementation to allow updating a subset of Gaussians". It runs on a dedicated thread that releases the GIL and waits on a pinned-memory signal set by DMA. Gaussian g is updated right after its last-touching micro-batch L_g = max{i | g ∈ S_i}, so Adam overlaps later micro-batches' fwd/bwd — [arXiv HTML](https://arxiv.org/html/2511.04951)
- "CPU Adam trailing time" (ms, random/TSP order): Bicycle 147.457/147.457, Rubble 137.27/127.11, Alameda 178.59/186.75, Ithaca 208.34/406.79, BigCity 21.85/23.36. In naive offloading, comms + CPU Adam are ">50%" of training time — [arXiv HTML](https://arxiv.org/html/2511.04951)
- Hardware: Threadripper PRO 5955WX (16C, 128 GB) + RTX 4090, and Xeon E5-2660 v3 (20C) + 2080 Ti. 1.38-1.92x over naive offloading. Per view, "less than 1%" of gaussians are accessed (BigCity average 0.39%, max 1.06%). Can render 100M gaussians on one RTX 4090 — [arXiv HTML](https://arxiv.org/html/2511.04951), [abs](https://arxiv.org/abs/2511.04951)
- **[UNVERIFIED, prior knowledge]** DeepSpeed's ZeRO-Offload CPUAdam is AVX2/AVX-512 vectorized and OpenMP-threaded, with tiled fp32 updates. ZeRO-Offload reports it as roughly 5-7x faster than PyTorch's CPU Adam.

**TideGS (Zhong, Shi, Chen, Sun, Zhao, Yuan, Li; arXiv 2605.20150)**
- SSD-CPU-GPU out-of-core training at more than 1B gaussians on one 24 GB GPU. It uses block-virtualized geometry, a hierarchical async pipeline, and trajectory-adaptive differential streaming that transfers only working-set deltas. The abstract does not say where the optimizer runs and gives no speed numbers — [abs](https://arxiv.org/abs/2605.20150)

### Inferences
- For a 400k-gaussian, 33%-visible CPU trainer, GS-Scale-style deferral reduces Adam traffic to about 33% of 770 MB plus 0.4 MB of counters, so about 12-13 ms instead of 35 ms. That is the same saving as plain sparse Adam, but it keeps dense-Adam semantics (momentum keeps moving unseen gaussians, approximately). It therefore likely avoids most of the Sparse Adam quality loss (−0.22 dB in 2601.16736). It does not fix L1 regularizers, which give a nonzero gradient everywhere. The deferral must treat "photometric grad == 0" as the skip condition and handle the regularizers separately, either with the DAR decoupled update or with an analytic catch-up of d steps of constant-sign L1 drift.
- CLM's "update right after last use" pattern maps to a CPU trainer as fusing Adam into the backward pass per tile batch (PennyGS already does fused backward-projection + Adam). In a CPU-only setting there is no GPU to overlap with, so the gain is cache locality (gradients still in L2), not hiding latency.
- At 22 GB/s with 4 vCPUs, the AVX2/AVX-512 Adam is already bandwidth-bound. More SIMD does not help, and only byte reduction does. Non-temporal streaming stores for params and moments may recover about 10-30% of effective bandwidth by avoiding read-for-ownership. **[inference]**

### Gaps
- No paper reports CPU Adam GB/s or ns/param. TideGS's optimizer placement and the existence of a separate "Lee 2026 offload" paper (beyond GS-Scale) were not verified.

## Q5. Multi-view / batched training (arXiv 2506.12727) and gradient accumulation

### Takeaway
Choi, Kim and Kim use 4 views per batch with one optimizer step per batch. They report 2.4-2.5x lower wall-clock than naive full multi-view training and better PSNR than single-view, but they do not measure optimizer cost. Batching amortizes Adam over B images, so with dense Adam the per-image optimizer cost drops by B. With sparse or deferred Adam, the touched set is the union of B views' visible sets, so the saving is smaller.

### Cited Findings
- Method: mini-batch of 4 views (2/4/8 ablated), averaged gradients, one update per batch. Thread-efficient partial rendering sub-samples pixels per view. 3D distance-aware D-SSIM. Multi-view ADC uses norm-then-add gradient statistics (E1 for split, E2 for clone). The pruning threshold is multiplied by the number of images. No lr change for 3DGS — [arXiv HTML](https://arxiv.org/html/2506.12727)
- Time (RTX 3090, Mip-NeRF360): 3DGS full multi-view 127 min, naive partial 105 min, ours 50 min. MCMC 181 / 135 / 74 min. Quality: Mip360 3DGS 29.28 -> 29.74 PSNR, MCMC 29.83 -> 30.42. T&T 3DGS 23.69 -> 24.02. Time-matched with half the iterations: 3DGS 26 min both, PSNR 29.28 -> 29.43 — [arXiv HTML](https://arxiv.org/html/2506.12727)
- The paper does not measure per-image optimizer cost. Preprocessing overhead grows with the number of images — [arXiv HTML](https://arxiv.org/html/2506.12727)
- 3DGS-LM also combines update directions from several image batches per LM step (25-70 images) — [arXiv HTML](https://arxiv.org/html/2409.12892v2)

### Inferences
- Dense Adam on the CPU trainer at B=4: 35 ms / 4 = 8.75 ms per image, saving about 26 ms per image-step, provided the total number of images seen stays at 7,000 (1,750 optimizer steps). Union visibility for 4 nearby views is likely about 45-60% rather than 33%. Sparse Adam over the union is therefore about 16-21 ms per batch, or 4-5 ms per image. **[inference]**
- The time-matched result (half the iterations, same time, +0.15 dB) suggests batching is quality-neutral to positive in their regime. Whether that holds at 7k images of compute is untested. Their pixel sub-sampling (partial rendering) also cuts per-pixel raster work, which is the dominant CPU cost.

### Gaps
- No direct measurement of optimizer cost per image under batching. No 3DGS paper isolates plain gradient accumulation (without partial rendering).

## Overall judgement for the CPU trainer (ranked by expected wall-clock gain / effort)
- (1) Deferred / sparse Adam with GS-Scale closed-form catch-up plus DAR-style decoupled regularizers (2509.15645 + 2601.16736): Adam 35 -> ~12 ms (−13% step), close to dense quality, and it fixes the regularizer issue. Low-moderate effort. **[inference]**
- (2) Multi-view batching B=2-4, one Adam per batch (2506.12727): optimizer per image cut to 1/B, with possible quality gain. Cheap to try. **[inference]**
- (3) Progressive freezing (2611.22059, software part): from mid-training, skip backward and Adam for up to 90% of gaussians. 1.9x on GPU at −0.2 dB. Needs the 6k/30k schedule compressed to 7k. **[inference]**
- (4) Byte cuts: momentum-free SH-rest, then bf16 v only (PennyGS saw −0.34 dB with bf16 m and v). Up to about −10 ms. **[inference]**
- (5) 3DGS² local Newton (2501.13975): the only path to a >3x end-to-end gain, because it cuts raster passes about 10x. High effort, no verified code, and the CPU per-pass overhead is unknown. A research bet.
- (6) LM / LM-RS / 3DGS²-TR: not recommended for CPU. Memory-heavy, small or negative measured wall-clock gains.
