# Ways to shrink the per-gaussian representation and per-splat compute during 3DGS training (2024-2026), and whether they transfer to a CPU trainer

Context for the transfer judgements is the CPU trainer: pure Rust, 4 vCPU with AVX2, about 22 GB/s, 400k gaussians at 59 floats each (48 of them SH3). That is 400k x 59 x 4 B ≈ 94 MB of parameters. Dense Adam reads and writes the parameters, the gradients and both moments, which comes to about 770 MB/step or about 35 ms. Raster takes about 100 ms per step.

Arithmetic used in the judgements below (my calculation, not from any paper): Adam traffic scales linearly with floats per gaussian, at about 13 MB/step per float per gaussian, or about 0.6 ms/step per float at 22 GB/s.
- 59 → 27 floats (DBS-style) gives about 16 ms Adam.
- 59 → 14 floats (flat RGB) gives about 8 ms Adam.

In standard 3DGS, view-dependent colour is evaluated once per gaussian per view in projection, not per pixel. A replacement colour model therefore changes O(N) work and Adam bandwidth, not the per-(splat, pixel) raster cost. The exception is kernel changes such as Beta, and deferred or reflection shading.

Unless marked otherwise, every number below comes from a fetched primary source. Items marked "(prior knowledge, not re-verified this session)" come from memory and should be checked.

## Q1. Methods that reduce SH or quantize during training (EAGLES, Compact3D/Lee, Reduced 3DGS, LightGaussian, c3dgs, "Gao 2026"), and the training speedups they report

### Takeaway
Of these, only EAGLES, Lee et al. and Reduced 3DGS change the representation during training.
- EAGLES and Lee et al. make training about 0-10% faster or even slower on GPU, because quantization, decoders or hash-grid MLPs add work while shrinking storage.
- Reduced 3DGS culls SH bands once, at 15K iterations. Its training is about 20% faster, mostly from having fewer primitives.
- LightGaussian and c3dgs are post-hoc compressors and do not speed up training.

None of these reports a large training speedup that comes from the colour representation itself.

### Cited Findings

**EAGLES** (Girish, Gupta, Shrivastava; arXiv 2312.04564, ECCV 2024)
- Colour, rotation and opacity are stored during training as quantized integer latents, with dimensions 16 / 8 / 1, and decoded by small MLPs. Position, scale and the base (DC) SH band are not quantized. — [arXiv HTML](https://arxiv.org/html/2312.04564)
- Training is progressive: render scale goes from 0.3 to 1.0 on a cosine schedule over the first 70% of 30K iterations.
- Densification runs every 175 iterations instead of 100, and influence-based pruning removes 15% of gaussians every 5K iterations up to 25K. — [arXiv HTML](https://arxiv.org/html/2312.04564)
- Training time, EAGLES vs 3DGS (re-run by the authors):

  | Dataset | 3DGS | EAGLES |
  |---|---|---|
  | Mip-NeRF360 | 23m20s | 21m34s |
  | T&T | 12m5s | 11m39s |
  | DB | 23m5s | 21m50s |

  That is only about 4-8% faster. The 3DGS paper's own reported time is 41m33s, which makes EAGLES look much faster than it is. — [arXiv HTML](https://arxiv.org/html/2312.04564)
- Peak training GPU memory drops: Bicycle 17.4G → 10G, Truck 8.5G → 5.3G, Playroom 9.6G → 7.1G. — [arXiv HTML](https://arxiv.org/html/2312.04564)
- Quality and storage:

  | Dataset | EAGLES PSNR / SSIM / LPIPS | EAGLES storage | 3DGS PSNR / SSIM / LPIPS | 3DGS storage |
  |---|---|---|---|---|
  | Mip-NeRF360 | 27.23 / 0.81 / 0.24 | 54 MB | 27.45 / 0.81 / 0.22 (re-run) | 745 MB |
  | T&T | 23.37 / 0.84 / 0.20 | 29 MB | 23.61 / 0.84 / 0.18 (reported) | 411 MB |
  | DB | 29.86 / 0.91 / 0.25 | 52 MB | 29.41 / 0.90 / 0.24 (reported) | — |

  The GPU model is not named. — [arXiv HTML](https://arxiv.org/html/2312.04564)

**Compact 3D Gaussian Representation** (Lee et al.; arXiv 2311.13681, CVPR 2024)
- A learnable mask on opacity and scale, trained with a straight-through estimator, prunes gaussians during training.
- SH is replaced by a hash grid over contracted positions plus a 2-layer, 64-channel MLP that takes view direction. The grid has 16 levels with 2 channels each, resolution 16-4096, and 2^19 entries.
- Scale and rotation go through residual VQ with C=64 and L=6, applied only in the last 1K iterations.
- Position and opacity are stored in 16-bit. — [arXiv HTML](https://arxiv.org/html/2311.13681)
- Training is slower than the 3DGS re-run on A100:

  | Dataset | 3DGS re-run (A100) | Lee et al. |
  |---|---|---|
  | Mip-NeRF360 | 24m07s | 33m06s |
  | T&T | 13m51s | 18m20s |
  | DB | 21m52s | 27m33s |

  The paper itself says training is "slightly longer than 3DGS". — [arXiv HTML](https://arxiv.org/html/2311.13681)
- Quality, storage and gaussian count, with the 3DGS re-run in brackets:

  | Dataset | PSNR / SSIM / LPIPS | Storage | Avg. gaussians |
  |---|---|---|---|
  | Mip-NeRF360 | 27.08 / 0.798 / 0.247 (3DGS 27.46 / 0.812 / 0.222) | 48.8 MB | 1.39M (3DGS 3.16M) |
  | T&T | 23.32 / 0.831 / 0.201 | 39.4 MB | — |
  | DB | 29.79 / 0.901 / 0.258 | 43.2 MB | — |

  — [arXiv HTML](https://arxiv.org/html/2311.13681)

**Reduced 3DGS** (Papantonakis et al.; arXiv 2406.17074, I3D/PACMCGIT 2024)
- A sparsity loss on SH runs during training. At 15K iterations, when densification ends, a one-time SH band cull runs per primitive with two tests:
  - colour-variance test: σ < 0.04 → keep only the DC colour;
  - colour-distance test: keep the lowest band q whose distance to the full colour is < 0.04.
- The resulting share of primitives using 0/1/2/3 bands is 89% / 0.1% / 2.7% / 8.2%. — [arXiv HTML](https://arxiv.org/html/2406.17074)
- Floats per gaussian are 14 + 3·Σ(2i+1). With 3 bands this is 59, and SH makes up 76%.
- Training GPU memory is unchanged, because the reductions happen after the memory peak. — [arXiv HTML](https://arxiv.org/html/2406.17074)
- Full-method training time (not SH-isolated) on RTX A6000:

  | Dataset | 3DGS | Reduced 3DGS |
  |---|---|---|
  | Mip-NeRF360 | 31m16s | 25m27s |
  | T&T | 17m47s | 14m0s |
  | DB | 28m2s | 22m4s |

  — [arXiv HTML](https://arxiv.org/html/2406.17074)
- Storage: Mip-NeRF360 744 → 29 MB, T&T 412 → 14 MB, DB 630 → 18 MB.
- Quality, SSIM / PSNR / LPIPS: Mip-NeRF360 0.813 / 27.42 / 0.217 → 0.809 / 27.10 / 0.226; DB 0.899 / 29.47 / 0.247 → 0.902 / 29.63 / 0.249.
- K-means (256 entries) and half-float storage are post-hoc. Half floats alone cost about 0.07 dB on Mip-NeRF360 (27.17 → 27.10) and cut size 38 → 29 MB. — [arXiv HTML](https://arxiv.org/html/2406.17074)

**LightGaussian** (Fan et al.; arXiv 2311.17245)
- It distills a full-degree SH teacher into a student with lower-degree SH. — [search summary of arXiv 2311.17245](https://arxiv.org/pdf/2311.17245)
- (Prior knowledge, not re-verified this session) This is a post-hoc pipeline on a trained 3DGS: global-significance pruning, then SH distillation down to degree 2 with pseudo-view augmentation, then vector quantization. It reports about 15x compression and 200+ FPS. It runs as extra fine-tuning after training and does not speed up training.

**c3dgs** (Niedermayr et al., "Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis", CVPR 2024, arXiv 2401.02436)
- (Prior knowledge, not re-verified) Post-hoc sensitivity-aware vector clustering of SH and shape, followed by quantization-aware fine-tuning. It reports up to 31x compression. It does not speed up training.

**SG-Splatting** (arXiv 2501.00342)
- It replaces SH3 with 3 orthogonal spherical gaussians: 15 SG parameters plus low-degree SH, where the SH degree (0-2) is chosen by projected size. SGs are enabled after 2K of 30K iterations. — [arXiv HTML](https://arxiv.org/html/2501.00342v1)
- Training time on RTX 3090:

  | Dataset | 3DGS | SG-Splatting |
  |---|---|---|
  | Mip-NeRF360 | 25m29s | 24m11s |
  | T&T | 14m40s | 13m21s |
  | DB | 24m18s | 22m25s |

  — [arXiv HTML](https://arxiv.org/html/2501.00342v1)
- FPS rises 247 → 334, and storage shrinks by about 47%.
- PSNR: Mip-NeRF360 27.401 → 27.267, T&T 23.728 → 23.465, DB 29.508 → 29.570. — [arXiv HTML](https://arxiv.org/html/2501.00342v1)

### Inferences
- On GPU, training time is dominated by raster and densification. Colour-parameter reductions therefore buy only 0-10% training speed; Reduced 3DGS's roughly 20% comes mostly from fewer primitives.
- On the bandwidth-bound CPU Adam, the same reductions matter much more. Cutting from 59 to about 14 floats for the 89% of gaussians Reduced 3DGS demotes to DC would cut Adam traffic by roughly 3-4x. This requires a ragged or two-pool layout so that culled bands are not stored at all; zero-masking them does not help.
- EAGLES-style latent quantization and Lee-style hash-grid colour add decoder or MLP work per gaussian per step. On a 4-core CPU that likely cancels the bandwidth gain, so they are not recommended.

### Gaps
- I found no 2026 paper by "Gao" on SH reduction. Searches for Gao, SH and 2026 returned nothing matching, so the coordinator should supply an arXiv ID.
- No paper isolates the training-time effect of SH culling alone.
- LightGaussian and c3dgs numbers were not re-fetched.

## Q2. View-dependent colour alternatives: floats per splat, training cost, quality

### Takeaway
Several 2025-2026 replacements for SH3 match or beat it at 3-15 colour floats per gaussian.
- Spherical Beta (DBS) uses 15 floats and NASG uses about 11.
- A tiny-MLP latent model (Hahlbohm 2026) uses 3 + 8 floats.
- Directional-opacity lobes (ARF, PennyGS) add 5 floats or fewer.

The only measured training speedup from the colour model alone is 1.32-1.45x, for the latent MLP on RTX 4090 with MCMC. Spherical Voronoi gives the best quality but no memory saving (48 floats). Spec-Gaussian and Scaffold-GS are heavier.

### Cited Findings

**Spherical Voronoi** (Di Sario, Rebain, Verbin, Grangetto, Tagliasacchi; arXiv 2512.14180, CVPR 2026)
- 8 sites per gaussian, each with a 3-float direction (temperature taken from its norm) and a 3-float radiance. That is 48 floats, the same as SH3.
- Evaluation is a dot product plus a softmax over the sites. — [arXiv HTML](https://arxiv.org/html/2512.14180)
- Quality on a Beta Splatting backbone, PSNR / SSIM / LPIPS:

  | Dataset | SH | SV |
  |---|---|---|
  | Mip-NeRF360 | 28.09 / 0.833 / 0.231 | 28.71 / 0.837 / 0.228 |
  | DB | 29.80 / 0.912 / 0.303 | 30.63 / 0.917 / 0.298 |
  | T&T | 24.50 / 0.869 / 0.175 | 25.00 / 0.874 / 0.170 |

  SG and SB are no better than SH. — [arXiv HTML](https://arxiv.org/html/2512.14180)
- Training time (bonsai and garden, 8 sites): SV 32 min, SH 30 min, SG 31 min, SB 28 min. FPS: SV 173, SH 107. The GPU is not stated.
- The reflection variant costs 2.43x training time. — [arXiv HTML](https://arxiv.org/html/2512.14180)

**Compact Neural Appearance Models** (Hahlbohm, Condor, Franke, Eisemann, Magnor; arXiv 2609.05255, Sep 2026)
- Per gaussian: an 8-dimensional fp16 latent plus a 3-float base colour, 28 bytes against 192 for SH3.
- A shared MLP decodes it: input is the frequency-encoded latent (16) plus degree-0-3 SH of the direction (16), then 2 hidden layers of 16, then 3 outputs. It has no biases and 816 parameters in total. — [arXiv HTML](https://arxiv.org/html/2609.05255)
- Training time on RTX 4090 with MCMC densification:

  | Dataset | SH3 | Neural | Speedup |
  |---|---|---|---|
  | Mip-NeRF360 | 6m12s | 4m17s | about 1.45x |
  | T&T/DB | 4m10s | 2m56s | about 1.42x |
  | NeRF-Synthetic | 1m22s | 1m02s | about 1.32x |

  — [arXiv HTML](https://arxiv.org/html/2609.05255)
- Quality, SH3 → Neural: Mip-NeRF360 28.36 / 0.836 / 0.222 → 28.55 / 0.831 / 0.227; T&T/DB 27.96 / 0.893 / 0.228 → 28.13 / 0.893 / 0.228.
- Other models in the same table:
  - NASG (44 B): 28.50 / 0.836 / 0.223 on Mip-NeRF360.
  - NASGabor (48 B): 28.54.
  - Spherical Voronoi with 7 sites (208 B): 28.64.

  — [arXiv HTML](https://arxiv.org/html/2609.05255)
- The MLP forward and backward run in fp16 on Tensor Cores, with an fp32 Adam master copy and loss scale 128. Gradients for features and base colour stay in fp32. — [arXiv HTML](https://arxiv.org/html/2609.05255)

**Beyond Spherical Harmonics / NASGabor** (Miazga, Condor, Didyk; arXiv 2606.09794, Jun 2026)
- Proposes the Normalized Anisotropic Spherical Gabor function.
- The abstract claims it is "up to five times more memory-efficient" and "more efficient to evaluate". The abstract gives no numbers. — [arXiv abs](https://arxiv.org/abs/2606.09794)
- Its numbers in the 2609.05255 table are 48 B and 28.54 PSNR on Mip-NeRF360, above. — [arXiv HTML 2609.05255](https://arxiv.org/html/2609.05255)

**Deformable Beta Splatting (DBS)** (Liu et al.; arXiv 2501.18630, SIGGRAPH 2025)
- The Spherical Beta colour has a base colour plus M lobes, each with a direction, a colour and a shininess. The paper gives the dimension as 3 + 6M, which is 15 with M=2, or 31% of SH3.
- Total is about 27 floats per primitive. That count is derived, not stated. — [arXiv HTML](https://arxiv.org/html/2501.18630v2)
- Results are on an RTX 6000 Ada.

  | | 3DGS | 3DGS-MCMC | DBS 30K |
  |---|---|---|---|
  | Mip-NeRF360 training time | 21m | 31m | 16m |
  | Mip-NeRF360 memory | 797.06 MB | — | 356.04 MB |
  | Mip-NeRF360 PSNR / SSIM / LPIPS | — | — | 28.60 / 0.844 / 0.182 |
  | T&T PSNR | — | — | 24.79 |
  | DB PSNR | — | — | 30.10 |

  The training speedup combines the kernel, the colour model and MCMC-style densification, so it is not attributable to colour alone. — [arXiv HTML](https://arxiv.org/html/2501.18630v2)

**Universal Beta Splatting (UBS)** (Rong Liu et al.; arXiv 2510.03312, ICLR 2026)
- N-dimensional anisotropic Beta kernels that model view dependence (and time) without SH or auxiliary networks. Gaussian splatting is a special case.
- Claims "up to +8.27 dB" on static scenes, a headline maximum that is not an average. — [arXiv abs](https://arxiv.org/abs/2510.03312); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a5976b741e26265ba8a69966f019d854-Abstract-Conference.html)
- Floats per splat and training time were not retrieved.

**ARF, Augmented Radiance Field** (Yang, Wu, Zhou, Huang; arXiv 2602.19916, ICLR 2026)
- Adds gaussians with view-dependent opacity α·((cos(min(θ/T, π)) + 1)/2)^exp(β). Each adds 5 parameters, and they number 10% of the existing gaussians. — [arXiv HTML](https://arxiv.org/html/2602.19916v1)
- It is post-hoc, added to an already optimized scene with the original attributes frozen except opacity. This adds about 15 min on Mip-NeRF360 (742 s + 168 s) on an RTX 4090.
- Mip-NeRF360 results:
  - MCMC with SH2: 28.89 / 0.848 / 0.171 at 467 MB.
  - MCMC with SH3: 28.96 / 0.849 / 0.170 at 722 MB.
  - On the plain 3DGS base: 27.21 → 28.39 PSNR.

  This shows SH2 plus directional-opacity lobes is about equal to SH3. — [arXiv HTML](https://arxiv.org/html/2602.19916v1)

**Spec-Gaussian** (Yang et al.; arXiv 2402.15870, NeurIPS 2024) (prior knowledge, not re-verified)
- Anisotropic spherical-gaussian features decoded by an MLP, with an anchor-based option. Heavier than SH. ARF reports Spec-Gaussian at 882 MB on Mip-NeRF360. — [ARF arXiv HTML](https://arxiv.org/html/2602.19916v1)

**Scaffold-GS / Octree-GS** (arXiv 2312.00109 and 2403.17898) (prior knowledge, not re-verified)
- Anchors carry a feature vector (32-dimensional by default), k=10 offsets and scaling. Neural gaussians' colour, opacity and covariance are decoded by per-view MLPs every frame.
- This gives low storage but adds MLP evaluations per visible neural gaussian per step.

### Inferences
- **For a CPU trainer, the best fits are explicit low-float models evaluated per gaussian in projection.** Options:
  - DC plus 1-2 Spherical-Beta or spherical-gaussian lobes (7-15 floats);
  - flat RGB plus directional opacity (PennyGS, ARF-like);
  - SH2 (27 floats) plus a few ARF-style lobes.

  Each cuts Adam traffic by 2-4x, roughly 35 ms → 9-17 ms, at negligible O(N) eval cost.
- **The latent-MLP model is the most compact but costs compute (my estimate).** It needs about 816 MAC forward and about 2.5k MAC backward per visible gaussian, about 1.3 GFLOP/step at 400k. On 4 AVX2 cores that is roughly 20-40 ms, which likely erases its bandwidth gain unless the MLP is batched as a GEMM.
- **Spherical Voronoi gives the best quality but saves no memory.** It is only attractive where quality is the goal.
- PennyGS's flat RGB plus opacity cone fits this direction: ARF independently shows directional-opacity lobes recover about +1.2 dB over plain 3DGS.

### Gaps
- No per-colour-model training time on CPU exists anywhere.
- The GPU for SV's timings is unknown.
- UBS parameter count and training time were not retrieved.
- NASGabor full-text numbers were not fetched.

## Q3. Adaptive SH degree per gaussian and progressive SH schedules

### Takeaway
Per-gaussian adaptive degree works well: Reduced 3DGS demotes 89% of primitives to DC only at about −0.3 dB on Mip-NeRF360. SG-Splatting picks the degree from projected size.

The standard 3DGS schedule raises the SH degree every 1000 iterations up to 3. Even before degree 3 is active, Adam still updates the full 48-float SH tensor unless the implementation slices it.

### Cited Findings
- Reduced 3DGS: one-shot per-primitive band culling at 15K iterations, with 0/1/2/3-band shares of 89% / 0.1% / 2.7% / 8.2%. — [arXiv HTML](https://arxiv.org/html/2406.17074)
- SG-Splatting: small gaussians use SH degree 0-2 chosen by projected size, large ones use degree 2, and SG lobes are added after 2K iterations. — [arXiv HTML](https://arxiv.org/html/2501.00342v1)
- StructGS (arXiv 2503.06462) proposes adaptive SH (title only, not fetched). — [arXiv PDF](https://arxiv.org/pdf/2503.06462)
- A low-rank SH paper (TCSVT 2026) describes learned per-primitive degree masks as flexible but more complex. — [PDF](https://zhuzhiwei99.github.io/projects/2026_LowRankSH/LowRankSH_TCSVT.pdf)
- ARF with SH2 is about equal to SH3 when lobes are added. — [arXiv HTML](https://arxiv.org/html/2602.19916v1)

### Inferences
- A cheap CPU win needs no new paper: skip Adam and gradients for inactive SH bands during the warm-up schedule.
  - SH degree 0 for iterations 0-999 means 14 floats instead of 59.
  - Degree 1 means 23 floats, and degree 2 means 38.

  Averaged over a 30K run this saves little, because degree 3 is active from iteration 3K. It saves more in short runs.
- A larger win: apply Reduced-3DGS-style culling at the end of densification and move demoted gaussians to a 14-float pool. This would cut Adam traffic by about 3x for the remaining half of training.

### Gaps
- No paper measured the training-speed effect of per-gaussian SH degree on its own.

## Q4. Cheaper kernels and primitives (Beta, bounded support, 2DGS, convex, textured) and kernel evaluation cost

### Takeaway
The Beta kernel (1−r²)^β has compact support but requires a pow, that is log plus exp, per (splat, pixel). It is therefore not cheaper per evaluation than exp(−½d²). Any gain comes from exact bounded footprints and fewer primitives.

No paper found here benchmarks kernel arithmetic cost in isolation.

### Cited Findings
- DBS kernel: B(x; b) = (1 − x)^β(b) with β = 4e^b, x = r² ∈ [0, 1], and exactly zero outside.
- The paper gives no kernel-cost benchmark, only end-to-end FPS: 123 vs 3DGS 116 on Mip-NeRF360. — [arXiv HTML](https://arxiv.org/html/2501.18630v2)
- UBS generalizes Beta kernels to N dimensions and claims a real-time CUDA implementation. — [arXiv abs](https://arxiv.org/abs/2510.03312)
- (Prior knowledge, not re-verified)
  - 2DGS (arXiv 2403.17888) uses ray-splat intersection with planar gaussians. It is more expensive per pixel than 3DGS.
  - 3D Convex Splatting (arXiv 2411.14974) uses a smooth convex hull of 6+ points, which is also costlier per pixel.
  - Textured Gaussians (arXiv 2411.18625) add per-gaussian texture maps, which adds memory and per-pixel texture lookups.

  None of these targets cheaper training.

### Inferences
- **For the CPU raster** (polynomial exp at about 100 ms/step):
  - A plain Beta kernel with fixed integer β (for example (1−r²)^4 via 2 squarings) would be cheaper than exp and exactly bounded. This lets tile and AABB culling use the true support instead of 3σ.
  - DBS's learnable real β needs a pow, so it gains nothing per evaluation.
  - A fixed-β polynomial kernel is a speculative design choice, not a published, benchmarked result.
- Bounded support plus exact ellipse-tile intersection likely reduces the count of (splat, pixel) pairs more than kernel arithmetic does.

### Gaps
- No primary-source measurement of kernel evaluation cost.
- Convex, 2DGS and Textured Gaussians numbers were not re-fetched this session.

## Q5. Mixed precision (fp16/bf16) in 3DGS training

### Takeaway
The evidence is thin.
- fp16 parameter storage costs about 0.07 dB (post-hoc).
- fp16 rasterization is reported as near-lossless in one 2026 paper, abstract only.
- One 2026 paper trains its colour MLP in fp16 with fp32 Adam masters.

I found no study of fully fp16 or bf16 3DGS parameter training.

### Cited Findings
- Reduced 3DGS: half-float storage gives 27.17 → 27.10 PSNR on Mip-NeRF360 and 38 → 29 MB. This is post-hoc. — [arXiv HTML](https://arxiv.org/html/2406.17074)
- "Accelerating 3D Gaussian Splatting using Tensor Cores" (Li et al.; arXiv 2605.17855, May 2026) says rendering "can be executed in FP16 with negligible quality degradation". It reports 1.65x end-to-end rendering, but the abstract gives no PSNR numbers. — [arXiv abs](https://arxiv.org/abs/2605.17855)
- HiGS (arXiv 2606.00352) does fp16 rasterization. A search summary attributes to it a large gap against an fp32 reference, which could not be verified and should be treated with caution. — [arXiv PDF](https://arxiv.org/pdf/2606.00352)
- Hahlbohm 2026 runs the MLP forward and backward in fp16 with loss scale 128 and fp32 Adam masters. Per-gaussian feature gradients stay fp32, and the features are stored in fp16. — [arXiv HTML](https://arxiv.org/html/2609.05255)

### Inferences
- **The highest-value CPU option is storing Adam moments m and v in bf16 or fp16 while keeping fp32 masters.** This would cut Adam bandwidth by about 30%.
- **Storing SH coefficients (not position) in fp16 should cost under 0.1 dB.** The basis is the 0.07 dB storage result, an analogy rather than a direct measurement.
- AVX2 has F16C for fast conversion, but no native bf16 arithmetic.

### Gaps
- No primary-source measurement of fp16 or bf16 optimizer state in 3DGS training.

## Overall CPU-transfer judgement (inference, ranked)
1. **Shrink colour floats during training.** Use flat RGB plus directional opacity (PennyGS), DC plus 1-2 spherical-beta or SG lobes, or SH2 plus ARF lobes.
   - Expected Adam cost: about 35 ms → 8-17 ms/step.
   - Expected quality: within ±0.3 dB, or better, of SH3, per DBS, ARF and PennyGS.
2. **Cull SH bands per gaussian at the end of densification, Reduced 3DGS style,** into a ragged or two-pool layout. Expect about 3x less Adam traffic after that point at about −0.3 dB.
3. **Skip inactive SH bands in Adam during the degree warm-up.** This is trivial to do.
4. **Store fp16 or bf16 Adam moments and SH.** This plausibly gives 25-50% less bandwidth, but no 3DGS training study confirms it.
5. **Avoid MLP or hash-grid colour (Lee, EAGLES decoders, Scaffold, Spec-Gaussian, latent MLP) and pow-based Beta kernels.** On 4 cores the compute adds more than the bandwidth it saves.
