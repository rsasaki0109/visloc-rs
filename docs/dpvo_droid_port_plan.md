# DPVO / DROID-SLAM / DPV-SLAM port plan

Direct-port ("ベタ移植") design for bringing the DROID-SLAM family's learned
visual front end into `visloc-rs`. Motivation: the current full-stack ATE on
EuRoC MH_01 is ~2.9 m with fragmented submaps against an ORB-SLAM3-SI target of
0.036 m at ~100% tracking; the accuracy ceiling is visual rotation/correspondence
quality, not the BA/pose-graph/IMU machinery already in this repo (see
`docs/visual_slam_literature_survey.md`, `docs/motion_based_vi_alignment.md`).
IMU fusion stays **our** machinery (DROID/DPVO/DPV-SLAM are visual-only); the
port supplies the correspondence/visual-odometry engine underneath it.

Sources are cited inline; anything not independently confirmed from a primary
source (paper text, official repo file) is marked **unverified**.

## 1. Architecture inventory

### 1.1 DROID-SLAM (Teed & Deng, NeurIPS 2021)

- Paper: [arXiv:2108.10869](https://arxiv.org/abs/2108.10869). Code:
  [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM).
- Component breakdown, sourced from
  [`droid_slam/droid_net.py`](https://github.com/princeton-vl/DROID-SLAM/blob/main/droid_slam/droid_net.py):

| Component | Shape / config | Source |
| --- | --- | --- |
| Feature net `fnet` | `BasicEncoder(output_dim=128, norm_fn='instance')`, applied once per frame | `droid_net.py` |
| Context net `cnet` | `BasicEncoder(output_dim=256, norm_fn='none')`, split into `net` (128 ch, `tanh`) and `inp` (128 ch, `relu`) | `droid_net.py` |
| Correlation volume | 4 pyramid levels, radius 3 → `cor_planes = 4·(2·3+1)² = 196` channels | `droid_net.py` |
| Correlation encoder | `Conv2d(196,128,k1)→ReLU→Conv2d(128,128,k3,pad1)→ReLU` | `droid_net.py` |
| Flow encoder | `Conv2d(4,128,k7,pad3)→ReLU→Conv2d(128,64,k3,pad1)→ReLU` | `droid_net.py` |
| ConvGRU | Hidden state 128 ch; input is corr(128)+flow(64)+context `inp`(128) concatenated (~256–320 ch pre-projection) | `droid_net.py` |
| Output heads | `weight` head → 2 ch, sigmoid (per-pixel confidence on `[u,v]`); `delta` head → 2 ch (flow-field revision); `eta` → 1 ch + `upmask` → 8×8×9 (learned convex upsampling of the 1/8-resolution field) | `droid_net.py` |

- Everything above runs at **1/8 input resolution** (standard RAFT-family
  downsampling that DROID inherits — Teed & Deng's own prior RAFT work; stated
  in the DROID-SLAM paper's method section).
- **Dense Bundle Adjustment (DBA)**: Gauss-Newton over SE3 camera poses plus
  **per-pixel inverse depth**, with the depth block's diagonal Hessian
  Schur-eliminated (classical BA structure, just at pixel density instead of
  landmark density). Implemented as a custom CUDA extension, not
  autograd/torch ops: `setup.py` builds `droid_backends` from
  `src/droid.cpp`, `src/droid_kernels.cu`, `src/correlation_kernels.cu`,
  `src/altcorr_kernel.cu` (verified from
  [`setup.py`](https://github.com/princeton-vl/DROID-SLAM/blob/main/setup.py)).
- **Frame graph**: a "Frame Graph Representation" of temporal + proximity
  edges drives which pairs get correlation/update/BA each step; DROID's BA is
  invoked repeatedly as new edges are added (this is the "recurrent iterative
  update" loop, driven from Python, not baked into the network graph — see §3).
- **Loop closure**: DROID-SLAM has **no dedicated loop-closure backend** — its
  only long-range consistency mechanism is the frame graph's own proximity
  edges reactivating when the trajectory revisits a place. The DPV-SLAM paper
  states this explicitly as its own motivation (§1.2 below).
- Network/CPU boundary: fnet/cnet/correlation-encoder/flow-encoder/ConvGRU/
  output-heads are the neural part (one forward pass per update iteration,
  called in a host-side loop); the DBA linear solve and SE3 retraction are
  pure numerical linear algebra with no learned parameters.

### 1.2 DPVO (Teed, Lipson, Deng, NeurIPS 2023)

- Paper: [arXiv:2208.04726](https://arxiv.org/abs/2208.04726). Code:
  [princeton-vl/DPVO](https://github.com/princeton-vl/DPVO).
- Component breakdown, sourced from
  [`dpvo/net.py`](https://github.com/princeton-vl/DPVO/blob/main/dpvo/net.py),
  [`dpvo/dpvo.py`](https://github.com/princeton-vl/DPVO/blob/main/dpvo/dpvo.py),
  [`dpvo/config.py`](https://github.com/princeton-vl/DPVO/blob/main/dpvo/config.py):

| Component | Shape / config | Source |
| --- | --- | --- |
| Patchifier `fnet` | `BasicEncoder4(output_dim=128, norm_fn='instance')` | `net.py` |
| Patchifier `inet` | `BasicEncoder4(output_dim=DIM=384, norm_fn='none')` | `net.py` |
| Patch size | 3×3 pixels | `net.py`, `config.py` |
| Patches per frame | **80** (`PATCHES_PER_FRAME = 80`, default `RANDOM` centroid selection strategy) | `config.py` — **corrects the ~96 figure in the task brief; 80 is the shipped default**, not a paper-stated 96 |
| Patch/keyframe buffer | `BUFFER_SIZE=4096`, `REMOVAL_WINDOW=20`, `OPTIMIZATION_WINDOW=12`, `PATCH_LIFETIME=12` | `config.py` |
| Keyframing | `KEYFRAME_INDEX=4`, `KEYFRAME_THRESH=12.5` (motion-magnitude threshold; a below-threshold frame is folded into a relative-pose delta and its factors dropped rather than kept) | `config.py`, `dpvo.py` |
| GRU update | Hidden dim 384 (=`DIM`); inputs = previous net state + `inet` context features + correlation features (2 pyramid levels × `(2·3+1)²=49` taps, linear-projected); block structure `LayerNorm→GatedResidual→LayerNorm→GatedResidual` with a neighbor-aggregation MLP pair (`c1`,`c2`) that pools over the patch graph's keyframe/pair dimensions (a lightweight relational/GNN-style layer inside the update, not present in DROID) | `net.py` |
| Output heads | `delta`: `ReLU→Linear(DIM,2)→GradientClip` (patch target-pixel correction); `weight`: `ReLU→Linear(DIM,2)→GradientClip→Sigmoid` (confidence); third head unused | `net.py` |
| Update iterations | **12** iterations during the ~8-frame initialization burst (`for itr in range(12): self.update()` when `n==8`); **1** iteration per incoming frame in steady state | `dpvo.py` |
| Patch graph edges | Forward/backward temporal edges inside the `PATCH_LIFETIME` window (`__edges_forw`/`__edges_back`, via `flatmeshgrid`); optional loop edges via `pg.edges_loop()` | `dpvo.py` |
| Bundle adjustment | `fastba.BA(poses, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, ...)`, **2** Gauss-Newton iterations per call, optimizes SE3 poses + patch inverse depths | `dpvo.py`, `ba.py` |

- **DBA/Schur complement**: confirmed from
  [`dpvo/ba.py`](https://github.com/princeton-vl/DPVO/blob/main/dpvo/ba.py) —
  a **pure-PyTorch reference implementation** exists (`block_matmul`,
  `block_solve`, an explicit `### solve w/ schur complement ###` comment
  computing `S = B − E·Q·Eᵀ` then back-substituting for depth), but production
  speed comes from a **native CUDA extension** `cuda_ba` (`ba.cpp`,
  `ba_cuda.cu`, `block_e.cu`, using Eigen 3.4.0) built in
  [`setup.py`](https://github.com/princeton-vl/DPVO/blob/main/setup.py). The
  math itself is the same Gauss-Newton-with-Schur-elimination structure
  `visloc-rs`'s own `pipelines/slam/src/bundle.rs` already implements for
  landmarks — DPVO's BA does not need porting as an algorithm, only its patch
  inverse-depth parameterization needs adding as a new residual/variable kind.
- Correlation: `cuda_corr` (`correlation.cpp`/`correlation_kernel.cu`) — the
  C++ shim only declares `corr_cuda_forward`/`corr_cuda_backward`, so the CUDA
  kernel body wasn't directly inspectable, but by construction (RAFT/DROID
  lineage, same author) it is a normalized dot-product cost volume between
  patch-anchored feature vectors and a target frame's feature map, average-
  pooled into a small pyramid, then bilinearly sampled at the current
  target-pixel estimate. This is classical math (no learned weights) —
  expressible as `matmul` + `avg_pool` + bilinear-sample (`grid_sample`
  equivalent), or reimplemented directly in Rust without going through ONNX at
  all (see §4).
- SE3/Sim3 ops: a `lietorch_backends` extension with **both** a GPU kernel
  (`lietorch_gpu.cu`) and a CPU implementation (`lietorch_cpu.cpp`) — confirms
  the Lie-algebra math itself is not inherently GPU-only; `visloc-core::geometry`
  already has an independent SE3/SO3 implementation, so this component is not
  ported either.
- Reported vs DROID-SLAM: DPVO's abstract states it "outperforms all prior
  work, including \[DROID\]... using a third of the memory while running 3x
  faster on average" (paper abstract, arXiv:2208.04726).

### 1.3 DPV-SLAM (Lipson, Teed, Deng, ECCV 2024)

- Paper: [arXiv:2408.01654](https://arxiv.org/abs/2408.01654) /
  [ECCV proceedings](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00272.pdf).
  Code: merged into [princeton-vl/DPVO](https://github.com/princeton-vl/DPVO)
  (the `LOOP_CLOSURE` / `CLASSIC_LOOP_CLOSURE` config flags in `config.py`
  above are literally DPV-SLAM's two backends, gated off by default).
- **Two loop-closure backends**, both confirmed structurally by
  `dpvo/config.py`'s field names and cross-checked against an automated
  extraction of the arXiv HTML (treat exact prose below as **paraphrase of an
  automated summary, not a verbatim table read** — re-check before quoting
  externally):
  1. **Mid-term / "proximity" backend** (`LOOP_CLOSURE=True`,
     `BACKEND_THRESH=64.0`, `MAX_EDGE_AGE=1000`, `GLOBAL_OPT_FREQ=15`):
     detected by **camera-pose proximity**, not appearance. DPV-SLAM keeps
     patch features for every past frame and creates uni-directional edges
     from old patches to the current frame when pose proximity crosses
     `BACKEND_THRESH`; because DPVO's correlation only needs the *destination*
     frame's dense features, no extra encoder pass is required for the
     revisit. These proximity edges are added into the **same** patch graph
     as ordinary temporal odometry edges and refined by the **same**
     GRU-update + BA machinery — it is not a separate module, just edges with
     a longer baseline. `GLOBAL_OPT_FREQ=15` triggers a periodic full
     (not windowed) BA pass.
  2. **Long-term / "classical" backend** (`CLASSIC_LOOP_CLOSURE=True`,
     `LOOP_RETR_THRESH=0.04`): classical **dBoW2** bag-of-words image
     retrieval over ORB features finds candidate revisits; candidates are
     verified with off-the-shelf keypoint matching, a structure-only BA to
     triangulate 3D points, 3D-3D matching between the two point clouds, and
     RANSAC+Umeyama rigid alignment. The resulting pose-graph edge is
     optimized by a **classical pose-graph optimizer running on the CPU in
     parallel to the main (GPU) odometry process**, then folded back in.
  3. Both edge types are mixed into one factor/scene graph and jointly
     bundle-adjusted with a CUDA block-sparse BA.
- Reported numbers (automated-extraction caveat above): EuRoC ATE ≈0.024 m
  (DPV-SLAM) vs ≈0.022 m (DROID-SLAM) at ~2.5× the speed (~50 vs ~20 fps);
  TartanAir DPV-SLAM ≈0.16 m vs DROID ≈0.24 m at ~3.4× the speed; KITTI:
  DPV-SLAM succeeds where DROID-SLAM is reported to fail/have no number; GPU
  memory 5–7 GB (DPV-SLAM) vs ~20 GB (DROID-SLAM).
- **This maps almost one-to-one onto `visloc-rs`'s existing
  `pipelines/slam/src/sparse_factor_graph.rs`** — see §4.

## 2. License verdicts

| Repo | Code license | Verified from | Weights |
| --- | --- | --- | --- |
| [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | **BSD 3-Clause**, © 2021 Princeton Vision & Learning Lab | Raw fetch of [`LICENSE`](https://github.com/princeton-vl/DROID-SLAM/blob/main/LICENSE) | `droid.pth`, Google Drive link in README, **no accompanying license/terms file** |
| [princeton-vl/DPVO](https://github.com/princeton-vl/DPVO) | **MIT**, © 2022 Princeton Vision & Learning Lab | Raw fetch of [`LICENSE`](https://github.com/princeton-vl/DPVO/blob/main/LICENSE) | `models.zip`, Google Drive link in README, **no accompanying license/terms file** |
| DPV-SLAM (same repo as DPVO) | Same MIT license (no separate LICENSE file; code lives in the DPVO tree) | Same as above | Same as above |

**Verdict: code is clean, weights are the open question.**

- Both BSD-3-Clause and MIT are permissive, non-copyleft, and fully compatible
  with `visloc-rs`'s own `MIT OR Apache-2.0` (checked in this repo's
  `Cargo.toml`). Neither license restricts commercial use or reimplementation.
- **A clean-room Rust reimplementation from the papers** (writing new code
  that implements the same architecture/math, without copying their Python or
  CUDA source) does not even trigger the attribution clause — that clause
  only binds redistributed copies of *their* source/binary.
- **Because this task explicitly asks for a direct port ("ベタ移植")** — i.e.
  translating their actual code, not just their published method — the BSD-3
  / MIT notice-and-disclaimer clauses do bind the translated files. This is
  trivial to satisfy: keep a `NOTICE`/attribution block citing "Princeton
  Vision & Learning Lab" and reproducing the two license texts for any file
  whose logic was translated line-for-line from DROID-SLAM/DPVO source. BSD-3
  additionally forbids using Princeton's name to promote derived products
  without permission — a passive citation is fine, marketing copy invoking
  their name is not.
- **Pretrained weights are the real gray area.** Neither repo ships a LICENSE
  file alongside its Google-Drive-hosted checkpoint, and OSS/ML practice
  generally treats trained weights as a separately protectable artifact from
  the training code unless a license explicitly says otherwise (contrast with
  licenses that explicitly cover weights, e.g. Meta's Llama license). Princeton's
  do not say anything about the weights either way.
  - **Recommendation, mirroring this repo's existing SuperPoint pattern**:
    do not check the converted `.onnx` weights into the `visloc-rs` repo or
    ship them in a distributed binary/crate. Instead ship a Python export
    script (parallel to `scripts/export_superpoint_onnx.py`) that a user runs
    themselves against the officially-downloaded checkpoint they fetch
    directly from princeton-vl's Google Drive link. This sidesteps the
    redistribution question entirely — exactly how `models/superpoint_1500.onnx`
    already works in this repo (user-run export, not a bundled artifact).
    Internal/local research use (this repo's own benchmark harness, not
    distributed) is low-risk either way.
  - If a shipped, redistributable ONNX weight file is ever required, either
    get explicit written permission from princeton-vl, or retrain equivalent
    encoders on permissively-licensed data (TartanAir's own license needs a
    separate check before that path is viable) as a `visloc-rs`-owned
    checkpoint.

## 3. ONNX exportability

The key structural fact, confirmed from both `droid_net.py` and `dpvo/net.py`:
**the multi-iteration refinement loop is host-side Python control flow that
calls the network module repeatedly — it is not baked into either network's
computation graph.** DROID's frame-graph BA loop and DPVO's `for itr in
range(12): self.update()` / one-call-per-frame steady state both live outside
the `nn.Module.forward`. This directly answers the loop-free-export question:
**one call = one static graph**, so ONNX export needs no `Loop`/`Scan` op —
the outer iteration becomes ordinary Rust control flow in the port, exactly
like the existing `ort` integration already does per-frame for SuperPoint.

| Piece | Neural? | ONNX-exportable? | Plan |
| --- | --- | --- | --- |
| `fnet`/`cnet` (DROID) or `fnet`/`inet` (DPVO) encoders | Yes | Yes — plain CNNs, same shape of problem `superpoint_onnx.rs` already solves | Export via a new `torch→onnx` script per §5 M1 |
| Correlation volume + lookup | No (classical dot-product + pooling + bilinear sample) | N/A — do **not** export; implement natively in Rust | Mirrors the existing pattern where SuperPoint's NMS/keypoint-selection/bilinear descriptor sampling are already done in Rust, not ONNX (`docs/superpoint_onnx_cuda_benchmark.md`) |
| GRU update cell (single call) | Yes | Yes, if patch/edge counts are padded to a fixed max per graph call (same trick the SuperPoint exporter used to fix a constant top-k keypoint count for a static graph) | Export per §5 M1/M2 |
| DBA / BA (Gauss-Newton + Schur) | No | N/A — pure math | Reuse/extend `bundle.rs`, do not port `ba.py`/`cuda_ba` |
| SE3/Sim3 retraction | No | N/A | Reuse `visloc_core::geometry` |

- `grid_sample`/`GridSample` is available from **ONNX opset 16** onward (PyTorch
  export historically failed at opset 9/12, added at 16); ONNX Runtime's CPU
  execution provider supports `GridSample` (confirmed via ORT issue tracker
  discussion of opset-16 `GridSample` on CPU EP, with 4D-input caveats around
  5D volumetric inputs that don't apply here). This matters only if the
  correlation math is exported rather than reimplemented natively — since the
  recommendation above is to reimplement it natively in Rust, this becomes a
  non-blocking fact rather than a hard dependency; note it in case a future
  milestone decides to export the lookup for parity-testing convenience.
- **No known ONNX/TensorRT export precedent for DPVO or DROID-SLAM's update
  operator was found** in official repos, issues, or community forks during
  this research pass — this is unexplored territory, not a well-trodden path.
  Budget schedule accordingly (§5, §6).
- CPU feasibility estimate (reasoning chain, not a benchmark — no DPVO/DROID
  ONNX artifact exists yet to actually time):
  - This machine's own measured baseline: SuperPoint ONNX CPU EP is
    165 ms/frame at 752×480 (`docs/superpoint_onnx_cuda_benchmark.md`, this
    host), 241 ms/frame on the Windows CUDA-verification host.
  - DPVO's `fnet` (128 ch) + `inet` (384 ch) are two `BasicEncoder4`-class CNN
    passes over the same input resolution — comparable convnet depth to
    SuperPoint's single backbone, so plausibly **1.5–2×** SuperPoint's cost
    for the encoder stage alone: **~250–450 ms/frame**.
  - Patch correlation (80 patches × 3×3, 2 pyramid levels, 49 taps) is a few
    thousand small dot products — negligible (<5 ms) as tight Rust/BLAS code,
    no GPU dependency.
  - One GRU update iteration over ~80 patch tokens (384-dim) is small relative
    to the encoder stage — plausibly tens of ms via `ort` CPU EP; the 12×
    iteration cost only hits once, during the ~8-frame startup burst.
  - BA (2 Gauss-Newton iterations, ≤80 patches × 12-frame window) is a much
    smaller sparse system than this repo's own per-landmark DBA solves in
    `bundle.rs` — sub-to-few-ms via `nalgebra`.
  - **Net estimate: ~300–500 ms/frame CPU steady state (≈2–3 fps)** — in the
    same ballpark as, or somewhat worse than, the existing SuperPoint CPU
    baseline; not real-time on this CPU-only machine, but usable for the
    offline/batch EuRoC ATE evaluation this repo already runs, and for an A/B
    research harness. Treat as a hypothesis to be measured once M1/M2 land,
    not a committed number.

## 4. Port surface mapping — the hybrid VI design

This is the central finding: **`visloc-rs` already contains a DROID-style
sparse factor graph scaffold, built in anticipation of exactly this port.**
`pipelines/slam/src/sparse_factor_graph.rs`'s own module doc says: *"A future
learned frontend can update the target correction, anisotropic information,
and damping on the same factor records without changing the metric BA/PnP
acceptance layer."* Its `SparseFactorMeasurement` struct —
`target_correction_px: [f32; 2]`, `information: [f32; 2]`, `damping: f32` — is
**structurally identical** to DPVO's per-edge GRU output (`delta`: 2-ch pixel
correction, `weight`: 2-ch confidence, plus the BA's `lmbda` damping scalar).
`SparseFactorKind::{Temporal, Proximity, Stereo}` with
`SparseFactorState::{Active, Inactive}` lifecycle, confidence-gated
reactivation, and an active-factor budget is already DPVO/DPV-SLAM's
patch-graph edge policy (temporal window + proximity edges + budget
enforcement + inactive-but-recoverable retention) minus the learned
measurement source.

| DROID/DPVO/DPV-SLAM piece | `visloc-rs` module it maps onto | What's missing |
| --- | --- | --- |
| Patch graph sliding window (`OPTIMIZATION_WINDOW=12`, `PATCH_LIFETIME=12`) | `OnlineSlamLocalBaConfig.window_size` (`online_slam_vi_ba.rs`) — already a trailing-keyframe-window BA trigger | Patch-level (not just keyframe-level) windowing; a thin wrapper, not a new mechanism |
| Patch graph temporal/proximity edges, active/inactive lifecycle, budget | `sparse_factor_graph.rs`'s `SparseFactorKind`/`SparseFactorState`/`enforce_active_budget` — **already implemented** | Only the *measurement source*: today `SparseFactorMeasurement::geometric()` sets confidence from shared-landmark support; needs a path where a learned update call populates `target_correction_px`/`information` via `update_measurement()` (already a public API) |
| DBA (pose + inverse-depth Gauss-Newton, Schur complement) | `pipelines/slam/src/bundle.rs`'s `BundleAdjustment` — already Schur-eliminates landmark blocks in exactly this structure | A new variable/residual kind: patch inverse-depth (one scalar per patch, not `Point3<f64>` per landmark) plus a residual that consumes `target_correction_px`+`information` as a weighted 2D correction instead of a raw pixel observation |
| DPV-SLAM proximity (mid-term) loop edges | `sparse_factor_graph.rs::SparseFactorKind::Proximity` + `broader_recovery_candidates` — **already implemented**, currently scored by shared-landmark geometry only | Feed proximity edges from the learned correlation/GRU update instead of (or in addition to) shared-landmark support |
| DPV-SLAM classical (long-term) loop closure (dBoW2 retrieval + PnP + RANSAC+Umeyama) | `online_slam.rs`'s appearance-loop-candidate pipeline + `map_atlas.rs`'s cross-submap PnP/RANSAC/scale-consensus/Sim3 welding — **already more rigorous** (metric-scale gates, covisibility disagreement bounds, MAD-based scale consensus) than DPV-SLAM's own verification | Nothing — this stays visloc-rs's own machinery; DPV-SLAM's classical backend does not need porting at all |
| IMU coupling | **Explicitly not in DROID/DPVO/DPV-SLAM** (visual-only systems) | `imu_preintegration.rs`'s `ImuPreintegrationFactor`, `vi_motion_initializer.rs`'s gyro-bias/gravity recovery chain, `NavigationStatePrior` marginalization — all **stay as-is** |

**Hybrid design in one sentence:** the DPVO-style update operator becomes a
new *measurement source* feeding `visloc-rs`'s existing sparse factor graph
and `BundleAdjustment`; it supplies per-patch correspondences + confidences,
our solver consumes them as weighted reprojection factors (a new patch
inverse-depth variable/residual pair) sitting in the **same** `BundleAdjustment`
struct that already carries `imu_factors`, `pairwise_pose_factors`, and
`navigation_state_prior` — so a patch-BA solve and an IMU-preintegration solve
are one joint Gauss-Newton problem, not two separately-reconciled ones. Metric
scale continues to come from stereo baseline / IMU (already decided), not from
DPVO's monocular-only formulation.

## 5. Staged milestones

Each milestone is independently testable in Rust with no new heavy
dependencies beyond `ort` (already used, `load-dynamic`, CPU EP only on this
machine), `nalgebra`, and `image`.

| # | Scope | Acceptance criteria | Rough effort |
| --- | --- | --- | --- |
| M1 | Python `torch→onnx` export scripts for DPVO's `fnet`/`inet` encoders and the GRU update cell (single iteration, fixed max patch/edge count via padding+masking — same trick used for SuperPoint's fixed top-k). | Cosine-sim/L2 parity vs PyTorch reference within a defined tolerance (e.g. 1e-4) on ≥20 saved `(image, patch-coords)` fixtures spanning EuRoC MH_01/03/05. | Small–medium (1–2 scripts, direct precedent in `scripts/export_superpoint_onnx.py`) |
| M2 | Rust patch extraction (native — centroid selection, 3×3 coords, no ONNX) + native correlation lookup against ONNX-derived feature maps + single GRU-iteration inference via `ort`. | Parity vs the Python fixture from M1 within the same tolerance; new module mirrors `crates/vision/src/features/superpoint_onnx.rs`'s structure (ONNX boundary ends at raw tensors, math done in Rust). | Medium–large (first native inference + math module) |
| M3 | Rust DBA extension: patch inverse-depth variable, weighted/damped residual consuming `target_correction_px`/`information`, Schur complement — extends `bundle.rs` rather than porting `ba.py`/`cuda_ba`. | Unit tests vs synthetic closed-form scenes; Schur-eliminated solution matches a dense (non-Schur) reference solve to numerical tolerance, following the existing `bundle.rs` test pattern. | Medium (extends a well-tested existing solver) |
| M4 | Sliding-window visual-only VO loop on EuRoC wiring M1–M3 into a new entry point (patch graph via `sparse_factor_graph.rs`, window via `online_slam_vi_ba.rs`-style trigger). | Visual-only ATE reported against (a) DPVO's own published EuRoC numbers and (b) this repo's existing SuperPoint+LightGlue baseline; report both, do not require beating either yet. | Large (first end-to-end integration) |
| M5 | IMU-coupled DBA: attach existing `ImuPreintegrationFactor` + `vi_motion_initializer.rs` gyro-bias/gravity chain to the M3 patch-BA as an additional residual family in the same `BundleAdjustment`. | `cost_breakdown`-style decomposition shows both visual (patch) and IMU residual blocks converging jointly; ATE on EuRoC MH_01/03/05 reported against the current tight-VI baseline (`docs/motion_based_vi_alignment.md`). | Medium (mostly plumbing once M3 produces BA-compatible residuals) |
| M6 | DPV-SLAM-style proximity loop closure: feed `sparse_factor_graph.rs`'s `Proximity` kind from the learned update instead of/alongside shared-landmark geometry; classical long-term loop closure reuses `online_slam.rs`/`map_atlas.rs` unchanged. | Loop-closure recall/precision on a EuRoC sequence with a genuine revisit (e.g. MH_05), measured against the existing appearance-loop-candidate pipeline's current recall. | Small–medium (mostly wiring — the single biggest existing-asset leverage point in this plan) |

## 6. Risk register

| Risk | Detail | Mitigation |
| --- | --- | --- |
| CPU latency | Estimated ~300–500 ms/frame steady-state (§3) — not real-time on this CPU-only machine (no cuDNN/CUDA available here). | Scope the port to offline/batch EuRoC evaluation first (M4–M6), same mode this repo already benchmarks in; real-time is a later, separate goal requiring a GPU host. |
| GRU export pitfalls | Dynamic per-call patch/edge counts don't map to a static ONNX graph. | Pad/mask to a fixed max count (same fix already used for SuperPoint's top-k keypoint export); verify masking doesn't leak into `LayerNorm`/aggregation statistics during M1 parity tests. |
| Training-free reuse limits | Weights were trained at specific resolutions/patch counts (TartanAir-based training data, per DPVO's training config); reuse on EuRoC at different resolution/patch settings may need light finetuning or explicit resolution-matching preprocessing. | Treat M4's ATE as a hypothesis, not a guarantee; if it's poor, fall back to the offline-oracle mitigation below before assuming the port itself is broken. |
| Weights license gray zone | No explicit license on either repo's pretrained checkpoint (§2). | Export-script pattern (not bundling weights), mirroring the existing SuperPoint precedent; escalate to explicit permission or retraining only if redistribution is ever required. |
| Numerical parity tolerance | Rust/`nalgebra` float ops will not bit-match PyTorch/CUDA; ONNX Runtime CPU EP vs CUDA EP already shows small non-deterministic differences for SuperPoint (`docs/superpoint_onnx_cuda_benchmark.md`: 1021.8 vs 1020.7 average kept keypoints). | Define parity tolerance up front per milestone (M1/M2: cosine-sim/L2 on tensors; M3: solved-state agreement, not bit-identical residuals) rather than discovering an ad hoc threshold under deadline pressure. |
| No prior ONNX/TensorRT export exists for DPVO/DROID | Confirmed absence of precedent (§3) — this plan is exploring new territory for the update operator's export, unlike SuperPoint which had a working reference. | Budget M1/M2 as genuinely uncertain-effort R&D, not routine porting; keep the offline oracle (below) as the fallback so the rest of the plan isn't blocked on export success. |
| Fallback: offline correspondence oracle | If the Rust port (M2/M3) stalls or produces poor parity, DPVO's own Python/CUDA reference can be run offline on EuRoC sequences to produce correspondence dumps consumed exactly like the existing pre-exported SuperPoint/LightGlue feature dumps, for an A/B comparison against the Rust port before fully trusting M4's ATE. | Keep this available throughout M1–M4; do not treat it as a permanent solution (it reintroduces the Python/PyTorch dependency this whole port aims to remove, per the existing SuperPoint ONNX motivation in `docs/superpoint_onnx_cuda_benchmark.md`). |

## Recommended first milestone

**M1** (Python ONNX export of DPVO's `fnet`/`inet` encoders + one GRU-update
iteration, with parity fixtures). It is the smallest self-contained step, has
a direct precedent already in this repo (`scripts/export_superpoint_onnx.py`),
and — because no ONNX export of this operator has ever been published for
DPVO/DROID — it is also the step most likely to surface an unexpected export
blocker (dynamic shapes, unsupported ops in the neighbor-aggregation MLP)
early, before any Rust code is written against an assumption that turns out
to be wrong.

## M1 results (2026-07-16)

Scripts: `scripts/export_dpvo_onnx.py`, `scripts/check_dpvo_onnx_parity.py`
(both self-contained; no new repo dependency — a throwaway venv at
`E:/tools/venvs/dpvo_export` holds `torch==2.13.0+cpu`, `onnx==1.22.0`,
`onnxruntime==1.27.0`, `onnxscript==0.7.1`, `numpy==2.5.1`). DPVO cloned to
`E:/tools/DPVO` (MIT, shallow clone, local use only, not vendored into this
repo). Both scripts read their own module docstrings for the full design
rationale; this section summarizes the outcome.

### Weight download: succeeded, and it's Dropbox, not Google Drive

The plan's §2 assumption (Google Drive checkpoint, `gdown` needed) was
**wrong** — `download_models_and_data.sh` actually fetches from Dropbox
(`https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip`), which `curl -L`
pulls directly with no auth/interactive step. `models.zip` → `dpvo.pth`
(13.5 MB, 98 tensors) downloaded and extracted to
`E:/tools/DPVO/models_extracted/dpvo.pth` without incident. All exports and
parity numbers below use these **real pretrained weights**, not random
init (a random-weight fallback path exists and was also verified to work,
see "Parity results" below).

### Exact `Update` operator signature, found in `dpvo/net.py`

```python
class Update(nn.Module):
    def forward(self, net, inp, corr, flow, ii, jj, kk):
        net = net + inp + self.corr(corr)          # corr: Linear(882,384)-ReLU-Linear-LN-ReLU-Linear
        net = self.norm(net)                        # LayerNorm(384, eps=1e-3)

        ix, jx = fastba.neighbors(kk, jj)            # CUDA: per-kk-group prev/next edge by jj order
        mask_ix = (ix >= 0).float().reshape(1,-1,1)
        mask_jx = (jx >= 0).float().reshape(1,-1,1)
        net = net + self.c1(mask_ix * net[:, ix])    # c1: Linear(384,384)-ReLU-Linear(384,384)
        net = net + self.c2(mask_jx * net[:, jx])    # c2: same shape

        net = net + self.agg_kk(net, kk)             # SoftAgg(384): grouped softmax+sum over kk
        net = net + self.agg_ij(net, ii*12345 + jj)  # SoftAgg(384): grouped softmax+sum over (ii,jj) pairs

        net = self.gru(net)                          # LN-GatedResidual-LN-GatedResidual, all dim 384
        return net, (self.d(net), self.w(net), None) # d: ReLU-Linear(384,2)-clip; w: same + Sigmoid
```

Shapes (`DIM=384`, patch size `P=3`, correlation radius 3, 2 pyramid
levels): `net`/`inp` are `(1, E, 384)`, `corr` is `(1, E, 882)`
(`= 2 levels × 49 taps × 3×3 patch pixels`), `ii`/`jj`/`kk`/`ix`/`jx` are all
`(E,)` int64 (`E` = current edge count in the patch graph — data-dependent,
changes every call). `flow` is accepted but **never read** inside `forward`
— dropped from the exported graph entirely. The checkpoint additionally
carries a vestigial `update.lmbda.*` 2-layer MLP that `net.py`'s current
`Update` class doesn't even define; `dpvo.py`'s own loader explicitly strips
`"update.lmbda"` keys before `load_state_dict`, and `export_dpvo_onnx.py`'s
`load_state_dict_subset` replicates that filter.

### What exported cleanly

`fnet.onnx` (128-ch, stride 4) and `inet.onnx` (384-ch context, stride 4) —
both `BasicEncoder4` instances (imported unmodified from `dpvo.extractor`,
which has zero CUDA/`torch_scatter` dependency) — exported with dynamic
`height`/`width` axes, opset 18, no surprises: plain conv/instance-norm/relu
graphs, same shape of problem `superpoint_1500.onnx` already solves. The
graphs bake in DPVO's own pre/post scaling (`2*(img/255)-0.5` in,
`fnet(img)/4.0` out) so the Rust-side contract is "raw `[0,255]` pixels in,
the exact tensor the rest of DPVO's pipeline consumes, out" — no hidden
scaling left for the Rust caller to get wrong.

The GRU update cell's **non-aggregation** math (context/corr fusion,
LayerNorm, the two `c1`/`c2` neighbor-gather branches, the GRU block, and
the `d`/`w` heads) also exported cleanly, as **two** static graphs (see next
section for why two). Both use an ordinary **dynamic `num_edges` axis**
rather than the pad-to-fixed-max trick the risk register anticipated —
every op left inside these two graphs (`Gather` by a precomputed index
tensor, elementwise ops, per-token `Linear`/`LayerNorm`) is shape-polymorphic
in the edge count already, so padding would only have wasted compute.
Confirmed working: the parity check below deliberately traces at `E=64` and
checks at `E=97` to exercise this.

One implementation subtlety worth flagging for M2: `net[:, ix]` with `ix`
containing `-1` (DPVO's "no neighbor" sentinel) relies on Python/NumPy-style
negative-index wraparound (index `-1` → last element, then zeroed out by
`mask_ix` *after* being fed through `c1`, whose bias terms mean a masked
contribution is **not exactly zero** — this is DPVO's own behavior, not a
bug introduced here). ONNX's `Gather` op supports the same negative-index
wraparound by spec, so this translated across without any special-casing —
confirmed by the parity numbers below, not just by reading the spec.

### What stayed host-side, and why (the SoftAgg / neighbors gap)

Two pieces resisted static-graph export, exactly as the risk register
anticipated ("softmax-aggregation may need host-side handling"):

1. **`fastba.neighbors(kk, jj)`** — a CUDA kernel (`fastba/ba.cpp`) that
   groups edge indices by `kk`, stable-sorts each group by `jj`, and returns
   each edge's previous/next sibling (or `-1`). This is pure integer
   bookkeeping over the *current* edge list, not a tensor computation — it
   has no ONNX equivalent worth having. Reimplemented host-side as
   `neighbors_cpu()` (pure Python, faithful translation of the C++), and its
   **output** (`ix`, `jx`) is simply fed into the pre-agg graph as an
   ordinary integer input tensor.
2. **`SoftAgg`** (`torch_scatter.scatter_softmax` + `scatter_sum`, called
   twice: `agg_kk` grouped by patch id, `agg_ij` grouped by `ii*12345+jj`
   edge-pair id) — a *data-dependent number of groups per call* (the patch
   graph's structure changes every frame). A static ONNX graph cannot
   allocate a `(num_groups, DIM)` scratch tensor whose `num_groups` is
   itself a traced value rather than a shape. This is a genuine, structural
   blocker, not a missing-op issue.

Consequence: the "one GRU update iteration" is exported as **two** static
sub-graphs with SoftAgg sandwiched between them on the host:

```
dpvo_update_pre_agg.onnx  : (net, inp, corr, ix, jx) -> net_pre_agg
host (Python now / Rust in M2):
    net_post_agg = net_pre_agg
                 + SoftAgg_kk(net_pre_agg, kk)
                 + SoftAgg_ij(net_pre_agg, ii*12345+jj)
dpvo_update_post_agg.onnx : net_post_agg -> (net_out, delta, weight)
```

`SoftAggReference` (in `export_dpvo_onnx.py`) reimplements `SoftAgg` in pure
PyTorch (`torch.unique` + `scatter_reduce_(reduce="amax")` + `scatter_add_`,
mathematically identical to `torch_scatter.scatter_softmax`/`scatter_sum`)
so the checkpoint's real `agg_kk`/`agg_ij` weights (`f`/`g`/`h`, each a
`Linear(384,384)`) still get used — this is small-scale, per-edge-graph math
(≤ a few thousand edges × 384 channels), exactly the kind of thing that
belongs as native Rust arithmetic in M2 rather than an ONNX Runtime call,
same philosophy as the correlation lookup already being scoped to native
Rust in §3/§4 of this doc. **M2 must port `neighbors_cpu` and
`SoftAggReference` to Rust** (both are small, well-specified, and now have
a working Python reference to test against).

Why this is *not* importing `dpvo.net`/`dpvo.blocks`/`dpvo.ba` directly:
those modules import CUDA-only extensions (`cuda_ba`, `cuda_corr`,
`lietorch_backends`) and the third-party `torch_scatter` package at module
scope; none are available on this CPU-only, no-CUDA-toolchain machine, and
no matching `torch_scatter` CPU wheel exists for `torch==2.13.0+cpu`.
`export_dpvo_onnx.py` instead imports `dpvo.extractor.BasicEncoder4`
directly (genuinely zero CUDA dependency) and re-implements the rest with
matching submodule names, so the official checkpoint still loads
`strict=True` onto every piece that needs weights.

### Parity results

All four graphs, real weights, ONNX Runtime `CPUExecutionProvider`, opset 18,
`torch==2.13.0+cpu`:

| Output | Shape | Max abs diff | Max rel diff | Threshold | Result |
| --- | --- | --- | --- | --- | --- |
| `fnet` fmap | (1,128,64,96) | 1.28e-06 | 1.14e-06 | 1e-4 | PASS |
| `inet` imap | (1,384,64,96) | 9.5e-07 | 3.2e-07 | 1e-4 | PASS |
| `dpvo_update_pre_agg` net_pre_agg | (1,97,384)* | 2.9e-06 | 5.2e-07 | 1e-4 | PASS |
| `dpvo_update_post_agg` net_out | (1,97,384)* | 9.5e-06 | 7.0e-07 | 1e-4 | PASS |
| `dpvo_update_post_agg` delta | (1,97,2)* | 2.5e-05 | 7.7e-07 | 1e-4 | PASS |
| `dpvo_update_post_agg` weight | (1,97,2)* | 5.8e-08 | 1.36 (near-zero denom) | 1e-4 | PASS |

\* traced/exported at `E=64`, checked at `E=97` to exercise the dynamic
`num_edges` axis end-to-end (not just re-running the export-time shape).

All six outputs pass the 1e-4 max-abs-diff threshold by 2-4 orders of
magnitude headroom — consistent with this repo's existing ONNX exports
(`docs/superpoint_onnx_runtime_plan.md`) and expected float32
CPU-EP-vs-eager-PyTorch noise, not a marginal pass. The random-weight
fallback path (`--checkpoint` omitted, same `--seed` on both sides since
random weights aren't persisted to disk) was also verified end-to-end and
passes identically — confirming the parity result is about graph-translation
correctness, not an artifact of the specific weights.

`weight`'s reported max-rel-diff of 1.36 is a false alarm from the
metric, not the model: `weight` is a sigmoid output pinned near 0 or 1 for
these random inputs, so its max-abs-diff denominator is near-zero and the
ratio is meaningless; the max-abs-diff (5.8e-08) is what governs the
PASS/FAIL threshold.

### BA / correlation / patchify fixtures for M2/M3 (best-effort, honestly-labeled)

Per the task's explicit guidance, these are **not** verified against
upstream's CUDA kernels (unavailable — no CUDA, and no local C++ build
environment set up here for a CPU-only `lietorch_backends`/`cuda_corr`
build), so each fixture's docstring/label states exactly what was and
wasn't cross-checked:

- **`fixtures/ba_fixture.npz`** — a tiny synthetic 3-frame/2-patch/4-edge
  scene, 2 Gauss-Newton iterations, using upstream's own
  Gauss-Newton-with-Schur-complement math (`block_matmul`/`block_solve`/the
  `BA()` structure, copied verbatim from `dpvo/ba.py` — confirmed to be
  upstream's real pure-PyTorch reference, just not importable directly
  because it also pulls in `torch_scatter`/`fastba`/`lietorch` at module
  scope) driven by a **from-scratch pure-PyTorch SE3** (`MiniSE3`:
  quaternion+translation, standard Lie-group conventions, left-perturbation
  `retr`). `MiniSE3` is self-validated (not upstream-cross-checked) via a
  float64 central-difference finite-difference check of the analytic
  `Ji`/`Jj` reprojection Jacobians against numerical differentiation:
  **max abs error 3.3e-09** — confirms the from-scratch SE3
  composition/inverse/action/adjoint formulas are internally consistent and
  correctly wired into the copied `projective_ops.transform` Jacobian
  formula, independent of not having lietorch's kernel to compare against.
  (An earlier float32-forward-difference version of this same check
  reported a spurious ~0.2-1.0 error that was pure truncation/roundoff noise
  on values of magnitude ~100, not a real bug — worth remembering if this
  check is ever "simplified" back to forward differences.)
- **`fixtures/patchify_fixture.npz`**, **`fixtures/correlation_fixture.npz`**
  — DPVO ships **no pure-Python reference** for `altcorr.patchify`/`corr`
  (both are CUDA-only `cuda_corr` calls). `patchify_cpu`/`corr_cpu` in
  `export_dpvo_onnx.py` are this script's **own** reimplementations
  (integer-window gather + upstream's own bilinear-blend arithmetic for
  patchify; `grid_sample`-based bilinear neighborhood sampling + normalized
  dot product for correlation, per this doc's §3 description of the
  otherwise-uninspectable kernel). Explicitly **not verified** against the
  true CUDA kernel; border/edge-clamp handling in particular may differ.
  Useful as an M2 regression fixture (does the Rust reimplementation match
  *a* reasonable reference), not as ground truth for "does this match
  upstream exactly."
- **`fixtures/update_cell_fixture.npz`** — the exact tensors used for the
  update-graph parity check above (`net`, `inp`, `corr`, `ix`, `jx`, plus
  `kk`/`ii`/`jj` and every intermediate: `net_pre_agg`, `net_post_agg`,
  `net_out`, `delta`, `weight`), for exact reuse by M2's Rust-side test.

### Blockers / open items for M2

1. **Port `neighbors_cpu` and `SoftAggReference` to Rust** (§ "What stayed
   host-side" above) — both are small, precisely specified now, and have a
   working Python reference + fixture to test against. This is the one
   piece of *new* logic M2 must write that isn't "call `ort` on an exported
   graph"; everything else in the update cell is now ONNX.
2. **Patch extraction and correlation lookup are still genuinely
   unverified against upstream** (no CUDA kernel available to cross-check
   this machine's — or a future Rust implementation's — output against).
   M2's native Rust patchify/correlation should be validated primarily
   against this repo's own downstream ATE/tracking metrics and internal
   consistency checks (e.g. the finite-difference-style self-check used for
   `MiniSE3` here), not treated as "matches DPVO" just because it matches
   this fixture.
3. **`weight`'s near-zero-denominator max-rel-diff metric is not a good
   parity metric for sigmoid/confidence outputs** — M2/M3's own parity
   tooling should lead with max-abs-diff (as this task specified) and treat
   relative-diff on saturating outputs as informational only.
4. No GPU/CUDA available on this machine for the whole M1-M6 arc — every
   subsequent milestone's parity/accuracy claims will be CPU-only, consistent
   with the risk register's existing "CPU latency" entry.

## M2 results (2026-07-17)

### Module layout

New crate module `crates/vision/src/dpvo/` (registered from `crates/vision/
src/lib.rs` as `pub mod dpvo;`), gated as a whole via `#![cfg(feature =
"onnx-inference")]` at the top of `dpvo/mod.rs` — a deliberate deviation
from `features::superpoint_onnx`/`lightglue_onnx`'s "always-visible stub,
runtime `FeatureDisabled` error" pattern, made because nothing in this
workspace calls into `dpvo` yet (M3+ wires it up); there is no existing
call site whose compilation needs protecting the way SuperPoint's CLI
plumbing did. Revisit if M3/M4 add a caller that needs the module name to
exist under `cargo build` without the feature.

| File | Contents |
| --- | --- |
| `dpvo/mod.rs` | Module doc (heavy — the SoftAgg-weights gap writeup lives here), `FNET_DIM`/`DIM`/`CORR_DIM`/`RES`/`PATCH`/`CORR_RADIUS`/`CORR_LEVELS` constants matching `manifest.json`. |
| `dpvo/onnx_session.rs` | `DpvoOnnxSession` — `ort`-backed wrapper around the 4 exported graphs (`load_from_paths`/`load_from_paths_with_backend`, reusing `features::superpoint_onnx::OnnxBackend`); `run_fnet`/`run_inet`/`run_update_pre_agg`/`run_update_post_agg`; `update_iteration` (the full pre-agg → host-`SoftAgg` → post-agg call, i.e. "one full update iteration" from the M2 scope). |
| `dpvo/patchify.rs` | `patchify_cpu` — native bilinear P×P patch extraction, ported from `export_dpvo_onnx.py`'s `patchify_cpu`. |
| `dpvo/correlation.rs` | `corr_cpu` — native normalized-dot-product correlation lookup with `grid_sample`-equivalent zero-padded bilinear sampling, ported from `export_dpvo_onnx.py`'s `corr_cpu`. Scoped to one shared target frame per call (see its module doc for the multi-`jj` batching left to M3/M4). |
| `dpvo/softagg.rs` | `SoftAgg` (grouped softmax-aggregation, `expand=True` path only) + `neighbors_cpu` (edge-neighbour bookkeeping), both pure Rust/`ndarray`, ported from `SoftAggReference`/`neighbors_cpu` in `export_dpvo_onnx.py`. `LinearWeights` helper for the `f`/`g`/`h` `Linear(384,384)` layers. |
| `dpvo/npz.rs` | Minimal `.npz` (uncompressed-ZIP-of-`.npy`) reader, written because no such reader exists anywhere in this workspace's `Cargo.lock` (checked directly: `grep -n '^name = ' Cargo.lock | grep -iE 'npy\|zip\|npz'` returns nothing) and the task forbids adding one. Not test-only — `SoftAgg::load_from_npz` uses it in production code. Only implements `ZIP_STORED` (uncompressed) entries, which is all `numpy.savez` ever produces and all these fixtures use (verified per-file via Python's `zipfile.ZipFile(...).infolist()[*].compress_type == 0` before writing this). |
| `crates/vision/tests/dpvo_onnx_parity.rs` | The 5 fixture-based integration tests (all `#[ignore]`-gated), plus a small `time_repeated` timing helper. |

Genuine gap found and closed during M2, not anticipated by M1: **`SoftAgg`'s
trained `f`/`g`/`h` `Linear(384,384)` weights (`update.agg_kk.*`/
`update.agg_ij.*`) never made it into any M1 artifact** — they live entirely
outside the ONNX boundary (SoftAgg stayed host-side by construction, see M1
results above), and M1's own fixture dump never captured them either. This
would have silently blocked the "SoftAgg fixture parity" test the task asked
for. Closed by extending `scripts/export_dpvo_onnx.py` with a
`dump_softagg_weights_fixture` step (writes `fixtures/
softagg_weights_fixture.npz`) rather than a throwaway one-off script, so
regenerating fixtures from scratch is self-sufficient again. Extraction
correctness was self-checked by recomputing `update_cell_fixture.npz`'s own
`net_post_agg` from the freshly-dumped weights: **max abs diff `0.0`**
(exact — same checkpoint, same modules, deterministic float ops). All four
`.onnx` graphs and all five fixtures were regenerated from the (now-updated)
tracked script and re-verified against `check_dpvo_onnx_parity.py` — same
PASS numbers as the M1 table above (`fmap`/`imap`/`net_pre_agg`/`net_out`/
`delta`/`weight`, 1.28e-06 to 2.48e-05 max abs diff, all `PASS` at the 1e-4
threshold — see that unchanged table).

### Parity table (Rust, `cargo test -p visloc-vision --features onnx-inference --test dpvo_onnx_parity -- --ignored`)

| Test | What it checks | Max abs diff | Threshold | Result |
| --- | --- | --- | --- | --- |
| `patchify_parity_against_fixture` | `patchify_cpu` vs `patchify_fixture.npz` | **0.000e0** (exact) | 1e-4 | PASS |
| `correlation_parity_against_fixture` | `corr_cpu` vs `correlation_fixture.npz` | 2.515e-7 | 1e-4 | PASS |
| `softagg_parity_against_fixture` | `net_pre_agg + agg_kk(...) + agg_ij(...)` vs `update_cell_fixture.npz`'s `net_post_agg`, weights from `softagg_weights_fixture.npz` | 3.815e-6 | 1e-4 | PASS |
| `update_cell_end_to_end_parity_against_fixture` | full `update_iteration` (`neighbors_cpu` → `ort` pre-agg → host `SoftAgg` → `ort` post-agg) vs fixture | net_out 1.049e-5, delta 5.531e-5, weight 5.661e-8 | 1e-4 | PASS (all three) |
| `fnet_inet_sessions_load_and_produce_finite_output_of_the_documented_shape` | `ort` loads/runs `fnet.onnx`/`inet.onnx`, output shape `(1,128,H/4,W/4)`/`(1,384,H/4,W/4)`, all-finite | — (shape/finiteness only; exact PyTorch-vs-ONNXRuntime numeric agreement already established in Python by M1, see above) | — | PASS |

All four numeric parity tests pass 2-5 orders of magnitude inside the 1e-4
threshold — not a marginal pass, consistent with M1's own headroom. The
`neighbors_cpu` output (`ix`/`jx`) is also checked bit-exactly (`assert_eq!`,
integer bookkeeping, no tolerance needed) inside the end-to-end test before
the ONNX/SoftAgg stages run, so a mismatch there is localized rather than
only surfacing as a downstream numeric drift.

Also unaffected and re-confirmed: `cargo test -p visloc-vision --features
onnx-inference` (148 passed, 0 failed — 128 pre-existing + 20 new always-on
unit tests for `neighbors_cpu`/`SoftAgg` math, `patchify_cpu` bilinear
cases, `corr_cpu` border/normalization cases, and the `npz` reader's
hand-built-archive round trip); `cargo test -p visloc-slam` (278+54+6+6+132+
10+9+4 = unaffected, all passing, 0 regressions — this milestone touched no
`visloc-slam` file); `cargo check --workspace --all-targets --features
image-io,onnx-inference` (clean); `cargo clippy -p visloc-vision
--all-targets --features onnx-inference` (clean, 0 warnings, after fixing a
type-complexity lint via a `UpdateCellOutput` type alias, a doc-list
indentation lint, and two `repeat().take()` → `repeat_n()` lints raised
during development).

### Per-stage CPU timings (rough, `Instant`-based, `--release`, this fixture's shapes)

Measured via the `time_repeated` helper in `dpvo_onnx_parity.rs`
(`--test-threads=1`, `--release`; **debug-build numbers are 10-100x slower
and not representative** — the first pass at these numbers was taken in a
debug build and was wildly misleading, e.g. `corr_cpu` at 224 ms instead of
0.5 ms, purely from unoptimized bounds-checked `ndarray` indexing; re-run in
`--release` before trusting any number here). Shapes are the fixtures' own
(small: 5 patches / 64 edges / 64×96 images), **not** representative of a
real EuRoC frame (752×480, ~80 patches) — treat these as a rough
per-primitive cost floor, not a per-frame budget; the encoder-cost
extrapolation below is the closest this gives to the latter.

| Stage | Shape | Time |
| --- | --- | --- |
| `patchify_cpu` | 5 patches × 128 ch × 3×3 | 0.044 ms/call |
| `corr_cpu` | 5 edges × 3×3 patch × 49 taps × 128 ch | 0.509 ms/call |
| `run_fnet` (ONNX, `ort` CPU EP) | 64×96 input → `(1,128,16,24)` | 4.755 ms/call |
| `run_inet` (ONNX, `ort` CPU EP) | 64×96 input → `(1,384,16,24)` | 2.227 ms/call |
| `SoftAgg` host step (`agg_kk`+`agg_ij`) | 64 edges × 384 dim | 2.213 ms/call |
| `update_iteration` (end-to-end: `neighbors_cpu` + `ort` pre-agg + host `SoftAgg` + `ort` post-agg) | 64 edges | 5.780 ms/call |

Two sanity checks against the plan doc's §3 CPU-feasibility estimate (a
reasoning chain, explicitly not a measurement, made before any DPVO ONNX
artifact existed):

* **One GRU update iteration**: §3 guessed "plausibly tens of ms via `ort`
  CPU EP". Measured: **5.78 ms** at 64 edges — better than the guess, and
  DPVO's real per-frame edge count (~80 patches × a handful of temporal
  neighbours each) is the same order of magnitude, so this stage is very
  unlikely to be the per-frame bottleneck.
* **Encoder stage** (`fnet`+`inet`): §3 guessed "~250-450 ms/frame" at full
  EuRoC resolution (752×480), by analogy by to SuperPoint's measured 165 ms
  at that resolution. Measured here at 64×96: `fnet` 4.76 ms + `inet`
  2.23 ms ≈ 7 ms combined. Convolutional cost scales roughly with pixel
  count (~58.75× more pixels at 752×480 vs 64×96), which would put full-res
  cost in the ballpark of **~410 ms combined** — landing inside §3's
  original estimated range, though this is an extrapolation from one small
  shape, not a direct measurement at full resolution (that is a natural M3
  follow-up: re-run `fnet_inet_sessions_load_and_produce_finite_output_of_
  the_documented_shape`-style timing at 752×480 once a real EuRoC frame is
  wired through, rather than trusting the scaling argument alone).

`patchify_cpu`/`corr_cpu` are negligible at this fixture's scale (5 patches)
and were already expected to stay negligible at DPVO's real ~80-patch scale
per §3 ("a few thousand small dot products... negligible (<5 ms)");
nothing measured here contradicts that.

### Blockers / open items for M3

1. **DBA / patch inverse-depth solver** (the plan doc's actual M3 scope) is
   entirely unstarted, as instructed — nothing in this milestone touched
   `bundle.rs` or `sparse_factor_graph.rs`.
2. **Production SoftAgg weight sourcing is still fixture-shaped, not a real
   deployment path.** `SoftAgg::load_from_npz` reads the `.npz` fixture
   format M2 had to invent (`{prefix}f_weight`/`{prefix}f_bias`/...); M3/M4
   need a real decision on how a production caller obtains these 12 tensors
   outside of a test fixture — likely folding
   `dump_softagg_weights_fixture` into whatever M4's actual model-loading
   entry point looks like (a sibling artifact next to the 4 `.onnx` files,
   not a 5th ONNX graph, since the op itself cannot be an ONNX graph — see
   M1 results). Low risk (the extraction is already proven exact), but
   undecided.
3. **`corr_cpu`'s single-shared-target-frame scope is real, not just a
   fixture simplification.** Assembling the full `(E, 882)` = 2-pyramid-
   level × 49-tap × 3×3-patch correlation tensor the update cell's ONNX
   graphs expect requires: (a) building a 2-level average-pooled pyramid
   per destination frame, (b) grouping edges by their `jj` (each edge can
   target a different frame), (c) calling `corr_cpu` once per distinct
   target frame (or batching it — not yet done either way), and (d)
   concatenating/flattening the two levels' taps in whatever byte order
   DPVO's own code expects. None of this is covered by any M1 fixture
   (`correlation_fixture.npz` only exercises one frame, one level), so it
   is unverified even against the "own reference" bar the rest of this
   milestone cleared. This is the single largest actual unknown standing
   between M2's primitives and a real per-frame DPVO forward pass — budget
   it explicitly in M3/M4 rather than assuming `corr_cpu` alone is
   sufficient.
4. **Patchify/correlation border-handling honesty caveat still applies
   unchanged** (M1's own caveat, reconfirmed, not newly discovered): these
   two ops are validated against `export_dpvo_onnx.py`'s own reference, not
   upstream's CUDA kernel. Nothing in M2 closes this gap; M4's downstream
   ATE/tracking metrics remain the intended arbiter per the risk register.
5. **No wiring into any pipeline** (`OnlineSlamPipeline`,
   `sparse_factor_graph.rs`, etc.) exists yet, as instructed — this module
   has zero callers today. `dpvo/mod.rs`'s whole-module feature gate should
   be revisited once M3/M4 add one (see that module's doc comment).
6. **Full-resolution (752×480), real-EuRoC-frame timing is still
   extrapolated, not measured** (see the timings section above) — a
   reasonable next step alongside M3/M4's first real integration pass,
   once real image data is flowing through `run_fnet`/`run_inet` rather
   than a synthetic 64×96 fixture.
