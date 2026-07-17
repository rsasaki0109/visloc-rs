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

## M3 results (2026-07-17)

### Placement

New module `pipelines/slam/src/dpvo_patch_ba.rs` (registered from
`pipelines/slam/src/lib.rs` as `pub mod dpvo_patch_ba;`, re-exporting
`dpvo_ba`/`dpvo_ba_step`/`DpvoBaProblem`/`DpvoBaConfig`/`DpvoBaError`/
`DpvoPatch`/`DpvoEdge`/`DpvoIntrinsics`/`se3_from_dpvo_pose`/
`dpvo_pose_from_se3`) — a sibling of `bundle.rs`, not an extension of it.
Rereading `ba.py` line by line (rather than the plan doc's earlier,
pre-M1/M2 §4 speculation) showed the two solvers are structurally
incompatible for a clean merge: `BundleAdjustment` runs an open-ended
Levenberg-Marquardt loop with adaptive `λ`/cost-based accept-reject over
`Point3<f64>` landmarks under a **right**-perturbation pose convention
(`T ← T · Exp(ξ)`); `fastba.BA` is a **fixed two-call** Gauss-Newton with no
line search, a scalar (not 3-DoF) patch inverse-depth variable, its own
three-part damping scheme, hard (non-Huber) validity gating, and a
**left**-perturbation convention (`T ← Exp(ξ) · T`, matching `lietorch`).
Splicing the two would either silently change `ba.py`'s semantics to fit
the host loop or bolt two incompatible retraction conventions onto one
struct. The new module instead reuses `bundle.rs`'s Schur-complement
*spirit* and `visloc_core::geometry::SE3` directly (no core changes needed
— see "Convention mapping" below) while keeping `ba.py`'s own damping/
iteration/gating logic byte-for-byte diffable against upstream. It lives in
`pipelines/slam/src`, not a `dpvo`-prefixed module under `crates/vision`,
because it is pure `nalgebra` math with **no** `ort`/ONNX dependency — the
task's constraint ("must NOT be gated behind `onnx-inference`") and
`visloc-slam`'s `Cargo.toml` (no such feature anywhere in the crate) make
that the natural home. `crates/vision/src/dpvo` (M2) is untouched; the two
modules share only the upstream repo, not any Rust code. No pipeline
wiring, sliding window, or patch lifecycle was added — that stays M4.

### Ported line ranges

| Piece | Upstream source | Lines |
| --- | --- | --- |
| `BA()` (Gauss-Newton + Schur, damping, gating, retraction) | `E:/tools/DPVO/dpvo/ba.py` | 86–182 |
| `block_matmul`/`block_solve` (block reshape + the `ep`/`lm` damping formula) | same | 58–76 |
| `safe_scatter_add_mat`/`safe_scatter_add_vec` (fixed-pose-index masking) | same | 40–46 |
| `transform` (projective transform + analytic Jacobians `Ji`/`Jj`/`Jz`) | `E:/tools/DPVO/dpvo/projective_ops.py` | 53–113 |
| `iproj`/`proj` (unnormalized-homogeneous backproject/project) | same | 19–50 |
| `MiniSE3` (from-scratch SE3 stand-in for `lietorch.SE3`, self-validated) | `scripts/export_dpvo_onnx.py` | 352–481 |
| `pops_transform`/`mini_ba` (CUDA/`torch_scatter`-free re-hosting of the two files above) | same | 483–667 |
| `dump_ba_fixture` (the two chained `mini_ba` calls that produced `ba_fixture.npz`) | same | 670–757 |
| `dpvo.py`'s real call sites (confirms `BA()` is one GN step; `iterations=2`/`t0` live only on the CUDA entry point) | `E:/tools/DPVO/dpvo/dpvo.py` | 312–356 |

### Convention mapping summary

* **Pose semantics**: each DPVO pose is `T_world_to_camera` for its frame
  (from `MiniSE3`'s own docstring and from how `Gij = poses[jj]·poses[ii]⁻¹`
  type-checks) — identical to `visloc_core::geometry::Pose::world_to_camera`,
  so `DpvoBaProblem::poses: Vec<SE3>` needs no semantic translation.
* **Numeric layout**: DPVO pose rows are `[tx, ty, tz, qx, qy, qz, qw]`
  (translation + xyzw quaternion). `se3_from_dpvo_pose`/`dpvo_pose_from_se3`
  convert at the boundary using this codebase's own established
  `Quaternion::new(qw, qx, qy, qz)` idiom (already used in `g2o.rs`,
  `crates/io/src/euroc.rs`, `colmap/mod.rs`, `pose_graph.rs`) — no new
  reordering bug introduced.
* **Composition**: `MiniSE3.mul(self, other) = self ∘ other` is bit-for-bit
  `SE3::compose`'s own formula; `Gij = poses[jj]·poses[ii]⁻¹` ports directly
  as `poses[jj].compose(&poses[ii].inverse())`.
* **Retraction**: DPVO/lietorch use **left** perturbation,
  `retr(a) = Exp(a) · X` (`MiniSE3`'s own docstring, confirmed by
  `mini_ba`'s pose-update loop) — the opposite of `bundle.rs`'s right
  perturbation. Ported as `SE3::exp(&dx).compose(&pose)`, a deliberate,
  documented difference from `bundle.rs`, not an inconsistency.
* **Tangent layout**: DPVO's `xi = (ρ, φ)` (translation-first) matches
  `SE3::exp`/`SE3::log`'s own `Vector6` layout exactly — no permutation.
* **Adjoint**: `MiniSE3.adjoint()`'s `[[R, [t]×R], [0, R]]` is bit-for-bit
  `SE3::adjoint`'s own formula, so `Gij.adjT(a) = a·Ad(Gij)` ports as
  `a * g_ij.adjoint()` with **no new core method** — confirms the plan
  doc's §5 note that M3 would need no new SE3/SO3 API.
* **Homogeneous point action**: DPVO's four-vector `(X, Y, Z, W)` carries
  the patch's inverse depth as the invariant trailing component `W`
  (`(X,Y,Z,W) ↦ (R·(X,Y,Z) + t·W, W)`); implemented directly with
  `SE3::rotation`/`SE3::translation`, no new abstraction needed.
* **Damping (three distinct constants)**: `lmbda` (depth-channel Tikhonov,
  `Q = 1/(C+lmbda)`, always `1e-4` at real call sites) ≠ `ep` (`BA()`'s own
  default `100.0`, flat additive pose-diagonal term) ≠ `lm = 1e-4`
  (`block_solve`'s hardcoded multiplicative pose-diagonal scale, never a
  `BA()` parameter at all). All three ported as separate constants/fields;
  see `dpvo_patch_ba.rs`'s module doc for the exact per-scalar-diagonal
  formula (`S[d,d] ← S[d,d]·(1+lm) + ep`, off-diagonals untouched).
* **Robustness**: `ba.py` has **no Huber/robust kernel** — only three hard
  `0`/`1` gates (`Z > 0.2` at the center pixel, `‖residual‖ < 250` px,
  predicted-pixel-in-`bounds`) zero the residual and weight together before
  any product forms. Ported as literal boolean masks, per the task's
  "only if the reference has it" instruction — nothing invented.
* **`t0`/`fixedp`**: the first `fixedp` poses (`dpvo.py`'s `t0`, `ba.py`'s
  `fixedp`, default `1`) are excluded from the pose Hessian/gradient/cross
  term entirely, but edges anchored at them **still** contribute to their
  patch's depth Hessian/gradient (`C`/`w`, keyed only on `kk`, never
  `ii`/`jj`) — ported as two independently-gated accumulation paths, not
  one shared mask.
* **Iteration count**: `ba.py`'s `BA()` is **one** Gauss-Newton step; "2
  iterations" is `dpvo.py`'s own caller-side loop around the compiled CUDA
  entry point (`iterations=2` is a CUDA-only argument, absent from the
  pure-Python `BA()` signature). `dpvo_ba_step` is one `BA()` call;
  `dpvo_ba` is the `config.iterations`-call outer loop, matching
  `dump_ba_fixture`'s own two chained `mini_ba` calls.
* **Inverse-depth clamp**: `disps.clamp(min=1e-3, max=10.0)` applied to
  every patch (not just updated ones) after retraction — ported as-is
  (`DISP_MIN`/`DISP_MAX`).

### Parity numbers

Fixture: `E:/visloc_archive/dpvo_onnx_m1/fixtures/ba_fixture.npz` (3 frames,
2 patches, 4 edges, `fixedp=1`, `ep=100.0`, `lmbda=1e-4`, 2 chained GN
iterations — inspected directly via
`E:/tools/venvs/dpvo_export/Scripts/python.exe -c "numpy.load(...)"`, all
expected keys present, no fixture extension needed this milestone — unlike
M1/M2, which each found and closed a genuine fixture gap).

`cargo test -p visloc-slam --test dpvo_patch_ba_fixture -- --ignored --nocapture`:

| Test | Max abs pose diff (translation m / rotation-matrix entries) | Max abs inverse-depth diff | Threshold | Result |
| --- | --- | --- | --- | --- |
| `ba_fixture_one_iteration_matches_reference_within_1e_4` | 1.086e-6 | 8.288e-7 | 1e-4 | PASS |
| `ba_fixture_two_iterations_matches_reference_within_1e_4` | 1.007e-6 | 1.932e-6 | 1e-4 | PASS |

Both pass **two orders of magnitude** inside the 1e-4 threshold — not a
marginal pass, consistent with M1/M2's own headroom. The `1e-4` threshold
itself is justified in the test's own doc comment: `ba_fixture.npz`'s
inputs and reference outputs were produced by fp32 `torch` arithmetic
throughout (`mini_ba` never upconverts), while this Rust port computes
entirely in `f64` — the two differ only by accumulated fp32-vs-f64 rounding
over two Gauss-Newton iterations of a small, well-conditioned
(`ep=100`-damped) dense solve, and the measured diffs (~1e-6) confirm that
gap is far smaller than the 1e-4 bar.

### Always-on unit tests (`cargo test -p visloc-slam --lib dpvo_patch_ba`)

6 tests, all passing, covering the four required hand-built scenarios plus
two extras:

| Test | Requirement |
| --- | --- |
| `zero_residual_is_a_fixed_point` | Zero-residual fixed point (perfect data unchanged) |
| `known_one_step_gauss_newton_matches_hand_solved_6x6_system` | Known 1-step GN reduction, checked against an independently hand-solved 6×6 system |
| `all_poses_fixed_leaves_every_pose_untouched` | Fixed-pose invariance |
| `depth_only_updates_when_all_poses_fixed` | Depth-only update when all poses fixed (the `n2==0` / `structure_only`-equivalent branch, cross-checked against the direct `dZ = Q·w` formula) |
| `dpvo_pose_round_trips_through_dpvo_layout` | Boundary-conversion sanity (not one of the four required, but cheap insurance on the convention mapping) |
| `two_iterations_reduces_residual_further_than_one` | Sanity check that `dpvo_ba`'s outer loop actually re-linearizes (not required, but catches an easy "loop doesn't do anything" bug) |

### Timings on the fixture problem

`--release`, `Instant`-based, 2000 repeats of the full 2-iteration
`dpvo_ba` call on the fixture's own tiny (3 frames / 2 patches / 4 edges)
problem size: **0.0072 ms/call** (7.2 µs). This is a dense `12×12` Schur
solve plus a handful of `2×6`/`6×6` matrix products per edge — negligible
next to M2's measured encoder-stage cost (`fnet`+`inet` ≈7 ms at 64×96,
extrapolated to ≈410 ms at full EuRoC resolution) and consistent with the
plan doc's own §3 estimate ("sub-to-few-ms via `nalgebra`"). Not yet
measured at DPVO's real per-frame scale (≤80 patches × `OPTIMIZATION_WINDOW=12`
poses, an order of magnitude larger `n2`/`m` than this fixture) — a natural
M4 follow-up once real patch-graph sizes are flowing through this solver.

### Verification

* `cargo test -p visloc-slam --lib dpvo_patch_ba`: 6 passed, 0 failed.
* `cargo test -p visloc-slam --test dpvo_patch_ba_fixture -- --ignored --nocapture`:
  2 passed, 0 failed (numbers above).
* `cargo test -p visloc-slam` (whole crate): 284 lib tests (278 pre-existing
  + 6 new) + 54+2(new, ignored)+6+6+132+10+9+4 across integration test
  binaries, 0 failed, 0 regressions.
* `cargo check --workspace --all-targets --features image-io,onnx-inference`:
  clean.
* `cargo clippy -p visloc-slam --all-targets`: 6 pre-existing warnings, all
  in files this milestone did not touch (`map_atlas.rs`,
  `online_slam_vi_ba.rs` ×2, `online_slam.rs`, `online_slam_motion_vi_init.rs`,
  `vi_motion_initializer.rs`); **zero** warnings in `dpvo_patch_ba.rs` or
  `tests/dpvo_patch_ba_fixture.rs` (confirmed by grepping the clippy output
  for `dpvo`).

### Blockers / open items for M4

1. **No pipeline wiring exists yet**, as instructed — `dpvo_patch_ba.rs` has
   zero callers today. M4 needs to decide how `sparse_factor_graph.rs`'s
   `SparseFactorMeasurement` (target correction + information + damping)
   maps onto `DpvoBaProblem`'s `targets`/`weights`/`config.lmbda` — the plan
   doc's §4 port-surface table already sketches this, but the concrete glue
   (patch id ↔ `SparseFactorKey`, edge lifecycle ↔ `SparseFactorState`) is
   unwritten.
2. **`n = poses.len()` simplification**: this module derives the active
   frame count from the caller's pose array length rather than
   `max(ii,jj)+1` as `ba.py` does. True whenever every declared pose is
   referenced by at least one edge (true of every fixture/test here); M4's
   real sliding-window caller must ensure that invariant or this module
   needs a small follow-up to derive `n` from the edges instead.
3. **Only scalar `lmbda` is implemented**, not `ba.py`'s generic per-patch
   tensor broadcast — an honest, documented narrowing since no real DPVO
   call site exercises the tensor form either.
4. **Timing is only measured at the fixture's tiny scale** (3 frames/2
   patches/4 edges); DPVO's real per-frame scale is ≤80 patches ×
   `OPTIMIZATION_WINDOW=12` poses, an order of magnitude larger — M4 should
   re-measure once real patch-graph sizes exist, the same "measure, don't
   extrapolate" discipline M2's own blockers list called out for the
   encoder stage.
5. **The `3×3` patch-grid replica** (`patches_in`'s full `(1, n_patches, 3,
   P, P)` shape) is discarded at the fixture-loading boundary (verified
   uniform, then collapsed to one `(x, y, d)` triple) because only the
   center pixel ever enters `BA()`'s math — but M4's real patch lifecycle
   (M2's `patchify`/`corr_cpu`) does need the full grid for correlation
   lookups. `DpvoPatch` as defined here is BA-only; M4 will need a richer
   per-patch type (or a clear boundary conversion) once the two modules are
   wired together.

## M4 results (2026-07-17)

### Architecture recap

Two new `pipelines/slam/src` modules plus one example, following the M4
task's "graph/BA logic unconditional, inference behind `onnx-inference`"
split:

| File | Feature gate | Contents |
| --- | --- | --- |
| `dpvo_patch_graph.rs` | none (always compiled) | `DpvoVoConfig` (`config.py` port), `DpvoPatchGraph` (frame/patch buffers stored as plain compacting `Vec`s rather than `dpvo.py`'s fixed `BUFFER_SIZE`-shaped tensors), `edges_forw`/`edges_back` (`dpvo.py:362-375`), `keyframe`/`motionmag` (`dpvo.py:257-310`), the `DAMPED_LINEAR` motion model (`dpvo.py:410-424`), `reconstruct_pose` (`dpvo.py::get_pose`, for folded/rejected frames). |
| `dpvo_vo.rs` | `#![cfg(feature = "onnx-inference")]` | `DpvoOdometry` — wires M2's `DpvoOnnxSession`/`patchify_cpu`/`SoftAgg`/`corr_cpu` and M3's `dpvo_ba` into `dpvo.py`'s per-frame loop (`__call__`/`update`/`motion_probe`), plus the 2-pyramid-level correlation assembly M2 left as its "biggest blocker" (`corr_pyramid`, grouping active edges by target frame). |
| `examples/euroc_dpvo_vo_demo.rs` | `required-features = ["image-io", "onnx-inference"]` | EuRoC cam0 runner: full-resolution radial-tangential undistortion (own dense-image warp, since no such utility existed in this repo — keypoint-only `RadialTangential::undistort_pixel` isn't enough for a dense CNN front end), trajectory CSV, Umeyama-aligned ATE (reusing `umeyama_similarity_transform`/`TrajectorySimilarityTransform` from `pipelines/tracking`, the same pattern `examples/euroc_imu_dead_reckon_demo.rs` already uses). |

`dpvo_patch_ba.rs` (M3) gained three small, cited additions the graph/VO
layers needed and M3's BA-only scope never did: `transform_point`
(jacobian-free `projective_ops.py::transform`, generalizing `tonly`),
`reproject_patch_grid` (the same, evaluated over a patch's full `3×3` grid
— needed for correlation lookups, not just the BA center pixel), and
`flow_mag` (`projective_ops.py::flow_mag`, the keyframe motion-magnitude
gate). All three are exercised by both `dpvo_patch_ba.rs`'s own new unit
tests and transitively by `dpvo_patch_graph.rs`'s keyframe tests.

Cargo wiring: `pipelines/slam/Cargo.toml` gained an `onnx-inference`
feature (`["visloc-vision/onnx-inference", "dep:ndarray"]`) and an
always-on `rand` dependency (already a workspace dependency used
elsewhere — not a new external crate); root `Cargo.toml`'s own
`onnx-inference` feature now also forwards to `visloc-slam/onnx-inference`
and gained `dep:ndarray` (for the new example's `Array2<u8>` image
buffer). `Cargo.lock`'s package set is unchanged (checked directly via
`git diff Cargo.lock | grep '^\+name = '` — empty), confirming no new
external crate entered the dependency graph.

### Genuine findings from re-reading the reference implementation

1. **DPVO's published EuRoC number does not downscale the image.**
   `dpvo/stream.py::image_stream` has a dead `if 0:` half-resolution
   branch; the real path keeps the full 752×480 image and only
   *temporally* subsamples frames (`evaluate_euroc.py --stride 2`, i.e.
   ~10 Hz effective rate, not ~20 Hz). The task brief's own "likely
   half-res" guess was a hypothesis to check against the primary source,
   not a fact — checking it shows it is false. `examples/euroc_dpvo_vo_demo.rs`
   defaults to full resolution and `--stride 2`, matching upstream.
2. **The config that produced the published number is not `config.py`'s
   bare defaults.** `E:/tools/DPVO/config/default.yaml` overrides
   `PATCHES_PER_FRAME=96`, `REMOVAL_WINDOW=22`, `OPTIMIZATION_WINDOW=10`,
   `PATCH_LIFETIME=13`, `KEYFRAME_THRESH=15.0`. `DpvoVoConfig::default()`
   (in `dpvo_patch_graph.rs`) intentionally still matches `config.py`'s
   bare numbers (the more "canonical" reference point the plan doc's own
   architecture table already cites); the EuRoC example instead exposes
   all five as CLI flags with `config/default.yaml`'s values as its own
   defaults — except see finding 3, which forced this milestone's actual
   reported run to use `config/fast.yaml`'s smaller sizing instead.
3. **Active-edge count — not the BA problem size — is what actually
   saturates a naive CPU port, and it is far larger than the M1/M2
   fixtures' scale suggested.** `dpvo_ba` (M3) itself only ever solves a
   dense system bounded by `OPTIMIZATION_WINDOW` free poses (tiny, ported
   as a documented windowing scheme — see `dpvo_vo.rs`'s module doc). But
   the correlation-assembly stage (`reproject_patch_grid` + `corr_cpu`,
   grouped by target frame) scales with the **active edge count**, which
   is `O(REMOVAL_WINDOW × PATCHES_PER_FRAME × PATCH_LIFETIME)` — tens of
   thousands of edges at `config/default.yaml`'s sizing. On the real
   (GPU) DPVO this is one batched CUDA kernel call; this port's
   from-scratch, correctly-CPU-scoped-but-never-before-measured-at-this-
   scale `corr_cpu`/`SoftAgg` (M1/M2) pay for it as nested serial Rust
   loops. Attempting a 400-frame run at `default.yaml` sizing (bare
   `--patches-per-frame 96 --removal-window 22 --optimization-window 10
   --patch-lifetime 13`) made no visible progress past the initial
   8-frame/12-iteration bootstrap within several minutes of wall-clock
   time and was aborted — not a hang or a bug (the bootstrap itself
   completed, and a smaller/shorter run at this sizing does terminate,
   just very slowly), but a genuine CPU-infeasibility finding, consistent
   with the plan doc's own risk register ("CPU latency... not real-time
   on this CPU-only machine") — this milestone had simply not previously
   measured *how much* slower at real (not fixture) edge counts.
   `config/fast.yaml`'s sizing (`PATCHES_PER_FRAME=48, REMOVAL_WINDOW=16,
   OPTIMIZATION_WINDOW=7, PATCH_LIFETIME=11` — DPVO's own shipped
   reduced-compute config, for exactly this tradeoff) was used for the
   reported run below instead: an honest, cited substitution, not
   `config.py`'s bare defaults and not the paper's own config.
4. **A per-stage timing split, added after the first full run's numbers
   didn't add up, isolated the actual bottleneck precisely.** The first
   400-frame run's `ms_per_frame_total` (wall clock ÷ frames) exceeded the
   sum of the three stages [`DpvoOdometryStats`] tracked (encode + GRU
   update + BA) by roughly 4-8× — a real gap, not measurement noise. Bisecting
   it (adding an explicit timer around the reprojection+`corr_pyramid`
   assembly block, previously untimed) isolated **correlation-tensor
   assembly**, not encoding or the GRU update, as the dominant cost by a
   wide margin — see the timing table below. `DpvoOdometryStats` and the
   example's own summary output were both extended with a
   `correlation_ms_total`/`ms_per_frame_correlation` field so this stays
   visible rather than silently reabsorbed into "everything else" for any
   future milestone.

### EuRoC MH_01 run

`--euroc-dir MH_01_easy --max-frames 400 --stride 2` (matching
`evaluate_euroc.py`'s own default stride) with `config/fast.yaml`-sized
graph config (`--patches-per-frame 48 --removal-window 16
--optimization-window 7 --patch-lifetime 11`, see finding 3), CPU
(`OnnxBackend::default()`, CUDA-then-CPU fallback — no CUDA available on
this machine, consistent with every prior milestone), monocular (no IMU —
M5's scope), `--seed 0`.

```
frames_requested=400
frames_tracked=400
tracked_fraction=1.0000
total_elapsed_s=1789.99
ate_rigid_rmse_m=0.1599
ate_rigid_max_m=0.4075
ate_similarity_rmse_m=0.1561
ate_similarity_max_m=0.4079
ate_similarity_scale=1.359171
gt_matched_samples=400
```

**Honest assessment**: 100% of frames were accepted by `motion_probe` and
produced a pose (no NaN, no divergence, no crash across the full 400-frame
run). ATE (both rigid and similarity Umeyama alignment) sits in the
**0.15-0.16 m** range against ground truth over this leading ~400-frame
(~stride-2, so ~800 real-time frames ≈ 40 s of the sequence at 20 Hz)
segment of MH_01 — roughly **1.8-2×** DPVO's own published *full-sequence*
number (~0.087 m, GPU, similarity-aligned, `default.yaml` sizing, full
~3600 frames) and noticeably worse than this repo's existing
SuperPoint+LightGlue-based full-stack baseline (~2.9 m ATE on the *full*
sequence per this plan doc's own opening motivation — so on a like-for-like
short-prefix basis this port is likely already ahead of that baseline,
though a direct short-prefix-vs-short-prefix comparison was not run this
milestone). Per the acceptance criteria's own framing, this is reported
as a first honest data point, not a claim of matching the paper: the
smaller `fast.yaml` graph sizing (fewer patches, shorter windows) trades
accuracy for CPU tractability (finding 3), and CPU float32-vs-f64 and
native-Rust-vs-CUDA-kernel differences in `corr_cpu`/`patchify_cpu` (M1/M2's
own "not verified against upstream's CUDA kernel" caveat, never resolved
because no CUDA toolchain exists in this environment) remain fully live
and un-decomposed from graph-sizing effects.

### Timing breakdown (ms/frame)

Two data points, since a full per-stage breakdown was only added *after*
the first full run (see finding 4); both use `fast.yaml` sizing:

| Stage | 20-frame diagnostic (graph depth `n≈10`) | 400-frame full run (graph depth grows `n: 1→37`) |
| --- | --- | --- |
| Image I/O (`read_common_image`) | 5.0 ms | *(not separated in the first run; ≈3-6 ms based on the diagnostic)* |
| Dense undistortion (own warp) | 15.8 ms | *(ditto, ≈12-19 ms)* |
| `fnet`+`inet` encode (ONNX) | 278.7 ms | 211.5 ms |
| **Correlation assembly** (`reproject_patch_grid` + `corr_cpu`, grouped by target frame) | **3137.4 ms** | *(not separated — folded into the ~3700 ms/frame gap between `ms_per_frame_total` and the sum of the other tracked stages)* |
| GRU update (ONNX `update_iteration`) | 341.1 ms | 533.3 ms |
| BA (`dpvo_ba`, windowed) | 9.6 ms | 10.7 ms |
| **Total** (wall clock ÷ frames) | 3828.9 ms | 4475.0 ms |

Correlation assembly is **~8-10× more expensive than every other stage
combined** — by far the dominant cost, and the reason `default.yaml`
sizing (finding 3) is currently infeasible for a multi-hundred-frame CPU
run. It also grows with active-graph depth (more target-frame groups, more
edges per group), consistent with the full run's larger
`ms_per_frame_total` once `n_frames` had grown to 37 vs. the diagnostic's
10. This is squarely a "naive nested Rust loop vs. batched CUDA kernel"
gap (see finding 3), not an algorithmic issue with the port's math — M2's
own fixture-scale timings (0.509 ms at 5 edges) were accurate at *that*
scale; nothing before this milestone had measured `corr_cpu` at DPVO's
real few-thousand-edge working set.

### Deviations from `dpvo.py`

* **Graph sizing**: `config/fast.yaml`, not `default.yaml` (finding 3) —
  a deliberate, cited, CPU-feasibility substitution, analogous to the M4
  brief's own suggested "downscale input resolution if too slow" lever,
  applied to graph size instead (finding 1 already ruled out resolution
  downscaling as what upstream itself does).
* **No loop closure** (`LOOP_CLOSURE`/`CLASSIC_LOOP_CLOSURE`) and **no
  global-BA fallback** (`dpvo.py::__run_global_BA`) — both explicitly
  Milestone M6 scope; the fallback is additionally unreachable by
  construction in this port (`dpvo_patch_graph.rs`'s module doc proves
  this from `REMOVAL_WINDOW`/`PATCH_LIFETIME`'s relative sizes under both
  shipped configs).
* **No inactive-edge retention** (`remove_factors(..., store=True)`) —
  edges aging out of the window are discarded, not archived, since
  nothing in M4 reads them back (no loop closure/global BA consumer
  exists yet).
* **Windowed BA**: every `update_step` call passes only `[frame_lo, n)` to
  `dpvo_ba`, not the full live trajectory — a derived-safe bound
  (`frame_lo = n.saturating_sub(REMOVAL_WINDOW + PATCH_LIFETIME)`, checked
  by a `debug_assert` at every call site) documented in `dpvo_vo.rs`'s
  module doc, needed because M3's `dpvo_ba` sizes its dense solve off
  `poses.len()` — not present as an explicit mechanism upstream (the CUDA
  kernel takes the full buffer plus explicit `t0`/`n` bounds instead).
* **`torch.median`/`torch.quantile(0.5)`** (patch-depth init and
  `motion_probe`'s gate) are ported as two distinct hand-derived functions
  (`median_recent_depth`, `torch_quantile_50`) with different documented
  tie-breaking conventions, rather than a single "median" helper — a
  correctness detail easy to conflate (both are cited/tested separately).
* **Dense image undistortion** (`examples/euroc_dpvo_vo_demo.rs`'s
  `undistort_image`) is a new, from-scratch forward-mapping warp — this
  repo previously only had `RadialTangential::undistort_pixel` (single
  keypoints), insufficient for a dense CNN front end; not part of
  `dpvo.py` itself (which delegates to `cv2.undistort`) but necessary
  glue with no prior in-repo utility to reuse.

### Blockers / open items for M5 (IMU coupling)

1. **Correlation-assembly CPU cost (finding 3/4) is the dominant open
   performance question**, not IMU coupling itself. M5 should either (a)
   budget for `fast.yaml`-scale graphs only, (b) invest in vectorizing
   `corr_cpu`/`patchify_cpu` (removing bounds-checked per-element
   `ndarray` indexing in the hot loops, precomputing per-target-frame
   pyramid slices once instead of per-edge) before attempting
   `default.yaml` sizing, or (c) accept CPU-only operation stays
   research-scale, matching the plan doc's own risk register. Do not
   assume M4's `fast.yaml` numbers extrapolate linearly to `default.yaml`
   sizing — they do not, given the nested-loop cost model measured here.
2. **`DpvoOdometry`'s pose stream is visual-only monocular** — no metric
   scale anchor beyond the initial random/median depth init (the
   `ate_similarity_scale=1.359` above is the Umeyama-recovered scale
   factor needed to match ground truth, i.e. genuine monocular scale
   drift, not a bug). M5's IMU coupling is exactly the mechanism the plan
   doc's §4 hybrid design expects to supply metric scale, via
   `ImuPreintegrationFactor`/`vi_motion_initializer.rs` attached to the
   *same* windowed `DpvoBaProblem`/`dpvo_ba` this milestone assembles, as
   an additional residual family in one joint solve.
3. **No production (non-fixture) `SoftAgg` weight export path exists
   yet** — M4 reused M2's `softagg_weights_fixture.npz` as-is (real
   checkpoint-derived weights, just a fixture-shaped filename/location);
   M5+ should decide whether this stays the permanent artifact name or
   moves alongside the four ONNX graphs in a real deployment layout.
4. **`dpvo_patch_graph.rs`'s own unit tests use small synthetic configs**
   (4-patch/frame toy graphs) to keep the bookkeeping tests fast and
   readable; no unit test exercises the real `fast.yaml`/`default.yaml`
   -scale graph directly — the EuRoC run above is the only evidence this
   milestone has at real scale, for both correctness and performance.
5. **Accuracy vs. upstream is still entangled with two un-decomposed
   factors**: `fast.yaml` (smaller) graph sizing vs. `default.yaml`, and
   M1/M2's own carried-forward "not verified against upstream's CUDA
   kernel" caveat for `corr_cpu`/`patchify_cpu`. A future milestone with
   more CPU budget (or a GPU host) could re-run at `default.yaml` sizing
   to separate "sizing hurts accuracy" from "the port's own
   correlation/patchify math has a real discrepancy vs. CUDA".

## M4-perf results (2026-07-17)

M4's own numbers identified correlation-tensor assembly — `reproject_patch_grid`
(`pipelines/slam/src/dpvo_patch_ba.rs`) + `corr_cpu`
(`crates/vision/src/dpvo/correlation.rs`) + the per-edge 2-pyramid-level
assembly `corr_pyramid` (`pipelines/slam/src/dpvo_vo.rs`) — as costing
**~3.1 s/frame** at DPVO's real `fast.yaml` working set, 8-10× the combined
encoder+GRU+BA cost and the dominant reason the M4 EuRoC run took 1789.99 s
for 400 frames. This milestone's target: bring that stage under
~300 ms/frame (order-of-magnitude), with no numerics change beyond
floating-point reordering.

### Hotspot findings (profiling, not guessing)

A micro-benchmark at DPVO's real per-frame scale (3000 edges, 3×3 patch
grid, 2 pyramid levels, radius-3/49-tap lookup, 128 channels — added as
`#[ignore]`-gated `--release` timing tests, not a correctness check; see
"Verification" below for exact invocations) isolated the cost split the
task asked for:

* **`reproject_patch_grid` is negligible**: 0.242-0.249 ms for 3000 edges
  (`dpvo_vo::tests::correlation_assembly_perf_at_realistic_working_set`).
  Confirms the plan doc's own §3 estimate ("a handful of matrix ops per
  edge... negligible") — the entire ~3.1 s was inside `corr_cpu`/
  `corr_pyramid`, not the reprojection math.
* **`corr_cpu` itself was ~11× slower than necessary for a reason with no
  algorithmic cause**: the pre-existing implementation indexed both
  `patch_feats` `(num_edges, channels, patch, patch)` and `target_fmap`
  `(channels, height, width)` in their native **channel-first** layout.
  Reading a per-pixel 128-channel feature vector therefore walked a stride
  of `patch²` (anchor side) or `height·width` (target side) elements
  between channels — a near-guaranteed cache miss per channel per corner
  per tap, and a memory-access pattern the compiler cannot autovectorize
  into a SIMD dot product. Two smaller, compounding inefficiencies rode
  along: the four bilinear corner weights were recomputed from scratch on
  every one of the 49 taps even though the tap offsets are integers (so the
  fractional part — and therefore every weight — is identical across all
  49 taps of a given patch pixel, proven in `correlation.rs`'s module doc);
  and a full weighted-neighbourhood vector was materialized before the dot
  product instead of fusing the two passes into one.
* **Single-threaded**: `rayon` was already a hard dependency of
  `pipelines/slam` and a dev-dependency of this workspace's own root crate
  (checked directly — `grep -n '^name = "rayon' Cargo.lock`, and
  `pipelines/slam/Cargo.toml:34`) before this milestone touched anything,
  so parallelizing `corr_cpu`'s independent per-edge work was in scope from
  the start per the task's own rule ("only if rayon is already in the
  tree").
* **A second, distinct hotspot only visible at the real per-frame
  integration scale, not in a single-call micro-benchmark**: `dpvo_vo.rs`'s
  `FramePyramid` re-transposed the *same* target frame's feature-map
  pyramid from scratch on every `update_step`/`motion_probe` call that
  referenced it, for as long as that frame stayed inside the active window
  (up to `REMOVAL_WINDOW`/`PATCH_LIFETIME` ≈ 11-16 calls). This is pure
  redundant work orthogonal to the per-call `corr_cpu` speedup — caching
  the transpose once, at the point the pyramid is built, removed it
  entirely rather than just making each occurrence faster (see "Caching"
  below).

### Changes made (in the order applied, per the task's approach)

1. **Layout + fusion rewrite of `corr_cpu`** (`crates/vision/src/dpvo/correlation.rs`):
   both inputs are now transposed once per call into channel-last scratch
   buffers (`ChannelLastImage`, `ChannelLastPatches`) so every per-corner
   access in the tap loop is a contiguous `channels`-length slice; bilinear
   weights are computed once per `(edge, patch-pixel)` (`BilinearWeights`)
   and reused across all 49 taps instead of being recomputed per tap; the
   old two-pass "accumulate a weighted neighbourhood vector, then dot it"
   is fused into one pass per corner via linearity of the dot product
   (`anchor·Σwₖcₖ = Σwₖ(anchor·cₖ)`) — a floating-point *reordering*, not a
   different computation. The channel dot product itself
   (`dot_slice`) is a straight contiguous `f32` slice loop, left for the
   compiler to autovectorize (verified only indirectly, via the measured
   speedup below — no disassembly inspection was done). **`corr_cpu`'s
   public signature, documented semantics, and existing fixture-based
   parity test are all unchanged** — the fixture test now simply exercises
   the rewritten implementation directly.
2. **Parallelism**: the outer loop over edges in `corr_cpu` is parallelized
   with `rayon::slice::ParallelSlice::par_chunks_mut` over the output
   array's contiguous per-edge chunks (each edge's output slice is
   independent). `rayon` was added as an **optional** dependency of
   `visloc-vision`, gated behind (and only pulled in by) the existing
   `onnx-inference` feature — not a new crate in the dependency graph
   (`git diff Cargo.lock` shows one new dependency *edge*,
   `visloc-vision → rayon`, zero new `[[package]]` entries).
3. **Caching** (`pipelines/slam/src/dpvo_vo.rs`): added
   `corr_cpu_prebuilt_target` (`crates/vision/src/dpvo/correlation.rs`), a
   twin of `corr_cpu` that accepts an already-channel-last-transposed
   target image (`ChannelLastImage`, now `pub`) instead of transposing one
   internally. `FramePyramid` now stores both pyramid levels as
   `ChannelLastImage`, built exactly once when the pyramid is constructed
   (frame arrival), and `corr_pyramid` calls `corr_cpu_prebuilt_target`
   instead of `corr_cpu` — eliminating the redundant per-`update_step`
   re-transpose entirely (confirmed nothing outside this module ever read
   the channel-first form, so no duplicate storage was needed).
   Numerically inert by construction (same deterministic transpose of the
   same input, computed once instead of repeatedly) — confirmed by the
   real EuRoC run below producing **bit-identical** ATE with and without
   this step.
4. **Not needed**: hoisting `reproject_patch_grid`'s per-edge invariant
   (`pose_j.compose(&pose_i.inverse())` and the quaternion→rotation-matrix
   conversion, previously recomputed once per of the 9 grid pixels instead
   of once per edge — `pipelines/slam/src/dpvo_patch_ba.rs`'s
   `ReprojectionInvariant`/`reprojection_invariant`/`project_with_invariant`)
   was applied anyway, since profiling (above) had already shown this stage
   negligible; it is a free, purely-algebraic factoring (bit-identical
   output, checked by the pre-existing `transform_point_center_matches_
   reproject_patch_grid_center_entry`/`reproject_patch_grid_offsets_are_
   distinct_for_a_tilted_view` tests, both still passing unchanged) kept
   because it was correct and cost nothing, not because it moved the
   needle.

### Before/after timing table

Micro-benchmarks, `--release`, synthetic realistic-scale inputs (not the
tiny M1/M2 fixture shapes):

| Benchmark | Before | After | Speedup |
| --- | --- | --- | --- |
| `corr_cpu`, 3000 edges × 128 ch × 3×3 patch × 49 taps, 1 pyramid level (`crates/vision`'s `corr_cpu_perf_at_realistic_working_set`; "before" = `corr_cpu_reference`, the naive channel-first implementation kept `cfg(test)`-only for this exact comparison) | 846.291 ms/call | 77.014 ms/call | **10.99×** |
| Full correlation-assembly, 3000 edges, single target-frame group: `reproject_patch_grid` + `corr_pyramid` (2 levels) (`dpvo_vo`'s `correlation_assembly_perf_at_realistic_working_set`) | — (not measured pre-rewrite at this exact split) | reproject: 0.242 ms; `corr_pyramid`: 152.4 ms; total: 152.7 ms | — |

Real EuRoC run (`examples/euroc_dpvo_vo_demo`, MH_01_easy, 400 frames,
stride 2, `fast.yaml` sizing: `--patches-per-frame 48 --removal-window 16
--optimization-window 7 --patch-lifetime 11 --seed 0`, CPU-only, monocular —
identical invocation to M4's own reported run except `--out-dir`):

| Stage (ms/frame, averaged over all 400 frames) | M4 baseline | M4-perf, step 1 only (fast `corr_cpu`, no caching) | M4-perf, step 1+2 (fast `corr_cpu` + pyramid caching) |
| --- | --- | --- | --- |
| Image I/O | *(not separated)* | 3.59 | 3.66 |
| Undistortion | *(not separated)* | 12.53 | 12.52 |
| Encoder (`fnet`+`inet`) | 211.5 | 197.52 | 199.69 |
| **Correlation assembly** | **~3719.5** (`4475.0 − 211.5 − 533.3 − 10.7`, i.e. M4's own reported total minus its other three tracked stages — M4 never isolated this number directly, see its own finding 4) | **472.38** | **369.67** |
| GRU update (`update_iteration`) | 533.3 | 554.58 | 559.11 |
| BA (`dpvo_ba`) | 10.7 | 11.08 | 11.13 |
| **Total** | **4475.0** | **1294.46** | **1205.05** |
| Total wall clock, 400 frames | 1789.99 s | 517.79 s | 482.02 s |

**Correlation-assembly speedup: 10.06×** (3719.5 → 369.67 ms/frame) — clears
the order-of-magnitude target. The absolute number (369.67 ms/frame) sits
modestly above the task's illustrative "~300 ms" figure, not fully under
it; **total per-frame cost dropped 3.71×** (4475.0 → 1205.05 ms), and
correlation assembly is no longer the dominant single stage (the GRU update
call is now the largest at 559.11 ms/frame — a pre-existing ONNX Runtime
CPU-EP cost this milestone did not touch, out of scope per the task). The
gap between the isolated-microbenchmark 11× and the real-run 10.06× is
attributable to real per-frame overhead the microbenchmark doesn't capture
(assembling multiple smaller per-target-frame groups rather than one big
batch, `HashMap`-based grouping, per-group array allocation) — a plausible,
not further-decomposed, explanation.

### ATE before/after

| Metric | M4 baseline | M4-perf (both steps) |
| --- | --- | --- |
| `ate_rigid_rmse_m` | 0.1599 | 0.1546 |
| `ate_rigid_max_m` | 0.4075 | 0.4040 |
| `ate_similarity_rmse_m` | 0.1561 | 0.1519 |
| `ate_similarity_max_m` | 0.4079 | 0.4033 |
| `ate_similarity_scale` | 1.359171 | 1.265951 |
| `tracked_fraction` | 1.0000 (400/400) | 1.0000 (400/400) |

Not bitwise-identical, but within noise of the kind the task's own
constraint anticipated ("fp reordering... only"): `corr_cpu`'s fixture
parity test (below) still passes at the *same* max-abs-diff as M2's
original numbers (`2.515e-7`, unchanged to the last reported digit), and a
new realistic-shape equivalence test bounds the fast/reference disagreement
at ≤ 1e-5 per correlation-tensor entry. Over a 400-frame monocular VO
trajectory built from thousands of GRU-update/BA iterations consuming
these tensors, such per-call differences compound into a *different but
comparably-accurate* trajectory rather than a bit-identical one — expected
and explicitly budgeted for, not a regression signal. Confirming this is
purely a numerics-of-parallel-reordering effect and not a caching bug: the
step-1-only run and the step-1+2 run below produced **exactly** the same
ATE to every reported digit (`0.1546`/`0.4040`/`0.1519`/`0.4033`/
`1.265951`) — caching changes *when* a deterministic transpose is computed,
never its value, so it could not have moved these numbers, and it didn't.

### Verification

* `cargo test -p visloc-vision --features onnx-inference` (debug):
  **149 passed**, 0 failed, 1 ignored (up from M2's 148 — the one new
  addition is `fast_matches_naive_reference_at_realistic_shape`, the
  realistic-shape fast/reference equivalence test the task required; the
  perf micro-benchmark is `#[ignore]`d, a timing report not a pass/fail
  check).
* `ORT_DYLIB_PATH=E:/tools/onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll
  cargo test -p visloc-vision --features onnx-inference --test
  dpvo_onnx_parity -- --ignored`: **5 passed**, 0 failed —
  `correlation_parity_against_fixture`'s max abs diff is `2.515e-7`,
  byte-for-byte the same value M2 originally reported, confirming the
  rewritten `corr_cpu` is still exactly consistent with the M1/M2 fixture.
* `cargo test -p visloc-slam --features onnx-inference` (debug): **304
  lib tests** + 54+2(ignored)+6+6+132+10+9+4 across integration binaries,
  0 failed, 0 regressions (matches M4's own reported counts plus this
  milestone's new `#[ignore]`d perf test).
* `cargo clippy -p visloc-vision --all-targets --features onnx-inference --
  -D warnings`: **clean, zero warnings.**
* `cargo clippy -p visloc-slam --all-targets --features onnx-inference`:
  same 6/9 pre-existing warnings as M3/M4 (in `map_atlas.rs`,
  `online_slam_vi_ba.rs` ×2, `online_slam.rs`,
  `online_slam_motion_vi_init.rs`, `vi_motion_initializer.rs` — none in any
  file this milestone touched, confirmed by grepping the clippy output for
  `dpvo`).
* `cargo check --workspace --all-targets --features image-io,onnx-inference`:
  clean (implied by the successful `--release` example build below; not
  re-run as a separate bare invocation this milestone since the release
  build already exercises the same feature set at a stricter optimization
  level).
* `git diff --stat Cargo.lock`: 1 line changed (`+"rayon",` inside
  `visloc-vision`'s existing `dependencies` array) — **zero new
  `[[package]]` entries**, confirming no new crate entered the dependency
  graph.

### Files changed

* `crates/vision/Cargo.toml` — `rayon` added as an optional dependency,
  gated into the existing `onnx-inference` feature.
* `crates/vision/src/dpvo/correlation.rs` — `corr_cpu` rewritten
  (channel-last transpose, hoisted bilinear weights, fused corner-dot,
  `rayon`-parallel outer loop); new `corr_cpu_prebuilt_target` (now-`pub`
  `ChannelLastImage`, plus private `ChannelLastPatches`/`BilinearWeights`/
  `dot_slice`); `corr_cpu_reference` (the pre-rewrite naive implementation,
  kept `cfg(test)`-only) plus a new realistic-shape equivalence test and an
  `#[ignore]`d perf micro-benchmark.
* `pipelines/slam/src/dpvo_patch_ba.rs` — `ReprojectionInvariant`/
  `reprojection_invariant`/`project_with_invariant` extracted from
  `project_no_jacobian`; `reproject_patch_grid` now computes the invariant
  once per edge instead of once per of its 9 grid pixels. Pure factoring,
  no numeric change (existing tests unchanged and passing).
* `pipelines/slam/src/dpvo_vo.rs` — `FramePyramid` now stores
  `ChannelLastImage` (built once at pyramid construction) instead of raw
  channel-first `Array3<f32>`; `corr_pyramid` takes `&ChannelLastImage` and
  calls `corr_cpu_prebuilt_target`; both call sites (`motion_probe`,
  `update_step`) and the existing `corr_pyramid_shape_matches_corr_dim`
  test updated accordingly; new `#[ignore]`d
  `correlation_assembly_perf_at_realistic_working_set` perf micro-benchmark.
* `docs/dpvo_droid_port_plan.md` — this section.
* Not touched: `crates/vision/src/dpvo/patchify.rs` (profiling showed
  `patchify_cpu` was never the bottleneck — M2's own numbers already had it
  at 0.044 ms/call for 5 patches, and nothing in this milestone's
  measurements implicated it at scale), `dpvo_patch_graph.rs`,
  `sparse_factor_graph.rs`, `bundle.rs` (BA was already sub-12 ms/frame,
  untouched, unaffected).

## M5 results (2026-07-17)

Milestone M5: couple the workspace's own battle-tested IMU preintegration
stack (`imu_preintegration.rs`, `vi_motion_initializer.rs`'s gyro-bias/
gravity-alignment chain) into the M3/M4 DPVO sliding window, to fix the
monocular scale drift M4-perf's own EuRoC run measured (similarity scale
1.266, ground truth 1.0). The joint-solve math (the milestone's stated
"trickiest part") is implemented and rigorously validated by synthetic
tests; the bootstrap chain that seeds it from a real EuRoC run is not yet
reliable, and this section reports that honestly rather than papering over
it — see "Honest negative: the bootstrap chain" below.

### Files changed

* `pipelines/slam/src/dpvo_vi_ba.rs` (**new**) — the IMU-coupled joint
  Gauss-Newton solve: `DpvoImuFactor`, `DpvoViWindow`, `DpvoViBaSolution`,
  `dpvo_vi_ba_step`/`dpvo_vi_ba`, and the private `imu_factor_jacobians`
  helper implementing the left-perturbation IMU Jacobian derivation (see
  below). A sibling of `dpvo_patch_ba.rs`, not an edit to it, for the same
  reason M3 gave for not merging into `bundle.rs`: composing new state
  alongside a tested, upstream-parity-verified solver is lower-risk than
  mutating it. Re-derives (does not call into) `dpvo_ba_step`'s own visual
  normal-equation assembly — a deliberate, tested duplication (see the
  module doc's "Visual assembly" section and
  `tests::zero_imu_factors_matches_visual_only_solve`, which cross-checks
  the duplication against the trusted original to <1e-9).
* `pipelines/slam/src/dpvo_patch_ba.rs` — **pure visibility change, no
  logic edit**: `transform_edge`, `EdgeGeometry` (+ its fields), `DISP_MIN`/
  `DISP_MAX`, `POSE_DIAG_LM` widened from private to `pub(crate)` so
  `dpvo_vi_ba.rs` can reuse them. Confirmed inert:
  `ba_fixture_one_iteration_matches_reference_within_1e_4`/
  `ba_fixture_two_iterations_matches_reference_within_1e_4` (the M3 upstream
  parity tests) still pass at the exact same max-abs-diff as M3's own
  report (1.086e-6 / 1.007e-6).
* `pipelines/slam/src/dpvo_vo.rs` — `DpvoOdometryConfig::imu: Option<DpvoImuConfig>`
  (`None` by default — every M4 call site keeps compiling/behaving
  unchanged); `DpvoImuConfig`, `DpvoImuDiagnostics`; `DpvoOdometry::push_imu`/
  `imu_diagnostics`; private `integrate_imu_for_new_frame` (banks
  `ImuPreintegrator` deltas between consecutive committed frames, keyed by
  stable `arrival_index` pairs), `try_imu_bootstrap` (the
  `estimate_gyro_bias` → `estimate_gravity_and_velocities` chain, gated on
  `gravity_norm_deviation_ratio`, run at most once), `prune_stale_imu_deltas`;
  `update_step` now branches to `dpvo_vi_ba` once bootstrapped, `dpvo_ba`
  (unchanged) otherwise. `DpvoOdometryConfig` lost its `Copy` derive
  (`SE3` isn't `Copy`) — checked non-breaking, every use was a `&self` field
  read.
* `pipelines/slam/src/lib.rs` — registered `dpvo_vi_ba`, re-exported its
  public items.
* `examples/euroc_dpvo_vo_demo.rs` — `--imu` (feeds `mav0/imu0/data.csv`,
  interleaved by timestamp with camera frames), `--imu-gravity-norm-deviation-ratio`,
  `--imu-min-bootstrap-factors`, `--imu-noise-scale` (a diagnostic knob added
  while investigating the bootstrap finding below); `se3_from_t_bs` (a
  verbatim copy of `examples/euroc_online_slam_vi_demo.rs`'s own helper,
  building `DpvoImuConfig::body_to_camera` from `cam0/sensor.yaml`'s `T_BS`);
  summary now echoes `imu_enabled`/`imu_bootstrapped`/`imu_gravity_world_*`/
  `imu_bias_*` alongside the usual ATE/scale fields.

### The jacobian-convention conversion (the hard math), summarized

`bundle.rs`'s existing Forster IMU factor derives residual/Jacobians under
`BundleAdjustment`'s **right**-perturbation convention (`T ← T·Exp(ξ)`) for
the body pose `body = imu_body_to_camera ∘ camera_pose`. DPVO's own poses
retract on the **left** (`T ← Exp(ξ)·T`, `dpvo_patch_ba.rs`'s own
convention). `dpvo_vi_ba.rs` needs `bundle.rs`'s exact residual/Jacobian
*content* (reused, not re-derived — re-deriving Forster's formulas a second
time would just be a second, independently-fallible copy) reinterpreted for
DPVO's camera pose's left perturbation. Two conjugations, both grounded in
the single Adjoint definition already doc-commented on
`visloc_core::geometry::SE3::adjoint` (`T·Exp(ξ)·T⁻¹ = Exp(Ad(T)·ξ)`):

1. **Right-perturbation of `body` → left-perturbation of `body`**:
   `J_left = J_right · Ad(T⁻¹)` (from `Exp(ξ)·T = T·Exp(Ad(T⁻¹)·ξ)`, itself
   the Adjoint definition applied to `T⁻¹`).
2. **Left-perturbation of `body` → left-perturbation of DPVO's camera pose**:
   with `C = imu_body_to_camera`, `body = C ∘ P`, a DPVO left perturbation of
   `P` induces `body_new = Exp(Ad(C)·ξ_P) ∘ body` (the Adjoint's own
   definition, conjugating `Exp(ξ_P)` by `C`) — so `J_P = J_left(body) ·
   Ad(C)`.

Combined: `J_P_i = J_right(body_i) · Ad(body_i⁻¹) · Ad(imu_body_to_camera)`,
symmetrically for `J_P_j`. Velocity Jacobians need no conversion (Euclidean,
convention-independent). A second, independent subtlety (the "sign
convention" section of the module doc): `dpvo_ba_step`'s visual accumulation
secretly relies on its residual's `target − predicted` sign trick to turn a
standard Gauss-Newton `-Jᵀr` into a `+Jᵀr` with no explicit minus anywhere;
the IMU residual has no such trick (it is a literal `∂r/∂ξ`), so combining
the two families in one shared `(B, v)` system needs the IMU block's RHS
contribution to carry the minus sign explicitly (`v += −Jᵀr`) while the
visual blocks keep their existing `+Jᵀr`.

**Verification**: `dpvo_vi_ba::tests::imu_factor_jacobian_matches_numeric_finite_difference`
checks every column of `J_pose_i`/`J_pose_j`/`J_vel_i`/`J_vel_j` (non-trivial
rotation+translation poses, non-identity `body_to_camera`) against a central
finite difference of the residual evaluated at the *actual*
`SE3::exp(&xi).compose(&pose)` retraction this crate uses elsewhere — max
error < 1e-5 (central-difference floor at `eps=1e-6`), for all 18 columns
(6+6 pose, 3+3 velocity). `tests::pure_imu_window_matches_textbook_gauss_newton_sign`
independently checks the sign convention (a pure-IMU window's residual norm
must *decrease* after one `dpvo_vi_ba_step` — confirmed). Both pass.

### Synthetic scale-recovery result

`tests::synthetic_window_recovers_metric_scale_within_two_percent`: a
4-frame window, 5 patches anchored in frame 0, starting-guess poses/inverse
depths scaled **1.4×** from a true constant-*acceleration* trajectory
(`a = (2, 0, 0) m/s²` — a first attempt at this test used constant
*velocity* and found it information-theoretically degenerate for scale
recovery, since `∂r/∂ξ` are then satisfied by *any* joint `(p, v)` rescale;
documented in the module doc's test comment as a real finding, not a detail),
with visual targets computed by literally reprojecting the wrong-scale
starting guess (an exact visual fixed point, since mono reprojection is
scale-invariant) and IMU factors built from the *true* metric kinematics.
After 30 `dpvo_vi_ba` iterations: **recovered scale 1.000002** against a
true scale of 1.0 (task's own bar was 2%; the actual result is four orders
of magnitude tighter). This is the joint solve's own correctness validated
in isolation from any bootstrap-quality question — see next section for why
that distinction matters.

Additional tests, all passing: `tests::zero_imu_factors_matches_visual_only_solve`
(this module's independently-derived visual assembly matches
`dpvo_ba_step`'s trusted original to <1e-9 given zero IMU factors) and the
sign-convention/jacobian tests above.

### EuRoC MH_01 run — honest negative: the bootstrap chain

`--euroc-dir MH_01_easy --max-frames 400 --stride 2 --seed 0`,
`fast.yaml`-equivalent graph sizing (`--patches-per-frame 48
--removal-window 16 --optimization-window 7 --patch-lifetime 11`, identical
to M4-perf's own reported run except `--imu`), CPU-only:

| Metric | M4-perf baseline (no IMU) | M5 (`--imu`) |
| --- | --- | --- |
| `ate_rigid_rmse_m` | 0.1546 | 24.4719 |
| `ate_rigid_max_m` | 0.4040 | 39.5517 |
| `ate_similarity_rmse_m` | 0.1519 | 0.1407 |
| `ate_similarity_max_m` | 0.4033 | 0.3500 |
| `ate_similarity_scale` | 1.265951 | **0.006035** |
| `tracked_fraction` | 1.0000 (400/400) | 1.0000 (400/400) |
| `total_elapsed_s` | 482.02 | 809.46 |

**This does not meet the acceptance target** (scale within `1.0±0.05`,
rigid ATE at or below similarity ATE) — reported per the task's own explicit
instruction to report honestly regardless. No crash, no NaN pose, no tracking
loss (100% tracked, matching the visual-only baseline); the *similarity*-
aligned ATE (which factors out scale) is actually slightly better than the
M4-perf baseline (0.1407 vs 0.1519 m), but the *rigid* ATE and recovered
scale show the joint solve converged to a badly wrong absolute scale rather
than the target metric one — the opposite of a graceful "imprecise but
metric-ish" result.

**Diagnosis** (four experiments, not a guess):

1. First attempt (`min_bootstrap_factors=3`, the initial default): bootstrap
   fired at frame ≈11, against a visual reconstruction with almost no time
   to stabilize. Scale collapsed to 0.0057.
2. Investigating why raising the threshold didn't obviously help surfaced a
   **genuine, separate bug**: the bootstrap was built from the graph's
   *current live* window, but `DpvoPatchGraph::keyframe`'s motion-magnitude
   folding (MH_01's opening seconds are close to stationary) discards
   low-motion frames faster than 10 usable factors (the estimators' own
   `MAX_ALIGNMENT_WINDOW`) could accumulate against a live-only view — at
   `min_bootstrap_factors=10` the bootstrap simply never fired at all within
   80 frames. Fixed by decoupling bootstrap evidence into its own
   `imu_bootstrap_history` (pose snapshots taken at bank time, independent
   of the BA window's own churn) — a real correctness fix, kept regardless
   of the remaining finding below.
3. With that fixed and `min_bootstrap_factors=10`: bootstrap now reliably
   fires (frame ≈11 again, since 10 factors accumulate quickly once not
   discarded), but the recovered gyro bias (`(-0.081, -0.182, 0.077)` rad/s)
   is implausibly large for a real MEMS gyro (EuRoC's own bias is normally
   `~1e-3` to `~1e-2` rad/s) and the joint solve still collapses scale to
   0.008. Raising the threshold further to 40 improved one axis
   (`bias_gyro_x` → `-0.006`, reasonable) but not the other two, and the
   collapse persisted.
4. To rule out "the joint solve is simply over-trusting a tight IMU
   covariance": reran with `--imu-noise-scale 50` (50× looser covariance-
   derived confidence). The bootstrap's own recovered gravity/bias was
   **byte-identical** to experiment 3 (expected — noise scale only affects
   the ongoing solve's weighting, not the bootstrap's own linear solve), and
   the collapse was, if anything, slightly worse (scale 0.0033). This rules
   out over-confident weighting as the cause: a systematically wrong
   absolute velocity/gravity model pulls scale toward a bad fixed point
   regardless of per-factor confidence, once every window's factors agree
   (wrongly) with each other over hundreds of frames.

**Root cause**: `estimate_gyro_bias`/`estimate_gravity_and_velocities`
(`vi_motion_initializer.rs`) were designed for, and everywhere else in this
codebase are run against, an **already metric, reasonably converged** set of
visual poses (stereo, or a monocular pipeline already anchored by a prior
successful VI init). DPVO's own reconstruction, at whatever point in a real
run enough IMU factors first accumulate, is still in its own uncalibrated,
non-metric scale and — during MH_01's slow, low-parallax opening seconds —
comparatively noisy rotation regime: a precondition mismatch these two
otherwise-correct, faithfully-reused functions were never built to tolerate.
Because M5's own design fixes the recovered bias/gravity forever after one
bootstrap attempt (the task's own staged-bias instruction), a single
bad-quality bootstrap poisons the rest of the run — there is no subsequent
opportunity for it to self-correct.

**A secondary, measured consequence**: the corrupted joint solve also
defeats `DpvoPatchGraph::keyframe`'s motion-magnitude-based folding — erratic
poses read as "high relative motion", keeping far more frames live than
intended (`frames_graph_n` reached **92** by frame ~371-400 with `--imu`,
versus **10-15** in a freshly-run 150-frame visual-only check on the same
sequence/config, consistent with M4's own report of `n: 1→37` over the full
400-frame no-IMU baseline). This is very likely why the `--imu` run's
`total_elapsed_s` (809.46 s) is ~1.7× the M4-perf baseline (482.02 s) — the
joint-solve math itself stays cheap (`ms_per_frame_ba` 13.85 vs 11.13,
barely different); the extra cost is the correlation/update stages doing
more work over a larger, wrongly-retained active window. Expected to
resolve on its own once the bootstrap-quality issue above is fixed, not a
separate mechanism to build.

### Visual-only path confirmed unchanged

A 150-frame run of the **same** command with `--imu` omitted:
`imu_enabled=false`, `imu_bootstrapped=false` throughout,
`tracked_fraction=1.0000`, `ate_similarity_scale=0.473211` — a plausible,
non-degenerate monocular scale-drift value (not directly comparable to the
400-frame M4-perf number; a different prefix length), confirming
`update_step`'s `else` branch (`dpvo_ba`, byte-for-byte the M4 call) still
runs and produces sane, non-collapsed output when `config.imu = None`. This
is also guaranteed structurally: every M5 addition to `dpvo_vo.rs` either
guards on `self.config.imu.clone()` returning `None` (early return, no
state ever mutated) or on `self.imu_bootstrapped` (which can only become
`true` inside `try_imu_bootstrap`, itself gated on `config.imu.is_some()`).

### Verification (verbatim)

* `ORT_DYLIB_PATH=E:/tools/onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll
  cargo test -p visloc-slam --features onnx-inference`: **308 lib tests**
  passed (up from M4-perf's 304 — the 4 new `dpvo_vi_ba` tests), 0 failed, 7
  ignored (perf/benchmark tests, unaffected by this milestone plus the new
  `dpvo_vi_ba` tests, none `#[ignore]`d); every integration test binary
  green, 0 failed anywhere: 54 passed + 1 ignored, 0 passed + 2 ignored
  (`dpvo_patch_ba_fixture.rs` — both its tests are `--ignored`-gated fixture
  tests, covered by the dedicated `--ignored` run below), 6/6, 6/6, 132/132,
  10/10, 9/9, 4/4 — unchanged counts from M4-perf plus the ignored-perf-test
  bump already noted.
* `ORT_DYLIB_PATH=... cargo test -p visloc-slam --features onnx-inference --
  --ignored`: **10 ignored tests passed, 0 failed** across all binaries —
  the lib's own 7 (`block_cholesky`/`incremental_pose_graph` benches plus
  `dpvo_vo::tests::correlation_assembly_perf_at_realistic_working_set`),
  `bundle_adjustment.rs`'s 1 (`bench_ba_sparse_solver`), and
  `dpvo_patch_ba_fixture.rs`'s 2 (`ba_fixture_one_iteration_matches_reference_within_1e_4`/
  `ba_fixture_two_iterations_matches_reference_within_1e_4`, the M3 upstream
  parity tests, unchanged max-abs-diff from M3's own report — confirming
  the `pub(crate)` visibility change above is inert).
* `cargo test -p visloc-vision --features onnx-inference`: **149 passed**,
  0 failed, 1 ignored — unaffected by this milestone (no `visloc-vision`
  file touched).
* `ORT_DYLIB_PATH=... cargo test -p visloc-vision --features onnx-inference
  --test dpvo_onnx_parity -- --ignored`: **5 passed**, 0 failed — same
  numbers as M2/M4-perf.
* `cargo check --workspace --all-targets --features image-io,onnx-inference`:
  clean.
* `cargo clippy -p visloc-slam --all-targets --features onnx-inference`:
  same 6 pre-existing warnings as M3/M4/M4-perf (`map_atlas.rs` ×3,
  `online_slam_vi_ba.rs` ×2, `vi_motion_initializer.rs`,
  `online_slam_motion_vi_init.rs`, `online_slam.rs` ×2 — confirmed by
  grepping clippy's output for `dpvo`: zero hits); **zero** warnings in
  `dpvo_vi_ba.rs`, `dpvo_vo.rs`, or `dpvo_patch_ba.rs`.
* `cargo clippy --example euroc_dpvo_vo_demo --features image-io,onnx-inference`:
  clean, zero warnings.
* `cargo build --release --example euroc_dpvo_vo_demo --features
  image-io,onnx-inference`: succeeded.
* `git diff --stat` on touched files: `dpvo_patch_ba.rs` (+25/-10, pure
  visibility, confirmed above); `dpvo_vi_ba.rs` new (~1080 lines incl.
  tests); `dpvo_vo.rs`, `lib.rs`, `examples/euroc_dpvo_vo_demo.rs` — all
  additive, no removed visual-only behavior.

### Blockers / open items for M6

1. **The bootstrap chain, not the joint solve, is the blocker.** The
   IMU-coupled Gauss-Newton math (Jacobian conversion, sign convention,
   Schur elimination with the augmented velocity block, implicit scale
   absorption) is validated by four passing synthetic/unit tests, including
   exact-to-6-decimal scale recovery from a deliberately wrong start. Do not
   rework `dpvo_vi_ba.rs`'s core math without new evidence it is wrong;
   rework the bootstrap instead.
2. **A genuinely monocular-aware VI bootstrap is needed.** Either (a) add
   an explicit scale unknown `s` to a gravity/velocity/scale joint linear
   solve (VINS-Mono/ORB-SLAM3-style monocular VI initialization) rather than
   reusing `estimate_gravity_and_velocities` as-is (which assumes metric
   input), or (b) gate promotion on more than a bare gravity-norm ratio —
   e.g. cross-validate the bootstrap's own resulting IMU residual, or
   require a minimum rotation/translation excitation — and (c) allow a
   bounded number of re-attempts / a rollback if the joint solve's own
   behavior (e.g. trending scale) looks pathological after bootstrapping,
   rather than fixing bias/gravity forever on the first attempt that merely
   clears the norm-ratio gate.
3. **`imu_bias_accel` is never estimated** (fixed at `0` for the whole run,
   a documented M5 narrowing) — the noise-scale experiment (finding 4 above)
   suggests this is not the primary cause of the observed collapse, but a
   real nonzero EuRoC accelerometer bias folds unmodeled into every
   gravity/velocity/position residual and should be revisited once the
   bootstrap's core reliability is fixed.
4. **Secondary performance regression** (`frames_graph_n` growing to ~92 vs
   ~37, ~1.7× wall clock) is very likely a symptom of finding 2, not a
   separate mechanism — re-measure once the bootstrap is fixed before
   assuming it needs its own work.
5. **`Ad(imu_body_to_camera)` is recomputed per IMU factor**, not hoisted
   once per window solve — a cost-negligible (single `6×6` product per
   factor, dwarfed by every other per-frame cost measured in this port),
   documented, trivial follow-up.
6. **No loop closure, no global-BA fallback** — unchanged from M4's own
   documented scope; M6 remains the milestone for DPV-SLAM-style proximity
   loop closure per the plan doc's original schedule.
