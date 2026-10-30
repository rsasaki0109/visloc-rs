# Next Development Plan (as of 2026-10-08)

This is the short answer to "what next". Current state and the list of
closed threads are in [`PLAN.md`](../PLAN.md). The previous version of this
plan (2026-07-02) is archived at
[`archive/next_development_plan_2026-07.md`](archive/next_development_plan_2026-07.md).
All of its phases have since landed or been closed.

Scope rule for this round: **no new GPU work.** GPU paths (wgpu SfM, 3DGS,
DPVO CUDA) stay as they are; nothing here depends on them.

**Update 2026-10-29: GPU work resumes on a local laptop.** The laptop has an
8 GB GPU and runs WSL2, where the GPU is reachable through CUDA but not
through wgpu (no NVIDIA Vulkan driver). New GPU work is therefore an opt-in
CUDA backend that follows the DPVO pattern
([`gsplat_cuda.md`](gsplat_cuda.md)); wgpu stays the main implementation.
The planned order:

1. CUDA 3DGS rasterizer: forward and backward, checked against the CPU
   oracle. Done.
2. CUDA trainer on top of it. Measure south-building PSNR and time against
   the wgpu trainer's recorded numbers. Done: 30k steps in 691 s, held-out
   PSNR 23.22 dB. The README's wgpu run reached 22.77 dB with a different
   SfM and the depth-normal loss. The first version took 978 s; sub-tile
   masks and direct global gradient adds cut a step from 30.9 to 21 GPU ms.
   The remaining large stages are the dense Adam (opt-in sparse SH Adam
   halves it), raster_bwd and the sort.
3. Score `gsplat_mesh`'s mesh on ETH3D, which gives 3DGS geometry a
   baseline (visloc-mvs's mesh: F1 79.3% at 2 cm). Then move TSDF
   integration to CUDA: it is 28 of the 34 mesh minutes on SmallCity.
   Baseline done: on terrains, playground and office the 3DGS mesh scores
   a mean F1 of 33.1% at 2 cm, against 80.0% for visloc-mvs's clouds
   ([`gsplat_cuda.md`](gsplat_cuda.md#geometry-on-eth3d)). visloc-mvs
   depth supervision (`--depth-prior`, weight 0.1) raises it to 35.0%:
   accuracy rises, completeness does not
   ([`gsplat_cuda.md`](gsplat_cuda.md#depth-prior-from-visloc-mvs)). The
   depth-normal loss (CUDA, weight 0.005) does not help: 32.0% alone, 34.0%
   with the prior ([`gsplat_cuda.md`](gsplat_cuda.md#depth-normal-loss)).
   Meshing the expected depth instead, which those losses train, with the
   prior at weight 30 reaches 63.6% (the prior's MVS clouds: 67.6%), at the
   price of the image fit
   ([`gsplat_cuda.md`](gsplat_cuda.md#meshing-the-depth-the-losses-train)).
   TSDF on the GPU done for the CUDA meshes: the same F1 in 4–5x less time
   and 1/6 of the memory, and playground at 1 cm in 4.2 GB, where the CPU
   TSDF ran out of 11 GB ([`gsplat_cuda.md`](gsplat_cuda.md#tsdf-on-the-gpu)).
4. The rasterizer optimizations from Kopanas's
   [3DGS CUDA write-up](https://grgkopanas.github.io/3dgs_cuda_blog/), one at
   a time. Stopped at about 21 ms per step: the two-stage sort, a vectorized
   Adam and dropping the gradient memset each gained under 1%. The dense
   Adam is bound by memory, and raster_bwd needs steps 5 and 6
   ([`gsplat_cuda.md`](gsplat_cuda.md#tried-and-not-adopted)).
5. Two-view verification on the GPU: 450 s of SmallCity's 600 s front end.
   On the WSL2 laptop the CPU matcher was the bottleneck instead (no wgpu):
   on south-building it took 780 s, and verification 578 s only because the
   two shared the cores. Matching on CUDA (`gsplat_photos_cuda`, the CPU's
   matches bit for bit) brings the front end to 18.4 s and the whole SfM
   from 945 s to 148 s (the same model), with verification at 5.3 s on the
   CPU, so verification stays there
   ([`gsplat_cuda.md`](gsplat_cuda.md#matching-on-the-gpu)).

## Track 1: Correctness and release hygiene

| Item | Status | Gate |
| --- | --- | --- |
| 0.2.1 patch release (distortion fixes) | Done: version bumped, CHANGELOG cut | Tag `v0.2.1` after main CI is green |
| 0.3.0 release | Done: every crate bumped to 0.3.0, CHANGELOG cut (highlights; the itemized log in [`archive/changelog_0.3.0_full.md`](archive/changelog_0.3.0_full.md)) | Tag `v0.3.0` after main CI is green; publish (crates.io / PyPI tokens) |
| 0.3.1 patch release | Done: every crate bumped to 0.3.1, CHANGELOG cut (`--help` on the release binaries, the intrinsics-refinement damping, the release workflow) | Tag `v0.3.1` after main CI is green; the `release` workflow then builds the binaries |
| `gsplat_photos` intrinsics refinement: damping | Done: the joint BA starts at λ = 1e4 (see CHANGELOG). Tried and not used: damping scaled by the Hessian's diagonal (Marquardt, λ₀ = 1e-4: 0–499 subset 10.0 cm median but 23 cameras over 1 m), λ ÷ 3 after an accepted step, 10 iterations instead of 20 | — |
| SmallCity 0–499: mapper initial pair across a revisit | Closed, not changed. The COLMAP-port mapper (as COLMAP) starts from images 66 and 324, 258 photos apart, because consecutive photos fail the 16° initial-pair angle; its model leaves image 66 32 m off and 319–352 bent by up to 29 m, which the intrinsics refinement (λ₀ = 1e4) pulls back to a 12.1 cm median, 23 cm max. Starting from a pair at most 30 photos apart (a prototype `init_max_image_gap`) avoids the bend but ends at 17.8 cm median, 71 cm max on that subset (23.4 / 8.1 cm against 24.9 / 7.9 cm on the other two): the revisit pair also closes the loop. An 8° angle behaves the same way (18.5 cm) | — |
| SmallCity 500–999: 24.9 cm median after `gsplat_photos` | Diagnosed, not changed. Locally consistent (3–5 cm after a Sim(3) refit per 100 photos), but the scale varies by ±2% with the period of the route around the block (±1% on 0–499, ±0.5% on 1000–1499), so the loop is warped. The self-calibrated focal length is 485.4 px against the release model's 486.97 (0–499: 488.8, 1000–1499: 484.6), and that subset's fx and fy end 0.9% apart (release: 0.02%). Holding the release model's intrinsics fixed gives 14.5 cm median and a ±1.3% warp, so the focal length explains about half of it. A shared focal length (`BaConfig::shared_focal`) leaves 500–999 at 24.5 cm and 1000–1499 at 8.1 cm but brings back 28 cameras over 1 m on 0–499, so it is not used | Better-conditioned focal length on street captures (e.g. a prior from EXIF when it is trusted), checked on all three subsets and ETH3D |
| crates.io readiness | Done: all 16 crates package and verify-build from their packaged sources; root package trimmed from ~83 MB to 0.87 MiB compressed with an `include` allowlist | `cargo publish --workspace` (needs a crates.io token) |
| BA distortion models | Done: pose/structure BA linearises every lens model (OPENCV, FULL_OPENCV, the fisheye models, FOV, Double Sphere); opt-in p1/p2 self-calibration; rig observations use the sensor's lens too (2026-10-22). Open: matrix-free / GPU backends fall back for distorted cameras | `rig_sfm` on a distorted rig re-measured (OpenLORIS) |

## Track 2: Sensor coverage (the empty cells in the README sensor table)

| Item | Status | Gate |
| --- | --- | --- |
| Monocular + IMU VIO on the Basalt port | Done (experimental): `basalt_euroc_vio_demo --mono`; synthetic scale error < 0.3%, 5–9 mm RMS; stereo path unchanged ([notes](mono_inertial_vio.md)) | EuRoC ATE (needs dataset access); static-start drift; mono in the online mapper |
| GNSS fusion | Done (synthetic): joint VO + GNSS pose graph with estimated alignment, lever arm, GNC outlier rejection, dropout bridging; ATE 5.25 m → 0.90 m ([notes](gnss_fusion.md)) | Real-data run (e.g. KITTI raw OXTS); geodetic-to-ENU conversion; IMU in the graph |
| Metric scale of a 360-camera model from its IMU (arXiv 2610.35554) | Done (Python script, 2026-10-28): `scripts/insta360_imu_scale.py` reproduces the paper's per-run alignment on OverMaps360_50 (Insta360 X5). 50 scenes against ARKit: −0.21% mean, 1.91% MAE, 2.49% SD (paper: 0.997 ratio, SD 2.3%); the joint solve 3.89% SD. The dataset's IMU CSV is raw counts. The paper's estimate reads about 1 point low: the models' camera centres carry a rolling-shutter error proportional to the angular velocity, which also makes the scale grow with the frame spacing; fitted, +0.73% mean, 1.86% MAE. Against the phone's GPS log (49 scenes) the corrected IMU is the closest estimate (MAE 1.34%; ARKit 2.25%); the +3–4% on scenes with long phone captures is ARKit reading low there ([notes](insta360_imu_scale.md)) Rust port: `visloc_slam::estimate_imu_scale` and `gsplat_photos --imu` (CSV or `.insv`) output metric models; on 136 OverMaps360 frames the metric model agrees with the dataset's IMU scale within 0.01%. Raw video in: `gsplat_photos --insv` takes the `.insv`'s two lens streams (one ffmpeg command) with the factory lens calibration from its metadata ([lens notes](insta360_lens.md)), one rig per frame with the back lens refined; on 200 frames each of five OverMaps360 walks SfM registers 99.4–100% of the faces and the metric scale lands within 4% of the dataset's IMU scale outdoors, with 640 or 1024 px lens faces alike; in an indoor mall 640 px faces give a weak gravity fit and 12% error, 1024 px faces 1.2%; the estimator fits the gyro's bias (1.8°/s on one walk's camera); `gsplat_photos --insv VID.insv --out run` cuts the frames itself with ffmpeg, and writes the lens faces to disk as it cuts them (200 frames with 1024 px faces in 8.6 GB) | Stitch the lenses (needs the recorded lens-to-lens angles decoded); other Insta360 models' calibration layouts |

## Track 3: Usability

| Item | Status | Gate |
| --- | --- | --- |
| Python bindings (pyo3 + maturin) | Done: `bindings/python` (Camera, Pose, COLMAP I/O, PnP localization, ATE/RPE), 42 pytest tests, CI job | Publish wheels to PyPI |
| ROS 2 nodes | Done: `ros2/visloc-ros2` VIO and localization nodes over pure-Rust DDS, tested end to end over RTPS, CI job | Test against a real ROS 2 install (Fast DDS / Cyclone); attach the online mapper to the VIO node |

## Track 4: Maintainability

| Item | Status | Gate |
| --- | --- | --- |
| Planning docs | Done: `PLAN.md` (233 KB) and the itemized 0.2.0 CHANGELOG (~4,800 lines) moved to [`archive/`](archive/); short handoff and changelog in their place | — |
| Split very large source files | Done: `incremental_sfm`, `bundle`, `unordered_sfm_demo` and `vio/aom` are now directories of submodules (pure moves; test names and counts unchanged, public API unchanged) | Remaining large files are single test modules (aom `tests.rs` 11k) whose test paths are used as filters; next candidates `rig_sfm.rs` (10.7k), `euroc_online_slam_vi_image_demo.rs` (10.2k), `integrate_rig_atlas_landmarks.rs` (9.6k) |

## Needs real data (not runnable from a dataset-less environment)

These stay open, but each needs EuRoC / OpenLORIS access to measure:

- **Real-time margin.** The slowest EuRoC sequence runs at RTF 1.06×. LM
  inner iterations are about 80% of VIO wall time; any speed-up must be
  checked against the VIO trajectory hash.
- **OpenLORIS 10k rig SfM RMSE parity.** 0.3890 m vs COLMAP's 0.3843 m.
- **Mono-inertial VIO ATE on EuRoC** (the code is in; only the measurement is missing).
- **GNSS fusion on real data** (for example KITTI raw OXTS as the GNSS source).

## On hold: GPU work (ETH3D MVS)

Recorded 2026-10-22. The ETH3D MVS and end-to-end work that followed this
plan ran on rented GPU pods; it is paused until the next round. Nothing below
is started.

**Where it stands** (ETH3D high-res training, 13 scenes, F1 at 2 cm; details
in [`mvs.md`](mvs.md)):

- visloc-mvs with ground-truth poses: **83.7%**. ACMMP reports 83.42%,
  HPM-MVS 84.58%, APD-MVS 86.84%. On the same L4, ACMMP's released code
  matches visloc-mvs's F1 over six scenes in four times the time
  ([`mvs.md`](mvs.md#against-acmmp-on-the-same-gpu)).
- End to end (gsplat_photos SfM, aligned to the ground truth, then
  visloc-mvs): **81.5%**. Aligned by the camera centres alone, 76.3%
  against 57.4% for COLMAP 4.2.1 CUDA's SfM and MVS on the 10 scenes it
  finished.
  The 13-scene end-to-end figure combines runs r49 to r53; it has not been
  re-measured on one commit since #312 and #313.
- The gap to APD-MVS is mostly four weakly textured scenes (r37, against
  APD's reported scores): office −11.7, playground −10.4, facade −8.1,
  kicker −5.1 points, about 2.7 of the 3.2 points of the mean. office loses
  on accuracy (wrong planes), playground on completeness.
- Already tried and not adopted: APD-MVS's deformable patches alone, and
  HPM-MVS's sampling, KNN prior and hierarchical priors
  ([`mvs.md`](mvs.md#tried-and-not-adopted-apd-mvss-deformable-patches)).

**Items, in the recommended order** (costs from earlier L4 runs: 13 scenes
of visloc-mvs end to end take about 2 hours, about $1.3):

| Item | What | Pod time (L4) | Cost |
| --- | --- | --- | --- |
| 1. APD-MVS on the same GPU | Done (r54, 2026-10-22, $2.7): on terrace, office, kicker, playground and facade APD-MVS scores 83.2% at 2 cm against visloc-mvs's 78.5% (its paper: 84.87%), ahead by 4.6 to 8.2 points on the four weakly textured scenes and behind on terrace, in 5.0 times the time ([`mvs.md`](mvs.md#against-apd-mvs-on-the-same-gpu-r54)). Depth maps were not kept | — | — |
| 2. APD's pipeline, not just its patches | A state-gated weak-pixel path next to the existing one: ambiguity-based weak/strong classification, coplanarity-verified anchors, candidates drawn only from anchor planes and a RANSAC-fitted plane, local refinement after each pass ([notes](apd_mvs_code_notes.md), §7, items 1–4). Started on CPU (2026-10-22): APD's strong-only median recovers the 2.4 points the deformable patches lose on office at 400 px, ending level with the defaults; its local refinement adds nothing ([`mvs.md`](mvs.md#tried-and-not-adopted-apd-mvss-deformable-patches)). Next: the strong median on the GPU, then 3200 px | about 10 runs, mostly subsets | $10–15 |
| 3. 13-scene re-measure | End to end and with ground-truth poses on one commit of current main | about 2 h | about $1.3 |
| 4. Official test set | Submit to the ETH3D high-res test benchmark, the only basis for a ranking claim | — | — |

Total, with failed starts and idle time: under about $25.

Not GPU work, but open from the same thread: electro's last unregistered
image (DSC_9291) and DSC_9292's 1.8 cm offset, diagnosed locally on the r49a
L4 exports ([`mvs.md`](mvs.md#electro-what-the-unregistered-images-cost-r51)).

**How the pod runs were done.** One RunPod NVIDIA L4 pod per run, from a
CUDA base image. The pod's start script checks out the branch at a fixed
commit and builds it. It receives the ETH3D archives (and, for end-to-end
runs, exported SfM models) through a token-protected upload server on port
8001, and serves `/workspace/out` (logs, scores, clouds) on port 8000. It runs
the official `ETH3DMultiViewEvaluation`, then deletes itself 20 minutes after
finishing or failing, and after a hard time limit. The deploy, upload and
fetch scripts lived in the agent's session scratchpad and are not in the
repository. The RunPod API key and the upload token come from the
environment and are never committed.

## Guardrails (unchanged)

- No mandatory OpenCV / ONNX / PyTorch / CUDA in default crates.
- New behavior is opt-in until measured; default outputs stay bit-identical.
- Claims are scoped to the benchmark registry and claim matrix; negative
  results are recorded, not deleted.
- Every behavior change updates README / CHANGELOG / the relevant `docs/`
  page, then passes `scripts/check.sh`.
