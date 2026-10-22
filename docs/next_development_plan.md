# Next Development Plan (as of 2026-10-08)

This is the short answer to "what next". Current state and the list of
closed threads are in [`PLAN.md`](../PLAN.md). The previous version of this
plan (2026-07-02) is archived at
[`archive/next_development_plan_2026-07.md`](archive/next_development_plan_2026-07.md).
All of its phases have since landed or been closed.

Scope rule for this round: **no new GPU work.** GPU paths (wgpu SfM, 3DGS,
DPVO CUDA) stay as they are; nothing here depends on them.

## Track 1: Correctness and release hygiene

| Item | Status | Gate |
| --- | --- | --- |
| 0.2.1 patch release (distortion fixes) | Done: version bumped, CHANGELOG cut | Tag `v0.2.1` after main CI is green |
| crates.io readiness | Done: all 16 crates package and verify-build from their packaged sources; root package trimmed from ~83 MB to 0.87 MiB compressed with an `include` allowlist | `cargo publish --workspace` (needs a crates.io token) |
| BA distortion models | Done: pose/structure BA linearises every lens model (OPENCV, FULL_OPENCV, the fisheye models, FOV, Double Sphere); opt-in p1/p2 self-calibration; rig observations use the sensor's lens too (2026-10-22). Open: matrix-free / GPU backends fall back for distorted cameras | `rig_sfm` on a distorted rig re-measured (OpenLORIS) |

## Track 2: Sensor coverage (the empty cells in the README sensor table)

| Item | Status | Gate |
| --- | --- | --- |
| Monocular + IMU VIO on the Basalt port | Done (experimental): `basalt_euroc_vio_demo --mono`; synthetic scale error < 0.3%, 5–9 mm RMS; stereo path unchanged ([notes](mono_inertial_vio.md)) | EuRoC ATE (needs dataset access); static-start drift; mono in the online mapper |
| GNSS fusion | Done (synthetic): joint VO + GNSS pose graph with estimated alignment, lever arm, GNC outlier rejection, dropout bridging; ATE 5.25 m → 0.90 m ([notes](gnss_fusion.md)) | Real-data run (e.g. KITTI raw OXTS); geodetic-to-ENU conversion; IMU in the graph |

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
