# Basalt vs visloc-rs — VI-SLAM gap analysis

Date: 2026-08-18 (updated 2026-08-20). Author: Claude (coordinator), takeover from Codex.
Target: **Basalt-equivalent VI-SLAM** (`https://github.com/VladyslavUsenko/basalt`).

**2026-08-20 pivot:** stop SuperPoint A/B as the primary path; execute a
**faithful (ベタ) port** driven by Basalt's published EuRoC config. See
[`basalt_faithful_port_plan.md`](basalt_faithful_port_plan.md) and
[`configs/basalt/euroc_config.json`](../configs/basalt/euroc_config.json).

## Progress snapshot (faithful track)

| Lever | Status |
|---|---|
| `euroc_config.json` imported + Rust bindings | **Done** (`BasaltVioConfig`) |
| Demo `--basalt-euroc-profile` / `--basalt-config` | **Done** (maps `vio_max_kfs`, sqrt marg, OF frontend) |
| SqrtToSqrt window + FEJ carry | **Done** (`vi_sqrt_window`, demo default on) |
| Frame-to-frame KLT → OnlineSlam | **Wired** (`OpticalFlowFeatureExtractor`, track-id descriptors, stereo LK bootstrap) |
| Track longevity / Basalt FB=0.04 | **Partial** — scaffold relaxes FB + iters; LSSD + Gaussian pyr still TODO (MH_01 smoke: ~2 tracked frames / 33 inliers then drop) |
| FAST detection + LSSD + stereo epipolar OF | **Todo** |
| NFR → pose graph (mapper_*) | **Partial** (NFR seed exists; not Basalt mapper) |

Honest scoring of the **faithful Basalt port**: ~**45%**
(backend + config + OF in the VI loop; quality gap vs Basalt KLT remains).


## Basalt architecture (the target)

Basalt "Visual-Inertial Odometry and Mapping" rests on four pillars:

1. **`basalt_vio` — visual-inertial odometry:**
   - Sliding-window **keyframe** VI bundle adjustment. Joint state = keyframe poses, velocities,
     gyro/accel biases, and 3D landmarks (SLAM points).
   - **IMU preintegration** couples frames; metric scale comes from the accelerometer (m/s^2),
     so there is no monocular scale gauge freedom in the coupled solve.
   - **Square Root Marginalization** (`Square Root Marginalization for Sliding-Window Bundle
     Adjustment`, ICCV'21) — the marginal of an outgoing keyframe is produced on the **square-root
     (QR) factor representation**, sparser and numerically better conditioned than a dense
     information-form Schur complement. This is the mechanism that keeps the fixed-lag window
     bounded and consistent.
   - **First-Estimate-Jacobian (FEJ)** linearization to avoid inconsistent double-counting when
     a marginalized state's value is later updated.
   - Feature tracking is **optical flow (KLT)** on the raw image pyramid, not independent
     2D-detector+descriptor matching per frame (robust to texture-poor / fast-motion scenes).
   - Robust kernels on all residuals; online camera-IMU calibration in the bundle.
   - Two-stage bootstrap: gyro/accel bias + gravity/attitude init, then motion-based VI batch
     init (cf. its publications) before the sliding window runs online.

2. **`basalt_mapper` — visual-inertial mapping (offline):**
   - Takes each marginalization record emitted during VIO and runs **Nonlinear Factor Recovery**
     (`Visual-Inertial Mapping with Non-Linear Factor Recovery`, RA-L 2019): the marginalized
     dense factor is re-approximated as a **sparse** set of relative-pose (red), roll-pitch
     (magenta), and BA covisibility (green) factors. That sparse ladder is what a global pose
     graph / loop closure can then consume without the dense prior.
   - Detect keypoints → geometric 2D-2D matching → track building → triangulation → optimize.

3. **Calibration** tools (DS camera model, B-spline IMU-cam trajectory).

4. **Simulation** harness for component tests.

## What visloc-rs already has (mapped 08-18)

| Capability | Status | Where |
|---|---|---|
| IMU preintegration (Forster T-RO 2017) with bias linearization Jacobians | **Yes**, full on-manifold + first-order bias correction | `imu_preintegration.rs` |
| Stationary-window VI init `(R_wb, b_g, b_a)`, gyro bias = mean, gravity read-out | **Yes** | `vi_initializer.rs`, `online_slam_vi_init.rs` |
| Motion-based VI init (ORB-SLAM3 VIBA1/2 analogue) `(v_w, b_g, b_a)` with scale fixed | **Yes** | `vi_motion_initializer.rs`, `online_slam_motion_vi_init.rs` |
| Sliding-window keyframe VI-BA (joint pose/vel/bias + 3D points) | **Yes** | `online_slam_vi_ba.rs` |
| DPVO patch-BA + IMU preintegration fused in **one** joint Gauss-Newton | **Yes** | `dpvo_vi_ba.rs` |
| Schur-complement marginalization into a dense Gaussian prior (classes, information form) | **Yes** | `marginalization.rs`, used in `online_slam_vi_ba.rs` |
| Fixed-lag pose-graph marginalization with KL-optimal **sparsification** | **Yes** | `online_slam.rs` (`marginalize_oldest` / `marginalize_oldest_sparsified`) |
| Robust kernels on BA residuals (`RobustKernel`) | **Yes** | `bundle.rs` + configs |
| Fixed body→camera SE(3) extrinsic composed into IMU residuals | **Yes** (as calibrated input, pre-given) | `bundle.rs`, `dpvo_vi_ba.rs` |
| Covisibility-selected local BA (ORB-SLAM-style neighborhood) | **Yes**, A/B-testable | `covisibility_ba.rs` |
| Loop closure / relocalization / pose-graph refinement | **Yes** (Sim(3) mirror + banded) | `online_slam.rs` |
| Running VI entry points (EuRoC, stereo VO) | **Yes** | `euroc_online_slam_vi_image_demo.rs`, `online_slam_stereo_vo_kitti_demo.rs` |
| Deep frontend (SuperPoint/LightGlue via ONNX) | **Yes** | `deep_frontend_two_view_demo.rs`, export scripts |

## The gap (Basalt-equivalent levers)

1. **Square-root (QR) marginalization — absent.** Basalt's headline consistency mechanism
   (ICCV'21) is a square-root / QR factorization marginal, not the dense information-form Schur
   complement currently in `marginalization.rs`. The current one is numerically sound but denser
   and more conditioning-sensitive; the QR form is the Basil/GTSAM-style route. **This is the
   biggest single architectural gap and the most defensible "Basalt-equivalent" target.**
   - Note: a *sparse, KL-optimal* fixed-lag marginal already exists for the Sim3 pose graph in
     `online_slam.rs` — that is a step toward the sparse-factor idea but not square-root.
   - Missing: a **square-root factor/residual representation** throughout the sliding window
     (currently residuals → dense `J^TJ`, `J^T r`; QR would maintain an upper-triangular `R`).

2. **Nonlinear Factor Recovery / marg-data export — absent.** visloc-rs has no analogue of
   `basalt_mapper`: it does not serialize each marginalization record and re-approximate it as
   sparse relative-pose / roll-pitch / BA-covisibility factors for an offline global map.
   The `marginalize_oldest_sparsified` KL path is the seed of the sparse-factor idea but is
   **pose-graph-only** (no recovered camera-landmark covariates). This is the feature that turns
   a bounded windowed VIO into a globally consistent visual-inertial map.

3. **FEJ consistency — only one mention.** `online_slam_vi_ba.rs` mentions "dense FEJ" in a doc
   comment for marginalization, but there is no systematic First-Estimate-Jacobian discipline
   across the sliding-window residuals (Basalt keeps FEJ as a hard invariant to keep the
   marginalized + active part mutually consistent). With Schur marginalization present and no
   FEJ, the risk of "information double-counting" inconsistency is real. **Gap.**

4. **Optical-flow (KLT) keyframe tracking — absent in the VI path.** The VI track relies on a
   deep (SuperPoint/LightGlue) or keypoint-detector frontend, not dense KLT pyramid optical flow.
   Basalt's robustness on texture-poor / fast-motion (MH_04/MH_05) comes substantially from KLT.
   Two viable paths: (a) add a KLT block-matcher to the vision crate for the VI frontend, or
   (b) keep the proven deep frontend but add temporal consistency / motion-adaptive tracking so
   the gap on MH_04/MH_05 (currently the open 52.28/15.24 cm cases) closes. **Structural gap
   (optionally filled by existing deep frontend — a design choice, not always a gap).**

5. **Online camera-IMU extrinsics / intrinsics estimation — absent (assumed calibrated).**
   The extrinsic is a fixed calibrated input everywhere; no online `T_bc` refinement. Basalt
   refines it in the bundle. Lower priority for a fixed EuRoC replay (calib is ground-truth-ish)
   but relevant for real devices.

6. **Two-priority sliding window / keyframe bookkeeping** — partially there via `online_slam_vi_ba`
   (trigger on new keyframes, fixed-lag window, gauge-fix anchor). This matches Basalt's keyframe
   design; no action beyond unifying the marginalization.

## Recommended sequencing (after the GT-free runset finishes)

Basalt-equivalence is a **windowed-consistency** problem first, a mapping problem second:

1. **Square-root (QR) sliding-window VI-BA.** Replace (or add alongside) the dense Schur
   marginalization in `online_slam_vi_ba.rs` with a QR/squareroot factorization of the windowed
   residual, marginalizing the outgoing keyframe on `R` (Basalt ICCV'21). Add FEJ discipline
   (point 3) in the same change — they belong together. Gate: reproduce MH_01-800f tracking
   ≥ 0.98 and Sim(3) ATE < 1.5 m / scale < 10 without regressing the 400f visual-only result
   (matches `docs/archive/next_development_plan_2026-07.md` Priority 3/4 gates).
2. **Marg-data + nonlinear factor recovery.** Emit a marginalization record per window step and
   add a `basalt_mapper`-style re-factorization (relative-pose / roll-pitch / BA-cov) to feed the
   existing Sim3 pose-graph and loop-closure stack. This is the "VI-SLAM == bounded VIO + global
   map" payoff.
3. **Frontend choice for MH_04/MH_05.** Add temporal/motion-adaptive tracking to the deep
   frontend, or integrate a KLT block matcher, behind existing gates. Re-run only after 1&2 show
   no regression.

## Constraints that stay binding

- Never feed GT into the VI estimator (post-hoc evaluation only) — same rule as the SfM track.
- No self-feedback: the VI state must not consume its own corrected posterior (Priority 4 gate).
- A claim is evidence-backed only on a frozen, failure-inclusive protocol; keep the R2 gates.
- Basalt is BSD-3-Clause; if any code becomes a port rather than a reimplementation, record
  provenance + license in the repo. Prefer a from-scratch Rust reimplementation of the *algorithms*
  (square-root marginalization, factor recovery) so no C++/license coupling enters the pipeline.
