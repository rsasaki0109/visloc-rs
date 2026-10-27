# Monocular visual-only odometry on the Basalt port (design / status)

**Status: opt-in.** Enable it with `config.vio_visual_only: true` (optional key, default
false) or `basalt_euroc_vio_demo --mono-vo`. Recommended EuRoC config:
`configs/basalt/euroc_config_mono_vo.json`. Every IMU path (stereo-inertial and
mono-inertial, see `docs/mono_inertial_vio.md`) is unchanged.

## Why

The estimator needs an IMU even with one camera. With no IMU samples:
1. `predict_nav` keeps the previous pose, so every frame sits at the origin.
2. Every triangulation candidate then fails the metric 5 cm baseline gate.
3. No landmark is ever created, so nothing moves the pose.

`tests/mono_inertial_synthetic.rs::monocular_without_imu_never_bootstraps` pins this
behaviour of the default configuration. Upstream C++ Basalt's visual-only estimator
(`SqrtKeypointVoEstimator`) is stereo, so there was no monocular visual-only mode to port.

## What the mode changes

Each mechanism follows the corresponding ORB-SLAM idea. No ORB-SLAM source is used (it is
GPLv3); the C++ functions are named as behaviour references, and deviations are listed.

| Piece | Default (IMU) path | Visual-only path | Reference |
|---|---|---|---|
| Initialization | Roll/pitch from the first accelerometer sample (`initial_nav_from_imu`) | **Two-view bootstrap.** Frames are held back (output has `active_state_count == 0`) until one shares at least 50 cam0 tracks with a reference frame and has at least 1 deg median parallax. A 5-point RANSAC on bearings (the in-crate OpenGV solver, `mapper::features::relative_pose_from_bearings`) gives the relative pose, scaled so the median scene depth is 1. The reference and that frame enter the window as the first two keyframes. | `Tracking::MonocularInitialization`, `TwoViewReconstruction`. Deviation: an essential matrix only, no homography/fundamental model selection. |
| Prediction | IMU preintegration | **Constant velocity**, scaled by elapsed time, then refined by a robust (Huber 2 px) motion-only fit to the active landmarks. Basalt keeps each state's first linearization point (FEJ), so a good first estimate matters. | `TrackWithMotionModel` |
| Triangulation gate | Baseline of at least 5 cm, inverse distance under 3 m^-1 | **Ray parallax of at least 0.25 deg**, inverse distance under 10 (in units of the bootstrap median depth). ORB-SLAM uses about 1.1 deg, but Basalt triangulates only from the active window: at 1 deg more than half the tracks never became landmarks on EuRoC. | New map point creation in `LocalMapping` |
| Gauge | 4-DoF: position, yaw and biases in the anchor prior | **7-DoF.** The anchor also pins roll/pitch. Until the first marginal prior exists, a scale row pins the second keyframe's position along the bootstrap baseline. Velocity/bias columns of every state get a weak zero prior: nothing else observes them, so the prior keeps them non-singular and moves no pose. This term is also added to the f32 trial objective so trial and linearized costs agree. | First keyframe plus initial map scale fixed (`Optimizer::LocalBundleAdjustment`) |
| Body frame | IMU, with the metric `T_imu_cam` | **Camera** (identity extrinsic). A metric lever arm cannot be applied in an arbitrary-scale world: doing so produced about 3% scale drift even on exact synthetic tracks. Single camera only. | - |
| IMU input | Used | **Ignored** entirely. MargData has `used_imu = false`. | - |

Implementation: `vio/estimator.rs` (`visual_only_bootstrap`, `visual_only_prediction`,
`visual_only_refine_pose`, `visual_only_two_view`, triangulation gates), `vio/window.rs`
(`VisualOnlyPrior`, `visual_only_factor`), `config.rs` (key), and the demo `--mono-vo`. The
demo writes `trajectory.tum` (online), plus `trajectory_refined_window.tum` (each frame's last
in-window pose) and `trajectory_refined_kf.tum` (pose relative to its reference keyframe,
composed with that keyframe's last pose). `EstimatorOutput::window_frames` provides the poses
for the refined outputs.

## Verification

**Default path unchanged.** `scripts/bench/basalt_vio_parity.sh` hashes `trajectory.tum` for
stereo and `--mono` on MH_01 and MH_04. The hashes are identical before and after this change.

**Unit test** (`mapper/features.rs`):
- `relative_pose_from_bearings_recovers_left_from_right_pose`: two-view known answer; pins the
  `T_left_right` convention.

**Synthetic tests** (`tests/mono_inertial_synthetic.rs`):
- `visual_only_mono_exact_tracks_recover_trajectory_up_to_similarity`: exact tracks give the
  camera trajectory to under 1e-5 m after Sim3, with no scale drift between halves.
- `visual_only_mono_ignores_imu_and_matches_without_marg_data`: feeding IMU samples changes
  nothing, and the MargData and no-MargData paths give identical poses.
- `visual_only_mono_noisy_tracks_stay_consistent` (release-only): with 0.5 px noise, the Sim3
  RMS is under 1% of path length and scale drift under 5%.

**Determinism.** Repeated EuRoC runs give identical sha256.

## EuRoC MH results

Full sequences, `configs/basalt/euroc_config_mono_vo.json`, online `trajectory.tum`, scored in
the camera frame (GT `T_w_b * T_BS`).

| Seq | Sim3 ATE m | 10 s windowed Sim3 median m | rotation RPE 1 s median / p90 deg |
|---|---|---|---|
| MH_01 | 0.217 | 0.0074 | 0.070 / 0.146 |
| MH_02 | 0.712 | 0.0093 | 0.073 / 0.167 |
| MH_03 | 2.916 | 0.0344 | 0.093 / 0.296 |
| MH_04 | 4.195 | 0.0783 | 0.111 / 0.323 |
| MH_05 | 1.535 | 0.0825 | 0.089 / 0.195 |

For reference, measured on the same machine with the same scoring:
- **ORB-SLAM3 C++ monocular** (stock `EuRoC.yaml`): Sim3 0.016-0.081 m, windowed 0.007-0.024
  m, rotation median 0.086-0.117 deg.
- **DPVO** (MH_01 / MH_05): Sim3 0.20 / 0.46 m, windowed 0.005 / 0.019 m.

Readings:
- Rotation is at or below ORB-SLAM3's median on every sequence.
- Local translation is at ORB-SLAM3 level on MH_01/02 and 1.5-4x worse on MH_03-05.
- Global drift is unbounded, as for any odometry without a map. Global Sim3 varies strongly
  with configuration (scale drift accumulates chaotically).

## Limitations and follow-ups

- **No map reuse**, so global scale and position drift. ORB-SLAM3-style tracking against a
  persistent local map is the natural next step. The online mapper already has a default-off
  `projection_rematch` / `incremental_local_mapping` / `local_ba_window`.
- **The mapper is not wired for visual-only packets.** The NFR mapper requires a full-rank
  marginal (scale stays in the visual-only null space), runs `match_stereo`, and uses SE3 loop
  factors.
- **Local accuracy on difficult sequences** (MH_04/05) still trails ORB-SLAM3.
- **Single camera only.** A visual-only stereo mode would be `SqrtKeypointVoEstimator`.
