# GNSS + Visual-Odometry Fusion

`visloc_slam::gnss_fusion` fuses a visual-odometry (VO / VIO) trajectory with a
GNSS fix stream in **one joint optimization**: VO relative-pose factors and
GNSS position factors are solved together with the GNSS-to-map alignment,
instead of using each fix only as a per-frame search prior (which is what
[`track_sequence_with_gnss_prior`](gnss_demo.md) does).

It is opt-in: none of the existing GNSS-prior examples or their check scripts
change behaviour.

## What it is (and is not)

- **Loosely coupled in the GNSS domain.** The measurement is the receiver's
  position solution (ENU metres plus a 3×3 covariance or horizontal/vertical
  accuracies). Raw pseudoranges, Doppler, carrier phase, satellite geometry,
  clock states, and RTK ambiguity resolution are **not** modelled.
- **Pose-graph fusion on the visual side.** VO enters as relative-pose
  factors between consecutive frames, not as image reprojection residuals, so
  this is not a visual–GNSS bundle adjustment. Landmarks are untouched.
- **Jointly estimated alignment.** The ENU-to-map similarity is an optimization
  variable alongside every frame pose, not a fixed pre-alignment.
- **No IMU.** There is no INS mechanization or IMU preintegration here; a VIO
  front end can feed its (gravity-aligned) trajectory in as the VO input.

## Model

States, all in the VO **map** frame:

| State | DoF | Notes |
| --- | --- | --- |
| Frame pose `(Rᵢ, pᵢ)` | 6 | camera-to-map; the oldest frame of the window is held fixed (gauge) |
| Frame log-scale `σᵢ` | 1 | only in `GnssScaleMode::ScaleDrift` (monocular scale drift, Sim(3)-style node) |
| Alignment `x_enu = s·R·x_map + t` | 4–7 | yaw + translation (`AlignmentRotationDof::YawOnly`), or full rotation (`Full`); `s` fixed to 1 in `Metric`, free otherwise |

Factors:

| Factor | Residual | Weighting |
| --- | --- | --- |
| VO relative pose `Zᵢⱼ = Tᵢ⁻¹Tⱼ` | `[e^{-σᵢ}Rᵢᵀ(pⱼ−pᵢ) − t_Z; Log(R_Zᵀ Rᵢᵀ Rⱼ); σⱼ − σᵢ]` | `VoNoiseModel`: translation sigma = floor + ratio·step length, rotation sigma per step, log-scale random walk |
| GNSS position | `s·R·pᵢ + t + R·Rᵢ·l − z` | per-fix 3×3 information; robust kernel or GNC; `l` = antenna lever arm in the camera frame |
| Alignment prior | `[t − t₀; Log(R R₀ᵀ); ln s − ln s₀]` | sliding window only (`AlignmentPriorSigmas`) |

All Jacobians are analytic and unit-tested against central finite differences
(`pipelines/slam/src/gnss_pose_graph.rs`).

The solver is Levenberg–Marquardt with Marquardt (`diag(H)`) damping on the
crate's block-sparse Cholesky (7×7 blocks; nodes in time order with the
alignment block last, so a VO chain plus GNSS factors forms an "arrow" matrix
that factors without fill). It reuses the SLAM crate's existing machinery:
`RobustKernel` (Huber/Cauchy IRLS), `gnc::GncState` (Graduated Non-Convexity),
and `block_cholesky`.

### Pipeline

1. **Time synchronization** — `visloc_fusion::interpolate_gnss_fix` resamples
   the receiver's stream (its own rate and clock offset) at each camera frame
   time. A fix within `max_nearest_offset` (20 ms) is used directly; otherwise
   the two bracketing fixes are linearly interpolated if they are at most
   `max_bracket_gap` (1.5 s) apart. Wider gaps are **dropouts**: those frames
   get no GNSS factor and are carried by the VO chain.
2. **Per-fix covariance** — the fix's full `PositionCovariance` if present,
   else its horizontal/vertical accuracies (vertical defaults to 2× horizontal),
   else `default_horizontal_sigma` / `default_vertical_sigma`. Interpolation
   blends the two covariances *convexly* (GNSS errors are time-correlated, so
   the midpoint is not treated as more accurate than either fix) and adds
   `motion_sigma_per_second` × distance-to-nearest-fix. When the camera is
   faster than the receiver, one fix feeds several frames; its information is
   split between them (`share_fix_information`) so it is not double counted.
3. **Alignment bootstrap** — RANSAC (2-point minimal samples for yaw-only,
   3-point for full rotation) plus a weighted Procrustes/Umeyama refit over the
   earliest fixes that span `min_horizontal_extent` (30 m), where VO drift is
   smallest. The lever arm is folded in by fixed-point iteration. Until the
   extent is reached the alignment is unobservable and `optimize` returns
   `GnssFusionError::AlignmentNotObservable`.
4. **Joint solve** — with the default `GnssRobustMode::Gnc` (truncated least
   squares, inlier scale at the χ²(3) 99.9 % gate ≈ 4.03σ): a least-squares
   warm start, GNC annealing, χ² classification of every GNSS factor, then a
   final refinement with the rejected fixes removed (`refine_without_outliers`).
   `GnssRobustMode::Kernel(Huber/Cauchy)` and `GnssRobustMode::None` are also
   available.

### Batch vs sliding window

- `GnssVoFusionConfig::window = None` (default): one full-batch solve over all
  frames, frame 0 fixed.
- `window = Some(n)`: a causal fixed-lag smoother. Frames and fixes are pushed
  in time order; every `window_update_stride` frames the last `n` frames are
  re-solved with the oldest one fixed at its current estimate and the previous
  alignment as a Gaussian prior. New frames are initialized by chaining VO
  motion onto the latest fused state. Each frame's reported ENU pose is the
  one from the last solve that contained it (the alignment keeps absorbing VO
  drift after a frame leaves the window). There is no marginalization: what
  dropped frames said about the alignment survives only through that prior.

## API

```rust
use visloc_rs::slam::gnss_fusion::{
    fuse_gnss_with_visual_odometry, GnssVoFusion, GnssVoFusionConfig,
};

// vo: Vec<TimedPose> (world-to-camera in the VO map frame, increasing time)
// gnss: MeasurementBuffer<GnssMeasurement> (ENU antenna positions)
let mut config = GnssVoFusionConfig::metric_gravity_aligned(); // or ::monocular()
config.lever_arm = nalgebra::Vector3::new(0.0, -1.5, -0.8); // antenna in camera frame
let fused = fuse_gnss_with_visual_odometry(&vo, &gnss, &config)?;
// fused.trajectory_enu, fused.alignment, fused.bootstrap,
// fused.frames[i].gnss (Inlier / Outlier / NoFix), fused.summary

// Incremental use:
let mut fusion = GnssVoFusion::new(config);
fusion.push_vo_pose(&vo[0])?;
fusion.push_gnss(fix);
let update = fusion.optimize()?; // bootstraps on first success
let enu = fusion.enu_trajectory();
```

Lower-level pieces, usable on their own:

- `visloc_fusion::{interpolate_gnss_fix, gnss_fix_covariance, GnssInterpolationConfig}` — time sync.
- `visloc_slam::{GnssPoseGraph, VoRelativeFactor, GnssPositionFactor, GnssAlignmentPrior, GnssPoseGraphConfig}` — the optimizer.
- `visloc_slam::{bootstrap_gnss_alignment, fit_gnss_alignment}` — robust / closed-form alignment.
- `visloc_slam::gnss_synthetic` — deterministic synthetic scenarios.
- `gnss_fusion::{position_rmse, aligned_position_rmse, transform_trajectory}` — evaluation helpers.

Choose the alignment/scale model to match the VO:

| VO | `rotation_dof` | `scale_mode` | Preset |
| --- | --- | --- | --- |
| Stereo VO / VIO, map frame gravity aligned with `+z` up | `YawOnly` | `Metric` | `metric_gravity_aligned()` |
| Metric VO, map frame = first camera frame | `Full` | `Metric` | — |
| Monocular VO | `Full` | `ScaleDrift` (or `GlobalScale`) | `monocular()` |

## Run It

```bash
cargo run --example gnss_vo_fusion_demo
cargo run --example gnss_vo_fusion_demo -- --scenario monocular
cargo run --example gnss_vo_fusion_demo -- --window 150
cargo run --example gnss_vo_fusion_demo -- --robust none   # see outliers pull LS
cargo run --example gnss_vo_fusion_demo -- --out-dir target/visloc_gnss_fusion_demo
sh scripts/check_gnss_fusion_demo_outputs.sh
```

The synthetic drive (`SyntheticGnssVoConfig::default()`): a ~1 km loop over
120 s, 600 camera frames at 5 Hz; VO with ~1°/min heading drift, 2 % scale
error, and per-step noise; a 2 Hz receiver with a 37 ms clock offset,
0.8 m / 1.5 m (horizontal / vertical) noise, a 15 s dropout, and three
multipath bursts (6 fixes offset by 21–28 m); antenna lever arm 1.5 m up and
0.8 m behind the camera. `--out-dir` writes `summary.json` and a per-frame
`trajectory.csv` (truth, VO, fused, GNSS status).

Results from the deterministic scenarios (debug build):

| Scenario | VO only, bootstrap-aligned | VO only, oracle-aligned to truth | Fused | Notes |
| --- | --- | --- | --- | --- |
| Metric, batch, GNC | 5.25 m | 2.24 m (SE(3)) | **0.90 m** | dropout frames 1.70 m; all 19 frames touching corrupted fixes rejected, none elsewhere; yaw error 0.75° |
| Metric, batch, `--robust none` | 5.25 m | 2.24 m | 1.33 m | frames near outliers 1.68 m RMSE vs 0.54 m with GNC |
| Metric, sliding window 150 | 5.25 m | 2.24 m | **1.04 m** | causal; 60 window solves |
| Monocular, batch, GNC | 185 m | 3.22 m (Sim(3)) | **0.37 m** | alignment scale error −2.2 %, rotation error 0.7° |

"Oracle-aligned" is the best-fit rigid (or similarity) transform of the VO onto
ground truth, i.e. the most favourable possible VO-only number; the fused
trajectory is evaluated with no alignment at all. In the metric case the
residual fused error is dominated by the unmodelled 2 % VO scale error, which
the metric model cannot absorb (the monocular `ScaleDrift` model can, hence its
lower error).

Runtime: the full-batch 600-frame demo takes ~16 s in an unoptimized debug
build (≈40 LM iterations including GNC); release builds are one to two orders
of magnitude faster.

## Tests

- `pipelines/slam/src/gnss_pose_graph.rs` unit tests: finite-difference checks
  of the GNSS, VO-relative, and alignment-prior Jacobians; RANSAC bootstrap
  recovery of yaw/scale/translation with outliers and of a full rotation;
  short-baseline rejection; optimizer alignment recovery with an outlier.
- `pipelines/fusion/tests/gnss_interpolation.rs`: nearest/interpolated
  selection, covariance blending and motion inflation, dropouts, covariance
  precedence.
- `pipelines/slam/tests/gnss_fusion.rs`: end-to-end synthetic drives — fused
  ATE well below VO-only, every outlier-touched frame rejected and none
  elsewhere, outliers do not pull the trajectory (vs plain least squares),
  dropout bridged, monocular scale recovered, sliding window, and
  unobservable-alignment / bad-timestamp errors.

## Limits

- Position-only GNSS: no velocity, heading, or raw-observation factors, and no
  receiver clock or multipath modelling beyond robust down-weighting.
- GNSS noise is treated as white per fix. Real receivers have time-correlated
  errors; the convex covariance blending and information sharing are
  conservative heuristics, not a colored-noise model.
- The VO noise model is a hand-set per-step sigma; VO covariances from the
  front end are not consumed yet, and the VO factors are only between
  consecutive frames (no loop closures inside this graph).
- Yaw-only alignment assumes a gravity-aligned map with `+z` up. With a full
  rotation, roll about a straight-line trajectory is unobservable until the
  path turns.
- The ENU frame is whatever the fixes are expressed in; geodetic (LLA/ECEF) to
  local ENU conversion is not provided here.
- The sliding window has no marginalization prior on the frames it drops; the
  oldest window frame is simply held fixed.
- Validated only on synthetic data so far; no public GNSS+camera dataset run is
  claimed.
