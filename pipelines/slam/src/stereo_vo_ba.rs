//! Multi-frame stereo bundle adjustment for stereo VO trajectories.
//!
//! Per-pair PnP underestimates absolute vertical motion on slope-following
//! sequences (e.g. KITTI seq08): the road follows the camera, so road-surface
//! stereo features yield `Δy_cam ≈ 0` between frames even when the camera
//! actually rises in world coordinates. Multi-view BA breaks this ambiguity
//! by requiring a single 3D landmark to explain its observations across many
//! frames — features that genuinely persist must obey rigid-body motion, and
//! the long-baseline geometry constrains the camera trajectory's vertical
//! component that pairwise stereo cannot.
//!
//! This module builds forward feature tracks from per-pair temporal matches,
//! initialises landmarks from the first stereo observation in each track, and
//! runs Schur-complement BA over all poses (with pose 0 fixed) plus all
//! long-tracked landmarks. The robust Huber kernel down-weights tracks that
//! cannot be explained as a rigid-body landmark (e.g. road texture matches
//! that "follow" the camera through pitched motion).

use std::collections::HashMap;

use nalgebra::{DMatrix, Point2, Point3, Vector3, Vector6};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::DescriptorMatch;
use visloc_vision::stereo_vo::StereoFeature;

use crate::imu_preintegration::{ImuPreintegrationFactor, ImuPreintegrator};
use crate::{
    BaConfig, BaError, BaResult, BaStereoObservation, BiasRandomWalkFactor, BundleAdjustment,
    GravityPrior, LinearSolver, PerPoseGravityPrior, PositionPrior, RobustKernel,
};

/// Single body-frame IMU sample for a stereo-VO BA inter-keyframe window.
/// `dt` is the time elapsed since the previous sample (or since the
/// window start for the first sample). `gyro` and `accel` are body-frame;
/// gravity is *not* pre-subtracted from `accel` — the BA-side IMU factor
/// handles gravity compensation through [`StereoVoBaImuInput::gravity_world`].
#[derive(Debug, Clone, PartialEq)]
pub struct StereoVoBaImuSample {
    /// Time step in seconds (must be `> 0`).
    pub dt: f64,
    /// Body-frame gyroscope reading (rad / s).
    pub gyro: Vector3<f64>,
    /// Body-frame accelerometer reading (m / s²), gravity *not* removed.
    pub accel: Vector3<f64>,
}

/// IMU input wired into a stereo-VO BA refinement. The caller supplies
/// `n_frames − 1` sample windows: `windows[i]` covers the integration
/// interval from keyframe `i` to keyframe `i + 1`. The refiner pre-
/// integrates each window with [`ImuPreintegrator::new_with_bias`] at
/// the initial bias linearisation point, registers a velocity / bias
/// state per keyframe, and pushes an [`ImuPreintegrationFactor`] for
/// every non-empty window. When `bias_random_walk_weight` is `Some(w)`,
/// every consecutive bias pair is also tied with a
/// [`BiasRandomWalkFactor`] of weight `w`.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoVoBaImuInput {
    /// `n_frames − 1` body-frame sample windows. An empty window (no
    /// samples) means "no IMU factor between those two keyframes" — the
    /// pair contributes neither velocity nor bias state, and any
    /// random-walk factor that would have referenced them is skipped.
    pub windows: Vec<Vec<StereoVoBaImuSample>>,
    /// World-frame gravity vector applied by every IMU factor. KITTI
    /// y-down convention: `(0, 9.81, 0)`. Pass `Vector3::zeros()` for
    /// gravity-free pre-integration (e.g. zero-g simulation tests).
    pub gravity_world: Vector3<f64>,
    /// Initial gyro bias linearisation point shared by every window.
    pub bias_gyro_init: Vector3<f64>,
    /// Initial accel bias linearisation point shared by every window.
    pub bias_acc_init: Vector3<f64>,
    /// 3-vector position residual weight `1/σ_p²` on every factor.
    pub weight_position: f64,
    /// 3-vector velocity residual weight `1/σ_v²` on every factor.
    pub weight_velocity: f64,
    /// 3-vector rotation residual weight `1/σ_R²` on every factor.
    pub weight_rotation: f64,
    /// When `Some(w)`, a [`BiasRandomWalkFactor`] is added between every
    /// pair of consecutive bias slots with weight `w` on
    /// `‖b_j − b_i‖²`. `None` disables the random-walk tie.
    pub bias_random_walk_weight: Option<f64>,
    /// When `true`, the first IMU-active keyframe's bias is held fixed
    /// at its initial value (gauge choice for an otherwise drift-free
    /// random walk).
    pub fix_first_bias: bool,
    /// When `true`, the first IMU-active keyframe's velocity is held
    /// fixed at its initial value. Pose 0 is always fixed for the BA
    /// gauge but velocity is an independent 3-DoF — pin it when a
    /// measured initial velocity is available.
    pub fix_first_velocity: bool,
}

impl StereoVoBaImuInput {
    /// Convenience constructor with weights, zero initial biases, no
    /// random-walk tie, gauge-fixed first bias, free first velocity.
    pub fn new(
        windows: Vec<Vec<StereoVoBaImuSample>>,
        gravity_world: Vector3<f64>,
        weight_position: f64,
        weight_velocity: f64,
        weight_rotation: f64,
    ) -> Self {
        Self {
            windows,
            gravity_world,
            bias_gyro_init: Vector3::zeros(),
            bias_acc_init: Vector3::zeros(),
            weight_position,
            weight_velocity,
            weight_rotation,
            bias_random_walk_weight: None,
            fix_first_bias: true,
            fix_first_velocity: false,
        }
    }
}

/// Refined IMU state from a BA solve that included
/// [`StereoVoBaImuInput`]. The vectors are indexed by keyframe id (so
/// entry `i` corresponds to `initial_poses[i]`). Keyframes that did not
/// participate in any IMU factor have their velocity left at the
/// build-time initial value and their bias at the input
/// `bias_*_init`.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoVoBaImuRefinement {
    /// World-frame velocity refined per keyframe.
    pub refined_velocities: Vec<Vector3<f64>>,
    /// Refined gyro bias per keyframe.
    pub refined_bias_gyro: Vec<Vector3<f64>>,
    /// Refined accelerometer bias per keyframe.
    pub refined_bias_acc: Vec<Vector3<f64>>,
}

/// Landmark initialisation strategy for [`refine_stereo_vo_with_ba`]. See
/// [`StereoVoBaConfig::landmark_init`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LandmarkInit {
    /// Triangulate the landmark from the first stereo observation in the
    /// track, transformed to world via that frame's initial pose. Default
    /// for backwards-compatibility.
    #[default]
    StereoSingleFrame,
    /// Linear DLT over ALL stereo observations in the track (3 equations
    /// per frame: `u_l`, `v_l`, `u_r`). Solved via SVD on a `(3n)×4`
    /// matrix.
    MultiViewDlt,
}

/// Configuration for [`refine_stereo_vo_with_ba`].
#[derive(Debug, Clone, PartialEq)]
pub struct StereoVoBaConfig {
    /// Minimum number of frames a track must span before it contributes a
    /// landmark to BA. Tracks shorter than this are dropped. The pairwise
    /// PnP-only path already exercises 2-frame tracks; the value of BA is in
    /// 3+ -frame tracks whose long baseline disambiguates ambiguous per-pair
    /// solutions.
    pub min_track_length: usize,
    /// Maximum cam-frame depth (metres) accepted for the seed stereo
    /// observation that initialises a landmark. Far-field points (large `Z`)
    /// have very noisy initial 3D positions from stereo triangulation and
    /// destabilise LM until refined.
    pub max_initial_depth_m: f64,
    /// Optional upper-row fraction (0..1) of the image. When set to `Some(r)`,
    /// only tracks whose FIRST observation has a left-image row `v < r * h`
    /// contribute landmarks. Tighter (smaller) values keep only above-horizon
    /// scenery (sky / buildings / trees) and exclude road-surface features
    /// that "follow" the camera through pitched motion on slope-following
    /// sequences. `None` (default) accepts every track regardless of row.
    pub max_seed_row_fraction: Option<f64>,
    /// Optional per-track pre-BA residual gate. When `Some(threshold_px)`,
    /// each candidate track's observations are projected against the initial
    /// poses + initial landmark, and the track is dropped if any single
    /// stereo residual exceeds `threshold_px` (Euclidean norm of the 3-vector
    /// `(du_l, dv, du_r)`). This rejects "pseudo-tracks" where chained
    /// temporal matches link slightly-different physical points — the
    /// dominant failure mode that made naive global BA degrade the
    /// trajectory on real KITTI sequences. Typical range `4.0..8.0`. `None`
    /// (default) admits every track regardless of fit.
    pub max_init_residual_px: Option<f64>,
    /// Optional minimum temporal-match confidence (in `(0, 1]`) applied to
    /// the temporal lookup before building forward tracks. Matches with
    /// `confidence < min` (or unset confidence) are excluded from chaining.
    /// Long chains of low-confidence matches were the second-largest source
    /// of pseudo-tracks. `None` (default) lets every temporal match
    /// contribute. A practical floor for SuperPoint/LightGlue is `0.8`.
    pub min_temporal_confidence: Option<f32>,
    /// Optional minimum number of tracks that must survive all filters
    /// before BA actually runs. When fewer than this many tracks remain,
    /// `refine_stereo_vo_with_ba` returns `Err(InsufficientTracks)` and the
    /// caller is expected to keep the initial poses. The gate guards against
    /// pathological sequences (low-feature highway, sparse texture) where
    /// the filtered track set is too small to support stable BA — joint
    /// optimisation in that regime tends to amplify whatever drift the
    /// initial trajectory already has. `None` (default) runs BA whenever
    /// there is at least one track.
    pub min_track_count: Option<usize>,
    /// Landmark initialisation strategy.
    ///
    /// `StereoSingleFrame` (legacy default) triangulates from the first
    /// stereo observation in the track, transformed to world via that
    /// frame's initial pose. Simple and fast, but for long tracks the
    /// initial 3D position is biased by the per-frame stereo depth noise
    /// and inflates the pre-BA residual.
    ///
    /// `MultiViewDlt` solves a linear DLT over ALL stereo observations in
    /// the track (3 equations per frame: `u_l`, `v_l`, `u_r`). Smaller
    /// initial residuals → tighter `max_init_residual_px` gates become
    /// viable → cleaner BA convergence.
    pub landmark_init: LandmarkInit,
    /// Optional sliding-window size. When `Some(w)`, BA processes
    /// overlapping `w`-frame windows instead of the entire trajectory at
    /// once: window 0 covers poses `[0, w)`, window 1 covers
    /// `[w-1, 2w-1)`, etc., so each window shares its first pose with the
    /// previous window's last pose. Within each window the first pose is
    /// fixed (the global gauge for window 0, the previously-refined
    /// boundary pose for subsequent windows), and only tracks confined to
    /// that window contribute observations. `None` (default) is global
    /// joint BA over all frames.
    pub window_size: Option<usize>,
    /// Optional level-world gravity prior. When `Some`, every non-anchor
    /// pose contributes a rotation-alignment residual `R_wc · g_world −
    /// g_camera_observed` to the BA solve (see [`GravityPrior`]). Useful on
    /// KITTI-style ground-vehicle sequences where the world frame is level
    /// and pitch / roll drift is the dominant rotational failure mode. In
    /// sliding-window mode the prior is applied to every window. `None`
    /// (default) runs BA without a gravity constraint.
    pub gravity_prior: Option<GravityPrior>,
    /// Optional absolute camera-centre prior. Use axis weights to constrain
    /// only the observable components needed by the experiment, e.g.
    /// `(0, w, 0)` for KITTI y/height or `(0, w, w)` for y+z grade priors.
    /// In sliding-window mode observations are remapped to each local window.
    pub position_prior: Option<PositionPrior>,
    /// Optional per-keyframe gravity prior. Each observation constrains
    /// `R_wc · g_world ≈ g_camera_observed` at the named frame. This is
    /// the online-friendly companion to [`Self::gravity_prior`] — the
    /// observation can be sourced per-frame from an accelerometer
    /// sample (rotated into the camera frame via the body→camera
    /// extrinsic), so a deployed VIO can use exactly the same prior
    /// path the BA exercises here. In sliding-window mode observations
    /// are remapped to each local window. `None` (default) runs BA
    /// without per-keyframe gravity constraints.
    pub per_pose_gravity_prior: Option<PerPoseGravityPrior>,
    /// Optional inter-keyframe IMU input. When `Some`, every inter-frame
    /// window in [`StereoVoBaImuInput::windows`] is pre-integrated and an
    /// [`ImuPreintegrationFactor`] is added to BA. Velocity and bias
    /// state are registered for every keyframe that participates. Not
    /// compatible with [`Self::window_size`] (the sliding window slices
    /// poses but not IMU samples). `None` (default) runs visual-only BA.
    pub imu_input: Option<StereoVoBaImuInput>,
    /// Number of leading poses to hold fixed as the gauge / map anchor. The
    /// default `1` fixes only pose 0 (the classic single-anchor window BA). A
    /// larger value fixes a *prefix* of poses: used by the streaming wrapper to
    /// run a "local map" BA over an extended backward window whose old frames are
    /// fixed (they anchor long-baseline landmarks) while only the recent frames
    /// are optimised — the fixed-keyframe local BA pattern. Values are clamped to
    /// `[1, n_frames-1]`.
    pub fix_pose_prefix: usize,
    /// Underlying BA solver config. Defaults to sparse Cholesky + Huber
    /// kernel sized to a 3 px reprojection residual norm.
    pub ba_config: BaConfig,
}

impl Default for StereoVoBaConfig {
    fn default() -> Self {
        Self {
            min_track_length: 3,
            max_initial_depth_m: 60.0,
            max_seed_row_fraction: None,
            max_init_residual_px: None,
            min_temporal_confidence: None,
            min_track_count: None,
            landmark_init: LandmarkInit::StereoSingleFrame,
            window_size: None,
            gravity_prior: None,
            position_prior: None,
            per_pose_gravity_prior: None,
            imu_input: None,
            fix_pose_prefix: 1,
            ba_config: BaConfig {
                max_iterations: 12,
                robust_kernel: RobustKernel::Huber { delta: 3.0 },
                linear_solver: LinearSolver::Sparse,
                ..BaConfig::default()
            },
        }
    }
}

/// Result of [`refine_stereo_vo_with_ba`].
#[derive(Debug, Clone, PartialEq)]
pub struct StereoVoBaRefinement {
    /// Refined `world_to_camera` pose per frame (same length as the input
    /// `initial_poses`). Pose 0 is unchanged (it is fixed as the anchor).
    pub refined_poses: Vec<Pose>,
    /// LM iteration trace and final cost from the BA solve.
    pub ba_result: BaResult,
    /// Number of multi-frame tracks that contributed landmarks to BA.
    pub track_count: usize,
    /// Number of `BaStereoObservation`s added across all tracks.
    pub observation_count: usize,
    /// Refined IMU state when [`StereoVoBaConfig::imu_input`] was set.
    /// `None` when the BA solve ran visual-only.
    pub imu_refinement: Option<StereoVoBaImuRefinement>,
}

/// Refine a stereo VO trajectory with multi-frame stereo bundle adjustment.
///
/// Build forward feature tracks by chaining `temporal_matches` across frames,
/// initialise each long track's landmark from its first valid stereo
/// observation, then run Schur-complement BA over all `initial_poses` (with
/// pose 0 fixed) plus all long-tracked landmarks. The metric scale is
/// anchored by the rectified-stereo baseline; no extra gauge fixing is
/// required.
// Public entry point: the stereo rig (camera, baseline), the per-frame VO
// sequence (poses, left/right features, stereo, temporal matches), and config.
#[allow(clippy::too_many_arguments)]
pub fn refine_stereo_vo_with_ba(
    camera: &Camera,
    baseline: f64,
    initial_poses: &[Pose],
    left_features: &[FeatureSet],
    right_features: &[FeatureSet],
    stereo_per_frame: &[Vec<StereoFeature>],
    temporal_matches: &[Vec<DescriptorMatch>],
    config: &StereoVoBaConfig,
) -> Result<StereoVoBaRefinement, StereoVoBaError> {
    let n_frames = initial_poses.len();
    if n_frames < config.min_track_length || n_frames < 2 {
        return Err(StereoVoBaError::TooFewFrames(n_frames));
    }
    if left_features.len() != n_frames
        || right_features.len() != n_frames
        || stereo_per_frame.len() != n_frames
    {
        return Err(StereoVoBaError::InputLengthMismatch);
    }
    if temporal_matches.len() != n_frames - 1 {
        return Err(StereoVoBaError::InputLengthMismatch);
    }
    if let Some(imu) = &config.imu_input {
        if imu.windows.len() != n_frames - 1 {
            return Err(StereoVoBaError::InvalidImuInput {
                reason: format!(
                    "imu_input.windows.len()={} must equal initial_poses.len()-1={}",
                    imu.windows.len(),
                    n_frames - 1
                ),
            });
        }
        if config.window_size.is_some() {
            return Err(StereoVoBaError::InvalidImuInput {
                reason:
                    "sliding-window BA (`window_size`) is not yet wired through the IMU factor stack"
                        .to_string(),
            });
        }
        for (i, samples) in imu.windows.iter().enumerate() {
            for (k, s) in samples.iter().enumerate() {
                if !(s.dt.is_finite() && s.dt > 0.0) {
                    return Err(StereoVoBaError::InvalidImuInput {
                        reason: format!("window[{i}] sample[{k}] has non-positive dt={}", s.dt),
                    });
                }
            }
        }
    }

    if let Some(w) = config.window_size {
        if w < config.min_track_length || w < 2 {
            return Err(StereoVoBaError::TooFewFrames(w));
        }
        // Sliding-window mode: refine overlapping windows of `w` frames each.
        // Successive windows share one boundary pose so the trajectory stays
        // continuous; that boundary pose, once refined, becomes the fixed
        // gauge of the next window.
        let mut refined = initial_poses.to_vec();
        let mut total_tracks = 0usize;
        let mut total_obs = 0usize;
        let mut agg_initial_cost = 0.0;
        let mut agg_final_cost = 0.0;
        let mut all_iterations: Vec<crate::BaIterationStats> = Vec::new();
        let mut converged_all = true;
        let mut start = 0usize;
        let mut sub_config = config.clone();
        sub_config.window_size = None;
        loop {
            let end = (start + w).min(n_frames);
            if end <= start + 1 {
                break;
            }
            if end - start >= config.min_track_length {
                sub_config.position_prior = config
                    .position_prior
                    .as_ref()
                    .map(|prior| slice_position_prior_for_window(prior, start, end));
                sub_config.per_pose_gravity_prior = config
                    .per_pose_gravity_prior
                    .as_ref()
                    .map(|prior| slice_per_pose_gravity_prior_for_window(prior, start, end));
                match refine_stereo_vo_with_ba(
                    camera,
                    baseline,
                    &refined[start..end],
                    &left_features[start..end],
                    &right_features[start..end],
                    &stereo_per_frame[start..end],
                    &temporal_matches[start..end - 1],
                    &sub_config,
                ) {
                    Ok(window_refinement) => {
                        for (offset, p) in window_refinement.refined_poses.iter().enumerate() {
                            refined[start + offset] = p.clone();
                        }
                        total_tracks += window_refinement.track_count;
                        total_obs += window_refinement.observation_count;
                        agg_initial_cost += window_refinement.ba_result.initial_cost;
                        agg_final_cost += window_refinement.ba_result.final_cost;
                        all_iterations.extend(window_refinement.ba_result.iterations);
                        converged_all = converged_all && window_refinement.ba_result.converged;
                    }
                    Err(StereoVoBaError::InsufficientTracks { .. })
                    | Err(StereoVoBaError::NoLongTracks) => {
                        // Skip this window; carry initial poses through.
                    }
                    Err(e) => return Err(e),
                }
            }
            if end >= n_frames {
                break;
            }
            start = end - 1; // overlap by one boundary frame
        }
        if total_tracks == 0 {
            return Err(StereoVoBaError::NoLongTracks);
        }
        return Ok(StereoVoBaRefinement {
            refined_poses: refined,
            ba_result: BaResult {
                initial_cost: agg_initial_cost,
                final_cost: agg_final_cost,
                iterations: all_iterations,
                converged: converged_all,
            },
            track_count: total_tracks,
            observation_count: total_obs,
            imu_refinement: None,
        });
    }

    let stereo_lookup: Vec<HashMap<usize, &StereoFeature>> = stereo_per_frame
        .iter()
        .map(|stereo| stereo.iter().map(|f| (f.left_index, f)).collect())
        .collect();

    let temporal_lookup: Vec<HashMap<usize, usize>> = temporal_matches
        .iter()
        .map(|matches| {
            matches
                .iter()
                .filter(|m| match config.min_temporal_confidence {
                    Some(min) => m
                        .confidence
                        .map(|c| c.is_finite() && c >= min)
                        .unwrap_or(false),
                    None => true,
                })
                .map(|m| (m.query_index, m.train_index))
                .collect()
        })
        .collect();

    let tracks = build_forward_tracks(
        n_frames,
        left_features,
        &temporal_lookup,
        config.min_track_length,
    );
    if tracks.is_empty() {
        return Err(StereoVoBaError::NoLongTracks);
    }

    let mut ba = BundleAdjustment::new(camera.clone());
    ba.set_stereo_baseline(baseline);
    if let Some(gravity) = config.gravity_prior.clone() {
        ba.set_gravity_prior(gravity);
    }
    if let Some(position_prior) = config.position_prior.clone() {
        ba.set_position_prior(position_prior);
    }
    if let Some(per_pose_gravity_prior) = config.per_pose_gravity_prior.clone() {
        ba.set_per_pose_gravity_prior(per_pose_gravity_prior);
    }
    for (i, pose) in initial_poses.iter().enumerate() {
        ba.add_pose(i as u64, pose.clone());
    }
    // Fix a leading prefix of poses as the gauge / local-map anchor (default 1 =
    // fix pose 0 only). A larger prefix anchors long-baseline landmarks over an
    // extended backward window while only the recent poses move.
    let fixed_prefix = config
        .fix_pose_prefix
        .clamp(1, n_frames.saturating_sub(1).max(1));
    for i in 0..fixed_prefix {
        ba.fix_pose(i as u64);
    }

    // IMU factor / velocity / bias wiring. Pre-integrate every supplied
    // window, register velocity & bias slots for each active keyframe,
    // then push the factors. Velocities are initialised from the
    // inter-keyframe pose-centre delta divided by the integrated time
    // (a coarse but reasonable seed); biases start at the input
    // linearisation point. Active keyframes are tracked so the
    // refinement extraction can fall back cleanly for non-IMU frames.
    let mut imu_active: Vec<bool> = vec![false; n_frames];
    if let Some(imu) = &config.imu_input {
        let bias_init = Vector6::from_iterator(
            imu.bias_gyro_init
                .iter()
                .copied()
                .chain(imu.bias_acc_init.iter().copied()),
        );
        for (window_index, samples) in imu.windows.iter().enumerate() {
            if samples.is_empty() {
                continue;
            }
            let from_id = window_index as u64;
            let to_id = (window_index + 1) as u64;
            let mut pre = ImuPreintegrator::new_with_bias(imu.bias_gyro_init, imu.bias_acc_init);
            for s in samples {
                pre.integrate_sample(s.gyro, s.accel, s.dt);
            }
            let delta = pre.delta();
            // Seed velocities from the inter-keyframe world-frame centre
            // delta, scaled by the integrated time. Only assign when the
            // slot is not yet populated so multi-window keyframes use
            // their first arriving seed and don't oscillate.
            let dt_total = delta.delta_time;
            let c_from = initial_poses[window_index].camera_center_world();
            let c_to = initial_poses[window_index + 1].camera_center_world();
            let v_seed = if dt_total > 0.0 {
                (c_to - c_from) / dt_total
            } else {
                Vector3::zeros()
            };
            if !imu_active[window_index] {
                ba.add_velocity(from_id, v_seed);
                ba.add_bias(from_id, bias_init);
                imu_active[window_index] = true;
            }
            if !imu_active[window_index + 1] {
                ba.add_velocity(to_id, v_seed);
                ba.add_bias(to_id, bias_init);
                imu_active[window_index + 1] = true;
            }
            ba.add_imu_factor(ImuPreintegrationFactor {
                keyframe_id_from: from_id,
                keyframe_id_to: to_id,
                delta,
                gravity_world: imu.gravity_world,
                weight_position: imu.weight_position,
                weight_velocity: imu.weight_velocity,
                weight_rotation: imu.weight_rotation,
            });
        }
        // Optional random-walk tie between consecutive bias slots.
        if let Some(rw_weight) = imu.bias_random_walk_weight {
            for i in 0..(n_frames - 1) {
                if imu_active[i] && imu_active[i + 1] {
                    ba.add_bias_random_walk_factor(BiasRandomWalkFactor {
                        keyframe_id_from: i as u64,
                        keyframe_id_to: (i + 1) as u64,
                        weight: rw_weight,
                    });
                }
            }
        }
        // Gauge anchors: optionally fix the very first IMU-active slot.
        if let Some(first_active) = imu_active.iter().position(|&a| a) {
            if imu.fix_first_bias {
                ba.fix_bias(first_active as u64);
            }
            if imu.fix_first_velocity {
                ba.fix_velocity(first_active as u64);
            }
        }
    }

    let mut observation_count = 0usize;
    let mut landmark_count = 0usize;
    let mut next_landmark_id: u64 = 0;

    let seed_row_threshold: Option<f64> = config
        .max_seed_row_fraction
        .map(|frac| (camera.height as f64) * frac);

    let intrinsics_for_residual: Option<(f64, f64, f64, f64)> =
        config.max_init_residual_px.and(camera.intrinsics());

    for track in &tracks {
        // Optional row-band filter: drop tracks whose first observation is
        // outside the configured upper-image band. Road-surface features
        // (lower image rows) genuinely violate rigid-body BA on slope
        // sequences (KITTI seq08) so removing them is a structural fix.
        if let Some(row_max) = seed_row_threshold {
            let (first_frame, first_idx) = track[0];
            let Some(first_kp) = left_features[first_frame].keypoints.get(first_idx) else {
                continue;
            };
            if first_kp.y >= row_max {
                continue;
            }
        }
        let init_world = match config.landmark_init {
            LandmarkInit::StereoSingleFrame => initial_landmark_world(
                track,
                &stereo_lookup,
                initial_poses,
                config.max_initial_depth_m,
            ),
            LandmarkInit::MultiViewDlt => initial_landmark_world_dlt(
                track,
                &stereo_lookup,
                left_features,
                right_features,
                initial_poses,
                camera,
                baseline,
                config.max_initial_depth_m,
            )
            .or_else(|| {
                // DLT can fail on degenerate geometry (collinear cameras +
                // far-field points). Fall back to the single-frame stereo
                // init so the track still gets a chance at BA.
                initial_landmark_world(
                    track,
                    &stereo_lookup,
                    initial_poses,
                    config.max_initial_depth_m,
                )
            }),
        };
        let Some(init_world) = init_world else {
            continue;
        };
        // Build the candidate observation list first; only commit if the
        // track has enough stereo-observed frames AND every per-observation
        // initial residual stays below the configured pre-BA gate.
        let mut candidate_obs: Vec<BaStereoObservation> = Vec::with_capacity(track.len());
        let landmark_id = next_landmark_id;
        let mut track_passes_residual_gate = true;
        for &(frame, left_idx) in track {
            let Some(left_kp) = left_features[frame].keypoints.get(left_idx) else {
                continue;
            };
            let Some(stereo) = stereo_lookup[frame].get(&left_idx) else {
                continue;
            };
            let Some(right_kp) = right_features[frame].keypoints.get(stereo.right_index) else {
                continue;
            };
            if let (Some(threshold), Some((fx, fy, cx, cy))) =
                (config.max_init_residual_px, intrinsics_for_residual)
            {
                // Predicted stereo projection of init_world from this frame's
                // initial pose. Reject the entire track if any observation
                // disagrees with the initial geometry by more than the gate.
                let pose = &initial_poses[frame];
                let xc = pose.transform_world_point(&init_world);
                if xc.z <= 0.0 {
                    track_passes_residual_gate = false;
                    break;
                }
                let u_l_pred = fx * xc.x / xc.z + cx;
                let v_pred = fy * xc.y / xc.z + cy;
                let u_r_pred = u_l_pred - fx * baseline / xc.z;
                let du_l = u_l_pred - left_kp.x;
                let dv = v_pred - left_kp.y;
                let du_r = u_r_pred - right_kp.x;
                let residual_norm = (du_l * du_l + dv * dv + du_r * du_r).sqrt();
                if !residual_norm.is_finite() || residual_norm > threshold {
                    track_passes_residual_gate = false;
                    break;
                }
            }
            candidate_obs.push(BaStereoObservation {
                keyframe_id: frame as u64,
                landmark_id,
                xy: Point2::new(left_kp.x, left_kp.y),
                u_right: right_kp.x,
            });
        }
        if !track_passes_residual_gate {
            continue;
        }
        if candidate_obs.len() < config.min_track_length {
            continue;
        }
        next_landmark_id += 1;
        ba.add_landmark(landmark_id, init_world);
        observation_count += candidate_obs.len();
        for obs in candidate_obs {
            ba.add_stereo_observation(obs);
        }
        landmark_count += 1;
    }

    if landmark_count == 0 {
        return Err(StereoVoBaError::NoLongTracks);
    }
    if let Some(required) = config.min_track_count {
        if landmark_count < required {
            return Err(StereoVoBaError::InsufficientTracks {
                count: landmark_count,
                required,
            });
        }
    }

    let ba_result = ba
        .optimize(&config.ba_config)
        .map_err(StereoVoBaError::Ba)?;

    let refined_poses: Vec<Pose> = (0..n_frames as u64)
        .map(|i| {
            ba.poses
                .get(&i)
                .cloned()
                .unwrap_or_else(|| initial_poses[i as usize].clone())
        })
        .collect();

    let imu_refinement = config.imu_input.as_ref().map(|imu| {
        let mut velocities = vec![Vector3::zeros(); n_frames];
        let mut bias_g = vec![imu.bias_gyro_init; n_frames];
        let mut bias_a = vec![imu.bias_acc_init; n_frames];
        for i in 0..n_frames as u64 {
            if let Some(v) = ba.velocities.get(&i) {
                velocities[i as usize] = *v;
            }
            if let Some(b) = ba.biases.get(&i) {
                bias_g[i as usize] = b.fixed_rows::<3>(0).into();
                bias_a[i as usize] = b.fixed_rows::<3>(3).into();
            }
        }
        StereoVoBaImuRefinement {
            refined_velocities: velocities,
            refined_bias_gyro: bias_g,
            refined_bias_acc: bias_a,
        }
    });

    Ok(StereoVoBaRefinement {
        refined_poses,
        ba_result,
        track_count: landmark_count,
        observation_count,
        imu_refinement,
    })
}

/// A left-image observation track: `(frame, left_keypoint_index, pixel)` per view.
type TrackObservations = Vec<(usize, usize, Point2<f64>)>;

/// A single reconstructed 3D point with its full multi-view track.
///
/// Unlike the per-frame stereo lift used by the streaming VO (one landmark per
/// frame, observed once), each `ReconstructedLandmark` is a *merged* track: the
/// same physical point seen by every frame the forward-track chaining linked it
/// to. This is the multi-view constraint that makes a global bundle adjustment
/// tighten reprojection to sub-pixel — i.e. COLMAP-grade structure suitable for
/// downstream novel-view synthesis (3DGS).
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructedLandmark {
    /// Refined world-frame position after BA.
    pub position: Point3<f64>,
    /// Every left-image observation of this point: `(frame, left_keypoint_index, pixel)`.
    pub observations: TrackObservations,
}

/// Output of [`reconstruct_stereo_vo_with_ba`]: a BA-grade sparse reconstruction
/// (refined poses + merged multi-view landmark tracks) ready for COLMAP export.
#[derive(Debug, Clone)]
pub struct StereoVoReconstruction {
    /// Refined `world_to_camera` pose per frame (pose 0 fixed as the anchor).
    pub refined_poses: Vec<Pose>,
    /// Merged multi-view landmark tracks (each observed by ≥ `min_track_length` frames).
    pub landmarks: Vec<ReconstructedLandmark>,
    /// LM iteration trace and final cost from the BA solve.
    pub ba_result: BaResult,
    /// Total left-image observations across all landmark tracks.
    pub observation_count: usize,
    /// Mean left-image reprojection error (px) before BA, over all observations.
    pub mean_reproj_before_px: f64,
    /// Mean left-image reprojection error (px) after BA — the structure-quality headline.
    pub mean_reproj_after_px: f64,
}

/// Build a COLMAP-grade sparse reconstruction from a stereo VO trajectory.
///
/// Where [`refine_stereo_vo_with_ba`] runs the same global bundle adjustment but
/// only returns the refined *poses* (discarding the structure), this entry point
/// keeps the merged multi-view landmark tracks so they can be written out as a
/// COLMAP model — every 3D point carries the full `TRACK[]` of frames that see
/// it, which is exactly what a per-frame stereo lift lacks and what a downstream
/// 3DGS optimizer needs to converge crisply.
///
/// Visual-only and global (no IMU, no sliding window): novel-view synthesis
/// wants one joint solve over every pose and point. Metric scale is anchored by
/// the rectified-stereo baseline; pose 0 is fixed as the gauge.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_stereo_vo_with_ba(
    camera: &Camera,
    baseline: f64,
    initial_poses: &[Pose],
    left_features: &[FeatureSet],
    right_features: &[FeatureSet],
    stereo_per_frame: &[Vec<StereoFeature>],
    temporal_matches: &[Vec<DescriptorMatch>],
    config: &StereoVoBaConfig,
) -> Result<StereoVoReconstruction, StereoVoBaError> {
    let n_frames = initial_poses.len();
    if n_frames < config.min_track_length || n_frames < 2 {
        return Err(StereoVoBaError::TooFewFrames(n_frames));
    }
    if left_features.len() != n_frames
        || right_features.len() != n_frames
        || stereo_per_frame.len() != n_frames
        || temporal_matches.len() != n_frames - 1
    {
        return Err(StereoVoBaError::InputLengthMismatch);
    }

    let stereo_lookup: Vec<HashMap<usize, &StereoFeature>> = stereo_per_frame
        .iter()
        .map(|stereo| stereo.iter().map(|f| (f.left_index, f)).collect())
        .collect();
    let temporal_lookup: Vec<HashMap<usize, usize>> = temporal_matches
        .iter()
        .map(|matches| {
            matches
                .iter()
                .filter(|m| match config.min_temporal_confidence {
                    Some(min) => m
                        .confidence
                        .map(|c| c.is_finite() && c >= min)
                        .unwrap_or(false),
                    None => true,
                })
                .map(|m| (m.query_index, m.train_index))
                .collect()
        })
        .collect();

    let tracks = build_forward_tracks(
        n_frames,
        left_features,
        &temporal_lookup,
        config.min_track_length,
    );
    if tracks.is_empty() {
        return Err(StereoVoBaError::NoLongTracks);
    }

    let mut ba = BundleAdjustment::new(camera.clone());
    ba.set_stereo_baseline(baseline);
    for (i, pose) in initial_poses.iter().enumerate() {
        ba.add_pose(i as u64, pose.clone());
    }
    ba.fix_pose(0);

    let seed_row_threshold: Option<f64> = config
        .max_seed_row_fraction
        .map(|frac| (camera.height as f64) * frac);
    let intrinsics_for_residual: Option<(f64, f64, f64, f64)> =
        config.max_init_residual_px.and(camera.intrinsics());

    // Per accepted landmark, remember its left-image observations so we can
    // emit the merged track once BA has refined the point.
    let mut landmark_obs: Vec<(u64, TrackObservations)> = Vec::new();
    let mut observation_count = 0usize;
    let mut next_landmark_id: u64 = 0;

    for track in &tracks {
        if let Some(row_max) = seed_row_threshold {
            let (first_frame, first_idx) = track[0];
            let Some(first_kp) = left_features[first_frame].keypoints.get(first_idx) else {
                continue;
            };
            if first_kp.y >= row_max {
                continue;
            }
        }
        let init_world = match config.landmark_init {
            LandmarkInit::StereoSingleFrame => initial_landmark_world(
                track,
                &stereo_lookup,
                initial_poses,
                config.max_initial_depth_m,
            ),
            LandmarkInit::MultiViewDlt => initial_landmark_world_dlt(
                track,
                &stereo_lookup,
                left_features,
                right_features,
                initial_poses,
                camera,
                baseline,
                config.max_initial_depth_m,
            )
            .or_else(|| {
                initial_landmark_world(
                    track,
                    &stereo_lookup,
                    initial_poses,
                    config.max_initial_depth_m,
                )
            }),
        };
        let Some(init_world) = init_world else {
            continue;
        };

        let landmark_id = next_landmark_id;
        let mut candidate_obs: Vec<BaStereoObservation> = Vec::with_capacity(track.len());
        let mut track_obs: TrackObservations = Vec::with_capacity(track.len());
        let mut track_passes_residual_gate = true;
        for &(frame, left_idx) in track {
            let Some(left_kp) = left_features[frame].keypoints.get(left_idx) else {
                continue;
            };
            let Some(stereo) = stereo_lookup[frame].get(&left_idx) else {
                continue;
            };
            let Some(right_kp) = right_features[frame].keypoints.get(stereo.right_index) else {
                continue;
            };
            if let (Some(threshold), Some((fx, fy, cx, cy))) =
                (config.max_init_residual_px, intrinsics_for_residual)
            {
                let pose = &initial_poses[frame];
                let xc = pose.transform_world_point(&init_world);
                if xc.z <= 0.0 {
                    track_passes_residual_gate = false;
                    break;
                }
                let u_l_pred = fx * xc.x / xc.z + cx;
                let v_pred = fy * xc.y / xc.z + cy;
                let u_r_pred = u_l_pred - fx * baseline / xc.z;
                let du_l = u_l_pred - left_kp.x;
                let dv = v_pred - left_kp.y;
                let du_r = u_r_pred - right_kp.x;
                let residual_norm = (du_l * du_l + dv * dv + du_r * du_r).sqrt();
                if !residual_norm.is_finite() || residual_norm > threshold {
                    track_passes_residual_gate = false;
                    break;
                }
            }
            let xy = Point2::new(left_kp.x, left_kp.y);
            candidate_obs.push(BaStereoObservation {
                keyframe_id: frame as u64,
                landmark_id,
                xy,
                u_right: right_kp.x,
            });
            track_obs.push((frame, left_idx, xy));
        }
        if !track_passes_residual_gate || candidate_obs.len() < config.min_track_length {
            continue;
        }
        next_landmark_id += 1;
        ba.add_landmark(landmark_id, init_world);
        observation_count += candidate_obs.len();
        for obs in candidate_obs {
            ba.add_stereo_observation(obs);
        }
        landmark_obs.push((landmark_id, track_obs));
    }

    if landmark_obs.is_empty() {
        return Err(StereoVoBaError::NoLongTracks);
    }
    if let Some(required) = config.min_track_count {
        if landmark_obs.len() < required {
            return Err(StereoVoBaError::InsufficientTracks {
                count: landmark_obs.len(),
                required,
            });
        }
    }

    let mean_reproj_before_px = mean_left_reprojection_px(&ba);
    let ba_result = ba
        .optimize(&config.ba_config)
        .map_err(StereoVoBaError::Ba)?;
    let mean_reproj_after_px = mean_left_reprojection_px(&ba);

    let refined_poses: Vec<Pose> = (0..n_frames as u64)
        .map(|i| {
            ba.poses
                .get(&i)
                .cloned()
                .unwrap_or_else(|| initial_poses[i as usize].clone())
        })
        .collect();

    let landmarks: Vec<ReconstructedLandmark> = landmark_obs
        .into_iter()
        .filter_map(|(id, observations)| {
            ba.landmarks.get(&id).map(|p| ReconstructedLandmark {
                position: *p,
                observations,
            })
        })
        .collect();

    Ok(StereoVoReconstruction {
        refined_poses,
        landmarks,
        ba_result,
        observation_count,
        mean_reproj_before_px,
        mean_reproj_after_px,
    })
}

/// Mean left-image reprojection error (px) over every stereo observation in a
/// BA problem, using the current pose/landmark estimates. The right-image term
/// is ignored so the figure is directly comparable to a monocular reprojection.
fn mean_left_reprojection_px(ba: &BundleAdjustment) -> f64 {
    let Some((fx, fy, cx, cy)) = ba.camera.intrinsics() else {
        return f64::NAN;
    };
    let mut sum = 0.0;
    let mut count = 0usize;
    for obs in &ba.stereo_observations {
        let (Some(pose), Some(point)) = (
            ba.poses.get(&obs.keyframe_id),
            ba.landmarks.get(&obs.landmark_id),
        ) else {
            continue;
        };
        let xc = pose.transform_world_point(point);
        if xc.z <= 0.0 {
            continue;
        }
        let u = fx * xc.x / xc.z + cx;
        let v = fy * xc.y / xc.z + cy;
        let du = u - obs.xy.x;
        let dv = v - obs.xy.y;
        sum += (du * du + dv * dv).sqrt();
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn slice_position_prior_for_window(
    prior: &PositionPrior,
    start: usize,
    end: usize,
) -> PositionPrior {
    let mut out = PositionPrior::new();
    for obs in &prior.observations {
        let frame = obs.keyframe_id as usize;
        if frame >= start && frame < end {
            let mut local = obs.clone();
            local.keyframe_id = (frame - start) as u64;
            out.push(local);
        }
    }
    out
}

fn slice_per_pose_gravity_prior_for_window(
    prior: &PerPoseGravityPrior,
    start: usize,
    end: usize,
) -> PerPoseGravityPrior {
    let mut out = PerPoseGravityPrior::new(prior.g_world, prior.weight);
    for obs in &prior.observations {
        let frame = obs.keyframe_id as usize;
        if frame >= start && frame < end {
            let mut local = obs.clone();
            local.keyframe_id = (frame - start) as u64;
            out.push(local);
        }
    }
    out
}

fn build_forward_tracks(
    n_frames: usize,
    left_features: &[FeatureSet],
    temporal_lookup: &[HashMap<usize, usize>],
    min_track_length: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut visited: HashMap<(usize, usize), ()> = HashMap::new();
    let mut tracks: Vec<Vec<(usize, usize)>> = Vec::new();
    for (start_frame, start_features) in left_features.iter().enumerate().take(n_frames) {
        for start_idx in 0..start_features.keypoints.len() {
            if visited.contains_key(&(start_frame, start_idx)) {
                continue;
            }
            let mut track = vec![(start_frame, start_idx)];
            visited.insert((start_frame, start_idx), ());
            let mut curr_frame = start_frame;
            let mut curr_idx = start_idx;
            while curr_frame + 1 < n_frames {
                let Some(map) = temporal_lookup.get(curr_frame) else {
                    break;
                };
                let Some(&next_idx) = map.get(&curr_idx) else {
                    break;
                };
                if visited.contains_key(&(curr_frame + 1, next_idx)) {
                    break;
                }
                track.push((curr_frame + 1, next_idx));
                visited.insert((curr_frame + 1, next_idx), ());
                curr_frame += 1;
                curr_idx = next_idx;
            }
            if track.len() >= min_track_length {
                tracks.push(track);
            }
        }
    }
    tracks
}

fn initial_landmark_world(
    track: &[(usize, usize)],
    stereo_lookup: &[HashMap<usize, &StereoFeature>],
    initial_poses: &[Pose],
    max_initial_depth_m: f64,
) -> Option<Point3<f64>> {
    for &(frame, left_idx) in track {
        let Some(stereo) = stereo_lookup[frame].get(&left_idx) else {
            continue;
        };
        if !stereo.point_cam.coords.iter().all(|v| v.is_finite()) {
            continue;
        }
        if stereo.point_cam.z <= 0.0 || stereo.point_cam.z > max_initial_depth_m {
            continue;
        }
        let pose = &initial_poses[frame];
        let world = pose.camera_to_world().transform_point(&stereo.point_cam);
        return Some(world);
    }
    None
}

/// Linear DLT triangulation over all stereo observations in a track. Each
/// stereo-observed frame contributes 3 rows to `A * [X, 1]^T = 0`: two from
/// the left-image projection equations and one from the right-image `u_r`
/// (the rectified-stereo `v_r = v_l` row is redundant with the left row).
///
/// Returns `None` when the track has fewer than 2 stereo-observed frames
/// (DLT is under-constrained), the SVD fails, or the recovered point fails
/// cheirality / depth gates at the first observation's frame.
// Per-track DLT triangulation inputs: the track plus the shared BA frame data
// (stereo lookup, features, poses, rig). Passed positionally for one call site.
#[allow(clippy::too_many_arguments)]
fn initial_landmark_world_dlt(
    track: &[(usize, usize)],
    stereo_lookup: &[HashMap<usize, &StereoFeature>],
    left_features: &[FeatureSet],
    right_features: &[FeatureSet],
    initial_poses: &[Pose],
    camera: &Camera,
    baseline: f64,
    max_initial_depth_m: f64,
) -> Option<Point3<f64>> {
    let (fx, fy, cx, cy) = camera.intrinsics()?;
    let mut rows: Vec<[f64; 4]> = Vec::with_capacity(track.len() * 3);
    let mut first_pose_index: Option<usize> = None;
    for &(frame, left_idx) in track {
        let Some(left_kp) = left_features[frame].keypoints.get(left_idx) else {
            continue;
        };
        let Some(stereo) = stereo_lookup[frame].get(&left_idx) else {
            continue;
        };
        let Some(right_kp) = right_features[frame].keypoints.get(stereo.right_index) else {
            continue;
        };
        if first_pose_index.is_none() {
            first_pose_index = Some(frame);
        }
        let pose = &initial_poses[frame];
        let r = pose.world_to_camera.rotation.to_rotation_matrix();
        let r = r.matrix();
        let t = pose.world_to_camera.translation;
        // Left projection matrix rows: P_left = K_left * [R | t]
        let p0 = [
            fx * r[(0, 0)] + cx * r[(2, 0)],
            fx * r[(0, 1)] + cx * r[(2, 1)],
            fx * r[(0, 2)] + cx * r[(2, 2)],
            fx * t.x + cx * t.z,
        ];
        let p1 = [
            fy * r[(1, 0)] + cy * r[(2, 0)],
            fy * r[(1, 1)] + cy * r[(2, 1)],
            fy * r[(1, 2)] + cy * r[(2, 2)],
            fy * t.y + cy * t.z,
        ];
        let p2 = [r[(2, 0)], r[(2, 1)], r[(2, 2)], t.z];
        // u_l row: u_l * p2 - p0 = 0
        rows.push([
            left_kp.x * p2[0] - p0[0],
            left_kp.x * p2[1] - p0[1],
            left_kp.x * p2[2] - p0[2],
            left_kp.x * p2[3] - p0[3],
        ]);
        // v_l row: v_l * p2 - p1 = 0
        rows.push([
            left_kp.y * p2[0] - p1[0],
            left_kp.y * p2[1] - p1[1],
            left_kp.y * p2[2] - p1[2],
            left_kp.y * p2[3] - p1[3],
        ]);
        // Right cam shares R; its origin is at +baseline along the left cam's
        // x-axis. Thus the right-cam world_to_camera translation is
        // `t - [b, 0, 0]^T` in the left-cam frame. The right `u_r` projection
        // row uses the same K but the shifted translation:
        //   u_r * (p2_right - p0_right) = 0
        // where p2_right == p2 (z-row unchanged), and the right p0 has
        // `t_right_x = t.x - baseline` in column 3.
        let p0_right_col3 = fx * (t.x - baseline) + cx * t.z;
        let p0_right = [p0[0], p0[1], p0[2], p0_right_col3];
        rows.push([
            right_kp.x * p2[0] - p0_right[0],
            right_kp.x * p2[1] - p0_right[1],
            right_kp.x * p2[2] - p0_right[2],
            right_kp.x * p2[3] - p0_right[3],
        ]);
    }
    if rows.len() < 6 {
        // Need at least 2 stereo-observed frames (6 rows) for an
        // over-determined 4-unknown linear system.
        return None;
    }
    let first_frame = first_pose_index?;
    let n = rows.len();
    let mut a = DMatrix::<f64>::zeros(n, 4);
    for (i, row) in rows.iter().enumerate() {
        for j in 0..4 {
            a[(i, j)] = row[j];
        }
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    // Right singular vector for the smallest singular value (= last row of V^T).
    let solution = v_t.row(3);
    let w = solution[3];
    if !w.is_finite() || w.abs() < 1e-12 {
        return None;
    }
    let world = Point3::new(solution[0] / w, solution[1] / w, solution[2] / w);
    if !world.coords.iter().all(|v| v.is_finite()) {
        return None;
    }
    // Cheirality + depth gate at the first observation's frame.
    let p_cam = initial_poses[first_frame].transform_world_point(&world);
    if !p_cam.z.is_finite() || p_cam.z <= 0.0 || p_cam.z > max_initial_depth_m {
        return None;
    }
    Some(world)
}

#[derive(Debug, Clone, PartialEq)]
pub enum StereoVoBaError {
    /// `initial_poses.len()` is less than the minimum required.
    TooFewFrames(usize),
    /// Per-frame slices have inconsistent lengths.
    InputLengthMismatch,
    /// No track of length `>= min_track_length` could be built from the
    /// supplied temporal matches.
    NoLongTracks,
    /// `config.min_track_count` was set and the filtered track set was
    /// below the gate. `count` is what survived; `required` is the gate.
    InsufficientTracks { count: usize, required: usize },
    /// `config.imu_input` was malformed (wrong number of windows, a
    /// non-positive sample dt, or incompatible with `window_size`).
    InvalidImuInput { reason: String },
    /// Underlying BA error.
    Ba(BaError),
}

impl std::fmt::Display for StereoVoBaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StereoVoBaError::TooFewFrames(n) => {
                write!(
                    f,
                    "stereo VO BA: only {n} frames, need at least min_track_length"
                )
            }
            StereoVoBaError::InputLengthMismatch => {
                write!(
                    f,
                    "stereo VO BA: per-frame input slices have inconsistent lengths"
                )
            }
            StereoVoBaError::NoLongTracks => {
                write!(
                    f,
                    "stereo VO BA: no tracks of sufficient length built from temporal matches"
                )
            }
            StereoVoBaError::InsufficientTracks { count, required } => write!(
                f,
                "stereo VO BA: only {count} tracks survived filters, below min_track_count={required}"
            ),
            StereoVoBaError::InvalidImuInput { reason } => {
                write!(f, "stereo VO BA: invalid imu_input — {reason}")
            }
            StereoVoBaError::Ba(err) => write!(f, "stereo VO BA: {err}"),
        }
    }
}

impl std::error::Error for StereoVoBaError {}

/// Parse a plain-text per-window IMU sample file.
///
/// Each non-empty, non-comment line carries seven whitespace-separated
/// numbers `dt gyro_x gyro_y gyro_z accel_x accel_y accel_z`. Lines
/// starting with `#` are treated as comments. The returned samples
/// preserve file order so callers can hand them straight to
/// [`ImuPreintegrator::integrate_sample`] without resorting on time.
///
/// `dt` must be a positive finite number; a non-positive `dt` aborts
/// the parse with a descriptive error so callers don't silently feed
/// the BA an unintegrable window.
pub fn parse_stereo_vo_imu_samples_txt(text: &str) -> Result<Vec<StereoVoBaImuSample>, String> {
    let mut out = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let nums: Vec<f64> = trimmed
            .split_whitespace()
            .map(|tok| {
                tok.parse::<f64>()
                    .map_err(|e| format!("line {}: cannot parse '{tok}': {e}", line_no + 1))
            })
            .collect::<Result<_, _>>()?;
        if nums.len() != 7 {
            return Err(format!(
                "line {}: expected 7 numbers (dt gyro accel), got {}",
                line_no + 1,
                nums.len()
            ));
        }
        let dt = nums[0];
        if !dt.is_finite() || dt <= 0.0 {
            return Err(format!(
                "line {}: dt must be a positive finite number, got {dt}",
                line_no + 1
            ));
        }
        out.push(StereoVoBaImuSample {
            dt,
            gyro: Vector3::new(nums[1], nums[2], nums[3]),
            accel: Vector3::new(nums[4], nums[5], nums[6]),
        });
    }
    Ok(out)
}

/// Slice a globally-timestamped IMU stream into per-keyframe pre-integration
/// windows that match the [`StereoVoBaImuInput::windows`] layout.
///
/// Given monotonically non-decreasing IMU timestamps `imu_timestamps_ns` and
/// the matching gyro / accel readings (same length), plus keyframe timestamps
/// `keyframe_timestamps_ns` (length `K >= 2`), the function returns a vector
/// of `K - 1` windows. Window `i` collects every IMU sample whose timestamp
/// `t` satisfies `kf[i] < t <= kf[i+1]`, with each sample's `dt` equal to the
/// gap from the previous timestamp in the window (the first sample is anchored
/// to `kf[i]`). If the last sample in the window does not reach `kf[i+1]`, a
/// trailing zero-order-hold step is appended using the last sample's
/// gyro/accel so the integrated interval matches the inter-keyframe duration.
///
/// Windows with no IMU coverage are returned empty, which signals
/// [`refine_stereo_vo_with_ba`] to skip wiring an IMU factor for that segment.
pub fn slice_imu_samples_for_keyframes(
    imu_timestamps_ns: &[i128],
    imu_gyro: &[Vector3<f64>],
    imu_accel: &[Vector3<f64>],
    keyframe_timestamps_ns: &[i128],
) -> Result<Vec<Vec<StereoVoBaImuSample>>, String> {
    if imu_gyro.len() != imu_timestamps_ns.len() {
        return Err(format!(
            "IMU timestamp / gyro length mismatch: {} vs {}",
            imu_timestamps_ns.len(),
            imu_gyro.len()
        ));
    }
    if imu_accel.len() != imu_timestamps_ns.len() {
        return Err(format!(
            "IMU timestamp / accel length mismatch: {} vs {}",
            imu_timestamps_ns.len(),
            imu_accel.len()
        ));
    }
    if keyframe_timestamps_ns.len() < 2 {
        return Err(format!(
            "need at least 2 keyframe timestamps, got {}",
            keyframe_timestamps_ns.len()
        ));
    }
    for (idx, pair) in keyframe_timestamps_ns.windows(2).enumerate() {
        if pair[1] < pair[0] {
            return Err(format!(
                "keyframe timestamps are not monotonically non-decreasing at index {idx}"
            ));
        }
    }
    for (idx, pair) in imu_timestamps_ns.windows(2).enumerate() {
        if pair[1] < pair[0] {
            return Err(format!(
                "IMU timestamps are not monotonically non-decreasing at index {idx}"
            ));
        }
    }

    let mut windows: Vec<Vec<StereoVoBaImuSample>> =
        Vec::with_capacity(keyframe_timestamps_ns.len() - 1);
    let mut cursor = 0usize;
    for pair in keyframe_timestamps_ns.windows(2) {
        let t_a = pair[0];
        let t_b = pair[1];
        let mut window: Vec<StereoVoBaImuSample> = Vec::new();
        let mut prev_t = t_a;

        while cursor < imu_timestamps_ns.len() && imu_timestamps_ns[cursor] <= t_a {
            cursor += 1;
        }
        while cursor < imu_timestamps_ns.len() && imu_timestamps_ns[cursor] <= t_b {
            let t = imu_timestamps_ns[cursor];
            let dt = (t - prev_t) as f64 * 1e-9;
            if dt > 0.0 {
                window.push(StereoVoBaImuSample {
                    dt,
                    gyro: imu_gyro[cursor],
                    accel: imu_accel[cursor],
                });
            }
            prev_t = t;
            cursor += 1;
        }
        if prev_t < t_b {
            if let Some(last) = window.last() {
                let tail_dt = (t_b - prev_t) as f64 * 1e-9;
                if tail_dt > 0.0 {
                    let tail = StereoVoBaImuSample {
                        dt: tail_dt,
                        gyro: last.gyro,
                        accel: last.accel,
                    };
                    window.push(tail);
                }
            }
        }
        windows.push(window);
    }

    Ok(windows)
}

#[cfg(test)]
// Tests build large `*Config` fixtures by tweaking a handful of fields off
// `Default::default()`; field-by-field assignment reads more clearly than a
// struct-update literal here and keeps each gate's relevant knobs together.
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::SE3;
    use visloc_vision::features::FeatureSet;

    fn kitti_camera() -> Camera {
        Camera::pinhole(0, 1241, 376, 718.856, 718.856, 607.193, 185.216)
    }

    fn synthetic_landmark_grid() -> Vec<Point3<f64>> {
        // 12 stationary landmarks spread in front of the cameras at depths 8..30 m
        // and across the image (X in [-4, 4], Y in [-2, 2]).
        let mut out = Vec::new();
        for &x in &[-4.0, -2.0, 0.0, 2.0, 4.0] {
            for &y in &[-2.0, 0.0, 1.5] {
                for &z in &[10.0, 20.0] {
                    out.push(Point3::new(x, y, z));
                }
            }
        }
        out
    }

    fn project_to_pixels(
        camera: &Camera,
        pose: &Pose,
        landmarks: &[Point3<f64>],
        baseline: f64,
    ) -> (FeatureSet, FeatureSet, Vec<StereoFeature>) {
        let (fx, fy, cx, cy) = camera.intrinsics().unwrap();
        let mut left_kp = Vec::new();
        let mut right_kp = Vec::new();
        let mut stereo = Vec::new();
        for (lm_idx, lm) in landmarks.iter().enumerate() {
            let p_cam = pose.transform_world_point(lm);
            if p_cam.z <= 0.5 {
                continue;
            }
            let u_l = fx * p_cam.x / p_cam.z + cx;
            let v = fy * p_cam.y / p_cam.z + cy;
            let u_r = u_l - fx * baseline / p_cam.z;
            let left_index = left_kp.len();
            let right_index = right_kp.len();
            left_kp.push(nalgebra::Point2::new(u_l, v));
            right_kp.push(nalgebra::Point2::new(u_r, v));
            stereo.push(StereoFeature {
                left_index,
                right_index,
                disparity: (u_l - u_r),
                point_cam: p_cam,
            });
            // Use lm_idx to make a unique descriptor so temporal matching is
            // straightforward (we don't actually use descriptors in BA tests).
            let _ = lm_idx;
        }
        let left_features = FeatureSet {
            keypoints: left_kp,
            descriptors: Vec::new(),
        };
        let right_features = FeatureSet {
            keypoints: right_kp,
            descriptors: Vec::new(),
        };
        (left_features, right_features, stereo)
    }

    /// Multi-view DLT landmark init should produce a near-perfect 3D point
    /// when the input observations are noise-free synthetic projections.
    #[test]
    fn multi_view_dlt_recovers_synthetic_landmark() {
        use std::collections::HashMap;

        let camera = kitti_camera();
        let baseline = 0.537;
        // Single landmark in front of the cameras.
        let truth = Point3::new(1.0, 0.5, 25.0);
        let poses = vec![
            Pose::identity(),
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -1.0)),
            },
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -2.0)),
            },
        ];
        let mut left_features = Vec::new();
        let mut right_features = Vec::new();
        let mut stereo_per_frame: Vec<Vec<StereoFeature>> = Vec::new();
        let (fx, fy, cx, cy) = camera.intrinsics().unwrap();
        for p in &poses {
            let cam = p.transform_world_point(&truth);
            let u_l = fx * cam.x / cam.z + cx;
            let v = fy * cam.y / cam.z + cy;
            let u_r = u_l - fx * baseline / cam.z;
            let left = FeatureSet {
                keypoints: vec![nalgebra::Point2::new(u_l, v)],
                descriptors: Vec::new(),
            };
            let right = FeatureSet {
                keypoints: vec![nalgebra::Point2::new(u_r, v)],
                descriptors: Vec::new(),
            };
            let stereo = StereoFeature {
                left_index: 0,
                right_index: 0,
                disparity: u_l - u_r,
                point_cam: cam,
            };
            left_features.push(left);
            right_features.push(right);
            stereo_per_frame.push(vec![stereo]);
        }
        let track: Vec<(usize, usize)> = vec![(0, 0), (1, 0), (2, 0)];
        let stereo_lookup: Vec<HashMap<usize, &StereoFeature>> = stereo_per_frame
            .iter()
            .map(|stereo| stereo.iter().map(|f| (f.left_index, f)).collect())
            .collect();
        let recovered = initial_landmark_world_dlt(
            &track,
            &stereo_lookup,
            &left_features,
            &right_features,
            &poses,
            &camera,
            baseline,
            80.0,
        )
        .expect("DLT should succeed on noise-free synthetic data");
        let err = (recovered - truth).norm();
        assert!(
            err < 1e-6,
            "multi-view DLT recovered point off by {err} for truth {truth:?}, got {recovered:?}"
        );
    }

    /// On a synthetic, stationary scene the refined poses should match the
    /// initial (correct) poses to within solver tolerance.
    #[test]
    fn ba_keeps_correct_trajectory_stable() {
        let camera = kitti_camera();
        let baseline = 0.537;
        let landmarks = synthetic_landmark_grid();

        // Three frames: forward motion only.
        let poses = vec![
            Pose::identity(),
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -0.8)),
            },
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -1.6)),
            },
        ];

        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut stereo = Vec::new();
        for p in &poses {
            let (l, r, s) = project_to_pixels(&camera, p, &landmarks, baseline);
            left.push(l);
            right.push(r);
            stereo.push(s);
        }

        // Identity temporal matches: each frame's left feature i maps to the
        // next frame's left feature i (synthetic scene, no missing matches).
        let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..poses.len() - 1)
            .map(|f| {
                let n = left[f].keypoints.len().min(left[f + 1].keypoints.len());
                (0..n)
                    .map(|i| DescriptorMatch {
                        query_index: i,
                        train_index: i,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: Some(1.0),
                    })
                    .collect()
            })
            .collect();

        let config = StereoVoBaConfig::default();
        let refinement = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &poses,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &config,
        )
        .expect("BA should succeed on synthetic scene");

        assert!(refinement.track_count > 0);
        assert!(refinement.observation_count > 0);
        // Pose 0 is fixed.
        assert!(
            (refinement.refined_poses[0].world_to_camera.translation
                - poses[0].world_to_camera.translation)
                .norm()
                < 1e-9
        );
        // Other poses should remain close to the (correct) input.
        for (idx, (refined, original)) in refinement
            .refined_poses
            .iter()
            .zip(poses.iter())
            .enumerate()
        {
            let dt =
                (refined.world_to_camera.translation - original.world_to_camera.translation).norm();
            assert!(
                dt < 1e-3,
                "frame {idx} drifted {dt} from a known-correct initialisation"
            );
        }
    }

    /// `reconstruct_stereo_vo_with_ba` returns merged multi-view tracks (each
    /// landmark observed by every frame it was chained through) and a global BA
    /// that does not worsen reprojection. On a noise-free synthetic scene the
    /// recovered landmark positions match truth and the tracks span all frames.
    #[test]
    fn reconstruct_emits_multiview_tracks_and_does_not_worsen_reprojection() {
        let camera = kitti_camera();
        let baseline = 0.537;
        let landmarks = synthetic_landmark_grid();

        // Four frames of forward motion (truth).
        let poses: Vec<Pose> = (0..4)
            .map(|i| Pose {
                world_to_camera: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(0.0, 0.0, -0.8 * i as f64),
                ),
            })
            .collect();

        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut stereo = Vec::new();
        for p in &poses {
            let (l, r, s) = project_to_pixels(&camera, p, &landmarks, baseline);
            left.push(l);
            right.push(r);
            stereo.push(s);
        }

        // Identity temporal matches: feature i → feature i in the next frame.
        let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..poses.len() - 1)
            .map(|f| {
                let n = left[f].keypoints.len().min(left[f + 1].keypoints.len());
                (0..n)
                    .map(|i| DescriptorMatch {
                        query_index: i,
                        train_index: i,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: Some(1.0),
                    })
                    .collect()
            })
            .collect();

        let config = StereoVoBaConfig::default();
        let recon = reconstruct_stereo_vo_with_ba(
            &camera,
            baseline,
            &poses,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &config,
        )
        .expect("reconstruction should succeed on synthetic scene");

        assert!(!recon.landmarks.is_empty(), "expected merged landmarks");
        assert_eq!(recon.refined_poses.len(), poses.len());

        // The defining property vs the per-frame writer: each landmark carries a
        // multi-view track. With identity matches every grid point is seen by all
        // 4 frames, so at least one landmark must span > 1 frame, and total
        // observations must exceed the landmark count.
        let max_track = recon
            .landmarks
            .iter()
            .map(|l| l.observations.len())
            .max()
            .unwrap();
        assert!(
            max_track >= 3,
            "expected a multi-view track spanning ≥3 frames, got {max_track}"
        );
        assert!(recon.observation_count > recon.landmarks.len());

        // Observations within a track must reference distinct frames in order.
        for l in &recon.landmarks {
            for w in l.observations.windows(2) {
                assert!(w[0].0 < w[1].0, "track frames must be strictly increasing");
            }
        }

        // Global BA must not worsen reprojection on a consistent scene.
        assert!(
            recon.mean_reproj_after_px <= recon.mean_reproj_before_px + 1e-6,
            "BA worsened reprojection: {} -> {}",
            recon.mean_reproj_before_px,
            recon.mean_reproj_after_px,
        );

        // Recovered landmark positions should match one of the truth grid points.
        for l in &recon.landmarks {
            let nearest = landmarks
                .iter()
                .map(|t| (t - l.position).norm())
                .fold(f64::INFINITY, f64::min);
            assert!(
                nearest < 1e-2,
                "reconstructed landmark {:?} not near any truth point (nearest {nearest})",
                l.position,
            );
        }
    }

    /// `fix_pose_prefix > 1` holds a leading prefix of poses fixed (the
    /// fixed-keyframe local-map anchor) while still optimising the trailing
    /// poses: the prefix is bit-for-bit unchanged, and a drifted trailing pose is
    /// pulled back toward truth by the long-baseline landmarks the fixed prefix
    /// anchors.
    #[test]
    fn fix_pose_prefix_anchors_leading_poses() {
        let camera = kitti_camera();
        let baseline = 0.537;
        let landmarks = synthetic_landmark_grid();

        // Five frames of forward motion (truth).
        let truth: Vec<Pose> = (0..5)
            .map(|i| Pose {
                world_to_camera: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(0.0, 0.0, -0.8 * i as f64),
                ),
            })
            .collect();

        // Observations are projected from the TRUE poses.
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut stereo = Vec::new();
        for p in &truth {
            let (l, r, s) = project_to_pixels(&camera, p, &landmarks, baseline);
            left.push(l);
            right.push(r);
            stereo.push(s);
        }
        let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..truth.len() - 1)
            .map(|f| {
                let n = left[f].keypoints.len().min(left[f + 1].keypoints.len());
                (0..n)
                    .map(|i| DescriptorMatch {
                        query_index: i,
                        train_index: i,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: Some(1.0),
                    })
                    .collect()
            })
            .collect();

        // Initial poses: truth, but the trailing two are drifted.
        let mut initial = truth.clone();
        for f in [3usize, 4] {
            initial[f] = Pose {
                world_to_camera: SE3::new(
                    UnitQuaternion::identity(),
                    truth[f].world_to_camera.translation + Vector3::new(0.06, 0.04, 0.05),
                ),
            };
        }

        let config = StereoVoBaConfig {
            fix_pose_prefix: 3,
            ..StereoVoBaConfig::default()
        };
        let refinement = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &initial,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &config,
        )
        .expect("BA should succeed");

        // The fixed prefix (poses 0,1,2) is bit-for-bit unchanged.
        for f in 0..3 {
            let d = (refinement.refined_poses[f].world_to_camera.translation
                - initial[f].world_to_camera.translation)
                .norm();
            assert!(d < 1e-12, "fixed prefix pose {f} moved by {d}");
        }
        // The drifted trailing poses are pulled back toward truth.
        for f in [3usize, 4] {
            let before = (initial[f].world_to_camera.translation
                - truth[f].world_to_camera.translation)
                .norm();
            let after = (refinement.refined_poses[f].world_to_camera.translation
                - truth[f].world_to_camera.translation)
                .norm();
            assert!(
                after < before * 0.5,
                "trailing pose {f}: expected correction, before {before:.4} after {after:.4}"
            );
        }
    }

    /// Inject a small drift on pose 2 and confirm BA corrects it back toward
    /// the correct trajectory.
    #[test]
    fn ba_corrects_injected_translation_drift() {
        let camera = kitti_camera();
        let baseline = 0.537;
        let landmarks = synthetic_landmark_grid();

        let correct_poses = vec![
            Pose::identity(),
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -0.8)),
            },
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -1.6)),
            },
        ];
        // Build observations from CORRECT poses (the "true" image content).
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut stereo = Vec::new();
        for p in &correct_poses {
            let (l, r, s) = project_to_pixels(&camera, p, &landmarks, baseline);
            left.push(l);
            right.push(r);
            stereo.push(s);
        }
        let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..correct_poses.len() - 1)
            .map(|f| {
                let n = left[f].keypoints.len().min(left[f + 1].keypoints.len());
                (0..n)
                    .map(|i| DescriptorMatch {
                        query_index: i,
                        train_index: i,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: Some(1.0),
                    })
                    .collect()
            })
            .collect();

        // Inject 5 cm drift on pose 2's y component (the seq08 failure mode).
        let mut drifted = correct_poses.clone();
        drifted[2].world_to_camera.translation.y += 0.05;

        let config = StereoVoBaConfig::default();
        let refinement = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &drifted,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &config,
        )
        .expect("BA should succeed on drifted synthetic scene");

        let initial_err = 0.05f64;
        let final_err = (refinement.refined_poses[2].world_to_camera.translation
            - correct_poses[2].world_to_camera.translation)
            .norm();
        assert!(
            final_err < initial_err * 0.2,
            "BA failed to correct injected drift: initial {initial_err:.4} final {final_err:.4}"
        );
    }

    /// Stereo VO BA refiner accepts an optional `gravity_prior` from
    /// `StereoVoBaConfig` and applies it on the constructed bundle. This is the
    /// wiring smoke — on a well-conditioned synthetic stereo scene, reprojection
    /// alone is enough to correct injected pitch drift, so the test only asserts
    /// that the prior is accepted, the refinement still converges, and the
    /// recovered rotation is at parity (no worse) with the no-prior baseline.
    /// Deeper "gravity prior recovers rotation" coverage lives in
    /// `tests/bundle_adjustment.rs::gravity_prior_recovers_pitched_pose`, which
    /// uses an under-constrained 6-point mono bundle where the prior is actually
    /// load-bearing.
    #[test]
    fn ba_with_gravity_prior_wires_through_config() {
        let camera = kitti_camera();
        let baseline = 0.537;
        let landmarks = synthetic_landmark_grid();

        let correct_poses = vec![
            Pose::identity(),
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -0.8)),
            },
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -1.6)),
            },
        ];
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut stereo = Vec::new();
        for p in &correct_poses {
            let (l, r, s) = project_to_pixels(&camera, p, &landmarks, baseline);
            left.push(l);
            right.push(r);
            stereo.push(s);
        }
        let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..correct_poses.len() - 1)
            .map(|f| {
                let n = left[f].keypoints.len().min(left[f + 1].keypoints.len());
                (0..n)
                    .map(|i| DescriptorMatch {
                        query_index: i,
                        train_index: i,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: Some(1.0),
                    })
                    .collect()
            })
            .collect();

        // Inject 0.05 rad pitch on pose 2 (rotation around camera x-axis),
        // preserving the truth camera centre.
        let pitch = 0.05_f64;
        let r2 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch);
        let truth_center_2 = -correct_poses[2].world_to_camera.translation;
        let mut drifted = correct_poses.clone();
        drifted[2].world_to_camera = SE3::new(r2, -(r2.transform_vector(&truth_center_2)));

        // Baseline: BA without gravity prior.
        let base_refinement = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &drifted,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &StereoVoBaConfig::default(),
        )
        .expect("BA should succeed without gravity prior");
        let base_rot_err = base_refinement.refined_poses[2]
            .world_to_camera
            .rotation
            .angle_to(&UnitQuaternion::identity());

        // With gravity prior: world up = +y (KITTI cameras y-down → gravity
        // points down in world, so g_world = (0, 9.81, 0) and the same
        // direction is observed in any level camera).
        let mut cfg_grav = StereoVoBaConfig::default();
        cfg_grav.gravity_prior = Some(GravityPrior {
            g_world: Vector3::new(0.0, 9.81, 0.0),
            g_camera_observed: Vector3::new(0.0, 9.81, 0.0),
            weight: 20.0,
        });
        let grav_refinement = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &drifted,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &cfg_grav,
        )
        .expect("BA should succeed with gravity prior");
        let grav_rot_err = grav_refinement.refined_poses[2]
            .world_to_camera
            .rotation
            .angle_to(&UnitQuaternion::identity());

        // On this well-constrained stereo scene reprojection alone reaches
        // ~0 rad rotation residual; the wiring assertion is parity with that
        // baseline within a small tolerance plus a hard 0.01 rad ceiling.
        let tol = 5.0e-3;
        assert!(
            grav_rot_err <= base_rot_err + tol,
            "gravity prior must not measurably worsen rotation error; \
             base={base_rot_err:.5} rad, with_prior={grav_rot_err:.5} rad"
        );
        assert!(
            grav_rot_err < 0.01,
            "gravity prior pass must keep pose 2 rotation < 0.01 rad; got {grav_rot_err:.5}"
        );
    }

    /// Stereo VO BA refiner accepts an optional `per_pose_gravity_prior`
    /// from `StereoVoBaConfig` and applies it on the constructed bundle.
    /// Like the global `gravity_prior` wiring smoke, this asserts the
    /// configured prior is wired through and does not measurably worsen
    /// the no-prior baseline rotation residual on a well-conditioned
    /// stereo scene. Deeper "per-pose prior recovers rotation" coverage
    /// lives in `tests/bundle_adjustment.rs::per_pose_gravity_prior_*`.
    #[test]
    fn ba_with_per_pose_gravity_prior_wires_through_config() {
        use crate::{PerPoseGravityObservation, PerPoseGravityPrior};

        let camera = kitti_camera();
        let baseline = 0.537;
        let landmarks = synthetic_landmark_grid();

        let correct_poses = vec![
            Pose::identity(),
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -0.8)),
            },
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -1.6)),
            },
        ];
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut stereo = Vec::new();
        for p in &correct_poses {
            let (l, r, s) = project_to_pixels(&camera, p, &landmarks, baseline);
            left.push(l);
            right.push(r);
            stereo.push(s);
        }
        let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..correct_poses.len() - 1)
            .map(|f| {
                let n = left[f].keypoints.len().min(left[f + 1].keypoints.len());
                (0..n)
                    .map(|i| DescriptorMatch {
                        query_index: i,
                        train_index: i,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: Some(1.0),
                    })
                    .collect()
            })
            .collect();

        let pitch = 0.05_f64;
        let r2 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch);
        let truth_center_2 = -correct_poses[2].world_to_camera.translation;
        let mut drifted = correct_poses.clone();
        drifted[2].world_to_camera = SE3::new(r2, -(r2.transform_vector(&truth_center_2)));

        // Baseline: BA without per-pose gravity prior.
        let base_refinement = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &drifted,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &StereoVoBaConfig::default(),
        )
        .expect("BA should succeed without per-pose gravity prior");
        let base_rot_err = base_refinement.refined_poses[2]
            .world_to_camera
            .rotation
            .angle_to(&UnitQuaternion::identity());

        // With per-pose gravity prior: every frame observes the level
        // direction (0, 9.81, 0) in camera coordinates — i.e. the
        // accelerometer-derived observation for a level camera.
        let mut cfg_grav = StereoVoBaConfig::default();
        let mut prior = PerPoseGravityPrior::new(Vector3::new(0.0, 9.81, 0.0), 20.0);
        for i in 0..correct_poses.len() {
            prior.push(PerPoseGravityObservation {
                keyframe_id: i as u64,
                g_camera_observed: Vector3::new(0.0, 9.81, 0.0),
                weight: 1.0,
            });
        }
        cfg_grav.per_pose_gravity_prior = Some(prior);
        let grav_refinement = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &drifted,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &cfg_grav,
        )
        .expect("BA should succeed with per-pose gravity prior");
        let grav_rot_err = grav_refinement.refined_poses[2]
            .world_to_camera
            .rotation
            .angle_to(&UnitQuaternion::identity());

        let tol = 5.0e-3;
        assert!(
            grav_rot_err <= base_rot_err + tol,
            "per-pose gravity prior must not measurably worsen rotation error; \
             base={base_rot_err:.5} rad, with_prior={grav_rot_err:.5} rad"
        );
        assert!(
            grav_rot_err < 0.01,
            "per-pose gravity prior pass must keep pose 2 rotation < 0.01 rad; got {grav_rot_err:.5}"
        );
    }

    /// `slice_per_pose_gravity_prior_for_window` keeps observations
    /// whose keyframe falls inside `[start, end)` and remaps their id
    /// to the local window's `0..(end-start)` range. Observations
    /// outside the window are dropped. This is the contract the
    /// sliding-window BA loop depends on.
    #[test]
    fn slice_per_pose_gravity_prior_remaps_local_window_ids() {
        use crate::{PerPoseGravityObservation, PerPoseGravityPrior};
        let mut prior = PerPoseGravityPrior::new(Vector3::new(0.0, 9.81, 0.0), 1.0);
        for id in 0..10u64 {
            prior.push(PerPoseGravityObservation {
                keyframe_id: id,
                g_camera_observed: Vector3::new(0.0, 9.81, id as f64 * 0.1),
                weight: 1.0,
            });
        }
        let sliced = slice_per_pose_gravity_prior_for_window(&prior, 3, 7);
        assert_eq!(sliced.g_world, prior.g_world);
        assert_eq!(sliced.weight, prior.weight);
        assert_eq!(sliced.observations.len(), 4);
        // Local ids should be 0..4 and the per-keyframe observations
        // should carry the original z-coordinate (3*0.1, 4*0.1, 5*0.1,
        // 6*0.1) so we can confirm the remap kept the right entries.
        for (local_id, expected_global) in (0..4u64).zip(3..7u64) {
            assert_eq!(sliced.observations[local_id as usize].keyframe_id, local_id);
            assert!(
                (sliced.observations[local_id as usize].g_camera_observed.z
                    - expected_global as f64 * 0.1)
                    .abs()
                    < 1.0e-12,
                "expected local id {local_id} to carry global id {expected_global}'s observation"
            );
        }
    }

    /// Drive `refine_stereo_vo_with_ba` with a `StereoVoBaImuInput` that
    /// matches the visual truth motion and a deliberately wrong velocity
    /// initialisation (which the refiner seeds from the inter-keyframe
    /// pose-centre delta — so the seed comes out correct here). The
    /// solver must (a) populate `imu_refinement`, (b) recover refined
    /// velocities consistent with the truth `(2 m, 0, 0)` motion over
    /// 1 second windows, and (c) leave the poses essentially unchanged
    /// since the visual + IMU residuals are already at the joint
    /// minimum.
    #[test]
    fn ba_with_imu_input_wires_through_config() {
        let camera = kitti_camera();
        let baseline = 0.537;
        let landmarks = synthetic_landmark_grid();

        // 3 frames with constant +x motion: world centres at 0 / 2 / 4 m,
        // each 1 s apart. Camera-frame velocity matches body-frame
        // velocity because the rotation is identity throughout.
        let centers = [0.0_f64, 2.0, 4.0];
        let poses: Vec<Pose> = centers
            .iter()
            .map(|&x| Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(-x, 0.0, 0.0)),
            })
            .collect();

        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut stereo = Vec::new();
        for p in &poses {
            let (l, r, s) = project_to_pixels(&camera, p, &landmarks, baseline);
            left.push(l);
            right.push(r);
            stereo.push(s);
        }
        let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..poses.len() - 1)
            .map(|f| {
                let n = left[f].keypoints.len().min(left[f + 1].keypoints.len());
                (0..n)
                    .map(|i| DescriptorMatch {
                        query_index: i,
                        train_index: i,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: Some(1.0),
                    })
                    .collect()
            })
            .collect();

        // Constant +x acceleration of 2 m/s² for 1 s → world centres at
        // 0, 1, 2 m. But our truth scene has centres at 0, 2, 4 m, so
        // pick constant velocity 2 m/s instead (zero accel). With zero
        // accel, gravity in the IMU equation cancels only if we set
        // `gravity_world = 0` — keep the test gravity-free for clarity.
        let dt_step = 0.05_f64;
        let steps_per_window = 20usize; // 1 s window
        let zero = Vector3::<f64>::zeros();
        let window: Vec<StereoVoBaImuSample> = (0..steps_per_window)
            .map(|_| StereoVoBaImuSample {
                dt: dt_step,
                gyro: zero,
                accel: zero, // zero accel in body frame; gravity = 0 below
            })
            .collect();
        let imu_input = StereoVoBaImuInput::new(
            vec![window.clone(), window],
            Vector3::zeros(),
            1.0,
            1.0,
            1.0,
        );

        let mut cfg = StereoVoBaConfig::default();
        cfg.imu_input = Some(imu_input);
        let refinement = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &poses,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &cfg,
        )
        .expect("BA with IMU input should succeed");

        let imu = refinement
            .imu_refinement
            .as_ref()
            .expect("imu_refinement should be populated when imu_input is set");
        assert_eq!(imu.refined_velocities.len(), poses.len());
        assert_eq!(imu.refined_bias_gyro.len(), poses.len());
        assert_eq!(imu.refined_bias_acc.len(), poses.len());

        // All three keyframes participate in at least one IMU window, so
        // each velocity slot should be close to the truth (+2, 0, 0).
        let truth_v = Vector3::new(2.0, 0.0, 0.0);
        for (i, v) in imu.refined_velocities.iter().enumerate() {
            let err = (v - truth_v).norm();
            assert!(
                err < 0.05,
                "keyframe {i} velocity {v:?} diverged from truth {truth_v:?}; err={err:.4}"
            );
        }

        // Poses should still match the input (the residuals are zero at
        // truth and the seed velocities come from the pose-centre delta
        // so the IMU factor contributes a near-zero residual too).
        for (i, (refined, original)) in refinement
            .refined_poses
            .iter()
            .zip(poses.iter())
            .enumerate()
        {
            let dt =
                (refined.world_to_camera.translation - original.world_to_camera.translation).norm();
            assert!(
                dt < 5.0e-3,
                "frame {i} drifted {dt} from truth under joint visual+IMU BA"
            );
        }
    }

    /// Gyro-bias observability scenario. The camera yaws around its
    /// y-axis at a known rate while staying at the world origin (rotation
    /// only). IMU samples carry the true body-frame angular velocity
    /// plus a non-zero true gyro bias; we feed BA the windows with
    /// `bias_gyro_init = 0` and `fix_first_bias = false`. The visual
    /// factor pins each frame's rotation, so the IMU rotation residual
    /// is paid by the bias slot — refined gyro bias must converge to the
    /// truth on every IMU-active keyframe. Accel bias stays at zero (no
    /// gravity, no linear motion) — this scenario isolates gyro bias
    /// observability.
    #[test]
    fn ba_with_imu_input_recovers_gyro_bias_under_rotation() {
        let camera = kitti_camera();
        let baseline = 0.537;
        let landmarks = synthetic_landmark_grid();

        // 4 keyframes; camera centre stays at origin, yaw advances by
        // omega_yaw rad each window. Landmarks at depth 10..20 m stay in
        // FOV for the full 0..0.18 rad sweep.
        let omega_yaw = 0.06_f64;
        let dt_window = 1.0_f64;
        let n_frames = 4usize;
        let poses: Vec<Pose> = (0..n_frames)
            .map(|i| {
                let yaw = omega_yaw * dt_window * i as f64;
                let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);
                // Camera centre at origin → world_to_camera translation = 0.
                Pose {
                    world_to_camera: SE3::new(r, Vector3::zeros()),
                }
            })
            .collect();

        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut stereo = Vec::new();
        for p in &poses {
            let (l, r, s) = project_to_pixels(&camera, p, &landmarks, baseline);
            left.push(l);
            right.push(r);
            stereo.push(s);
        }
        let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..poses.len() - 1)
            .map(|f| {
                let n = left[f].keypoints.len().min(left[f + 1].keypoints.len());
                (0..n)
                    .map(|i| DescriptorMatch {
                        query_index: i,
                        train_index: i,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: Some(1.0),
                    })
                    .collect()
            })
            .collect();

        // Body-frame angular velocity sign: with
        // `world_to_camera = exp(+yaw_i ey)` the body (camera) frame is
        // rotating around its own y axis at rate `−omega_yaw` (so that
        // `R_i^T R_j = exp((yaw_i − yaw_j) ey) = exp(−omega_yaw·Δt ey)`
        // matches what pre-integration with `gyro = −omega_yaw ey + truth_bias`
        // produces). True bias offsets the +y component by `true_bias_y`;
        // with `bias_gyro_init = 0` the pre-integrated ΔR is off by
        // `−true_bias_y · dt_window` rad — the bias slot must absorb that.
        let true_bias_y = 0.015_f64;
        let true_bias = Vector3::new(0.0, true_bias_y, 0.0);
        let dt_step = 0.05_f64;
        let steps_per_window = (dt_window / dt_step).round() as usize; // 20
        let imu_gyro_sample = Vector3::new(0.0, -omega_yaw + true_bias_y, 0.0);
        let window: Vec<StereoVoBaImuSample> = (0..steps_per_window)
            .map(|_| StereoVoBaImuSample {
                dt: dt_step,
                gyro: imu_gyro_sample,
                accel: Vector3::zeros(),
            })
            .collect();
        let windows = vec![window; n_frames - 1];

        let mut imu_input = StereoVoBaImuInput::new(
            windows,
            Vector3::zeros(),
            1.0,  // weight_position
            1.0,  // weight_velocity
            50.0, // weight_rotation — strong so rotation residual drives the bias
        );
        imu_input.fix_first_bias = false; // bias slot 0 is free to move
        imu_input.fix_first_velocity = true; // anchor velocity gauge at 0
        imu_input.bias_random_walk_weight = Some(10.0); // tie bias slots together

        let mut cfg = StereoVoBaConfig::default();
        cfg.imu_input = Some(imu_input);
        let refinement = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &poses,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &cfg,
        )
        .expect("BA with IMU input should succeed");

        let imu = refinement
            .imu_refinement
            .as_ref()
            .expect("imu_refinement populated when imu_input is set");
        assert_eq!(imu.refined_bias_gyro.len(), poses.len());
        assert_eq!(imu.refined_bias_acc.len(), poses.len());

        // Every keyframe is IMU-active (4 keyframes, 3 chained windows),
        // so all bias slots should land near truth. Tolerance accounts
        // for the first-order linearisation of the bias correction
        // (true bias ~0.015 from init=0).
        let tol_gyro = 5.0e-3;
        for (i, b) in imu.refined_bias_gyro.iter().enumerate() {
            let err = (b - true_bias).norm();
            assert!(
                err < tol_gyro,
                "keyframe {i} gyro bias {b:?} diverged from truth {true_bias:?}; \
                 err={err:.5} > tol={tol_gyro}"
            );
        }

        // Accel bias should stay close to zero: with gravity = 0 and
        // zero accel samples, no signal drives the accel-bias slots away
        // from the init. The random-walk tie keeps them mutually close.
        let tol_acc = 5.0e-3;
        for (i, b) in imu.refined_bias_acc.iter().enumerate() {
            assert!(
                b.norm() < tol_acc,
                "keyframe {i} accel bias {b:?} drifted from zero under no-gravity / no-accel scene; \
                 norm={:.5} > tol={tol_acc}",
                b.norm()
            );
        }

        // Sanity: poses should still match the input rotation closely.
        for (i, (refined, original)) in refinement
            .refined_poses
            .iter()
            .zip(poses.iter())
            .enumerate()
        {
            let r_err = refined
                .world_to_camera
                .rotation
                .angle_to(&original.world_to_camera.rotation);
            assert!(
                r_err < 2.0e-3,
                "frame {i} rotation drifted {r_err:.5} rad under joint visual+IMU BA"
            );
        }
    }

    #[test]
    fn imu_samples_txt_round_trips_through_pre_integration() {
        let text = "\
# dt gx gy gz ax ay az
0.01 0.0 0.0 0.0 2.0 0.0 0.0

0.01 0.0 0.0 0.0 2.0 0.0 0.0
0.01 0.0 0.0 0.0 2.0 0.0 0.0
";
        let samples = parse_stereo_vo_imu_samples_txt(text).expect("parse");
        assert_eq!(samples.len(), 3);
        assert!((samples[0].dt - 0.01).abs() < 1e-12);
        assert_eq!(samples[1].accel, Vector3::new(2.0, 0.0, 0.0));
        assert_eq!(samples[2].gyro, Vector3::zeros());
    }

    #[test]
    fn imu_samples_txt_rejects_bad_lines() {
        // Wrong column count.
        let err = parse_stereo_vo_imu_samples_txt("0.01 0.0 0.0 0.0 2.0 0.0\n")
            .expect_err("should fail on 6 columns");
        assert!(err.contains("7 numbers"));
        // Negative dt.
        let err = parse_stereo_vo_imu_samples_txt("-0.01 0 0 0 0 0 0\n")
            .expect_err("should fail on negative dt");
        assert!(err.contains("positive"));
        // Garbage token.
        let err = parse_stereo_vo_imu_samples_txt("foo 0 0 0 0 0 0\n")
            .expect_err("should fail on non-numeric");
        assert!(err.contains("cannot parse"));
    }

    /// `imu_input.windows.len()` mismatch must produce a structured
    /// error rather than a panic. Same for `window_size` + IMU.
    #[test]
    fn ba_with_imu_input_validates_window_count_and_sliding_window() {
        let camera = kitti_camera();
        let baseline = 0.537;
        let landmarks = synthetic_landmark_grid();
        let poses = vec![
            Pose::identity(),
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -1.0)),
            },
            Pose {
                world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -2.0)),
            },
        ];
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut stereo = Vec::new();
        for p in &poses {
            let (l, r, s) = project_to_pixels(&camera, p, &landmarks, baseline);
            left.push(l);
            right.push(r);
            stereo.push(s);
        }
        let temporal_matches: Vec<Vec<DescriptorMatch>> = (0..poses.len() - 1)
            .map(|f| {
                let n = left[f].keypoints.len().min(left[f + 1].keypoints.len());
                (0..n)
                    .map(|i| DescriptorMatch {
                        query_index: i,
                        train_index: i,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: Some(1.0),
                    })
                    .collect()
            })
            .collect();

        // Wrong window count (1 instead of 2).
        let mut cfg = StereoVoBaConfig::default();
        cfg.imu_input = Some(StereoVoBaImuInput::new(
            vec![vec![]],
            Vector3::zeros(),
            1.0,
            1.0,
            1.0,
        ));
        let err = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &poses,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &cfg,
        )
        .expect_err("should fail with InvalidImuInput");
        assert!(matches!(err, StereoVoBaError::InvalidImuInput { .. }));

        // Right window count but `window_size` is also set.
        cfg.imu_input = Some(StereoVoBaImuInput::new(
            vec![vec![], vec![]],
            Vector3::zeros(),
            1.0,
            1.0,
            1.0,
        ));
        cfg.window_size = Some(2);
        let err = refine_stereo_vo_with_ba(
            &camera,
            baseline,
            &poses,
            &left,
            &right,
            &stereo,
            &temporal_matches,
            &cfg,
        )
        .expect_err("should fail with InvalidImuInput (sliding window)");
        assert!(matches!(err, StereoVoBaError::InvalidImuInput { .. }));
    }

    /// Slicing a globally-timestamped IMU stream should land each sample in
    /// the window covering its timestamp and emit `dt` gaps anchored at the
    /// preceding keyframe. A trailing fragment is required when the last
    /// IMU sample stops short of the next keyframe.
    #[test]
    fn slice_imu_samples_buckets_by_keyframe_intervals() {
        let imu_t = vec![
            50_000_000i128, // 0.05 s — falls in window 0 (kf0..kf1).
            150_000_000,    // 0.15 s — also in window 0.
            350_000_000,    // 0.35 s — in window 1 (kf1..kf2).
            900_000_000,    // 0.9 s  — in window 2 (kf2..kf3); trailing fragment expected.
        ];
        let gyro = vec![
            Vector3::new(0.1, 0.0, 0.0),
            Vector3::new(0.2, 0.0, 0.0),
            Vector3::new(0.3, 0.0, 0.0),
            Vector3::new(0.4, 0.0, 0.0),
        ];
        let accel = vec![
            Vector3::new(0.0, 0.0, 9.81),
            Vector3::new(0.0, 0.0, 9.82),
            Vector3::new(0.0, 0.0, 9.83),
            Vector3::new(0.0, 0.0, 9.84),
        ];
        let kf_t = vec![0i128, 200_000_000, 400_000_000, 1_000_000_000];

        let windows = slice_imu_samples_for_keyframes(&imu_t, &gyro, &accel, &kf_t).unwrap();
        assert_eq!(windows.len(), 3);

        // Window 0: samples at 0.05s + 0.15s plus a 0.05s ZOH tail closing
        // the interval at 0.20s (3 entries, total dt = 0.20s).
        assert_eq!(windows[0].len(), 3);
        assert!((windows[0][0].dt - 0.05).abs() < 1e-9);
        assert!((windows[0][1].dt - 0.10).abs() < 1e-9);
        assert_eq!(windows[0][1].gyro, Vector3::new(0.2, 0.0, 0.0));
        assert!((windows[0][2].dt - 0.05).abs() < 1e-9);
        assert_eq!(windows[0][2].gyro, Vector3::new(0.2, 0.0, 0.0));
        let total_w0: f64 = windows[0].iter().map(|s| s.dt).sum();
        assert!((total_w0 - 0.20).abs() < 1e-9);

        // Window 1: sample at 0.35s (dt=0.15) plus a 0.05s tail to 0.40s.
        assert_eq!(windows[1].len(), 2);
        assert!((windows[1][0].dt - 0.15).abs() < 1e-9);
        assert_eq!(windows[1][0].accel, Vector3::new(0.0, 0.0, 9.83));
        assert!((windows[1][1].dt - 0.05).abs() < 1e-9);

        // Window 2: sample at 0.9s (dt=0.5) plus trailing ZOH fragment to 1.0s (dt=0.1).
        assert_eq!(windows[2].len(), 2);
        assert!((windows[2][0].dt - 0.5).abs() < 1e-9);
        assert_eq!(windows[2][0].gyro, Vector3::new(0.4, 0.0, 0.0));
        assert!((windows[2][1].dt - 0.1).abs() < 1e-9);
        // Tail step holds the last sample's gyro/accel.
        assert_eq!(windows[2][1].gyro, Vector3::new(0.4, 0.0, 0.0));
        assert_eq!(windows[2][1].accel, Vector3::new(0.0, 0.0, 9.84));

        // Total integrated time across all windows matches the keyframe span.
        let total_dt: f64 = windows.iter().flat_map(|w| w.iter()).map(|s| s.dt).sum();
        assert!((total_dt - 1.0).abs() < 1e-9);
    }

    /// A keyframe interval that the IMU stream never covers should produce
    /// an empty window (so [`refine_stereo_vo_with_ba`] silently skips the
    /// IMU factor for that segment).
    #[test]
    fn slice_imu_samples_emits_empty_window_when_no_coverage() {
        let imu_t = vec![10_000_000i128, 20_000_000];
        let gyro = vec![Vector3::zeros(), Vector3::zeros()];
        let accel = vec![Vector3::zeros(), Vector3::zeros()];
        let kf_t = vec![0i128, 100_000_000, 200_000_000];

        let windows = slice_imu_samples_for_keyframes(&imu_t, &gyro, &accel, &kf_t).unwrap();
        assert_eq!(windows.len(), 2);
        // Window 0: 2 samples + 1 ZOH tail filling the remaining gap to 0.1s.
        assert_eq!(windows[0].len(), 3);
        // Window 1 has no IMU samples and no `last` to extend, so it stays empty.
        assert!(windows[1].is_empty(), "window 1 has no IMU coverage");
    }

    #[test]
    fn slice_imu_samples_validates_lengths_and_monotonicity() {
        let gyro = vec![Vector3::zeros(); 2];
        let accel = vec![Vector3::zeros(); 2];

        // Length mismatch.
        let err = slice_imu_samples_for_keyframes(&[0i128, 1, 2], &gyro, &accel, &[0i128, 1])
            .expect_err("length mismatch should fail");
        assert!(err.contains("length mismatch"));

        // Fewer than two keyframes.
        let err = slice_imu_samples_for_keyframes(&[0i128, 1], &gyro, &accel, &[0i128])
            .expect_err("single keyframe should fail");
        assert!(err.contains("at least 2 keyframe"));

        // Non-monotonic keyframes.
        let err = slice_imu_samples_for_keyframes(&[0i128, 1], &gyro, &accel, &[1i128, 0])
            .expect_err("non-monotonic should fail");
        assert!(err.contains("keyframe timestamps"));
    }
}
