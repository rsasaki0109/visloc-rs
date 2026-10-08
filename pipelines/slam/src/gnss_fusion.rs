//! GNSS + visual-odometry fusion: a joint (full-batch or sliding-window)
//! estimator over VO relative-pose factors and GNSS position factors.
//!
//! This turns GNSS from a per-frame *search prior* (see
//! `visloc_fusion::GnssMeasurement`'s `LocalizationPriorProvider` impl) into
//! a genuine fused estimate. The pipeline is:
//!
//! 1. **Time synchronization.** Each GNSS fix stream is resampled at the
//!    camera frame times by [`visloc_fusion::interpolate_gnss_fix`]: a fix within
//!    `max_nearest_offset` is used directly,
//!    otherwise the two bracketing fixes are linearly interpolated when they
//!    are at most `max_bracket_gap` apart. Larger
//!    gaps are **dropouts**: the affected frames simply get no GNSS factor and
//!    are carried by the VO chain.
//! 2. **Per-fix covariance.** A fix's full position covariance is used when
//!    present, else its horizontal/vertical accuracies, else configurable
//!    defaults ([`visloc_fusion::gnss_fix_covariance`]). Interpolation blends covariances
//!    conservatively (GNSS errors are time-correlated, so the midpoint is *not*
//!    treated as more accurate) and adds an unmodelled-motion term. When the
//!    camera runs faster than the receiver, one fix feeds several frames; its
//!    information is split between them
//!    ([`GnssVoFusionConfig::share_fix_information`]) so it is not counted
//!    twice.
//! 3. **Alignment bootstrap.** The GNSS-ENU-to-map alignment (yaw +
//!    translation, plus scale for monocular VO, or a full rotation for VO whose
//!    map frame is not gravity aligned) is initialized by RANSAC + weighted
//!    Procrustes/Umeyama over the earliest fixes that span
//!    [`GnssBootstrapConfig::min_horizontal_extent`] metres
//!    ([`crate::bootstrap_gnss_alignment`]).
//! 4. **Joint optimization.** [`crate::GnssPoseGraph`] then estimates the
//!    frame poses (map frame, optional Sim(3)-style per-frame scale) *and* the
//!    alignment together, with the antenna lever arm, per-fix covariances, and
//!    a robust kernel / Graduated Non-Convexity that rejects multipath jumps.
//!
//! [`fuse_gnss_with_visual_odometry`] is the one-call batch entry point; the
//! [`GnssVoFusion`] struct is the incremental interface (push frames and fixes,
//! call [`GnssVoFusion::optimize`] whenever a fused estimate is wanted), which
//! with [`GnssVoFusionConfig::window`] becomes a fixed-lag sliding-window
//! smoother.
//!
//! # Coupling level
//!
//! This is **loosely coupled in the GNSS domain**: the receiver's position
//! solution (ENU, with covariance) is the measurement. Raw pseudoranges,
//! Doppler, carrier phase, and satellite geometry are not modelled. On the
//! visual side the factors are VO relative poses, not image reprojections, so
//! it is a pose-graph fusion rather than a visual-GNSS bundle adjustment.

use nalgebra::{SMatrix, Vector3};
use visloc_core::geometry::{Pose, SE3};
use visloc_fusion::{GnssMeasurement, MeasurementBuffer, Timed, TimedPose, Timestamp};

use crate::gnss_pose_graph::{
    bootstrap_gnss_alignment, fit_gnss_alignment, AlignmentBootstrapConfig,
    AlignmentCorrespondence, GnssAlignmentPrior, GnssGraphNode, GnssPoseGraph, GnssPoseGraphError,
    GnssPositionFactor, VoRelativeFactor,
};

pub use crate::gnss_pose_graph::{
    AlignmentBootstrap, AlignmentRotationDof, GnssAlignment, GnssPoseGraphConfig,
    GnssPoseGraphResult, GnssRobustMode, GnssScaleMode, GNSS_CHI2_3DOF_999,
};
pub use visloc_fusion::{
    gnss_fix_covariance, interpolate_gnss_fix, GnssFixSource, GnssInterpolationConfig,
    InterpolatedGnssFix,
};

/// Noise model turning VO relative motion into factor information.
#[derive(Debug, Clone, PartialEq)]
pub struct VoNoiseModel {
    /// Translation sigma floor per step (VO units).
    pub translation_sigma_floor: f64,
    /// Translation sigma as a fraction of the step length.
    pub translation_sigma_ratio: f64,
    /// Rotation sigma per step (radians, per axis).
    pub rotation_sigma: f64,
    /// Log-scale random-walk sigma per step (used only for
    /// [`GnssScaleMode::ScaleDrift`]).
    pub log_scale_sigma: f64,
}

impl Default for VoNoiseModel {
    fn default() -> Self {
        Self {
            translation_sigma_floor: 0.01,
            translation_sigma_ratio: 0.02,
            rotation_sigma: 0.002,
            log_scale_sigma: 0.005,
        }
    }
}

impl VoNoiseModel {
    /// `(6×6 information over [translation; rotation], log-scale information)`
    /// for one relative motion.
    pub fn information(&self, relative: &SE3) -> (SMatrix<f64, 6, 6>, f64) {
        let sigma_t = (self.translation_sigma_floor
            + self.translation_sigma_ratio * relative.translation.norm())
        .max(1e-9);
        let sigma_r = self.rotation_sigma.max(1e-9);
        let mut information = SMatrix::<f64, 6, 6>::zeros();
        for axis in 0..3 {
            information[(axis, axis)] = 1.0 / (sigma_t * sigma_t);
            information[(axis + 3, axis + 3)] = 1.0 / (sigma_r * sigma_r);
        }
        let sigma_s = self.log_scale_sigma.max(1e-9);
        (information, 1.0 / (sigma_s * sigma_s))
    }
}

/// When and how the GNSS-ENU-to-map alignment is bootstrapped.
#[derive(Debug, Clone, PartialEq)]
pub struct GnssBootstrapConfig {
    /// Minimum frames with a GNSS fix before bootstrapping.
    pub min_fixes: usize,
    /// Minimum horizontal extent (metres) of the inlier fixes: below it the
    /// yaw (and scale) are unobservable and the bootstrap waits.
    pub min_horizontal_extent: f64,
    pub ransac_iterations: usize,
    /// RANSAC gate in fix sigmas (plus [`Self::inlier_threshold_floor`]).
    pub inlier_threshold_sigmas: f64,
    /// Additive RANSAC slack (metres) absorbing VO drift in the bootstrap
    /// window.
    pub inlier_threshold_floor: f64,
    pub seed: u64,
}

impl Default for GnssBootstrapConfig {
    fn default() -> Self {
        Self {
            min_fixes: 10,
            min_horizontal_extent: 30.0,
            ransac_iterations: 300,
            inlier_threshold_sigmas: 3.0,
            inlier_threshold_floor: 2.0,
            seed: 0x5eed_9a55,
        }
    }
}

/// Per-update alignment random walk used by the sliding window: when the
/// window no longer starts at the first frame, the previous alignment becomes
/// a Gaussian prior with these sigmas.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentPriorSigmas {
    pub translation: f64,
    pub rotation: f64,
    pub log_scale: f64,
}

impl Default for AlignmentPriorSigmas {
    fn default() -> Self {
        Self {
            translation: 0.5,
            rotation: 0.01,
            log_scale: 0.01,
        }
    }
}

/// Configuration for [`GnssVoFusion`] / [`fuse_gnss_with_visual_odometry`].
#[derive(Debug, Clone, PartialEq)]
pub struct GnssVoFusionConfig {
    pub interpolation: GnssInterpolationConfig,
    /// Split a fix's information between the frames it feeds.
    pub share_fix_information: bool,
    pub vo_noise: VoNoiseModel,
    /// GNSS antenna position in the camera frame (metres).
    pub lever_arm: Vector3<f64>,
    pub bootstrap: GnssBootstrapConfig,
    /// Joint optimizer settings, including the alignment rotation DoF, scale
    /// mode, and the GNSS robust kernel / GNC.
    pub optimizer: GnssPoseGraphConfig,
    /// Sliding-window length in frames; `None` optimizes all frames.
    pub window: Option<usize>,
    /// Frames between optimizations in [`fuse_gnss_with_visual_odometry`]'s
    /// sliding-window replay.
    pub window_update_stride: usize,
    pub alignment_prior: AlignmentPriorSigmas,
}

impl Default for GnssVoFusionConfig {
    fn default() -> Self {
        Self {
            interpolation: GnssInterpolationConfig::default(),
            share_fix_information: true,
            vo_noise: VoNoiseModel::default(),
            lever_arm: Vector3::zeros(),
            bootstrap: GnssBootstrapConfig::default(),
            optimizer: GnssPoseGraphConfig::default(),
            window: None,
            window_update_stride: 10,
            alignment_prior: AlignmentPriorSigmas::default(),
        }
    }
}

impl GnssVoFusionConfig {
    /// Metric, gravity-aligned VO/VIO (`+z` up): yaw + translation alignment.
    pub fn metric_gravity_aligned() -> Self {
        Self::default()
    }

    /// Monocular VO whose map frame is an arbitrary camera frame: full
    /// rotation, global scale, and per-frame scale drift.
    pub fn monocular() -> Self {
        let mut config = Self::default();
        config.optimizer.rotation_dof = AlignmentRotationDof::Full;
        config.optimizer.scale_mode = GnssScaleMode::ScaleDrift;
        config
    }
}

/// Errors from the fusion front end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GnssFusionError {
    /// Frames must be pushed in strictly increasing time order.
    NonMonotonicTimestamp {
        previous: Timestamp,
        next: Timestamp,
    },
    /// At least two frames are needed.
    TooFewFrames,
    /// The alignment could not be bootstrapped (too few fixes, too little
    /// horizontal motion, or no consistent consensus).
    AlignmentNotObservable,
    Optimizer(GnssPoseGraphError),
}

impl std::fmt::Display for GnssFusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonMonotonicTimestamp { previous, next } => write!(
                f,
                "frame timestamp {} ns is not after {} ns",
                next.as_nanoseconds(),
                previous.as_nanoseconds()
            ),
            Self::TooFewFrames => write!(f, "GNSS/VO fusion needs at least two frames"),
            Self::AlignmentNotObservable => {
                write!(f, "GNSS-to-map alignment is not observable yet")
            }
            Self::Optimizer(error) => write!(f, "GNSS pose graph: {error}"),
        }
    }
}

impl std::error::Error for GnssFusionError {}

impl From<GnssPoseGraphError> for GnssFusionError {
    fn from(error: GnssPoseGraphError) -> Self {
        Self::Optimizer(error)
    }
}

/// GNSS outcome of one frame at its latest optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameGnssStatus {
    /// The frame has not been part of an optimization yet.
    NotOptimized,
    /// No usable fix (dropout, or bracket not yet complete).
    NoFix,
    /// Fix accepted; `squared_residual` is the whitened `rᵀΩr`.
    Inlier { squared_residual: f64 },
    /// Fix rejected by the robust solve / χ² gate.
    Outlier { squared_residual: f64 },
}

/// Per-frame fused state and GNSS status.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedFrameReport {
    pub timestamp: Timestamp,
    /// Fused camera-to-map pose.
    pub camera_to_map: SE3,
    /// Camera-to-ENU pose from the latest optimization that included the
    /// frame (`None` if it was never optimized).
    pub camera_to_enu: Option<SE3>,
    /// Fused local log-scale (non-zero only in scale-drift mode).
    pub log_scale: f64,
    pub gnss: FrameGnssStatus,
}

/// Summary of one [`GnssVoFusion::optimize`] call.
#[derive(Debug, Clone, PartialEq)]
pub struct GnssFusionUpdate {
    /// The alignment was bootstrapped during this call.
    pub bootstrapped: bool,
    /// Index of the first (fixed) frame of the optimized window.
    pub window_start: usize,
    pub window_frame_count: usize,
    pub gnss_factor_count: usize,
    pub outlier_count: usize,
    pub alignment: GnssAlignment,
    pub optimizer: GnssPoseGraphResult,
}

#[derive(Debug, Clone)]
struct FrameRecord {
    timestamp: Timestamp,
    vo_camera_to_map: SE3,
    state: GnssGraphNode,
    /// ENU pose from the latest optimization that included this frame.
    camera_to_enu: Option<SE3>,
    status: FrameGnssStatus,
}

/// Incremental GNSS/VO fusion engine. Push VO poses (world-to-camera, in the
/// VO map frame) and GNSS fixes (ENU) in any interleaving, then call
/// [`Self::optimize`].
#[derive(Debug, Clone)]
pub struct GnssVoFusion {
    config: GnssVoFusionConfig,
    frames: Vec<FrameRecord>,
    gnss: MeasurementBuffer<GnssMeasurement>,
    alignment: Option<GnssAlignment>,
    bootstrap: Option<AlignmentBootstrap>,
    optimizer_runs: usize,
}

impl GnssVoFusion {
    pub fn new(config: GnssVoFusionConfig) -> Self {
        Self {
            config,
            frames: Vec::new(),
            gnss: MeasurementBuffer::new(),
            alignment: None,
            bootstrap: None,
            optimizer_runs: 0,
        }
    }

    pub fn config(&self) -> &GnssVoFusionConfig {
        &self.config
    }

    /// Append a VO pose. Its fused state is initialized by chaining the VO
    /// relative motion onto the previous frame's *fused* state, so corrections
    /// propagate to new frames immediately.
    pub fn push_vo_pose(&mut self, pose: &TimedPose) -> Result<(), GnssFusionError> {
        let vo = pose.value.camera_to_world();
        let state = match self.frames.last() {
            None => GnssGraphNode::new(vo.clone()),
            Some(previous) => {
                if pose.timestamp <= previous.timestamp {
                    return Err(GnssFusionError::NonMonotonicTimestamp {
                        previous: previous.timestamp,
                        next: pose.timestamp,
                    });
                }
                let relative = previous.vo_camera_to_map.inverse().compose(&vo);
                let base = &previous.state;
                GnssGraphNode {
                    camera_to_map: SE3::new(
                        base.camera_to_map.rotation * relative.rotation,
                        base.camera_to_map.translation
                            + base.camera_to_map.rotation
                                * (relative.translation * base.log_scale.exp()),
                    ),
                    log_scale: base.log_scale,
                }
            }
        };
        self.frames.push(FrameRecord {
            timestamp: pose.timestamp,
            vo_camera_to_map: vo,
            state,
            camera_to_enu: None,
            status: FrameGnssStatus::NotOptimized,
        });
        Ok(())
    }

    pub fn push_gnss(&mut self, measurement: GnssMeasurement) {
        self.gnss.push(measurement);
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn gnss_fix_count(&self) -> usize {
        self.gnss.len()
    }

    pub fn optimizer_runs(&self) -> usize {
        self.optimizer_runs
    }

    /// Current GNSS-ENU-to-map alignment (`None` until bootstrapped).
    pub fn alignment(&self) -> Option<&GnssAlignment> {
        self.alignment.as_ref()
    }

    /// Bootstrap result (`None` until bootstrapped).
    pub fn bootstrap(&self) -> Option<&AlignmentBootstrap> {
        self.bootstrap.as_ref()
    }

    pub fn frame_reports(&self) -> Vec<FusedFrameReport> {
        self.frames
            .iter()
            .map(|frame| FusedFrameReport {
                timestamp: frame.timestamp,
                camera_to_map: frame.state.camera_to_map.clone(),
                camera_to_enu: frame.camera_to_enu.clone(),
                log_scale: frame.state.log_scale,
                gnss: frame.status,
            })
            .collect()
    }

    /// Fused trajectory in the VO map frame (world-to-camera poses).
    pub fn map_trajectory(&self) -> Vec<TimedPose> {
        self.frames
            .iter()
            .map(|frame| {
                Timed::new(
                    frame.timestamp,
                    Pose {
                        world_to_camera: frame.state.camera_to_map.inverse(),
                    },
                )
            })
            .collect()
    }

    /// Fused trajectory in ENU (world-to-camera poses with world = ENU).
    ///
    /// Each frame uses the ENU pose from the latest optimization that
    /// included it. In sliding-window mode the alignment keeps absorbing VO
    /// drift after a frame leaves the window, so re-projecting old map-frame
    /// states through the *current* alignment would be wrong; frames not yet
    /// optimized use the current alignment. `None` until bootstrapped.
    pub fn enu_trajectory(&self) -> Option<Vec<TimedPose>> {
        let alignment = self.alignment.as_ref()?;
        Some(
            self.frames
                .iter()
                .map(|frame| {
                    let camera_to_enu = frame.camera_to_enu.clone().unwrap_or_else(|| {
                        alignment.transform_camera_to_map(&frame.state.camera_to_map)
                    });
                    Timed::new(
                        frame.timestamp,
                        Pose {
                            world_to_camera: camera_to_enu.inverse(),
                        },
                    )
                })
                .collect(),
        )
    }

    /// Interpolated fixes for frames `start..`, with information sharing.
    fn window_fixes(&self, start: usize) -> Vec<Option<InterpolatedGnssFix>> {
        let mut fixes: Vec<Option<InterpolatedGnssFix>> = self.frames[start..]
            .iter()
            .map(|frame| {
                interpolate_gnss_fix(&self.gnss, frame.timestamp, &self.config.interpolation)
            })
            .collect();
        if self.config.share_fix_information {
            let mut usage = vec![0.0_f64; self.gnss.len()];
            for fix in fixes.iter().flatten() {
                for (index, weight) in fix.source_weights() {
                    if weight > 0.0 {
                        usage[index] += 1.0;
                    }
                }
            }
            for fix in fixes.iter_mut().flatten() {
                let factor = fix
                    .source_weights()
                    .iter()
                    .filter(|(_, w)| *w > 0.0)
                    .map(|(index, w)| w * usage[*index])
                    .sum::<f64>()
                    .max(1.0);
                fix.covariance *= factor;
            }
        }
        fixes
    }

    fn try_bootstrap(&mut self) -> bool {
        let fixes = self.window_fixes(0);
        let correspondences: Vec<AlignmentCorrespondence> = self
            .frames
            .iter()
            .zip(&fixes)
            .filter_map(|(frame, fix)| {
                let fix = fix.as_ref()?;
                Some(AlignmentCorrespondence {
                    camera_to_map: frame.state.camera_to_map.clone(),
                    position_enu: fix.position_enu,
                    horizontal_sigma: fix.horizontal_sigma(),
                    vertical_sigma: fix.vertical_sigma(),
                })
            })
            .collect();
        let bootstrap = &self.config.bootstrap;
        let config = AlignmentBootstrapConfig {
            rotation_dof: self.config.optimizer.rotation_dof,
            scale_mode: self.config.optimizer.scale_mode,
            lever_arm: self.config.lever_arm,
            ransac_iterations: bootstrap.ransac_iterations,
            inlier_threshold_sigmas: bootstrap.inlier_threshold_sigmas,
            inlier_threshold_floor: bootstrap.inlier_threshold_floor,
            min_inliers: bootstrap.min_fixes,
            min_horizontal_extent: bootstrap.min_horizontal_extent,
            seed: bootstrap.seed,
        };
        // Use the earliest fixes that span the required extent (where VO drift
        // is smallest), growing the prefix if consensus fails.
        let mut prefix = shortest_prefix_with_extent(
            &correspondences,
            bootstrap.min_fixes,
            bootstrap.min_horizontal_extent,
        );
        while let Some(length) = prefix {
            if let Some(result) = bootstrap_gnss_alignment(&correspondences[..length], &config) {
                self.alignment = Some(result.alignment.clone());
                self.bootstrap = Some(result);
                return true;
            }
            prefix = (length < correspondences.len())
                .then(|| (length + length / 2 + 1).min(correspondences.len()));
        }
        false
    }

    /// Bootstrap the alignment if needed, then jointly optimize the window
    /// (all frames when [`GnssVoFusionConfig::window`] is `None`).
    ///
    /// Returns [`GnssFusionError::AlignmentNotObservable`] while the alignment
    /// cannot be bootstrapped yet; the frames keep their VO-propagated states.
    pub fn optimize(&mut self) -> Result<GnssFusionUpdate, GnssFusionError> {
        if self.frames.len() < 2 {
            return Err(GnssFusionError::TooFewFrames);
        }
        let bootstrapped = if self.alignment.is_none() {
            if !self.try_bootstrap() {
                return Err(GnssFusionError::AlignmentNotObservable);
            }
            true
        } else {
            false
        };
        let alignment = self.alignment.clone().expect("alignment bootstrapped");

        let start = self
            .config
            .window
            .map_or(0, |window| self.frames.len().saturating_sub(window.max(2)));
        let fixes = self.window_fixes(start);

        let mut graph = GnssPoseGraph::new();
        graph.lever_arm = self.config.lever_arm;
        graph.alignment = alignment.clone();
        for (offset, frame) in self.frames[start..].iter().enumerate() {
            let id = (start + offset) as u64;
            graph.add_node(id, frame.state.clone());
            if offset > 0 {
                let previous = &self.frames[start + offset - 1];
                let relative = previous
                    .vo_camera_to_map
                    .inverse()
                    .compose(&frame.vo_camera_to_map);
                let (information, log_scale_information) =
                    self.config.vo_noise.information(&relative);
                graph.add_relative_factor(VoRelativeFactor {
                    from: id - 1,
                    to: id,
                    relative,
                    information,
                    log_scale_information,
                });
            }
        }
        graph.fix_node(start as u64);
        let mut factor_frames = Vec::new();
        for (offset, fix) in fixes.iter().enumerate() {
            let Some(fix) = fix else { continue };
            let Some(information) = fix.covariance.try_inverse() else {
                continue;
            };
            graph.add_gnss_factor(GnssPositionFactor {
                node: (start + offset) as u64,
                position_enu: fix.position_enu,
                information,
            });
            factor_frames.push(start + offset);
        }
        if start > 0 {
            let sigmas = &self.config.alignment_prior;
            graph.alignment_prior = Some(GnssAlignmentPrior::from_sigmas(
                alignment,
                sigmas.translation,
                sigmas.rotation,
                sigmas.log_scale,
            ));
        }

        let result = graph.optimize(&self.config.optimizer)?;
        self.optimizer_runs += 1;

        for (offset, frame) in self.frames[start..].iter_mut().enumerate() {
            frame.state = graph.nodes[&((start + offset) as u64)].clone();
            frame.camera_to_enu = Some(
                graph
                    .alignment
                    .transform_camera_to_map(&frame.state.camera_to_map),
            );
            frame.status = FrameGnssStatus::NoFix;
        }
        for (factor, &frame_index) in factor_frames.iter().enumerate() {
            let squared_residual = result.gnss_squared_residuals[factor];
            self.frames[frame_index].status = if result.gnss_inliers[factor] {
                FrameGnssStatus::Inlier { squared_residual }
            } else {
                FrameGnssStatus::Outlier { squared_residual }
            };
        }
        // Frames pushed after the window keep chaining from the fused states.
        self.alignment = Some(graph.alignment.clone());

        Ok(GnssFusionUpdate {
            bootstrapped,
            window_start: start,
            window_frame_count: self.frames.len() - start,
            gnss_factor_count: result.gnss_factor_count,
            outlier_count: result.outlier_count(),
            alignment: graph.alignment,
            optimizer: result,
        })
    }
}

fn shortest_prefix_with_extent(
    correspondences: &[AlignmentCorrespondence],
    min_count: usize,
    min_extent: f64,
) -> Option<usize> {
    let first = correspondences.first()?;
    let mut max_distance: f64 = 0.0;
    for (index, c) in correspondences.iter().enumerate() {
        let d = c.position_enu - first.position_enu;
        max_distance = max_distance.max((d.x * d.x + d.y * d.y).sqrt());
        if index + 1 >= min_count.max(2) && max_distance >= min_extent {
            return Some(index + 1);
        }
    }
    (correspondences.len() >= min_count.max(2)).then_some(correspondences.len())
}

/// Counts over a finished fusion run.
#[derive(Debug, Clone, PartialEq)]
pub struct GnssFusionSummary {
    pub frame_count: usize,
    pub gnss_fix_count: usize,
    pub frames_with_inlier_fix: usize,
    pub frames_with_outlier_fix: usize,
    /// Frames without a usable fix (dropouts).
    pub frames_without_fix: usize,
    pub optimizer_runs: usize,
}

/// Output of [`fuse_gnss_with_visual_odometry`].
#[derive(Debug, Clone, PartialEq)]
pub struct GnssVoFusionResult {
    /// Fused poses in ENU (world-to-camera, world = ENU), one per VO frame.
    pub trajectory_enu: Vec<TimedPose>,
    /// Fused poses in the VO map frame.
    pub trajectory_map: Vec<TimedPose>,
    /// Final GNSS-ENU-to-map alignment.
    pub alignment: GnssAlignment,
    /// The RANSAC bootstrap the joint solve started from.
    pub bootstrap: AlignmentBootstrap,
    pub frames: Vec<FusedFrameReport>,
    pub summary: GnssFusionSummary,
    /// Result of the last joint optimization.
    pub last_update: GnssFusionUpdate,
}

/// Fuse a VO trajectory with a GNSS fix stream.
///
/// With [`GnssVoFusionConfig::window`] `= None` this is one full-batch joint
/// optimization. With `Some(n)` the run is replayed causally: frames are
/// pushed in time order together with every fix up to the frame time, and an
/// `n`-frame sliding window is optimized every
/// [`GnssVoFusionConfig::window_update_stride`] frames (plus once at the end).
pub fn fuse_gnss_with_visual_odometry(
    vo_trajectory: &[TimedPose],
    gnss: &MeasurementBuffer<GnssMeasurement>,
    config: &GnssVoFusionConfig,
) -> Result<GnssVoFusionResult, GnssFusionError> {
    if vo_trajectory.len() < 2 {
        return Err(GnssFusionError::TooFewFrames);
    }
    let mut fusion = GnssVoFusion::new(config.clone());
    let fixes = gnss.as_slice();
    let last_update = match config.window {
        None => {
            for pose in vo_trajectory {
                fusion.push_vo_pose(pose)?;
            }
            for fix in fixes {
                fusion.push_gnss(fix.clone());
            }
            fusion.optimize()?
        }
        Some(_) => {
            let stride = config.window_update_stride.max(1);
            let mut next_fix = 0;
            let mut last = None;
            for (index, pose) in vo_trajectory.iter().enumerate() {
                fusion.push_vo_pose(pose)?;
                while next_fix < fixes.len() && fixes[next_fix].timestamp <= pose.timestamp {
                    fusion.push_gnss(fixes[next_fix].clone());
                    next_fix += 1;
                }
                if (index + 1) % stride == 0 {
                    match fusion.optimize() {
                        Ok(update) => last = Some(update),
                        Err(GnssFusionError::AlignmentNotObservable) => {}
                        Err(error) => return Err(error),
                    }
                }
            }
            for fix in &fixes[next_fix..] {
                fusion.push_gnss(fix.clone());
            }
            match fusion.optimize() {
                Ok(update) => update,
                Err(GnssFusionError::AlignmentNotObservable) if last.is_some() => {
                    last.expect("checked")
                }
                Err(error) => return Err(error),
            }
        }
    };

    let frames = fusion.frame_reports();
    let count = |predicate: fn(&FrameGnssStatus) -> bool| {
        frames.iter().filter(|frame| predicate(&frame.gnss)).count()
    };
    let summary = GnssFusionSummary {
        frame_count: frames.len(),
        gnss_fix_count: fusion.gnss_fix_count(),
        frames_with_inlier_fix: count(|s| matches!(s, FrameGnssStatus::Inlier { .. })),
        frames_with_outlier_fix: count(|s| matches!(s, FrameGnssStatus::Outlier { .. })),
        frames_without_fix: count(|s| {
            matches!(s, FrameGnssStatus::NoFix | FrameGnssStatus::NotOptimized)
        }),
        optimizer_runs: fusion.optimizer_runs(),
    };
    Ok(GnssVoFusionResult {
        trajectory_enu: fusion.enu_trajectory().expect("bootstrapped"),
        trajectory_map: fusion.map_trajectory(),
        alignment: fusion.alignment().cloned().expect("bootstrapped"),
        bootstrap: fusion.bootstrap().cloned().expect("bootstrapped"),
        frames,
        summary,
        last_update,
    })
}

/// Position RMSE (metres) between two equally-indexed trajectories' camera
/// centres, with no alignment. Returns `None` for empty or mismatched input.
pub fn position_rmse(estimate: &[TimedPose], reference: &[TimedPose]) -> Option<f64> {
    if estimate.is_empty() || estimate.len() != reference.len() {
        return None;
    }
    let sum: f64 = estimate
        .iter()
        .zip(reference)
        .map(|(a, b)| {
            (a.value.camera_center_world() - b.value.camera_center_world()).norm_squared()
        })
        .sum();
    Some((sum / estimate.len() as f64).sqrt())
}

/// Absolute trajectory error (RMSE, metres) after the best-fit rigid
/// (`with_scale = false`) or similarity alignment of `estimate` onto
/// `reference` — the usual oracle-aligned ATE.
pub fn aligned_position_rmse(
    estimate: &[TimedPose],
    reference: &[TimedPose],
    with_scale: bool,
) -> Option<f64> {
    if estimate.len() < 3 || estimate.len() != reference.len() {
        return None;
    }
    let correspondences: Vec<AlignmentCorrespondence> = estimate
        .iter()
        .zip(reference)
        .map(|(a, b)| AlignmentCorrespondence {
            camera_to_map: a.value.camera_to_world(),
            position_enu: b.value.camera_center_world().coords,
            horizontal_sigma: 1.0,
            vertical_sigma: 1.0,
        })
        .collect();
    let alignment = fit_gnss_alignment(
        &correspondences,
        AlignmentRotationDof::Full,
        if with_scale {
            GnssScaleMode::GlobalScale
        } else {
            GnssScaleMode::Metric
        },
        &Vector3::zeros(),
    )?;
    let sum: f64 = correspondences
        .iter()
        .map(|c| {
            (alignment.transform_point(&c.camera_to_map.translation) - c.position_enu)
                .norm_squared()
        })
        .sum();
    Some((sum / correspondences.len() as f64).sqrt())
}

/// Apply an alignment to a map-frame trajectory, giving ENU poses.
pub fn transform_trajectory(alignment: &GnssAlignment, trajectory: &[TimedPose]) -> Vec<TimedPose> {
    trajectory
        .iter()
        .map(|pose| {
            Timed::new(
                pose.timestamp,
                Pose {
                    world_to_camera: alignment
                        .transform_camera_to_map(&pose.value.camera_to_world())
                        .inverse(),
                },
            )
        })
        .collect()
}
