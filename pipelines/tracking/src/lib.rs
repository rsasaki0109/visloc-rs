#![forbid(unsafe_code)]
//! Lightweight localization-based tracking scaffold.
//!
//! This crate does not implement full SLAM. It keeps temporal state around
//! repeated localization calls, exposes motion-model pose priors, tracks
//! failure/lost/relocalization events, and leaves keyframe management, map
//! updates, loop closure, and bundle adjustment to future layers.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Write as _;
use std::path::Path;

use nalgebra::{Matrix3, Point3, Quaternion, Rotation3, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{
    CameraId, Frame, FrameId, LandmarkDescriptorStore, LocalizationFailureReason,
    LocalizationResult, QueryImage, VisualMap,
};
use visloc_localization::{
    map_provider_stats, CandidateSelector, DescriptorProvider, InMemoryMapProvider,
    IntersectCandidateSelector, LocalizationPipeline, LocalizationPrior, MapProvider,
    MapProviderStats, RadiusLandmarkSelector,
};
use visloc_vision::features::{FeatureExtractor, FeatureSet};
use visloc_vision::matching::Matcher;
use visloc_vision::ransac::RobustPoseEstimator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingState {
    Uninitialized,
    Tracking,
    Lost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingEvent {
    Initialized,
    Tracked,
    TrackingFailed,
    Lost,
    Relocalized,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingConfig {
    pub min_successive_failures_to_lost: usize,
    pub last_pose_candidate_radius: Option<f64>,
    pub max_pose_prior_translation_error: Option<f64>,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub max_mean_reprojection_error: Option<f64>,
    /// Restrict descriptor-matching candidates to a local map built from the
    /// covisibility graph around the last successfully tracked keyframe. When
    /// the local map is too small (e.g. early-startup, lost-state, or below
    /// `min_local_map_landmarks`), the tracker falls back to the full map.
    pub covisibility_local_map: Option<CovisibilityLocalMapConfig>,
    /// When `true`, hand the motion-model pose prior to the PnP RANSAC as a
    /// warm-start hypothesis (ORB-SLAM3-style motion-only BA seed). Random
    /// samples must beat the prior's inlier count to win, so a well-aligned
    /// prior short-circuits RANSAC on hard scenes while a misaligned prior
    /// gracefully degrades to the standard random search. Off by default to
    /// preserve the existing behaviour where the prior is consumed only as a
    /// candidate-radius filter.
    pub pnp_pose_prior_warm_start: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CovisibilityLocalMapConfig {
    /// Cap on number of co-visible keyframes (the reference keyframe is always
    /// included). Set to `None` for unlimited.
    pub max_keyframes: Option<usize>,
    /// Minimum number of shared landmarks between a candidate keyframe and the
    /// reference keyframe for the candidate to enter the local map.
    pub min_shared_landmarks: usize,
    /// If the resulting local-map landmark set is smaller than this, fall back
    /// to the full descriptor store. Guards against accidentally collapsing the
    /// match candidate pool in degenerate early-startup states.
    pub min_local_map_landmarks: usize,
}

impl Default for CovisibilityLocalMapConfig {
    fn default() -> Self {
        Self {
            max_keyframes: Some(10),
            min_shared_landmarks: 15,
            min_local_map_landmarks: 30,
        }
    }
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            min_successive_failures_to_lost: 3,
            last_pose_candidate_radius: None,
            max_pose_prior_translation_error: None,
            min_inliers: 0,
            min_inlier_ratio: 0.0,
            max_mean_reprojection_error: None,
            covisibility_local_map: None,
            pnp_pose_prior_warm_start: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackingFailureReason {
    InsufficientInliers {
        inlier_count: usize,
        min_inliers: usize,
    },
    InlierRatioTooLow {
        inlier_ratio: f64,
        min_inlier_ratio: f64,
    },
    MeanReprojectionErrorTooHigh {
        reprojection_error: f64,
        max_reprojection_error: f64,
    },
    PosePriorTranslationErrorExceeded {
        translation_error: f64,
        max_translation_error: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingResult {
    pub frame_id: FrameId,
    pub state: TrackingState,
    pub event: TrackingEvent,
    pub successive_failures: usize,
    pub pose_prior: Option<Pose>,
    pub used_pose_prior: bool,
    pub used_external_localization_prior: bool,
    pub external_localization_prior_radius: Option<f64>,
    pub tracking_failure_reason: Option<TrackingFailureReason>,
    pub map_landmark_count: usize,
    pub map_stats: MapProviderStats,
    pub localization: LocalizationResult,
    /// Number of landmarks in the covisibility-derived local map used to
    /// restrict descriptor matching for this frame. `None` when the local-map
    /// filter was disabled or fell back to the full map.
    pub covisibility_local_map_size: Option<usize>,
}

impl TrackingResult {
    pub fn localization_prior(&self, radius: f64) -> LocalizationPrior {
        if let Some(pose_prior) = self.pose_prior.clone() {
            LocalizationPrior::from_pose(pose_prior, radius)
        } else {
            LocalizationPrior::none()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectorySample {
    pub frame_id: FrameId,
    pub pose: Pose,
    pub state: TrackingState,
    pub event: TrackingEvent,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub reprojection_error: Option<f64>,
}

impl TrajectorySample {
    pub fn from_tracking_result(result: &TrackingResult) -> Option<Self> {
        if !result.localization.success {
            return None;
        }

        Some(Self {
            frame_id: result.frame_id,
            pose: result.localization.pose.clone()?,
            state: result.state,
            event: result.event,
            inlier_count: result.localization.inlier_count,
            inlier_ratio: result.localization.inlier_ratio,
            reprojection_error: result.localization.reprojection_error,
        })
    }

    pub fn camera_center_world(&self) -> Point3<f64> {
        self.pose.camera_center_world()
    }

    pub fn to_csv_record(&self) -> String {
        let center = self.camera_center_world();
        let q = self.pose.world_to_camera.rotation.quaternion();
        let t = self.pose.world_to_camera.translation;
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{:?},{:?},{},{},{}",
            self.frame_id,
            center.x,
            center.y,
            center.z,
            q.w,
            q.i,
            q.j,
            q.k,
            t.x,
            t.y,
            t.z,
            self.state,
            self.event,
            self.inlier_count,
            self.inlier_ratio,
            optional_f64_csv(self.reprojection_error)
        )
    }

    pub fn to_kitti_pose_record(&self) -> String {
        let camera_to_world = self.pose.camera_to_world().matrix();
        let mut values = Vec::with_capacity(12);
        for row in 0..3 {
            for column in 0..4 {
                values.push(camera_to_world[(row, column)].to_string());
            }
        }
        values.join(" ")
    }

    pub fn to_tum_pose_record(&self) -> String {
        let camera_to_world = self.pose.camera_to_world();
        let q = camera_to_world.rotation.quaternion();
        let t = camera_to_world.translation;
        format!(
            "{} {} {} {} {} {} {} {}",
            self.frame_id,
            export_f64(t.x),
            export_f64(t.y),
            export_f64(t.z),
            export_f64(q.i),
            export_f64(q.j),
            export_f64(q.k),
            export_f64(q.w)
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseTrajectory {
    samples: Vec<TrajectorySample>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TumTrajectoryParseError {
    pub line_number: usize,
    pub line: String,
    pub message: String,
}

impl TumTrajectoryParseError {
    fn new(line_number: usize, line: &str, message: impl Into<String>) -> Self {
        Self {
            line_number,
            line: line.to_string(),
            message: message.into(),
        }
    }
}

impl fmt::Display for TumTrajectoryParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid TUM trajectory line {}: {} ({})",
            self.line_number, self.line, self.message
        )
    }
}

impl std::error::Error for TumTrajectoryParseError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KittiTrajectoryParseError {
    pub line_number: usize,
    pub line: String,
    pub message: String,
}

impl KittiTrajectoryParseError {
    fn new(line_number: usize, line: &str, message: impl Into<String>) -> Self {
        Self {
            line_number,
            line: line.to_string(),
            message: message.into(),
        }
    }
}

impl fmt::Display for KittiTrajectoryParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid KITTI trajectory line {}: {} ({})",
            self.line_number, self.line, self.message
        )
    }
}

impl std::error::Error for KittiTrajectoryParseError {}

#[derive(Debug)]
pub enum TrajectoryFileError {
    Io(std::io::Error),
    TumParse(TumTrajectoryParseError),
    KittiParse(KittiTrajectoryParseError),
}

impl fmt::Display for TrajectoryFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "trajectory file I/O error: {error}"),
            Self::TumParse(error) => write!(formatter, "{error}"),
            Self::KittiParse(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for TrajectoryFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::TumParse(error) => Some(error),
            Self::KittiParse(error) => Some(error),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectorySummary {
    pub pose_count: usize,
    pub first_frame_id: Option<FrameId>,
    pub last_frame_id: Option<FrameId>,
    pub total_path_length: f64,
    pub mean_inlier_count: Option<f64>,
    pub mean_inlier_ratio: Option<f64>,
    pub mean_reprojection_error: Option<f64>,
    pub min_camera_center_world: Option<[f64; 3]>,
    pub max_camera_center_world: Option<[f64; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryTranslationError {
    pub frame_id: FrameId,
    pub translation_error: f64,
}

impl TrajectoryTranslationError {
    pub fn to_csv_record(&self) -> String {
        format!("{},{}", self.frame_id, self.translation_error)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TrajectoryAlignment {
    #[default]
    None,
    FirstMatchedTranslation,
    /// Rigid SE(3) alignment via Umeyama (1991): finds the rotation +
    /// translation that minimises the sum of squared distances between
    /// frame-id-matched camera centres. Scale is fixed at 1.
    Umeyama,
    /// Similarity SO(3) + scale + translation alignment via Umeyama
    /// (1991): the standard ATE protocol used to compare monocular
    /// / scale-ambiguous SLAM systems against ground truth.
    UmeyamaWithScale,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectorySimilarityTransform {
    pub scale: f64,
    pub rotation: Rotation3<f64>,
    pub translation: Vector3<f64>,
}

impl TrajectorySimilarityTransform {
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            rotation: Rotation3::identity(),
            translation: Vector3::zeros(),
        }
    }

    pub fn pure_translation(translation: Vector3<f64>) -> Self {
        Self {
            scale: 1.0,
            rotation: Rotation3::identity(),
            translation,
        }
    }

    pub fn apply(&self, point: &Point3<f64>) -> Point3<f64> {
        let v = self.rotation * (point.coords * self.scale) + self.translation;
        Point3::from(v)
    }
}

/// Closed-form similarity registration (Umeyama 1991).
///
/// Returns the transform `T = (scale, R, t)` that minimises
/// `sum_i || T(source_i) - target_i ||^2`. When `with_scale` is false
/// the scale is fixed at 1 (rigid SE(3) Procrustes). Returns `None`
/// when there are fewer than two correspondences or the source point
/// cloud has zero spread.
pub fn umeyama_similarity_transform(
    source: &[Point3<f64>],
    target: &[Point3<f64>],
    with_scale: bool,
) -> Option<TrajectorySimilarityTransform> {
    if source.len() != target.len() || source.len() < 2 {
        return None;
    }
    let n = source.len() as f64;
    let source_centroid = source
        .iter()
        .fold(Vector3::<f64>::zeros(), |acc, p| acc + p.coords)
        / n;
    let target_centroid = target
        .iter()
        .fold(Vector3::<f64>::zeros(), |acc, p| acc + p.coords)
        / n;

    let mut sigma = Matrix3::<f64>::zeros();
    let mut source_variance = 0.0f64;
    for (s, t) in source.iter().zip(target.iter()) {
        let xs = s.coords - source_centroid;
        let xt = t.coords - target_centroid;
        sigma += xt * xs.transpose();
        source_variance += xs.norm_squared();
    }
    sigma /= n;
    source_variance /= n;
    if source_variance <= f64::EPSILON {
        return None;
    }

    let svd = sigma.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let singular_values = svd.singular_values;

    let det_uv = u.determinant() * v_t.determinant();
    let mut s_diag = Vector3::new(1.0, 1.0, 1.0);
    if det_uv < 0.0 {
        s_diag.z = -1.0;
    }
    let s_matrix = Matrix3::from_diagonal(&s_diag);
    let rotation_matrix = u * s_matrix * v_t;
    let rotation = Rotation3::from_matrix_unchecked(rotation_matrix);

    let scale = if with_scale {
        let trace_ds = singular_values.x * s_diag.x
            + singular_values.y * s_diag.y
            + singular_values.z * s_diag.z;
        trace_ds / source_variance
    } else {
        1.0
    };

    let translation = target_centroid - rotation * (source_centroid * scale);

    Some(TrajectorySimilarityTransform {
        scale,
        rotation,
        translation,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryErrorSummary {
    pub estimated_pose_count: usize,
    pub reference_pose_count: usize,
    pub matched_pose_count: usize,
    pub missing_reference_count: usize,
    pub missing_estimate_count: usize,
    pub mean_translation_error: Option<f64>,
    pub rmse_translation_error: Option<f64>,
    pub max_translation_error: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrajectoryEvaluationConfig {
    pub max_mean_translation_error: Option<f64>,
    pub max_rmse_translation_error: Option<f64>,
    pub max_max_translation_error: Option<f64>,
    pub min_matched_pose_count: Option<usize>,
    pub min_match_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KittiOdometryBenchmarkConfig {
    /// Segment lengths in meters used by the KITTI odometry leaderboard.
    /// The current public KITTI leaderboard reports `[100, 200, ..., 800]`.
    /// Older devkit/debug runs often use short windows such as
    /// `[5, 10, 50, 100, 150, ..., 400]`; pass those explicitly when a
    /// sequence is too short for the official windows.
    pub segment_lengths_m: Vec<f64>,
    /// Frame stride for start indices. KITTI's devkit uses every frame.
    pub start_frame_step: usize,
}

impl Default for KittiOdometryBenchmarkConfig {
    fn default() -> Self {
        Self {
            segment_lengths_m: vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            start_frame_step: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KittiOdometrySegmentError {
    pub first_frame_id: FrameId,
    pub last_frame_id: FrameId,
    pub length_m: f64,
    /// KITTI translational error for this segment, as a fraction of segment
    /// length. Multiply by 100 for percent.
    pub translational_error_ratio: f64,
    /// KITTI rotational error for this segment in degrees per meter.
    pub rotational_error_deg_per_m: f64,
}

impl KittiOdometrySegmentError {
    pub fn translational_error_percent(&self) -> f64 {
        self.translational_error_ratio * 100.0
    }

    pub fn to_csv_record(&self) -> String {
        format!(
            "{},{},{},{},{},{}",
            self.first_frame_id,
            self.last_frame_id,
            self.length_m,
            self.translational_error_ratio,
            self.translational_error_percent(),
            self.rotational_error_deg_per_m
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KittiOdometryBenchmarkSummary {
    pub estimated_pose_count: usize,
    pub reference_pose_count: usize,
    pub matched_pose_count: usize,
    pub segment_count: usize,
    pub mean_translational_error_percent: Option<f64>,
    pub mean_rotational_error_deg_per_m: Option<f64>,
    pub segment_errors: Vec<KittiOdometrySegmentError>,
}

impl KittiOdometryBenchmarkSummary {
    pub fn max_translational_error_percent(&self) -> Option<f64> {
        self.segment_errors
            .iter()
            .map(KittiOdometrySegmentError::translational_error_percent)
            .reduce(f64::max)
    }

    pub fn max_rotational_error_deg_per_m(&self) -> Option<f64> {
        self.segment_errors
            .iter()
            .map(|error| error.rotational_error_deg_per_m)
            .reduce(f64::max)
    }

    pub fn segment_errors_csv(&self) -> String {
        let mut output = String::from(
            "first_frame_id,last_frame_id,length_m,translational_error_ratio,translational_error_percent,rotational_error_deg_per_m\n",
        );
        for error in &self.segment_errors {
            output.push_str(&error.to_csv_record());
            output.push('\n');
        }
        output
    }

    pub fn to_json(&self) -> String {
        let mut output = String::new();
        writeln!(&mut output, "{{").unwrap();
        writeln!(
            &mut output,
            "  \"estimated_pose_count\": {},",
            self.estimated_pose_count
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"reference_pose_count\": {},",
            self.reference_pose_count
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"matched_pose_count\": {},",
            self.matched_pose_count
        )
        .unwrap();
        writeln!(&mut output, "  \"segment_count\": {},", self.segment_count).unwrap();
        writeln!(
            &mut output,
            "  \"mean_translational_error_percent\": {},",
            optional_f64_json(self.mean_translational_error_percent)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"mean_rotational_error_deg_per_m\": {},",
            optional_f64_json(self.mean_rotational_error_deg_per_m)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"max_translational_error_percent\": {},",
            optional_f64_json(self.max_translational_error_percent())
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"max_rotational_error_deg_per_m\": {}",
            optional_f64_json(self.max_rotational_error_deg_per_m())
        )
        .unwrap();
        output.push_str("}\n");
        output
    }

    pub fn write_segment_errors_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.segment_errors_csv())
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

/// Configuration for the TUM-style relative pose error (RPE).
///
/// RPE (Sturm et al., 2012) measures *local* drift: for each pair of poses a
/// fixed number of steps apart it compares the relative motion of the estimate
/// against the reference. Unlike the absolute trajectory error (ATE), RPE needs
/// no global alignment — a relative motion is invariant to any rigid transform
/// applied to the whole trajectory — which makes it the canonical companion
/// metric to ATE for SLAM back-ends and visual odometry front-ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelativePoseErrorConfig {
    /// Step gap Δ between the two poses of each relative-motion pair, counted in
    /// matched (frame-id-intersected) trajectory steps. `delta = 1` reports the
    /// per-step drift; larger values probe drift accumulated over longer
    /// windows.
    pub delta: usize,
    /// Stride between successive start indices. `1` evaluates every pose.
    pub start_step: usize,
}

impl Default for RelativePoseErrorConfig {
    fn default() -> Self {
        Self {
            delta: 1,
            start_step: 1,
        }
    }
}

/// One relative-pose-error sample over a single Δ-spaced pair.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativePoseError {
    pub first_frame_id: FrameId,
    pub last_frame_id: FrameId,
    /// Translational drift of the relative motion, in meters.
    pub translation_error: f64,
    /// Rotational drift of the relative motion, in degrees.
    pub rotation_error_deg: f64,
}

impl RelativePoseError {
    pub fn to_csv_record(&self) -> String {
        format!(
            "{},{},{},{}",
            self.first_frame_id,
            self.last_frame_id,
            self.translation_error,
            self.rotation_error_deg
        )
    }
}

/// Distribution statistics for one RPE channel (translation or rotation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativePoseErrorStatistics {
    /// Root-mean-square error — the headline RPE figure reported in the TUM
    /// protocol.
    pub rmse: f64,
    pub mean: f64,
    pub median: f64,
    /// Population standard deviation (`rmse² = mean² + std²`).
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl RelativePoseErrorStatistics {
    /// Statistics of a non-empty error sample. Returns `None` when `values` is
    /// empty.
    fn from_values(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let sum_sq = values.iter().map(|v| v * v).sum::<f64>();
        let rmse = (sum_sq / n).sqrt();
        let std = (sum_sq / n - mean * mean).max(0.0).sqrt();
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let middle = sorted.len() / 2;
        let median = if sorted.len() % 2 == 1 {
            sorted[middle]
        } else {
            0.5 * (sorted[middle - 1] + sorted[middle])
        };
        Some(Self {
            rmse,
            mean,
            median,
            std,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        })
    }
}

/// Aggregated TUM-style relative pose error over all Δ-spaced pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativePoseErrorSummary {
    pub estimated_pose_count: usize,
    pub reference_pose_count: usize,
    pub matched_pose_count: usize,
    /// The step gap Δ used (clamped to at least 1).
    pub delta: usize,
    pub pair_count: usize,
    /// Translational error statistics in meters (`None` when no pair exists).
    pub translation: Option<RelativePoseErrorStatistics>,
    /// Rotational error statistics in degrees (`None` when no pair exists).
    pub rotation_deg: Option<RelativePoseErrorStatistics>,
    pub errors: Vec<RelativePoseError>,
}

impl RelativePoseErrorSummary {
    pub fn errors_csv(&self) -> String {
        let mut output =
            String::from("first_frame_id,last_frame_id,translation_error,rotation_error_deg\n");
        for error in &self.errors {
            output.push_str(&error.to_csv_record());
            output.push('\n');
        }
        output
    }

    pub fn write_errors_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.errors_csv())
    }

    pub fn to_json(&self) -> String {
        let mut output = String::new();
        writeln!(&mut output, "{{").unwrap();
        writeln!(
            &mut output,
            "  \"estimated_pose_count\": {},",
            self.estimated_pose_count
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"reference_pose_count\": {},",
            self.reference_pose_count
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"matched_pose_count\": {},",
            self.matched_pose_count
        )
        .unwrap();
        writeln!(&mut output, "  \"delta\": {},", self.delta).unwrap();
        writeln!(&mut output, "  \"pair_count\": {},", self.pair_count).unwrap();
        writeln!(
            &mut output,
            "  \"translation_m\": {},",
            relative_pose_error_statistics_json(self.translation)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"rotation_deg\": {}",
            relative_pose_error_statistics_json(self.rotation_deg)
        )
        .unwrap();
        output.push_str("}\n");
        output
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryEvaluationResult {
    pub passed: bool,
    pub summary: TrajectoryErrorSummary,
    pub config: TrajectoryEvaluationConfig,
    pub match_ratio: Option<f64>,
    pub failures: Vec<TrajectoryEvaluationFailure>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrajectoryEvaluationFailure {
    MeanTranslationErrorTooHigh { actual: f64, maximum: f64 },
    RmseTranslationErrorTooHigh { actual: f64, maximum: f64 },
    MaxTranslationErrorTooHigh { actual: f64, maximum: f64 },
    NotEnoughMatchedPoses { actual: usize, minimum: usize },
    MatchRatioTooLow { actual: f64, minimum: f64 },
    NoMatchedPoses,
}

impl TrajectoryErrorSummary {
    pub fn from_trajectories(estimated: &PoseTrajectory, reference: &PoseTrajectory) -> Self {
        let errors = estimated.translation_errors_against(reference);
        Self::from_translation_errors(estimated, reference, &errors)
    }

    fn from_translation_errors(
        estimated: &PoseTrajectory,
        reference: &PoseTrajectory,
        errors: &[TrajectoryTranslationError],
    ) -> Self {
        let matched_pose_count = errors.len();
        let estimated_ids = estimated.frame_id_set();
        let reference_ids = reference.frame_id_set();
        let missing_reference_count = estimated_ids.difference(&reference_ids).count();
        let missing_estimate_count = reference_ids.difference(&estimated_ids).count();

        let (mean_translation_error, rmse_translation_error, max_translation_error) =
            if errors.is_empty() {
                (None, None, None)
            } else {
                let sum = errors
                    .iter()
                    .map(|error| error.translation_error)
                    .sum::<f64>();
                let squared_sum = errors
                    .iter()
                    .map(|error| error.translation_error * error.translation_error)
                    .sum::<f64>();
                let max = errors
                    .iter()
                    .map(|error| error.translation_error)
                    .fold(0.0_f64, f64::max);
                (
                    Some(sum / matched_pose_count as f64),
                    Some((squared_sum / matched_pose_count as f64).sqrt()),
                    Some(max),
                )
            };

        Self {
            estimated_pose_count: estimated.len(),
            reference_pose_count: reference.len(),
            matched_pose_count,
            missing_reference_count,
            missing_estimate_count,
            mean_translation_error,
            rmse_translation_error,
            max_translation_error,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"estimated_pose_count\": {},\n",
                "  \"reference_pose_count\": {},\n",
                "  \"matched_pose_count\": {},\n",
                "  \"missing_reference_count\": {},\n",
                "  \"missing_estimate_count\": {},\n",
                "  \"mean_translation_error\": {},\n",
                "  \"rmse_translation_error\": {},\n",
                "  \"max_translation_error\": {}\n",
                "}}\n"
            ),
            self.estimated_pose_count,
            self.reference_pose_count,
            self.matched_pose_count,
            self.missing_reference_count,
            self.missing_estimate_count,
            optional_f64_json(self.mean_translation_error),
            optional_f64_json(self.rmse_translation_error),
            optional_f64_json(self.max_translation_error)
        )
    }

    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"estimated_pose_count\": {}, ",
                "\"reference_pose_count\": {}, ",
                "\"matched_pose_count\": {}, ",
                "\"missing_reference_count\": {}, ",
                "\"missing_estimate_count\": {}, ",
                "\"mean_translation_error\": {}, ",
                "\"rmse_translation_error\": {}, ",
                "\"max_translation_error\": {}",
                "}}"
            ),
            self.estimated_pose_count,
            self.reference_pose_count,
            self.matched_pose_count,
            self.missing_reference_count,
            self.missing_estimate_count,
            optional_f64_json(self.mean_translation_error),
            optional_f64_json(self.rmse_translation_error),
            optional_f64_json(self.max_translation_error)
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn evaluate(&self, config: TrajectoryEvaluationConfig) -> TrajectoryEvaluationResult {
        TrajectoryEvaluationResult::from_summary(self.clone(), config)
    }
}

impl TrajectoryEvaluationResult {
    pub fn from_summary(
        summary: TrajectoryErrorSummary,
        config: TrajectoryEvaluationConfig,
    ) -> Self {
        let match_ratio = if summary.reference_pose_count == 0 {
            None
        } else {
            Some(summary.matched_pose_count as f64 / summary.reference_pose_count as f64)
        };
        let mut failures = Vec::new();

        if summary.matched_pose_count == 0 {
            failures.push(TrajectoryEvaluationFailure::NoMatchedPoses);
        }

        if let Some(maximum) = config.max_mean_translation_error {
            match summary.mean_translation_error {
                Some(actual) if actual > maximum => {
                    failures.push(TrajectoryEvaluationFailure::MeanTranslationErrorTooHigh {
                        actual,
                        maximum,
                    })
                }
                None => failures.push(TrajectoryEvaluationFailure::NoMatchedPoses),
                _ => {}
            }
        }

        if let Some(maximum) = config.max_rmse_translation_error {
            match summary.rmse_translation_error {
                Some(actual) if actual > maximum => {
                    failures.push(TrajectoryEvaluationFailure::RmseTranslationErrorTooHigh {
                        actual,
                        maximum,
                    })
                }
                None => failures.push(TrajectoryEvaluationFailure::NoMatchedPoses),
                _ => {}
            }
        }

        if let Some(maximum) = config.max_max_translation_error {
            match summary.max_translation_error {
                Some(actual) if actual > maximum => {
                    failures.push(TrajectoryEvaluationFailure::MaxTranslationErrorTooHigh {
                        actual,
                        maximum,
                    })
                }
                None => failures.push(TrajectoryEvaluationFailure::NoMatchedPoses),
                _ => {}
            }
        }

        if let Some(minimum) = config.min_matched_pose_count {
            if summary.matched_pose_count < minimum {
                failures.push(TrajectoryEvaluationFailure::NotEnoughMatchedPoses {
                    actual: summary.matched_pose_count,
                    minimum,
                });
            }
        }

        if let Some(minimum) = config.min_match_ratio {
            match match_ratio {
                Some(actual) if actual < minimum => {
                    failures.push(TrajectoryEvaluationFailure::MatchRatioTooLow { actual, minimum })
                }
                None => failures.push(TrajectoryEvaluationFailure::MatchRatioTooLow {
                    actual: 0.0,
                    minimum,
                }),
                _ => {}
            }
        }

        failures.dedup();
        let passed = failures.is_empty();

        Self {
            passed,
            summary,
            config,
            match_ratio,
            failures,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"match_ratio\": {},\n",
                "  \"config\": {},\n",
                "  \"summary\": {},\n",
                "  \"failures\": {}\n",
                "}}\n"
            ),
            self.passed,
            optional_f64_json(self.match_ratio),
            self.config.to_json_inline(),
            self.summary.to_json_inline(),
            trajectory_evaluation_failures_json(&self.failures)
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

impl TrajectoryEvaluationConfig {
    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"max_mean_translation_error\": {}, ",
                "\"max_rmse_translation_error\": {}, ",
                "\"max_max_translation_error\": {}, ",
                "\"min_matched_pose_count\": {}, ",
                "\"min_match_ratio\": {}",
                "}}"
            ),
            optional_f64_json(self.max_mean_translation_error),
            optional_f64_json(self.max_rmse_translation_error),
            optional_f64_json(self.max_max_translation_error),
            optional_usize_json(self.min_matched_pose_count),
            optional_f64_json(self.min_match_ratio)
        )
    }
}

impl TrajectoryEvaluationFailure {
    pub fn reason(&self) -> &'static str {
        match self {
            Self::MeanTranslationErrorTooHigh { .. } => "mean_translation_error_too_high",
            Self::RmseTranslationErrorTooHigh { .. } => "rmse_translation_error_too_high",
            Self::MaxTranslationErrorTooHigh { .. } => "max_translation_error_too_high",
            Self::NotEnoughMatchedPoses { .. } => "not_enough_matched_poses",
            Self::MatchRatioTooLow { .. } => "match_ratio_too_low",
            Self::NoMatchedPoses => "no_matched_poses",
        }
    }

    fn to_json_inline(&self) -> String {
        match self {
            Self::MeanTranslationErrorTooHigh { actual, maximum }
            | Self::RmseTranslationErrorTooHigh { actual, maximum }
            | Self::MaxTranslationErrorTooHigh { actual, maximum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"maximum\": {}}}",
                self.reason(),
                actual,
                maximum
            ),
            Self::NotEnoughMatchedPoses { actual, minimum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"minimum\": {}}}",
                self.reason(),
                actual,
                minimum
            ),
            Self::MatchRatioTooLow { actual, minimum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"minimum\": {}}}",
                self.reason(),
                actual,
                minimum
            ),
            Self::NoMatchedPoses => {
                format!("{{\"reason\": \"{}\"}}", self.reason())
            }
        }
    }
}

impl TrajectorySummary {
    pub fn from_trajectory(trajectory: &PoseTrajectory) -> Self {
        let pose_count = trajectory.len();
        let first_frame_id = trajectory.samples.first().map(|sample| sample.frame_id);
        let last_frame_id = trajectory.samples.last().map(|sample| sample.frame_id);
        let total_path_length = trajectory.total_path_length();
        let mean_reprojection_error = trajectory.mean_reprojection_error();

        let (mean_inlier_count, mean_inlier_ratio) = if pose_count == 0 {
            (None, None)
        } else {
            let inlier_count_sum: usize = trajectory
                .samples
                .iter()
                .map(|sample| sample.inlier_count)
                .sum();
            let inlier_ratio_sum: f64 = trajectory
                .samples
                .iter()
                .map(|sample| sample.inlier_ratio)
                .sum();
            (
                Some(inlier_count_sum as f64 / pose_count as f64),
                Some(inlier_ratio_sum / pose_count as f64),
            )
        };

        let (min_camera_center_world, max_camera_center_world) =
            trajectory.camera_center_bounds_world();

        Self {
            pose_count,
            first_frame_id,
            last_frame_id,
            total_path_length,
            mean_inlier_count,
            mean_inlier_ratio,
            mean_reprojection_error,
            min_camera_center_world,
            max_camera_center_world,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"pose_count\": {},\n",
                "  \"first_frame_id\": {},\n",
                "  \"last_frame_id\": {},\n",
                "  \"total_path_length\": {},\n",
                "  \"mean_inlier_count\": {},\n",
                "  \"mean_inlier_ratio\": {},\n",
                "  \"mean_reprojection_error\": {},\n",
                "  \"min_camera_center_world\": {},\n",
                "  \"max_camera_center_world\": {}\n",
                "}}\n"
            ),
            self.pose_count,
            optional_frame_id_json(self.first_frame_id),
            optional_frame_id_json(self.last_frame_id),
            self.total_path_length,
            optional_f64_json(self.mean_inlier_count),
            optional_f64_json(self.mean_inlier_ratio),
            optional_f64_json(self.mean_reprojection_error),
            optional_vec3_json(self.min_camera_center_world),
            optional_vec3_json(self.max_camera_center_world)
        )
    }
}

impl PoseTrajectory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_tracking_results(results: &[TrackingResult]) -> Self {
        let mut trajectory = Self::new();
        for result in results {
            trajectory.push_result(result);
        }
        trajectory
    }

    pub fn from_tum_poses_str(text: &str) -> Result<Self, TumTrajectoryParseError> {
        let mut trajectory = Self::new();
        for (line_index, line) in text.lines().enumerate() {
            let line_number = line_index + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let fields = trimmed.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 8 {
                return Err(TumTrajectoryParseError::new(
                    line_number,
                    line,
                    format!("expected 8 fields, got {}", fields.len()),
                ));
            }

            let frame_id = parse_tum_frame_id(fields[0], line_number, line)?;
            let tx = parse_tum_f64(fields[1], "tx", line_number, line)?;
            let ty = parse_tum_f64(fields[2], "ty", line_number, line)?;
            let tz = parse_tum_f64(fields[3], "tz", line_number, line)?;
            let qx = parse_tum_f64(fields[4], "qx", line_number, line)?;
            let qy = parse_tum_f64(fields[5], "qy", line_number, line)?;
            let qz = parse_tum_f64(fields[6], "qz", line_number, line)?;
            let qw = parse_tum_f64(fields[7], "qw", line_number, line)?;
            let Some(rotation) = UnitQuaternion::try_new(Quaternion::new(qw, qx, qy, qz), 1.0e-12)
            else {
                return Err(TumTrajectoryParseError::new(
                    line_number,
                    line,
                    "quaternion norm is too small",
                ));
            };

            let camera_to_world = SE3::new(rotation, Vector3::new(tx, ty, tz));
            trajectory.push_sample(TrajectorySample {
                frame_id,
                pose: Pose {
                    world_to_camera: camera_to_world.inverse(),
                },
                state: TrackingState::Tracking,
                event: TrackingEvent::Tracked,
                inlier_count: 0,
                inlier_ratio: 0.0,
                reprojection_error: None,
            });
        }
        Ok(trajectory)
    }

    pub fn read_tum_poses(path: impl AsRef<Path>) -> Result<Self, TrajectoryFileError> {
        let text = std::fs::read_to_string(path).map_err(TrajectoryFileError::Io)?;
        Self::from_tum_poses_str(&text).map_err(TrajectoryFileError::TumParse)
    }

    pub fn from_kitti_poses_str(text: &str) -> Result<Self, KittiTrajectoryParseError> {
        let mut trajectory = Self::new();
        let mut pose_index = 0_u64;
        for (line_index, line) in text.lines().enumerate() {
            let line_number = line_index + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let fields = trimmed.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 12 {
                return Err(KittiTrajectoryParseError::new(
                    line_number,
                    line,
                    format!("expected 12 fields, got {}", fields.len()),
                ));
            }

            let mut values = [0.0_f64; 12];
            for (index, field) in fields.iter().enumerate() {
                values[index] = parse_kitti_f64(field, index, line_number, line)?;
            }

            let rotation = Matrix3::new(
                values[0], values[1], values[2], values[4], values[5], values[6], values[8],
                values[9], values[10],
            );
            let translation = Vector3::new(values[3], values[7], values[11]);
            let camera_to_world = SE3::new(
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation)),
                translation,
            );
            trajectory.push_sample(TrajectorySample {
                frame_id: pose_index,
                pose: Pose {
                    world_to_camera: camera_to_world.inverse(),
                },
                state: TrackingState::Tracking,
                event: TrackingEvent::Tracked,
                inlier_count: 0,
                inlier_ratio: 0.0,
                reprojection_error: None,
            });
            pose_index += 1;
        }
        Ok(trajectory)
    }

    pub fn read_kitti_poses(path: impl AsRef<Path>) -> Result<Self, TrajectoryFileError> {
        let text = std::fs::read_to_string(path).map_err(TrajectoryFileError::Io)?;
        Self::from_kitti_poses_str(&text).map_err(TrajectoryFileError::KittiParse)
    }

    pub fn push_result(&mut self, result: &TrackingResult) -> bool {
        if let Some(sample) = TrajectorySample::from_tracking_result(result) {
            self.samples.push(sample);
            true
        } else {
            false
        }
    }

    pub fn push_sample(&mut self, sample: TrajectorySample) {
        self.samples.push(sample);
    }

    pub fn samples(&self) -> &[TrajectorySample] {
        &self.samples
    }

    pub fn into_samples(self) -> Vec<TrajectorySample> {
        self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn frame_ids(&self) -> Vec<FrameId> {
        self.samples.iter().map(|sample| sample.frame_id).collect()
    }

    fn frame_id_set(&self) -> HashSet<FrameId> {
        self.samples.iter().map(|sample| sample.frame_id).collect()
    }

    pub fn camera_centers_world(&self) -> Vec<Point3<f64>> {
        self.samples
            .iter()
            .map(TrajectorySample::camera_center_world)
            .collect()
    }

    pub fn camera_center_bounds_world(&self) -> (Option<[f64; 3]>, Option<[f64; 3]>) {
        let Some(first) = self.samples.first() else {
            return (None, None);
        };

        let first_center = first.camera_center_world();
        let mut min = [first_center.x, first_center.y, first_center.z];
        let mut max = min;

        for sample in self.samples.iter().skip(1) {
            let center = sample.camera_center_world();
            min[0] = min[0].min(center.x);
            min[1] = min[1].min(center.y);
            min[2] = min[2].min(center.z);
            max[0] = max[0].max(center.x);
            max[1] = max[1].max(center.y);
            max[2] = max[2].max(center.z);
        }

        (Some(min), Some(max))
    }

    fn trajectory_alignment_transform(
        &self,
        reference_by_frame_id: &HashMap<FrameId, Point3<f64>>,
        alignment: TrajectoryAlignment,
    ) -> TrajectorySimilarityTransform {
        match alignment {
            TrajectoryAlignment::None => TrajectorySimilarityTransform::identity(),
            TrajectoryAlignment::FirstMatchedTranslation => self
                .samples
                .iter()
                .find_map(|sample| {
                    let reference_center = reference_by_frame_id.get(&sample.frame_id)?;
                    Some(TrajectorySimilarityTransform::pure_translation(
                        reference_center.coords - sample.camera_center_world().coords,
                    ))
                })
                .unwrap_or_else(TrajectorySimilarityTransform::identity),
            TrajectoryAlignment::Umeyama | TrajectoryAlignment::UmeyamaWithScale => {
                let mut source = Vec::with_capacity(self.samples.len());
                let mut target = Vec::with_capacity(self.samples.len());
                for sample in &self.samples {
                    if let Some(reference_center) = reference_by_frame_id.get(&sample.frame_id) {
                        source.push(sample.camera_center_world());
                        target.push(*reference_center);
                    }
                }
                let with_scale = matches!(alignment, TrajectoryAlignment::UmeyamaWithScale);
                umeyama_similarity_transform(&source, &target, with_scale)
                    .unwrap_or_else(TrajectorySimilarityTransform::identity)
            }
        }
    }

    pub fn total_path_length(&self) -> f64 {
        self.samples
            .windows(2)
            .map(|window| {
                (window[1].camera_center_world() - window[0].camera_center_world()).norm()
            })
            .sum()
    }

    pub fn kitti_odometry_benchmark_against(
        &self,
        reference: &PoseTrajectory,
        config: &KittiOdometryBenchmarkConfig,
    ) -> KittiOdometryBenchmarkSummary {
        let estimated_by_frame_id: HashMap<FrameId, &TrajectorySample> = self
            .samples
            .iter()
            .map(|sample| (sample.frame_id, sample))
            .collect();
        let matched: Vec<(&TrajectorySample, &TrajectorySample)> = reference
            .samples
            .iter()
            .filter_map(|reference_sample| {
                let estimated_sample = estimated_by_frame_id.get(&reference_sample.frame_id)?;
                Some((*estimated_sample, reference_sample))
            })
            .collect();

        let distances = cumulative_reference_distances(&matched);
        let mut segment_errors = Vec::new();
        let start_step = config.start_frame_step.max(1);
        for first_index in (0..matched.len()).step_by(start_step) {
            let first_distance = distances[first_index];
            for &length_m in &config.segment_lengths_m {
                if !(length_m.is_finite() && length_m > 0.0) {
                    continue;
                }
                let target_distance = first_distance + length_m;
                let Some(last_index) = first_index_for_distance(&distances, target_distance) else {
                    continue;
                };
                if last_index <= first_index {
                    continue;
                }
                let (estimated_first, reference_first) = matched[first_index];
                let (estimated_last, reference_last) = matched[last_index];
                let estimated_delta = relative_camera_to_world(estimated_first, estimated_last);
                let reference_delta = relative_camera_to_world(reference_first, reference_last);
                let error = estimated_delta.inverse().compose(&reference_delta);
                let translation_error = error.translation.norm() / length_m;
                let rotation_error = error.rotation.angle().to_degrees() / length_m;
                segment_errors.push(KittiOdometrySegmentError {
                    first_frame_id: reference_first.frame_id,
                    last_frame_id: reference_last.frame_id,
                    length_m,
                    translational_error_ratio: translation_error,
                    rotational_error_deg_per_m: rotation_error,
                });
            }
        }

        let mean_translational_error_percent = if segment_errors.is_empty() {
            None
        } else {
            Some(
                segment_errors
                    .iter()
                    .map(KittiOdometrySegmentError::translational_error_percent)
                    .sum::<f64>()
                    / segment_errors.len() as f64,
            )
        };
        let mean_rotational_error_deg_per_m = if segment_errors.is_empty() {
            None
        } else {
            Some(
                segment_errors
                    .iter()
                    .map(|error| error.rotational_error_deg_per_m)
                    .sum::<f64>()
                    / segment_errors.len() as f64,
            )
        };
        KittiOdometryBenchmarkSummary {
            estimated_pose_count: self.len(),
            reference_pose_count: reference.len(),
            matched_pose_count: matched.len(),
            segment_count: segment_errors.len(),
            mean_translational_error_percent,
            mean_rotational_error_deg_per_m,
            segment_errors,
        }
    }

    /// Compute the TUM-style relative pose error (RPE) of this trajectory
    /// against `reference`.
    ///
    /// Poses are matched by frame id; the i-th and (i+Δ)-th matched poses form
    /// each relative-motion pair, where Δ is `config.delta`. For each pair the
    /// relative camera motion of the estimate (`ΔP`) and of the reference
    /// (`ΔQ`) are compared via the residual `E = ΔQ⁻¹ · ΔP`, contributing its
    /// translation norm (m) and rotation angle (deg). Because relative motion
    /// is invariant to a global rigid transform of either trajectory, RPE needs
    /// no prior alignment — its defining advantage over the ATE.
    pub fn relative_pose_error_against(
        &self,
        reference: &PoseTrajectory,
        config: &RelativePoseErrorConfig,
    ) -> RelativePoseErrorSummary {
        let estimated_by_frame_id: HashMap<FrameId, &TrajectorySample> = self
            .samples
            .iter()
            .map(|sample| (sample.frame_id, sample))
            .collect();
        let matched: Vec<(&TrajectorySample, &TrajectorySample)> = reference
            .samples
            .iter()
            .filter_map(|reference_sample| {
                let estimated_sample = estimated_by_frame_id.get(&reference_sample.frame_id)?;
                Some((*estimated_sample, reference_sample))
            })
            .collect();

        let delta = config.delta.max(1);
        let start_step = config.start_step.max(1);
        let mut errors = Vec::new();
        let mut first_index = 0;
        while first_index + delta < matched.len() {
            let last_index = first_index + delta;
            let (estimated_first, reference_first) = matched[first_index];
            let (estimated_last, reference_last) = matched[last_index];
            let estimated_delta = relative_camera_to_world(estimated_first, estimated_last);
            let reference_delta = relative_camera_to_world(reference_first, reference_last);
            let residual = reference_delta.inverse().compose(&estimated_delta);
            errors.push(RelativePoseError {
                first_frame_id: reference_first.frame_id,
                last_frame_id: reference_last.frame_id,
                translation_error: residual.translation.norm(),
                rotation_error_deg: residual.rotation.angle().to_degrees(),
            });
            first_index += start_step;
        }

        let translation_values: Vec<f64> = errors.iter().map(|e| e.translation_error).collect();
        let rotation_values: Vec<f64> = errors.iter().map(|e| e.rotation_error_deg).collect();
        RelativePoseErrorSummary {
            estimated_pose_count: self.len(),
            reference_pose_count: reference.len(),
            matched_pose_count: matched.len(),
            delta,
            pair_count: errors.len(),
            translation: RelativePoseErrorStatistics::from_values(&translation_values),
            rotation_deg: RelativePoseErrorStatistics::from_values(&rotation_values),
            errors,
        }
    }

    pub fn mean_reprojection_error(&self) -> Option<f64> {
        let mut sum = 0.0;
        let mut count = 0;
        for sample in &self.samples {
            if let Some(error) = sample.reprojection_error {
                sum += error;
                count += 1;
            }
        }

        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    pub fn to_csv(&self) -> String {
        let mut output = String::from(
            "frame_id,camera_center_x,camera_center_y,camera_center_z,qw,qx,qy,qz,tx,ty,tz,state,event,inlier_count,inlier_ratio,reprojection_error\n",
        );
        for sample in &self.samples {
            output.push_str(&sample.to_csv_record());
            output.push('\n');
        }
        output
    }

    pub fn write_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_csv())
    }

    pub fn to_kitti_poses(&self) -> String {
        let mut output = String::new();
        for sample in &self.samples {
            output.push_str(&sample.to_kitti_pose_record());
            output.push('\n');
        }
        output
    }

    pub fn write_kitti_poses(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_kitti_poses())
    }

    pub fn to_tum_poses(&self) -> String {
        let mut output = String::new();
        for sample in &self.samples {
            output.push_str(&sample.to_tum_pose_record());
            output.push('\n');
        }
        output
    }

    pub fn write_tum_poses(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_tum_poses())
    }

    pub fn translation_errors_against(
        &self,
        reference: &PoseTrajectory,
    ) -> Vec<TrajectoryTranslationError> {
        self.translation_errors_against_with_alignment(reference, TrajectoryAlignment::None)
    }

    pub fn translation_errors_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
    ) -> Vec<TrajectoryTranslationError> {
        let reference_by_frame_id = reference
            .samples
            .iter()
            .map(|sample| (sample.frame_id, sample.camera_center_world()))
            .collect::<HashMap<_, _>>();
        let transform = self.trajectory_alignment_transform(&reference_by_frame_id, alignment);

        self.samples
            .iter()
            .filter_map(|sample| {
                let reference_center = reference_by_frame_id.get(&sample.frame_id)?;
                let aligned_center = transform.apply(&sample.camera_center_world());
                Some(TrajectoryTranslationError {
                    frame_id: sample.frame_id,
                    translation_error: (aligned_center - reference_center).norm(),
                })
            })
            .collect()
    }

    pub fn translation_error_summary_against(
        &self,
        reference: &PoseTrajectory,
    ) -> TrajectoryErrorSummary {
        TrajectoryErrorSummary::from_trajectories(self, reference)
    }

    pub fn translation_error_summary_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
    ) -> TrajectoryErrorSummary {
        let errors = self.translation_errors_against_with_alignment(reference, alignment);
        TrajectoryErrorSummary::from_translation_errors(self, reference, &errors)
    }

    pub fn translation_errors_csv_against(&self, reference: &PoseTrajectory) -> String {
        self.translation_errors_csv_against_with_alignment(reference, TrajectoryAlignment::None)
    }

    pub fn translation_errors_csv_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
    ) -> String {
        let mut output = String::from("frame_id,translation_error\n");
        for error in self.translation_errors_against_with_alignment(reference, alignment) {
            output.push_str(&error.to_csv_record());
            output.push('\n');
        }
        output
    }

    pub fn write_translation_errors_csv_against(
        &self,
        reference: &PoseTrajectory,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        std::fs::write(path, self.translation_errors_csv_against(reference))
    }

    pub fn write_translation_errors_csv_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        std::fs::write(
            path,
            self.translation_errors_csv_against_with_alignment(reference, alignment),
        )
    }

    pub fn write_translation_error_summary_json_against(
        &self,
        reference: &PoseTrajectory,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        self.translation_error_summary_against(reference)
            .write_json(path)
    }

    pub fn write_translation_error_summary_json_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        self.translation_error_summary_against_with_alignment(reference, alignment)
            .write_json(path)
    }

    pub fn to_html_report(&self) -> String {
        let summary = self.summary();
        let svg = trajectory_svg(self);
        let mut output = String::new();
        output.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
        output
            .push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
        output.push_str("<title>visloc-rs trajectory report</title>\n");
        output.push_str("<style>");
        output.push_str(
            "body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f9;color:#182026}\
             main{max-width:1080px;margin:0 auto;padding:28px}\
             h1{font-size:24px;margin:0 0 8px}\
             .sub{margin:0 0 22px;color:#52616b}\
             .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}\
             .metric{background:white;border:1px solid #dde3ea;border-radius:8px;padding:12px}\
             .label{display:block;font-size:12px;color:#65727e}\
             .value{display:block;font-size:22px;font-weight:700;margin-top:4px}\
             .panel{background:white;border:1px solid #dde3ea;border-radius:8px;padding:16px;margin-top:14px}\
             .legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 0;color:#52616b;font-size:13px}\
             .swatch{display:inline-block;width:26px;height:4px;border-radius:2px;vertical-align:middle;margin-right:6px}\
             code{background:#eef2f5;border-radius:4px;padding:2px 4px}\
             svg{width:100%;height:auto;display:block}",
        );
        output.push_str("</style>\n</head>\n<body>\n<main>\n");
        output.push_str("<h1>visloc-rs trajectory report</h1>\n");
        output.push_str(
            "<p class=\"sub\">Estimated camera-center trajectory from sequence localization.</p>\n",
        );
        output.push_str("<section class=\"grid\">\n");
        push_metric_card(&mut output, "Poses", &summary.pose_count.to_string());
        push_metric_card(
            &mut output,
            "First frame",
            &format_optional_frame_id(summary.first_frame_id),
        );
        push_metric_card(
            &mut output,
            "Last frame",
            &format_optional_frame_id(summary.last_frame_id),
        );
        push_metric_card(
            &mut output,
            "Path length",
            &format_optional_metric(Some(summary.total_path_length), "m"),
        );
        push_metric_card(
            &mut output,
            "Mean inliers",
            &format_optional_count(summary.mean_inlier_count),
        );
        push_metric_card(
            &mut output,
            "Mean reprojection",
            &format_optional_metric(summary.mean_reprojection_error, "px"),
        );
        output.push_str("</section>\n");
        output.push_str("<section class=\"panel\">\n");
        output.push_str(&svg);
        output.push_str(
            "<div class=\"legend\"><span><span class=\"swatch\" style=\"background:#e0574f\"></span>estimated trajectory</span></div>\n",
        );
        output.push_str("</section>\n</main>\n</body>\n</html>\n");
        output
    }

    pub fn write_html_report(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_html_report())
    }

    pub fn to_html_report_against(&self, reference: &PoseTrajectory) -> String {
        self.to_html_report_against_with_alignment(reference, TrajectoryAlignment::None)
    }

    pub fn to_html_report_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
    ) -> String {
        let summary = self.translation_error_summary_against_with_alignment(reference, alignment);
        let errors = self.translation_errors_against_with_alignment(reference, alignment);
        let svg = trajectory_comparison_svg(self, reference, alignment);
        let mut output = String::new();
        output.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
        output
            .push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
        output.push_str("<title>visloc-rs trajectory evaluation</title>\n");
        output.push_str("<style>");
        output.push_str(
            "body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f9;color:#182026}\
             main{max-width:1080px;margin:0 auto;padding:28px}\
             h1{font-size:24px;margin:0 0 8px}\
             .sub{margin:0 0 22px;color:#52616b}\
             .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}\
             .metric{background:white;border:1px solid #dde3ea;border-radius:8px;padding:12px}\
             .label{display:block;font-size:12px;color:#65727e}\
             .value{display:block;font-size:22px;font-weight:700;margin-top:4px}\
             .panel{background:white;border:1px solid #dde3ea;border-radius:8px;padding:16px;margin-top:14px}\
             .legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 0;color:#52616b;font-size:13px}\
             .swatch{display:inline-block;width:26px;height:4px;border-radius:2px;vertical-align:middle;margin-right:6px}\
             table{width:100%;border-collapse:collapse;font-size:13px}\
             th,td{text-align:right;border-bottom:1px solid #e7ecf0;padding:6px 8px}\
             th:first-child,td:first-child{text-align:left}\
             code{background:#eef2f5;border-radius:4px;padding:2px 4px}\
             svg{width:100%;height:auto;display:block}",
        );
        output.push_str("</style>\n</head>\n<body>\n<main>\n");
        output.push_str("<h1>visloc-rs trajectory evaluation</h1>\n");
        let _ = writeln!(
            output,
            "<p class=\"sub\">Estimated trajectory compared with reference poses. Alignment: <code>{alignment:?}</code></p>"
        );
        output.push_str("<section class=\"grid\">\n");
        push_metric_card(
            &mut output,
            "Estimated poses",
            &summary.estimated_pose_count.to_string(),
        );
        push_metric_card(
            &mut output,
            "Reference poses",
            &summary.reference_pose_count.to_string(),
        );
        push_metric_card(
            &mut output,
            "Matched poses",
            &summary.matched_pose_count.to_string(),
        );
        push_metric_card(
            &mut output,
            "Mean error",
            &format_optional_metric(summary.mean_translation_error, "m"),
        );
        push_metric_card(
            &mut output,
            "RMSE",
            &format_optional_metric(summary.rmse_translation_error, "m"),
        );
        push_metric_card(
            &mut output,
            "Max error",
            &format_optional_metric(summary.max_translation_error, "m"),
        );
        output.push_str("</section>\n");
        output.push_str("<section class=\"panel\">\n");
        output.push_str(&svg);
        output.push_str(
            "<div class=\"legend\"><span><span class=\"swatch\" style=\"background:#e0574f\"></span>estimated</span><span><span class=\"swatch\" style=\"background:#2676c9\"></span>reference</span><span><span class=\"swatch\" style=\"background:#9aa7b2\"></span>matched error</span></div>\n",
        );
        output.push_str("</section>\n");
        output.push_str("<section class=\"panel\">\n<h2>Matched translation errors</h2>\n");
        output.push_str(
            "<table><thead><tr><th>frame</th><th>translation error</th></tr></thead><tbody>\n",
        );
        for error in errors.iter().take(80) {
            let _ = writeln!(
                output,
                "<tr><td>{}</td><td>{}</td></tr>",
                error.frame_id,
                format_optional_metric(Some(error.translation_error), "m")
            );
        }
        if errors.len() > 80 {
            let _ = writeln!(
                output,
                "<tr><td colspan=\"2\">{} more rows omitted</td></tr>",
                errors.len() - 80
            );
        }
        output.push_str("</tbody></table>\n</section>\n</main>\n</body>\n</html>\n");
        output
    }

    pub fn write_html_report_against(
        &self,
        reference: &PoseTrajectory,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        std::fs::write(path, self.to_html_report_against(reference))
    }

    pub fn write_html_report_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        std::fs::write(
            path,
            self.to_html_report_against_with_alignment(reference, alignment),
        )
    }

    pub fn summary(&self) -> TrajectorySummary {
        TrajectorySummary::from_trajectory(self)
    }

    pub fn to_summary_json(&self) -> String {
        self.summary().to_json()
    }

    pub fn write_summary_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_summary_json())
    }
}

fn optional_f64_csv(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn push_metric_card(output: &mut String, label: &str, value: &str) {
    let _ = writeln!(
        output,
        "<div class=\"metric\"><span class=\"label\">{}</span><span class=\"value\">{}</span></div>",
        label, value
    );
}

fn format_optional_metric(value: Option<f64>, unit: &str) -> String {
    value
        .map(|value| format!("{value:.4} {unit}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_count(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.1}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_frame_id(value: Option<FrameId>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

fn trajectory_svg(trajectory: &PoseTrajectory) -> String {
    let points = trajectory.camera_centers_world();
    let projection = TrajectorySvgProjection::from_points(&points, &[]);

    let mut output = String::new();
    output.push_str("<svg viewBox=\"0 0 900 520\" role=\"img\" aria-label=\"trajectory plot\">\n");
    output.push_str("<rect x=\"0\" y=\"0\" width=\"900\" height=\"520\" fill=\"#fbfcfd\"/>\n");
    output.push_str("<g stroke=\"#e4e9ef\" stroke-width=\"1\">\n");
    for x in [80, 228, 376, 524, 672, 820] {
        let _ = writeln!(output, "<line x1=\"{x}\" y1=\"54\" x2=\"{x}\" y2=\"450\"/>");
    }
    for y in [54, 133, 212, 291, 370, 450] {
        let _ = writeln!(output, "<line x1=\"80\" y1=\"{y}\" x2=\"820\" y2=\"{y}\"/>");
    }
    output.push_str("</g>\n");
    output.push_str("<g fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n");
    push_polyline(&mut output, &points, &projection, "#e0574f", 4);
    output.push_str("</g>\n");
    push_points(&mut output, &points, &projection, "#e0574f");
    output.push_str(
        "<text x=\"80\" y=\"486\" fill=\"#65727e\" font-size=\"13\">top-down camera-center trajectory</text>\n",
    );
    output.push_str("</svg>\n");
    output
}

fn trajectory_comparison_svg(
    estimated: &PoseTrajectory,
    reference: &PoseTrajectory,
    alignment: TrajectoryAlignment,
) -> String {
    let reference_by_frame_id = reference
        .samples
        .iter()
        .map(|sample| (sample.frame_id, sample.camera_center_world()))
        .collect::<HashMap<_, _>>();
    let transform = estimated.trajectory_alignment_transform(&reference_by_frame_id, alignment);
    let estimated_points = estimated
        .samples
        .iter()
        .map(|sample| transform.apply(&sample.camera_center_world()))
        .collect::<Vec<_>>();
    let reference_points = reference.camera_centers_world();
    let projection = TrajectorySvgProjection::from_points(&estimated_points, &reference_points);

    let mut output = String::new();
    output.push_str("<svg viewBox=\"0 0 900 520\" role=\"img\" aria-label=\"trajectory plot\">\n");
    output.push_str("<rect x=\"0\" y=\"0\" width=\"900\" height=\"520\" fill=\"#fbfcfd\"/>\n");
    output.push_str("<g stroke=\"#e4e9ef\" stroke-width=\"1\">\n");
    for x in [80, 228, 376, 524, 672, 820] {
        let _ = writeln!(output, "<line x1=\"{x}\" y1=\"54\" x2=\"{x}\" y2=\"450\"/>");
    }
    for y in [54, 133, 212, 291, 370, 450] {
        let _ = writeln!(output, "<line x1=\"80\" y1=\"{y}\" x2=\"820\" y2=\"{y}\"/>");
    }
    output.push_str("</g>\n");
    output.push_str("<g fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n");
    push_polyline(&mut output, &estimated_points, &projection, "#e0574f", 4);
    push_polyline(&mut output, &reference_points, &projection, "#2676c9", 4);
    output.push_str("</g>\n");

    output.push_str("<g stroke=\"#9aa7b2\" stroke-width=\"1.5\" stroke-dasharray=\"4 5\">\n");
    for sample in &estimated.samples {
        let Some(reference_center) = reference_by_frame_id.get(&sample.frame_id) else {
            continue;
        };
        let estimated_center = transform.apply(&sample.camera_center_world());
        let (ex, ey) = projection.project(&estimated_center);
        let (rx, ry) = projection.project(reference_center);
        let _ = writeln!(
            output,
            "<line x1=\"{ex:.2}\" y1=\"{ey:.2}\" x2=\"{rx:.2}\" y2=\"{ry:.2}\"/>"
        );
    }
    output.push_str("</g>\n");
    push_points(&mut output, &reference_points, &projection, "#2676c9");
    push_points(&mut output, &estimated_points, &projection, "#e0574f");
    output.push_str(
        "<text x=\"80\" y=\"486\" fill=\"#65727e\" font-size=\"13\">top-down camera-center trajectory</text>\n",
    );
    output.push_str("</svg>\n");
    output
}

fn push_polyline(
    output: &mut String,
    points: &[Point3<f64>],
    projection: &TrajectorySvgProjection,
    color: &str,
    stroke_width: usize,
) {
    if points.is_empty() {
        return;
    }

    let mut point_text = String::new();
    for point in points {
        let (x, y) = projection.project(point);
        let _ = write!(point_text, "{x:.2},{y:.2} ");
    }
    let _ = writeln!(
        output,
        "<polyline points=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>",
        point_text.trim_end(),
        color,
        stroke_width
    );
}

fn push_points(
    output: &mut String,
    points: &[Point3<f64>],
    projection: &TrajectorySvgProjection,
    color: &str,
) {
    output.push_str("<g>\n");
    for point in points {
        let (x, y) = projection.project(point);
        let _ = writeln!(
            output,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"4\" fill=\"{color}\"/>"
        );
    }
    output.push_str("</g>\n");
}

#[derive(Debug, Clone, Copy)]
struct TrajectorySvgProjection {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    axis_y: usize,
}

impl TrajectorySvgProjection {
    fn from_points(estimated: &[Point3<f64>], reference: &[Point3<f64>]) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for point in estimated.iter().chain(reference.iter()) {
            min[0] = min[0].min(point.x);
            min[1] = min[1].min(point.y);
            min[2] = min[2].min(point.z);
            max[0] = max[0].max(point.x);
            max[1] = max[1].max(point.y);
            max[2] = max[2].max(point.z);
        }

        if !min[0].is_finite() {
            return Self {
                min_x: -1.0,
                max_x: 1.0,
                min_y: -1.0,
                max_y: 1.0,
                axis_y: 2,
            };
        }

        let spread_y = max[1] - min[1];
        let spread_z = max[2] - min[2];
        let axis_y = if spread_z >= spread_y { 2 } else { 1 };
        let (mut min_x, mut max_x) = padded_range(min[0], max[0]);
        let (mut min_y, mut max_y) = padded_range(min[axis_y], max[axis_y]);
        let x_span = max_x - min_x;
        let y_span = max_y - min_y;
        if x_span > y_span {
            let delta = (x_span - y_span) * 0.5;
            min_y -= delta;
            max_y += delta;
        } else {
            let delta = (y_span - x_span) * 0.5;
            min_x -= delta;
            max_x += delta;
        }

        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            axis_y,
        }
    }

    fn project(&self, point: &Point3<f64>) -> (f64, f64) {
        let plot_left = 80.0;
        let plot_top = 54.0;
        let plot_width = 740.0;
        let plot_height = 396.0;
        let horizontal = (point.x - self.min_x) / (self.max_x - self.min_x);
        let vertical_value = if self.axis_y == 2 { point.z } else { point.y };
        let vertical = (vertical_value - self.min_y) / (self.max_y - self.min_y);
        (
            plot_left + horizontal * plot_width,
            plot_top + (1.0 - vertical) * plot_height,
        )
    }
}

fn padded_range(min: f64, max: f64) -> (f64, f64) {
    let span = max - min;
    if span.abs() < 1.0e-12 {
        (min - 1.0, max + 1.0)
    } else {
        let padding = span * 0.08;
        (min - padding, max + padding)
    }
}

fn parse_tum_frame_id(
    value: &str,
    line_number: usize,
    line: &str,
) -> Result<FrameId, TumTrajectoryParseError> {
    value.parse::<FrameId>().map_err(|error| {
        TumTrajectoryParseError::new(line_number, line, format!("invalid frame_id: {error}"))
    })
}

fn parse_tum_f64(
    value: &str,
    field_name: &str,
    line_number: usize,
    line: &str,
) -> Result<f64, TumTrajectoryParseError> {
    value.parse::<f64>().map_err(|error| {
        TumTrajectoryParseError::new(line_number, line, format!("invalid {field_name}: {error}"))
    })
}

fn parse_kitti_f64(
    value: &str,
    field_index: usize,
    line_number: usize,
    line: &str,
) -> Result<f64, KittiTrajectoryParseError> {
    value.parse::<f64>().map_err(|error| {
        KittiTrajectoryParseError::new(
            line_number,
            line,
            format!("invalid field {field_index}: {error}"),
        )
    })
}

fn cumulative_reference_distances(matched: &[(&TrajectorySample, &TrajectorySample)]) -> Vec<f64> {
    let mut distances = Vec::with_capacity(matched.len());
    let mut cumulative = 0.0;
    for (index, (_, reference_sample)) in matched.iter().enumerate() {
        if index > 0 {
            let previous = matched[index - 1].1.camera_center_world();
            let current = reference_sample.camera_center_world();
            cumulative += (current - previous).norm();
        }
        distances.push(cumulative);
    }
    distances
}

fn first_index_for_distance(distances: &[f64], target_distance: f64) -> Option<usize> {
    distances
        .iter()
        .position(|distance| *distance >= target_distance)
}

fn relative_camera_to_world(first: &TrajectorySample, last: &TrajectorySample) -> SE3 {
    let first_camera_to_world = first.pose.world_to_camera.inverse();
    let last_camera_to_world = last.pose.world_to_camera.inverse();
    first_camera_to_world
        .inverse()
        .compose(&last_camera_to_world)
}

fn export_f64(value: f64) -> String {
    if value.abs() < 1.0e-15 {
        "0".to_string()
    } else {
        value.to_string()
    }
}

fn optional_f64_json(value: Option<f64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn optional_usize_json(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn relative_pose_error_statistics_json(stats: Option<RelativePoseErrorStatistics>) -> String {
    match stats {
        None => "null".to_string(),
        Some(s) => format!(
            "{{ \"rmse\": {}, \"mean\": {}, \"median\": {}, \"std\": {}, \"min\": {}, \"max\": {} }}",
            s.rmse, s.mean, s.median, s.std, s.min, s.max
        ),
    }
}

fn optional_frame_id_json(value: Option<FrameId>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn optional_vec3_json(value: Option<[f64; 3]>) -> String {
    value
        .map(|value| format!("[{}, {}, {}]", value[0], value[1], value[2]))
        .unwrap_or_else(|| "null".to_string())
}

fn trajectory_evaluation_failures_json(failures: &[TrajectoryEvaluationFailure]) -> String {
    let mut output = String::from("[");
    for (index, failure) in failures.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.push_str(&failure.to_json_inline());
    }
    output.push(']');
    output
}

fn tracking_evaluation_failures_json(failures: &[TrackingEvaluationFailure]) -> String {
    let mut output = String::from("[");
    for (index, failure) in failures.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.push_str(&failure.to_json_inline());
    }
    output.push(']');
    output
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackingStats {
    pub first_frame_id: Option<FrameId>,
    pub last_frame_id: Option<FrameId>,
    pub frame_count: usize,
    pub successful_frame_count: usize,
    pub failed_frame_count: usize,
    pub lost_count: usize,
    pub relocalization_count: usize,
    pub pose_prior_used_count: usize,
    pub external_localization_prior_used_count: usize,
    pub tracking_quality_gate_failure_count: usize,
    pub total_inlier_count: usize,
    pub total_correspondence_count: usize,
    pub covisibility_local_map_used_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackingEvaluationConfig {
    pub min_success_rate: Option<f64>,
    pub max_failure_rate: Option<f64>,
    pub max_lost_count: Option<usize>,
    pub max_tracking_quality_gate_failure_count: Option<usize>,
    pub min_external_localization_prior_usage_rate: Option<f64>,
    pub min_overall_inlier_ratio: Option<f64>,
    pub min_mean_inliers_per_successful_frame: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingEvaluationResult {
    pub passed: bool,
    pub stats: TrackingStats,
    pub config: TrackingEvaluationConfig,
    pub failures: Vec<TrackingEvaluationFailure>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackingEvaluationFailure {
    SuccessRateTooLow { actual: f64, minimum: f64 },
    FailureRateTooHigh { actual: f64, maximum: f64 },
    LostCountTooHigh { actual: usize, maximum: usize },
    QualityGateFailureCountTooHigh { actual: usize, maximum: usize },
    ExternalLocalizationPriorUsageRateTooLow { actual: f64, minimum: f64 },
    OverallInlierRatioTooLow { actual: f64, minimum: f64 },
    MeanInliersPerSuccessfulFrameTooLow { actual: f64, minimum: f64 },
}

impl TrackingStats {
    pub fn from_results(results: &[TrackingResult]) -> Self {
        let mut stats = Self::default();
        for result in results {
            if stats.first_frame_id.is_none() {
                stats.first_frame_id = Some(result.frame_id);
            }
            stats.last_frame_id = Some(result.frame_id);
            stats.frame_count += 1;
            if result.localization.success {
                stats.successful_frame_count += 1;
            } else {
                stats.failed_frame_count += 1;
            }
            if result.event == TrackingEvent::Lost {
                stats.lost_count += 1;
            }
            if result.event == TrackingEvent::Relocalized {
                stats.relocalization_count += 1;
            }
            if result.used_pose_prior {
                stats.pose_prior_used_count += 1;
            }
            if result.used_external_localization_prior {
                stats.external_localization_prior_used_count += 1;
            }
            if result.tracking_failure_reason.is_some() {
                stats.tracking_quality_gate_failure_count += 1;
            }
            stats.total_inlier_count += result.localization.inlier_count;
            stats.total_correspondence_count += result.localization.correspondence_count;
            if result.covisibility_local_map_size.is_some() {
                stats.covisibility_local_map_used_count += 1;
            }
        }
        stats
    }

    pub fn success_rate(&self) -> f64 {
        ratio(self.successful_frame_count, self.frame_count)
    }

    pub fn failure_rate(&self) -> f64 {
        ratio(self.failed_frame_count, self.frame_count)
    }

    pub fn pose_prior_usage_rate(&self) -> f64 {
        ratio(self.pose_prior_used_count, self.frame_count)
    }

    pub fn external_localization_prior_usage_rate(&self) -> f64 {
        ratio(
            self.external_localization_prior_used_count,
            self.frame_count,
        )
    }

    pub fn overall_inlier_ratio(&self) -> f64 {
        ratio(self.total_inlier_count, self.total_correspondence_count)
    }

    pub fn mean_inliers_per_successful_frame(&self) -> f64 {
        ratio(self.total_inlier_count, self.successful_frame_count)
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"first_frame_id\": {},\n",
                "  \"last_frame_id\": {},\n",
                "  \"frame_count\": {},\n",
                "  \"successful_frame_count\": {},\n",
                "  \"failed_frame_count\": {},\n",
                "  \"lost_count\": {},\n",
                "  \"relocalization_count\": {},\n",
                "  \"pose_prior_used_count\": {},\n",
                "  \"external_localization_prior_used_count\": {},\n",
                "  \"tracking_quality_gate_failure_count\": {},\n",
                "  \"total_inlier_count\": {},\n",
                "  \"total_correspondence_count\": {},\n",
                "  \"success_rate\": {},\n",
                "  \"failure_rate\": {},\n",
                "  \"pose_prior_usage_rate\": {},\n",
                "  \"external_localization_prior_usage_rate\": {},\n",
                "  \"overall_inlier_ratio\": {},\n",
                "  \"mean_inliers_per_successful_frame\": {}\n",
                "}}\n"
            ),
            optional_frame_id_json(self.first_frame_id),
            optional_frame_id_json(self.last_frame_id),
            self.frame_count,
            self.successful_frame_count,
            self.failed_frame_count,
            self.lost_count,
            self.relocalization_count,
            self.pose_prior_used_count,
            self.external_localization_prior_used_count,
            self.tracking_quality_gate_failure_count,
            self.total_inlier_count,
            self.total_correspondence_count,
            self.success_rate(),
            self.failure_rate(),
            self.pose_prior_usage_rate(),
            self.external_localization_prior_usage_rate(),
            self.overall_inlier_ratio(),
            self.mean_inliers_per_successful_frame(),
        )
    }

    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"first_frame_id\": {}, ",
                "\"last_frame_id\": {}, ",
                "\"frame_count\": {}, ",
                "\"successful_frame_count\": {}, ",
                "\"failed_frame_count\": {}, ",
                "\"lost_count\": {}, ",
                "\"relocalization_count\": {}, ",
                "\"pose_prior_used_count\": {}, ",
                "\"external_localization_prior_used_count\": {}, ",
                "\"tracking_quality_gate_failure_count\": {}, ",
                "\"total_inlier_count\": {}, ",
                "\"total_correspondence_count\": {}, ",
                "\"success_rate\": {}, ",
                "\"failure_rate\": {}, ",
                "\"pose_prior_usage_rate\": {}, ",
                "\"external_localization_prior_usage_rate\": {}, ",
                "\"overall_inlier_ratio\": {}, ",
                "\"mean_inliers_per_successful_frame\": {}",
                "}}"
            ),
            optional_frame_id_json(self.first_frame_id),
            optional_frame_id_json(self.last_frame_id),
            self.frame_count,
            self.successful_frame_count,
            self.failed_frame_count,
            self.lost_count,
            self.relocalization_count,
            self.pose_prior_used_count,
            self.external_localization_prior_used_count,
            self.tracking_quality_gate_failure_count,
            self.total_inlier_count,
            self.total_correspondence_count,
            self.success_rate(),
            self.failure_rate(),
            self.pose_prior_usage_rate(),
            self.external_localization_prior_usage_rate(),
            self.overall_inlier_ratio(),
            self.mean_inliers_per_successful_frame(),
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn evaluate(&self, config: TrackingEvaluationConfig) -> TrackingEvaluationResult {
        TrackingEvaluationResult::from_stats(self.clone(), config)
    }
}

impl TrackingEvaluationResult {
    pub fn from_stats(stats: TrackingStats, config: TrackingEvaluationConfig) -> Self {
        let mut failures = Vec::new();

        if let Some(minimum) = config.min_success_rate {
            let actual = stats.success_rate();
            if actual < minimum {
                failures.push(TrackingEvaluationFailure::SuccessRateTooLow { actual, minimum });
            }
        }

        if let Some(maximum) = config.max_failure_rate {
            let actual = stats.failure_rate();
            if actual > maximum {
                failures.push(TrackingEvaluationFailure::FailureRateTooHigh { actual, maximum });
            }
        }

        if let Some(maximum) = config.max_lost_count {
            if stats.lost_count > maximum {
                failures.push(TrackingEvaluationFailure::LostCountTooHigh {
                    actual: stats.lost_count,
                    maximum,
                });
            }
        }

        if let Some(maximum) = config.max_tracking_quality_gate_failure_count {
            if stats.tracking_quality_gate_failure_count > maximum {
                failures.push(TrackingEvaluationFailure::QualityGateFailureCountTooHigh {
                    actual: stats.tracking_quality_gate_failure_count,
                    maximum,
                });
            }
        }

        if let Some(minimum) = config.min_external_localization_prior_usage_rate {
            let actual = stats.external_localization_prior_usage_rate();
            if actual < minimum {
                failures.push(
                    TrackingEvaluationFailure::ExternalLocalizationPriorUsageRateTooLow {
                        actual,
                        minimum,
                    },
                );
            }
        }

        if let Some(minimum) = config.min_overall_inlier_ratio {
            let actual = stats.overall_inlier_ratio();
            if actual < minimum {
                failures
                    .push(TrackingEvaluationFailure::OverallInlierRatioTooLow { actual, minimum });
            }
        }

        if let Some(minimum) = config.min_mean_inliers_per_successful_frame {
            let actual = stats.mean_inliers_per_successful_frame();
            if actual < minimum {
                failures.push(
                    TrackingEvaluationFailure::MeanInliersPerSuccessfulFrameTooLow {
                        actual,
                        minimum,
                    },
                );
            }
        }

        let passed = failures.is_empty();
        Self {
            passed,
            stats,
            config,
            failures,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"config\": {},\n",
                "  \"stats\": {},\n",
                "  \"failures\": {}\n",
                "}}\n"
            ),
            self.passed,
            self.config.to_json_inline(),
            self.stats.to_json_inline(),
            tracking_evaluation_failures_json(&self.failures)
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

impl TrackingEvaluationConfig {
    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"min_success_rate\": {}, ",
                "\"max_failure_rate\": {}, ",
                "\"max_lost_count\": {}, ",
                "\"max_tracking_quality_gate_failure_count\": {}, ",
                "\"min_external_localization_prior_usage_rate\": {}, ",
                "\"min_overall_inlier_ratio\": {}, ",
                "\"min_mean_inliers_per_successful_frame\": {}",
                "}}"
            ),
            optional_f64_json(self.min_success_rate),
            optional_f64_json(self.max_failure_rate),
            optional_usize_json(self.max_lost_count),
            optional_usize_json(self.max_tracking_quality_gate_failure_count),
            optional_f64_json(self.min_external_localization_prior_usage_rate),
            optional_f64_json(self.min_overall_inlier_ratio),
            optional_f64_json(self.min_mean_inliers_per_successful_frame)
        )
    }
}

impl TrackingEvaluationFailure {
    pub fn reason(&self) -> &'static str {
        match self {
            Self::SuccessRateTooLow { .. } => "success_rate_too_low",
            Self::FailureRateTooHigh { .. } => "failure_rate_too_high",
            Self::LostCountTooHigh { .. } => "lost_count_too_high",
            Self::QualityGateFailureCountTooHigh { .. } => "quality_gate_failure_count_too_high",
            Self::ExternalLocalizationPriorUsageRateTooLow { .. } => {
                "external_localization_prior_usage_rate_too_low"
            }
            Self::OverallInlierRatioTooLow { .. } => "overall_inlier_ratio_too_low",
            Self::MeanInliersPerSuccessfulFrameTooLow { .. } => {
                "mean_inliers_per_successful_frame_too_low"
            }
        }
    }

    fn to_json_inline(&self) -> String {
        match self {
            Self::SuccessRateTooLow { actual, minimum }
            | Self::ExternalLocalizationPriorUsageRateTooLow { actual, minimum }
            | Self::OverallInlierRatioTooLow { actual, minimum }
            | Self::MeanInliersPerSuccessfulFrameTooLow { actual, minimum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"minimum\": {}}}",
                self.reason(),
                actual,
                minimum
            ),
            Self::FailureRateTooHigh { actual, maximum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"maximum\": {}}}",
                self.reason(),
                actual,
                maximum
            ),
            Self::LostCountTooHigh { actual, maximum }
            | Self::QualityGateFailureCountTooHigh { actual, maximum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"maximum\": {}}}",
                self.reason(),
                actual,
                maximum
            ),
        }
    }
}

pub fn tracking_results_to_html_report(results: &[TrackingResult]) -> String {
    let stats = TrackingStats::from_results(results);
    let mut output = String::new();
    output.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    output.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    output.push_str("<title>visloc-rs tracking report</title>\n");
    output.push_str("<style>");
    output.push_str(
        "body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f9;color:#182026}\
         main{max-width:1120px;margin:0 auto;padding:28px}\
         h1{font-size:24px;margin:0 0 8px}\
         h2{font-size:18px;margin:0 0 10px}\
         .sub{margin:0 0 22px;color:#52616b}\
         .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}\
         .metric{background:white;border:1px solid #dde3ea;border-radius:8px;padding:12px}\
         .label{display:block;font-size:12px;color:#65727e}\
         .value{display:block;font-size:22px;font-weight:700;margin-top:4px}\
         .panel{background:white;border:1px solid #dde3ea;border-radius:8px;padding:16px;margin-top:14px}\
         table{width:100%;border-collapse:collapse;font-size:13px}\
         th,td{text-align:right;border-bottom:1px solid #e7ecf0;padding:6px 8px;vertical-align:top}\
         th:first-child,td:first-child,th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4),th:last-child,td:last-child{text-align:left}\
         .ok{color:#198754;font-weight:700}.fail{color:#c23b3b;font-weight:700}\
         svg{width:100%;height:auto;display:block}",
    );
    output.push_str("</style>\n</head>\n<body>\n<main>\n");
    output.push_str("<h1>visloc-rs tracking report</h1>\n");
    output.push_str("<p class=\"sub\">Frame-by-frame sequence-localization state, failures, priors, and inlier diagnostics.</p>\n");
    output.push_str("<section class=\"grid\">\n");
    push_metric_card(&mut output, "Frames", &stats.frame_count.to_string());
    push_metric_card(
        &mut output,
        "Success rate",
        &format!("{:.1}%", stats.success_rate() * 100.0),
    );
    push_metric_card(
        &mut output,
        "Failed frames",
        &stats.failed_frame_count.to_string(),
    );
    push_metric_card(&mut output, "Lost events", &stats.lost_count.to_string());
    push_metric_card(
        &mut output,
        "Relocalized",
        &stats.relocalization_count.to_string(),
    );
    push_metric_card(
        &mut output,
        "Motion prior",
        &format!("{:.1}%", stats.pose_prior_usage_rate() * 100.0),
    );
    push_metric_card(
        &mut output,
        "External prior",
        &format!(
            "{} ({:.1}%)",
            stats.external_localization_prior_used_count,
            stats.external_localization_prior_usage_rate() * 100.0
        ),
    );
    push_metric_card(
        &mut output,
        "Mean inliers",
        &format!("{:.1}", stats.mean_inliers_per_successful_frame()),
    );
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n");
    output.push_str(&tracking_timeline_svg(results));
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n<h2>Frames</h2>\n");
    output.push_str("<table><thead><tr><th>frame</th><th>success</th><th>state</th><th>event</th><th>inliers</th><th>ratio</th><th>reprojection</th><th>priors</th><th>reason</th></tr></thead><tbody>\n");
    for result in results.iter().take(160) {
        let success_class = if result.localization.success {
            "ok"
        } else {
            "fail"
        };
        let success_text = if result.localization.success {
            "ok"
        } else {
            "failed"
        };
        let reason = tracking_result_reason(result);
        let _ = writeln!(
            output,
            "<tr><td>{}</td><td class=\"{}\">{}</td><td>{:?}</td><td>{:?}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            result.frame_id,
            success_class,
            success_text,
            result.state,
            result.event,
            result.localization.inlier_count,
            result.localization.inlier_ratio,
            format_optional_metric(result.localization.reprojection_error, "px"),
            tracking_prior_text(result),
            html_escape(&reason),
        );
    }
    if results.len() > 160 {
        let _ = writeln!(
            output,
            "<tr><td colspan=\"9\">{} more rows omitted</td></tr>",
            results.len() - 160
        );
    }
    output.push_str("</tbody></table>\n</section>\n</main>\n</body>\n</html>\n");
    output
}

pub fn write_tracking_results_html_report(
    results: &[TrackingResult],
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    std::fs::write(path, tracking_results_to_html_report(results))
}

pub fn tracking_results_to_csv(results: &[TrackingResult]) -> String {
    let mut output = String::from(
        "frame_id,state,event,success,successive_failures,used_pose_prior,used_external_localization_prior,external_localization_prior_radius,tracking_failure_reason,localization_failure_reason,candidate_landmark_count,match_count,correspondence_count,inlier_count,outlier_count,inlier_ratio,reprojection_error,median_reprojection_error,max_reprojection_error,map_cameras,map_keyframes,map_landmarks,map_descriptors\n",
    );
    for result in results {
        let _ = writeln!(
            output,
            "{},{:?},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            result.frame_id,
            result.state,
            result.event,
            result.localization.success,
            result.successive_failures,
            result.used_pose_prior,
            result.used_external_localization_prior,
            optional_f64_csv(result.external_localization_prior_radius),
            csv_escape(&format_optional_debug(&result.tracking_failure_reason)),
            csv_escape(&format_optional_debug(&result.localization.failure_reason)),
            result.localization.candidate_landmark_count,
            result.localization.match_count,
            result.localization.correspondence_count,
            result.localization.inlier_count,
            result.localization.outlier_count,
            result.localization.inlier_ratio,
            optional_f64_csv(result.localization.reprojection_error),
            optional_f64_csv(result.localization.median_reprojection_error),
            optional_f64_csv(result.localization.max_reprojection_error),
            result.map_stats.camera_count,
            result.map_stats.keyframe_count,
            result.map_stats.landmark_count,
            result.map_stats.descriptor_count,
        );
    }
    output
}

pub fn write_tracking_results_csv(
    results: &[TrackingResult],
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    std::fs::write(path, tracking_results_to_csv(results))
}

fn tracking_timeline_svg(results: &[TrackingResult]) -> String {
    let mut output = String::new();
    output
        .push_str("<svg viewBox=\"0 0 900 190\" role=\"img\" aria-label=\"tracking timeline\">\n");
    output.push_str("<rect x=\"0\" y=\"0\" width=\"900\" height=\"190\" fill=\"#fbfcfd\"/>\n");
    output.push_str(
        "<line x1=\"70\" y1=\"92\" x2=\"830\" y2=\"92\" stroke=\"#d8e0e7\" stroke-width=\"2\"/>\n",
    );
    if results.is_empty() {
        output.push_str(
            "<text x=\"70\" y=\"102\" fill=\"#65727e\" font-size=\"14\">no frames</text>\n",
        );
        output.push_str("</svg>\n");
        return output;
    }

    let denom = results.len().saturating_sub(1).max(1) as f64;
    for (index, result) in results.iter().enumerate() {
        let x = 70.0 + (index as f64 / denom) * 760.0;
        let radius = if result.localization.success {
            8.0
        } else {
            9.0
        };
        let color = tracking_event_color(result);
        let y = if result.localization.success {
            78.0
        } else {
            108.0
        };
        let _ = writeln!(
            output,
            "<line x1=\"{x:.2}\" y1=\"92\" x2=\"{x:.2}\" y2=\"{y:.2}\" stroke=\"{color}\" stroke-width=\"2\"/>"
        );
        let _ = writeln!(
            output,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"{radius}\" fill=\"{color}\"/>"
        );
        if index == 0 || index + 1 == results.len() || result.event != TrackingEvent::Tracked {
            let _ = writeln!(
                output,
                "<text x=\"{:.2}\" y=\"142\" fill=\"#52616b\" font-size=\"12\" text-anchor=\"middle\">{}</text>",
                x, result.frame_id
            );
            let _ = writeln!(
                output,
                "<text x=\"{:.2}\" y=\"158\" fill=\"#52616b\" font-size=\"11\" text-anchor=\"middle\">{:?}</text>",
                x, result.event
            );
        }
    }
    output.push_str(
        "<text x=\"70\" y=\"32\" fill=\"#52616b\" font-size=\"13\">success/relocalization</text>\n",
    );
    output.push_str(
        "<text x=\"70\" y=\"176\" fill=\"#52616b\" font-size=\"13\">failure/lost</text>\n",
    );
    output.push_str("</svg>\n");
    output
}

fn tracking_event_color(result: &TrackingResult) -> &'static str {
    match result.event {
        TrackingEvent::Initialized => "#2676c9",
        TrackingEvent::Tracked => "#198754",
        TrackingEvent::TrackingFailed => "#d9822b",
        TrackingEvent::Lost => "#c23b3b",
        TrackingEvent::Relocalized => "#7b4ab8",
    }
}

fn tracking_result_reason(result: &TrackingResult) -> String {
    if let Some(reason) = &result.tracking_failure_reason {
        format!("{reason:?}")
    } else if let Some(reason) = &result.localization.failure_reason {
        format!("{reason:?}")
    } else {
        String::new()
    }
}

fn tracking_prior_text(result: &TrackingResult) -> String {
    let mut priors = Vec::new();
    if result.used_pose_prior {
        priors.push("motion".to_string());
    }
    if result.used_external_localization_prior {
        let label = if let Some(radius) = result.external_localization_prior_radius {
            format!("external({radius:.3}m)")
        } else {
            "external".to_string()
        };
        priors.push(label);
    }

    if priors.is_empty() {
        "none".to_string()
    } else {
        priors.join(" + ")
    }
}

fn format_optional_debug<T: fmt::Debug>(value: &Option<T>) -> String {
    value
        .as_ref()
        .map(|value| format!("{value:?}"))
        .unwrap_or_default()
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub trait MotionModel {
    fn predict_pose(
        &self,
        frame: &Frame,
        last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose>;

    fn observe(&mut self, _result: &TrackingResult) {}

    fn reset(&mut self) {}
}

pub trait VisualOdometryFrontend {
    type Error;

    fn estimate_relative_pose(
        &self,
        previous_frame: &Frame,
        current_frame: &Frame,
    ) -> Result<Option<VisualOdometryEstimate>, Self::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisualOdometryEstimate {
    pub previous_frame_id: FrameId,
    pub current_frame_id: FrameId,
    pub previous_to_current: SE3,
    pub match_count: usize,
    pub inlier_count: usize,
    pub mean_reprojection_error: Option<f64>,
}

impl VisualOdometryEstimate {
    pub fn new(
        previous_frame_id: FrameId,
        current_frame_id: FrameId,
        previous_to_current: SE3,
    ) -> Self {
        Self {
            previous_frame_id,
            current_frame_id,
            previous_to_current,
            match_count: 0,
            inlier_count: 0,
            mean_reprojection_error: None,
        }
    }

    pub fn pose_prior_from_previous_pose(&self, previous_pose: &Pose) -> Pose {
        Pose {
            world_to_camera: self
                .previous_to_current
                .compose(&previous_pose.world_to_camera),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisualOdometryPosePrior {
    pub estimate: VisualOdometryEstimate,
    pub pose: Pose,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisualOdometryPriorProvider<F> {
    frontend: F,
}

impl<F> VisualOdometryPriorProvider<F> {
    pub fn new(frontend: F) -> Self {
        Self { frontend }
    }

    pub fn frontend(&self) -> &F {
        &self.frontend
    }

    pub fn frontend_mut(&mut self) -> &mut F {
        &mut self.frontend
    }

    pub fn into_inner(self) -> F {
        self.frontend
    }
}

impl<F> VisualOdometryPriorProvider<F>
where
    F: VisualOdometryFrontend,
{
    pub fn predict_pose_prior(
        &self,
        previous_frame: &Frame,
        previous_pose: &Pose,
        current_frame: &Frame,
    ) -> Result<Option<VisualOdometryPosePrior>, F::Error> {
        let Some(estimate) = self
            .frontend
            .estimate_relative_pose(previous_frame, current_frame)?
        else {
            return Ok(None);
        };
        let pose = estimate.pose_prior_from_previous_pose(previous_pose);
        Ok(Some(VisualOdometryPosePrior { estimate, pose }))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoopVisualOdometryFrontend;

impl VisualOdometryFrontend for NoopVisualOdometryFrontend {
    type Error = std::convert::Infallible;

    fn estimate_relative_pose(
        &self,
        _previous_frame: &Frame,
        _current_frame: &Frame,
    ) -> Result<Option<VisualOdometryEstimate>, Self::Error> {
        Ok(None)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConstantPoseMotionModel;

impl MotionModel for ConstantPoseMotionModel {
    fn predict_pose(
        &self,
        _frame: &Frame,
        _last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        last_successful_pose.cloned()
    }
}

/// Inertial pose predictor: a "loosely coupled" replacement for
/// [`ConstantPoseMotionModel`] / [`ConstantVelocityMotionModel`] that
/// integrates body-frame IMU samples forward from the previous
/// successful pose to produce the next pose prior. Mirrors the inputs of
/// [`crate::Tracker`]'s motion-model slot so it can be dropped in via
/// `Tracker::new_with_motion_model`.
///
/// Lifecycle:
///
/// 1. The caller pushes inter-frame IMU samples via
///    [`Self::push_imu_measurement`] (mutable, accumulates in
///    `pending_samples`).
/// 2. The tracker invokes [`MotionModel::predict_pose`] (read-only): the
///    model forward-Eulers `(R_bw, v_w, p_bw)` from `last_successful_pose`
///    using the pending samples + the configured gravity / biases + the
///    current `velocity_world` and returns the predicted
///    `world_to_camera` pose.
/// 3. The tracker invokes [`MotionModel::observe`]: on success the
///    pending samples are drained (the next inter-frame window starts
///    fresh). The model itself does NOT re-estimate `velocity_world`
///    from the tracker's output — that update is the caller's
///    responsibility (e.g., from a downstream local VI-BA's refined
///    velocity, or from a finite-difference of camera centres over the
///    integrated window). Without an explicit update, the predictor
///    re-uses the last set velocity, which is the same constant-velocity
///    behaviour [`ConstantVelocityMotionModel`] gives.
///
/// Coordinate conventions match the IMU pre-integrator: gyro and accel
/// are body-frame; gravity is in the world frame (KITTI y-down default:
/// `(0, 9.81, 0)`); accel includes gravity such that a stationary IMU
/// reads `R_wb^T · (−gravity_world)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuPredictiveMotionModel {
    /// Static configuration (gravity, biases). Update biases via
    /// [`Self::set_biases`] when a downstream VI-BA refines them.
    pub config: ImuPredictiveMotionModelConfig,
    /// Buffered `(gyro, accel, dt)` tuples to integrate at the next
    /// `predict_pose` call.
    pending_samples: Vec<ImuPredictivePendingSample>,
    /// World-frame velocity at the last successful pose. Used as the
    /// initial velocity of the strapdown integration. The model does
    /// NOT auto-update this on `observe`; downstream code (e.g., the
    /// local VI-BA in `OnlineSlamPipeline`) should call
    /// [`Self::set_velocity_world`] with the refined velocity after the
    /// next BA pass.
    velocity_world: nalgebra::Vector3<f64>,
    /// `true` when at least one pending sample was integrated by the
    /// most recent `predict_pose` call. Used by `observe` to decide
    /// whether to drain the buffer (a `predict_pose` call without any
    /// samples must NOT drain anything pushed *after* it).
    last_predict_consumed_samples: bool,
    /// Camera pose passed to the previous successful `observe`. Used by
    /// the carry-forward path to re-anchor the integration before
    /// committing the advanced `velocity_world`. Stays `None` until the
    /// first successful frame; ignored entirely when
    /// `config.carry_forward_velocity_world` is `false`.
    last_successful_pose: Option<Pose>,
}

/// Static parameters for [`ImuPredictiveMotionModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ImuPredictiveMotionModelConfig {
    /// World-frame gravity vector. KITTI y-down default `(0, 9.81, 0)`;
    /// EuRoC z-up convention uses `(0, 0, -9.81)`.
    pub gravity_world: nalgebra::Vector3<f64>,
    /// Gyro bias subtracted from every sample before integration.
    pub bias_gyro: nalgebra::Vector3<f64>,
    /// Accel bias subtracted from every sample before integration.
    pub bias_acc: nalgebra::Vector3<f64>,
    /// `T_BS` body-from-sensor (= body-from-camera) rigid transform:
    /// `p_body = body_to_sensor · p_sensor`. Used by `predict_pose` to
    /// (a) convert the input camera pose to a body pose before
    /// strapdown integration, and (b) convert the integrated body pose
    /// back to a camera pose on the way out. Defaults to identity
    /// (body == camera, the assumption the original wire-up made).
    /// EuRoC's `cam0/sensor.yaml::T_BS` is exactly this transform: pass
    /// the parsed [`SE3`] verbatim. The camera-relative offset is
    /// ~0.1 m on EuRoC, so the identity default is a usable
    /// approximation but a metric-tight prediction wants the real
    /// extrinsic.
    pub body_to_sensor: SE3,
    /// When `true`, `observe` re-integrates the pending IMU
    /// samples from the previously-tracked pose to advance
    /// `velocity_world` for the next frame. Without this, the seed
    /// velocity stays frozen at the last `set_velocity_world`
    /// (i.e., last VI-BA mirror) until the next mirror fires — so on
    /// frames between mirrors, the strapdown integration restarts from
    /// the KF-time velocity rather than the velocity at the just-tracked
    /// frame. Defaults to `false` for backwards compatibility.
    pub carry_forward_velocity_world: bool,
}

impl Default for ImuPredictiveMotionModelConfig {
    fn default() -> Self {
        Self {
            gravity_world: nalgebra::Vector3::new(0.0, 9.81, 0.0),
            bias_gyro: nalgebra::Vector3::zeros(),
            bias_acc: nalgebra::Vector3::zeros(),
            body_to_sensor: SE3::identity(),
            carry_forward_velocity_world: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ImuPredictivePendingSample {
    gyro: nalgebra::Vector3<f64>,
    accel: nalgebra::Vector3<f64>,
    dt: f64,
}

impl ImuPredictiveMotionModel {
    pub fn new(config: ImuPredictiveMotionModelConfig) -> Self {
        Self {
            config,
            pending_samples: Vec::new(),
            velocity_world: nalgebra::Vector3::zeros(),
            last_predict_consumed_samples: false,
            last_successful_pose: None,
        }
    }

    /// Append one body-frame `(gyro, accel)` sample with the elapsed
    /// time `dt` (seconds) since the previous sample. Non-positive
    /// `dt` is silently dropped to keep raw IMU replays robust.
    pub fn push_imu_measurement(
        &mut self,
        gyro: nalgebra::Vector3<f64>,
        accel: nalgebra::Vector3<f64>,
        dt: f64,
    ) {
        if dt <= 0.0 || dt.is_nan() {
            return;
        }
        self.pending_samples
            .push(ImuPredictivePendingSample { gyro, accel, dt });
    }

    /// Overwrite the world-frame velocity carried into the next
    /// `predict_pose`. Call this after a downstream solver (e.g., the
    /// `OnlineSlamPipeline` local VI-BA) refines the velocity at the
    /// most recent keyframe.
    pub fn set_velocity_world(&mut self, velocity_world: nalgebra::Vector3<f64>) {
        self.velocity_world = velocity_world;
    }

    /// Non-mutating finite-difference body-frame world-velocity from
    /// two successive camera poses and the elapsed time between them.
    /// Returns `None` when `dt_seconds` is not strictly positive/finite.
    /// Callers that want to write the result into `velocity_world`
    /// should use [`Self::update_velocity_from_camera_pose_difference`];
    /// callers that need to combine multiple finite-differences (e.g.
    /// the Phase-25 ThreePoseSmoother refresh policy averages two)
    /// use this directly to avoid intermediate state writes.
    pub fn body_velocity_from_camera_pose_difference(
        &self,
        prev: &Pose,
        curr: &Pose,
        dt_seconds: f64,
    ) -> Option<nalgebra::Vector3<f64>> {
        if dt_seconds <= 0.0 || !dt_seconds.is_finite() {
            return None;
        }
        let body_prev = self
            .config
            .body_to_sensor
            .compose(&prev.world_to_camera)
            .inverse()
            .translation;
        let body_curr = self
            .config
            .body_to_sensor
            .compose(&curr.world_to_camera)
            .inverse()
            .translation;
        Some((body_curr - body_prev) / dt_seconds)
    }

    /// Update `velocity_world` to a finite-difference estimate from two
    /// successive camera poses and the elapsed time between them.
    /// Internally converts each camera pose to a body pose via
    /// `body_to_sensor` so the velocity is the body's world-frame
    /// velocity (the integrator's expected initial-velocity semantics),
    /// not the camera's. Non-positive `dt_seconds` is silently dropped.
    /// This is the recommended hook for callers that do not run a
    /// downstream VI-BA (which would normally refine velocity) — without
    /// it, `velocity_world` stays at the constructor default (zero) and
    /// the position integration only picks up the quadratic accel term,
    /// systematically under-predicting motion on a moving body.
    pub fn update_velocity_from_camera_pose_difference(
        &mut self,
        prev: &Pose,
        curr: &Pose,
        dt_seconds: f64,
    ) {
        if let Some(v) = self.body_velocity_from_camera_pose_difference(prev, curr, dt_seconds) {
            self.velocity_world = v;
        }
    }

    /// Overwrite the gyro / accel bias linearisation points.
    pub fn set_biases(
        &mut self,
        bias_gyro: nalgebra::Vector3<f64>,
        bias_acc: nalgebra::Vector3<f64>,
    ) {
        self.config.bias_gyro = bias_gyro;
        self.config.bias_acc = bias_acc;
    }

    /// Read-only access to the current pending-sample count, mostly for
    /// tests / diagnostics.
    pub fn pending_samples_len(&self) -> usize {
        self.pending_samples.len()
    }

    /// Sum of `dt` over all currently-pending IMU samples. Useful for
    /// callers (e.g. [`AdaptiveImuPoseMotionModel`]) that need to know
    /// the elapsed wall-clock time between the previous successful
    /// `observe()` (which drained the buffer) and the moment this is
    /// queried — typically immediately before the next `observe()`
    /// drain. Returns `0.0` when no samples are pending.
    pub fn pending_samples_total_dt(&self) -> f64 {
        self.pending_samples.iter().map(|s| s.dt).sum()
    }

    /// Read-only access to the current world-frame velocity.
    pub fn velocity_world(&self) -> nalgebra::Vector3<f64> {
        self.velocity_world
    }
}

impl MotionModel for ImuPredictiveMotionModel {
    fn predict_pose(
        &self,
        _frame: &Frame,
        _last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        let prev = last_successful_pose?;
        if self.pending_samples.is_empty() {
            return Some(prev.clone());
        }
        // Convert the input camera pose `T_cw = prev.world_to_camera` to
        // a body pose `T_bw = T_bs · T_cw` (body-from-world). Then take
        // its inverse to get `T_wb`, whose rotation is the body-to-world
        // orientation `R_wb` (transforms body-frame vectors to world)
        // and whose translation is the body centre in world `p_wb`.
        // With `body_to_sensor = identity` this reduces to the original
        // `r_bw = R_wc, p_bw = camera_center_world` extraction.
        let t_bw_initial = self.config.body_to_sensor.compose(&prev.world_to_camera);
        let t_wb_initial = t_bw_initial.inverse();
        let mut r_wb = t_wb_initial.rotation;
        let mut p_wb = t_wb_initial.translation;
        let mut v_w = self.velocity_world;
        for sample in &self.pending_samples {
            let gyro_unbiased = sample.gyro - self.config.bias_gyro;
            let accel_unbiased = sample.accel - self.config.bias_acc;
            let accel_world = r_wb.transform_vector(&accel_unbiased) + self.config.gravity_world;
            // Forward-Euler strapdown step. Position uses the velocity at
            // step start (mid-point integration would be more accurate but
            // matches the post-IMU-factor pre-integration semantics: tiny
            // `dt`s amortise the second-order error).
            p_wb += v_w * sample.dt + 0.5 * accel_world * sample.dt * sample.dt;
            v_w += accel_world * sample.dt;
            r_wb *= UnitQuaternion::from_scaled_axis(gyro_unbiased * sample.dt);
        }
        // Build the new body-in-world pose, invert to body-from-world,
        // and compose with `T_cb = body_to_sensor⁻¹` to recover the new
        // camera-from-world pose: `T_cw_new = T_cb · T_bw_new`.
        let t_wb_new = SE3::new(r_wb, p_wb);
        let t_bw_new = t_wb_new.inverse();
        let t_cw_new = self.config.body_to_sensor.inverse().compose(&t_bw_new);
        Some(Pose {
            world_to_camera: t_cw_new,
        })
    }

    fn observe(&mut self, result: &TrackingResult) {
        if !result.localization.success {
            return;
        }
        // Carry-forward path: when enabled and a previous successful
        // pose exists, re-run the same strapdown integration that
        // `predict_pose` performed (using the *previous* pose as anchor)
        // and commit the post-integration `v_w` as the new initial
        // velocity for the next `predict_pose` call. Without this, the
        // seed velocity stays frozen at the last `set_velocity_world`
        // (i.e. last VI-BA mirror) for every frame in the KF window,
        // so per-frame predictions silently restart from the KF-time
        // velocity instead of the velocity at the just-tracked frame.
        if self.config.carry_forward_velocity_world {
            if let Some(prev) = self.last_successful_pose.as_ref() {
                let t_bw_initial = self.config.body_to_sensor.compose(&prev.world_to_camera);
                let t_wb_initial = t_bw_initial.inverse();
                let mut r_wb = t_wb_initial.rotation;
                let mut v_w = self.velocity_world;
                for sample in &self.pending_samples {
                    let gyro_unbiased = sample.gyro - self.config.bias_gyro;
                    let accel_unbiased = sample.accel - self.config.bias_acc;
                    let accel_world =
                        r_wb.transform_vector(&accel_unbiased) + self.config.gravity_world;
                    v_w += accel_world * sample.dt;
                    r_wb *= UnitQuaternion::from_scaled_axis(gyro_unbiased * sample.dt);
                }
                self.velocity_world = v_w;
            }
            if let Some(pose) = result.localization.pose.as_ref() {
                self.last_successful_pose = Some(pose.clone());
            }
        }
        // Drain the pending window after a successful frame so the next
        // inter-frame integration starts fresh.
        self.pending_samples.clear();
        self.last_predict_consumed_samples = false;
    }

    fn reset(&mut self) {
        self.pending_samples.clear();
        self.velocity_world = nalgebra::Vector3::zeros();
        self.last_predict_consumed_samples = false;
        self.last_successful_pose = None;
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConstantVelocityMotionModel {
    previous_successful_pose: Option<Pose>,
    latest_successful_pose: Option<Pose>,
}

impl ConstantVelocityMotionModel {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MotionModel for ConstantVelocityMotionModel {
    fn predict_pose(
        &self,
        _frame: &Frame,
        _last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        let (Some(previous), Some(latest)) = (
            self.previous_successful_pose.as_ref(),
            self.latest_successful_pose.as_ref(),
        ) else {
            return last_successful_pose.cloned();
        };

        let previous_center = previous.camera_center_world();
        let latest_center = latest.camera_center_world();
        let predicted_center = latest_center + (latest_center - previous_center);
        let rotation = latest.world_to_camera.rotation;
        let translation = -(rotation.transform_vector(&predicted_center.coords));
        Some(Pose::from_world_to_camera(rotation, translation))
    }

    fn observe(&mut self, result: &TrackingResult) {
        if !result.localization.success {
            return;
        }
        let Some(pose) = result.localization.pose.as_ref() else {
            return;
        };
        self.previous_successful_pose = self.latest_successful_pose.take();
        self.latest_successful_pose = Some(pose.clone());
    }

    fn reset(&mut self) {
        self.previous_successful_pose = None;
        self.latest_successful_pose = None;
    }
}

/// Configuration for the adaptive IMU↔ConstantPose motion model
/// ([`AdaptiveImuPoseMotionModel`]).
///
/// The Phase-23 EuRoC sweep
/// (`docs/motion_based_vi_alignment.md` §Phase-23 #2 follow-up)
/// established a clean accuracy↔survival trade-off:
///
/// - `--motion-model imu` produces tight pre-cliff trajectories
///   (V2_01 rigid ATE `0.0021 m`, similarity scale `1.000044`) but
///   the IMU's predictive aggressiveness triggers the
///   `--max-pose-jump-meters` gate at the universal cliff.
/// - `--motion-model pose` survives 25-313 % longer (MH_01 7 → 29
///   keyframes) but degrades rigid ATE 4-100× and collapses the
///   similarity scale.
///
/// The adaptive model defaults to IMU mode for accuracy and
/// transparently falls back to constant-pose after enough consecutive
/// tracking failures, then switches back to IMU once the tracker has
/// recovered. The intent is to keep IMU's tight predictions on the
/// healthy regime while the pose model carries the tracker through
/// the cliff transition.
/// How the [`AdaptiveImuPoseMotionModel`] refreshes the wrapped IMU
/// model's `velocity_world` at every Pose → IMU mode switch.
///
/// The motivation comes from the Phase-23 #4 oscillation: while the
/// adaptive wrapper sits in Pose mode, the IMU keeps absorbing raw
/// samples but never sees a successful visual `observe`, so its seed
/// `velocity_world` rapidly drifts away from the true body motion. The
/// first IMU prediction after the switch-back then mispredicts and
/// fires another failure — the wrapper oscillates Pose↔IMU.
///
/// The Phase-24 hook introduced [`Self::FiniteDifference`] to address
/// that. Phase-25 added [`Self::ZeroReset`] and
/// [`Self::ThreePoseSmoother`] as A/B alternatives after the
/// finite-difference variant produced only V1_01 wins (MH_01 / V2_01
/// were neutral-to-worse).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuVelocityRefreshPolicy {
    /// Phase-23 #4 behavior. Leave the IMU's `velocity_world`
    /// unchanged at every switch-back. The IMU continues integrating
    /// from whatever seed velocity it last computed; pose-mode-induced
    /// staleness is unmitigated.
    None,
    /// Phase-24 behavior (the post-refactor default). Recompute
    /// `velocity_world` from a finite-difference of the two most
    /// recent successful visual poses, divided by the IMU's
    /// pending-sample dt sum captured at the moment of the latest
    /// successful `observe`. Cheapest non-trivial reset; injects PnP
    /// noise from both poses into the velocity estimate.
    FiniteDifference,
    /// Phase-25 #1 (zero-reset) variant. Overwrite `velocity_world`
    /// with the zero vector at every switch-back. Cheapest possible
    /// reset; useful as a control when the cliff-region pose-mode
    /// poses are themselves PnP-noise-dominated (in which case any
    /// non-zero finite-difference seed is worse than zero).
    ZeroReset,
    /// Phase-25 #2 (smoothed finite-difference) variant. Computes two
    /// finite-difference velocities across the three most recent
    /// successful visual poses (oldest→previous and
    /// previous→latest) and averages them, then writes the result
    /// into `velocity_world`. Falls back to single
    /// finite-difference behavior when fewer than three poses are
    /// available. Aims to halve the PnP-noise variance compared with
    /// [`Self::FiniteDifference`].
    ThreePoseSmoother,
}

impl Default for ImuVelocityRefreshPolicy {
    /// Phase-25 default: three-pose smoothed finite-difference reset
    /// on switch. Empirically (Phase-25 EuRoC sweep, see
    /// `target/euroc_phase25_refresh_policy_ab/SUMMARY.md`) strictly
    /// improves on or matches [`Self::FiniteDifference`] on every
    /// 3-seq × 2-threshold case tested: identical at f=3/s=10 where
    /// the hook never fires or its result is washed out, and -25 %
    /// V2_01 / -1 % MH_01 / identical V1_01 rigid ATE at f=2/s=5
    /// compared with Phase-24's FiniteDifference default.
    fn default() -> Self {
        Self::ThreePoseSmoother
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveImuPoseMotionModelConfig {
    /// Number of consecutive failed-tracking frames (under IMU mode)
    /// that triggers a switch to constant-pose mode. Lower values
    /// react faster but oscillate more on noisy regimes.
    pub failures_to_switch_to_pose: usize,
    /// Number of consecutive successful-tracking frames (under
    /// constant-pose mode) that triggers a switch back to IMU mode.
    /// Higher values bias toward stability — the model stays in pose
    /// mode longer before re-trusting the IMU prediction.
    pub successes_to_switch_to_imu: usize,
    /// Policy for refreshing the wrapped IMU model's `velocity_world`
    /// at every Pose → IMU switch-back. See
    /// [`ImuVelocityRefreshPolicy`] for the semantics of each variant
    /// and the motivation behind the Phase-24 / Phase-25 thread.
    pub imu_velocity_refresh_policy: ImuVelocityRefreshPolicy,
}

impl Default for AdaptiveImuPoseMotionModelConfig {
    fn default() -> Self {
        Self {
            failures_to_switch_to_pose: 2,
            successes_to_switch_to_imu: 5,
            imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::default(),
        }
    }
}

/// Diagnostic snapshot of which inner motion model the adaptive
/// wrapper is currently dispatching predictions through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMotionMode {
    Imu,
    Pose,
}

/// Adaptive motion model that wraps an [`ImuPredictiveMotionModel`]
/// and a [`ConstantPoseMotionModel`] and dispatches `predict_pose`
/// through whichever inner model the per-frame
/// failure / success counters select. Both inner models are kept
/// fed at all times by [`Self::observe`] so the switch is
/// instantaneous when it fires (the IMU's pending-sample buffer and
/// last-successful-pose anchor stay current even while
/// constant-pose is dispatching predictions).
///
/// The IMU sample stream is forwarded into the inner IMU model via
/// the public [`Self::imu_mut`] accessor — the [`MotionModel`] trait
/// itself does not expose a sample-feeding entry point, so callers
/// who construct an `AdaptiveImuPoseMotionModel` and want the IMU
/// integration to stay current must push samples directly into the
/// wrapped model.
#[derive(Debug, Clone)]
pub struct AdaptiveImuPoseMotionModel {
    imu: ImuPredictiveMotionModel,
    pose: ConstantPoseMotionModel,
    config: AdaptiveImuPoseMotionModelConfig,
    mode: AdaptiveMotionMode,
    consecutive_failures_under_imu: usize,
    consecutive_successes_under_pose: usize,
    switches_to_pose: u64,
    switches_to_imu: u64,
    /// Pose from the third-most-recent successful `observe()`. Used
    /// only by the
    /// [`ImuVelocityRefreshPolicy::ThreePoseSmoother`] policy to form
    /// a second finite-difference velocity that gets averaged with the
    /// most recent finite-difference. `None` until at least three
    /// successful observations have occurred since construction or the
    /// last [`Self::reset`].
    oldest_successful_pose: Option<Pose>,
    /// Pose from the second-most-recent successful `observe()`. Paired
    /// with [`Self::latest_successful_pose`] +
    /// [`Self::dt_between_latest_two_observations`] to recompute the
    /// IMU `velocity_world` from a visual finite-difference at every
    /// Pose → IMU switch event under
    /// [`ImuVelocityRefreshPolicy::FiniteDifference`] /
    /// [`ImuVelocityRefreshPolicy::ThreePoseSmoother`].
    previous_successful_pose: Option<Pose>,
    latest_successful_pose: Option<Pose>,
    /// Wall-clock seconds elapsed between
    /// [`Self::oldest_successful_pose`] and
    /// [`Self::previous_successful_pose`]. Captured as the value of
    /// [`Self::dt_between_latest_two_observations`] one shift ago.
    /// Used only by [`ImuVelocityRefreshPolicy::ThreePoseSmoother`].
    dt_between_previous_two_observations: f64,
    /// Wall-clock seconds elapsed between
    /// [`Self::previous_successful_pose`] and
    /// [`Self::latest_successful_pose`], captured as the IMU's
    /// `pending_samples_total_dt()` value at the moment of the
    /// `latest_successful_pose`'s `observe()` call (i.e. before that
    /// call drained the pending buffer).
    dt_between_latest_two_observations: f64,
    /// Cumulative number of `imu_velocity_refresh_policy` hooks that
    /// actually fired (i.e. switches at which the configured policy
    /// found enough state to write a new `velocity_world`). Smaller
    /// than or equal to [`Self::switches_to_imu`]; useful for
    /// telemetry on whether the refresh policy is engaging.
    velocity_refreshes_on_switch_to_imu: u64,
}

impl AdaptiveImuPoseMotionModel {
    pub fn new(
        imu: ImuPredictiveMotionModel,
        pose: ConstantPoseMotionModel,
        config: AdaptiveImuPoseMotionModelConfig,
    ) -> Self {
        Self {
            imu,
            pose,
            config,
            mode: AdaptiveMotionMode::Imu,
            consecutive_failures_under_imu: 0,
            consecutive_successes_under_pose: 0,
            switches_to_pose: 0,
            switches_to_imu: 0,
            oldest_successful_pose: None,
            previous_successful_pose: None,
            latest_successful_pose: None,
            dt_between_previous_two_observations: 0.0,
            dt_between_latest_two_observations: 0.0,
            velocity_refreshes_on_switch_to_imu: 0,
        }
    }

    /// Construct an adaptive model with default-config inner models.
    /// Convenience for callers that want a one-shot setup with the
    /// default IMU gravity / biases / extrinsics.
    pub fn with_defaults() -> Self {
        Self::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig::default(),
        )
    }

    pub fn imu(&self) -> &ImuPredictiveMotionModel {
        &self.imu
    }

    /// Mutable access to the wrapped IMU model. Callers use this to
    /// forward raw IMU samples via
    /// [`ImuPredictiveMotionModel::push_imu_measurement`] and to
    /// mirror VI-BA-refined velocity / biases into the IMU state.
    pub fn imu_mut(&mut self) -> &mut ImuPredictiveMotionModel {
        &mut self.imu
    }

    pub fn config(&self) -> &AdaptiveImuPoseMotionModelConfig {
        &self.config
    }

    pub fn mode(&self) -> AdaptiveMotionMode {
        self.mode
    }

    /// Cumulative number of times the wrapper has switched from
    /// IMU → ConstantPose since construction (or last `reset`).
    pub fn switches_to_pose(&self) -> u64 {
        self.switches_to_pose
    }

    /// Cumulative number of times the wrapper has switched from
    /// ConstantPose → IMU since construction (or last `reset`).
    pub fn switches_to_imu(&self) -> u64 {
        self.switches_to_imu
    }

    /// Cumulative number of times the Phase-24
    /// refresh-IMU-velocity-on-switch-to-IMU hook has actually fired
    /// (i.e. the wrapper switched back to IMU AND
    /// `refresh_imu_velocity_on_switch_to_imu` was enabled AND both
    /// previous + latest visual poses were available AND
    /// `dt_between_latest_two_observations > 0`). Less than or equal
    /// to [`Self::switches_to_imu`].
    pub fn velocity_refreshes_on_switch_to_imu(&self) -> u64 {
        self.velocity_refreshes_on_switch_to_imu
    }
}

impl MotionModel for AdaptiveImuPoseMotionModel {
    fn predict_pose(
        &self,
        frame: &Frame,
        last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        match self.mode {
            AdaptiveMotionMode::Imu => {
                self.imu
                    .predict_pose(frame, last_result, last_successful_pose)
            }
            AdaptiveMotionMode::Pose => {
                self.pose
                    .predict_pose(frame, last_result, last_successful_pose)
            }
        }
    }

    fn observe(&mut self, result: &TrackingResult) {
        // Capture the IMU's pending-sample dt sum BEFORE forwarding to
        // inner models — `imu.observe()` drains the pending buffer on
        // success, so this is the only window in which we can read the
        // wall-clock time elapsed since the previous successful
        // `observe()`. Used by the Phase-24
        // refresh-IMU-velocity-on-switch hook to compute a visual
        // finite-difference velocity at the moment of a Pose → IMU
        // switch.
        let pending_dt_before_observe = self.imu.pending_samples_total_dt();
        // Keep both inner models current regardless of which one is
        // currently dispatching predictions — when the switch fires
        // the previously-dormant model must have a coherent state.
        self.imu.observe(result);
        self.pose.observe(result);
        if result.localization.success {
            if let Some(pose) = result.localization.pose.as_ref() {
                self.oldest_successful_pose = self.previous_successful_pose.take();
                self.previous_successful_pose = self.latest_successful_pose.take();
                self.latest_successful_pose = Some(pose.clone());
                self.dt_between_previous_two_observations = self.dt_between_latest_two_observations;
                self.dt_between_latest_two_observations = pending_dt_before_observe;
            }
            match self.mode {
                AdaptiveMotionMode::Imu => {
                    self.consecutive_failures_under_imu = 0;
                }
                AdaptiveMotionMode::Pose => {
                    self.consecutive_successes_under_pose += 1;
                    if self.consecutive_successes_under_pose
                        >= self.config.successes_to_switch_to_imu
                    {
                        self.mode = AdaptiveMotionMode::Imu;
                        self.switches_to_imu += 1;
                        self.consecutive_failures_under_imu = 0;
                        self.consecutive_successes_under_pose = 0;
                        self.maybe_refresh_imu_velocity_on_switch_to_imu();
                    }
                }
            }
        } else {
            match self.mode {
                AdaptiveMotionMode::Imu => {
                    self.consecutive_failures_under_imu += 1;
                    if self.consecutive_failures_under_imu >= self.config.failures_to_switch_to_pose
                    {
                        self.mode = AdaptiveMotionMode::Pose;
                        self.switches_to_pose += 1;
                        self.consecutive_failures_under_imu = 0;
                        self.consecutive_successes_under_pose = 0;
                    }
                }
                AdaptiveMotionMode::Pose => {
                    self.consecutive_successes_under_pose = 0;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.imu.reset();
        self.pose.reset();
        self.mode = AdaptiveMotionMode::Imu;
        self.consecutive_failures_under_imu = 0;
        self.consecutive_successes_under_pose = 0;
        self.switches_to_pose = 0;
        self.switches_to_imu = 0;
        self.oldest_successful_pose = None;
        self.previous_successful_pose = None;
        self.latest_successful_pose = None;
        self.dt_between_previous_two_observations = 0.0;
        self.dt_between_latest_two_observations = 0.0;
        self.velocity_refreshes_on_switch_to_imu = 0;
    }
}

impl AdaptiveImuPoseMotionModel {
    /// Phase-24 / Phase-25 IMU-velocity-refresh hook. Called at the
    /// instant of every Pose → IMU mode transition from
    /// [`MotionModel::observe`]. Dispatches on
    /// [`AdaptiveImuPoseMotionModelConfig::imu_velocity_refresh_policy`]
    /// and increments [`Self::velocity_refreshes_on_switch_to_imu`]
    /// every time it actually writes a new `velocity_world` (so the
    /// counter is `0` under [`ImuVelocityRefreshPolicy::None`] and
    /// whenever the configured policy lacks enough state to compute a
    /// value). Silent no-op when the policy cannot fire — the IMU
    /// then continues with its current `velocity_world`.
    fn maybe_refresh_imu_velocity_on_switch_to_imu(&mut self) {
        match self.config.imu_velocity_refresh_policy {
            ImuVelocityRefreshPolicy::None => {}
            ImuVelocityRefreshPolicy::ZeroReset => {
                self.imu.set_velocity_world(nalgebra::Vector3::zeros());
                self.velocity_refreshes_on_switch_to_imu += 1;
            }
            ImuVelocityRefreshPolicy::FiniteDifference => {
                let (Some(prev), Some(curr)) = (
                    self.previous_successful_pose.as_ref(),
                    self.latest_successful_pose.as_ref(),
                ) else {
                    return;
                };
                let dt = self.dt_between_latest_two_observations;
                let Some(v) = self
                    .imu
                    .body_velocity_from_camera_pose_difference(prev, curr, dt)
                else {
                    return;
                };
                self.imu.set_velocity_world(v);
                self.velocity_refreshes_on_switch_to_imu += 1;
            }
            ImuVelocityRefreshPolicy::ThreePoseSmoother => {
                let (Some(prev), Some(curr)) = (
                    self.previous_successful_pose.as_ref(),
                    self.latest_successful_pose.as_ref(),
                ) else {
                    return;
                };
                let dt_latest = self.dt_between_latest_two_observations;
                let Some(v_latest) = self
                    .imu
                    .body_velocity_from_camera_pose_difference(prev, curr, dt_latest)
                else {
                    return;
                };
                // If the oldest pose + a valid older dt are available,
                // compute a second finite-difference and average. When
                // they're not (fewer than 3 successes), fall back to
                // single-finite-difference semantics so the policy
                // degrades gracefully into FiniteDifference rather
                // than no-op'ing.
                let v_write = match self.oldest_successful_pose.as_ref() {
                    Some(oldest) => self
                        .imu
                        .body_velocity_from_camera_pose_difference(
                            oldest,
                            prev,
                            self.dt_between_previous_two_observations,
                        )
                        .map(|v_prev| (v_prev + v_latest) * 0.5)
                        .unwrap_or(v_latest),
                    None => v_latest,
                };
                self.imu.set_velocity_world(v_write);
                self.velocity_refreshes_on_switch_to_imu += 1;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tracker<P, M = ConstantPoseMotionModel> {
    localization_pipeline: P,
    motion_model: M,
    config: TrackingConfig,
    state: TrackingState,
    successive_failures: usize,
    last_result: Option<TrackingResult>,
    last_successful_frame_id: Option<FrameId>,
    last_successful_pose: Option<Pose>,
    stats: TrackingStats,
}

#[derive(Debug, Clone)]
pub struct ImageTracker<X, T = Tracker<LocalizationPipeline>> {
    pub extractor: X,
    pub tracker: T,
}

impl<X> ImageTracker<X, Tracker<LocalizationPipeline, ConstantPoseMotionModel>>
where
    X: FeatureExtractor,
{
    pub fn new(extractor: X, config: TrackingConfig) -> Self {
        Self {
            extractor,
            tracker: Tracker::new(LocalizationPipeline::default(), config),
        }
    }
}

impl<X, T> ImageTracker<X, T>
where
    X: FeatureExtractor,
{
    pub fn with_tracker(extractor: X, tracker: T) -> Self {
        Self { extractor, tracker }
    }

    pub fn tracker(&self) -> &T {
        &self.tracker
    }

    pub fn tracker_mut(&mut self) -> &mut T {
        &mut self.tracker
    }

    pub fn into_parts(self) -> (X, T) {
        (self.extractor, self.tracker)
    }
}

impl<P> Tracker<P, ConstantPoseMotionModel>
where
    P: FrameLocalizer,
{
    pub fn new(localization_pipeline: P, config: TrackingConfig) -> Self {
        Self::with_motion_model(localization_pipeline, ConstantPoseMotionModel, config)
    }
}

impl<P, M> Tracker<P, M>
where
    P: FrameLocalizer,
    M: MotionModel,
{
    pub fn with_motion_model(
        localization_pipeline: P,
        motion_model: M,
        config: TrackingConfig,
    ) -> Self {
        Self {
            localization_pipeline,
            motion_model,
            config,
            state: TrackingState::Uninitialized,
            successive_failures: 0,
            last_result: None,
            last_successful_frame_id: None,
            last_successful_pose: None,
            stats: TrackingStats::default(),
        }
    }

    pub fn state(&self) -> TrackingState {
        self.state
    }

    pub fn successive_failures(&self) -> usize {
        self.successive_failures
    }

    pub fn last_result(&self) -> Option<&TrackingResult> {
        self.last_result.as_ref()
    }

    pub fn last_successful_frame_id(&self) -> Option<FrameId> {
        self.last_successful_frame_id
    }

    pub fn last_successful_pose(&self) -> Option<&Pose> {
        self.last_successful_pose.as_ref()
    }

    pub fn stats(&self) -> &TrackingStats {
        &self.stats
    }

    pub fn motion_model(&self) -> &M {
        &self.motion_model
    }

    /// Mutable access to the configured motion model. Use this to feed
    /// out-of-band inputs (e.g., raw IMU samples into
    /// [`ImuPredictiveMotionModel`]) that the per-frame `track_frame*`
    /// path does not surface.
    pub fn motion_model_mut(&mut self) -> &mut M {
        &mut self.motion_model
    }

    pub fn reset(&mut self) {
        self.state = TrackingState::Uninitialized;
        self.successive_failures = 0;
        self.last_result = None;
        self.last_successful_frame_id = None;
        self.last_successful_pose = None;
        self.stats = TrackingStats::default();
        self.motion_model.reset();
    }

    /// Override the tracker's per-frame history with a successful
    /// relocalization recovery result. Called by callers that detect
    /// a primary `track_frame` failure and recover via a separate
    /// `FrameLocalizer` (e.g. the relocalization-on-tracker-death stage
    /// in `OnlineSlamPipeline`).
    ///
    /// Reverts the failed-frame side-effects from the primary attempt
    /// (`successive_failures` counter, `last_result` / `last_successful_*`
    /// fields, `motion_model.observe(failed_result)`) and re-runs them
    /// as if the recovered result had been the primary outcome. The
    /// failed-frame audit counter (`stats.failed_frame_count`) is left
    /// unchanged so the caller can tell that primary tracking dropped
    /// the frame before relocalization rescued it.
    ///
    /// No-op when `result.localization.success == false` (callers must
    /// gate on success before invoking).
    pub fn accept_relocalization_result(&mut self, result: TrackingResult) {
        if !result.localization.success {
            return;
        }
        self.state = TrackingState::Tracking;
        self.successive_failures = 0;
        self.last_successful_frame_id = Some(result.frame_id);
        self.last_successful_pose = result.localization.pose.clone();
        self.last_result = Some(result.clone());
        self.stats.relocalization_count += 1;
        self.stats.successful_frame_count += 1;
        if result.localization.inlier_count > 0 {
            self.stats.total_inlier_count += result.localization.inlier_count;
            self.stats.total_correspondence_count += result.localization.correspondence_count;
        }
        self.motion_model.observe(&result);
    }

    pub fn pose_prior_for_frame(&self, frame: &Frame) -> Option<Pose> {
        self.motion_model.predict_pose(
            frame,
            self.last_result.as_ref(),
            self.last_successful_pose.as_ref(),
        )
    }

    pub fn localization_prior_for_frame(&self, frame: &Frame, radius: f64) -> LocalizationPrior {
        if let Some(pose_prior) = self.pose_prior_for_frame(frame) {
            LocalizationPrior::from_pose(pose_prior, radius)
        } else {
            LocalizationPrior::none()
        }
    }

    pub fn track_frame(&mut self, frame: &Frame, map: &VisualMap) -> TrackingResult {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.track_frame_with_descriptor_store(frame, map, &descriptor_store)
    }

    pub fn track_frames(&mut self, frames: &[Frame], map: &VisualMap) -> Vec<TrackingResult> {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.track_frames_with_descriptor_store(frames, map, &descriptor_store)
    }

    pub fn track_frame_with_provider<P2>(&mut self, frame: &Frame, provider: &P2) -> TrackingResult
    where
        P2: MapProvider + DescriptorProvider,
    {
        let map = provider.visual_map();
        let map_stats = map_provider_stats(provider);
        if let Some(descriptor_store) = provider.landmark_descriptor_store() {
            self.track_frame_with_descriptor_store_and_map_stats(
                frame,
                map,
                descriptor_store,
                map_stats,
            )
        } else {
            let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
            let map_stats = MapProviderStats {
                descriptor_count: descriptor_store.len(),
                ..map_stats
            };
            self.track_frame_with_descriptor_store_and_map_stats(
                frame,
                map,
                &descriptor_store,
                map_stats,
            )
        }
    }

    pub fn track_frames_with_provider<P2>(
        &mut self,
        frames: &[Frame],
        provider: &P2,
    ) -> Vec<TrackingResult>
    where
        P2: MapProvider + DescriptorProvider,
    {
        frames
            .iter()
            .map(|frame| self.track_frame_with_provider(frame, provider))
            .collect()
    }

    pub fn track_frame_with_prior_submap_provider<P2>(
        &mut self,
        frame: &Frame,
        provider: &P2,
        radius: f64,
    ) -> TrackingResult
    where
        P2: MapProvider + DescriptorProvider,
    {
        if let Some(pose_prior) = self.pose_prior_for_frame(frame) {
            let submap_provider = InMemoryMapProvider::from_provider_radius(
                provider,
                pose_prior.camera_center_world(),
                radius,
            );
            self.track_frame_with_provider(frame, &submap_provider)
        } else {
            self.track_frame_with_provider(frame, provider)
        }
    }

    pub fn track_frames_with_prior_submap_provider<P2>(
        &mut self,
        frames: &[Frame],
        provider: &P2,
        radius: f64,
    ) -> Vec<TrackingResult>
    where
        P2: MapProvider + DescriptorProvider,
    {
        frames
            .iter()
            .map(|frame| self.track_frame_with_prior_submap_provider(frame, provider, radius))
            .collect()
    }

    pub fn track_frame_with_localization_prior_submap_provider<P2>(
        &mut self,
        frame: &Frame,
        provider: &P2,
        prior: &LocalizationPrior,
    ) -> TrackingResult
    where
        P2: MapProvider + DescriptorProvider,
    {
        if let (Some(center_world), Some(radius)) = (prior.center_world(), prior.radius) {
            let submap_provider =
                InMemoryMapProvider::from_provider_radius(provider, center_world, radius);
            let mut result = self.track_frame_with_provider(frame, &submap_provider);
            result.used_external_localization_prior = true;
            result.external_localization_prior_radius = Some(radius);
            self.stats.external_localization_prior_used_count += 1;
            self.last_result = Some(result.clone());
            result
        } else {
            self.track_frame_with_provider(frame, provider)
        }
    }

    pub fn track_frames_with_localization_prior_submap_provider<'a, P2, I>(
        &mut self,
        frames_and_priors: I,
        provider: &P2,
    ) -> Vec<TrackingResult>
    where
        P2: MapProvider + DescriptorProvider,
        I: IntoIterator<Item = (&'a Frame, Option<&'a LocalizationPrior>)>,
    {
        frames_and_priors
            .into_iter()
            .map(|(frame, prior)| {
                if let Some(prior) = prior {
                    self.track_frame_with_localization_prior_submap_provider(frame, provider, prior)
                } else {
                    self.track_frame_with_provider(frame, provider)
                }
            })
            .collect()
    }

    pub fn track_frame_with_descriptor_store(
        &mut self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> TrackingResult {
        let map_stats = MapProviderStats {
            camera_count: map.cameras.len(),
            landmark_count: map.landmarks.len(),
            keyframe_count: map.keyframes.len(),
            descriptor_count: descriptor_store.len(),
        };
        self.track_frame_with_descriptor_store_and_map_stats(
            frame,
            map,
            descriptor_store,
            map_stats,
        )
    }

    pub fn track_frames_with_descriptor_store(
        &mut self,
        frames: &[Frame],
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Vec<TrackingResult> {
        frames
            .iter()
            .map(|frame| self.track_frame_with_descriptor_store(frame, map, descriptor_store))
            .collect()
    }

    fn track_frame_with_descriptor_store_and_map_stats(
        &mut self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        map_stats: MapProviderStats,
    ) -> TrackingResult {
        let pose_prior = self.motion_model.predict_pose(
            frame,
            self.last_result.as_ref(),
            self.last_successful_pose.as_ref(),
        );
        let used_pose_prior =
            pose_prior.is_some() && self.config.last_pose_candidate_radius.is_some();

        let covisibility_local_store =
            self.build_covisibility_local_map_store(map, descriptor_store);
        let covisibility_local_map_size =
            covisibility_local_store.as_ref().map(|store| store.len());
        let active_descriptor_store: &LandmarkDescriptorStore = covisibility_local_store
            .as_ref()
            .unwrap_or(descriptor_store);

        let mut localization = if self.config.pnp_pose_prior_warm_start {
            self.localization_pipeline
                .localize_frame_with_pose_prior_warm_start_and_descriptor_store(
                    frame,
                    map,
                    active_descriptor_store,
                    pose_prior.as_ref(),
                    self.config.last_pose_candidate_radius,
                )
        } else {
            self.localization_pipeline
                .localize_frame_with_pose_prior_and_descriptor_store(
                    frame,
                    map,
                    active_descriptor_store,
                    pose_prior.as_ref(),
                    self.config.last_pose_candidate_radius,
                )
        };
        let tracking_failure_reason =
            self.apply_tracking_quality_gate(pose_prior.as_ref(), &mut localization);

        let previous_state = self.state;
        let event = if localization.success {
            self.state = TrackingState::Tracking;
            self.successive_failures = 0;
            match previous_state {
                TrackingState::Uninitialized => TrackingEvent::Initialized,
                TrackingState::Tracking => TrackingEvent::Tracked,
                TrackingState::Lost => TrackingEvent::Relocalized,
            }
        } else {
            self.successive_failures += 1;
            if self.successive_failures >= self.config.min_successive_failures_to_lost {
                self.state = TrackingState::Lost;
                TrackingEvent::Lost
            } else if self.state == TrackingState::Uninitialized {
                self.state = TrackingState::Uninitialized;
                TrackingEvent::TrackingFailed
            } else {
                self.state = TrackingState::Tracking;
                TrackingEvent::TrackingFailed
            }
        };

        let result = TrackingResult {
            frame_id: frame.id,
            state: self.state,
            event,
            successive_failures: self.successive_failures,
            pose_prior,
            used_pose_prior,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason,
            map_landmark_count: map_stats.landmark_count,
            map_stats,
            localization,
            covisibility_local_map_size,
        };

        self.update_history(&result);
        self.motion_model.observe(&result);
        result
    }

    /// Build a covisibility-graph-derived descriptor store, if the feature is
    /// enabled and the surrounding state allows it (tracker is in `Tracking`
    /// state with a known reference keyframe in `map`, the local-map landmark
    /// set is above `min_local_map_landmarks`). Returns `None` to signal that
    /// the caller should fall through to the original descriptor store.
    fn build_covisibility_local_map_store(
        &self,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Option<LandmarkDescriptorStore> {
        let config = self.config.covisibility_local_map.as_ref()?;
        if self.state != TrackingState::Tracking {
            return None;
        }
        let last_id = self.last_successful_frame_id?;
        let reference_kf = covisibility_pick_reference_keyframe(map, last_id)?;
        let local_landmarks = covisibility_local_map_landmarks(map, reference_kf, config);
        if local_landmarks.len() < config.min_local_map_landmarks {
            return None;
        }
        let mut filtered = LandmarkDescriptorStore::new();
        // Sort the local-landmark id set before iterating so downstream
        // consumers see a deterministic insertion order independent of
        // the per-process `HashSet` SipHash seed. Even though the
        // `LandmarkDescriptorStore` itself is a `HashMap`, this insulates
        // any caller that depends on observed insertion order.
        let mut local_landmark_ids: Vec<u64> = local_landmarks.iter().copied().collect();
        local_landmark_ids.sort();
        for landmark_id in &local_landmark_ids {
            if let Some(descriptor) = descriptor_store.get(*landmark_id) {
                filtered.insert(*landmark_id, descriptor.to_vec());
            }
        }
        if filtered.len() < config.min_local_map_landmarks {
            return None;
        }
        Some(filtered)
    }

    fn update_history(&mut self, result: &TrackingResult) {
        if self.stats.first_frame_id.is_none() {
            self.stats.first_frame_id = Some(result.frame_id);
        }
        self.stats.last_frame_id = Some(result.frame_id);
        self.stats.frame_count += 1;
        if result.localization.success {
            self.stats.successful_frame_count += 1;
            self.last_successful_frame_id = Some(result.frame_id);
            self.last_successful_pose = result.localization.pose.clone();
        } else {
            self.stats.failed_frame_count += 1;
        }

        if result.event == TrackingEvent::Lost {
            self.stats.lost_count += 1;
        }
        if result.event == TrackingEvent::Relocalized {
            self.stats.relocalization_count += 1;
        }
        if result.used_pose_prior {
            self.stats.pose_prior_used_count += 1;
        }
        if result.tracking_failure_reason.is_some() {
            self.stats.tracking_quality_gate_failure_count += 1;
        }
        if result.covisibility_local_map_size.is_some() {
            self.stats.covisibility_local_map_used_count += 1;
        }
        self.stats.total_inlier_count += result.localization.inlier_count;
        self.stats.total_correspondence_count += result.localization.correspondence_count;

        self.last_result = Some(result.clone());
    }

    fn apply_tracking_quality_gate(
        &self,
        pose_prior: Option<&Pose>,
        localization: &mut LocalizationResult,
    ) -> Option<TrackingFailureReason> {
        if !localization.success {
            return None;
        }

        if localization.inlier_count < self.config.min_inliers {
            *localization = localization.clone().rejected_by_quality_gate();
            return Some(TrackingFailureReason::InsufficientInliers {
                inlier_count: localization.inlier_count,
                min_inliers: self.config.min_inliers,
            });
        }

        if localization.inlier_ratio < self.config.min_inlier_ratio {
            *localization = localization.clone().rejected_by_quality_gate();
            return Some(TrackingFailureReason::InlierRatioTooLow {
                inlier_ratio: localization.inlier_ratio,
                min_inlier_ratio: self.config.min_inlier_ratio,
            });
        }

        if let (Some(reprojection_error), Some(max_reprojection_error)) = (
            localization.reprojection_error,
            self.config.max_mean_reprojection_error,
        ) {
            if reprojection_error > max_reprojection_error {
                *localization = localization.clone().rejected_by_quality_gate();
                return Some(TrackingFailureReason::MeanReprojectionErrorTooHigh {
                    reprojection_error,
                    max_reprojection_error,
                });
            }
        }

        let max_translation_error = self.config.max_pose_prior_translation_error?;
        let pose_prior = pose_prior?;
        let estimated_pose = localization.pose.as_ref()?;

        let translation_error =
            (estimated_pose.camera_center_world() - pose_prior.camera_center_world()).norm();
        if translation_error <= max_translation_error {
            return None;
        }

        *localization = localization.clone().rejected_by_quality_gate();
        Some(TrackingFailureReason::PosePriorTranslationErrorExceeded {
            translation_error,
            max_translation_error,
        })
    }
}

/// Resolve the reference keyframe for covisibility-based local-map selection.
///
/// Prefers a keyframe whose `frame.id` matches `last_id` exactly (when the
/// last successful frame was itself promoted to a keyframe). Otherwise falls
/// back to the keyframe with the largest `frame.id <= last_id` so we still
/// anchor to a temporally-nearby past keyframe.
fn covisibility_pick_reference_keyframe(map: &VisualMap, last_id: FrameId) -> Option<u64> {
    if map.keyframes.contains_key(&last_id) {
        return Some(last_id);
    }
    let mut best: Option<u64> = None;
    for kf_id in map.keyframes.keys() {
        if *kf_id > last_id {
            continue;
        }
        match best {
            None => best = Some(*kf_id),
            Some(current) if *kf_id > current => best = Some(*kf_id),
            _ => {}
        }
    }
    best
}

/// Compute the covisibility-derived local-map landmark set: union of
/// landmarks observed by the reference keyframe and the keyframes that share
/// at least `min_shared_landmarks` landmarks with it (capped at
/// `max_keyframes` co-visible neighbours, ranked by descending shared count).
fn covisibility_local_map_landmarks(
    map: &VisualMap,
    reference_kf_id: u64,
    config: &CovisibilityLocalMapConfig,
) -> HashSet<u64> {
    let mut local_landmarks: HashSet<u64> = HashSet::new();

    let Some(reference_kf) = map.keyframes.get(&reference_kf_id) else {
        return local_landmarks;
    };

    let reference_landmarks: HashSet<u64> = reference_kf
        .observations
        .iter()
        .map(|obs| obs.landmark_id)
        .collect();
    if reference_landmarks.is_empty() {
        return local_landmarks;
    }
    local_landmarks.extend(reference_landmarks.iter().copied());

    let mut shared_counts: HashMap<u64, usize> = HashMap::new();
    for landmark_id in &reference_landmarks {
        let Some(landmark) = map.landmarks.get(landmark_id) else {
            continue;
        };
        for obs in &landmark.observations {
            let kf_id = obs.frame_id;
            if kf_id == reference_kf_id {
                continue;
            }
            if !map.keyframes.contains_key(&kf_id) {
                continue;
            }
            *shared_counts.entry(kf_id).or_insert(0) += 1;
        }
    }

    let mut ranked: Vec<(u64, usize)> = shared_counts
        .into_iter()
        .filter(|(_, count)| *count >= config.min_shared_landmarks)
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if let Some(cap) = config.max_keyframes {
        ranked.truncate(cap);
    }

    for (kf_id, _) in &ranked {
        if let Some(kf) = map.keyframes.get(kf_id) {
            for obs in &kf.observations {
                local_landmarks.insert(obs.landmark_id);
            }
        }
    }

    local_landmarks
}

#[cfg(test)]
mod covisibility_local_map_tests {
    use super::*;
    use nalgebra::Point2;
    use visloc_core::types::{Camera, Keyframe, Landmark, Observation};

    fn make_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn make_keyframe(id: u64) -> Keyframe {
        Keyframe {
            frame: Frame::new(id, 1),
            observations: Vec::new(),
        }
    }

    fn make_landmark(id: u64) -> Landmark {
        Landmark::new(id, Point3::new(0.0, 0.0, 5.0))
    }

    fn link_observation(map: &mut VisualMap, kf_id: u64, landmark_id: u64) {
        let obs = Observation {
            frame_id: kf_id,
            landmark_id,
            keypoint_index: 0,
            xy: Point2::new(0.0, 0.0),
        };
        if let Some(kf) = map.keyframes.get_mut(&kf_id) {
            kf.observations.push(obs.clone());
        }
        if let Some(lm) = map.landmarks.get_mut(&landmark_id) {
            lm.observations.push(obs);
        }
    }

    fn make_three_kf_map() -> VisualMap {
        let mut map = VisualMap::new();
        let camera = make_camera();
        map.cameras.insert(camera.id, camera);
        for kf_id in [1, 2, 3] {
            map.keyframes.insert(kf_id, make_keyframe(kf_id));
        }
        for lm_id in 1..=20 {
            map.landmarks.insert(lm_id, make_landmark(lm_id));
        }
        for lm_id in 1..=10 {
            link_observation(&mut map, 1, lm_id);
            link_observation(&mut map, 2, lm_id);
        }
        for lm_id in 11..=15 {
            link_observation(&mut map, 2, lm_id);
            link_observation(&mut map, 3, lm_id);
        }
        for lm_id in 16..=20 {
            link_observation(&mut map, 3, lm_id);
        }
        map
    }

    #[test]
    fn picks_exact_keyframe_when_last_frame_id_matches() {
        let map = make_three_kf_map();
        assert_eq!(covisibility_pick_reference_keyframe(&map, 2), Some(2));
    }

    #[test]
    fn picks_nearest_prior_keyframe_when_last_frame_id_misses() {
        let map = make_three_kf_map();
        assert_eq!(covisibility_pick_reference_keyframe(&map, 5), Some(3));
    }

    #[test]
    fn returns_none_when_no_keyframe_is_in_past() {
        let map = make_three_kf_map();
        assert!(covisibility_pick_reference_keyframe(&map, 0).is_none());
    }

    #[test]
    fn covisibility_local_map_includes_reference_landmarks_and_neighbours() {
        let map = make_three_kf_map();
        let config = CovisibilityLocalMapConfig {
            max_keyframes: Some(5),
            min_shared_landmarks: 1,
            min_local_map_landmarks: 1,
        };
        let local = covisibility_local_map_landmarks(&map, 2, &config);
        // KF=2 sees landmarks {1..=15}. Co-visible with KF=1 (shares 1..=10) and
        // KF=3 (shares 11..=15). Union ∪ KF1 ∪ KF3 covers 1..=20.
        for lm in 1..=20 {
            assert!(local.contains(&lm), "missing landmark {}", lm);
        }
    }

    #[test]
    fn covisibility_local_map_drops_low_shared_neighbours() {
        let map = make_three_kf_map();
        let config = CovisibilityLocalMapConfig {
            max_keyframes: Some(5),
            min_shared_landmarks: 6, // KF3 shares only 5 with KF2, so it's dropped
            min_local_map_landmarks: 1,
        };
        let local = covisibility_local_map_landmarks(&map, 2, &config);
        // KF1 shares 10 with KF2 → kept (landmarks 1..=10 contributed via KF1).
        // KF3 shares 5 with KF2 → dropped, so landmarks 16..=20 (only in KF3)
        // should NOT appear.
        for lm in 1..=15 {
            assert!(local.contains(&lm), "expected landmark {}", lm);
        }
        for lm in 16..=20 {
            assert!(!local.contains(&lm), "unexpected landmark {}", lm);
        }
    }

    #[test]
    fn covisibility_local_map_respects_max_keyframes_cap() {
        let map = make_three_kf_map();
        let config = CovisibilityLocalMapConfig {
            max_keyframes: Some(1), // only the strongest neighbour
            min_shared_landmarks: 1,
            min_local_map_landmarks: 1,
        };
        let local = covisibility_local_map_landmarks(&map, 2, &config);
        // Strongest neighbour of KF2 is KF1 (10 shared) > KF3 (5 shared).
        // So KF3-only landmarks (16..=20) must NOT appear.
        for lm in 16..=20 {
            assert!(!local.contains(&lm), "unexpected landmark {} from KF3", lm);
        }
        // But reference-only landmarks 11..=15 (only in KF2 + KF3) should still
        // appear because they are in the reference KF.
        for lm in 11..=15 {
            assert!(local.contains(&lm), "expected reference landmark {}", lm);
        }
    }
}

impl<X, P, M> ImageTracker<X, Tracker<P, M>>
where
    X: FeatureExtractor,
    P: FrameLocalizer,
    M: MotionModel,
{
    pub fn reset(&mut self) {
        self.tracker.reset();
    }

    pub fn track_frame_image(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        map: &VisualMap,
    ) -> Result<TrackingResult, X::Error> {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.track_frame_image_with_descriptor_store(
            frame_id,
            camera_id,
            image,
            map,
            &descriptor_store,
        )
    }

    pub fn track_frame_images<'a, I>(
        &mut self,
        frames: I,
        map: &VisualMap,
    ) -> Result<Vec<TrackingResult>, X::Error>
    where
        X::Image: 'a,
        I: IntoIterator<Item = (FrameId, CameraId, &'a X::Image)>,
    {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.track_frame_images_with_descriptor_store(frames, map, &descriptor_store)
    }

    pub fn track_frame_image_with_descriptor_store(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Result<TrackingResult, X::Error> {
        let features = self.extractor.extract(image)?;
        let frame = frame_from_features(frame_id, camera_id, features);
        Ok(self
            .tracker
            .track_frame_with_descriptor_store(&frame, map, descriptor_store))
    }

    pub fn track_frame_images_with_descriptor_store<'a, I>(
        &mut self,
        frames: I,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Result<Vec<TrackingResult>, X::Error>
    where
        X::Image: 'a,
        I: IntoIterator<Item = (FrameId, CameraId, &'a X::Image)>,
    {
        let mut results = Vec::new();
        for (frame_id, camera_id, image) in frames {
            results.push(self.track_frame_image_with_descriptor_store(
                frame_id,
                camera_id,
                image,
                map,
                descriptor_store,
            )?);
        }
        Ok(results)
    }

    pub fn track_frame_image_with_provider<P2>(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        provider: &P2,
    ) -> Result<TrackingResult, X::Error>
    where
        P2: MapProvider + DescriptorProvider,
    {
        let features = self.extractor.extract(image)?;
        let frame = frame_from_features(frame_id, camera_id, features);
        Ok(self.tracker.track_frame_with_provider(&frame, provider))
    }

    pub fn track_frame_images_with_provider<'a, I, P2>(
        &mut self,
        frames: I,
        provider: &P2,
    ) -> Result<Vec<TrackingResult>, X::Error>
    where
        X::Image: 'a,
        I: IntoIterator<Item = (FrameId, CameraId, &'a X::Image)>,
        P2: MapProvider + DescriptorProvider,
    {
        let mut results = Vec::new();
        for (frame_id, camera_id, image) in frames {
            results
                .push(self.track_frame_image_with_provider(frame_id, camera_id, image, provider)?);
        }
        Ok(results)
    }

    pub fn track_frame_image_with_prior_submap_provider<P2>(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        provider: &P2,
        radius: f64,
    ) -> Result<TrackingResult, X::Error>
    where
        P2: MapProvider + DescriptorProvider,
    {
        let features = self.extractor.extract(image)?;
        let frame = frame_from_features(frame_id, camera_id, features);
        Ok(self
            .tracker
            .track_frame_with_prior_submap_provider(&frame, provider, radius))
    }

    pub fn track_frame_image_with_localization_prior_submap_provider<P2>(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        provider: &P2,
        prior: &LocalizationPrior,
    ) -> Result<TrackingResult, X::Error>
    where
        P2: MapProvider + DescriptorProvider,
    {
        let features = self.extractor.extract(image)?;
        let frame = frame_from_features(frame_id, camera_id, features);
        Ok(self
            .tracker
            .track_frame_with_localization_prior_submap_provider(&frame, provider, prior))
    }

    pub fn track_frame_images_with_prior_submap_provider<'a, I, P2>(
        &mut self,
        frames: I,
        provider: &P2,
        radius: f64,
    ) -> Result<Vec<TrackingResult>, X::Error>
    where
        X::Image: 'a,
        I: IntoIterator<Item = (FrameId, CameraId, &'a X::Image)>,
        P2: MapProvider + DescriptorProvider,
    {
        let mut results = Vec::new();
        for (frame_id, camera_id, image) in frames {
            results.push(self.track_frame_image_with_prior_submap_provider(
                frame_id, camera_id, image, provider, radius,
            )?);
        }
        Ok(results)
    }
}

fn frame_from_features(frame_id: FrameId, camera_id: CameraId, features: FeatureSet) -> Frame {
    Frame {
        id: frame_id,
        camera_id,
        keypoints: features.keypoints,
        descriptors: features.descriptors,
        pose: None,
    }
}

pub trait FrameLocalizer {
    fn localize_frame_with_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> LocalizationResult;

    fn localize_frame_with_pose_prior_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        _pose_prior: Option<&Pose>,
        _candidate_radius: Option<f64>,
    ) -> LocalizationResult {
        self.localize_frame_with_descriptor_store(frame, map, descriptor_store)
    }

    /// Variant that, in addition to the radius candidate filter, ALSO threads
    /// the pose prior into the PnP RANSAC as a warm-start hypothesis. Default
    /// impl falls back to the non-warm-start variant so existing implementors
    /// don't need to change.
    fn localize_frame_with_pose_prior_warm_start_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: Option<&Pose>,
        candidate_radius: Option<f64>,
    ) -> LocalizationResult {
        self.localize_frame_with_pose_prior_and_descriptor_store(
            frame,
            map,
            descriptor_store,
            pose_prior,
            candidate_radius,
        )
    }
}

impl<M, S, E> FrameLocalizer for LocalizationPipeline<M, S, E>
where
    M: Matcher + Clone,
    S: CandidateSelector + Clone,
    E: RobustPoseEstimator + Clone,
{
    fn localize_frame_with_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> LocalizationResult {
        LocalizationPipeline::localize_frame_with_descriptor_store(
            self,
            frame,
            map,
            descriptor_store,
        )
    }

    fn localize_frame_with_pose_prior_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: Option<&Pose>,
        candidate_radius: Option<f64>,
    ) -> LocalizationResult {
        let Some(radius) = candidate_radius else {
            return self.localize_frame_with_descriptor_store(frame, map, descriptor_store);
        };
        let Some(pose_prior) = pose_prior else {
            return self.localize_frame_with_descriptor_store(frame, map, descriptor_store);
        };

        let radius_selector = RadiusLandmarkSelector::new(pose_prior.camera_center_world(), radius);
        let candidate_selector =
            IntersectCandidateSelector::new(self.candidate_selector.clone(), radius_selector);

        LocalizationPipeline::localize_frame_with_candidate_selector_and_descriptor_store(
            self,
            frame,
            map,
            descriptor_store,
            candidate_selector,
        )
    }

    fn localize_frame_with_pose_prior_warm_start_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: Option<&Pose>,
        candidate_radius: Option<f64>,
    ) -> LocalizationResult {
        let Some(pose_prior) = pose_prior else {
            return self.localize_frame_with_descriptor_store(frame, map, descriptor_store);
        };
        let Some(camera) = map.cameras.get(&frame.camera_id).cloned() else {
            return LocalizationResult::failure(
                LocalizationFailureReason::MissingCamera {
                    camera_id: frame.camera_id,
                },
                0,
                0,
                0,
            );
        };
        let query = QueryImage::from_frame(frame, camera);

        if let Some(radius) = candidate_radius {
            let radius_selector =
                RadiusLandmarkSelector::new(pose_prior.camera_center_world(), radius);
            let candidate_selector =
                IntersectCandidateSelector::new(self.candidate_selector.clone(), radius_selector);
            self.localize_with_candidate_selector_and_descriptor_store_and_pose_prior(
                &query,
                map,
                descriptor_store,
                candidate_selector,
                Some(pose_prior),
            )
        } else {
            self.localize_with_candidate_selector_and_descriptor_store_and_pose_prior(
                &query,
                map,
                descriptor_store,
                self.candidate_selector.clone(),
                Some(pose_prior),
            )
        }
    }
}

#[cfg(test)]
mod imu_predictive_motion_tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    use visloc_core::geometry::Pose;
    use visloc_core::types::Frame;

    fn make_dummy_frame() -> Frame {
        Frame::new(1, 1)
    }

    fn rotation_angle_deg(a: &Pose, b: &Pose) -> f64 {
        let q_a = a.world_to_camera.rotation;
        let q_b = b.world_to_camera.rotation;
        q_a.rotation_to(&q_b).angle().to_degrees()
    }

    #[test]
    fn imu_predictive_motion_returns_last_pose_when_no_samples_pushed() {
        let model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        let prev =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 5.0));
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted pose");
        assert_eq!(
            predicted.world_to_camera.translation,
            prev.world_to_camera.translation
        );
        assert_eq!(
            predicted.world_to_camera.rotation,
            prev.world_to_camera.rotation
        );
    }

    #[test]
    fn imu_predictive_motion_returns_none_when_no_previous_pose() {
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        assert!(model
            .predict_pose(&make_dummy_frame(), None, None)
            .is_none());
    }

    #[test]
    fn imu_predictive_motion_stationary_under_gravity_holds_pose() {
        // Body stationary at world origin, gravity_world = (0, 0, -9.81)
        // (z-up). Accelerometer reads R_wb^T · (−g) = (0, 0, 9.81) when the
        // body is identity-oriented. With zero velocity / zero gyro, the
        // predicted pose must match the input exactly (no drift).
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::new(0.0, 0.0, -9.81),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        for _ in 0..10 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81), 0.05);
        }
        let prev = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        let predicted_center = predicted.camera_center_world();
        assert!(
            predicted_center.coords.norm() < 1.0e-9,
            "predicted center should stay at origin under stationary IMU, got {:?}",
            predicted_center
        );
        assert!(rotation_angle_deg(&predicted, &prev) < 1.0e-6);
    }

    #[test]
    fn imu_predictive_motion_pure_yaw_rotation_propagates_rotation() {
        // Zero gravity scene, body rotates at π/2 rad/s around world-z for
        // 1.0 s. Accel reading is zero (free fall in zero gravity).
        // Predicted pose's rotation should be a +90° yaw of the input.
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        let yaw_rate = std::f64::consts::FRAC_PI_2;
        for _ in 0..100 {
            model.push_imu_measurement(Vector3::new(0.0, 0.0, yaw_rate), Vector3::zeros(), 0.01);
        }
        let prev = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        // The body rotated +π/2 about world-z → R_bw is now a +90° rotation
        // about world-z → R_wc = R_bw⁻¹ is a -90° rotation about world-z.
        let expected_r_bw = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw_rate);
        let expected_r_wc = expected_r_bw.inverse();
        let actual_r_wc = predicted.world_to_camera.rotation;
        let angle_err = actual_r_wc.rotation_to(&expected_r_wc).angle().to_degrees();
        assert!(
            angle_err < 0.1,
            "yaw rotation drift too large: {angle_err} deg"
        );
    }

    #[test]
    fn imu_predictive_motion_constant_velocity_translates_position() {
        // Zero gravity, zero gyro, zero accel, but `velocity_world = (1, 0, 0)`
        // and 1 s of pending samples. Predicted position must be the input
        // position + (1, 0, 0).
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        model.set_velocity_world(Vector3::new(1.0, 0.0, 0.0));
        for _ in 0..100 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.01);
        }
        let prev = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        let center = predicted.camera_center_world();
        assert!(
            (center.x - 1.0).abs() < 1.0e-9,
            "predicted x should be 1.0 (start 0 + 1 m/s * 1 s), got {}",
            center.x
        );
        assert!(center.y.abs() < 1.0e-9);
        assert!(center.z.abs() < 1.0e-9);
    }

    #[test]
    fn imu_predictive_motion_observe_drains_pending_window_on_success() {
        use visloc_core::types::LocalizationSuccess;
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        for _ in 0..3 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.05);
        }
        assert_eq!(model.pending_samples_len(), 3);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let localization = LocalizationResult::success(LocalizationSuccess {
            pose,
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        let success = TrackingResult {
            frame_id: 1,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        };
        model.observe(&success);
        assert_eq!(model.pending_samples_len(), 0);
    }

    #[test]
    fn imu_predictive_motion_reset_clears_velocity_and_samples() {
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        model.set_velocity_world(Vector3::new(3.0, 0.0, 0.0));
        model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.05);
        model.reset();
        assert_eq!(model.pending_samples_len(), 0);
        assert_eq!(model.velocity_world(), Vector3::zeros());
    }

    #[test]
    fn imu_predictive_motion_carry_forward_default_off_leaves_velocity_frozen() {
        use visloc_core::types::LocalizationSuccess;
        // Default config: carry_forward_velocity_world = false. After
        // observe, velocity_world must equal the value set before push
        // (Phase-7 / pre-Phase-22 behaviour).
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        model.set_velocity_world(Vector3::new(1.0, 0.0, 0.0));
        for _ in 0..10 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::new(2.0, 0.0, 0.0), 0.1);
        }
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let localization = LocalizationResult::success(LocalizationSuccess {
            pose,
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        let result = TrackingResult {
            frame_id: 1,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        };
        model.observe(&result);
        let v = model.velocity_world();
        assert!((v - Vector3::new(1.0, 0.0, 0.0)).norm() < 1.0e-12);
    }

    #[test]
    fn imu_predictive_motion_carry_forward_on_advances_velocity_per_frame() {
        use visloc_core::types::LocalizationSuccess;
        // Zero gravity, identity body_to_sensor, accel (2,0,0) m/s² for
        // 1.0 s with initial v=(1,0,0). The body integrates to
        // v=(1+2*1.0,0,0)=(3,0,0). With carry-forward on, observe must
        // commit this back into velocity_world.
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            carry_forward_velocity_world: true,
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        model.set_velocity_world(Vector3::new(1.0, 0.0, 0.0));
        // Seed last_successful_pose with the body-at-origin pose by
        // running an initial observe (no pending samples → integration
        // is a no-op; effect is to populate last_successful_pose).
        let pose_zero = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let localization_zero = LocalizationResult::success(LocalizationSuccess {
            pose: pose_zero.clone(),
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        let make_result = |pose: Pose| TrackingResult {
            frame_id: 1,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization: LocalizationResult::success(LocalizationSuccess {
                pose,
                candidate_landmark_count: 4,
                match_count: 4,
                correspondence_count: 4,
                inliers: vec![0, 1, 2, 3],
                inlier_query_indices: vec![0, 1, 2, 3],
                inlier_landmark_ids: vec![1, 2, 3, 4],
                inlier_reprojection_errors: vec![0.0; 4],
                mean_reprojection_error: 0.0,
                median_reprojection_error: 0.0,
                max_reprojection_error: 0.0,
            }),
            covisibility_local_map_size: None,
        };
        // First observe: populate last_successful_pose. No samples yet.
        model.observe(&TrackingResult {
            localization: localization_zero,
            ..make_result(pose_zero.clone())
        });
        // Second window: push 1.0 s of accel (2,0,0) samples, then observe.
        for _ in 0..10 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::new(2.0, 0.0, 0.0), 0.1);
        }
        // The "tracked" pose passed to this observe is irrelevant for
        // the v_w commit (the integration anchors on the *previous*
        // pose); use any plausible value.
        let pose_next =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-2.0, 0.0, 0.0));
        model.observe(&make_result(pose_next));
        let v = model.velocity_world();
        assert!(
            (v - Vector3::new(3.0, 0.0, 0.0)).norm() < 1.0e-9,
            "velocity_world should have advanced from (1,0,0) by ∫(2,0,0) dt = (2,0,0) to (3,0,0); got {v:?}",
        );
    }

    #[test]
    fn imu_predictive_motion_carry_forward_reset_clears_last_successful_pose() {
        use visloc_core::types::LocalizationSuccess;
        // After reset, a single observe with carry-forward on but no
        // prior pose should NOT touch velocity_world (no anchor to
        // integrate from). Verifies the optional gate.
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            carry_forward_velocity_world: true,
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let localization = LocalizationResult::success(LocalizationSuccess {
            pose: pose.clone(),
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        let result = TrackingResult {
            frame_id: 1,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        };
        model.observe(&result);
        model.set_velocity_world(Vector3::new(5.0, 0.0, 0.0));
        model.push_imu_measurement(Vector3::zeros(), Vector3::new(99.0, 0.0, 0.0), 0.1);
        model.reset();
        // After reset, last_successful_pose is cleared; next observe
        // with pending samples must NOT advance velocity_world (it stays
        // at the post-reset zero).
        model.push_imu_measurement(Vector3::zeros(), Vector3::new(99.0, 0.0, 0.0), 0.1);
        model.observe(&result);
        assert_eq!(model.velocity_world(), Vector3::zeros());
    }

    #[test]
    fn imu_predictive_motion_t_bs_offset_preserves_extrinsic_under_translation() {
        // Body translates at +1 m/s along world-x for 0.5 s under
        // zero-gravity, zero gyro, zero accel. The camera sits at a
        // body-frame offset of (0.1, 0, 0): `body_to_sensor.translation`
        // is the body-frame coords of the sensor origin (i.e., the
        // camera origin expressed in body coords). The body moves from
        // 0 to 0.5 m, so the body should end at world-x 0.5 m and the
        // camera should end at world-x 0.6 m. The starting camera pose
        // is set up consistent with body-at-origin, so this is a clean
        // round-trip test of the T_BS plumbing.
        let body_to_sensor = SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0));
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            body_to_sensor: body_to_sensor.clone(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        model.set_velocity_world(Vector3::new(1.0, 0.0, 0.0));
        for _ in 0..50 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.01);
        }
        // Starting camera pose: body at world origin with identity
        // rotation → camera centre in world = body_to_sensor.translation.
        // The world_to_camera SE3 with `R_cw = I, t_cw = -(0.1, 0, 0)`
        // places `camera_center_world = (0.1, 0, 0)`.
        let prev =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.1, 0.0, 0.0));
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        let center = predicted.camera_center_world();
        assert!(
            (center.x - 0.6).abs() < 1.0e-9,
            "predicted camera x should be 0.6 (body 0.5 + extrinsic 0.1), got {}",
            center.x
        );
        assert!(center.y.abs() < 1.0e-9);
        assert!(center.z.abs() < 1.0e-9);
        // Camera orientation should be unchanged (pure translation).
        assert!(rotation_angle_deg(&predicted, &prev) < 1.0e-6);
    }

    #[test]
    fn imu_predictive_motion_update_velocity_from_camera_pose_diff_recovers_body_velocity() {
        // Body moved from (0,0,0) to (0.5, 0, 0) over 0.5 s → world-frame
        // body velocity should be (1, 0, 0). With `body_to_sensor =
        // identity` body==camera, so the camera-pose difference directly
        // reflects body motion.
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        let prev = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let curr =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.5, 0.0, 0.0));
        model.update_velocity_from_camera_pose_difference(&prev, &curr, 0.5);
        let v = model.velocity_world();
        assert!((v.x - 1.0).abs() < 1.0e-9);
        assert!(v.y.abs() < 1.0e-9);
        assert!(v.z.abs() < 1.0e-9);
    }

    #[test]
    fn imu_predictive_motion_update_velocity_with_lever_arm_uses_body_centre() {
        // Body translates from (0,0,0) to (0.5, 0, 0). The camera sits
        // 0.1 m ahead of body in body-x. So the camera moves from
        // (0.1, 0, 0) to (0.6, 0, 0). The velocity-update method must
        // strip the lever-arm offset and report the body's velocity
        // (1, 0, 0) instead of the camera's (which would be the same
        // here, but with a rotation the two would differ).
        let body_to_sensor = SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0));
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig {
            body_to_sensor,
            ..ImuPredictiveMotionModelConfig::default()
        });
        // Camera centre = (0.1, 0, 0) at t=0; (0.6, 0, 0) at t=0.5.
        // World_to_camera translation = -R_cw * camera_center_world.
        let prev =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.1, 0.0, 0.0));
        let curr =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.6, 0.0, 0.0));
        model.update_velocity_from_camera_pose_difference(&prev, &curr, 0.5);
        let v = model.velocity_world();
        assert!((v.x - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn imu_predictive_motion_update_velocity_rejects_nonpositive_dt() {
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        model.set_velocity_world(Vector3::new(5.0, 0.0, 0.0));
        let p = Pose::identity();
        model.update_velocity_from_camera_pose_difference(&p, &p, 0.0);
        assert_eq!(model.velocity_world(), Vector3::new(5.0, 0.0, 0.0));
        model.update_velocity_from_camera_pose_difference(&p, &p, -1.0);
        assert_eq!(model.velocity_world(), Vector3::new(5.0, 0.0, 0.0));
        model.update_velocity_from_camera_pose_difference(&p, &p, f64::NAN);
        assert_eq!(model.velocity_world(), Vector3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn imu_predictive_motion_t_bs_offset_propagates_rotation_with_lever_arm() {
        // Body rotates +π/2 about world-z over 1.0 s under zero gravity,
        // zero accel, zero initial velocity. Camera sits 0.1 m ahead of
        // body in body-frame x. As the body rotates +90° about z, the
        // camera's world position should trace an arc: start (0.1, 0, 0)
        // → end (0, 0.1, 0). With `body_to_sensor = identity` the camera
        // would just rotate in place at the origin — so this test is
        // specifically validating the T_BS lever-arm contribution to the
        // predicted camera centre.
        let body_to_sensor = SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0));
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            body_to_sensor: body_to_sensor.clone(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        let yaw_rate = std::f64::consts::FRAC_PI_2;
        for _ in 0..100 {
            model.push_imu_measurement(Vector3::new(0.0, 0.0, yaw_rate), Vector3::zeros(), 0.01);
        }
        // Starting camera pose corresponds to body at world origin
        // identity-oriented → camera centre = (0.1, 0, 0).
        let prev =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.1, 0.0, 0.0));
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        let center = predicted.camera_center_world();
        // After +90° yaw, body's +x-axis points to world +y. Camera is
        // at body-x=0.1, so its world position is (0, 0.1, 0).
        assert!(
            center.x.abs() < 1.0e-3,
            "predicted camera x should be ≈0 after +90° yaw with lever arm, got {}",
            center.x
        );
        assert!(
            (center.y - 0.1).abs() < 1.0e-3,
            "predicted camera y should be ≈0.1 after +90° yaw, got {}",
            center.y
        );
        assert!(center.z.abs() < 1.0e-9);
    }
}

#[cfg(test)]
mod umeyama_alignment_tests {
    use super::*;
    use nalgebra::{Point3, Rotation3, UnitQuaternion, Vector3};

    fn make_source() -> Vec<Point3<f64>> {
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(2.5, -1.0, 0.5),
        ]
    }

    fn transform_points(
        points: &[Point3<f64>],
        scale: f64,
        rotation: &Rotation3<f64>,
        translation: &Vector3<f64>,
    ) -> Vec<Point3<f64>> {
        points
            .iter()
            .map(|p| Point3::from(rotation * (p.coords * scale) + translation))
            .collect()
    }

    #[test]
    fn umeyama_recovers_pure_translation() {
        let source = make_source();
        let translation = Vector3::new(5.0, -2.0, 1.5);
        let target = transform_points(&source, 1.0, &Rotation3::identity(), &translation);
        let transform = umeyama_similarity_transform(&source, &target, false).expect("transform");
        assert!((transform.scale - 1.0).abs() < 1e-12);
        assert!((transform.translation - translation).norm() < 1e-10);
        assert!(transform.rotation.matrix().relative_eq(
            Rotation3::<f64>::identity().matrix(),
            1e-10,
            1e-10
        ));
    }

    #[test]
    fn umeyama_recovers_rotation_and_translation() {
        let source = make_source();
        let rotation = Rotation3::from_euler_angles(0.3, -0.4, 0.5);
        let translation = Vector3::new(-1.0, 2.0, -3.0);
        let target = transform_points(&source, 1.0, &rotation, &translation);
        let transform = umeyama_similarity_transform(&source, &target, false).expect("transform");
        assert!((transform.scale - 1.0).abs() < 1e-10);
        assert!((transform.translation - translation).norm() < 1e-8);
        assert!(transform
            .rotation
            .matrix()
            .relative_eq(rotation.matrix(), 1e-8, 1e-8));
        let aligned: Vec<Point3<f64>> = source.iter().map(|p| transform.apply(p)).collect();
        for (a, t) in aligned.iter().zip(target.iter()) {
            assert!((a - t).norm() < 1e-8);
        }
    }

    #[test]
    fn umeyama_recovers_similarity_with_scale() {
        let source = make_source();
        let scale = 3.4;
        let rotation = Rotation3::from_euler_angles(-0.2, 0.8, 0.1);
        let translation = Vector3::new(2.0, 0.5, -1.0);
        let target = transform_points(&source, scale, &rotation, &translation);
        let transform = umeyama_similarity_transform(&source, &target, true).expect("transform");
        assert!((transform.scale - scale).abs() < 1e-8);
        assert!((transform.translation - translation).norm() < 1e-7);
        assert!(transform
            .rotation
            .matrix()
            .relative_eq(rotation.matrix(), 1e-8, 1e-8));
    }

    #[test]
    fn umeyama_returns_none_for_insufficient_points() {
        let source = vec![Point3::new(0.0, 0.0, 0.0)];
        let target = vec![Point3::new(1.0, 0.0, 0.0)];
        assert!(umeyama_similarity_transform(&source, &target, false).is_none());
    }

    #[test]
    fn umeyama_returns_none_for_zero_variance_source() {
        let source = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0)];
        let target = vec![Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)];
        assert!(umeyama_similarity_transform(&source, &target, false).is_none());
    }

    fn pose_with_camera_center(frame_id: FrameId, center: Point3<f64>) -> TrajectorySample {
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -center.coords);
        TrajectorySample {
            frame_id,
            pose,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            inlier_count: 0,
            inlier_ratio: 0.0,
            reprojection_error: None,
        }
    }

    fn build_trajectory(centers: &[(FrameId, Point3<f64>)]) -> PoseTrajectory {
        let mut trajectory = PoseTrajectory::new();
        for (id, p) in centers {
            trajectory.push_sample(pose_with_camera_center(*id, *p));
        }
        trajectory
    }

    #[test]
    fn trajectory_alignment_umeyama_drives_error_to_zero_for_similarity_perturbation() {
        let scale = 2.0;
        let rotation = Rotation3::from_euler_angles(0.1, -0.2, 0.3);
        let translation = Vector3::new(7.0, -4.0, 2.0);
        let reference_centers: Vec<(FrameId, Point3<f64>)> = (0..6)
            .map(|i| {
                let p = Point3::new(i as f64, (i as f64).sin(), (i as f64).cos());
                (i as FrameId, p)
            })
            .collect();
        let estimated_centers: Vec<(FrameId, Point3<f64>)> = reference_centers
            .iter()
            .map(|(id, p)| {
                let inv_translation = -translation;
                let inv_rotation = rotation.inverse();
                let inv_scale = 1.0 / scale;
                let pre = p.coords + inv_translation;
                let body = inv_rotation * pre * inv_scale;
                (*id, Point3::from(body))
            })
            .collect();
        let reference = build_trajectory(&reference_centers);
        let estimated = build_trajectory(&estimated_centers);

        let aligned_summary = estimated.translation_error_summary_against_with_alignment(
            &reference,
            TrajectoryAlignment::UmeyamaWithScale,
        );
        let rmse = aligned_summary.rmse_translation_error.expect("rmse");
        assert!(
            rmse < 1e-7,
            "expected near-zero ATE after similarity alignment, got {rmse}"
        );

        let raw_summary = estimated.translation_error_summary_against_with_alignment(
            &reference,
            TrajectoryAlignment::None,
        );
        let raw_rmse = raw_summary.rmse_translation_error.expect("raw rmse");
        assert!(
            raw_rmse > 1.0,
            "expected substantial raw error before alignment, got {raw_rmse}"
        );
    }

    #[test]
    fn trajectory_alignment_umeyama_rigid_does_not_remove_scale() {
        let reference_centers: Vec<(FrameId, Point3<f64>)> = (0..5)
            .map(|i| (i as FrameId, Point3::new(i as f64, 0.0, 0.0)))
            .collect();
        let estimated_centers: Vec<(FrameId, Point3<f64>)> = reference_centers
            .iter()
            .map(|(id, p)| (*id, Point3::from(p.coords * 0.5)))
            .collect();
        let reference = build_trajectory(&reference_centers);
        let estimated = build_trajectory(&estimated_centers);

        let rigid = estimated.translation_error_summary_against_with_alignment(
            &reference,
            TrajectoryAlignment::Umeyama,
        );
        let scaled = estimated.translation_error_summary_against_with_alignment(
            &reference,
            TrajectoryAlignment::UmeyamaWithScale,
        );
        let rigid_rmse = rigid.rmse_translation_error.expect("rigid rmse");
        let scaled_rmse = scaled.rmse_translation_error.expect("scaled rmse");
        assert!(
            scaled_rmse < 1e-8,
            "expected near-zero ATE after similarity alignment, got {scaled_rmse}"
        );
        assert!(
            rigid_rmse > 0.1,
            "expected non-trivial residual under rigid alignment of half-scale trajectory, got {rigid_rmse}"
        );
    }

    fn rpe_sample(frame_id: FrameId, camera_to_world: SE3) -> TrajectorySample {
        let world_to_camera = camera_to_world.inverse();
        TrajectorySample {
            frame_id,
            pose: Pose::from_world_to_camera(world_to_camera.rotation, world_to_camera.translation),
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            inlier_count: 0,
            inlier_ratio: 0.0,
            reprojection_error: None,
        }
    }

    fn rpe_trajectory(poses: &[(FrameId, SE3)]) -> PoseTrajectory {
        let mut trajectory = PoseTrajectory::new();
        for (id, camera_to_world) in poses {
            trajectory.push_sample(rpe_sample(*id, camera_to_world.clone()));
        }
        trajectory
    }

    #[test]
    fn relative_pose_error_is_zero_when_estimate_matches_reference() {
        let poses: Vec<(FrameId, SE3)> = (0..8)
            .map(|i| {
                let c2w = SE3::new(
                    UnitQuaternion::from_euler_angles(0.0, 0.0, 0.05 * i as f64),
                    Vector3::new(i as f64, (i as f64 * 0.3).sin(), 0.0),
                );
                (i as FrameId, c2w)
            })
            .collect();
        let trajectory = rpe_trajectory(&poses);

        let summary = trajectory
            .relative_pose_error_against(&trajectory, &RelativePoseErrorConfig::default());

        assert_eq!(summary.pair_count, 7);
        let translation = summary.translation.expect("translation stats");
        let rotation = summary.rotation_deg.expect("rotation stats");
        assert!(
            translation.rmse < 1e-9,
            "translation rmse {}",
            translation.rmse
        );
        assert!(rotation.rmse < 1e-9, "rotation rmse {}", rotation.rmse);
    }

    #[test]
    fn relative_pose_error_is_invariant_to_a_global_rigid_transform() {
        // RPE compares *relative* motion, which is unchanged by left-multiplying
        // every pose by a fixed rigid transform — so a globally displaced and
        // rotated estimate scores ~0 RPE while its raw (unaligned) ATE is large.
        let global = SE3::new(
            UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
            Vector3::new(7.0, -4.0, 2.0),
        );
        let reference_poses: Vec<(FrameId, SE3)> = (0..10)
            .map(|i| {
                let c2w = SE3::new(
                    UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1 * i as f64),
                    Vector3::new(i as f64, (i as f64).sin(), (i as f64).cos()),
                );
                (i as FrameId, c2w)
            })
            .collect();
        let estimated_poses: Vec<(FrameId, SE3)> = reference_poses
            .iter()
            .map(|(id, c2w)| (*id, global.compose(c2w)))
            .collect();
        let reference = rpe_trajectory(&reference_poses);
        let estimated = rpe_trajectory(&estimated_poses);

        let summary = estimated.relative_pose_error_against(
            &reference,
            &RelativePoseErrorConfig {
                delta: 2,
                start_step: 1,
            },
        );
        assert_eq!(summary.delta, 2);
        let translation = summary.translation.expect("translation stats");
        let rotation = summary.rotation_deg.expect("rotation stats");
        assert!(
            translation.rmse < 1e-9 && rotation.rmse < 1e-9,
            "RPE should vanish under a global rigid transform, got t={} r={}",
            translation.rmse,
            rotation.rmse
        );

        // Contrast: the unaligned absolute trajectory error is far from zero.
        let ate = estimated
            .translation_error_summary_against_with_alignment(&reference, TrajectoryAlignment::None)
            .rmse_translation_error
            .expect("ate rmse");
        assert!(
            ate > 1.0,
            "expected large raw ATE for the displaced estimate, got {ate}"
        );
    }

    #[test]
    fn relative_pose_error_recovers_a_constant_per_step_drift() {
        // Reference advances one meter per step with no rotation. The estimate's
        // per-step relative motion carries a fixed extra transform `drift`, so
        // every residual equals `drift`: translation 0.1 m, rotation 5 deg.
        let step = SE3::new(UnitQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0));
        let drift = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 5.0_f64.to_radians()),
            Vector3::new(0.1, 0.0, 0.0),
        );

        let mut reference_poses = vec![(0u64, SE3::identity())];
        let mut estimated_poses = vec![(0u64, SE3::identity())];
        for i in 1..7u64 {
            let previous_reference = reference_poses[(i - 1) as usize].1.clone();
            reference_poses.push((i, previous_reference.compose(&step)));
            let previous_estimate = estimated_poses[(i - 1) as usize].1.clone();
            estimated_poses.push((i, previous_estimate.compose(&step).compose(&drift)));
        }
        let reference = rpe_trajectory(&reference_poses);
        let estimated = rpe_trajectory(&estimated_poses);

        let summary =
            estimated.relative_pose_error_against(&reference, &RelativePoseErrorConfig::default());
        let translation = summary.translation.expect("translation stats");
        let rotation = summary.rotation_deg.expect("rotation stats");
        // Tolerances absorb float accumulation through the multi-step compose
        // chain and quaternion angle extraction.
        assert!(
            (translation.mean - 0.1).abs() < 1e-6,
            "translation mean {}",
            translation.mean
        );
        assert!(
            translation.std < 1e-6,
            "translation std {}",
            translation.std
        );
        assert!(
            (rotation.mean - 5.0).abs() < 1e-6,
            "rotation mean {}",
            rotation.mean
        );
        assert!(rotation.std < 1e-6, "rotation std {}", rotation.std);
    }

    #[test]
    fn relative_pose_error_has_no_pairs_when_delta_exceeds_overlap() {
        let poses: Vec<(FrameId, SE3)> = (0..3)
            .map(|i| {
                (
                    i as FrameId,
                    SE3::new(UnitQuaternion::identity(), Vector3::new(i as f64, 0.0, 0.0)),
                )
            })
            .collect();
        let trajectory = rpe_trajectory(&poses);
        let summary = trajectory.relative_pose_error_against(
            &trajectory,
            &RelativePoseErrorConfig {
                delta: 5,
                start_step: 1,
            },
        );
        assert_eq!(summary.pair_count, 0);
        assert!(summary.translation.is_none());
        assert!(summary.rotation_deg.is_none());
    }

    fn fake_success_tracking_result(frame_id: u64) -> TrackingResult {
        fake_success_tracking_result_with_pose(
            frame_id,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros()),
        )
    }

    fn fake_success_tracking_result_with_pose(frame_id: u64, pose: Pose) -> TrackingResult {
        use visloc_core::types::LocalizationSuccess;
        let localization = LocalizationResult::success(LocalizationSuccess {
            pose,
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        TrackingResult {
            frame_id,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        }
    }

    fn fake_failure_tracking_result(frame_id: u64) -> TrackingResult {
        let localization = LocalizationResult::failure(
            visloc_core::types::LocalizationFailureReason::QualityGateFailed,
            0,
            0,
            0,
        );
        TrackingResult {
            frame_id,
            state: TrackingState::Tracking,
            event: TrackingEvent::TrackingFailed,
            successive_failures: 1,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: Some(TrackingFailureReason::InsufficientInliers {
                inlier_count: 0,
                min_inliers: 10,
            }),
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        }
    }

    #[test]
    fn adaptive_motion_starts_in_imu_mode() {
        let model = AdaptiveImuPoseMotionModel::with_defaults();
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_pose(), 0);
        assert_eq!(model.switches_to_imu(), 0);
    }

    #[test]
    fn adaptive_motion_switches_to_pose_after_consecutive_failures() {
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 2,
                successes_to_switch_to_imu: 5,
                ..AdaptiveImuPoseMotionModelConfig::default()
            },
        );
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        model.observe(&fake_failure_tracking_result(2));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        assert_eq!(model.switches_to_pose(), 1);
    }

    #[test]
    fn adaptive_motion_switches_back_to_imu_after_consecutive_successes() {
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 3,
                ..AdaptiveImuPoseMotionModelConfig::default()
            },
        );
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        // Three consecutive successes flip back to IMU.
        model.observe(&fake_success_tracking_result(2));
        model.observe(&fake_success_tracking_result(3));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        model.observe(&fake_success_tracking_result(4));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_pose(), 1);
        assert_eq!(model.switches_to_imu(), 1);
    }

    #[test]
    fn adaptive_motion_intermittent_failures_reset_pose_recovery_counter() {
        // While in Pose mode, a stray failure resets the
        // success-streak counter so the wrapper does not prematurely
        // switch back to IMU on a noisy regime.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 3,
                ..AdaptiveImuPoseMotionModelConfig::default()
            },
        );
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        model.observe(&fake_success_tracking_result(2));
        model.observe(&fake_success_tracking_result(3));
        model.observe(&fake_failure_tracking_result(4));
        model.observe(&fake_success_tracking_result(5));
        model.observe(&fake_success_tracking_result(6));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        model.observe(&fake_success_tracking_result(7));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
    }

    #[test]
    fn adaptive_motion_reset_clears_counters_and_mode() {
        let mut model = AdaptiveImuPoseMotionModel::with_defaults();
        // Force a switch.
        for f in 1..=3 {
            model.observe(&fake_failure_tracking_result(f));
        }
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        model.reset();
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_pose(), 0);
        assert_eq!(model.switches_to_imu(), 0);
    }

    // ---- Phase-24: IMU-velocity-refresh-on-switch-to-IMU ---------------

    /// World-frame camera-centre pose helper for adaptive Phase-24
    /// tests. `from_world_to_camera` takes `translation = -R · centre`,
    /// so for an identity rotation we negate the supplied `centre`.
    fn pose_at_world_centre(centre: Vector3<f64>) -> Pose {
        Pose::from_world_to_camera(UnitQuaternion::identity(), -centre)
    }

    #[test]
    fn adaptive_motion_refresh_imu_velocity_on_switch_recomputes_from_pose_diff() {
        // Refresh-on-switch enabled (the Phase-24 default). Drive a
        // failure to enter Pose mode, then two successes (each preceded
        // by an IMU sample so the pending-dt sum is well-defined). At
        // the second success we trip the switch-back threshold and the
        // refresh hook should rewrite IMU velocity_world to the visual
        // finite-difference (camera body moved +0.4m along world-x in
        // 0.1s ⇒ velocity_world ≈ (4, 0, 0)).
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::FiniteDifference,
            },
        );
        // Seed a non-zero stale velocity so we can prove the refresh
        // overwrote it.
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        // Push one IMU sample with dt=0.1, then a successful obs at the
        // origin → previous_successful_pose = origin, dt = 0.1.
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        // Push another IMU sample with dt=0.1, then a successful obs
        // displaced +0.4m along world-x. This is the second success →
        // mode switches back to IMU and the refresh hook fires.
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_imu(), 1);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        let v = model.imu().velocity_world();
        assert!(
            (v.x - 4.0).abs() < 1.0e-9 && v.y.abs() < 1.0e-9 && v.z.abs() < 1.0e-9,
            "refresh hook should overwrite stale velocity_world with visual \
             finite-difference (0.4 m / 0.1 s = 4 m/s along x); got {v:?}",
        );
    }

    #[test]
    fn adaptive_motion_refresh_disabled_leaves_imu_velocity_stale_on_switch() {
        // Same sequence as the previous test, but with the refresh
        // flag disabled. The seeded stale velocity_world must survive
        // the switch-back-to-IMU transition unchanged (this recovers
        // the Phase-23 #4 no-refresh behavior for A/B comparison).
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::None,
            },
        );
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_imu(), 1);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 0);
        let v = model.imu().velocity_world();
        assert_eq!(
            v,
            Vector3::new(99.0, 0.0, 0.0),
            "refresh disabled: stale velocity_world must survive the \
             switch-back unchanged; got {v:?}",
        );
    }

    #[test]
    fn adaptive_motion_reset_clears_refresh_state_and_counters() {
        // reset() must clear the Phase-24 recent-pose tracking state
        // (previous/latest successful pose, dt between them, and the
        // refresh counter) on top of the existing reset semantics.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::FiniteDifference,
            },
        );
        model.observe(&fake_failure_tracking_result(1));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        model.reset();
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 0);
        // After reset, a fresh failure → switch sequence must NOT
        // trigger a refresh (no recent poses remembered).
        model.observe(&fake_failure_tracking_result(10));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            11,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        // This second success triggers a switch-back. At that moment
        // we only have ONE post-reset success on record before this
        // one — previous_successful_pose was set by frame 11 and
        // latest_successful_pose will be set by frame 12. Both
        // present + positive dt ⇒ refresh DOES fire. The point of
        // this assertion is that the per-instance counter restarts
        // from zero after reset(), not that refreshes stop.
        model.observe(&fake_success_tracking_result_with_pose(
            12,
            pose_at_world_centre(Vector3::new(0.2, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
    }

    // ---- Phase-25: ZeroReset + ThreePoseSmoother A/B variants ----------

    #[test]
    fn adaptive_motion_refresh_zero_reset_zeroes_imu_velocity_on_switch() {
        // The ZeroReset policy must overwrite the IMU's
        // `velocity_world` with the zero vector at every switch-back,
        // independent of whether any visual poses are on record. The
        // counter must increment because the policy fired.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::ZeroReset,
            },
        );
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_imu(), 1);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        let v = model.imu().velocity_world();
        assert_eq!(
            v,
            Vector3::zeros(),
            "ZeroReset policy must overwrite stale velocity_world with \
             zeros at switch; got {v:?}",
        );
    }

    #[test]
    fn adaptive_motion_refresh_three_pose_smoother_averages_two_finite_diffs() {
        // Three successive successful observations along world-x at
        //   t0: pose @ x=0          dt = irrelevant (first success)
        //   t1: pose @ x=0.4   dt1 = 0.1   ⇒ v_prev   = 4 m/s
        //   t2: pose @ x=1.2   dt2 = 0.1   ⇒ v_latest = 8 m/s
        // ThreePoseSmoother averages them ⇒ writes (4+8)/2 = 6 m/s.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 3,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::ThreePoseSmoother,
            },
        );
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        // 1st success: oldest=None, prev=None, latest=pose@0
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        // 2nd success: oldest=None, prev=pose@0, latest=pose@0.4
        // dt_latest=0.1, dt_prev=0.1 (carried from the prior tick — but
        // there was no previous pose so the smoother's oldest branch
        // does not trigger). Counter still 0 (no switch yet).
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        // 3rd success: oldest=pose@0, prev=pose@0.4, latest=pose@1.2.
        // dt_prev=0.1 (was dt_latest one shift ago), dt_latest=0.1.
        // consecutive_successes reaches threshold → switch + smoother
        // fires with both 3-pose entries available.
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            4,
            pose_at_world_centre(Vector3::new(1.2, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_imu(), 1);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        let v = model.imu().velocity_world();
        assert!(
            (v.x - 6.0).abs() < 1.0e-9 && v.y.abs() < 1.0e-9 && v.z.abs() < 1.0e-9,
            "ThreePoseSmoother should average two finite-diffs \
             (4 m/s and 8 m/s along x ⇒ 6 m/s); got {v:?}",
        );
    }

    #[test]
    fn adaptive_motion_refresh_three_pose_smoother_falls_back_when_only_two_poses() {
        // Same setup as the FiniteDifference test (only TWO successful
        // poses before the switch), but with the policy set to
        // ThreePoseSmoother. The fallback must compute a single
        // finite-difference (4 m/s along x) rather than no-op.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::ThreePoseSmoother,
            },
        );
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        let v = model.imu().velocity_world();
        assert!(
            (v.x - 4.0).abs() < 1.0e-9 && v.y.abs() < 1.0e-9 && v.z.abs() < 1.0e-9,
            "ThreePoseSmoother fallback (only 2 poses) should match \
             single finite-difference; got {v:?}",
        );
    }
}
