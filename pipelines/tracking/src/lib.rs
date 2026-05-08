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
    CameraId, Frame, FrameId, LandmarkDescriptorStore, LocalizationResult, VisualMap,
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
    pub tracking_failure_reason: Option<TrackingFailureReason>,
    pub map_landmark_count: usize,
    pub map_stats: MapProviderStats,
    pub localization: LocalizationResult,
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

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
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

    fn translation_alignment_offset(
        &self,
        reference_by_frame_id: &HashMap<FrameId, Point3<f64>>,
        alignment: TrajectoryAlignment,
    ) -> Vector3<f64> {
        match alignment {
            TrajectoryAlignment::None => Vector3::zeros(),
            TrajectoryAlignment::FirstMatchedTranslation => self
                .samples
                .iter()
                .find_map(|sample| {
                    let reference_center = reference_by_frame_id.get(&sample.frame_id)?;
                    Some(sample.camera_center_world() - *reference_center)
                })
                .unwrap_or_else(Vector3::zeros),
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
        let translation_offset =
            self.translation_alignment_offset(&reference_by_frame_id, alignment);

        self.samples
            .iter()
            .filter_map(|sample| {
                let reference_center = reference_by_frame_id.get(&sample.frame_id)?;
                Some(TrajectoryTranslationError {
                    frame_id: sample.frame_id,
                    translation_error: ((sample.camera_center_world() - *reference_center)
                        - translation_offset)
                        .norm(),
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
    let offset = estimated.translation_alignment_offset(&reference_by_frame_id, alignment);
    let estimated_points = estimated
        .samples
        .iter()
        .map(|sample| sample.camera_center_world() - offset)
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
        let estimated_center = sample.camera_center_world() - offset;
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
    pub tracking_quality_gate_failure_count: usize,
    pub total_inlier_count: usize,
    pub total_correspondence_count: usize,
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
            if result.tracking_failure_reason.is_some() {
                stats.tracking_quality_gate_failure_count += 1;
            }
            stats.total_inlier_count += result.localization.inlier_count;
            stats.total_correspondence_count += result.localization.correspondence_count;
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
                "  \"tracking_quality_gate_failure_count\": {},\n",
                "  \"total_inlier_count\": {},\n",
                "  \"total_correspondence_count\": {},\n",
                "  \"success_rate\": {},\n",
                "  \"failure_rate\": {},\n",
                "  \"pose_prior_usage_rate\": {},\n",
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
            self.tracking_quality_gate_failure_count,
            self.total_inlier_count,
            self.total_correspondence_count,
            self.success_rate(),
            self.failure_rate(),
            self.pose_prior_usage_rate(),
            self.overall_inlier_ratio(),
            self.mean_inliers_per_successful_frame(),
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
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
        "Mean inliers",
        &format!("{:.1}", stats.mean_inliers_per_successful_frame()),
    );
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n");
    output.push_str(&tracking_timeline_svg(results));
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n<h2>Frames</h2>\n");
    output.push_str("<table><thead><tr><th>frame</th><th>success</th><th>state</th><th>event</th><th>inliers</th><th>ratio</th><th>reprojection</th><th>prior</th><th>reason</th></tr></thead><tbody>\n");
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
            if result.used_pose_prior { "yes" } else { "no" },
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
        "frame_id,state,event,success,successive_failures,used_pose_prior,tracking_failure_reason,localization_failure_reason,candidate_landmark_count,match_count,correspondence_count,inlier_count,outlier_count,inlier_ratio,reprojection_error,median_reprojection_error,max_reprojection_error,map_cameras,map_keyframes,map_landmarks,map_descriptors\n",
    );
    for result in results {
        let _ = writeln!(
            output,
            "{},{:?},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            result.frame_id,
            result.state,
            result.event,
            result.localization.success,
            result.successive_failures,
            result.used_pose_prior,
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

    pub fn reset(&mut self) {
        self.state = TrackingState::Uninitialized;
        self.successive_failures = 0;
        self.last_result = None;
        self.last_successful_frame_id = None;
        self.last_successful_pose = None;
        self.stats = TrackingStats::default();
        self.motion_model.reset();
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
            self.track_frame_with_provider(frame, &submap_provider)
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
        let mut localization = self
            .localization_pipeline
            .localize_frame_with_pose_prior_and_descriptor_store(
                frame,
                map,
                descriptor_store,
                pose_prior.as_ref(),
                self.config.last_pose_candidate_radius,
            );
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
            tracking_failure_reason,
            map_landmark_count: map_stats.landmark_count,
            map_stats,
            localization,
        };

        self.update_history(&result);
        self.motion_model.observe(&result);
        result
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
}
