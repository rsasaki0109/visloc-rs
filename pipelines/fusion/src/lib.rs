#![forbid(unsafe_code)]
//! Loose-coupling sensor-fusion foundations.
//!
//! This crate does not implement a full GNSS/INS/VIO optimizer. It provides
//! timestamped measurement types and conversions into localization priors so
//! visual localization, tracking, and SLAM pipelines can use external sensors
//! without depending on a specific robotics stack.

use std::collections::HashMap;

use nalgebra::{Matrix3, Point3, SMatrix, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::{Frame, FrameId};
use visloc_localization::LocalizationPrior;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    nanoseconds: i128,
}

impl Timestamp {
    pub fn from_nanoseconds(nanoseconds: i128) -> Self {
        Self { nanoseconds }
    }

    pub fn from_seconds_nanoseconds(seconds: i64, nanoseconds: u32) -> Self {
        Self {
            nanoseconds: seconds as i128 * 1_000_000_000 + nanoseconds as i128,
        }
    }

    pub fn as_nanoseconds(&self) -> i128 {
        self.nanoseconds
    }

    pub fn duration_since(&self, earlier: Self) -> Option<TimeDelta> {
        let nanoseconds = self.nanoseconds.checked_sub(earlier.nanoseconds)?;
        if nanoseconds < 0 {
            None
        } else {
            Some(TimeDelta { nanoseconds })
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimeDelta {
    nanoseconds: i128,
}

impl TimeDelta {
    pub fn from_nanoseconds(nanoseconds: i128) -> Self {
        Self { nanoseconds }
    }

    pub fn as_nanoseconds(&self) -> i128 {
        self.nanoseconds
    }

    pub fn as_seconds_f64(&self) -> f64 {
        self.nanoseconds as f64 / 1_000_000_000.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Timed<T> {
    pub timestamp: Timestamp,
    pub value: T,
}

impl<T> Timed<T> {
    pub fn new(timestamp: Timestamp, value: T) -> Self {
        Self { timestamp, value }
    }
}

impl<T> TimedMeasurement for Timed<T> {
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

pub type TimedFrame = Timed<Frame>;
pub type TimedPose = Timed<Pose>;

pub trait TimedMeasurement {
    fn timestamp(&self) -> Timestamp;
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameTimestampIndex {
    timestamps: HashMap<FrameId, Timestamp>,
}

impl FrameTimestampIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_timed_frames<I>(frames: I) -> Self
    where
        I: IntoIterator<Item = TimedFrame>,
    {
        let mut index = Self::new();
        for frame in frames {
            index.insert_frame_id(frame.value.id, frame.timestamp);
        }
        index
    }

    pub fn insert_frame(&mut self, frame: &Frame, timestamp: Timestamp) -> Option<Timestamp> {
        self.insert_frame_id(frame.id, timestamp)
    }

    pub fn insert_frame_id(
        &mut self,
        frame_id: FrameId,
        timestamp: Timestamp,
    ) -> Option<Timestamp> {
        self.timestamps.insert(frame_id, timestamp)
    }

    pub fn timestamp_for_frame(&self, frame: &Frame) -> Option<Timestamp> {
        self.timestamp_for_frame_id(frame.id)
    }

    pub fn timestamp_for_frame_id(&self, frame_id: FrameId) -> Option<Timestamp> {
        self.timestamps.get(&frame_id).copied()
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (FrameId, Timestamp)> + '_ {
        self.timestamps
            .iter()
            .map(|(&frame_id, &timestamp)| (frame_id, timestamp))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PriorConfig {
    pub default_radius: f64,
    pub min_radius: f64,
    pub confidence_multiplier: f64,
}

impl Default for PriorConfig {
    fn default() -> Self {
        Self {
            default_radius: 50.0,
            min_radius: 5.0,
            confidence_multiplier: 3.0,
        }
    }
}

pub trait LocalizationPriorProvider: TimedMeasurement {
    fn localization_prior(&self, config: &PriorConfig) -> Option<LocalizationPrior>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FramePriorSyncSummary {
    pub frame_count: usize,
    pub measurement_count: usize,
    pub matched_frame_count: usize,
    pub missing_measurement_count: usize,
}

impl FramePriorSyncSummary {
    pub fn matched_frame_ratio(&self) -> f64 {
        if self.frame_count == 0 {
            0.0
        } else {
            self.matched_frame_count as f64 / self.frame_count as f64
        }
    }

    pub fn all_frames_matched(&self) -> bool {
        self.missing_measurement_count == 0
    }

    pub fn evaluate(
        &self,
        config: FramePriorSyncEvaluationConfig,
    ) -> FramePriorSyncEvaluationResult {
        let mut failures = Vec::new();
        if let Some(minimum) = config.min_matched_frame_count {
            if self.matched_frame_count < minimum {
                failures.push(FramePriorSyncEvaluationFailure::MatchedFrameCountTooLow {
                    actual: self.matched_frame_count,
                    minimum,
                });
            }
        }
        if let Some(minimum) = config.min_matched_frame_ratio {
            let actual = self.matched_frame_ratio();
            if actual < minimum {
                failures.push(FramePriorSyncEvaluationFailure::MatchedFrameRatioTooLow {
                    actual,
                    minimum,
                });
            }
        }

        FramePriorSyncEvaluationResult {
            passed: failures.is_empty(),
            summary: *self,
            config,
            failures,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FramePriorSyncEvaluationConfig {
    pub min_matched_frame_count: Option<usize>,
    pub min_matched_frame_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FramePriorSyncEvaluationFailure {
    MatchedFrameCountTooLow { actual: usize, minimum: usize },
    MatchedFrameRatioTooLow { actual: f64, minimum: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct FramePriorSyncEvaluationResult {
    pub passed: bool,
    pub summary: FramePriorSyncSummary,
    pub config: FramePriorSyncEvaluationConfig,
    pub failures: Vec<FramePriorSyncEvaluationFailure>,
}

pub type PoseCovarianceMatrix = SMatrix<f64, 6, 6>;

#[derive(Debug, Clone, PartialEq)]
pub struct PositionCovariance {
    pub matrix: Matrix3<f64>,
}

impl PositionCovariance {
    pub fn new(matrix: Matrix3<f64>) -> Self {
        Self { matrix }
    }

    pub fn from_standard_deviations(standard_deviations: Vector3<f64>) -> Self {
        Self {
            matrix: Matrix3::from_diagonal(&standard_deviations.map(|sigma| sigma * sigma)),
        }
    }

    pub fn max_standard_deviation(&self) -> Option<f64> {
        max_diagonal_standard_deviation(self.matrix.diagonal().iter().copied())
    }

    pub fn horizontal_standard_deviation(&self) -> Option<f64> {
        max_diagonal_standard_deviation([self.matrix[(0, 0)], self.matrix[(1, 1)]])
    }

    pub fn vertical_standard_deviation(&self) -> Option<f64> {
        standard_deviation_from_variance(self.matrix[(2, 2)])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseCovariance {
    pub matrix: PoseCovarianceMatrix,
}

impl PoseCovariance {
    pub fn new(matrix: PoseCovarianceMatrix) -> Self {
        Self { matrix }
    }

    pub fn from_translation_rotation_standard_deviations(
        translation_standard_deviations: Vector3<f64>,
        rotation_standard_deviations: Vector3<f64>,
    ) -> Self {
        let mut matrix = PoseCovarianceMatrix::zeros();
        for axis in 0..3 {
            matrix[(axis, axis)] =
                translation_standard_deviations[axis] * translation_standard_deviations[axis];
            matrix[(axis + 3, axis + 3)] =
                rotation_standard_deviations[axis] * rotation_standard_deviations[axis];
        }
        Self { matrix }
    }

    pub fn max_translation_standard_deviation(&self) -> Option<f64> {
        max_diagonal_standard_deviation([
            self.matrix[(0, 0)],
            self.matrix[(1, 1)],
            self.matrix[(2, 2)],
        ])
    }

    pub fn max_rotation_standard_deviation(&self) -> Option<f64> {
        max_diagonal_standard_deviation([
            self.matrix[(3, 3)],
            self.matrix[(4, 4)],
            self.matrix[(5, 5)],
        ])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementBuffer<T> {
    measurements: Vec<T>,
}

impl<T> Default for MeasurementBuffer<T> {
    fn default() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }
}

impl<T> MeasurementBuffer<T>
where
    T: TimedMeasurement,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_measurements<I>(measurements: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut buffer = Self::new();
        for measurement in measurements {
            buffer.push(measurement);
        }
        buffer
    }

    pub fn push(&mut self, measurement: T) {
        let timestamp = measurement.timestamp();
        let index = self
            .measurements
            .partition_point(|existing| existing.timestamp() <= timestamp);
        self.measurements.insert(index, measurement);
    }

    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.measurements.iter()
    }

    pub fn latest_before_or_at(&self, timestamp: Timestamp) -> Option<&T> {
        let index = self
            .measurements
            .partition_point(|measurement| measurement.timestamp() <= timestamp);
        index
            .checked_sub(1)
            .and_then(|latest_index| self.measurements.get(latest_index))
    }

    pub fn nearest(&self, timestamp: Timestamp, tolerance: TimeDelta) -> Option<&T> {
        let index = self
            .measurements
            .partition_point(|measurement| measurement.timestamp() < timestamp);

        let before = index
            .checked_sub(1)
            .and_then(|before_index| self.measurements.get(before_index));
        let after = self.measurements.get(index);

        [before, after]
            .into_iter()
            .flatten()
            .filter_map(|measurement| {
                timestamp_distance(measurement.timestamp(), timestamp)
                    .filter(|distance| *distance <= tolerance)
                    .map(|distance| (distance, measurement))
            })
            .min_by_key(|(distance, _)| distance.as_nanoseconds())
            .map(|(_, measurement)| measurement)
    }

    pub fn nearest_localization_prior(
        &self,
        timestamp: Timestamp,
        tolerance: TimeDelta,
        config: &PriorConfig,
    ) -> Option<LocalizationPrior>
    where
        T: LocalizationPriorProvider,
    {
        self.nearest(timestamp, tolerance)?
            .localization_prior(config)
    }

    pub fn nearest_for_frame(
        &self,
        frame: &Frame,
        frame_timestamps: &FrameTimestampIndex,
        tolerance: TimeDelta,
    ) -> Option<&T> {
        let timestamp = frame_timestamps.timestamp_for_frame(frame)?;
        self.nearest(timestamp, tolerance)
    }

    pub fn nearest_for_frame_id(
        &self,
        frame_id: FrameId,
        frame_timestamps: &FrameTimestampIndex,
        tolerance: TimeDelta,
    ) -> Option<&T> {
        let timestamp = frame_timestamps.timestamp_for_frame_id(frame_id)?;
        self.nearest(timestamp, tolerance)
    }

    pub fn nearest_localization_prior_for_frame(
        &self,
        frame: &Frame,
        frame_timestamps: &FrameTimestampIndex,
        tolerance: TimeDelta,
        config: &PriorConfig,
    ) -> Option<LocalizationPrior>
    where
        T: LocalizationPriorProvider,
    {
        self.nearest_for_frame(frame, frame_timestamps, tolerance)?
            .localization_prior(config)
    }

    pub fn nearest_localization_prior_for_frame_id(
        &self,
        frame_id: FrameId,
        frame_timestamps: &FrameTimestampIndex,
        tolerance: TimeDelta,
        config: &PriorConfig,
    ) -> Option<LocalizationPrior>
    where
        T: LocalizationPriorProvider,
    {
        self.nearest_for_frame_id(frame_id, frame_timestamps, tolerance)?
            .localization_prior(config)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FramePriorSource<T> {
    pub frame_timestamps: FrameTimestampIndex,
    pub measurements: MeasurementBuffer<T>,
    pub tolerance: TimeDelta,
    pub prior_config: PriorConfig,
}

impl<T> FramePriorSource<T>
where
    T: TimedMeasurement,
{
    pub fn new(
        frame_timestamps: FrameTimestampIndex,
        measurements: MeasurementBuffer<T>,
        tolerance: TimeDelta,
    ) -> Self {
        Self {
            frame_timestamps,
            measurements,
            tolerance,
            prior_config: PriorConfig::default(),
        }
    }

    pub fn with_prior_config(mut self, prior_config: PriorConfig) -> Self {
        self.prior_config = prior_config;
        self
    }

    pub fn timestamp_for_frame(&self, frame: &Frame) -> Option<Timestamp> {
        self.frame_timestamps.timestamp_for_frame(frame)
    }

    pub fn timestamp_for_frame_id(&self, frame_id: FrameId) -> Option<Timestamp> {
        self.frame_timestamps.timestamp_for_frame_id(frame_id)
    }

    pub fn nearest_measurement_for_frame(&self, frame: &Frame) -> Option<&T> {
        self.measurements
            .nearest_for_frame(frame, &self.frame_timestamps, self.tolerance)
    }

    pub fn nearest_measurement_for_frame_id(&self, frame_id: FrameId) -> Option<&T> {
        self.measurements
            .nearest_for_frame_id(frame_id, &self.frame_timestamps, self.tolerance)
    }

    pub fn frame_count(&self) -> usize {
        self.frame_timestamps.len()
    }

    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    pub fn sync_summary(&self) -> FramePriorSyncSummary {
        let matched_frame_count = self
            .frame_timestamps
            .iter()
            .filter(|(frame_id, _)| self.nearest_measurement_for_frame_id(*frame_id).is_some())
            .count();
        let frame_count = self.frame_count();

        FramePriorSyncSummary {
            frame_count,
            measurement_count: self.measurement_count(),
            matched_frame_count,
            missing_measurement_count: frame_count.saturating_sub(matched_frame_count),
        }
    }

    pub fn evaluate_sync(
        &self,
        config: FramePriorSyncEvaluationConfig,
    ) -> FramePriorSyncEvaluationResult {
        self.sync_summary().evaluate(config)
    }
}

impl<T> FramePriorSource<T>
where
    T: LocalizationPriorProvider,
{
    pub fn localization_prior_for_frame(&self, frame: &Frame) -> Option<LocalizationPrior> {
        self.measurements.nearest_localization_prior_for_frame(
            frame,
            &self.frame_timestamps,
            self.tolerance,
            &self.prior_config,
        )
    }

    pub fn localization_prior_for_frame_id(&self, frame_id: FrameId) -> Option<LocalizationPrior> {
        self.measurements.nearest_localization_prior_for_frame_id(
            frame_id,
            &self.frame_timestamps,
            self.tolerance,
            &self.prior_config,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GnssMeasurement {
    pub timestamp: Timestamp,
    pub position_world: Point3<f64>,
    pub horizontal_accuracy: Option<f64>,
    pub vertical_accuracy: Option<f64>,
    pub position_covariance: Option<PositionCovariance>,
}

impl GnssMeasurement {
    pub fn new(timestamp: Timestamp, position_world: Point3<f64>) -> Self {
        Self {
            timestamp,
            position_world,
            horizontal_accuracy: None,
            vertical_accuracy: None,
            position_covariance: None,
        }
    }

    pub fn with_accuracy(
        mut self,
        horizontal_accuracy: Option<f64>,
        vertical_accuracy: Option<f64>,
    ) -> Self {
        self.horizontal_accuracy = horizontal_accuracy;
        self.vertical_accuracy = vertical_accuracy;
        self
    }

    pub fn with_position_covariance(mut self, position_covariance: PositionCovariance) -> Self {
        self.position_covariance = Some(position_covariance);
        self
    }

    pub fn search_radius(&self, config: &PriorConfig) -> f64 {
        let accuracy_radius = [self.horizontal_accuracy, self.vertical_accuracy]
            .into_iter()
            .flatten()
            .fold(None, |max_value: Option<f64>, value| {
                Some(max_value.map_or(value, |current| current.max(value)))
            })
            .or_else(|| {
                self.position_covariance
                    .as_ref()
                    .and_then(PositionCovariance::max_standard_deviation)
            })
            .map(|sigma| sigma * config.confidence_multiplier)
            .unwrap_or(config.default_radius);
        accuracy_radius.max(config.min_radius)
    }
}

impl TimedMeasurement for GnssMeasurement {
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

impl LocalizationPriorProvider for GnssMeasurement {
    fn localization_prior(&self, config: &PriorConfig) -> Option<LocalizationPrior> {
        Some(LocalizationPrior::from_position(
            self.position_world,
            self.search_radius(config),
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PosePriorMeasurement {
    pub timestamp: Timestamp,
    pub pose: Pose,
    pub translation_sigma: Option<f64>,
    pub pose_covariance: Option<PoseCovariance>,
}

impl PosePriorMeasurement {
    pub fn new(timestamp: Timestamp, pose: Pose) -> Self {
        Self {
            timestamp,
            pose,
            translation_sigma: None,
            pose_covariance: None,
        }
    }

    pub fn with_translation_sigma(mut self, translation_sigma: f64) -> Self {
        self.translation_sigma = Some(translation_sigma);
        self
    }

    pub fn with_pose_covariance(mut self, pose_covariance: PoseCovariance) -> Self {
        self.pose_covariance = Some(pose_covariance);
        self
    }

    pub fn search_radius(&self, config: &PriorConfig) -> f64 {
        self.translation_sigma
            .or_else(|| {
                self.pose_covariance
                    .as_ref()
                    .and_then(PoseCovariance::max_translation_standard_deviation)
            })
            .map(|sigma| sigma * config.confidence_multiplier)
            .unwrap_or(config.default_radius)
            .max(config.min_radius)
    }
}

impl TimedMeasurement for PosePriorMeasurement {
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

impl LocalizationPriorProvider for PosePriorMeasurement {
    fn localization_prior(&self, config: &PriorConfig) -> Option<LocalizationPrior> {
        Some(LocalizationPrior::from_pose(
            self.pose.clone(),
            self.search_radius(config),
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImuMeasurement {
    pub timestamp: Timestamp,
    pub angular_velocity: Vector3<f64>,
    pub linear_acceleration: Vector3<f64>,
    pub orientation: Option<UnitQuaternion<f64>>,
}

impl ImuMeasurement {
    pub fn new(
        timestamp: Timestamp,
        angular_velocity: Vector3<f64>,
        linear_acceleration: Vector3<f64>,
    ) -> Self {
        Self {
            timestamp,
            angular_velocity,
            linear_acceleration,
            orientation: None,
        }
    }

    pub fn with_orientation(mut self, orientation: UnitQuaternion<f64>) -> Self {
        self.orientation = Some(orientation);
        self
    }
}

impl TimedMeasurement for ImuMeasurement {
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

fn timestamp_distance(left: Timestamp, right: Timestamp) -> Option<TimeDelta> {
    if left >= right {
        left.duration_since(right)
    } else {
        right.duration_since(left)
    }
}

fn max_diagonal_standard_deviation<I>(variances: I) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    variances
        .into_iter()
        .filter_map(standard_deviation_from_variance)
        .max_by(f64::total_cmp)
}

fn standard_deviation_from_variance(variance: f64) -> Option<f64> {
    if variance.is_finite() && variance >= 0.0 {
        Some(variance.sqrt())
    } else {
        None
    }
}
