#![forbid(unsafe_code)]
//! Lightweight localization-based tracking scaffold.
//!
//! This crate does not implement full SLAM. It keeps temporal state around
//! repeated localization calls, exposes motion-model pose priors, tracks
//! failure/lost/relocalization events, and leaves keyframe management, map
//! updates, loop closure, and bundle adjustment to future layers.

use nalgebra::Point3;
use visloc_core::geometry::Pose;
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
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseTrajectory {
    samples: Vec<TrajectorySample>,
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

    pub fn camera_centers_world(&self) -> Vec<Point3<f64>> {
        self.samples
            .iter()
            .map(TrajectorySample::camera_center_world)
            .collect()
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
