#![forbid(unsafe_code)]
//! Local mapping scaffolding.
//!
//! This crate intentionally does not mutate maps yet. It starts v0.3 by
//! defining keyframe-selection interfaces that can consume tracking results and
//! later feed staged map updates, triangulation, and local refinement.

use nalgebra::{DMatrix, Point2, Point3};
use visloc_core::geometry::Pose;
use visloc_core::types::{
    CameraId, FrameId, Keyframe, Landmark, LandmarkId, Observation, VisualMap,
};
use visloc_tracking::{TrackingEvent, TrackingResult};

mod stereo_replenish;
pub use stereo_replenish::{
    build_stereo_metric_points, build_stereo_replenish_candidates, StereoReplenishConfig,
};

pub trait KeyframePolicy {
    fn evaluate(&mut self, result: &TrackingResult) -> KeyframeDecision;

    fn reset(&mut self);
}

#[derive(Debug, Clone, PartialEq)]
pub struct KeyframePolicyConfig {
    pub min_frame_id_gap: u64,
    pub min_translation: f64,
    pub select_relocalized_frames: bool,
    pub tracked_landmark_keyframe_ratio: Option<f64>,
    pub min_tracked_landmarks_for_quality_keyframe: usize,
    /// When `Some(n)`, a frame that would otherwise be promoted to a
    /// keyframe (i.e. it already cleared the frame-id-gap and translation
    /// gates) is instead rejected with
    /// [`KeyframeDecisionReason::InsufficientTrackingQuality`] if its PnP
    /// inlier count is below `n`. Guards against promoting a keyframe from
    /// a marginal localization (e.g. a handful of inliers surviving a
    /// tracking failure) that would poison the covisibility graph and
    /// local map for subsequent frames. `None` (the default) preserves
    /// legacy always-promote behaviour.
    pub min_inliers: Option<usize>,
    /// Companion to `min_inliers`: when `Some(r)`, also requires the PnP
    /// inlier ratio to be at least `r`. Either configured threshold not
    /// being met rejects the promotion. `None` (the default) preserves
    /// legacy always-promote behaviour.
    pub min_inlier_ratio: Option<f64>,
}

impl Default for KeyframePolicyConfig {
    fn default() -> Self {
        Self {
            min_frame_id_gap: 5,
            min_translation: 1.0,
            select_relocalized_frames: true,
            tracked_landmark_keyframe_ratio: None,
            min_tracked_landmarks_for_quality_keyframe: 20,
            min_inliers: None,
            min_inlier_ratio: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KeyframeDecision {
    pub frame_id: FrameId,
    pub selected: bool,
    pub reason: KeyframeDecisionReason,
    pub last_keyframe_frame_id: Option<FrameId>,
    pub selected_keyframe_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KeyframeDecisionReason {
    NotLocalized,
    MissingPose,
    FirstSuccessfulFrame,
    Relocalized,
    TrackedLandmarkDrop {
        frame_id_gap: u64,
        tracked_landmarks: usize,
        last_keyframe_tracked_landmarks: usize,
        min_tracked_landmark_ratio: f64,
    },
    ThresholdsMet {
        frame_id_gap: u64,
        translation: f64,
    },
    FrameIdGapTooSmall {
        frame_id_gap: u64,
        min_frame_id_gap: u64,
    },
    TranslationTooSmall {
        translation: f64,
        min_translation: f64,
    },
    InsufficientTrackingQuality {
        inlier_count: usize,
        inlier_ratio: f64,
        min_inliers: Option<usize>,
        min_inlier_ratio: Option<f64>,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SimpleKeyframePolicy {
    config: KeyframePolicyConfig,
    last_keyframe_frame_id: Option<FrameId>,
    last_keyframe_pose: Option<Pose>,
    last_keyframe_tracked_landmark_count: Option<usize>,
    selected_keyframe_count: usize,
}

impl SimpleKeyframePolicy {
    pub fn new(config: KeyframePolicyConfig) -> Self {
        Self {
            config,
            last_keyframe_frame_id: None,
            last_keyframe_pose: None,
            last_keyframe_tracked_landmark_count: None,
            selected_keyframe_count: 0,
        }
    }

    pub fn config(&self) -> &KeyframePolicyConfig {
        &self.config
    }

    pub fn last_keyframe_frame_id(&self) -> Option<FrameId> {
        self.last_keyframe_frame_id
    }

    pub fn last_keyframe_pose(&self) -> Option<&Pose> {
        self.last_keyframe_pose.as_ref()
    }

    pub fn selected_keyframe_count(&self) -> usize {
        self.selected_keyframe_count
    }

    pub fn last_keyframe_tracked_landmark_count(&self) -> Option<usize> {
        self.last_keyframe_tracked_landmark_count
    }

    fn selected(
        &mut self,
        result: &TrackingResult,
        pose: &Pose,
        reason: KeyframeDecisionReason,
    ) -> KeyframeDecision {
        self.last_keyframe_frame_id = Some(result.frame_id);
        self.last_keyframe_pose = Some(pose.clone());
        self.last_keyframe_tracked_landmark_count = Some(result.localization.inlier_count);
        self.selected_keyframe_count += 1;
        self.decision(result.frame_id, true, reason)
    }

    fn rejected(
        &self,
        result: &TrackingResult,
        reason: KeyframeDecisionReason,
    ) -> KeyframeDecision {
        self.decision(result.frame_id, false, reason)
    }

    fn decision(
        &self,
        frame_id: FrameId,
        selected: bool,
        reason: KeyframeDecisionReason,
    ) -> KeyframeDecision {
        KeyframeDecision {
            frame_id,
            selected,
            reason,
            last_keyframe_frame_id: self.last_keyframe_frame_id,
            selected_keyframe_count: self.selected_keyframe_count,
        }
    }

    fn tracked_landmark_drop_reason(
        &self,
        result: &TrackingResult,
        frame_id_gap: u64,
    ) -> Option<KeyframeDecisionReason> {
        let min_tracked_landmark_ratio = self.config.tracked_landmark_keyframe_ratio?;
        if !(0.0..=1.0).contains(&min_tracked_landmark_ratio) {
            return None;
        }

        let last_keyframe_tracked_landmarks = self.last_keyframe_tracked_landmark_count?;
        if last_keyframe_tracked_landmarks < self.config.min_tracked_landmarks_for_quality_keyframe
        {
            return None;
        }

        let tracked_landmarks = result.localization.inlier_count;
        if tracked_landmarks < self.config.min_tracked_landmarks_for_quality_keyframe {
            return None;
        }

        let threshold = last_keyframe_tracked_landmarks as f64 * min_tracked_landmark_ratio;
        if tracked_landmarks < last_keyframe_tracked_landmarks
            && (tracked_landmarks as f64) <= threshold
        {
            return Some(KeyframeDecisionReason::TrackedLandmarkDrop {
                frame_id_gap,
                tracked_landmarks,
                last_keyframe_tracked_landmarks,
                min_tracked_landmark_ratio,
            });
        }

        None
    }

    /// When a frame has cleared the frame-id-gap and translation gates and
    /// would otherwise be promoted, check it against the optional
    /// `min_inliers` / `min_inlier_ratio` tracking-quality floors. Returns
    /// `Some` (rejecting the promotion) when either configured threshold is
    /// not met; `None` (both thresholds absent, or both satisfied) leaves
    /// the promotion to proceed as `ThresholdsMet`.
    fn insufficient_tracking_quality_reason(
        &self,
        result: &TrackingResult,
    ) -> Option<KeyframeDecisionReason> {
        let min_inliers = self.config.min_inliers;
        let min_inlier_ratio = self.config.min_inlier_ratio;
        let inlier_count = result.localization.inlier_count;
        let inlier_ratio = result.localization.inlier_ratio;

        let fails_min_inliers = min_inliers.is_some_and(|min| inlier_count < min);
        let fails_min_inlier_ratio = min_inlier_ratio.is_some_and(|min| inlier_ratio < min);
        if fails_min_inliers || fails_min_inlier_ratio {
            Some(KeyframeDecisionReason::InsufficientTrackingQuality {
                inlier_count,
                inlier_ratio,
                min_inliers,
                min_inlier_ratio,
            })
        } else {
            None
        }
    }
}

impl KeyframePolicy for SimpleKeyframePolicy {
    fn evaluate(&mut self, result: &TrackingResult) -> KeyframeDecision {
        if !result.localization.success {
            return self.rejected(result, KeyframeDecisionReason::NotLocalized);
        }

        let Some(pose) = result.localization.pose.as_ref() else {
            return self.rejected(result, KeyframeDecisionReason::MissingPose);
        };

        let Some(last_frame_id) = self.last_keyframe_frame_id else {
            return self.selected(result, pose, KeyframeDecisionReason::FirstSuccessfulFrame);
        };

        if result.event == TrackingEvent::Relocalized && self.config.select_relocalized_frames {
            return self.selected(result, pose, KeyframeDecisionReason::Relocalized);
        }

        let frame_id_gap = result.frame_id.saturating_sub(last_frame_id);
        if frame_id_gap < self.config.min_frame_id_gap {
            return self.rejected(
                result,
                KeyframeDecisionReason::FrameIdGapTooSmall {
                    frame_id_gap,
                    min_frame_id_gap: self.config.min_frame_id_gap,
                },
            );
        }

        if let Some(reason) = self.tracked_landmark_drop_reason(result, frame_id_gap) {
            return self.selected(result, pose, reason);
        }

        let translation = self
            .last_keyframe_pose
            .as_ref()
            .map(|last_pose| (pose.camera_center_world() - last_pose.camera_center_world()).norm())
            .unwrap_or(f64::INFINITY);
        if translation < self.config.min_translation {
            return self.rejected(
                result,
                KeyframeDecisionReason::TranslationTooSmall {
                    translation,
                    min_translation: self.config.min_translation,
                },
            );
        }

        if let Some(reason) = self.insufficient_tracking_quality_reason(result) {
            return self.rejected(result, reason);
        }

        self.selected(
            result,
            pose,
            KeyframeDecisionReason::ThresholdsMet {
                frame_id_gap,
                translation,
            },
        )
    }

    fn reset(&mut self) {
        self.last_keyframe_frame_id = None;
        self.last_keyframe_pose = None;
        self.last_keyframe_tracked_landmark_count = None;
        self.selected_keyframe_count = 0;
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StagedMapUpdate {
    pub keyframes: Vec<Keyframe>,
    pub landmarks: Vec<Landmark>,
    pub observations: Vec<Observation>,
}

impl StagedMapUpdate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stage_keyframe(&mut self, keyframe: Keyframe) {
        self.keyframes.push(keyframe);
    }

    pub fn stage_landmark(&mut self, landmark: Landmark) {
        self.landmarks.push(landmark);
    }

    pub fn stage_observation(&mut self, observation: Observation) {
        self.observations.push(observation);
    }

    pub fn with_keyframe(mut self, keyframe: Keyframe) -> Self {
        self.stage_keyframe(keyframe);
        self
    }

    pub fn with_landmark(mut self, landmark: Landmark) -> Self {
        self.stage_landmark(landmark);
        self
    }

    pub fn with_observation(mut self, observation: Observation) -> Self {
        self.stage_observation(observation);
        self
    }

    pub fn stage_triangulated_landmark(&mut self, triangulated: TriangulatedLandmark) {
        let mut landmark = triangulated.landmark;
        let observations = std::mem::take(&mut landmark.observations);
        self.stage_landmark(landmark);
        for observation in observations {
            self.stage_observation(observation);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty() && self.landmarks.is_empty() && self.observations.is_empty()
    }

    pub fn validate_against(&self, map: &VisualMap) -> MapUpdateValidationReport {
        let mut report = MapUpdateValidationReport::default();
        self.validate_keyframes(map, &mut report);
        self.validate_landmarks(map, &mut report);
        self.validate_observations(map, &mut report);
        report
    }

    pub fn apply_to(
        self,
        map: &mut VisualMap,
    ) -> Result<AppliedMapUpdate, MapUpdateValidationReport> {
        let report = self.validate_against(map);
        if !report.is_valid() {
            return Err(report);
        }

        let applied = AppliedMapUpdate {
            keyframe_count: self.keyframes.len(),
            landmark_count: self.landmarks.len(),
            observation_count: self.observations.len(),
        };

        for keyframe in self.keyframes {
            map.keyframes.insert(keyframe.frame.id, keyframe);
        }
        for landmark in self.landmarks {
            map.landmarks.insert(landmark.id, landmark);
        }
        for observation in self.observations {
            if let Some(keyframe) = map.keyframes.get_mut(&observation.frame_id) {
                keyframe.observations.push(observation.clone());
            }
            if let Some(landmark) = map.landmarks.get_mut(&observation.landmark_id) {
                landmark.observations.push(observation);
            }
        }

        Ok(applied)
    }

    fn validate_keyframes(&self, map: &VisualMap, report: &mut MapUpdateValidationReport) {
        let mut staged_frame_ids = Vec::new();
        for keyframe in &self.keyframes {
            let frame_id = keyframe.frame.id;
            if map.keyframes.contains_key(&frame_id) {
                report.push(MapUpdateValidationIssue::KeyframeAlreadyExists { frame_id });
            }
            if staged_frame_ids.contains(&frame_id) {
                report.push(MapUpdateValidationIssue::DuplicateStagedKeyframe { frame_id });
            }
            staged_frame_ids.push(frame_id);

            if !map.cameras.contains_key(&keyframe.frame.camera_id) {
                report.push(MapUpdateValidationIssue::MissingCameraForKeyframe {
                    frame_id,
                    camera_id: keyframe.frame.camera_id,
                });
            }
        }
    }

    fn validate_landmarks(&self, map: &VisualMap, report: &mut MapUpdateValidationReport) {
        let mut staged_landmark_ids = Vec::new();
        for landmark in &self.landmarks {
            if map.landmarks.contains_key(&landmark.id) {
                report.push(MapUpdateValidationIssue::LandmarkAlreadyExists {
                    landmark_id: landmark.id,
                });
            }
            if staged_landmark_ids.contains(&landmark.id) {
                report.push(MapUpdateValidationIssue::DuplicateStagedLandmark {
                    landmark_id: landmark.id,
                });
            }
            staged_landmark_ids.push(landmark.id);
        }
    }

    fn validate_observations(&self, map: &VisualMap, report: &mut MapUpdateValidationReport) {
        let mut staged_observations = Vec::new();
        for observation in &self.observations {
            if staged_observations.contains(&observation_key(observation)) {
                report.push(MapUpdateValidationIssue::DuplicateStagedObservation {
                    frame_id: observation.frame_id,
                    landmark_id: observation.landmark_id,
                    keypoint_index: observation.keypoint_index,
                });
            }
            staged_observations.push(observation_key(observation));

            let keyframe = self
                .keyframes
                .iter()
                .find(|keyframe| keyframe.frame.id == observation.frame_id)
                .or_else(|| map.keyframes.get(&observation.frame_id));
            if keyframe.is_none() {
                report.push(MapUpdateValidationIssue::ObservationMissingKeyframe {
                    frame_id: observation.frame_id,
                    landmark_id: observation.landmark_id,
                    keypoint_index: observation.keypoint_index,
                });
            }

            let landmark_exists = self
                .landmarks
                .iter()
                .any(|landmark| landmark.id == observation.landmark_id)
                || map.landmarks.contains_key(&observation.landmark_id);
            if !landmark_exists {
                report.push(MapUpdateValidationIssue::ObservationMissingLandmark {
                    frame_id: observation.frame_id,
                    landmark_id: observation.landmark_id,
                    keypoint_index: observation.keypoint_index,
                });
            }

            let Some(keyframe) = keyframe else {
                continue;
            };
            if observation.keypoint_index >= keyframe.frame.keypoints.len() {
                report.push(MapUpdateValidationIssue::ObservationKeypointOutOfBounds {
                    frame_id: observation.frame_id,
                    landmark_id: observation.landmark_id,
                    keypoint_index: observation.keypoint_index,
                    keypoint_count: keyframe.frame.keypoints.len(),
                });
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AppliedMapUpdate {
    pub keyframe_count: usize,
    pub landmark_count: usize,
    pub observation_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MapUpdateValidationReport {
    pub issues: Vec<MapUpdateValidationIssue>,
}

impl MapUpdateValidationReport {
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }

    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }

    pub fn push(&mut self, issue: MapUpdateValidationIssue) {
        self.issues.push(issue);
    }

    pub fn into_result(self) -> Result<(), Self> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(self)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapUpdateValidationIssue {
    KeyframeAlreadyExists {
        frame_id: FrameId,
    },
    DuplicateStagedKeyframe {
        frame_id: FrameId,
    },
    MissingCameraForKeyframe {
        frame_id: FrameId,
        camera_id: CameraId,
    },
    LandmarkAlreadyExists {
        landmark_id: LandmarkId,
    },
    DuplicateStagedLandmark {
        landmark_id: LandmarkId,
    },
    ObservationMissingKeyframe {
        frame_id: FrameId,
        landmark_id: LandmarkId,
        keypoint_index: usize,
    },
    ObservationMissingLandmark {
        frame_id: FrameId,
        landmark_id: LandmarkId,
        keypoint_index: usize,
    },
    ObservationKeypointOutOfBounds {
        frame_id: FrameId,
        landmark_id: LandmarkId,
        keypoint_index: usize,
        keypoint_count: usize,
    },
    DuplicateStagedObservation {
        frame_id: FrameId,
        landmark_id: LandmarkId,
        keypoint_index: usize,
    },
}

fn observation_key(observation: &Observation) -> (FrameId, LandmarkId, usize) {
    (
        observation.frame_id,
        observation.landmark_id,
        observation.keypoint_index,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalMapWindowConfig {
    pub max_keyframes: usize,
}

impl Default for LocalMapWindowConfig {
    fn default() -> Self {
        Self { max_keyframes: 5 }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LocalMapWindow {
    pub anchor_frame_id: Option<FrameId>,
    pub keyframe_ids: Vec<FrameId>,
    pub landmark_ids: Vec<LandmarkId>,
    pub observation_count: usize,
}

impl LocalMapWindow {
    pub fn from_anchor(
        map: &VisualMap,
        anchor_frame_id: FrameId,
        config: &LocalMapWindowConfig,
    ) -> Self {
        let max_keyframes = config.max_keyframes.max(1);
        let mut keyframe_ids = map
            .keyframes
            .keys()
            .copied()
            .filter(|frame_id| *frame_id <= anchor_frame_id)
            .collect::<Vec<_>>();
        keyframe_ids.sort_unstable();
        if keyframe_ids.len() > max_keyframes {
            keyframe_ids = keyframe_ids[keyframe_ids.len() - max_keyframes..].to_vec();
        }

        Self::from_keyframe_ids(map, Some(anchor_frame_id), keyframe_ids)
    }

    pub fn from_recent(map: &VisualMap, config: &LocalMapWindowConfig) -> Self {
        let Some(anchor_frame_id) = map.keyframes.keys().max().copied() else {
            return Self::default();
        };
        Self::from_anchor(map, anchor_frame_id, config)
    }

    pub fn from_keyframe_ids(
        map: &VisualMap,
        anchor_frame_id: Option<FrameId>,
        mut keyframe_ids: Vec<FrameId>,
    ) -> Self {
        keyframe_ids.sort_unstable();
        keyframe_ids.dedup();
        keyframe_ids.retain(|frame_id| map.keyframes.contains_key(frame_id));

        let mut landmark_ids = Vec::new();
        let mut observation_count = 0;
        for keyframe_id in &keyframe_ids {
            let Some(keyframe) = map.keyframes.get(keyframe_id) else {
                continue;
            };
            observation_count += keyframe.observations.len();
            for observation in &keyframe.observations {
                if map.landmarks.contains_key(&observation.landmark_id) {
                    landmark_ids.push(observation.landmark_id);
                }
            }
        }
        landmark_ids.sort_unstable();
        landmark_ids.dedup();

        Self {
            anchor_frame_id,
            keyframe_ids,
            landmark_ids,
            observation_count,
        }
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframe_ids.len()
    }

    pub fn landmark_count(&self) -> usize {
        self.landmark_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keyframe_ids.is_empty()
    }
}

pub type LandmarkCandidateId = u64;

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkCandidate {
    pub id: LandmarkCandidateId,
    pub observations: Vec<LandmarkCandidateObservation>,
    pub descriptor: Option<Vec<f32>>,
}

impl LandmarkCandidate {
    pub fn new(id: LandmarkCandidateId) -> Self {
        Self {
            id,
            observations: Vec::new(),
            descriptor: None,
        }
    }

    pub fn with_observation(mut self, observation: LandmarkCandidateObservation) -> Self {
        self.observations.push(observation);
        self
    }

    pub fn with_descriptor(mut self, descriptor: Vec<f32>) -> Self {
        self.descriptor = Some(descriptor);
        self
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    pub fn is_triangulatable(&self, min_observations: usize) -> bool {
        self.observations.len() >= min_observations.max(2)
    }

    pub fn validate_against(
        &self,
        map: &VisualMap,
        window: Option<&LocalMapWindow>,
        config: &LandmarkCandidateValidationConfig,
    ) -> LandmarkCandidateValidationReport {
        let mut report = LandmarkCandidateValidationReport::default();
        let min_observations = config.min_observations.max(2);
        if self.observations.len() < min_observations {
            report.push(LandmarkCandidateValidationIssue::TooFewObservations {
                observation_count: self.observations.len(),
                min_observations,
            });
        }

        let mut seen = Vec::new();
        for observation in &self.observations {
            let key = candidate_observation_key(observation);
            if seen.contains(&key) {
                report.push(LandmarkCandidateValidationIssue::DuplicateObservation {
                    frame_id: observation.frame_id,
                    keypoint_index: observation.keypoint_index,
                });
            }
            seen.push(key);

            let Some(keyframe) = map.keyframes.get(&observation.frame_id) else {
                report.push(LandmarkCandidateValidationIssue::MissingKeyframe {
                    frame_id: observation.frame_id,
                });
                continue;
            };

            if observation.keypoint_index >= keyframe.frame.keypoints.len() {
                report.push(LandmarkCandidateValidationIssue::KeypointOutOfBounds {
                    frame_id: observation.frame_id,
                    keypoint_index: observation.keypoint_index,
                    keypoint_count: keyframe.frame.keypoints.len(),
                });
            }

            if let Some(window) = window {
                if !window.keyframe_ids.contains(&observation.frame_id) {
                    report.push(
                        LandmarkCandidateValidationIssue::ObservationOutsideLocalWindow {
                            frame_id: observation.frame_id,
                        },
                    );
                }
            }
        }

        report
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkCandidateObservation {
    pub frame_id: FrameId,
    pub keypoint_index: usize,
    pub xy: Point2<f64>,
}

impl LandmarkCandidateObservation {
    pub fn new(frame_id: FrameId, keypoint_index: usize, xy: Point2<f64>) -> Self {
        Self {
            frame_id,
            keypoint_index,
            xy,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandmarkCandidateValidationConfig {
    pub min_observations: usize,
}

impl Default for LandmarkCandidateValidationConfig {
    fn default() -> Self {
        Self {
            min_observations: 2,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LandmarkCandidateValidationReport {
    pub issues: Vec<LandmarkCandidateValidationIssue>,
}

impl LandmarkCandidateValidationReport {
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }

    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }

    pub fn push(&mut self, issue: LandmarkCandidateValidationIssue) {
        self.issues.push(issue);
    }

    pub fn into_result(self) -> Result<(), Self> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(self)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LandmarkCandidateValidationIssue {
    TooFewObservations {
        observation_count: usize,
        min_observations: usize,
    },
    MissingKeyframe {
        frame_id: FrameId,
    },
    KeypointOutOfBounds {
        frame_id: FrameId,
        keypoint_index: usize,
        keypoint_count: usize,
    },
    DuplicateObservation {
        frame_id: FrameId,
        keypoint_index: usize,
    },
    ObservationOutsideLocalWindow {
        frame_id: FrameId,
    },
}

fn candidate_observation_key(observation: &LandmarkCandidateObservation) -> (FrameId, usize) {
    (observation.frame_id, observation.keypoint_index)
}

pub trait Triangulator {
    fn triangulate(
        &self,
        candidate: &LandmarkCandidate,
        map: &VisualMap,
    ) -> Result<TriangulatedLandmark, TriangulationFailureReason>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearTriangulator {
    pub config: TriangulationConfig,
}

impl Default for LinearTriangulator {
    fn default() -> Self {
        Self::new(TriangulationConfig::default())
    }
}

impl LinearTriangulator {
    pub fn new(config: TriangulationConfig) -> Self {
        Self { config }
    }
}

impl Triangulator for LinearTriangulator {
    fn triangulate(
        &self,
        candidate: &LandmarkCandidate,
        map: &VisualMap,
    ) -> Result<TriangulatedLandmark, TriangulationFailureReason> {
        let validation_config = LandmarkCandidateValidationConfig {
            min_observations: self.config.min_observations,
        };
        let validation = candidate.validate_against(map, None, &validation_config);
        if !validation.is_valid() {
            return Err(TriangulationFailureReason::CandidateValidationFailed(
                validation,
            ));
        }

        let mut a = DMatrix::<f64>::zeros(candidate.observations.len() * 2, 4);
        for (observation_index, observation) in candidate.observations.iter().enumerate() {
            let keyframe = map
                .keyframes
                .get(&observation.frame_id)
                .expect("candidate validation ensures keyframe exists");
            let camera = map.cameras.get(&keyframe.frame.camera_id).ok_or(
                TriangulationFailureReason::MissingCamera {
                    frame_id: keyframe.frame.id,
                    camera_id: keyframe.frame.camera_id,
                },
            )?;
            let pose =
                keyframe
                    .frame
                    .pose
                    .as_ref()
                    .ok_or(TriangulationFailureReason::MissingPose {
                        frame_id: keyframe.frame.id,
                    })?;
            let normalized = camera.normalize_pixel(&observation.xy).ok_or(
                TriangulationFailureReason::UnsupportedCameraModel {
                    frame_id: keyframe.frame.id,
                    camera_id: keyframe.frame.camera_id,
                },
            )?;
            let matrix = pose.matrix();
            let row0 = matrix.row(0);
            let row1 = matrix.row(1);
            let row2 = matrix.row(2);
            let row_x = observation_index * 2;
            let row_y = row_x + 1;
            for column in 0..4 {
                a[(row_x, column)] = normalized.x * row2[column] - row0[column];
                a[(row_y, column)] = normalized.y * row2[column] - row1[column];
            }
        }

        let svd = a.svd(false, true);
        let Some(v_t) = svd.v_t else {
            return Err(TriangulationFailureReason::DegenerateGeometry);
        };
        let homogeneous = v_t.row(v_t.nrows() - 1);
        let w = homogeneous[3];
        if w.abs() <= self.config.min_homogeneous_scale {
            return Err(TriangulationFailureReason::DegenerateGeometry);
        }
        let position = Point3::new(homogeneous[0] / w, homogeneous[1] / w, homogeneous[2] / w);

        let mut reprojection_errors = Vec::new();
        for observation in &candidate.observations {
            let keyframe = map
                .keyframes
                .get(&observation.frame_id)
                .expect("candidate validation ensures keyframe exists");
            let camera = map
                .cameras
                .get(&keyframe.frame.camera_id)
                .expect("camera was checked before solving");
            let pose = keyframe
                .frame
                .pose
                .as_ref()
                .expect("pose was checked before solving");
            let point_camera = pose.transform_world_point(&position);
            if self.config.require_positive_depth && point_camera.z <= 0.0 {
                return Err(TriangulationFailureReason::PointBehindCamera {
                    frame_id: keyframe.frame.id,
                });
            }
            let projected = camera.project(&point_camera).ok_or(
                TriangulationFailureReason::ProjectionFailed {
                    frame_id: keyframe.frame.id,
                    camera_id: keyframe.frame.camera_id,
                },
            )?;
            reprojection_errors.push((projected - observation.xy).norm());
        }

        let mean_reprojection_error = mean(&reprojection_errors);
        let max_reprojection_error = reprojection_errors.iter().copied().fold(0.0_f64, f64::max);
        if let Some(max_mean_reprojection_error) = self.config.max_mean_reprojection_error {
            if mean_reprojection_error > max_mean_reprojection_error {
                return Err(TriangulationFailureReason::ReprojectionErrorTooHigh {
                    mean_reprojection_error,
                    max_mean_reprojection_error,
                });
            }
        }

        let mut landmark = Landmark::new(candidate.id, position);
        landmark.descriptor = candidate.descriptor.clone();
        landmark.observations = candidate
            .observations
            .iter()
            .map(|observation| Observation {
                frame_id: observation.frame_id,
                landmark_id: candidate.id,
                keypoint_index: observation.keypoint_index,
                xy: observation.xy,
            })
            .collect();

        Ok(TriangulatedLandmark {
            landmark,
            observation_count: candidate.observations.len(),
            mean_reprojection_error,
            max_reprojection_error,
            reprojection_errors,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TriangulationConfig {
    pub min_observations: usize,
    pub min_homogeneous_scale: f64,
    pub require_positive_depth: bool,
    pub max_mean_reprojection_error: Option<f64>,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_observations: 2,
            min_homogeneous_scale: 1.0e-12,
            require_positive_depth: true,
            max_mean_reprojection_error: Some(2.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TriangulatedLandmark {
    pub landmark: Landmark,
    pub observation_count: usize,
    pub mean_reprojection_error: f64,
    pub max_reprojection_error: f64,
    pub reprojection_errors: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TriangulationFailureReason {
    CandidateValidationFailed(LandmarkCandidateValidationReport),
    MissingCamera {
        frame_id: FrameId,
        camera_id: CameraId,
    },
    UnsupportedCameraModel {
        frame_id: FrameId,
        camera_id: CameraId,
    },
    MissingPose {
        frame_id: FrameId,
    },
    DegenerateGeometry,
    PointBehindCamera {
        frame_id: FrameId,
    },
    ProjectionFailed {
        frame_id: FrameId,
        camera_id: CameraId,
    },
    ReprojectionErrorTooHigh {
        mean_reprojection_error: f64,
        max_mean_reprojection_error: f64,
    },
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

pub trait LocalRefiner {
    fn refine(
        &self,
        map: &VisualMap,
        local_window: &LocalMapWindow,
        staged_update: &mut StagedMapUpdate,
    ) -> LocalRefinementResult;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoopLocalRefiner;

impl LocalRefiner for NoopLocalRefiner {
    fn refine(
        &self,
        _map: &VisualMap,
        _local_window: &LocalMapWindow,
        _staged_update: &mut StagedMapUpdate,
    ) -> LocalRefinementResult {
        LocalRefinementResult {
            refined: false,
            reason: LocalRefinementReason::Noop,
            keyframe_count: 0,
            landmark_count: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRefinementResult {
    pub refined: bool,
    pub reason: LocalRefinementReason,
    pub keyframe_count: usize,
    pub landmark_count: usize,
}

impl LocalRefinementResult {
    pub fn skipped(reason: LocalRefinementReason) -> Self {
        Self {
            refined: false,
            reason,
            keyframe_count: 0,
            landmark_count: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalRefinementReason {
    Noop,
    NoSelectedKeyframe,
    Refined,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalMappingPipeline<
    K = SimpleKeyframePolicy,
    T = LinearTriangulator,
    R = NoopLocalRefiner,
> {
    pub keyframe_policy: K,
    pub triangulator: T,
    pub local_refiner: R,
    pub local_window_config: LocalMapWindowConfig,
    pub candidate_validation_config: LandmarkCandidateValidationConfig,
}

impl Default for LocalMappingPipeline<SimpleKeyframePolicy, LinearTriangulator, NoopLocalRefiner> {
    fn default() -> Self {
        Self {
            keyframe_policy: SimpleKeyframePolicy::default(),
            triangulator: LinearTriangulator::default(),
            local_refiner: NoopLocalRefiner,
            local_window_config: LocalMapWindowConfig::default(),
            candidate_validation_config: LandmarkCandidateValidationConfig::default(),
        }
    }
}

impl<K, T> LocalMappingPipeline<K, T, NoopLocalRefiner>
where
    K: KeyframePolicy,
    T: Triangulator,
{
    pub fn new(
        keyframe_policy: K,
        triangulator: T,
        local_window_config: LocalMapWindowConfig,
        candidate_validation_config: LandmarkCandidateValidationConfig,
    ) -> Self {
        Self {
            keyframe_policy,
            triangulator,
            local_refiner: NoopLocalRefiner,
            local_window_config,
            candidate_validation_config,
        }
    }
}

impl<K, T, R> LocalMappingPipeline<K, T, R>
where
    K: KeyframePolicy,
    T: Triangulator,
    R: LocalRefiner,
{
    pub fn with_refiner(
        keyframe_policy: K,
        triangulator: T,
        local_refiner: R,
        local_window_config: LocalMapWindowConfig,
        candidate_validation_config: LandmarkCandidateValidationConfig,
    ) -> Self {
        Self {
            keyframe_policy,
            triangulator,
            local_refiner,
            local_window_config,
            candidate_validation_config,
        }
    }

    pub fn reset(&mut self) {
        self.keyframe_policy.reset();
    }

    pub fn process_keyframe<I>(
        &mut self,
        map: &VisualMap,
        tracking_result: &TrackingResult,
        keyframe: Keyframe,
        candidates: I,
    ) -> LocalMappingResult
    where
        I: IntoIterator<Item = LandmarkCandidate>,
    {
        let keyframe_decision = self.keyframe_policy.evaluate(tracking_result);
        let mut staged_update = StagedMapUpdate::new();
        let mut triangulated_landmarks = Vec::new();
        let mut candidate_failures = Vec::new();

        if !keyframe_decision.selected {
            let local_window = LocalMapWindow::from_recent(map, &self.local_window_config);
            let staged_update_validation = staged_update.validate_against(map);
            let refinement =
                LocalRefinementResult::skipped(LocalRefinementReason::NoSelectedKeyframe);
            return LocalMappingResult {
                keyframe_decision,
                local_window,
                staged_update,
                triangulated_landmarks,
                candidate_failures,
                refinement,
                staged_update_validation,
            };
        }

        staged_update.stage_keyframe(keyframe.clone());

        let mut working_map = map.clone();
        working_map.keyframes.insert(keyframe.frame.id, keyframe);
        let local_window = LocalMapWindow::from_anchor(
            &working_map,
            tracking_result.frame_id,
            &self.local_window_config,
        );

        for candidate in candidates {
            let validation = candidate.validate_against(
                &working_map,
                Some(&local_window),
                &self.candidate_validation_config,
            );
            if !validation.is_valid() {
                candidate_failures.push(LandmarkCandidateMappingFailure {
                    candidate_id: candidate.id,
                    reason: LandmarkCandidateMappingFailureReason::CandidateValidationFailed(
                        validation,
                    ),
                });
                continue;
            }

            match self.triangulator.triangulate(&candidate, &working_map) {
                Ok(triangulated) => {
                    staged_update.stage_triangulated_landmark(triangulated.clone());
                    triangulated_landmarks.push(triangulated);
                }
                Err(reason) => {
                    candidate_failures.push(LandmarkCandidateMappingFailure {
                        candidate_id: candidate.id,
                        reason: LandmarkCandidateMappingFailureReason::TriangulationFailed(reason),
                    });
                }
            }
        }

        let refinement = self
            .local_refiner
            .refine(&working_map, &local_window, &mut staged_update);
        let staged_update_validation = staged_update.validate_against(map);
        LocalMappingResult {
            keyframe_decision,
            local_window,
            staged_update,
            triangulated_landmarks,
            candidate_failures,
            refinement,
            staged_update_validation,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalMappingResult {
    pub keyframe_decision: KeyframeDecision,
    pub local_window: LocalMapWindow,
    pub staged_update: StagedMapUpdate,
    pub triangulated_landmarks: Vec<TriangulatedLandmark>,
    pub candidate_failures: Vec<LandmarkCandidateMappingFailure>,
    pub refinement: LocalRefinementResult,
    pub staged_update_validation: MapUpdateValidationReport,
}

impl LocalMappingResult {
    pub fn is_ready_to_apply(&self) -> bool {
        self.keyframe_decision.selected
            && self.staged_update_validation.is_valid()
            && self.candidate_failures.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkCandidateMappingFailure {
    pub candidate_id: LandmarkCandidateId,
    pub reason: LandmarkCandidateMappingFailureReason,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LandmarkCandidateMappingFailureReason {
    CandidateValidationFailed(LandmarkCandidateValidationReport),
    TriangulationFailed(TriangulationFailureReason),
}
