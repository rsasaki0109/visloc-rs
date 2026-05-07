#![forbid(unsafe_code)]
//! Local mapping scaffolding.
//!
//! This crate intentionally does not mutate maps yet. It starts v0.3 by
//! defining keyframe-selection interfaces that can consume tracking results and
//! later feed staged map updates, triangulation, and local refinement.

use visloc_core::geometry::Pose;
use visloc_core::types::{
    CameraId, FrameId, Keyframe, Landmark, LandmarkId, Observation, VisualMap,
};
use visloc_tracking::{TrackingEvent, TrackingResult};

pub trait KeyframePolicy {
    fn evaluate(&mut self, result: &TrackingResult) -> KeyframeDecision;

    fn reset(&mut self);
}

#[derive(Debug, Clone, PartialEq)]
pub struct KeyframePolicyConfig {
    pub min_frame_id_gap: u64,
    pub min_translation: f64,
    pub select_relocalized_frames: bool,
}

impl Default for KeyframePolicyConfig {
    fn default() -> Self {
        Self {
            min_frame_id_gap: 5,
            min_translation: 1.0,
            select_relocalized_frames: true,
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
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SimpleKeyframePolicy {
    config: KeyframePolicyConfig,
    last_keyframe_frame_id: Option<FrameId>,
    last_keyframe_pose: Option<Pose>,
    selected_keyframe_count: usize,
}

impl SimpleKeyframePolicy {
    pub fn new(config: KeyframePolicyConfig) -> Self {
        Self {
            config,
            last_keyframe_frame_id: None,
            last_keyframe_pose: None,
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

    fn selected(
        &mut self,
        result: &TrackingResult,
        pose: &Pose,
        reason: KeyframeDecisionReason,
    ) -> KeyframeDecision {
        self.last_keyframe_frame_id = Some(result.frame_id);
        self.last_keyframe_pose = Some(pose.clone());
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
