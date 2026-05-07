#![forbid(unsafe_code)]
//! Minimal online SLAM orchestration.
//!
//! This crate wires tracking and local mapping together. It is not a full SLAM
//! system: loop closure, global pose graph optimization, dense mapping, and
//! production bundle adjustment remain outside this MVP layer.

use visloc_core::types::{Frame, Keyframe, Observation, VisualMap};
use visloc_localization::LocalizationPipeline;
use visloc_mapping::{
    AppliedMapUpdate, KeyframePolicy, LandmarkCandidate, LinearTriangulator, LocalMappingPipeline,
    LocalMappingResult, SimpleKeyframePolicy, Triangulator,
};
use visloc_tracking::{
    ConstantPoseMotionModel, FrameLocalizer, MotionModel, Tracker, TrackingConfig, TrackingResult,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnlineSlamConfig {
    pub apply_map_updates: bool,
}

impl Default for OnlineSlamConfig {
    fn default() -> Self {
        Self {
            apply_map_updates: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamPipeline<T, M> {
    pub map: VisualMap,
    pub tracker: T,
    pub mapper: M,
    pub config: OnlineSlamConfig,
}

impl Default
    for OnlineSlamPipeline<
        Tracker<LocalizationPipeline, ConstantPoseMotionModel>,
        LocalMappingPipeline<SimpleKeyframePolicy, LinearTriangulator>,
    >
{
    fn default() -> Self {
        Self {
            map: VisualMap::new(),
            tracker: Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            mapper: LocalMappingPipeline::default(),
            config: OnlineSlamConfig::default(),
        }
    }
}

impl<T, M> OnlineSlamPipeline<T, M> {
    pub fn new(map: VisualMap, tracker: T, mapper: M, config: OnlineSlamConfig) -> Self {
        Self {
            map,
            tracker,
            mapper,
            config,
        }
    }

    pub fn map(&self) -> &VisualMap {
        &self.map
    }

    pub fn map_mut(&mut self) -> &mut VisualMap {
        &mut self.map
    }
}

impl<P, Motion, K, Tri> OnlineSlamPipeline<Tracker<P, Motion>, LocalMappingPipeline<K, Tri>>
where
    P: FrameLocalizer,
    Motion: MotionModel,
    K: KeyframePolicy,
    Tri: Triangulator,
{
    pub fn process_frame<I>(&mut self, frame: &Frame, candidates: I) -> OnlineSlamResult
    where
        I: IntoIterator<Item = LandmarkCandidate>,
    {
        let tracking = self.tracker.track_frame(frame, &self.map);
        let mut mapping = None;
        let mut applied_update = None;

        if tracking.localization.success {
            let keyframe = keyframe_from_tracking_result(frame, &tracking);
            let mapping_result = self
                .mapper
                .process_keyframe(&self.map, &tracking, keyframe, candidates);
            if self.config.apply_map_updates && mapping_result.staged_update_validation.is_valid() {
                if let Ok(applied) = mapping_result.staged_update.clone().apply_to(&mut self.map) {
                    applied_update = Some(applied);
                }
            }
            mapping = Some(mapping_result);
        }

        OnlineSlamResult {
            tracking,
            mapping,
            applied_update,
            map_keyframe_count: self.map.keyframes.len(),
            map_landmark_count: self.map.landmarks.len(),
        }
    }

    pub fn reset_sequence_state(&mut self) {
        self.tracker.reset();
        self.mapper.reset();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamResult {
    pub tracking: TrackingResult,
    pub mapping: Option<LocalMappingResult>,
    pub applied_update: Option<AppliedMapUpdate>,
    pub map_keyframe_count: usize,
    pub map_landmark_count: usize,
}

impl OnlineSlamResult {
    pub fn tracking_succeeded(&self) -> bool {
        self.tracking.localization.success
    }

    pub fn map_was_updated(&self) -> bool {
        self.applied_update.is_some()
    }
}

fn keyframe_from_tracking_result(frame: &Frame, tracking: &TrackingResult) -> Keyframe {
    let mut frame = frame.clone();
    frame.pose = tracking.localization.pose.clone();

    let observations = tracking
        .localization
        .inlier_query_indices
        .iter()
        .zip(tracking.localization.inlier_landmark_ids.iter())
        .filter_map(|(keypoint_index, landmark_id)| {
            frame.keypoints.get(*keypoint_index).map(|xy| Observation {
                frame_id: frame.id,
                landmark_id: *landmark_id,
                keypoint_index: *keypoint_index,
                xy: *xy,
            })
        })
        .collect();

    Keyframe {
        frame,
        observations,
    }
}
