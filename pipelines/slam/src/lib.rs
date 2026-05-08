#![forbid(unsafe_code)]
//! Minimal online SLAM orchestration.
//!
//! This crate wires tracking and local mapping together. It is not a full SLAM
//! system: it can report lightweight loop-closure candidates, but global pose
//! graph optimization, dense mapping, and production bundle adjustment remain
//! outside this MVP layer.

use std::collections::HashSet;

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
    pub loop_closure: LoopClosureConfig,
}

impl Default for OnlineSlamConfig {
    fn default() -> Self {
        Self {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopClosureConfig {
    pub enabled: bool,
    pub min_frame_id_gap: u64,
    pub min_shared_landmarks: usize,
    pub min_shared_landmark_ratio_percent: u8,
    pub max_candidates: usize,
}

impl Default for LoopClosureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_frame_id_gap: 5,
            min_shared_landmarks: 12,
            min_shared_landmark_ratio_percent: 40,
            max_candidates: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopClosureCandidate {
    pub query_frame_id: u64,
    pub matched_keyframe_id: u64,
    pub shared_landmark_count: usize,
    pub query_inlier_count: usize,
    pub keyframe_observation_count: usize,
    pub shared_landmark_ratio: f64,
    pub score: f64,
    pub geometrically_verified: bool,
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
        let loop_closure_candidates =
            detect_loop_closure_candidates(frame, &tracking, &self.map, &self.config.loop_closure);

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
            loop_closure_candidates,
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
    pub loop_closure_candidates: Vec<LoopClosureCandidate>,
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

    pub fn has_loop_closure_candidate(&self) -> bool {
        !self.loop_closure_candidates.is_empty()
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

fn detect_loop_closure_candidates(
    frame: &Frame,
    tracking: &TrackingResult,
    map: &VisualMap,
    config: &LoopClosureConfig,
) -> Vec<LoopClosureCandidate> {
    if !config.enabled || !tracking.localization.success {
        return Vec::new();
    }

    let query_landmarks = tracking
        .localization
        .inlier_landmark_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    if query_landmarks.is_empty() {
        return Vec::new();
    }

    let mut candidates = map
        .keyframes
        .values()
        .filter_map(|keyframe| {
            if frame.id.abs_diff(keyframe.frame.id) < config.min_frame_id_gap {
                return None;
            }

            let keyframe_landmarks = keyframe
                .observations
                .iter()
                .map(|observation| observation.landmark_id)
                .collect::<HashSet<_>>();
            if keyframe_landmarks.is_empty() {
                return None;
            }

            let shared_landmark_count = query_landmarks.intersection(&keyframe_landmarks).count();
            if shared_landmark_count < config.min_shared_landmarks {
                return None;
            }

            let denominator = query_landmarks.len().min(keyframe_landmarks.len());
            let shared_landmark_ratio = shared_landmark_count as f64 / denominator as f64;
            let required_ratio = f64::from(config.min_shared_landmark_ratio_percent) / 100.0;
            if shared_landmark_ratio < required_ratio {
                return None;
            }

            let score = shared_landmark_ratio * shared_landmark_count as f64;
            Some(LoopClosureCandidate {
                query_frame_id: frame.id,
                matched_keyframe_id: keyframe.frame.id,
                shared_landmark_count,
                query_inlier_count: query_landmarks.len(),
                keyframe_observation_count: keyframe_landmarks.len(),
                shared_landmark_ratio,
                score,
                geometrically_verified: true,
            })
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.shared_landmark_count.cmp(&a.shared_landmark_count))
            .then_with(|| a.matched_keyframe_id.cmp(&b.matched_keyframe_id))
    });
    candidates.truncate(config.max_candidates);
    candidates
}
