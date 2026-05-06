#![forbid(unsafe_code)]
//! Local mapping scaffolding.
//!
//! This crate intentionally does not mutate maps yet. It starts v0.3 by
//! defining keyframe-selection interfaces that can consume tracking results and
//! later feed staged map updates, triangulation, and local refinement.

use visloc_core::geometry::Pose;
use visloc_core::types::FrameId;
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
