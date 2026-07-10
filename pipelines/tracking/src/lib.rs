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
use visloc_core::geometry::{Pose, Sim3, SE3};
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
    /// When `true`, scale `max_pose_prior_translation_error` by the number
    /// of frames elapsed since the last successful track (capped at
    /// `pose_jump_gap_scaling_max_multiplier`, floored at 1). A pose prior
    /// that has gone stale because tracking failed for several frames in a
    /// row is compared against a proportionally widened gate instead of the
    /// fixed radius, so a good PnP solution arriving after a gap is not
    /// rejected purely because the (frozen) prior is far from it. When the
    /// immediately preceding frame tracked successfully the gap is 1 and
    /// the gate is unchanged from today's fixed-radius behaviour. Off by
    /// default.
    pub pose_jump_gap_scaling: bool,
    /// Cap on the gap-scaling multiplier applied when `pose_jump_gap_scaling`
    /// is enabled. Prevents an extended tracking outage from inflating the
    /// gate to an unbounded radius.
    pub pose_jump_gap_scaling_max_multiplier: usize,
    /// ORB-SLAM3-style projection-guided tracking: when a pose prior is
    /// available, restrict descriptor matching to a per-landmark projection
    /// window instead of an appearance-global search, with a widen-retry
    /// ladder and post-hoc local-map refinement. `None` (the default)
    /// preserves today's appearance-global-then-warm-start behaviour
    /// bit-for-bit; enabling the feature can only ADD tracking chances,
    /// because the widen-retry ladder falls back to the existing
    /// appearance-global path if every projection attempt fails.
    pub projection_guided_tracking: Option<ProjectionGuidedTrackingConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectionGuidedTrackingConfig {
    /// Initial per-landmark projection-window radius, in pixels.
    pub search_radius_px: f64,
    /// Multiplier applied to `search_radius_px` on each widen-retry attempt
    /// after a stage-1 projection attempt fails the pose-estimation quality
    /// gate.
    pub widen_factor: f64,
    /// Maximum number of widen-retry attempts after the initial one. Total
    /// projection attempts per frame is `1 + max_widen_retries`; if all of
    /// them fail, the tracker falls back to the appearance-global path.
    pub max_widen_retries: u32,
    /// When `true`, after ANY successful pose estimate (projection path or
    /// appearance-global fallback), project the covisibility local map with
    /// the estimated pose and re-optimize over the union of harvested and
    /// existing inlier correspondences. Requires `covisibility_local_map`
    /// to also be configured (the local-map landmark set the brief and
    /// `build_covisibility_local_map_store` compute); otherwise this stage
    /// is a no-op since there is no local-map landmark set to project.
    pub local_map_refinement: bool,
    /// Per-landmark projection-window radius, in pixels, used by the
    /// local-map refinement stage. Independent of `search_radius_px` since
    /// refinement starts from an already-estimated (not merely predicted)
    /// pose and can afford a tighter window.
    pub refinement_search_radius_px: f64,
}

impl Default for ProjectionGuidedTrackingConfig {
    fn default() -> Self {
        Self {
            search_radius_px: 15.0,
            widen_factor: 2.0,
            max_widen_retries: 2,
            local_map_refinement: true,
            refinement_search_radius_px: 8.0,
        }
    }
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
            pose_jump_gap_scaling: false,
            pose_jump_gap_scaling_max_multiplier: 10,
            projection_guided_tracking: None,
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

pub mod motion;
pub use motion::*;

pub mod report;
pub use report::*;

pub mod tracker;
pub use tracker::*;

pub mod trajectory;
pub use trajectory::*;
