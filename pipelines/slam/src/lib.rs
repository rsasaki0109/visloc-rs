#![forbid(unsafe_code)]
//! Minimal online SLAM orchestration.
//!
//! This crate wires tracking and local mapping together. It is not a full SLAM
//! system: it can report lightweight loop-closure candidates, but global pose
//! graph optimization, dense mapping, and production bundle adjustment remain
//! outside this MVP layer.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use nalgebra::{DMatrix, DVector, Matrix6, Point2, Point3, Vector6};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Camera, Frame, Keyframe, Observation, VisualMap};
use visloc_localization::LocalizationPipeline;
use visloc_mapping::{
    AppliedMapUpdate, KeyframePolicy, LandmarkCandidate, LinearTriangulator, LocalMappingPipeline,
    LocalMappingResult, SimpleKeyframePolicy, Triangulator,
};
use visloc_tracking::{
    ConstantPoseMotionModel, FrameLocalizer, MotionModel, Tracker, TrackingConfig, TrackingResult,
};
use visloc_vision::pnp::Correspondence2D3D;
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_vision::two_view::{
    EightPointEssentialMatrixEstimator, RelativePoseEstimator, TwoViewCorrespondence,
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
    /// `true` while the candidate has not been rejected by an explicit
    /// verifier. When [`verify_loop_closure_candidates`] runs, this becomes
    /// `LoopClosureVerification::verified`.
    pub geometrically_verified: bool,
    /// Optional verifier output. `Some` when [`verify_loop_closure_candidates`]
    /// (or another caller) has explicitly run a [`LoopClosureVerifier`] over
    /// the candidate; `None` when only the shared-landmark heuristic has
    /// produced the candidate.
    pub verification: Option<LoopClosureVerification>,
}

/// Configuration thresholds for [`EssentialMatrixLoopClosureVerifier`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopClosureVerifierConfig {
    /// Minimum number of inliers an essential-matrix RANSAC fit must produce
    /// for the candidate to be accepted.
    pub min_inliers: usize,
    /// Minimum inlier ratio (inliers / supplied correspondences) for
    /// acceptance.
    pub min_inlier_ratio: f64,
    /// Maximum allowed mean Sampson distance, in normalized image-plane units
    /// (multiply by focal length to convert to pixels).
    pub max_mean_sampson_error: f64,
    /// Translation scale applied when recovering the relative pose. Two-view
    /// geometry leaves translation up to scale; this default is what
    /// [`LoopClosureVerification::relative_pose`] uses unless callers wrap
    /// the verifier with their own scale source. Defaults to `1.0`.
    pub default_translation_scale: f64,
}

impl Default for LoopClosureVerifierConfig {
    fn default() -> Self {
        Self {
            min_inliers: 8,
            min_inlier_ratio: 0.5,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: 1.0,
        }
    }
}

/// Reason a [`LoopClosureVerification`] rejected a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopClosureVerificationFailureReason {
    /// Fewer correspondences than the verifier's minimum requirement.
    InsufficientCorrespondences,
    /// The essential-matrix RANSAC failed to find a consensus.
    EssentialEstimationFailed,
    /// The RANSAC produced fewer inliers than `min_inliers`.
    TooFewInliers,
    /// The inlier ratio fell below `min_inlier_ratio`.
    LowInlierRatio,
    /// Mean Sampson error exceeded `max_mean_sampson_error`.
    HighSampsonError,
}

/// Output of running a [`LoopClosureVerifier`] on a candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopClosureVerification {
    pub verified: bool,
    pub correspondence_count: usize,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    /// Mean Sampson distance reported by an essential-matrix verifier (in
    /// normalized image-plane units). `0.0` and uninformative for
    /// PnP-based verifiers; check [`Self::mean_reprojection_error_px`] in
    /// that case.
    pub mean_sampson_error: f64,
    pub score: f64,
    pub failure_reason: Option<LoopClosureVerificationFailureReason>,
    /// Recovered relative pose (older keyframe → current frame) when the
    /// underlying RANSAC converged. `Some` even for non-`verified` cases as
    /// long as a pose was recovered; consult `verified` and `failure_reason`
    /// before consuming. For essential-matrix verifiers the translation is
    /// scaled by [`LoopClosureVerifierConfig::default_translation_scale`];
    /// for PnP verifiers it is in metric units (the keyframe pose carries
    /// the world scale).
    pub relative_pose: Option<SE3>,
    /// Mean reprojection error (in pixels) reported by a PnP-based verifier.
    /// `None` for essential-matrix verifiers.
    pub mean_reprojection_error_px: Option<f64>,
}

/// Trait for a loop-closure candidate verifier. Concrete implementations
/// receive 2D-2D correspondences in pixel coordinates between the older
/// keyframe (`previous_xy`) and the current/query frame (`current_xy`) plus
/// the shared camera intrinsics.
pub trait LoopClosureVerifier {
    fn verify(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> LoopClosureVerification;
}

/// Geometric verifier that runs the classical essential-matrix RANSAC from
/// `visloc-vision::two_view` on the supplied correspondences and reports
/// inlier statistics, mean Sampson error, and a combined score.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EssentialMatrixLoopClosureVerifier {
    pub estimator: RelativePoseEstimator<EightPointEssentialMatrixEstimator>,
    pub config: LoopClosureVerifierConfig,
}

impl EssentialMatrixLoopClosureVerifier {
    pub fn new(
        estimator: RelativePoseEstimator<EightPointEssentialMatrixEstimator>,
        config: LoopClosureVerifierConfig,
    ) -> Self {
        Self { estimator, config }
    }
}

impl LoopClosureVerifier for EssentialMatrixLoopClosureVerifier {
    fn verify(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> LoopClosureVerification {
        let correspondence_count = correspondences.len();
        let minimum = self
            .estimator
            .ransac
            .estimator
            .min_correspondences
            .max(self.config.min_inliers);
        if correspondence_count < minimum {
            return LoopClosureVerification {
                verified: false,
                correspondence_count,
                inlier_count: 0,
                inlier_ratio: 0.0,
                mean_sampson_error: f64::INFINITY,
                score: 0.0,
                failure_reason: Some(
                    LoopClosureVerificationFailureReason::InsufficientCorrespondences,
                ),
                relative_pose: None,
                mean_reprojection_error_px: None,
            };
        }

        let Some(relative_pose) = self.estimator.estimate_with_scale(
            correspondences,
            camera,
            self.config.default_translation_scale,
        ) else {
            return LoopClosureVerification {
                verified: false,
                correspondence_count,
                inlier_count: 0,
                inlier_ratio: 0.0,
                mean_sampson_error: f64::INFINITY,
                score: 0.0,
                failure_reason: Some(
                    LoopClosureVerificationFailureReason::EssentialEstimationFailed,
                ),
                relative_pose: None,
                mean_reprojection_error_px: None,
            };
        };

        let inlier_count = relative_pose.inliers.len();
        let inlier_ratio = inlier_count as f64 / correspondence_count as f64;
        let mean_sampson = relative_pose.mean_sampson_error;

        let mut failure_reason = None;
        if inlier_count < self.config.min_inliers {
            failure_reason = Some(LoopClosureVerificationFailureReason::TooFewInliers);
        } else if inlier_ratio < self.config.min_inlier_ratio {
            failure_reason = Some(LoopClosureVerificationFailureReason::LowInlierRatio);
        } else if mean_sampson > self.config.max_mean_sampson_error {
            failure_reason = Some(LoopClosureVerificationFailureReason::HighSampsonError);
        }
        let verified = failure_reason.is_none();
        let inlier_volume = inlier_ratio * inlier_count as f64;
        let denominator = mean_sampson.max(1.0e-6);
        let score = if denominator.is_finite() {
            inlier_volume / denominator
        } else {
            inlier_volume
        };
        LoopClosureVerification {
            verified,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: mean_sampson,
            score,
            failure_reason,
            relative_pose: Some(relative_pose.previous_to_current),
            mean_reprojection_error_px: None,
        }
    }
}

/// Configuration thresholds for [`PnPLoopClosureVerifier`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PnPLoopClosureVerifierConfig {
    /// Minimum number of PnP RANSAC inliers an accepted candidate must
    /// produce.
    pub min_inliers: usize,
    /// Minimum inlier ratio (inliers / supplied 2D-3D correspondences).
    pub min_inlier_ratio: f64,
    /// Maximum allowed mean reprojection error (in pixels) for inliers.
    pub max_mean_reprojection_error_px: f64,
}

impl Default for PnPLoopClosureVerifierConfig {
    fn default() -> Self {
        Self {
            min_inliers: 8,
            min_inlier_ratio: 0.5,
            max_mean_reprojection_error_px: 4.0,
        }
    }
}

/// PnP-based loop-closure verifier. Reuses the project's [`PnPRansac`] to
/// re-localize the current frame against landmarks observed by the candidate
/// keyframe; if the recovered absolute pose has enough inliers and a small
/// reprojection error, the candidate is accepted and the relative pose
/// (older keyframe → current frame) is filled into
/// [`LoopClosureVerification::relative_pose`].
///
/// Compared with [`EssentialMatrixLoopClosureVerifier`], this verifier:
///
/// - operates on 2D-3D correspondences instead of 2D-2D, so it checks the
///   candidate against the actual 3D map structure rather than two-view
///   geometry alone;
/// - returns metric translations (the keyframe pose carries the world scale),
///   so callers do not need to plug in a separate `default_translation_scale`;
/// - is preferable when the older keyframe has sufficient triangulated
///   landmarks visible from the current frame, which is the common case for
///   in-map loop closures.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PnPLoopClosureVerifier<R = PnPRansac> {
    pub ransac: R,
    pub config: PnPLoopClosureVerifierConfig,
}

impl<R> PnPLoopClosureVerifier<R>
where
    R: RobustPoseEstimator,
{
    pub fn new(ransac: R, config: PnPLoopClosureVerifierConfig) -> Self {
        Self { ransac, config }
    }

    /// Run PnP RANSAC on `correspondences` and turn the report into a
    /// [`LoopClosureVerification`]. `keyframe_pose` is the older keyframe's
    /// stored `world_to_camera` SE3; the recovered current-frame pose is
    /// composed with its inverse to populate `relative_pose`.
    pub fn verify(
        &self,
        correspondences: &[Correspondence2D3D],
        keyframe_pose: &Pose,
        camera: &Camera,
    ) -> LoopClosureVerification {
        let correspondence_count = correspondences.len();
        if correspondence_count < self.config.min_inliers {
            return LoopClosureVerification {
                verified: false,
                correspondence_count,
                inlier_count: 0,
                inlier_ratio: 0.0,
                mean_sampson_error: 0.0,
                score: 0.0,
                failure_reason: Some(
                    LoopClosureVerificationFailureReason::InsufficientCorrespondences,
                ),
                relative_pose: None,
                mean_reprojection_error_px: None,
            };
        }

        let Some(report) = self.ransac.estimate(correspondences, camera) else {
            return LoopClosureVerification {
                verified: false,
                correspondence_count,
                inlier_count: 0,
                inlier_ratio: 0.0,
                mean_sampson_error: 0.0,
                score: 0.0,
                failure_reason: Some(
                    LoopClosureVerificationFailureReason::EssentialEstimationFailed,
                ),
                relative_pose: None,
                mean_reprojection_error_px: None,
            };
        };

        let inlier_count = report.inliers.len();
        let inlier_ratio = inlier_count as f64 / correspondence_count as f64;
        let mean_reprojection_error_px = report.mean_reprojection_error;

        let mut failure_reason = None;
        if inlier_count < self.config.min_inliers {
            failure_reason = Some(LoopClosureVerificationFailureReason::TooFewInliers);
        } else if inlier_ratio < self.config.min_inlier_ratio {
            failure_reason = Some(LoopClosureVerificationFailureReason::LowInlierRatio);
        } else if mean_reprojection_error_px > self.config.max_mean_reprojection_error_px {
            failure_reason = Some(LoopClosureVerificationFailureReason::HighSampsonError);
        }
        let verified = failure_reason.is_none();
        let inlier_volume = inlier_ratio * inlier_count as f64;
        let denominator = mean_reprojection_error_px.max(1.0e-6);
        let score = if denominator.is_finite() {
            inlier_volume / denominator
        } else {
            inlier_volume
        };
        let relative_pose = report
            .pose
            .world_to_camera
            .compose(&keyframe_pose.world_to_camera.inverse());
        LoopClosureVerification {
            verified,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: 0.0,
            score,
            failure_reason,
            relative_pose: Some(relative_pose),
            mean_reprojection_error_px: Some(mean_reprojection_error_px),
        }
    }
}

/// Build 2D-3D correspondences for a loop-closure candidate by intersecting
/// the current frame's tracking inliers with the older keyframe's observed
/// landmarks. Each shared landmark contributes one entry pairing the current
/// frame's pixel observation with the landmark's world position.
pub fn correspondences_2d3d_for_loop_candidate(
    current_frame: &Frame,
    current_inlier_query_indices: &[usize],
    current_inlier_landmark_ids: &[u64],
    keyframe: &Keyframe,
    map: &VisualMap,
) -> Vec<Correspondence2D3D> {
    let keyframe_landmark_ids: HashSet<u64> = keyframe
        .observations
        .iter()
        .map(|observation| observation.landmark_id)
        .collect();
    let mut correspondences = Vec::new();
    for (query_index, landmark_id) in current_inlier_query_indices
        .iter()
        .zip(current_inlier_landmark_ids.iter())
    {
        if !keyframe_landmark_ids.contains(landmark_id) {
            continue;
        }
        let Some(landmark) = map.landmarks.get(landmark_id) else {
            continue;
        };
        let Some(query_xy) = current_frame.keypoints.get(*query_index) else {
            continue;
        };
        correspondences.push(Correspondence2D3D {
            point2d: *query_xy,
            point3d: landmark.position,
        });
    }
    correspondences
}

/// Build pixel-space two-view correspondences for a loop-closure candidate
/// from the current frame's tracking inliers and an older keyframe's
/// observations. Each shared landmark id contributes one correspondence
/// `(keyframe_xy, current_xy)`.
pub fn correspondences_for_loop_candidate(
    current_frame: &Frame,
    current_inlier_query_indices: &[usize],
    current_inlier_landmark_ids: &[u64],
    keyframe: &Keyframe,
) -> Vec<TwoViewCorrespondence> {
    let keyframe_lookup: HashMap<u64, Point2<f64>> = keyframe
        .observations
        .iter()
        .map(|observation| (observation.landmark_id, observation.xy))
        .collect();
    let mut correspondences = Vec::new();
    for (query_index, landmark_id) in current_inlier_query_indices
        .iter()
        .zip(current_inlier_landmark_ids.iter())
    {
        let Some(keyframe_xy) = keyframe_lookup.get(landmark_id) else {
            continue;
        };
        let Some(query_xy) = current_frame.keypoints.get(*query_index) else {
            continue;
        };
        correspondences.push(TwoViewCorrespondence {
            previous_xy: *keyframe_xy,
            current_xy: *query_xy,
        });
    }
    correspondences
}

/// Pose-graph-style constraint between two keyframes derived from a verified
/// loop-closure candidate. This is intentionally a lightweight data type — no
/// solver lives in this crate yet — so downstream optimization layers can
/// adopt it without committing to a specific backend.
///
/// `relative_pose` represents the rigid transform that takes a point in
/// `from_keyframe_id`'s camera frame to `to_keyframe_id`'s camera frame, with
/// the translation scaled by the verifier's
/// [`LoopClosureVerifierConfig::default_translation_scale`] (or whatever
/// scale the caller chose to apply before constructing the constraint).
#[derive(Debug, Clone, PartialEq)]
pub struct LoopClosureConstraint {
    pub from_keyframe_id: u64,
    pub to_keyframe_id: u64,
    pub relative_pose: SE3,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub mean_sampson_error: f64,
    pub score: f64,
}

impl LoopClosureConstraint {
    /// Builds a constraint from a verified candidate. Returns `None` when the
    /// candidate has no verifier output, when the verifier rejected it, or
    /// when no relative pose was recovered.
    pub fn from_verified_candidate(
        candidate: &LoopClosureCandidate,
    ) -> Option<LoopClosureConstraint> {
        let verification = candidate.verification.as_ref()?;
        if !verification.verified {
            return None;
        }
        let relative_pose = verification.relative_pose.clone()?;
        Some(LoopClosureConstraint {
            from_keyframe_id: candidate.matched_keyframe_id,
            to_keyframe_id: candidate.query_frame_id,
            relative_pose,
            inlier_count: verification.inlier_count,
            inlier_ratio: verification.inlier_ratio,
            mean_sampson_error: verification.mean_sampson_error,
            score: verification.score,
        })
    }
}

/// Convenience helper that builds a constraint per verified candidate. Keeps
/// the same ordering as the input slice and silently drops candidates that
/// were not verified or lack a recovered relative pose.
pub fn loop_closure_constraints_from_candidates(
    candidates: &[LoopClosureCandidate],
) -> Vec<LoopClosureConstraint> {
    candidates
        .iter()
        .filter_map(LoopClosureConstraint::from_verified_candidate)
        .collect()
}

/// Run `verifier` on every supplied candidate, mutating each
/// [`LoopClosureCandidate`] in place: `verification` is set to the verifier
/// output and `geometrically_verified` is replaced with
/// `LoopClosureVerification::verified`. Candidates whose matched keyframe is
/// no longer in `map` are left untouched.
pub fn verify_loop_closure_candidates<V>(
    candidates: &mut [LoopClosureCandidate],
    current_frame: &Frame,
    tracking: &TrackingResult,
    map: &VisualMap,
    camera: &Camera,
    verifier: &V,
) where
    V: LoopClosureVerifier,
{
    for candidate in candidates.iter_mut() {
        let Some(keyframe) = map.keyframes.get(&candidate.matched_keyframe_id) else {
            continue;
        };
        let correspondences = correspondences_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
        );
        let verification = verifier.verify(&correspondences, camera);
        candidate.geometrically_verified = verification.verified;
        candidate.verification = Some(verification);
    }
}

/// Run a PnP-based [`PnPLoopClosureVerifier`] on every supplied candidate.
/// For each candidate this builds 2D-3D correspondences via
/// [`correspondences_2d3d_for_loop_candidate`], runs PnP RANSAC on them, and
/// updates `verification` and `geometrically_verified` in place.
/// Candidates whose matched keyframe is no longer in `map`, or whose stored
/// keyframe pose is missing, are left untouched.
pub fn verify_loop_closure_candidates_pnp<R>(
    candidates: &mut [LoopClosureCandidate],
    current_frame: &Frame,
    tracking: &TrackingResult,
    map: &VisualMap,
    camera: &Camera,
    verifier: &PnPLoopClosureVerifier<R>,
) where
    R: RobustPoseEstimator,
{
    for candidate in candidates.iter_mut() {
        let Some(keyframe) = map.keyframes.get(&candidate.matched_keyframe_id) else {
            continue;
        };
        let Some(keyframe_pose) = keyframe.frame.pose.as_ref() else {
            continue;
        };
        let correspondences = correspondences_2d3d_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
            map,
        );
        let verification = verifier.verify(&correspondences, keyframe_pose, camera);
        candidate.geometrically_verified = verification.verified;
        candidate.verification = Some(verification);
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

pub fn online_slam_results_to_html_report(results: &[OnlineSlamResult]) -> String {
    let samples = slam_report_samples(results);
    let loop_candidates = results
        .iter()
        .flat_map(|result| result.loop_closure_candidates.iter())
        .collect::<Vec<_>>();
    let mut output = String::new();
    output.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    output.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    output.push_str("<title>visloc-rs online SLAM loop report</title>\n");
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
         th:first-child,td:first-child{text-align:left}\
         .ok{color:#198754;font-weight:700}.warn{color:#a15c00;font-weight:700}\
         svg{width:100%;height:auto;display:block}",
    );
    output.push_str("</style>\n</head>\n<body>\n<main>\n");
    output.push_str("<h1>visloc-rs online SLAM loop report</h1>\n");
    output.push_str("<p class=\"sub\">Top-down tracked camera centers with diagnostic loop-closure candidate edges. This report does not imply global pose-graph optimization.</p>\n");
    output.push_str("<section class=\"grid\">\n");
    push_metric_card(&mut output, "Frames", &results.len().to_string());
    push_metric_card(&mut output, "Tracked poses", &samples.len().to_string());
    push_metric_card(
        &mut output,
        "Loop candidates",
        &loop_candidates.len().to_string(),
    );
    push_metric_card(
        &mut output,
        "Final keyframes",
        &results
            .last()
            .map(|result| result.map_keyframe_count.to_string())
            .unwrap_or_else(|| "0".to_string()),
    );
    push_metric_card(
        &mut output,
        "Final landmarks",
        &results
            .last()
            .map(|result| result.map_landmark_count.to_string())
            .unwrap_or_else(|| "0".to_string()),
    );
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n");
    output.push_str(&online_slam_loop_svg(&samples, &loop_candidates));
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n<h2>Loop Closure Candidates</h2>\n");
    output.push_str("<table><thead><tr><th>query frame</th><th>matched keyframe</th><th>shared landmarks</th><th>ratio</th><th>score</th><th>verified</th><th>verifier inliers</th><th>verifier inlier ratio</th><th>mean error</th><th>verifier score</th><th>failure</th></tr></thead><tbody>\n");
    if loop_candidates.is_empty() {
        output.push_str("<tr><td colspan=\"11\">no loop candidates reported</td></tr>\n");
    }
    for candidate in &loop_candidates {
        let verified_class = if candidate.geometrically_verified {
            "ok"
        } else {
            "warn"
        };
        let verified_text = match candidate.verification.as_ref() {
            Some(verification) if verification.verified => "yes",
            Some(_) => "rejected",
            None => "candidate",
        };
        let (inlier_count_text, inlier_ratio_text, mean_text, verifier_score_text, failure_text) =
            if let Some(verification) = candidate.verification.as_ref() {
                (
                    verification.inlier_count.to_string(),
                    format!("{:.3}", verification.inlier_ratio),
                    if let Some(px) = verification.mean_reprojection_error_px {
                        format!("{px:.4} px")
                    } else if verification.mean_sampson_error.is_finite() {
                        format!("{:.4}", verification.mean_sampson_error)
                    } else {
                        "n/a".to_string()
                    },
                    format!("{:.3}", verification.score),
                    verification
                        .failure_reason
                        .as_ref()
                        .map(format_loop_closure_failure_reason)
                        .unwrap_or_else(|| "&mdash;".to_string()),
                )
            } else {
                (
                    "&mdash;".to_string(),
                    "&mdash;".to_string(),
                    "&mdash;".to_string(),
                    "&mdash;".to_string(),
                    "&mdash;".to_string(),
                )
            };
        let _ = writeln!(
            output,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td class=\"{}\">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            candidate.query_frame_id,
            candidate.matched_keyframe_id,
            candidate.shared_landmark_count,
            candidate.shared_landmark_ratio,
            candidate.score,
            verified_class,
            verified_text,
            inlier_count_text,
            inlier_ratio_text,
            mean_text,
            verifier_score_text,
            failure_text,
        );
    }
    output.push_str("</tbody></table>\n</section>\n");

    let constraints: Vec<LoopClosureConstraint> = loop_candidates
        .iter()
        .filter_map(|candidate| LoopClosureConstraint::from_verified_candidate(candidate))
        .collect();
    output.push_str("<section class=\"panel\">\n<h2>Loop Closure Constraints</h2>\n");
    output.push_str("<p class=\"sub\">Each row is a verified candidate turned into a `LoopClosureConstraint` ready for a future pose-graph layer. No global optimization runs in this report.</p>\n");
    output.push_str("<table><thead><tr><th>from keyframe</th><th>to keyframe</th><th>inliers</th><th>inlier ratio</th><th>mean Sampson</th><th>score</th><th>relative translation</th></tr></thead><tbody>\n");
    if constraints.is_empty() {
        output.push_str("<tr><td colspan=\"7\">no verified loop constraints</td></tr>\n");
    }
    for constraint in &constraints {
        let translation = constraint.relative_pose.translation;
        let _ = writeln!(
            output,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.4}</td><td>{:.3}</td><td>[{:.3}, {:.3}, {:.3}]</td></tr>",
            constraint.from_keyframe_id,
            constraint.to_keyframe_id,
            constraint.inlier_count,
            constraint.inlier_ratio,
            constraint.mean_sampson_error,
            constraint.score,
            translation.x,
            translation.y,
            translation.z,
        );
    }
    output.push_str("</tbody></table>\n</section>\n</main>\n</body>\n</html>\n");
    output
}

fn format_loop_closure_failure_reason(reason: &LoopClosureVerificationFailureReason) -> String {
    match reason {
        LoopClosureVerificationFailureReason::InsufficientCorrespondences => {
            "insufficient correspondences".to_string()
        }
        LoopClosureVerificationFailureReason::EssentialEstimationFailed => {
            "essential RANSAC failed".to_string()
        }
        LoopClosureVerificationFailureReason::TooFewInliers => "too few inliers".to_string(),
        LoopClosureVerificationFailureReason::LowInlierRatio => "low inlier ratio".to_string(),
        LoopClosureVerificationFailureReason::HighSampsonError => "high Sampson error".to_string(),
    }
}

pub fn write_online_slam_results_html_report(
    results: &[OnlineSlamResult],
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    std::fs::write(path, online_slam_results_to_html_report(results))
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

#[derive(Debug, Clone, Copy)]
struct SlamReportSample {
    frame_id: u64,
    x: f64,
    y: f64,
    z: f64,
}

fn slam_report_samples(results: &[OnlineSlamResult]) -> Vec<SlamReportSample> {
    results
        .iter()
        .filter_map(|result| {
            let pose = result.tracking.localization.pose.as_ref()?;
            let center = pose.camera_center_world();
            Some(SlamReportSample {
                frame_id: result.tracking.frame_id,
                x: center.x,
                y: center.y,
                z: center.z,
            })
        })
        .collect()
}

fn online_slam_loop_svg(
    samples: &[SlamReportSample],
    candidates: &[&LoopClosureCandidate],
) -> String {
    let projection = SlamReportProjection::from_samples(samples);
    let by_frame_id = samples
        .iter()
        .map(|sample| (sample.frame_id, *sample))
        .collect::<HashMap<_, _>>();
    let mut output = String::new();
    output.push_str("<svg viewBox=\"0 0 900 520\" role=\"img\" aria-label=\"online SLAM loop candidate plot\">\n");
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
    if !samples.is_empty() {
        let points = samples
            .iter()
            .map(|sample| {
                let (x, y) = projection.project(sample);
                format!("{x:.2},{y:.2}")
            })
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            output,
            "<polyline points=\"{points}\" stroke=\"#2676c9\" stroke-width=\"4\"/>"
        );
    }
    output.push_str("</g>\n");
    output.push_str("<g fill=\"none\" stroke=\"#f0a202\" stroke-width=\"4\" stroke-dasharray=\"10 7\" stroke-linecap=\"round\">\n");
    for candidate in candidates {
        let (Some(query), Some(matched)) = (
            by_frame_id.get(&candidate.query_frame_id),
            by_frame_id.get(&candidate.matched_keyframe_id),
        ) else {
            continue;
        };
        let (qx, qy) = projection.project(query);
        let (mx, my) = projection.project(matched);
        let _ = writeln!(
            output,
            "<line x1=\"{qx:.2}\" y1=\"{qy:.2}\" x2=\"{mx:.2}\" y2=\"{my:.2}\"/>"
        );
    }
    output.push_str("</g>\n<g>\n");
    for sample in samples {
        let (x, y) = projection.project(sample);
        let _ = writeln!(
            output,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"6\" fill=\"#2676c9\"/>"
        );
        let _ = writeln!(
            output,
            "<text x=\"{:.2}\" y=\"{:.2}\" fill=\"#52616b\" font-size=\"12\" text-anchor=\"middle\">{}</text>",
            x,
            y + 22.0,
            sample.frame_id
        );
    }
    output.push_str("</g>\n");
    output.push_str("<rect x=\"80\" y=\"468\" width=\"14\" height=\"6\" fill=\"#2676c9\"/>\n");
    output.push_str(
        "<text x=\"102\" y=\"476\" fill=\"#52616b\" font-size=\"13\">tracked camera path</text>\n",
    );
    output.push_str("<line x1=\"278\" y1=\"472\" x2=\"320\" y2=\"472\" stroke=\"#f0a202\" stroke-width=\"4\" stroke-dasharray=\"10 7\"/>\n");
    output.push_str(
        "<text x=\"330\" y=\"476\" fill=\"#52616b\" font-size=\"13\">loop candidate edge</text>\n",
    );
    output.push_str("</svg>\n");
    output
}

#[derive(Debug, Clone, Copy)]
struct SlamReportProjection {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    axis_y: usize,
}

impl SlamReportProjection {
    fn from_samples(samples: &[SlamReportSample]) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for sample in samples {
            min[0] = min[0].min(sample.x);
            min[1] = min[1].min(sample.y);
            min[2] = min[2].min(sample.z);
            max[0] = max[0].max(sample.x);
            max[1] = max[1].max(sample.y);
            max[2] = max[2].max(sample.z);
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

    fn project(&self, sample: &SlamReportSample) -> (f64, f64) {
        let plot_left = 80.0;
        let plot_top = 54.0;
        let plot_width = 740.0;
        let plot_height = 396.0;
        let horizontal = (sample.x - self.min_x) / (self.max_x - self.min_x);
        let vertical_value = if self.axis_y == 2 { sample.z } else { sample.y };
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

fn push_metric_card(output: &mut String, label: &str, value: &str) {
    let _ = writeln!(
        output,
        "<div class=\"metric\"><span class=\"label\">{label}</span><span class=\"value\">{value}</span></div>"
    );
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
                verification: None,
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

/// Kind of an edge inside a [`PoseGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoseGraphEdgeKind {
    /// Sequential odometry edge between consecutive keyframes.
    Sequential,
    /// Loop-closure edge backed by a verified [`LoopClosureConstraint`].
    LoopClosure,
}

/// Edge in a sparse [`PoseGraph`]. Encodes a measured `previous_to_current`
/// SE3 between two keyframes plus a positive weight used by translation-only
/// least squares.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphEdge {
    pub from: u64,
    pub to: u64,
    pub measurement: SE3,
    pub kind: PoseGraphEdgeKind,
    pub weight: f64,
}

/// Single Gauss-Newton step diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphOptimizationStep {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub mean_translation_correction: f64,
    pub max_translation_correction: f64,
}

/// Robust kernel applied to each pose-graph edge's residual norm-squared.
/// Down-weights edges whose squared residual exceeds the kernel threshold so
/// outlier loop closures cannot dominate the least-squares solve.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RobustKernel {
    /// Standard squared-error cost (`ρ(s) = s`).
    #[default]
    None,
    /// Huber kernel: quadratic for `s ≤ δ²`, linear in `√s` beyond.
    /// `delta` is the threshold on residual norm where the kernel switches
    /// from quadratic to linear.
    Huber { delta: f64 },
    /// Cauchy / Lorentzian kernel: `ρ(s) = c² · log(1 + s / c²)`.
    /// `c` is the soft-saturation scale on residual norm.
    Cauchy { c: f64 },
}

impl RobustKernel {
    /// Applied cost `ρ(s)` for `s = ||r||²`.
    pub fn cost(&self, s: f64) -> f64 {
        match *self {
            RobustKernel::None => s,
            RobustKernel::Huber { delta } => {
                let delta_sq = delta * delta;
                if s <= delta_sq {
                    s
                } else {
                    2.0 * delta * s.sqrt() - delta_sq
                }
            }
            RobustKernel::Cauchy { c } => {
                let c_sq = c * c;
                c_sq * (1.0 + s / c_sq).ln()
            }
        }
    }

    /// Influence weight `ρ'(s)` used as a multiplier on each edge's normal-equation
    /// contribution (a.k.a. IRLS weight).
    pub fn weight(&self, s: f64) -> f64 {
        match *self {
            RobustKernel::None => 1.0,
            RobustKernel::Huber { delta } => {
                let delta_sq = delta * delta;
                if s <= delta_sq {
                    1.0
                } else {
                    delta / s.sqrt()
                }
            }
            RobustKernel::Cauchy { c } => {
                let c_sq = c * c;
                1.0 / (1.0 + s / c_sq)
            }
        }
    }
}

/// Configuration for [`PoseGraph::optimize_se3_iterative`].
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSe3Config {
    /// Hard cap on iterations (including rejected LM steps).
    pub max_iterations: usize,
    /// Convergence threshold on the largest per-node 6-vector update of the
    /// most recent accepted step.
    pub step_tolerance: f64,
    /// Convergence threshold on the absolute cost change between two
    /// successive accepted steps.
    pub cost_tolerance: f64,
    /// Robust kernel applied to each edge's squared residual norm.
    pub robust_kernel: RobustKernel,
    /// Initial Levenberg-Marquardt damping `λ`. `None` runs pure
    /// Gauss-Newton (every step is accepted unconditionally). `Some(λ₀)`
    /// enables LM: solve `(H + λI) δ = -g`, accept if cost decreases (and
    /// scale `λ` down by `lambda_decrease_factor`), otherwise reject and
    /// scale `λ` up by `lambda_increase_factor`.
    pub initial_lambda: Option<f64>,
    /// Multiplier applied to `λ` after a rejected LM step.
    pub lambda_increase_factor: f64,
    /// Multiplier applied to `λ` after an accepted LM step.
    pub lambda_decrease_factor: f64,
    /// Upper bound on `λ`. When a step is rejected and `λ * factor > max_lambda`,
    /// the optimizer gives up and returns `converged: false`.
    pub max_lambda: f64,
    /// Lower bound on `λ`. Decreases stop here so `λ` cannot collapse to zero.
    pub min_lambda: f64,
}

impl Default for PoseGraphSe3Config {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            step_tolerance: 1e-6,
            cost_tolerance: 1e-9,
            robust_kernel: RobustKernel::None,
            initial_lambda: None,
            lambda_increase_factor: 10.0,
            lambda_decrease_factor: 0.1,
            max_lambda: 1e12,
            min_lambda: 1e-9,
        }
    }
}

/// Per-iteration diagnostics for [`PoseGraph::optimize_se3_iterative`].
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSe3IterationStats {
    pub iteration: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub max_step_norm: f64,
    /// LM damping `λ` used for this iteration (`0.0` for pure Gauss-Newton).
    pub lambda: f64,
    /// `true` when the trial step was kept; only false for rejected LM steps.
    pub step_accepted: bool,
}

/// Result of a full SE(3) Gauss-Newton run.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSe3Result {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: Vec<PoseGraphSe3IterationStats>,
    pub converged: bool,
}

/// Errors returned by [`PoseGraph::optimize_translations_once`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoseGraphError {
    /// No anchor was specified before optimization.
    NoAnchor,
    /// An edge or anchor referenced a node that is missing from the graph.
    MissingNode(u64),
    /// The graph contains no edges, so there is nothing to optimize.
    NoEdges,
    /// The graph contains no non-anchor nodes (all variables are fixed).
    NoVariables,
    /// The Gauss-Newton normal equations were singular, e.g., because the
    /// graph has disconnected components or rank-deficient constraints.
    SingularSystem,
}

impl std::fmt::Display for PoseGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoseGraphError::NoAnchor => write!(f, "pose graph has no anchor"),
            PoseGraphError::MissingNode(id) => write!(f, "pose graph is missing node {id}"),
            PoseGraphError::NoEdges => write!(f, "pose graph has no edges"),
            PoseGraphError::NoVariables => write!(f, "pose graph has no non-anchor nodes"),
            PoseGraphError::SingularSystem => {
                write!(f, "pose graph translation Gauss-Newton system was singular")
            }
        }
    }
}

impl std::error::Error for PoseGraphError {}

/// Sparse pose graph keyed by keyframe id. Stores per-node poses plus a flat
/// list of sequential and loop-closure edges, and provides a single
/// translation-only Gauss-Newton step that keeps node rotations fixed.
///
/// This is intentionally a skeleton: rotations are not optimized, the solver
/// is a single linear least-squares step rather than an iterative SE3 solver,
/// and there is no incremental incremental map update. Future milestones can
/// extend the same data type with full SE3 Jacobians, robust kernels, or a
/// production solver.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseGraph {
    /// Keyframe id → pose. `BTreeMap` keeps the iteration order deterministic
    /// so the variable layout in the linear system is reproducible.
    pub poses: BTreeMap<u64, Pose>,
    /// Edges in insertion order.
    pub edges: Vec<PoseGraphEdge>,
    /// Anchor keyframe id; its pose is held fixed during optimization.
    pub anchor: Option<u64>,
}

impl PoseGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a pose for `keyframe_id`.
    pub fn add_pose(&mut self, keyframe_id: u64, pose: Pose) {
        self.poses.insert(keyframe_id, pose);
    }

    /// Designate `keyframe_id` as the anchor whose pose stays fixed during
    /// translation optimization. Replaces any previously selected anchor.
    pub fn anchor(&mut self, keyframe_id: u64) {
        self.anchor = Some(keyframe_id);
    }

    /// Add a sequential odometry edge with weight `1.0`.
    pub fn add_sequential_edge(&mut self, from: u64, to: u64, measurement: SE3) {
        self.edges.push(PoseGraphEdge {
            from,
            to,
            measurement,
            kind: PoseGraphEdgeKind::Sequential,
            weight: 1.0,
        });
    }

    /// Append a loop-closure constraint as a graph edge. The verifier-derived
    /// inlier count is reused as the edge weight (clamped to a minimum of
    /// `1.0`) so loops with more inliers carry more pull on the solver.
    pub fn add_loop_closure_constraint(&mut self, constraint: &LoopClosureConstraint) {
        let weight = (constraint.inlier_count as f64).max(1.0);
        self.edges.push(PoseGraphEdge {
            from: constraint.from_keyframe_id,
            to: constraint.to_keyframe_id,
            measurement: constraint.relative_pose.clone(),
            kind: PoseGraphEdgeKind::LoopClosure,
            weight,
        });
    }

    /// Sum of squared edge translation residuals in world coordinates.
    /// Rotation residuals are ignored — this is a translation-only metric
    /// that matches what [`Self::optimize_translations_once`] minimizes.
    pub fn translation_cost(&self) -> f64 {
        let mut total = 0.0;
        for edge in &self.edges {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let displacement = expected_world_displacement(to, &edge.measurement);
            let actual = to.camera_center_world() - from.camera_center_world();
            let residual = actual - displacement;
            total += edge.weight * residual.norm_squared();
        }
        total
    }

    /// Solve a single Gauss-Newton step on the translation residuals while
    /// holding rotations fixed. With linear-in-translation residuals the
    /// "single step" is the exact least-squares optimum of the underlying
    /// linear system, not a Newton iteration that needs to be repeated.
    pub fn optimize_translations_once(
        &mut self,
    ) -> Result<PoseGraphOptimizationStep, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        let anchor_pose = self
            .poses
            .get(&anchor_id)
            .ok_or(PoseGraphError::MissingNode(anchor_id))?
            .clone();
        let anchor_center = anchor_pose.camera_center_world();
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }

        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if id == anchor_id {
                continue;
            }
            let next = node_index.len();
            node_index.insert(id, next);
        }
        let variable_count = node_index.len();
        if variable_count == 0 {
            return Err(PoseGraphError::NoVariables);
        }

        let row_count = self.edges.len() * 3;
        let column_count = variable_count * 3;
        let mut a = DMatrix::<f64>::zeros(row_count, column_count);
        let mut b = DVector::<f64>::zeros(row_count);

        for edge in &self.edges {
            if !self.poses.contains_key(&edge.from) {
                return Err(PoseGraphError::MissingNode(edge.from));
            }
            if !self.poses.contains_key(&edge.to) {
                return Err(PoseGraphError::MissingNode(edge.to));
            }
        }

        let cost_before = self.translation_cost();

        for (edge_index, edge) in self.edges.iter().enumerate() {
            let to_pose = &self.poses[&edge.to];
            let displacement = expected_world_displacement(to_pose, &edge.measurement);
            let mut rhs = displacement;
            let weight = edge.weight.sqrt();
            let row = edge_index * 3;

            if let Some(&j) = node_index.get(&edge.to) {
                for k in 0..3 {
                    a[(row + k, j * 3 + k)] += weight;
                }
            } else {
                rhs -= anchor_center.coords;
            }
            if let Some(&i) = node_index.get(&edge.from) {
                for k in 0..3 {
                    a[(row + k, i * 3 + k)] -= weight;
                }
            } else {
                rhs += anchor_center.coords;
            }
            for k in 0..3 {
                b[row + k] = weight * rhs[k];
            }
        }

        let ata = a.transpose() * &a;
        let atb = a.transpose() * &b;
        let solution = ata.lu().solve(&atb).ok_or(PoseGraphError::SingularSystem)?;

        let mut total_correction = 0.0;
        let mut max_correction: f64 = 0.0;
        for (&id, &i) in &node_index {
            let new_center = Point3::new(solution[i * 3], solution[i * 3 + 1], solution[i * 3 + 2]);
            let pose = self
                .poses
                .get_mut(&id)
                .ok_or(PoseGraphError::MissingNode(id))?;
            let old_center = pose.camera_center_world();
            let correction_norm = (new_center - old_center).norm();
            total_correction += correction_norm;
            if correction_norm > max_correction {
                max_correction = correction_norm;
            }
            let rotation_matrix = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            pose.world_to_camera.translation = -(rotation_matrix * new_center.coords);
        }

        let cost_after = self.translation_cost();
        let mean_translation_correction = if variable_count > 0 {
            total_correction / variable_count as f64
        } else {
            0.0
        };

        Ok(PoseGraphOptimizationStep {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            cost_before,
            cost_after,
            mean_translation_correction,
            max_translation_correction: max_correction,
        })
    }

    /// Sum of squared SE(3) residuals: r_e = log(meas_e⁻¹ · T_to · T_from⁻¹),
    /// weighted by `edge.weight`. Unlike [`Self::translation_cost`], this
    /// includes both the translation and rotation components of every edge.
    pub fn se3_cost(&self) -> f64 {
        self.robust_se3_cost(&RobustKernel::None)
    }

    /// Robust SE(3) cost: `Σ_e edge.weight · ρ(||r_e||²)` where `ρ` is the
    /// supplied [`RobustKernel`]. With [`RobustKernel::None`] this matches
    /// [`Self::se3_cost`].
    pub fn robust_se3_cost(&self, kernel: &RobustKernel) -> f64 {
        let mut total = 0.0;
        for edge in &self.edges {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
            let r = edge.measurement.inverse().compose(&predicted).log();
            total += edge.weight * kernel.cost(r.norm_squared());
        }
        total
    }

    /// Run a full SE(3) Gauss-Newton optimization with right-perturbation
    /// updates `T_i ← T_i · Exp(δ_i)`. Uses the first-order BCH approximation
    /// `J_r⁻¹(r) ≈ I`, so each edge contributes:
    ///
    /// - residual: `r_e = log(meas_e⁻¹ · T_to · T_from⁻¹)` (6-vector),
    /// - Jacobians: `∂r/∂δ_to = Ad(T_from)`, `∂r/∂δ_from = -Ad(T_from)`.
    ///
    /// The anchor pose is held fixed; all other poses are updated. Returns the
    /// per-iteration cost trace plus a `converged` flag derived from the
    /// configured tolerances.
    pub fn optimize_se3_iterative(
        &mut self,
        config: &PoseGraphSe3Config,
    ) -> Result<PoseGraphSe3Result, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if !self.poses.contains_key(&anchor_id) {
            return Err(PoseGraphError::MissingNode(anchor_id));
        }
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }
        for edge in &self.edges {
            if !self.poses.contains_key(&edge.from) {
                return Err(PoseGraphError::MissingNode(edge.from));
            }
            if !self.poses.contains_key(&edge.to) {
                return Err(PoseGraphError::MissingNode(edge.to));
            }
        }

        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if id == anchor_id {
                continue;
            }
            let next = node_index.len();
            node_index.insert(id, next);
        }
        let variable_count = node_index.len();
        if variable_count == 0 {
            return Err(PoseGraphError::NoVariables);
        }

        let kernel = config.robust_kernel;
        let initial_cost = self.robust_se3_cost(&kernel);
        let mut iterations: Vec<PoseGraphSe3IterationStats> =
            Vec::with_capacity(config.max_iterations);
        let mut converged = false;
        let mut current_cost = initial_cost;
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let dim = variable_count * 6;

        for iteration in 0..config.max_iterations {
            let mut h = DMatrix::<f64>::zeros(dim, dim);
            let mut g = DVector::<f64>::zeros(dim);

            for edge in &self.edges {
                let t_from = &self.poses[&edge.from].world_to_camera;
                let t_to = &self.poses[&edge.to].world_to_camera;
                let predicted = t_to.compose(&t_from.inverse());
                let r = edge.measurement.inverse().compose(&predicted).log();
                let ad_from = t_from.adjoint();
                let robust_weight = kernel.weight(r.norm_squared());
                let weight = edge.weight * robust_weight;
                let ata = ad_from.transpose() * ad_from;
                let atr = ad_from.transpose() * r;

                let i_from = node_index.get(&edge.from).copied();
                let i_to = node_index.get(&edge.to).copied();

                if let Some(j) = i_to {
                    add_block6(&mut h, j * 6, j * 6, weight, &ata);
                    add_segment6(&mut g, j * 6, weight, &atr);
                }
                if let Some(i) = i_from {
                    add_block6(&mut h, i * 6, i * 6, weight, &ata);
                    add_segment6(&mut g, i * 6, -weight, &atr);
                }
                if let (Some(j), Some(i)) = (i_to, i_from) {
                    let cross = -ata;
                    let cross_t = cross.transpose();
                    add_block6(&mut h, j * 6, i * 6, weight, &cross);
                    add_block6(&mut h, i * 6, j * 6, weight, &cross_t);
                }
            }

            let mut h_damped = h.clone();
            if lambda > 0.0 {
                for k in 0..dim {
                    h_damped[(k, k)] += lambda;
                }
            }
            let neg_g = -&g;
            let delta = solve_normal_equations(&h_damped, &neg_g)?;

            // Tentatively apply the step so we can evaluate the new cost.
            let mut max_step_norm: f64 = 0.0;
            let cost_before = current_cost;
            let saved_poses = if config.initial_lambda.is_some() {
                Some(self.poses.clone())
            } else {
                None
            };
            for (&id, &i) in &node_index {
                let block = i * 6;
                let mut xi = Vector6::<f64>::zeros();
                for k in 0..6 {
                    xi[k] = delta[block + k];
                }
                let step = xi.norm();
                if step > max_step_norm {
                    max_step_norm = step;
                }
                let pose = self
                    .poses
                    .get_mut(&id)
                    .ok_or(PoseGraphError::MissingNode(id))?;
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
            }

            let cost_after = self.robust_se3_cost(&kernel);
            let step_accepted = match config.initial_lambda {
                None => true,
                Some(_) => cost_after < cost_before,
            };

            if !step_accepted {
                if let Some(saved) = saved_poses {
                    self.poses = saved;
                }
                lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                iterations.push(PoseGraphSe3IterationStats {
                    iteration,
                    cost_before,
                    cost_after,
                    max_step_norm,
                    lambda,
                    step_accepted: false,
                });
                if lambda >= config.max_lambda {
                    // λ saturated without finding a downhill step → bail.
                    break;
                }
                continue;
            }

            iterations.push(PoseGraphSe3IterationStats {
                iteration,
                cost_before,
                cost_after,
                max_step_norm,
                lambda,
                step_accepted: true,
            });
            current_cost = cost_after;
            if config.initial_lambda.is_some() {
                lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
            }

            if max_step_norm < config.step_tolerance {
                converged = true;
                break;
            }
            if (cost_before - cost_after).abs() < config.cost_tolerance {
                converged = true;
                break;
            }
        }

        Ok(PoseGraphSe3Result {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }
}

/// Solve `H · x = b` preferring Cholesky (SPD path) and falling back to LU
/// for ill-conditioned or rank-deficient systems.
fn solve_normal_equations(
    h: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, PoseGraphError> {
    if let Some(chol) = h.clone().cholesky() {
        return Ok(chol.solve(b));
    }
    h.clone()
        .lu()
        .solve(b)
        .ok_or(PoseGraphError::SingularSystem)
}

fn add_block6(h: &mut DMatrix<f64>, row: usize, col: usize, weight: f64, block: &Matrix6<f64>) {
    for r in 0..6 {
        for c in 0..6 {
            h[(row + r, col + c)] += weight * block[(r, c)];
        }
    }
}

fn add_segment6(g: &mut DVector<f64>, start: usize, weight: f64, v: &Vector6<f64>) {
    for k in 0..6 {
        g[start + k] += weight * v[k];
    }
}

/// Compute the relative SE3 `previous_to_current` such that
/// `to_pose.world_to_camera == relative * from_pose.world_to_camera`. This is
/// the same convention used by [`PoseGraphEdge::measurement`].
pub fn relative_world_to_camera(from_pose: &Pose, to_pose: &Pose) -> SE3 {
    to_pose
        .world_to_camera
        .compose(&from_pose.world_to_camera.inverse())
}

/// Translation-only constraint on camera centers in world coordinates implied
/// by `measurement` together with `to_pose`'s rotation: `c_to - c_from`
/// equals this displacement.
fn expected_world_displacement(to_pose: &Pose, measurement: &SE3) -> nalgebra::Vector3<f64> {
    let rotation_matrix = to_pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    -(rotation_matrix.transpose() * measurement.translation)
}
