#![forbid(unsafe_code)]
//! Minimal online SLAM orchestration.
//!
//! This crate wires tracking and local mapping together. It is not a full SLAM
//! system: it can report lightweight loop-closure candidates, but global pose
//! graph optimization, dense mapping, and production bundle adjustment remain
//! outside this MVP layer.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use nalgebra::Point2;
use visloc_core::geometry::SE3;
use visloc_core::types::{Camera, Frame, Keyframe, Observation, VisualMap};
use visloc_localization::LocalizationPipeline;
use visloc_mapping::{
    AppliedMapUpdate, KeyframePolicy, LandmarkCandidate, LinearTriangulator, LocalMappingPipeline,
    LocalMappingResult, SimpleKeyframePolicy, Triangulator,
};
use visloc_tracking::{
    ConstantPoseMotionModel, FrameLocalizer, MotionModel, Tracker, TrackingConfig, TrackingResult,
};
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
    pub mean_sampson_error: f64,
    pub score: f64,
    pub failure_reason: Option<LoopClosureVerificationFailureReason>,
    /// Recovered relative pose (older keyframe → current frame) when the
    /// underlying RANSAC converged. `Some` even for non-`verified` cases as
    /// long as a pose was recovered; consult `verified` and `failure_reason`
    /// before consuming. Translation is scaled by
    /// [`LoopClosureVerifierConfig::default_translation_scale`].
    pub relative_pose: Option<SE3>,
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
        }
    }
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
    output.push_str("<table><thead><tr><th>query frame</th><th>matched keyframe</th><th>shared landmarks</th><th>ratio</th><th>score</th><th>verified</th><th>verifier inliers</th><th>verifier inlier ratio</th><th>mean Sampson</th><th>verifier score</th><th>failure</th></tr></thead><tbody>\n");
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
                    if verification.mean_sampson_error.is_finite() {
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
