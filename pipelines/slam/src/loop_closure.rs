//! Loop-closure detection and geometric verification: candidate scan,
//! essential-matrix / PnP / hybrid verifiers, and constraint construction.

use super::*;

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
    /// Query keypoint index paired with an older-map landmark id by the
    /// appearance PnP correspondence builder. Empty for other sources.
    pub pnp_query_landmark_pairs: Vec<(usize, u64)>,
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
    /// The hybrid verifier ran both backends successfully, but the recovered
    /// essential-matrix and PnP relative poses disagreed beyond the configured
    /// translation-direction or rotation tolerances.
    PoseDisagreement,
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

    /// Confidence-aware variant: when `weights[i]` is high, correspondence
    /// `i` is preferred during RANSAC sampling (PROSAC-style). Default
    /// implementation falls back to the unweighted `verify` so existing
    /// implementors don't need to change. Verifiers backed by RANSAC can
    /// override to thread weights into `EssentialRansac::estimate_with_weights`.
    fn verify_with_weights(
        &self,
        correspondences: &[TwoViewCorrespondence],
        weights: Option<&[f32]>,
        camera: &Camera,
    ) -> LoopClosureVerification {
        let _ = weights;
        self.verify(correspondences, camera)
    }
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
        self.verify_with_weights(correspondences, None, camera)
    }

    fn verify_with_weights(
        &self,
        correspondences: &[TwoViewCorrespondence],
        weights: Option<&[f32]>,
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

        let relative_pose = match weights {
            Some(w) if w.len() == correspondences.len() => {
                self.estimator.estimate_with_scale_and_weights(
                    correspondences,
                    camera,
                    self.config.default_translation_scale,
                    w,
                )
            }
            _ => self.estimator.estimate_with_scale(
                correspondences,
                camera,
                self.config.default_translation_scale,
            ),
        };
        let Some(relative_pose) = relative_pose else {
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
        self.verify_impl(correspondences, keyframe_pose, camera, None)
    }

    /// Run PnP RANSAC with optional confidence-weighted sampling. This keeps the
    /// same acceptance thresholds as [`Self::verify`] but lets matchers such as
    /// LightGlue bias RANSAC toward high-confidence correspondences.
    pub fn verify_with_weights(
        &self,
        correspondences: &[Correspondence2D3D],
        keyframe_pose: &Pose,
        camera: &Camera,
        weights: Option<&[f32]>,
    ) -> LoopClosureVerification {
        self.verify_impl(correspondences, keyframe_pose, camera, weights)
    }

    fn verify_impl(
        &self,
        correspondences: &[Correspondence2D3D],
        keyframe_pose: &Pose,
        camera: &Camera,
        weights: Option<&[f32]>,
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

        let report = match weights {
            Some(weights)
                if weights.len() == correspondence_count
                    && weights
                        .iter()
                        .any(|value| value.is_finite() && *value > 0.0) =>
            {
                self.ransac
                    .estimate_with_weights(correspondences, camera, weights)
            }
            _ => self.ransac.estimate(correspondences, camera),
        };
        let Some(report) = report else {
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
            confidence: None,
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

    /// Lift this loop-closure edge into a [`PairwisePoseFactor`] so it can be
    /// added to a [`BundleAdjustment`] alongside reprojection residuals.
    ///
    /// The relative pose `T_rel = T_to · T_fromⁱ` (the constraint's
    /// `relative_pose`) is used verbatim as the BA measurement; `weight` is
    /// `1 / σ²` for an isotropic SE(3) measurement noise σ. A robust default is
    /// to scale from the verifier's inlier count, e.g.
    /// `weight = (inlier_count as f64) * base_weight`, but the choice is left
    /// to the caller because verifier output differs by backend (essential
    /// matrix vs PnP vs hybrid).
    pub fn to_pairwise_pose_factor(&self, weight: f64) -> PairwisePoseFactor {
        PairwisePoseFactor {
            keyframe_id_from: self.from_keyframe_id,
            keyframe_id_to: self.to_keyframe_id,
            measurement: Pose {
                world_to_camera: self.relative_pose.clone(),
            },
            weight,
        }
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

/// Convert a slice of verified [`LoopClosureConstraint`]s into BA-ready
/// [`PairwisePoseFactor`]s, all sharing the same scalar `weight`. Convenience
/// wrapper for the common case where every loop edge is treated as
/// equally-informative. For per-edge weighting (e.g. scaling by
/// `inlier_count`), call [`LoopClosureConstraint::to_pairwise_pose_factor`]
/// directly.
pub fn pairwise_pose_factors_from_loop_closures(
    constraints: &[LoopClosureConstraint],
    weight: f64,
) -> Vec<PairwisePoseFactor> {
    constraints
        .iter()
        .map(|c| c.to_pairwise_pose_factor(weight))
        .collect()
}

/// Run `verify_one` on every candidate whose `matched_keyframe_id` still
/// resolves in `map`. Candidates without a matching keyframe are silently
/// skipped (the matched keyframe may have been deleted between candidate
/// generation and verification). When `verify_one` returns `Some`, the
/// candidate is mutated in place: `geometrically_verified` is set to
/// `LoopClosureVerification::verified` and `verification` is set to the
/// returned struct. `None` skips writing.
///
/// This is the shared iteration shell between
/// [`verify_loop_closure_candidates`],
/// [`verify_loop_closure_candidates_pnp`], and
/// [`verify_loop_closure_candidates_hybrid`]. Each public wrapper supplies a
/// closure that builds correspondences and calls its backend; this helper
/// owns the candidate iteration and write-back so the public API stays
/// uniform when new backends are added.
fn verify_each_candidate<F>(
    candidates: &mut [LoopClosureCandidate],
    map: &VisualMap,
    mut verify_one: F,
) where
    F: FnMut(&Keyframe) -> Option<LoopClosureVerification>,
{
    for candidate in candidates.iter_mut() {
        let Some(keyframe) = map.keyframes.get(&candidate.matched_keyframe_id) else {
            continue;
        };
        let Some(verification) = verify_one(keyframe) else {
            continue;
        };
        candidate.geometrically_verified = verification.verified;
        candidate.verification = Some(verification);
    }
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
    verify_each_candidate(candidates, map, |keyframe| {
        let correspondences = correspondences_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
        );
        Some(verifier.verify(&correspondences, camera))
    });
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
    verify_each_candidate(candidates, map, |keyframe| {
        let keyframe_pose = keyframe.frame.pose.as_ref()?;
        let correspondences = correspondences_2d3d_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
            map,
        );
        Some(verifier.verify(&correspondences, keyframe_pose, camera))
    });
}

/// Configuration for [`HybridLoopClosureVerifier`]: maximum allowed
/// disagreement between the essential-matrix and PnP recovered poses before
/// the hybrid verifier rejects the candidate as inconsistent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridLoopClosureVerifierConfig {
    /// Maximum allowed angle (in radians) between the essential and PnP
    /// translation directions. Compared on unit vectors so essential's
    /// scale-up-to-translation ambiguity does not trigger spurious failures.
    pub max_translation_direction_disagreement_rad: f64,
    /// Maximum allowed rotation angle between the essential and PnP rotation
    /// components.
    pub max_rotation_disagreement_rad: f64,
}

impl Default for HybridLoopClosureVerifierConfig {
    fn default() -> Self {
        Self {
            max_translation_direction_disagreement_rad: 0.20,
            max_rotation_disagreement_rad: 0.20,
        }
    }
}

/// Loop-closure verifier that consults both the essential-matrix and PnP
/// backends and reports a consensus decision: the candidate is accepted iff
/// both verifiers accept it AND their recovered relative poses agree to
/// within the configured rotation / translation-direction tolerances. This
/// catches ambiguity where a 2D-2D essential fit looks plausible but
/// disagrees with the 3D map structure (or vice versa).
///
/// The combined [`LoopClosureVerification`] uses the PnP relative pose
/// (metric, no scale parameter), the minimum of both verifiers' inlier
/// counts (conservative), and reports both `mean_sampson_error` and
/// `mean_reprojection_error_px`. When either backend rejects, the failure
/// reason is propagated; if both pass but the poses disagree the failure
/// reason is [`LoopClosureVerificationFailureReason::PoseDisagreement`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HybridLoopClosureVerifier<R = PnPRansac> {
    pub essential: EssentialMatrixLoopClosureVerifier,
    pub pnp: PnPLoopClosureVerifier<R>,
    pub config: HybridLoopClosureVerifierConfig,
}

impl<R> HybridLoopClosureVerifier<R>
where
    R: RobustPoseEstimator,
{
    pub fn new(
        essential: EssentialMatrixLoopClosureVerifier,
        pnp: PnPLoopClosureVerifier<R>,
        config: HybridLoopClosureVerifierConfig,
    ) -> Self {
        Self {
            essential,
            pnp,
            config,
        }
    }
}

/// Run a [`HybridLoopClosureVerifier`] on every supplied candidate. For each
/// candidate this builds both 2D-2D and 2D-3D correspondences, runs the two
/// backends in turn, combines them into a consensus
/// [`LoopClosureVerification`], and writes the result back into
/// `verification` / `geometrically_verified` in place. Candidates whose
/// matched keyframe is no longer in `map` or whose stored keyframe pose is
/// missing are left untouched.
pub fn verify_loop_closure_candidates_hybrid<R>(
    candidates: &mut [LoopClosureCandidate],
    current_frame: &Frame,
    tracking: &TrackingResult,
    map: &VisualMap,
    camera: &Camera,
    verifier: &HybridLoopClosureVerifier<R>,
) where
    R: RobustPoseEstimator,
{
    verify_each_candidate(candidates, map, |keyframe| {
        let keyframe_pose = keyframe.frame.pose.as_ref()?;
        let two_view = correspondences_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
        );
        let pnp_corrs = correspondences_2d3d_for_loop_candidate(
            current_frame,
            &tracking.localization.inlier_query_indices,
            &tracking.localization.inlier_landmark_ids,
            keyframe,
            map,
        );
        let essential_v = verifier.essential.verify(&two_view, camera);
        let pnp_v = verifier.pnp.verify(&pnp_corrs, keyframe_pose, camera);
        Some(combine_hybrid_verifications(
            &essential_v,
            &pnp_v,
            &verifier.config,
        ))
    });
}

/// One keyframe's appearance in a pairwise loop-closure scan: the frame id is
/// what the produced [`LoopClosureCandidate`] uses for `query_frame_id` /
/// `matched_keyframe_id`. The keypoint and descriptor slices are typically
/// borrowed from a [`FeatureSet`] (see [`PairwiseKeyframeView::from_features`]).
#[derive(Debug, Clone, Copy)]
pub struct PairwiseKeyframeView<'a> {
    pub frame_id: u64,
    pub keypoints: &'a [Point2<f64>],
    pub descriptors: &'a [Vec<f32>],
}

impl<'a> PairwiseKeyframeView<'a> {
    pub fn from_features(frame_id: u64, features: &'a FeatureSet) -> Self {
        Self {
            frame_id,
            keypoints: &features.keypoints,
            descriptors: &features.descriptors,
        }
    }
}

/// Configuration for [`scan_pairwise_loop_closures`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairwiseLoopClosureScannerConfig {
    /// Minimum frame-id gap between two keyframes for a pair to be eligible.
    /// Pairs `(i, j)` with `keyframes[j].frame_id - keyframes[i].frame_id <
    /// min_keyframe_id_gap` are skipped, so adjacent keyframes (which share
    /// most of their tracks) are never confused for loops.
    pub min_keyframe_id_gap: u64,
    /// Minimum number of descriptor matches a pair needs before its
    /// correspondences are even handed to the verifier. Cheap reject for
    /// keyframes whose appearance has no overlap.
    pub min_matches: usize,
}

impl Default for PairwiseLoopClosureScannerConfig {
    fn default() -> Self {
        Self {
            min_keyframe_id_gap: 20,
            min_matches: 30,
        }
    }
}

/// Walk every keyframe pair `(i, j)` with `i < j` and a sufficient frame-id
/// gap, brute-force match descriptors, and run the verifier on the resulting
/// 2D-2D correspondences. Returns one [`LoopClosureCandidate`] per accepted
/// pair, with `verification` populated and `geometrically_verified = true`.
/// Rejected pairs are dropped (vs the `verify_loop_closure_candidates_*`
/// helpers which mutate caller-supplied candidates in place); this routine is
/// meant for the "I have a list of keyframes and want loop *detection*, not
/// just verification" case.
///
/// `query_frame_id` is the later keyframe and `matched_keyframe_id` is the
/// earlier one — the same convention `LoopClosureConstraint` builders use.
/// `shared_landmark_count` and `keyframe_observation_count` are filled with
/// inlier and total-match counts respectively (no map is consulted), so the
/// candidate is ready for [`LoopClosureConstraint::from_verified_candidate`].
pub fn scan_pairwise_loop_closures<M, V>(
    keyframes: &[PairwiseKeyframeView],
    matcher: &M,
    verifier: &V,
    camera: &Camera,
    config: &PairwiseLoopClosureScannerConfig,
) -> Vec<LoopClosureCandidate>
where
    M: Matcher,
    V: LoopClosureVerifier,
{
    let mut out = Vec::new();
    for i in 0..keyframes.len() {
        for j in (i + 1)..keyframes.len() {
            let from = &keyframes[i];
            let to = &keyframes[j];
            let gap = to.frame_id.saturating_sub(from.frame_id);
            if gap < config.min_keyframe_id_gap {
                continue;
            }
            let matches = matcher.match_descriptors(from.descriptors, to.descriptors);
            if matches.len() < config.min_matches {
                continue;
            }
            let mut correspondences: Vec<TwoViewCorrespondence> = Vec::with_capacity(matches.len());
            let mut weights: Vec<f32> = Vec::with_capacity(matches.len());
            let mut any_confidence = false;
            for m in &matches {
                let Some(prev) = from.keypoints.get(m.query_index) else {
                    continue;
                };
                let Some(curr) = to.keypoints.get(m.train_index) else {
                    continue;
                };
                correspondences.push(TwoViewCorrespondence {
                    previous_xy: *prev,
                    current_xy: *curr,
                });
                if let Some(c) = m.confidence {
                    any_confidence = true;
                    weights.push(c);
                } else {
                    weights.push(1.0);
                }
            }
            if correspondences.len() < config.min_matches {
                continue;
            }
            let weights_slice = if any_confidence {
                Some(weights.as_slice())
            } else {
                None
            };
            let verification =
                verifier.verify_with_weights(&correspondences, weights_slice, camera);
            if !verification.verified {
                continue;
            }
            out.push(LoopClosureCandidate {
                query_frame_id: to.frame_id,
                matched_keyframe_id: from.frame_id,
                shared_landmark_count: verification.inlier_count,
                query_inlier_count: verification.inlier_count,
                keyframe_observation_count: matches.len(),
                shared_landmark_ratio: verification.inlier_ratio,
                score: verification.score,
                geometrically_verified: true,
                verification: Some(verification),
                pnp_query_landmark_pairs: Vec::new(),
            });
        }
    }
    out
}

fn combine_hybrid_verifications(
    essential: &LoopClosureVerification,
    pnp: &LoopClosureVerification,
    config: &HybridLoopClosureVerifierConfig,
) -> LoopClosureVerification {
    // Inherit the minimum (conservative) inlier count and ratio so the
    // combined diagnostics never overstate either backend's evidence.
    let inlier_count = essential.inlier_count.min(pnp.inlier_count);
    let inlier_ratio = essential.inlier_ratio.min(pnp.inlier_ratio);
    let correspondence_count = essential.correspondence_count.min(pnp.correspondence_count);
    let score = essential.score.min(pnp.score);

    if !essential.verified {
        return LoopClosureVerification {
            verified: false,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: essential.mean_sampson_error,
            score,
            failure_reason: essential.failure_reason,
            relative_pose: pnp.relative_pose.clone(),
            mean_reprojection_error_px: pnp.mean_reprojection_error_px,
        };
    }
    if !pnp.verified {
        return LoopClosureVerification {
            verified: false,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: essential.mean_sampson_error,
            score,
            failure_reason: pnp.failure_reason,
            relative_pose: pnp.relative_pose.clone(),
            mean_reprojection_error_px: pnp.mean_reprojection_error_px,
        };
    }
    // Both verified — check pose agreement.
    let (Some(ess_pose), Some(pnp_pose)) =
        (essential.relative_pose.as_ref(), pnp.relative_pose.as_ref())
    else {
        return LoopClosureVerification {
            verified: false,
            correspondence_count,
            inlier_count,
            inlier_ratio,
            mean_sampson_error: essential.mean_sampson_error,
            score,
            failure_reason: Some(LoopClosureVerificationFailureReason::PoseDisagreement),
            relative_pose: pnp.relative_pose.clone(),
            mean_reprojection_error_px: pnp.mean_reprojection_error_px,
        };
    };

    let direction_disagreement =
        translation_direction_disagreement_rad(&ess_pose.translation, &pnp_pose.translation);
    let rotation_disagreement = ess_pose.rotation.rotation_to(&pnp_pose.rotation).angle();
    let agreement_ok = direction_disagreement <= config.max_translation_direction_disagreement_rad
        && rotation_disagreement <= config.max_rotation_disagreement_rad;

    let failure_reason = if agreement_ok {
        None
    } else {
        Some(LoopClosureVerificationFailureReason::PoseDisagreement)
    };
    LoopClosureVerification {
        verified: agreement_ok,
        correspondence_count,
        inlier_count,
        inlier_ratio,
        mean_sampson_error: essential.mean_sampson_error,
        score,
        failure_reason,
        relative_pose: Some(pnp_pose.clone()),
        mean_reprojection_error_px: pnp.mean_reprojection_error_px,
    }
}

fn translation_direction_disagreement_rad(
    a: &nalgebra::Vector3<f64>,
    b: &nalgebra::Vector3<f64>,
) -> f64 {
    let na = a.norm();
    let nb = b.norm();
    if na < 1.0e-9 || nb < 1.0e-9 {
        return 0.0;
    }
    let dir_a = a / na;
    let dir_b = b / nb;
    dir_a.dot(&dir_b).clamp(-1.0, 1.0).acos()
}

pub(crate) fn detect_loop_closure_candidates(
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
                pnp_query_landmark_pairs: Vec::new(),
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
