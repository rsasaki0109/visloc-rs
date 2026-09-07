//! Bounded camera-centre alignment between independently gauged rig submaps.
//!
//! A rig frame is represented once in each local submap by its world-to-camera
//! [`Pose`].  The adapter turns the common rig-frame camera centres into the
//! [`SubmapPointMatch`] representation consumed by the landmark-only Sim(3)
//! verifier.  It also derives the local-gauge rotation from the two camera
//! attitudes and uses a bounded medoid consensus before invoking that
//! verifier.
//!
//! The input may contain a large trajectory, but all quadratic work is over
//! the retained boundary only.  Common rig-frame ids are sorted in ascending
//! order and, by default, the earliest `max_boundary_frames` are retained.
//! An opt-in uniform sampler can instead cover the whole common-id span while
//! remaining bounded.  This makes the adapter suitable for a moving seam
//! whose newest window is still being registered without allowing the cost of
//! a frame Cartesian product to grow with the full trajectories.
//!
//! Callers conventionally pass the newer submap window as `source_frames` and
//! the previous/Atlas-adjacent window as `target_frames`.  In that orientation,
//! the earliest common global rig-frame ids are the seam boundary by
//! construction unless the uniform boundary sampler is explicitly selected.
//!
//! The optional fixed-rotation and generic fallback paths are deliberately
//! wrapper-only and opt-in. They retain at most B = `max_boundary_frames`
//! matches (64 by default). Rotation consensus and median pairwise scene
//! scale require O(B²) temporary state, not an all-image matrix. The public
//! configuration permits larger B; callers must retain a suitable cap.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use visloc_core::geometry::{Pose, Sim3};

use crate::{
    estimate_submap_sim3_constraint, SubmapPointMatch, SubmapSim3AlignmentConfig,
    SubmapSim3Constraint, SubmapSim3Rejection, SubmapSim3RejectionReason,
};

/// Configuration for the bounded rig-frame seam adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSubmapAlignmentConfig {
    /// Maximum number of common rig frames retained for the seam.
    pub max_boundary_frames: usize,
    /// Policy used to select the bounded common-frame seam boundary.
    pub boundary_sampling: RigSubmapBoundarySampling,
    /// Angular radius of the independent local-gauge rotation consensus.
    pub rotation_consensus_deg: f64,
    /// Minimum fraction of retained rig-frame rotations in that consensus.
    pub min_rotation_consensus_ratio: f64,
    /// Existing robust 3D Sim(3) alignment and residual gates.
    pub alignment: SubmapSim3AlignmentConfig,
    /// Allow the rig-only fixed-rotation camera-centre fallback after the
    /// generic Sim(3) estimator rejects. Disabled by default because this
    /// fallback does not add independent landmark geometry.
    pub allow_fixed_rotation_fallback: bool,
    /// Use the orientation-constrained camera-centre estimator as the primary
    /// alignment method. This is diagnostic-only; a generic fallback is
    /// separately opt-in through `allow_generic_fallback_after_fixed_rotation`.
    pub prefer_fixed_rotation_alignment: bool,
    /// When fixed-rotation alignment is primary, allow the existing generic
    /// landmark-style Sim(3) estimator to run only after that primary attempt
    /// rejects. This is diagnostic-only and disabled by default so enabling
    /// the fixed-primary path never silently changes its acceptance policy.
    pub allow_generic_fallback_after_fixed_rotation: bool,
    /// In fixed-rotation primary/fallback paths, force unit scale and use a
    /// robust translation-only estimator. Generic landmark Sim(3) alignment
    /// is never affected by this diagnostic switch.
    pub force_fixed_rotation_unit_scale: bool,
}

impl Default for RigSubmapAlignmentConfig {
    fn default() -> Self {
        Self {
            max_boundary_frames: 64,
            boundary_sampling: RigSubmapBoundarySampling::Earliest,
            rotation_consensus_deg: 5.0,
            min_rotation_consensus_ratio: 0.6,
            alignment: SubmapSim3AlignmentConfig::default(),
            allow_fixed_rotation_fallback: false,
            prefer_fixed_rotation_alignment: false,
            allow_generic_fallback_after_fixed_rotation: false,
            force_fixed_rotation_unit_scale: false,
        }
    }
}

/// Deterministic policy for selecting the bounded common-frame seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigSubmapBoundarySampling {
    /// Preserve the historical behavior: retain the lowest sorted frame ids.
    Earliest,
    /// Retain endpoint-inclusive, evenly spaced ids across the full overlap.
    UniformAcrossOverlap,
}

/// Estimator path that produced (or rejected) a rig-submap alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigSubmapAlignmentMethod {
    /// The existing robust landmark-style Sim(3) estimator.
    Generic,
    /// The fixed-consensus orientation estimator was selected as primary.
    FixedRotationPrimary,
    /// The fixed-consensus orientation estimator was attempted after generic
    /// Sim(3) rejection.
    FixedRotationFallback,
    /// The fixed-consensus orientation estimator rejected first, then the
    /// generic landmark-style Sim(3) estimator accepted as a fallback.
    GenericFallback,
}

impl RigSubmapAlignmentMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::FixedRotationPrimary => "fixed-primary",
            Self::FixedRotationFallback => "fixed-fallback",
            Self::GenericFallback => "generic-fallback",
        }
    }
}

/// Why a rig-submap camera-centre seam was not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigSubmapAlignmentRejectionReason {
    /// The bounded seam or rotation-consensus settings are not finite or are
    /// outside their fail-closed domains.
    InvalidConfiguration,
    /// A source or target trajectory contains the same global rig-frame id
    /// more than once.
    DuplicateFrameId,
    /// A source or target pose (including its rotation or translation) is not
    /// finite.
    NonFinitePose,
    /// A finite pose produced a non-finite camera centre.
    NonFiniteCameraCenter,
    /// The retained common boundary is smaller than the Sim(3) minimum.
    InsufficientCommonFrames,
    /// The independent camera-attitude rotations do not form a large enough
    /// consensus cluster.
    LowRotationConsensus,
    /// The existing robust Sim(3) estimator rejected the camera-centre
    /// matches.  The detailed rejection is retained in the error.
    Sim3Rejected,
}

/// Side of a rig-frame input that supplied a malformed record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigSubmapInputSide {
    Source,
    Target,
}

/// Deterministic and auditable counts from a rig-submap alignment attempt.
///
/// Counts are populated on both success and rejection.  In particular,
/// `common_frame_count` is the full unique intersection while
/// `retained_frame_count` is the bounded seam actually passed to the Sim(3)
/// estimator.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSubmapAlignmentDiagnostics {
    pub source_frame_count: usize,
    pub target_frame_count: usize,
    pub source_unique_frame_count: usize,
    pub target_unique_frame_count: usize,
    pub common_frame_count: usize,
    pub retained_frame_count: usize,
    pub retained_frame_ids: Vec<u64>,
    pub rotation_candidate_count: usize,
    pub rotation_consensus_count: usize,
    pub rotation_consensus_ratio: f64,
    pub rotation_consensus_max_disagreement_deg: f64,
    pub rotation_consensus_threshold_deg: f64,
    pub rotation_medoid_frame_id: Option<u64>,
    pub camera_centre_match_count: usize,
    /// Which estimator path was selected or produced the rejection.
    pub alignment_method: Option<RigSubmapAlignmentMethod>,
    pub sim3_inlier_count: Option<usize>,
    pub sim3_inlier_ratio: Option<f64>,
    /// The first generic Sim(3) rejection, retained when the fixed-rotation
    /// fallback is attempted or used so the original gate remains auditable.
    pub generic_sim3_rejection: Option<Box<SubmapSim3Rejection>>,
    /// Rejection from the fixed-rotation primary estimator. This is retained
    /// when the optional generic fallback is attempted or accepted.
    pub fixed_rotation_rejection: Option<Box<SubmapSim3Rejection>>,
    /// Rejection from the generic estimator after fixed-rotation primary
    /// rejection, when that diagnostic fallback also fails.
    pub generic_fallback_rejection: Option<Box<SubmapSim3Rejection>>,
    /// Whether a secondary estimator was attempted after the primary path.
    pub fallback_attempted: bool,
    /// Whether that secondary estimator supplied the accepted constraint.
    pub fallback_used: bool,
    /// The fixed-rotation fallback rejection, if the generic-primary path's
    /// attempted fixed-rotation fallback also failed.
    pub fallback_rejection: Option<Box<SubmapSim3Rejection>>,
}

impl RigSubmapAlignmentDiagnostics {
    fn from_counts(
        source_frame_count: usize,
        target_frame_count: usize,
        source_unique_frame_count: usize,
        target_unique_frame_count: usize,
        common_frame_count: usize,
        retained_frame_ids: Vec<u64>,
        rotation_consensus_threshold_deg: f64,
    ) -> Self {
        Self {
            source_frame_count,
            target_frame_count,
            source_unique_frame_count,
            target_unique_frame_count,
            common_frame_count,
            retained_frame_count: retained_frame_ids.len(),
            retained_frame_ids,
            rotation_candidate_count: 0,
            rotation_consensus_count: 0,
            rotation_consensus_ratio: 0.0,
            rotation_consensus_max_disagreement_deg: 0.0,
            rotation_consensus_threshold_deg,
            rotation_medoid_frame_id: None,
            camera_centre_match_count: 0,
            alignment_method: None,
            sim3_inlier_count: None,
            sim3_inlier_ratio: None,
            generic_sim3_rejection: None,
            fixed_rotation_rejection: None,
            generic_fallback_rejection: None,
            fallback_attempted: false,
            fallback_used: false,
            fallback_rejection: None,
        }
    }
}

/// Typed rejection carrying the partial seam diagnostics and, when the
/// wrapper reached the robust estimator, its detailed rejection.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSubmapAlignmentRejection {
    pub reason: RigSubmapAlignmentRejectionReason,
    pub diagnostics: Box<RigSubmapAlignmentDiagnostics>,
    pub offending_side: Option<RigSubmapInputSide>,
    pub offending_frame_id: Option<u64>,
    /// Rejection from the primary robust estimator (generic or fixed).
    pub sim3_rejection: Option<Box<SubmapSim3Rejection>>,
    /// Rejection from the optional secondary estimator (fixed or generic).
    pub fallback_rejection: Option<Box<SubmapSim3Rejection>>,
}

/// Error alias using the conventional `Error` name for callers that prefer
/// `Result<_, RigSubmapAlignmentError>`.
pub type RigSubmapAlignmentError = RigSubmapAlignmentRejection;

impl fmt::Display for RigSubmapAlignmentRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.reason {
            RigSubmapAlignmentRejectionReason::InvalidConfiguration => write!(
                f,
                "rig-submap alignment has an invalid boundary or rotation-consensus configuration"
            ),
            RigSubmapAlignmentRejectionReason::DuplicateFrameId => {
                write!(f, "rig-submap seam contains a duplicate frame id")
            }
            RigSubmapAlignmentRejectionReason::NonFinitePose => {
                write!(f, "rig-submap seam contains a non-finite pose")
            }
            RigSubmapAlignmentRejectionReason::NonFiniteCameraCenter => {
                write!(f, "rig-submap seam contains a non-finite camera centre")
            }
            RigSubmapAlignmentRejectionReason::InsufficientCommonFrames => write!(
                f,
                "rig-submap seam has {} retained common frames",
                self.diagnostics.retained_frame_count
            ),
            RigSubmapAlignmentRejectionReason::LowRotationConsensus => write!(
                f,
                "rig-submap rotation consensus ratio {} is below the configured minimum",
                self.diagnostics.rotation_consensus_ratio,
            ),
            RigSubmapAlignmentRejectionReason::Sim3Rejected => {
                write!(f, "rig-submap Sim(3) alignment was rejected")
            }
        }
    }
}

impl Error for RigSubmapAlignmentRejection {}

/// A verified camera-centre seam constraint plus the bounded evidence used to
/// admit it.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSubmapAlignmentResult {
    pub constraint: SubmapSim3Constraint,
    pub diagnostics: RigSubmapAlignmentDiagnostics,
}

/// Estimate a `target local <- source local` Sim(3) from corresponding global
/// rig-frame poses in two independently reconstructed submap gauges.
///
/// Callers should pass the newer submap window as `source_frames` and the
/// previous/Atlas-adjacent window as `target_frames`; the earliest common ids
/// then identify the new-window seam boundary.
///
/// The source and target slices are indexed by global rig-frame id.  Duplicate
/// ids are rejected.  Their unique intersection is sorted by id and sampled
/// according to `config.boundary_sampling`, bounded by
/// `config.max_boundary_frames`, before any rotation consensus or Sim(3) work.
/// For each retained frame, the camera-centre pair is represented by a
/// [`SubmapPointMatch`] whose source and target ids are that global rig-frame
/// id.  The independent rotation candidate is exactly
/// `q_target_cw.inverse() * q_source_cw`.
pub fn estimate_rig_submap_sim3_constraint(
    source_submap_id: u64,
    target_submap_id: u64,
    source_frames: &[(u64, Pose)],
    target_frames: &[(u64, Pose)],
    config: &RigSubmapAlignmentConfig,
) -> Result<RigSubmapAlignmentResult, RigSubmapAlignmentRejection> {
    if !valid_config(config) {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::InvalidConfiguration,
            diagnostics: Box::new(RigSubmapAlignmentDiagnostics::from_counts(
                source_frames.len(),
                target_frames.len(),
                0,
                0,
                0,
                Vec::new(),
                config.rotation_consensus_deg,
            )),
            offending_side: None,
            offending_frame_id: None,
            sim3_rejection: None,
            fallback_rejection: None,
        });
    }

    let source_unique = unique_frame_ids(source_frames);
    let target_unique = unique_frame_ids(target_frames);
    let base_diagnostics = RigSubmapAlignmentDiagnostics::from_counts(
        source_frames.len(),
        target_frames.len(),
        source_unique.len(),
        target_unique.len(),
        0,
        Vec::new(),
        config.rotation_consensus_deg,
    );

    if source_unique.len() != source_frames.len() {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::DuplicateFrameId,
            diagnostics: Box::new(base_diagnostics),
            offending_side: Some(RigSubmapInputSide::Source),
            offending_frame_id: duplicate_frame_id(source_frames),
            sim3_rejection: None,
            fallback_rejection: None,
        });
    }
    if target_unique.len() != target_frames.len() {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::DuplicateFrameId,
            diagnostics: Box::new(base_diagnostics),
            offending_side: Some(RigSubmapInputSide::Target),
            offending_frame_id: duplicate_frame_id(target_frames),
            sim3_rejection: None,
            fallback_rejection: None,
        });
    }

    if let Some((side, frame_id)) = first_nonfinite_pose(source_frames, target_frames) {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::NonFinitePose,
            diagnostics: Box::new(base_diagnostics),
            offending_side: Some(side),
            offending_frame_id: Some(frame_id),
            sim3_rejection: None,
            fallback_rejection: None,
        });
    }

    if let Some((side, frame_id)) = first_nonfinite_camera_center(source_frames, target_frames) {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::NonFiniteCameraCenter,
            diagnostics: Box::new(base_diagnostics),
            offending_side: Some(side),
            offending_frame_id: Some(frame_id),
            sim3_rejection: None,
            fallback_rejection: None,
        });
    }

    let common_frame_ids = source_unique
        .keys()
        .filter(|frame_id| target_unique.contains_key(frame_id))
        .copied()
        .collect::<Vec<_>>();
    let retained_frame_ids = sample_boundary_frame_ids(
        &common_frame_ids,
        config.max_boundary_frames,
        config.boundary_sampling,
    );
    let mut diagnostics = RigSubmapAlignmentDiagnostics::from_counts(
        source_frames.len(),
        target_frames.len(),
        source_unique.len(),
        target_unique.len(),
        common_frame_ids.len(),
        retained_frame_ids.clone(),
        config.rotation_consensus_deg,
    );

    let required_common_frames = config.alignment.min_correspondences.max(3);
    if retained_frame_ids.len() < required_common_frames {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::InsufficientCommonFrames,
            diagnostics: Box::new(diagnostics),
            offending_side: None,
            offending_frame_id: None,
            sim3_rejection: None,
            fallback_rejection: None,
        });
    }

    let mut matches = Vec::with_capacity(retained_frame_ids.len());
    let mut rotations = Vec::with_capacity(retained_frame_ids.len());
    for frame_id in &retained_frame_ids {
        let source_pose = source_unique
            .get(frame_id)
            .expect("retained frame id came from source map");
        let target_pose = target_unique
            .get(frame_id)
            .expect("retained frame id came from target map");
        let source_center = source_pose.camera_center_world();
        let target_center = target_pose.camera_center_world();
        if !point_is_finite(&source_center) {
            diagnostics.camera_centre_match_count = matches.len();
            return Err(RigSubmapAlignmentRejection {
                reason: RigSubmapAlignmentRejectionReason::NonFiniteCameraCenter,
                diagnostics: Box::new(diagnostics),
                offending_side: Some(RigSubmapInputSide::Source),
                offending_frame_id: Some(*frame_id),
                sim3_rejection: None,
                fallback_rejection: None,
            });
        }
        if !point_is_finite(&target_center) {
            diagnostics.camera_centre_match_count = matches.len();
            return Err(RigSubmapAlignmentRejection {
                reason: RigSubmapAlignmentRejectionReason::NonFiniteCameraCenter,
                diagnostics: Box::new(diagnostics),
                offending_side: Some(RigSubmapInputSide::Target),
                offending_frame_id: Some(*frame_id),
                sim3_rejection: None,
                fallback_rejection: None,
            });
        }
        matches.push(SubmapPointMatch {
            source_landmark_id: *frame_id,
            target_landmark_id: *frame_id,
            source_point: source_center,
            target_point: target_center,
        });
        rotations.push(IndependentRotationCandidate {
            frame_id: *frame_id,
            rotation: target_pose.world_to_camera.rotation.inverse()
                * source_pose.world_to_camera.rotation,
        });
    }
    diagnostics.camera_centre_match_count = matches.len();
    diagnostics.rotation_candidate_count = rotations.len();

    let consensus = rotation_consensus(&rotations, config.rotation_consensus_deg);
    diagnostics.rotation_consensus_count = consensus.indices.len();
    diagnostics.rotation_consensus_ratio =
        consensus.indices.len() as f64 / rotations.len().max(1) as f64;
    diagnostics.rotation_consensus_max_disagreement_deg = consensus.max_disagreement_deg;
    diagnostics.rotation_medoid_frame_id = Some(consensus.medoid_frame_id);
    if !diagnostics.rotation_consensus_ratio.is_finite()
        || diagnostics.rotation_consensus_ratio < config.min_rotation_consensus_ratio
    {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::LowRotationConsensus,
            diagnostics: Box::new(diagnostics),
            offending_side: None,
            offending_frame_id: None,
            sim3_rejection: None,
            fallback_rejection: None,
        });
    }

    if config.prefer_fixed_rotation_alignment {
        diagnostics.alignment_method = Some(RigSubmapAlignmentMethod::FixedRotationPrimary);
        match estimate_fixed_rotation_fallback(
            source_submap_id,
            target_submap_id,
            &matches,
            &consensus.rotation,
            &config.alignment,
            config.force_fixed_rotation_unit_scale,
        ) {
            Ok(constraint) => {
                diagnostics.sim3_inlier_count = Some(constraint.inlier_match_indices.len());
                diagnostics.sim3_inlier_ratio = Some(constraint.inlier_ratio);
                return Ok(RigSubmapAlignmentResult {
                    constraint,
                    diagnostics,
                });
            }
            Err(rejection) => {
                diagnostics.sim3_inlier_count = Some(rejection.inlier_count);
                diagnostics.sim3_inlier_ratio = Some(rejection.inlier_ratio);
                diagnostics.fixed_rotation_rejection = Some(Box::new(rejection.clone()));
                if config.allow_generic_fallback_after_fixed_rotation {
                    diagnostics.fallback_attempted = true;
                    diagnostics.alignment_method = Some(RigSubmapAlignmentMethod::GenericFallback);
                    match estimate_submap_sim3_constraint(
                        source_submap_id,
                        target_submap_id,
                        &matches,
                        &consensus.rotation,
                        &config.alignment,
                    ) {
                        Ok(constraint) => {
                            diagnostics.fallback_used = true;
                            diagnostics.sim3_inlier_count =
                                Some(constraint.inlier_match_indices.len());
                            diagnostics.sim3_inlier_ratio = Some(constraint.inlier_ratio);
                            return Ok(RigSubmapAlignmentResult {
                                constraint,
                                diagnostics,
                            });
                        }
                        Err(generic_rejection) => {
                            diagnostics.sim3_inlier_count = Some(generic_rejection.inlier_count);
                            diagnostics.sim3_inlier_ratio = Some(generic_rejection.inlier_ratio);
                            diagnostics.generic_fallback_rejection =
                                Some(Box::new(generic_rejection.clone()));
                            return Err(RigSubmapAlignmentRejection {
                                reason: RigSubmapAlignmentRejectionReason::Sim3Rejected,
                                diagnostics: Box::new(diagnostics),
                                offending_side: None,
                                offending_frame_id: None,
                                sim3_rejection: Some(Box::new(rejection)),
                                fallback_rejection: Some(Box::new(generic_rejection)),
                            });
                        }
                    }
                }
                return Err(RigSubmapAlignmentRejection {
                    reason: RigSubmapAlignmentRejectionReason::Sim3Rejected,
                    diagnostics: Box::new(diagnostics),
                    offending_side: None,
                    offending_frame_id: None,
                    sim3_rejection: Some(Box::new(rejection)),
                    fallback_rejection: None,
                });
            }
        }
    }

    diagnostics.alignment_method = Some(RigSubmapAlignmentMethod::Generic);
    let constraint = match estimate_submap_sim3_constraint(
        source_submap_id,
        target_submap_id,
        &matches,
        &consensus.rotation,
        &config.alignment,
    ) {
        Ok(constraint) => constraint,
        Err(sim3_rejection) => {
            diagnostics.sim3_inlier_count = Some(sim3_rejection.inlier_count);
            diagnostics.sim3_inlier_ratio = Some(sim3_rejection.inlier_ratio);
            diagnostics.generic_sim3_rejection = Some(Box::new(sim3_rejection.clone()));
            if !config.allow_fixed_rotation_fallback {
                return Err(RigSubmapAlignmentRejection {
                    reason: RigSubmapAlignmentRejectionReason::Sim3Rejected,
                    diagnostics: Box::new(diagnostics),
                    offending_side: None,
                    offending_frame_id: None,
                    sim3_rejection: Some(Box::new(sim3_rejection)),
                    fallback_rejection: None,
                });
            }

            diagnostics.fallback_attempted = true;
            diagnostics.alignment_method = Some(RigSubmapAlignmentMethod::FixedRotationFallback);
            match estimate_fixed_rotation_fallback(
                source_submap_id,
                target_submap_id,
                &matches,
                &consensus.rotation,
                &config.alignment,
                config.force_fixed_rotation_unit_scale,
            ) {
                Ok(constraint) => {
                    diagnostics.fallback_used = true;
                    diagnostics.sim3_inlier_count = Some(constraint.inlier_match_indices.len());
                    diagnostics.sim3_inlier_ratio = Some(constraint.inlier_ratio);
                    constraint
                }
                Err(fallback_rejection) => {
                    diagnostics.fallback_rejection = Some(Box::new(fallback_rejection.clone()));
                    return Err(RigSubmapAlignmentRejection {
                        reason: RigSubmapAlignmentRejectionReason::Sim3Rejected,
                        diagnostics: Box::new(diagnostics),
                        offending_side: None,
                        offending_frame_id: None,
                        sim3_rejection: Some(Box::new(sim3_rejection)),
                        fallback_rejection: Some(Box::new(fallback_rejection)),
                    });
                }
            }
        }
    };
    diagnostics.sim3_inlier_count = Some(constraint.inlier_match_indices.len());
    diagnostics.sim3_inlier_ratio = Some(constraint.inlier_ratio);
    Ok(RigSubmapAlignmentResult {
        constraint,
        diagnostics,
    })
}

/// Estimate a camera-centre Sim(3) while keeping the already verified rig
/// rotation fixed.
///
/// This is intentionally private to the rig wrapper.  It is not a replacement
/// for landmark-based Sim(3): it is only useful when camera-centre geometry is
/// sufficient to recover scale and translation after the generic estimator
/// rejects a degenerate point cloud.  The retained seam is bounded by the
/// caller (64 by default). Each two-point hypothesis evaluates O(B) residuals,
/// and the leave-one-out stability check is O(B²). Median pairwise scene
/// scale uses O(B²) temporary state; hypothesis/refit state is O(B).
fn estimate_fixed_rotation_fallback(
    source_submap_id: u64,
    target_submap_id: u64,
    matches: &[SubmapPointMatch],
    rotation: &UnitQuaternion<f64>,
    config: &SubmapSim3AlignmentConfig,
    force_unit_scale: bool,
) -> Result<SubmapSim3Constraint, SubmapSim3Rejection> {
    if force_unit_scale {
        return estimate_fixed_rotation_unit_scale(
            source_submap_id,
            target_submap_id,
            matches,
            rotation,
            config,
        );
    }
    let count = matches.len();
    let required = config.min_correspondences.max(3);
    if count < required {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::TooFewCorrespondences,
            count,
        ));
    }
    if matches.iter().any(|point_match| {
        !point_is_finite(&point_match.source_point) || !point_is_finite(&point_match.target_point)
    }) {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::NonFinitePoint,
            count,
        ));
    }
    let unique_source = matches
        .iter()
        .map(|point_match| point_match.source_landmark_id)
        .collect::<BTreeSet<_>>();
    let unique_target = matches
        .iter()
        .map(|point_match| point_match.target_landmark_id)
        .collect::<BTreeSet<_>>();
    if unique_source.len() != count || unique_target.len() != count {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::NonUniqueCorrespondences,
            count,
        ));
    }

    let target_points = matches
        .iter()
        .map(|point_match| point_match.target_point)
        .collect::<Vec<_>>();
    let target_scene_scale = fallback_median_pairwise_distance(&target_points);
    if !target_scene_scale.is_finite() || target_scene_scale <= 1.0e-12 {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::InvalidTargetSceneScale,
            count,
        ));
    }
    let threshold = (target_scene_scale * config.max_inlier_residual_ratio).max(1.0e-9);
    let mut rng = StdRng::seed_from_u64(config.random_seed);
    let mut best_inliers = Vec::new();
    let mut best_mean = f64::INFINITY;

    for _ in 0..config.ransac_iterations.max(1) {
        let Some([first, second]) = sample_two(count, &mut rng) else {
            break;
        };
        let pair = [matches[first], matches[second]];
        let Some((scale, translation)) = fit_scale_translation_fixed_rotation(&pair, rotation)
        else {
            continue;
        };
        if !valid_fallback_scale(scale, config) {
            continue;
        }
        let transform = Sim3::new(*rotation, translation, scale);
        let residuals = matches
            .iter()
            .map(|point_match| {
                (transform.transform_point(&point_match.source_point) - point_match.target_point)
                    .norm()
            })
            .collect::<Vec<_>>();
        let inliers = residuals
            .iter()
            .enumerate()
            .filter_map(|(index, residual)| (*residual <= threshold).then_some(index))
            .collect::<Vec<_>>();
        let mean = if inliers.is_empty() {
            f64::INFINITY
        } else {
            inliers.iter().map(|index| residuals[*index]).sum::<f64>() / inliers.len() as f64
        };
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && mean < best_mean)
        {
            best_inliers = inliers;
            best_mean = mean;
        }
    }

    if best_inliers.is_empty() {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::NoRobustFit,
            count,
        ));
    }
    let mut rejection = fallback_rejection(SubmapSim3RejectionReason::TooFewInliers, count);
    rejection.inlier_count = best_inliers.len();
    rejection.inlier_ratio = best_inliers.len() as f64 / count as f64;
    if best_inliers.len() < config.min_inliers.max(3) {
        return Err(rejection);
    }
    if rejection.inlier_ratio < config.min_inlier_ratio {
        rejection.reason = SubmapSim3RejectionReason::LowInlierRatio;
        return Err(rejection);
    }

    let first_inliers = best_inliers
        .iter()
        .map(|&index| matches[index])
        .collect::<Vec<_>>();
    let Some((first_scale, first_translation)) =
        fit_scale_translation_fixed_rotation(&first_inliers, rotation)
    else {
        rejection.reason = SubmapSim3RejectionReason::NoRobustFit;
        return Err(rejection);
    };
    if !valid_fallback_scale(first_scale, config) {
        rejection.reason = SubmapSim3RejectionReason::ScaleOutOfBounds;
        return Err(rejection);
    }
    let first_transform = Sim3::new(*rotation, first_translation, first_scale);
    best_inliers = matches
        .iter()
        .enumerate()
        .filter_map(|(index, point_match)| {
            ((first_transform.transform_point(&point_match.source_point)
                - point_match.target_point)
                .norm()
                <= threshold)
                .then_some(index)
        })
        .collect();
    rejection.inlier_count = best_inliers.len();
    rejection.inlier_ratio = best_inliers.len() as f64 / count as f64;
    if best_inliers.len() < config.min_inliers.max(3) {
        rejection.reason = SubmapSim3RejectionReason::TooFewInliers;
        return Err(rejection);
    }
    if rejection.inlier_ratio < config.min_inlier_ratio {
        rejection.reason = SubmapSim3RejectionReason::LowInlierRatio;
        return Err(rejection);
    }

    let inliers = best_inliers
        .iter()
        .map(|&index| matches[index])
        .collect::<Vec<_>>();
    let Some((scale, translation)) = fit_scale_translation_fixed_rotation(&inliers, rotation)
    else {
        rejection.reason = SubmapSim3RejectionReason::NoRobustFit;
        return Err(rejection);
    };
    if !valid_fallback_scale(scale, config) {
        rejection.reason = SubmapSim3RejectionReason::ScaleOutOfBounds;
        return Err(rejection);
    }
    let transform = Sim3::new(*rotation, translation, scale);
    let mean_residual_ratio = inliers
        .iter()
        .map(|point_match| {
            (transform.transform_point(&point_match.source_point) - point_match.target_point).norm()
        })
        .sum::<f64>()
        / inliers.len() as f64
        / target_scene_scale;
    rejection.mean_residual_ratio = Some(mean_residual_ratio);
    if !mean_residual_ratio.is_finite() || mean_residual_ratio > config.max_mean_residual_ratio {
        rejection.reason = SubmapSim3RejectionReason::HighMeanResidual;
        return Err(rejection);
    }

    // The rotation is the independently verified consensus by construction.
    // Keep the diagnostic explicit so callers can distinguish this from a
    // generic estimator that happened to fit a similar orientation.
    let rotation_disagreement_deg = 0.0;
    rejection.rotation_disagreement_deg = Some(rotation_disagreement_deg);
    if rotation_disagreement_deg > config.max_rotation_disagreement_deg {
        rejection.reason = SubmapSim3RejectionReason::RotationInconsistent;
        return Err(rejection);
    }
    let leave_one_out_log_scale_mad =
        fixed_rotation_leave_one_out_log_scale_mad(&inliers, rotation, scale)
            .unwrap_or(f64::INFINITY);
    rejection.leave_one_out_log_scale_mad = Some(leave_one_out_log_scale_mad);
    if !leave_one_out_log_scale_mad.is_finite()
        || leave_one_out_log_scale_mad > config.max_leave_one_out_log_scale_mad
    {
        rejection.reason = SubmapSim3RejectionReason::UnstableLeaveOneOutScale;
        return Err(rejection);
    }

    let inlier_ratio = best_inliers.len() as f64 / count as f64;
    Ok(SubmapSim3Constraint {
        source_submap_id,
        target_submap_id,
        target_from_source: transform,
        correspondence_count: count,
        inlier_match_indices: best_inliers,
        inlier_ratio,
        mean_residual_ratio,
        rotation_disagreement_deg,
        leave_one_out_log_scale_mad,
        target_scene_scale,
    })
}

/// Estimate a fixed-rotation, unit-scale camera-centre constraint.
///
/// Every retained correspondence supplies one translation hypothesis.  Each
/// hypothesis is scored against the same bounded residual threshold, then the
/// best inlier set is refit by its translation mean and reclassified once.
/// The final scale and leave-one-out scale MAD are exactly one and zero, so
/// the existing inlier, residual, rotation, and scale gates remain the policy
/// that decides admission.  The retained seam is bounded by the caller, hence
/// hypothesis scoring uses O(B²) residual work in the worst case. Median
/// pairwise scene scale uses O(B²) temporary state; hypothesis state is O(B).
fn estimate_fixed_rotation_unit_scale(
    source_submap_id: u64,
    target_submap_id: u64,
    matches: &[SubmapPointMatch],
    rotation: &UnitQuaternion<f64>,
    config: &SubmapSim3AlignmentConfig,
) -> Result<SubmapSim3Constraint, SubmapSim3Rejection> {
    let count = matches.len();
    let required = config.min_correspondences.max(3);
    if count < required {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::TooFewCorrespondences,
            count,
        ));
    }
    if !rotation.coords.iter().all(|value| value.is_finite())
        || matches.iter().any(|point_match| {
            !point_is_finite(&point_match.source_point)
                || !point_is_finite(&point_match.target_point)
        })
    {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::NonFinitePoint,
            count,
        ));
    }
    let unique_source = matches
        .iter()
        .map(|point_match| point_match.source_landmark_id)
        .collect::<BTreeSet<_>>();
    let unique_target = matches
        .iter()
        .map(|point_match| point_match.target_landmark_id)
        .collect::<BTreeSet<_>>();
    if unique_source.len() != count || unique_target.len() != count {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::NonUniqueCorrespondences,
            count,
        ));
    }
    if !valid_fallback_scale(1.0, config) {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::ScaleOutOfBounds,
            count,
        ));
    }
    let target_scene_scale = fallback_median_pairwise_distance(
        &matches
            .iter()
            .map(|point_match| point_match.target_point)
            .collect::<Vec<_>>(),
    );
    if !target_scene_scale.is_finite() || target_scene_scale <= 1.0e-12 {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::InvalidTargetSceneScale,
            count,
        ));
    }
    let threshold = (target_scene_scale * config.max_inlier_residual_ratio).max(1.0e-9);
    let mut hypothesis_indices = (0..count).collect::<Vec<_>>();
    hypothesis_indices.sort_by_key(|&index| {
        (
            matches[index].source_landmark_id,
            matches[index].target_landmark_id,
            index,
        )
    });
    let mut best_inliers = Vec::new();
    let mut best_mean = f64::INFINITY;
    let mut best_hypothesis = (u64::MAX, u64::MAX, usize::MAX);
    for index in hypothesis_indices {
        let translation = matches[index].target_point.coords
            - rotation.transform_vector(&matches[index].source_point.coords);
        if !translation.iter().all(|value| value.is_finite()) {
            continue;
        }
        let transform = Sim3::new(*rotation, translation, 1.0);
        let residuals = matches
            .iter()
            .map(|point_match| {
                (transform.transform_point(&point_match.source_point) - point_match.target_point)
                    .norm()
            })
            .collect::<Vec<_>>();
        let inliers = residuals
            .iter()
            .enumerate()
            .filter_map(|(match_index, residual)| (*residual <= threshold).then_some(match_index))
            .collect::<Vec<_>>();
        let mean = if inliers.is_empty() {
            f64::INFINITY
        } else {
            inliers
                .iter()
                .map(|&match_index| residuals[match_index])
                .sum::<f64>()
                / inliers.len() as f64
        };
        let hypothesis_key = (
            matches[index].source_landmark_id,
            matches[index].target_landmark_id,
            index,
        );
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len()
                && (mean.total_cmp(&best_mean).is_lt()
                    || (mean.total_cmp(&best_mean).is_eq() && hypothesis_key < best_hypothesis)))
        {
            best_inliers = inliers;
            best_mean = mean;
            best_hypothesis = hypothesis_key;
        }
    }

    if best_inliers.is_empty() {
        return Err(fallback_rejection(
            SubmapSim3RejectionReason::NoRobustFit,
            count,
        ));
    }
    let mut rejection = fallback_rejection(SubmapSim3RejectionReason::TooFewInliers, count);
    rejection.inlier_count = best_inliers.len();
    rejection.inlier_ratio = best_inliers.len() as f64 / count as f64;
    if best_inliers.len() < config.min_inliers.max(3) {
        return Err(rejection);
    }
    if rejection.inlier_ratio < config.min_inlier_ratio {
        rejection.reason = SubmapSim3RejectionReason::LowInlierRatio;
        return Err(rejection);
    }

    let first_translation = mean_translation_fixed_rotation(&best_inliers, matches, rotation);
    if !first_translation.iter().all(|value| value.is_finite()) {
        rejection.reason = SubmapSim3RejectionReason::NoRobustFit;
        return Err(rejection);
    }
    let first_transform = Sim3::new(*rotation, first_translation, 1.0);
    best_inliers = matches
        .iter()
        .enumerate()
        .filter_map(|(index, point_match)| {
            ((first_transform.transform_point(&point_match.source_point)
                - point_match.target_point)
                .norm()
                <= threshold)
                .then_some(index)
        })
        .collect();
    rejection.inlier_count = best_inliers.len();
    rejection.inlier_ratio = best_inliers.len() as f64 / count as f64;
    if best_inliers.len() < config.min_inliers.max(3) {
        rejection.reason = SubmapSim3RejectionReason::TooFewInliers;
        return Err(rejection);
    }
    if rejection.inlier_ratio < config.min_inlier_ratio {
        rejection.reason = SubmapSim3RejectionReason::LowInlierRatio;
        return Err(rejection);
    }

    let translation = mean_translation_fixed_rotation(&best_inliers, matches, rotation);
    if !translation.iter().all(|value| value.is_finite()) {
        rejection.reason = SubmapSim3RejectionReason::NoRobustFit;
        return Err(rejection);
    }
    let transform = Sim3::new(*rotation, translation, 1.0);
    let mean_residual_ratio = best_inliers
        .iter()
        .map(|&index| {
            (transform.transform_point(&matches[index].source_point) - matches[index].target_point)
                .norm()
        })
        .sum::<f64>()
        / best_inliers.len() as f64
        / target_scene_scale;
    rejection.mean_residual_ratio = Some(mean_residual_ratio);
    if !mean_residual_ratio.is_finite() || mean_residual_ratio > config.max_mean_residual_ratio {
        rejection.reason = SubmapSim3RejectionReason::HighMeanResidual;
        return Err(rejection);
    }

    let rotation_disagreement_deg = 0.0;
    rejection.rotation_disagreement_deg = Some(rotation_disagreement_deg);
    if rotation_disagreement_deg > config.max_rotation_disagreement_deg {
        rejection.reason = SubmapSim3RejectionReason::RotationInconsistent;
        return Err(rejection);
    }
    let leave_one_out_log_scale_mad = 0.0;
    rejection.leave_one_out_log_scale_mad = Some(leave_one_out_log_scale_mad);
    if leave_one_out_log_scale_mad > config.max_leave_one_out_log_scale_mad {
        rejection.reason = SubmapSim3RejectionReason::UnstableLeaveOneOutScale;
        return Err(rejection);
    }

    let inlier_ratio = best_inliers.len() as f64 / count as f64;
    Ok(SubmapSim3Constraint {
        source_submap_id,
        target_submap_id,
        target_from_source: transform,
        correspondence_count: count,
        inlier_match_indices: best_inliers,
        inlier_ratio,
        mean_residual_ratio,
        rotation_disagreement_deg,
        leave_one_out_log_scale_mad,
        target_scene_scale,
    })
}

fn mean_translation_fixed_rotation(
    indices: &[usize],
    matches: &[SubmapPointMatch],
    rotation: &UnitQuaternion<f64>,
) -> Vector3<f64> {
    indices.iter().fold(Vector3::zeros(), |sum, &index| {
        sum + matches[index].target_point.coords
            - rotation.transform_vector(&matches[index].source_point.coords)
    }) / indices.len().max(1) as f64
}

fn fallback_rejection(
    reason: SubmapSim3RejectionReason,
    correspondence_count: usize,
) -> SubmapSim3Rejection {
    SubmapSim3Rejection {
        reason,
        correspondence_count,
        inlier_count: 0,
        inlier_ratio: 0.0,
        mean_residual_ratio: None,
        rotation_disagreement_deg: None,
        leave_one_out_log_scale_mad: None,
    }
}

fn fit_scale_translation_fixed_rotation(
    matches: &[SubmapPointMatch],
    rotation: &UnitQuaternion<f64>,
) -> Option<(f64, Vector3<f64>)> {
    if matches.len() < 2 {
        return None;
    }
    let source_mean = matches.iter().fold(Vector3::zeros(), |sum, point_match| {
        sum + point_match.source_point.coords
    }) / matches.len() as f64;
    let target_mean = matches.iter().fold(Vector3::zeros(), |sum, point_match| {
        sum + point_match.target_point.coords
    }) / matches.len() as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for point_match in matches {
        let source = rotation * (point_match.source_point.coords - source_mean);
        let target = point_match.target_point.coords - target_mean;
        numerator += source.dot(&target);
        denominator += source.norm_squared();
    }
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 1.0e-12 {
        return None;
    }
    let scale = numerator / denominator;
    let translation = target_mean - scale * (rotation * source_mean);
    (scale.is_finite() && translation.iter().all(|value| value.is_finite()))
        .then_some((scale, translation))
}

fn valid_fallback_scale(scale: f64, config: &SubmapSim3AlignmentConfig) -> bool {
    scale.is_finite() && scale >= config.min_scale && scale <= config.max_scale
}

fn sample_two(count: usize, rng: &mut StdRng) -> Option<[usize; 2]> {
    if count < 2 {
        return None;
    }
    for _ in 0..64 {
        let first = rng.gen_range(0..count);
        let second = rng.gen_range(0..count);
        if first != second {
            return Some([first, second]);
        }
    }
    None
}

fn fallback_median_pairwise_distance(points: &[Point3<f64>]) -> f64 {
    let mut distances = Vec::new();
    for first in 0..points.len() {
        for second in (first + 1)..points.len() {
            let distance = (points[first] - points[second]).norm();
            if distance.is_finite() && distance > 0.0 {
                distances.push(distance);
            }
        }
    }
    fallback_median(distances).unwrap_or(0.0)
}

fn fixed_rotation_leave_one_out_log_scale_mad(
    matches: &[SubmapPointMatch],
    rotation: &UnitQuaternion<f64>,
    reference_scale: f64,
) -> Option<f64> {
    if matches.len() < 4 || !reference_scale.is_finite() || reference_scale <= 0.0 {
        return None;
    }
    let mut deviations = Vec::with_capacity(matches.len());
    for omitted in 0..matches.len() {
        let kept = matches
            .iter()
            .enumerate()
            .filter_map(|(index, point_match)| (index != omitted).then_some(*point_match))
            .collect::<Vec<_>>();
        let (scale, _) = fit_scale_translation_fixed_rotation(&kept, rotation)?;
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        deviations.push((scale / reference_scale).ln().abs());
    }
    fallback_median(deviations)
}

fn fallback_median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

/// Alias emphasizing that the returned Sim(3) is built from rig camera
/// centres.  It has the same bounded implementation and result type as
/// [`estimate_rig_submap_sim3_constraint`].
pub fn estimate_rig_submap_camera_centre_constraint(
    source_submap_id: u64,
    target_submap_id: u64,
    source_frames: &[(u64, Pose)],
    target_frames: &[(u64, Pose)],
    config: &RigSubmapAlignmentConfig,
) -> Result<RigSubmapAlignmentResult, RigSubmapAlignmentRejection> {
    estimate_rig_submap_sim3_constraint(
        source_submap_id,
        target_submap_id,
        source_frames,
        target_frames,
        config,
    )
}

#[derive(Debug, Clone, Copy)]
struct IndependentRotationCandidate {
    frame_id: u64,
    rotation: UnitQuaternion<f64>,
}

#[derive(Debug, Clone)]
struct RotationConsensus {
    rotation: UnitQuaternion<f64>,
    medoid_frame_id: u64,
    indices: Vec<usize>,
    max_disagreement_deg: f64,
}

/// Select the largest angular cluster and use its deterministic medoid as the
/// independent rotation.  The pairwise matrix is computed once, so this is
/// O(B²) in both rotations and angular evaluations.
fn rotation_consensus(
    candidates: &[IndependentRotationCandidate],
    radius_deg: f64,
) -> RotationConsensus {
    debug_assert!(!candidates.is_empty());
    let count = candidates.len();
    let radius = radius_deg.max(0.0).to_radians();
    let mut distances = vec![vec![0.0; count]; count];
    for i in 0..count {
        for j in (i + 1)..count {
            let distance = candidates[i]
                .rotation
                .rotation_to(&candidates[j].rotation)
                .angle();
            distances[i][j] = distance;
            distances[j][i] = distance;
        }
    }

    let mut medoid_index = 0;
    let mut medoid_support = 0;
    let mut medoid_distance_sum = f64::INFINITY;
    let mut medoid_cluster = Vec::new();
    for candidate_index in 0..count {
        let cluster = (0..count)
            .filter(|&other_index| distances[candidate_index][other_index] <= radius)
            .collect::<Vec<_>>();
        let support = cluster.len();
        let distance_sum = cluster
            .iter()
            .map(|&other_index| distances[candidate_index][other_index])
            .sum::<f64>();
        let frame_id = candidates[candidate_index].frame_id;
        let current_frame_id = candidates[medoid_index].frame_id;
        let better = support > medoid_support
            || (support == medoid_support
                && (distance_sum < medoid_distance_sum
                    || (distance_sum == medoid_distance_sum && frame_id < current_frame_id)));
        if better {
            medoid_index = candidate_index;
            medoid_support = support;
            medoid_distance_sum = distance_sum;
            medoid_cluster = cluster;
        }
    }
    if medoid_cluster.is_empty() {
        medoid_cluster.push(medoid_index);
    }
    // The input is sorted by frame id before this function is called.  Keep
    // the cluster in that order so the inlier evidence is reproducible.
    medoid_cluster.sort_unstable();
    let max_disagreement_deg = medoid_cluster
        .iter()
        .map(|&index| distances[medoid_index][index])
        .fold(0.0_f64, f64::max)
        .to_degrees();
    RotationConsensus {
        rotation: candidates[medoid_index].rotation,
        medoid_frame_id: candidates[medoid_index].frame_id,
        indices: medoid_cluster,
        max_disagreement_deg,
    }
}

/// Select at most `max_boundary_frames` from a sorted, unique overlap.
///
/// The intersection itself is built by the caller in O(common) map scans;
/// this pass is O(common + B) time and O(B) output state.  Integer positions
/// are used for the uniform policy so the endpoints are exact and no sampled
/// id can be duplicated when the cap is smaller than the overlap.
fn sample_boundary_frame_ids(
    common_frame_ids: &[u64],
    max_boundary_frames: usize,
    sampling: RigSubmapBoundarySampling,
) -> Vec<u64> {
    if common_frame_ids.is_empty() || max_boundary_frames == 0 {
        return Vec::new();
    }
    if common_frame_ids.len() <= max_boundary_frames {
        return common_frame_ids.to_vec();
    }
    match sampling {
        RigSubmapBoundarySampling::Earliest => common_frame_ids[..max_boundary_frames].to_vec(),
        RigSubmapBoundarySampling::UniformAcrossOverlap => {
            if max_boundary_frames == 1 {
                return vec![common_frame_ids[0]];
            }
            let span = (common_frame_ids.len() - 1) as u128;
            let intervals = (max_boundary_frames - 1) as u128;
            (0..max_boundary_frames)
                .map(|sample_index| {
                    let position = (sample_index as u128 * span / intervals) as usize;
                    common_frame_ids[position]
                })
                .collect()
        }
    }
}

fn unique_frame_ids(frames: &[(u64, Pose)]) -> BTreeMap<u64, &Pose> {
    let mut unique = BTreeMap::new();
    for (frame_id, pose) in frames {
        unique.entry(*frame_id).or_insert(pose);
    }
    unique
}

fn valid_config(config: &RigSubmapAlignmentConfig) -> bool {
    config.max_boundary_frames > 0
        && config.rotation_consensus_deg.is_finite()
        && config.rotation_consensus_deg > 0.0
        && config.rotation_consensus_deg <= 180.0
        && config.min_rotation_consensus_ratio.is_finite()
        && config.min_rotation_consensus_ratio > 0.0
        && config.min_rotation_consensus_ratio <= 1.0
}

fn duplicate_frame_id(frames: &[(u64, Pose)]) -> Option<u64> {
    let mut seen = BTreeSet::new();
    frames
        .iter()
        .find_map(|(frame_id, _)| (!seen.insert(*frame_id)).then_some(*frame_id))
}

fn first_nonfinite_pose(
    source_frames: &[(u64, Pose)],
    target_frames: &[(u64, Pose)],
) -> Option<(RigSubmapInputSide, u64)> {
    source_frames
        .iter()
        .find_map(|(frame_id, pose)| {
            (!pose_is_finite(pose)).then_some((RigSubmapInputSide::Source, *frame_id))
        })
        .or_else(|| {
            target_frames.iter().find_map(|(frame_id, pose)| {
                (!pose_is_finite(pose)).then_some((RigSubmapInputSide::Target, *frame_id))
            })
        })
}

fn first_nonfinite_camera_center(
    source_frames: &[(u64, Pose)],
    target_frames: &[(u64, Pose)],
) -> Option<(RigSubmapInputSide, u64)> {
    source_frames
        .iter()
        .find_map(|(frame_id, pose)| {
            (!point_is_finite(&pose.camera_center_world()))
                .then_some((RigSubmapInputSide::Source, *frame_id))
        })
        .or_else(|| {
            target_frames.iter().find_map(|(frame_id, pose)| {
                (!point_is_finite(&pose.camera_center_world()))
                    .then_some((RigSubmapInputSide::Target, *frame_id))
            })
        })
}

fn pose_is_finite(pose: &Pose) -> bool {
    pose.world_to_camera
        .rotation
        .coords
        .iter()
        .all(|value| value.is_finite())
        && pose
            .world_to_camera
            .translation
            .iter()
            .all(|value| value.is_finite())
}

fn point_is_finite(point: &Point3<f64>) -> bool {
    point.coords.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::Sim3;

    type RigFrames = Vec<(u64, Pose)>;

    fn pose_from_center(rotation: UnitQuaternion<f64>, center: Point3<f64>) -> Pose {
        Pose::from_world_to_camera(rotation, -rotation.transform_vector(&center.coords))
    }

    fn fixture(count: usize) -> (RigFrames, RigFrames, Sim3) {
        let truth = Sim3::new(
            UnitQuaternion::from_euler_angles(0.09, -0.13, 0.2),
            Vector3::new(1.3, -0.7, 0.5),
            2.4,
        );
        let mut source = Vec::with_capacity(count);
        let mut target = Vec::with_capacity(count);
        for index in 0..count {
            let frame_id = index as u64;
            let centre = Point3::new(
                index as f64 * 0.17,
                (index as f64 * 0.31).sin() * 0.7,
                (index as f64 * 0.21).cos() * 0.5,
            );
            let source_rotation = UnitQuaternion::from_euler_angles(
                index as f64 * 0.01,
                index as f64 * -0.007,
                index as f64 * 0.013,
            );
            let target_center = truth.transform_point(&centre);
            let target_rotation = source_rotation * truth.rotation.inverse();
            source.push((frame_id, pose_from_center(source_rotation, centre)));
            target.push((frame_id, pose_from_center(target_rotation, target_center)));
        }
        // Deliberately scramble input order: the adapter's result must still
        // use the ascending frame-id seam boundary.
        source.reverse();
        target.rotate_left((count / 3).min(count));
        (source, target, truth)
    }

    fn unit_scale_fixture(count: usize) -> (RigFrames, RigFrames, Sim3) {
        let truth = Sim3::new(
            UnitQuaternion::from_euler_angles(0.09, -0.13, 0.2),
            Vector3::new(1.3, -0.7, 0.5),
            1.0,
        );
        let mut source = Vec::with_capacity(count);
        let mut target = Vec::with_capacity(count);
        for index in 0..count {
            let frame_id = index as u64;
            let centre = Point3::new(
                index as f64 * 0.17,
                (index as f64 * 0.31).sin() * 0.7,
                (index as f64 * 0.21).cos() * 0.5,
            );
            let source_rotation = UnitQuaternion::from_euler_angles(
                index as f64 * 0.01,
                index as f64 * -0.007,
                index as f64 * 0.013,
            );
            let target_center = truth.transform_point(&centre);
            let target_rotation = source_rotation * truth.rotation.inverse();
            source.push((frame_id, pose_from_center(source_rotation, centre)));
            target.push((frame_id, pose_from_center(target_rotation, target_center)));
        }
        source.reverse();
        target.rotate_left((count / 3).min(count));
        (source, target, truth)
    }

    fn permissive_config() -> RigSubmapAlignmentConfig {
        RigSubmapAlignmentConfig {
            alignment: SubmapSim3AlignmentConfig {
                min_correspondences: 3,
                min_inliers: 3,
                min_inlier_ratio: 0.5,
                max_inlier_residual_ratio: 0.1,
                max_mean_residual_ratio: 0.1,
                max_rotation_disagreement_deg: 10.0,
                max_leave_one_out_log_scale_mad: 0.1,
                min_second_to_first_singular_ratio: 0.0,
                ransac_iterations: 512,
                ..SubmapSim3AlignmentConfig::default()
            },
            ..RigSubmapAlignmentConfig::default()
        }
    }

    fn collinear_fixture(count: usize) -> (RigFrames, RigFrames, Sim3) {
        let truth = Sim3::new(
            UnitQuaternion::from_euler_angles(0.11, -0.17, 0.23),
            Vector3::new(1.1, -0.6, 0.4),
            2.4,
        );
        let mut source = Vec::with_capacity(count);
        let mut target = Vec::with_capacity(count);
        for index in 0..count {
            let frame_id = index as u64;
            let centre = Point3::new(index as f64 * 0.4, 0.0, 0.0);
            let source_rotation = UnitQuaternion::from_euler_angles(
                index as f64 * 0.013,
                index as f64 * -0.009,
                index as f64 * 0.007,
            );
            let target_center = truth.transform_point(&centre);
            let target_rotation = source_rotation * truth.rotation.inverse();
            source.push((frame_id, pose_from_center(source_rotation, centre)));
            target.push((frame_id, pose_from_center(target_rotation, target_center)));
        }
        (source, target, truth)
    }

    fn fixed_rotation_mismatch_fixture(
        count: usize,
        rotation_offset_deg: f64,
    ) -> (RigFrames, RigFrames, Sim3) {
        let (source, mut target, truth) = fixture(count);
        let offset =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), rotation_offset_deg.to_radians());
        let consensus_rotation = truth.rotation * offset;
        for (frame_id, target_pose) in &mut target {
            let target_center = target_pose.camera_center_world();
            let source_rotation = source
                .iter()
                .find(|(source_frame_id, _)| source_frame_id == frame_id)
                .expect("fixture source frame id")
                .1
                .world_to_camera
                .rotation;
            let target_rotation = source_rotation * consensus_rotation.inverse();
            *target_pose = pose_from_center(target_rotation, target_center);
        }
        (source, target, truth)
    }

    fn fallback_test_config() -> RigSubmapAlignmentConfig {
        let mut config = permissive_config();
        config.alignment.min_correspondences = 8;
        config.alignment.min_inliers = 8;
        config.alignment.min_second_to_first_singular_ratio = 0.01;
        config.allow_fixed_rotation_fallback = false;
        config
    }

    #[test]
    fn known_sim3_is_recovered_from_rig_camera_centres() {
        let (source, target, truth) = fixture(24);
        let result =
            estimate_rig_submap_sim3_constraint(11, 17, &source, &target, &permissive_config())
                .expect("known Sim3 should pass");
        assert_eq!(result.constraint.source_submap_id, 11);
        assert_eq!(result.constraint.target_submap_id, 17);
        assert_eq!(
            result.diagnostics.alignment_method,
            Some(RigSubmapAlignmentMethod::Generic)
        );
        assert!((result.constraint.target_from_source.scale - truth.scale).abs() < 1.0e-9);
        assert!(
            (result.constraint.target_from_source.translation - truth.translation).norm() < 1.0e-9
        );
        assert!(
            result
                .constraint
                .target_from_source
                .rotation
                .rotation_to(&truth.rotation)
                .angle()
                < 1.0e-9
        );
        assert_eq!(
            result.diagnostics.retained_frame_ids,
            (0..24).collect::<Vec<_>>()
        );
    }

    #[test]
    fn centre_and_rotation_outliers_are_rejected_by_the_two_consensus_stages() {
        let (source, mut target, truth) = fixture(30);
        for index in [2_usize, 5, 8] {
            target[index].1 = pose_from_center(
                UnitQuaternion::from_euler_angles(1.2, -0.4, 0.7),
                Point3::new(40.0 + index as f64, -30.0, 20.0),
            );
        }
        let result =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &permissive_config())
                .expect("minority outliers should be robustly rejected");
        assert!(result.diagnostics.rotation_consensus_ratio > 0.8);
        assert!(result.constraint.inlier_ratio > 0.8);
        assert!((result.constraint.target_from_source.scale - truth.scale).abs() < 1.0e-6);
        assert!(
            (result.constraint.target_from_source.translation - truth.translation).norm() < 1.0e-6
        );
    }

    #[test]
    fn duplicate_ids_are_rejected_before_intersection() {
        let (mut source, target, _) = fixture(12);
        source.push(
            source
                .iter()
                .find(|(frame_id, _)| *frame_id == 0)
                .unwrap()
                .clone(),
        );
        let error =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &permissive_config())
                .unwrap_err();
        assert_eq!(
            error.reason,
            RigSubmapAlignmentRejectionReason::DuplicateFrameId
        );
        assert_eq!(error.offending_side, Some(RigSubmapInputSide::Source));
        assert_eq!(error.offending_frame_id, Some(0));
    }

    #[test]
    fn earliest_boundary_cap_is_sorted_and_deterministic() {
        let (source, target, _) = fixture(20);
        let mut config = permissive_config();
        config.max_boundary_frames = 5;
        assert_eq!(
            config.boundary_sampling,
            RigSubmapBoundarySampling::Earliest
        );
        let first = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("five-frame seam should pass");
        let second = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("same input should be deterministic");
        assert_eq!(first, second);
        assert_eq!(first.diagnostics.retained_frame_ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn uniform_boundary_cap_is_endpoint_inclusive_and_deterministic() {
        let (source, target, _) = fixture(20);
        let mut config = permissive_config();
        config.max_boundary_frames = 5;
        config.boundary_sampling = RigSubmapBoundarySampling::UniformAcrossOverlap;
        let first = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("uniform five-frame seam should pass");
        let second = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("same uniform input should be deterministic");
        assert_eq!(first, second);
        assert_eq!(first.diagnostics.retained_frame_ids, vec![0, 4, 9, 14, 19]);
    }

    #[test]
    fn uniform_boundary_sampling_has_no_duplicates_on_sparse_ids() {
        let common = vec![10, 20, 30, 100, 200, 500, 900];
        assert_eq!(
            sample_boundary_frame_ids(&common, 4, RigSubmapBoundarySampling::UniformAcrossOverlap,),
            vec![10, 30, 200, 900]
        );
        assert_eq!(
            sample_boundary_frame_ids(&common, 1, RigSubmapBoundarySampling::UniformAcrossOverlap,),
            vec![10]
        );
        assert_eq!(
            sample_boundary_frame_ids(&common, 32, RigSubmapBoundarySampling::UniformAcrossOverlap,),
            common
        );
    }

    #[test]
    fn low_rotation_consensus_is_typed() {
        let (source, mut target, _) = fixture(12);
        for (index, (_, target_pose)) in target.iter_mut().enumerate() {
            let centre = target_pose.camera_center_world();
            let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, (index as f64 + 1.0) * 0.4);
            *target_pose = pose_from_center(rotation, centre);
        }
        let mut config = permissive_config();
        config.min_rotation_consensus_ratio = 0.9;
        let error =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config).unwrap_err();
        assert_eq!(
            error.reason,
            RigSubmapAlignmentRejectionReason::LowRotationConsensus
        );
    }

    #[test]
    fn invalid_boundary_and_rotation_configurations_fail_closed_at_entry() {
        let (source, target, _) = fixture(3);
        let mut invalid_configs = Vec::new();

        let mut config = permissive_config();
        config.max_boundary_frames = 0;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.rotation_consensus_deg = 0.0;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.rotation_consensus_deg = 180.1;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.rotation_consensus_deg = f64::NAN;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.min_rotation_consensus_ratio = 0.0;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.min_rotation_consensus_ratio = 1.1;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.min_rotation_consensus_ratio = f64::NAN;
        invalid_configs.push(config);

        for config in invalid_configs {
            let error = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
                .expect_err("invalid configuration must reject before alignment");
            assert_eq!(
                error.reason,
                RigSubmapAlignmentRejectionReason::InvalidConfiguration
            );
            assert!(error.to_string().contains("invalid"));
        }
    }

    #[test]
    fn fixed_rotation_fallback_is_opt_in_for_collinear_centres() {
        let (source, target, truth) = collinear_fixture(24);
        let config = fallback_test_config();
        let error =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config).unwrap_err();
        assert_eq!(
            error.reason,
            RigSubmapAlignmentRejectionReason::Sim3Rejected
        );
        assert!(!error.diagnostics.fallback_attempted);
        assert_eq!(
            error.diagnostics.alignment_method,
            Some(RigSubmapAlignmentMethod::Generic)
        );
        assert!(!RigSubmapAlignmentConfig::default().allow_fixed_rotation_fallback);
        assert!(!RigSubmapAlignmentConfig::default().prefer_fixed_rotation_alignment);
        assert!(!RigSubmapAlignmentConfig::default().allow_generic_fallback_after_fixed_rotation);
        assert!(!RigSubmapAlignmentConfig::default().force_fixed_rotation_unit_scale);
        assert_eq!(
            error
                .sim3_rejection
                .as_ref()
                .expect("generic rejection is retained")
                .reason,
            SubmapSim3RejectionReason::DegenerateSourceGeometry
        );

        let mut opted_in = config;
        opted_in.allow_fixed_rotation_fallback = true;
        let result = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &opted_in)
            .expect("fixed orientation recovers the collinear gauge");
        assert!(result.diagnostics.fallback_attempted);
        assert!(result.diagnostics.fallback_used);
        assert_eq!(
            result.diagnostics.alignment_method,
            Some(RigSubmapAlignmentMethod::FixedRotationFallback)
        );
        assert!(result.diagnostics.generic_sim3_rejection.is_some());
        assert!((result.constraint.target_from_source.scale - truth.scale).abs() < 1.0e-9);
        assert!(
            (result.constraint.target_from_source.translation - truth.translation).norm() < 1.0e-9
        );
        assert!(
            result
                .constraint
                .target_from_source
                .rotation
                .rotation_to(&truth.rotation)
                .angle()
                < 1.0e-9
        );
    }

    #[test]
    fn fixed_rotation_fallback_rejects_centre_outliers_robustly() {
        let (source, mut target, truth) = collinear_fixture(32);
        for index in [3_usize, 11, 27] {
            let rotation = target[index].1.world_to_camera.rotation;
            target[index].1 =
                pose_from_center(rotation, Point3::new(90.0 + index as f64, -70.0, 40.0));
        }
        let mut config = fallback_test_config();
        config.allow_fixed_rotation_fallback = true;
        let result = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("a minority of centre outliers should be rejected");
        assert!(result.diagnostics.fallback_used);
        assert!(result.constraint.inlier_ratio > 0.8);
        assert!((result.constraint.target_from_source.scale - truth.scale).abs() < 1.0e-6);
        assert!(
            (result.constraint.target_from_source.translation - truth.translation).norm() < 1.0e-6
        );
        assert!(!result.constraint.inlier_match_indices.contains(&3));
        assert!(!result.constraint.inlier_match_indices.contains(&11));
        assert!(!result.constraint.inlier_match_indices.contains(&27));
    }

    #[test]
    fn fixed_rotation_unit_scale_recovers_known_gauge_with_outliers() {
        let (source, mut target, truth) = unit_scale_fixture(32);
        let outlier_frame_ids = [target[3].0, target[11].0, target[27].0];
        for index in [3_usize, 11, 27] {
            let rotation = target[index].1.world_to_camera.rotation;
            target[index].1 =
                pose_from_center(rotation, Point3::new(90.0 + index as f64, -70.0, 40.0));
        }
        let mut config = permissive_config();
        config.prefer_fixed_rotation_alignment = true;
        config.force_fixed_rotation_unit_scale = true;
        let first = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("unit-scale fixed rotation should reject a minority of outliers");
        let second = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("unit-scale fixed rotation should be deterministic");
        assert_eq!(first, second);
        assert_eq!(
            first.diagnostics.alignment_method,
            Some(RigSubmapAlignmentMethod::FixedRotationPrimary)
        );
        assert!(!first.diagnostics.fallback_attempted);
        assert!(!first.diagnostics.fallback_used);
        assert!(first.constraint.inlier_ratio > 0.8);
        assert_eq!(first.constraint.target_from_source.scale, 1.0);
        assert!(
            (first.constraint.target_from_source.translation - truth.translation).norm() < 1.0e-9
        );
        assert_eq!(first.constraint.leave_one_out_log_scale_mad, 0.0);
        for frame_id in outlier_frame_ids {
            let retained_index = first
                .diagnostics
                .retained_frame_ids
                .iter()
                .position(|retained_id| *retained_id == frame_id)
                .expect("outlier frame is retained");
            assert!(!first
                .constraint
                .inlier_match_indices
                .contains(&retained_index));
        }
    }

    #[test]
    fn fixed_rotation_unit_scale_rejects_malformed_degenerate_and_invalid_scale() {
        let rotation = UnitQuaternion::identity();
        let mut config = permissive_config().alignment;
        let malformed = vec![
            SubmapPointMatch {
                source_landmark_id: 0,
                target_landmark_id: 0,
                source_point: Point3::new(f64::NAN, 0.0, 0.0),
                target_point: Point3::origin(),
            },
            SubmapPointMatch {
                source_landmark_id: 1,
                target_landmark_id: 1,
                source_point: Point3::new(1.0, 0.0, 0.0),
                target_point: Point3::new(1.0, 0.0, 0.0),
            },
            SubmapPointMatch {
                source_landmark_id: 2,
                target_landmark_id: 2,
                source_point: Point3::new(2.0, 0.0, 0.0),
                target_point: Point3::new(2.0, 0.0, 0.0),
            },
        ];
        let error = estimate_fixed_rotation_unit_scale(0, 1, &malformed, &rotation, &config)
            .expect_err("non-finite points must fail closed");
        assert_eq!(error.reason, SubmapSim3RejectionReason::NonFinitePoint);

        let degenerate = malformed
            .into_iter()
            .enumerate()
            .map(|(index, mut point_match)| {
                point_match.source_point = Point3::new(index as f64, 0.0, 0.0);
                point_match.target_point = Point3::origin();
                point_match
            })
            .collect::<Vec<_>>();
        let error = estimate_fixed_rotation_unit_scale(0, 1, &degenerate, &rotation, &config)
            .expect_err("zero target scene scale must fail closed");
        assert_eq!(
            error.reason,
            SubmapSim3RejectionReason::InvalidTargetSceneScale
        );

        config.min_scale = 1.01;
        let (source, target, _) = unit_scale_fixture(12);
        let matches = source
            .iter()
            .zip(target.iter())
            .map(
                |((frame_id, source_pose), (_, target_pose))| SubmapPointMatch {
                    source_landmark_id: *frame_id,
                    target_landmark_id: *frame_id,
                    source_point: source_pose.camera_center_world(),
                    target_point: target_pose.camera_center_world(),
                },
            )
            .collect::<Vec<_>>();
        let error = estimate_fixed_rotation_unit_scale(0, 1, &matches, &rotation, &config)
            .expect_err("unit scale outside configured bounds must reject");
        assert_eq!(error.reason, SubmapSim3RejectionReason::ScaleOutOfBounds);
    }

    #[test]
    fn fixed_rotation_unit_scale_switch_is_default_off_and_preserves_default_path() {
        let (source, target, _) = collinear_fixture(24);
        let mut config = fallback_test_config();
        config.allow_fixed_rotation_fallback = true;
        let baseline = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("legacy fixed fallback should pass");
        let mut explicit_default = config.clone();
        explicit_default.force_fixed_rotation_unit_scale = false;
        let explicit =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &explicit_default)
                .expect("explicit false must retain legacy path");
        assert_eq!(baseline, explicit);
        assert!(!RigSubmapAlignmentConfig::default().force_fixed_rotation_unit_scale);
    }

    #[test]
    fn fixed_rotation_fallback_rejects_invalid_scale_and_unstable_loo() {
        let (source, target, _) = collinear_fixture(24);
        let mut invalid_scale = fallback_test_config();
        invalid_scale.allow_fixed_rotation_fallback = true;
        invalid_scale.alignment.min_scale = 10.0;
        let error = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &invalid_scale)
            .unwrap_err();
        assert!(error.diagnostics.fallback_attempted);
        assert_eq!(
            error
                .diagnostics
                .fallback_rejection
                .as_ref()
                .expect("fallback rejection is retained")
                .reason,
            SubmapSim3RejectionReason::NoRobustFit
        );

        let mut unstable_loo = fallback_test_config();
        unstable_loo.allow_fixed_rotation_fallback = true;
        unstable_loo.alignment.max_leave_one_out_log_scale_mad = -1.0;
        let error =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &unstable_loo).unwrap_err();
        assert_eq!(
            error
                .diagnostics
                .fallback_rejection
                .as_ref()
                .expect("fallback rejection is retained")
                .reason,
            SubmapSim3RejectionReason::UnstableLeaveOneOutScale
        );
        assert!(error
            .diagnostics
            .fallback_rejection
            .as_ref()
            .and_then(|rejection| rejection.leave_one_out_log_scale_mad)
            .is_some());
    }

    #[test]
    fn fixed_rotation_fallback_is_deterministic() {
        let (source, target, _) = collinear_fixture(24);
        let mut config = fallback_test_config();
        config.allow_fixed_rotation_fallback = true;
        config.alignment.random_seed = 73;
        let first = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("first deterministic run");
        let second = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("second deterministic run");
        assert_eq!(first, second);
    }

    #[test]
    fn malformed_input_never_attempts_fixed_rotation_fallback() {
        let (mut source, target, _) = collinear_fixture(24);
        source[4].1 = pose_from_center(UnitQuaternion::identity(), Point3::new(f64::NAN, 0.0, 0.0));
        let mut config = fallback_test_config();
        config.allow_fixed_rotation_fallback = true;
        let error =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config).unwrap_err();
        assert_eq!(
            error.reason,
            RigSubmapAlignmentRejectionReason::NonFinitePose
        );
        assert!(!error.diagnostics.fallback_attempted);
        assert!(error.sim3_rejection.is_none());
        assert!(error.fallback_rejection.is_none());
    }

    #[test]
    fn fixed_rotation_primary_is_deterministic_and_skips_generic_estimator() {
        let (source, target, truth) = collinear_fixture(24);
        let mut config = fallback_test_config();
        config.prefer_fixed_rotation_alignment = true;
        let first = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("fixed-primary should recover the collinear gauge");
        let second = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("fixed-primary should be deterministic");
        assert_eq!(first, second);
        assert_eq!(
            first.diagnostics.alignment_method,
            Some(RigSubmapAlignmentMethod::FixedRotationPrimary)
        );
        assert!(!first.diagnostics.fallback_attempted);
        assert!(!first.diagnostics.fallback_used);
        assert!(first.diagnostics.generic_sim3_rejection.is_none());
        assert!((first.constraint.target_from_source.scale - truth.scale).abs() < 1.0e-9);

        let mut rejecting = config;
        rejecting.alignment.min_scale = 10.0;
        let error =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &rejecting).unwrap_err();
        assert_eq!(
            error.diagnostics.alignment_method,
            Some(RigSubmapAlignmentMethod::FixedRotationPrimary)
        );
        assert!(!error.diagnostics.fallback_attempted);
        assert!(error.sim3_rejection.is_some());
        assert!(error.diagnostics.fixed_rotation_rejection.is_some());
        assert_eq!(
            error.sim3_rejection.as_ref().unwrap().reason,
            SubmapSim3RejectionReason::NoRobustFit
        );
    }

    #[test]
    fn alignment_method_names_are_stable() {
        assert_eq!(RigSubmapAlignmentMethod::Generic.as_str(), "generic");
        assert_eq!(
            RigSubmapAlignmentMethod::FixedRotationPrimary.as_str(),
            "fixed-primary"
        );
        assert_eq!(
            RigSubmapAlignmentMethod::FixedRotationFallback.as_str(),
            "fixed-fallback"
        );
        assert_eq!(
            RigSubmapAlignmentMethod::GenericFallback.as_str(),
            "generic-fallback"
        );
    }

    #[test]
    fn generic_fallback_after_fixed_primary_is_opt_in_and_deterministic() {
        let (source, target, truth) = fixed_rotation_mismatch_fixture(24, 20.0);
        let mut config = fallback_test_config();
        config.prefer_fixed_rotation_alignment = true;
        config.alignment.max_rotation_disagreement_deg = 60.0;

        let fixed_only = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect_err("fixed-primary should reject the mismatched rig rotation");
        assert_eq!(
            fixed_only.diagnostics.alignment_method,
            Some(RigSubmapAlignmentMethod::FixedRotationPrimary)
        );
        assert!(!fixed_only.diagnostics.fallback_attempted);
        assert!(fixed_only.diagnostics.fixed_rotation_rejection.is_some());
        assert!(fixed_only.diagnostics.generic_fallback_rejection.is_none());

        config.allow_generic_fallback_after_fixed_rotation = true;
        let first = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("generic fallback should recover the centre geometry");
        let second = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("generic fallback should be deterministic");
        assert_eq!(first, second);
        assert_eq!(
            first.diagnostics.alignment_method,
            Some(RigSubmapAlignmentMethod::GenericFallback)
        );
        assert!(first.diagnostics.fallback_attempted);
        assert!(first.diagnostics.fallback_used);
        assert!(first.diagnostics.fixed_rotation_rejection.is_some());
        assert!(first.diagnostics.generic_fallback_rejection.is_none());
        assert!(
            (first
                .constraint
                .target_from_source
                .rotation
                .rotation_to(&truth.rotation))
            .angle()
                < 1.0e-9
        );
        assert!((first.constraint.target_from_source.scale - truth.scale).abs() < 1.0e-9);
    }

    #[test]
    fn generic_fallback_after_fixed_primary_retains_both_rejections() {
        let (source, target, _) = fixed_rotation_mismatch_fixture(24, 20.0);
        let mut config = fallback_test_config();
        config.prefer_fixed_rotation_alignment = true;
        config.allow_generic_fallback_after_fixed_rotation = true;
        config.alignment.max_rotation_disagreement_deg = 5.0;
        let error = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect_err("the strict generic fallback rotation gate should reject");
        assert_eq!(
            error.diagnostics.alignment_method,
            Some(RigSubmapAlignmentMethod::GenericFallback)
        );
        assert!(error.diagnostics.fallback_attempted);
        assert!(!error.diagnostics.fallback_used);
        assert!(error.diagnostics.fixed_rotation_rejection.is_some());
        assert!(error.diagnostics.generic_fallback_rejection.is_some());
        assert_eq!(
            error
                .sim3_rejection
                .as_ref()
                .expect("primary fixed rejection")
                .reason,
            error
                .diagnostics
                .fixed_rotation_rejection
                .as_ref()
                .expect("diagnostic fixed rejection")
                .reason
        );
        assert_eq!(
            error
                .fallback_rejection
                .as_ref()
                .expect("fallback generic rejection")
                .reason,
            error
                .diagnostics
                .generic_fallback_rejection
                .as_ref()
                .expect("diagnostic generic rejection")
                .reason
        );
    }
}
