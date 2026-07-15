//! Covisibility-selected local bundle adjustment over an existing `VisualMap`.
//!
//! This is the ORB-SLAM-style local BA shape, but kept as an explicit API so it
//! can be A/B tested against temporal-window BA without changing the online
//! pipeline default. A caller supplies the active keyframe id; the selector
//! pulls in high-covisibility neighbor keyframes as variables, fixed boundary
//! keyframes that observe the same local landmarks, and the shared landmark set.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::Vector3;
use visloc_core::types::{Camera, FrameId, LandmarkId, Observation, VisualMap};

use crate::{
    BaConfig, BaError, BaGeneralStereoObservation, BaObservation, BaResult, BundleAdjustment,
    LinearSolver, PositionPrior, PositionPriorObservation, RobustKernel,
};

/// A keyframe ranked by how many selected landmarks it shares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovisibilityKeyframeScore {
    pub keyframe_id: FrameId,
    pub shared_landmark_count: usize,
}

/// Selection diagnostics for one covisibility local-BA problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CovisibilityLocalBaSelection {
    /// Keyframe that seeded the local BA problem.
    pub active_keyframe_id: FrameId,
    /// Variable keyframes. The active keyframe is always first, followed by
    /// high-covisibility neighbors.
    pub optimized_keyframe_ids: Vec<FrameId>,
    /// Fixed keyframes outside the optimized set that observe local landmarks.
    pub fixed_keyframe_ids: Vec<FrameId>,
    /// Landmarks included in the local BA.
    pub landmark_ids: Vec<LandmarkId>,
    /// Number of monocular reprojection observations added to BA.
    pub observation_count: usize,
    /// Ranked covisible candidates before the neighbor cap is applied.
    pub neighbor_candidates: Vec<CovisibilityKeyframeScore>,
    /// Ranked boundary candidates before the boundary cap is applied.
    pub boundary_candidates: Vec<CovisibilityKeyframeScore>,
    /// `true` when the selector had to retry boundary-keyframe selection with
    /// [`CovisibilityLocalBaConfig::fallback_min_boundary_observations`] after
    /// the primary boundary threshold produced no eligible local landmarks.
    pub boundary_fallback_used: bool,
}

impl CovisibilityLocalBaSelection {
    fn all_keyframe_ids(&self) -> Vec<FrameId> {
        let mut ids = self.optimized_keyframe_ids.clone();
        ids.extend(self.fixed_keyframe_ids.iter().copied());
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

/// Configuration for [`refine_visual_map_with_covisibility_ba`].
#[derive(Debug, Clone, PartialEq)]
pub struct CovisibilityLocalBaConfig {
    /// Maximum number of variable neighbor keyframes, excluding the active
    /// keyframe. Set to `0` for active-only pose/landmark refinement.
    pub max_neighbor_keyframes: usize,
    /// Minimum shared landmarks for a keyframe to become a variable neighbor.
    pub min_shared_landmarks: usize,
    /// Maximum fixed boundary keyframes. These keyframes anchor local map scale
    /// and pose gauge while their poses stay unchanged.
    pub max_boundary_keyframes: usize,
    /// Minimum observations of local landmarks for a keyframe to become a fixed
    /// boundary keyframe.
    pub min_boundary_observations: usize,
    /// Optional lower boundary-keyframe threshold used only when the primary
    /// boundary threshold produces no local landmarks. This is an A/B lever for
    /// weak covisibility windows; `None` preserves the strict single-pass
    /// selector.
    pub fallback_min_boundary_observations: Option<usize>,
    /// Minimum number of selected-keyframe observations a landmark needs before
    /// it contributes to BA. `2` rejects single-view points.
    pub min_observations_per_landmark: usize,
    /// Minimum number of selected local-landmark observations required on the
    /// active keyframe before BA runs. This avoids optimizing a newly promoted
    /// but weak active keyframe that does not actually constrain the local
    /// problem.
    pub min_active_observations: usize,
    /// Optional guard for large optimized windows with too little fixed
    /// boundary support. When `Some(n)`, windows with at least `n` optimized
    /// keyframes must also have at least
    /// [`Self::boundary_support_min_fixed_keyframes`] fixed boundary keyframes.
    pub boundary_support_min_optimized_keyframes: Option<usize>,
    /// Fixed-boundary keyframe floor used with
    /// [`Self::boundary_support_min_optimized_keyframes`].
    pub boundary_support_min_fixed_keyframes: usize,
    /// Optional runtime cap on local landmarks. Landmarks are ranked by how many
    /// optimized and total local keyframes observe them.
    pub max_landmarks: Option<usize>,
    /// Optional reprojection threshold for post-BA outlier diagnostics.
    pub outlier_reprojection_threshold_px: Option<f64>,
    /// When `true`, observations above
    /// [`Self::outlier_reprojection_threshold_px`] are removed from both
    /// keyframes and landmarks after a successful BA.
    pub remove_outlier_observations: bool,
    /// Replace a monocular left factor with the calibrated 4D
    /// `(u_l,v_l,u_r,v_r)` factor when a matching right-image sidecar exists.
    /// `false` preserves the historical local-BA objective.
    pub use_general_stereo_observations: bool,
    /// Optional pre-solve right-image reprojection gate for general stereo
    /// sidecars. Stale or incorrectly associated right measurements fall back
    /// to the ordinary monocular factor instead of entering BA.
    pub general_stereo_max_initial_right_reprojection_error_px: Option<f64>,
    /// Optional gauge/global-anchoring prior weight. When `Some(w)` with a
    /// finite `w > 0`, every optimized (non-fixed) keyframe in the window gets
    /// an absolute camera-center prior pulling it toward its pre-BA (tracking)
    /// estimate with per-axis weight `w` (cost `Σ w·‖C − C_pre‖²`). This pins
    /// the local window's gauge to the global map so a locally-consistent solve
    /// cannot drift the whole window — the EuRoC MH_05 covisibility-BA failure
    /// mode. `None` (default) leaves the solve anchored only by its fixed
    /// boundary keyframes (legacy behavior).
    pub pose_anchor_prior_weight: Option<f64>,
    /// Underlying Schur-complement BA configuration.
    pub ba_config: BaConfig,
}

impl Default for CovisibilityLocalBaConfig {
    fn default() -> Self {
        Self {
            max_neighbor_keyframes: 10,
            min_shared_landmarks: 15,
            max_boundary_keyframes: 10,
            min_boundary_observations: 5,
            fallback_min_boundary_observations: None,
            min_observations_per_landmark: 2,
            min_active_observations: 1,
            boundary_support_min_optimized_keyframes: None,
            boundary_support_min_fixed_keyframes: 0,
            max_landmarks: None,
            outlier_reprojection_threshold_px: Some(5.0),
            remove_outlier_observations: false,
            use_general_stereo_observations: false,
            general_stereo_max_initial_right_reprojection_error_px: Some(5.0),
            pose_anchor_prior_weight: None,
            ba_config: BaConfig {
                max_iterations: 12,
                robust_kernel: RobustKernel::Huber { delta: 3.0 },
                linear_solver: LinearSolver::Sparse,
                ..BaConfig::default()
            },
        }
    }
}

/// Result of one covisibility local-BA solve.
#[derive(Debug, Clone, PartialEq)]
pub struct CovisibilityLocalBaResult {
    pub selection: CovisibilityLocalBaSelection,
    pub ba_result: BaResult,
    pub mean_reprojection_before_px: f64,
    pub mean_reprojection_after_px: f64,
    /// Largest camera-centre displacement applied to an optimized keyframe.
    pub max_pose_translation_correction_m: f64,
    /// Largest world-to-camera rotation change applied to an optimized keyframe.
    pub max_pose_rotation_correction_rad: f64,
    pub updated_keyframe_count: usize,
    pub updated_landmark_count: usize,
    pub outlier_observation_count: usize,
    pub removed_observation_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CovisibilityLocalBaError {
    ActiveKeyframeMissing(FrameId),
    MissingPose(FrameId),
    MissingCamera(FrameId),
    UnsupportedCameraModel,
    NoReferenceLandmarks(FrameId),
    NoLocalLandmarks,
    NoObservations,
    InsufficientActiveObservations {
        keyframe_id: FrameId,
        observation_count: usize,
        min_observations: usize,
        boundary_fallback_used: bool,
    },
    InsufficientBoundaryKeyframes {
        optimized_keyframe_count: usize,
        fixed_keyframe_count: usize,
        min_optimized_keyframes: usize,
        min_fixed_keyframes: usize,
        boundary_fallback_used: bool,
    },
    QualityGateRejected {
        outlier_observation_count: usize,
        observation_count: usize,
        max_outlier_observation_ratio: f64,
    },
    /// Write-back rejected because too large a fraction of the solved
    /// optimized landmarks landed behind (or on) an observing optimized
    /// camera. Signals a degenerate/under-constrained solve; see
    /// [`behind_camera_optimized_landmark_ratio`].
    BehindCameraGateRejected {
        behind_camera_landmark_ratio: f64,
        max_behind_camera_landmark_ratio: f64,
    },
    /// Write-back rejected because the selected window did not carry enough
    /// fixed boundary keyframes relative to its optimized keyframes; see
    /// [`fixed_to_optimized_ratio_satisfied`].
    FixedSupportRatioRejected {
        optimized_keyframe_count: usize,
        fixed_keyframe_count: usize,
        required_fixed_keyframes: usize,
        min_fixed_to_optimized_ratio: f64,
    },
    /// Write-back rejected because the solved keyframe poses moved farther
    /// than the configured transactional safety bounds.
    PoseCorrectionGateRejected {
        translation_correction_m: f64,
        rotation_correction_rad: f64,
        max_translation_correction_m: Option<f64>,
        max_rotation_correction_rad: Option<f64>,
    },
    Ba(BaError),
}

impl std::fmt::Display for CovisibilityLocalBaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ActiveKeyframeMissing(id) => write!(f, "active keyframe {id} is missing"),
            Self::MissingPose(id) => write!(f, "keyframe {id} has no pose"),
            Self::MissingCamera(id) => write!(f, "camera for keyframe {id} is missing"),
            Self::UnsupportedCameraModel => write!(f, "camera model is not supported by BA"),
            Self::NoReferenceLandmarks(id) => {
                write!(f, "active keyframe {id} observes no known landmarks")
            }
            Self::NoLocalLandmarks => write!(f, "covisibility selection produced no landmarks"),
            Self::NoObservations => write!(f, "covisibility selection produced no observations"),
            Self::InsufficientActiveObservations {
                keyframe_id,
                observation_count,
                min_observations,
                ..
            } => write!(
                f,
                "active keyframe {keyframe_id} has {observation_count} selected observations, below required {min_observations}"
            ),
            Self::InsufficientBoundaryKeyframes {
                optimized_keyframe_count,
                fixed_keyframe_count,
                min_optimized_keyframes,
                min_fixed_keyframes,
                ..
            } => write!(
                f,
                "covisibility local BA selected {optimized_keyframe_count} optimized keyframes but only {fixed_keyframe_count} fixed boundary keyframes; at least {min_fixed_keyframes} are required once optimized keyframes reach {min_optimized_keyframes}"
            ),
            Self::QualityGateRejected {
                outlier_observation_count,
                observation_count,
                max_outlier_observation_ratio,
            } => write!(
                f,
                "covisibility local BA rejected by quality gate: {outlier_observation_count}/{observation_count} outlier observations exceeds ratio {max_outlier_observation_ratio}"
            ),
            Self::BehindCameraGateRejected {
                behind_camera_landmark_ratio,
                max_behind_camera_landmark_ratio,
            } => write!(
                f,
                "covisibility local BA rejected by behind-camera gate: {behind_camera_landmark_ratio} behind-camera landmark ratio exceeds max {max_behind_camera_landmark_ratio}"
            ),
            Self::FixedSupportRatioRejected {
                optimized_keyframe_count,
                fixed_keyframe_count,
                required_fixed_keyframes,
                min_fixed_to_optimized_ratio,
            } => write!(
                f,
                "covisibility local BA rejected by fixed-support ratio gate: {fixed_keyframe_count} fixed boundary keyframes below required {required_fixed_keyframes} for {optimized_keyframe_count} optimized keyframes at ratio {min_fixed_to_optimized_ratio}"
            ),
            Self::PoseCorrectionGateRejected {
                translation_correction_m,
                rotation_correction_rad,
                max_translation_correction_m,
                max_rotation_correction_rad,
            } => write!(
                f,
                "covisibility local BA rejected by pose-correction gate: translation {translation_correction_m} m (max {max_translation_correction_m:?}), rotation {rotation_correction_rad} rad (max {max_rotation_correction_rad:?})"
            ),
            Self::Ba(err) => write!(f, "bundle adjustment failed: {err}"),
        }
    }
}

impl std::error::Error for CovisibilityLocalBaError {}

impl From<BaError> for CovisibilityLocalBaError {
    fn from(value: BaError) -> Self {
        Self::Ba(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ObservationKey {
    frame_id: FrameId,
    landmark_id: LandmarkId,
    keypoint_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct LandmarkScore {
    landmark_id: LandmarkId,
    optimized_observation_count: usize,
    total_observation_count: usize,
}

#[derive(Debug)]
struct LocalLandmarkSelection {
    boundary_candidates: Vec<CovisibilityKeyframeScore>,
    fixed_keyframe_ids: Vec<FrameId>,
    ba_keyframe_ids: Vec<FrameId>,
    landmark_ids: Vec<LandmarkId>,
}

struct LocalLandmarkSelectionInput<'a> {
    map: &'a VisualMap,
    active_keyframe_id: FrameId,
    camera_id: u64,
    seed_landmarks: &'a BTreeSet<LandmarkId>,
    optimized_keyframe_ids: &'a [FrameId],
    optimized_set: &'a BTreeSet<FrameId>,
    config: &'a CovisibilityLocalBaConfig,
    min_boundary_observations: usize,
}

/// Select the covisibility local-BA window without running the optimizer.
pub fn select_covisibility_local_ba_window(
    map: &VisualMap,
    active_keyframe_id: FrameId,
    config: &CovisibilityLocalBaConfig,
) -> Result<CovisibilityLocalBaSelection, CovisibilityLocalBaError> {
    let active_kf = map.keyframes.get(&active_keyframe_id).ok_or(
        CovisibilityLocalBaError::ActiveKeyframeMissing(active_keyframe_id),
    )?;
    if active_kf.frame.pose.is_none() {
        return Err(CovisibilityLocalBaError::MissingPose(active_keyframe_id));
    }
    let camera = map
        .cameras
        .get(&active_kf.frame.camera_id)
        .ok_or(CovisibilityLocalBaError::MissingCamera(active_keyframe_id))?;
    if camera.intrinsics().is_none() {
        return Err(CovisibilityLocalBaError::UnsupportedCameraModel);
    }

    let reference_landmarks = observed_landmark_set(&active_kf.observations, map);
    if reference_landmarks.is_empty() {
        return Err(CovisibilityLocalBaError::NoReferenceLandmarks(
            active_keyframe_id,
        ));
    }

    let mut neighbor_candidates = rank_keyframes_by_shared_landmarks(
        map,
        active_keyframe_id,
        active_kf.frame.camera_id,
        &reference_landmarks,
        config.min_shared_landmarks,
        None,
    );
    let mut neighbor_ids: Vec<FrameId> = neighbor_candidates
        .iter()
        .take(config.max_neighbor_keyframes)
        .map(|score| score.keyframe_id)
        .collect();
    neighbor_ids.sort_unstable();

    let mut optimized_keyframe_ids = Vec::with_capacity(neighbor_ids.len() + 1);
    optimized_keyframe_ids.push(active_keyframe_id);
    optimized_keyframe_ids.extend(neighbor_ids);
    optimized_keyframe_ids.dedup();
    let optimized_set: BTreeSet<FrameId> = optimized_keyframe_ids.iter().copied().collect();

    let seed_landmarks = union_observed_landmarks(map, &optimized_keyframe_ids);
    if seed_landmarks.is_empty() {
        return Err(CovisibilityLocalBaError::NoLocalLandmarks);
    }

    let mut local_selection =
        select_local_landmarks_with_boundary_threshold(LocalLandmarkSelectionInput {
            map,
            active_keyframe_id,
            camera_id: active_kf.frame.camera_id,
            seed_landmarks: &seed_landmarks,
            optimized_keyframe_ids: &optimized_keyframe_ids,
            optimized_set: &optimized_set,
            config,
            min_boundary_observations: config.min_boundary_observations,
        });
    let mut boundary_fallback_used = false;
    if local_selection.landmark_ids.is_empty() {
        if let Some(fallback_min) = config.fallback_min_boundary_observations {
            let fallback_min = fallback_min.max(1);
            if fallback_min < config.min_boundary_observations {
                let fallback_selection =
                    select_local_landmarks_with_boundary_threshold(LocalLandmarkSelectionInput {
                        map,
                        active_keyframe_id,
                        camera_id: active_kf.frame.camera_id,
                        seed_landmarks: &seed_landmarks,
                        optimized_keyframe_ids: &optimized_keyframe_ids,
                        optimized_set: &optimized_set,
                        config,
                        min_boundary_observations: fallback_min,
                    });
                if !fallback_selection.landmark_ids.is_empty() {
                    local_selection = fallback_selection;
                    boundary_fallback_used = true;
                }
            }
        }
    }
    if local_selection.landmark_ids.is_empty() {
        return Err(CovisibilityLocalBaError::NoLocalLandmarks);
    }
    if let Some(min_optimized) = config.boundary_support_min_optimized_keyframes {
        let min_fixed = config.boundary_support_min_fixed_keyframes;
        if min_fixed > 0
            && optimized_keyframe_ids.len() >= min_optimized.max(1)
            && local_selection.fixed_keyframe_ids.len() < min_fixed
        {
            return Err(CovisibilityLocalBaError::InsufficientBoundaryKeyframes {
                optimized_keyframe_count: optimized_keyframe_ids.len(),
                fixed_keyframe_count: local_selection.fixed_keyframe_ids.len(),
                min_optimized_keyframes: min_optimized.max(1),
                min_fixed_keyframes: min_fixed,
                boundary_fallback_used,
            });
        }
    }

    let landmark_set: BTreeSet<LandmarkId> = local_selection.landmark_ids.iter().copied().collect();
    let observation_count =
        count_selected_observations(map, &local_selection.ba_keyframe_ids, &landmark_set);
    if observation_count == 0 {
        return Err(CovisibilityLocalBaError::NoObservations);
    }
    let active_observation_count =
        count_selected_observations(map, &[active_keyframe_id], &landmark_set);
    if active_observation_count < config.min_active_observations {
        return Err(CovisibilityLocalBaError::InsufficientActiveObservations {
            keyframe_id: active_keyframe_id,
            observation_count: active_observation_count,
            min_observations: config.min_active_observations,
            boundary_fallback_used,
        });
    }

    // Keep diagnostics deterministic even if caller maps came from HashMaps.
    neighbor_candidates.sort_by(score_order);
    local_selection.boundary_candidates.sort_by(score_order);

    Ok(CovisibilityLocalBaSelection {
        active_keyframe_id,
        optimized_keyframe_ids,
        fixed_keyframe_ids: local_selection.fixed_keyframe_ids,
        landmark_ids: local_selection.landmark_ids,
        observation_count,
        neighbor_candidates,
        boundary_candidates: local_selection.boundary_candidates,
        boundary_fallback_used,
    })
}

/// Run covisibility-selected local BA and write refined poses/landmarks back to
/// `map`.
pub fn refine_visual_map_with_covisibility_ba(
    map: &mut VisualMap,
    active_keyframe_id: FrameId,
    config: &CovisibilityLocalBaConfig,
) -> Result<CovisibilityLocalBaResult, CovisibilityLocalBaError> {
    let selection = select_covisibility_local_ba_window(map, active_keyframe_id, config)?;
    let active_camera = map
        .keyframes
        .get(&selection.active_keyframe_id)
        .ok_or(CovisibilityLocalBaError::ActiveKeyframeMissing(
            selection.active_keyframe_id,
        ))?
        .frame
        .camera_id;
    let camera = map
        .cameras
        .get(&active_camera)
        .ok_or(CovisibilityLocalBaError::MissingCamera(
            selection.active_keyframe_id,
        ))?
        .clone();

    let mut ba = build_ba_from_selection(map, &camera, &selection, config)?;
    let mean_reprojection_before_px = mean_reprojection_px(&ba);
    let ba_result = ba.optimize(&config.ba_config)?;
    let mean_reprojection_after_px = mean_reprojection_px(&ba);
    let mut max_pose_translation_correction_m = 0.0_f64;
    let mut max_pose_rotation_correction_rad = 0.0_f64;
    for keyframe_id in &selection.optimized_keyframe_ids {
        let (Some(before), Some(after)) = (
            map.keyframes
                .get(keyframe_id)
                .and_then(|keyframe| keyframe.frame.pose.as_ref()),
            ba.poses.get(keyframe_id),
        ) else {
            continue;
        };
        max_pose_translation_correction_m = max_pose_translation_correction_m
            .max((after.camera_center_world() - before.camera_center_world()).norm());
        let rotation_delta =
            before.world_to_camera.rotation.inverse() * after.world_to_camera.rotation;
        max_pose_rotation_correction_rad =
            max_pose_rotation_correction_rad.max(rotation_delta.angle());
    }

    let mut updated_keyframe_count = 0usize;
    for keyframe_id in &selection.optimized_keyframe_ids {
        if ba.fixed_poses.contains(keyframe_id) {
            continue;
        }
        let Some(refined_pose) = ba.poses.get(keyframe_id).cloned() else {
            continue;
        };
        if let Some(keyframe) = map.keyframes.get_mut(keyframe_id) {
            keyframe.frame.pose = Some(refined_pose);
            updated_keyframe_count += 1;
        }
    }

    let mut updated_landmark_count = 0usize;
    for landmark_id in &selection.landmark_ids {
        if ba.fixed_landmarks.contains(landmark_id) {
            continue;
        }
        let Some(refined_position) = ba.landmarks.get(landmark_id).copied() else {
            continue;
        };
        if let Some(landmark) = map.landmarks.get_mut(landmark_id) {
            landmark.position = refined_position;
            // The stored stereo covariance was linearized at the seed point.
            // Until BA marginal covariance recovery is wired here, dropping a
            // stale matrix is safer than using it with a moved landmark.
            map.landmark_position_covariances.remove(landmark_id);
            updated_landmark_count += 1;
        }
    }

    let outlier_keys = match config.outlier_reprojection_threshold_px {
        Some(threshold) if threshold.is_finite() && threshold > 0.0 => selected_outlier_keys(
            map,
            &camera,
            &selection,
            threshold,
            config.use_general_stereo_observations,
        ),
        _ => BTreeSet::new(),
    };
    let outlier_observation_count = outlier_keys.len();
    let removed_observation_count = if config.remove_outlier_observations {
        remove_observations(map, &outlier_keys)
    } else {
        0
    };

    Ok(CovisibilityLocalBaResult {
        selection,
        ba_result,
        mean_reprojection_before_px,
        mean_reprojection_after_px,
        max_pose_translation_correction_m,
        max_pose_rotation_correction_rad,
        updated_keyframe_count,
        updated_landmark_count,
        outlier_observation_count,
        removed_observation_count,
    })
}

fn build_ba_from_selection(
    map: &VisualMap,
    camera: &Camera,
    selection: &CovisibilityLocalBaSelection,
    config: &CovisibilityLocalBaConfig,
) -> Result<BundleAdjustment, CovisibilityLocalBaError> {
    let mut ba = BundleAdjustment::new(camera.clone());

    for keyframe_id in &selection.optimized_keyframe_ids {
        let keyframe = map.keyframes.get(keyframe_id).ok_or(
            CovisibilityLocalBaError::ActiveKeyframeMissing(*keyframe_id),
        )?;
        let pose = keyframe
            .frame
            .pose
            .clone()
            .ok_or(CovisibilityLocalBaError::MissingPose(*keyframe_id))?;
        ba.add_pose(*keyframe_id, pose);
    }

    for keyframe_id in &selection.fixed_keyframe_ids {
        let Some(keyframe) = map.keyframes.get(keyframe_id) else {
            continue;
        };
        let Some(pose) = keyframe.frame.pose.clone() else {
            continue;
        };
        ba.add_pose(*keyframe_id, pose);
        ba.fix_pose(*keyframe_id);
    }

    let no_boundary_gauge = selection.fixed_keyframe_ids.is_empty();
    if no_boundary_gauge {
        if let Some(anchor_id) = selection.optimized_keyframe_ids.first() {
            ba.fix_pose(*anchor_id);
        }
    }

    for landmark_id in &selection.landmark_ids {
        let Some(landmark) = map.landmarks.get(landmark_id) else {
            continue;
        };
        ba.add_landmark(*landmark_id, landmark.position);
    }
    if no_boundary_gauge {
        if let Some(anchor_landmark) = selection.landmark_ids.first() {
            ba.fix_landmark(*anchor_landmark);
        }
    }

    let landmark_set: BTreeSet<LandmarkId> = selection.landmark_ids.iter().copied().collect();
    for keyframe_id in selection.all_keyframe_ids() {
        let Some(keyframe) = map.keyframes.get(&keyframe_id) else {
            continue;
        };
        for obs in &keyframe.observations {
            if !landmark_set.contains(&obs.landmark_id) {
                continue;
            }
            if ba.poses.contains_key(&obs.frame_id) && ba.landmarks.contains_key(&obs.landmark_id) {
                let stereo = config
                    .use_general_stereo_observations
                    .then(|| {
                        map.stereo_observations
                            .iter()
                            .find(|stereo| {
                                stereo.frame_id == obs.frame_id
                                    && stereo.landmark_id == obs.landmark_id
                            })
                            .and_then(|stereo| {
                                map.cameras
                                    .get(&stereo.right_camera_id)
                                    .map(|camera| (stereo, camera))
                            })
                            .filter(|(stereo, right_camera)| {
                                let Some(max_error) =
                                    config.general_stereo_max_initial_right_reprojection_error_px
                                else {
                                    return true;
                                };
                                if !max_error.is_finite() || max_error <= 0.0 {
                                    return false;
                                }
                                let Some(pose) = ba.poses.get(&obs.frame_id) else {
                                    return false;
                                };
                                let Some(point_world) = ba.landmarks.get(&obs.landmark_id) else {
                                    return false;
                                };
                                let point_left = pose.transform_world_point(point_world);
                                right_camera
                                    .project(&stereo.left_to_right.transform_point(&point_left))
                                    .is_some_and(|predicted| {
                                        (predicted - stereo.xy_right).norm() <= max_error
                                    })
                            })
                    })
                    .flatten();
                if let Some((stereo, right_camera)) = stereo {
                    ba.add_general_stereo_observation(BaGeneralStereoObservation {
                        keyframe_id: obs.frame_id,
                        landmark_id: obs.landmark_id,
                        xy_left: obs.xy,
                        xy_right: stereo.xy_right,
                        right_camera: right_camera.clone(),
                        left_to_right: stereo.left_to_right.clone(),
                    });
                } else {
                    ba.add_observation(BaObservation {
                        keyframe_id: obs.frame_id,
                        landmark_id: obs.landmark_id,
                        xy: obs.xy,
                    });
                }
            }
        }
    }

    if let Some(weight) = config.pose_anchor_prior_weight {
        if weight.is_finite() && weight > 0.0 {
            let mut prior = PositionPrior::new();
            for keyframe_id in &selection.optimized_keyframe_ids {
                if ba.fixed_poses.contains(keyframe_id) {
                    continue;
                }
                let Some(pose) = ba.poses.get(keyframe_id) else {
                    continue;
                };
                prior.push(PositionPriorObservation {
                    keyframe_id: *keyframe_id,
                    camera_center_world: pose.camera_center_world(),
                    axis_weights: Vector3::new(weight, weight, weight),
                });
            }
            if !prior.observations.is_empty() {
                ba.set_position_prior(prior);
            }
        }
    }

    Ok(ba)
}

fn observed_landmark_set(observations: &[Observation], map: &VisualMap) -> BTreeSet<LandmarkId> {
    observations
        .iter()
        .map(|obs| obs.landmark_id)
        .filter(|landmark_id| map.landmarks.contains_key(landmark_id))
        .collect()
}

fn union_observed_landmarks(map: &VisualMap, keyframe_ids: &[FrameId]) -> BTreeSet<LandmarkId> {
    let mut out = BTreeSet::new();
    for keyframe_id in keyframe_ids {
        let Some(keyframe) = map.keyframes.get(keyframe_id) else {
            continue;
        };
        for obs in &keyframe.observations {
            if map.landmarks.contains_key(&obs.landmark_id) {
                out.insert(obs.landmark_id);
            }
        }
    }
    out
}

fn select_local_landmarks_with_boundary_threshold(
    input: LocalLandmarkSelectionInput<'_>,
) -> LocalLandmarkSelection {
    let boundary_candidates = rank_keyframes_by_shared_landmarks(
        input.map,
        input.active_keyframe_id,
        input.camera_id,
        input.seed_landmarks,
        input.min_boundary_observations.max(1),
        Some(input.optimized_set),
    );
    let fixed_keyframe_ids: Vec<FrameId> = boundary_candidates
        .iter()
        .take(input.config.max_boundary_keyframes)
        .map(|score| score.keyframe_id)
        .collect();

    let mut ba_keyframe_ids = input.optimized_keyframe_ids.to_vec();
    ba_keyframe_ids.extend(fixed_keyframe_ids.iter().copied());
    ba_keyframe_ids.sort_unstable();
    ba_keyframe_ids.dedup();

    let mut landmark_scores =
        score_local_landmarks(input.map, input.optimized_set, &ba_keyframe_ids);
    landmark_scores.retain(|score| {
        score.optimized_observation_count > 0
            && score.total_observation_count >= input.config.min_observations_per_landmark
            && input.map.landmarks.contains_key(&score.landmark_id)
    });
    landmark_scores.sort_by(|a, b| {
        b.optimized_observation_count
            .cmp(&a.optimized_observation_count)
            .then_with(|| b.total_observation_count.cmp(&a.total_observation_count))
            .then_with(|| a.landmark_id.cmp(&b.landmark_id))
    });
    if let Some(max_landmarks) = input.config.max_landmarks {
        landmark_scores.truncate(max_landmarks);
    }
    let mut landmark_ids: Vec<LandmarkId> = landmark_scores
        .iter()
        .map(|score| score.landmark_id)
        .collect();
    landmark_ids.sort_unstable();

    LocalLandmarkSelection {
        boundary_candidates,
        fixed_keyframe_ids,
        ba_keyframe_ids,
        landmark_ids,
    }
}

fn rank_keyframes_by_shared_landmarks(
    map: &VisualMap,
    active_keyframe_id: FrameId,
    camera_id: u64,
    landmarks: &BTreeSet<LandmarkId>,
    min_count: usize,
    excluded: Option<&BTreeSet<FrameId>>,
) -> Vec<CovisibilityKeyframeScore> {
    let mut scores = Vec::new();
    let mut keyframe_ids: Vec<FrameId> = map.keyframes.keys().copied().collect();
    keyframe_ids.sort_unstable();
    for keyframe_id in keyframe_ids {
        if keyframe_id == active_keyframe_id {
            continue;
        }
        if excluded
            .map(|set| set.contains(&keyframe_id))
            .unwrap_or(false)
        {
            continue;
        }
        let Some(keyframe) = map.keyframes.get(&keyframe_id) else {
            continue;
        };
        if keyframe.frame.camera_id != camera_id || keyframe.frame.pose.is_none() {
            continue;
        }
        let count = keyframe
            .observations
            .iter()
            .map(|obs| obs.landmark_id)
            .collect::<BTreeSet<_>>()
            .intersection(landmarks)
            .count();
        if count >= min_count {
            scores.push(CovisibilityKeyframeScore {
                keyframe_id,
                shared_landmark_count: count,
            });
        }
    }
    scores.sort_by(score_order);
    scores
}

fn score_order(a: &CovisibilityKeyframeScore, b: &CovisibilityKeyframeScore) -> std::cmp::Ordering {
    b.shared_landmark_count
        .cmp(&a.shared_landmark_count)
        .then_with(|| a.keyframe_id.cmp(&b.keyframe_id))
}

fn score_local_landmarks(
    map: &VisualMap,
    optimized_keyframes: &BTreeSet<FrameId>,
    ba_keyframe_ids: &[FrameId],
) -> Vec<LandmarkScore> {
    let mut scores: BTreeMap<LandmarkId, LandmarkScore> = BTreeMap::new();
    for keyframe_id in ba_keyframe_ids {
        let Some(keyframe) = map.keyframes.get(keyframe_id) else {
            continue;
        };
        for obs in &keyframe.observations {
            if !map.landmarks.contains_key(&obs.landmark_id) {
                continue;
            }
            let slot = scores.entry(obs.landmark_id).or_insert(LandmarkScore {
                landmark_id: obs.landmark_id,
                optimized_observation_count: 0,
                total_observation_count: 0,
            });
            slot.total_observation_count += 1;
            if optimized_keyframes.contains(keyframe_id) {
                slot.optimized_observation_count += 1;
            }
        }
    }
    scores.into_values().collect()
}

fn count_selected_observations(
    map: &VisualMap,
    keyframe_ids: &[FrameId],
    landmark_ids: &BTreeSet<LandmarkId>,
) -> usize {
    keyframe_ids
        .iter()
        .filter_map(|id| map.keyframes.get(id))
        .map(|keyframe| {
            keyframe
                .observations
                .iter()
                .filter(|obs| landmark_ids.contains(&obs.landmark_id))
                .count()
        })
        .sum()
}

fn mean_reprojection_px(ba: &BundleAdjustment) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for obs in &ba.observations {
        let (Some(pose), Some(point)) = (
            ba.poses.get(&obs.keyframe_id),
            ba.landmarks.get(&obs.landmark_id),
        ) else {
            continue;
        };
        let Some(predicted) = ba.camera.project(&pose.transform_world_point(point)) else {
            continue;
        };
        sum += (predicted - obs.xy).norm();
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Mean pixel reprojection error of the observations represented by a
/// covisibility selection in a written-back map. This is intentionally public
/// so transactional callers can apply an additional pose constraint (for
/// example loop-welding rotation anchoring) and re-run the same observable
/// quality check before committing their cloned map.
pub fn mean_selected_reprojection_px(
    map: &VisualMap,
    selection: &CovisibilityLocalBaSelection,
) -> Option<f64> {
    let selected_landmarks: BTreeSet<LandmarkId> = selection.landmark_ids.iter().copied().collect();
    let mut keyframe_ids = selection.optimized_keyframe_ids.clone();
    keyframe_ids.extend(selection.fixed_keyframe_ids.iter().copied());
    keyframe_ids.sort_unstable();
    keyframe_ids.dedup();

    let mut sum = 0.0;
    let mut count = 0usize;
    for keyframe_id in keyframe_ids {
        let keyframe = map.keyframes.get(&keyframe_id)?;
        let pose = keyframe.frame.pose.as_ref()?;
        let camera = map.cameras.get(&keyframe.frame.camera_id)?;
        for observation in &keyframe.observations {
            if !selected_landmarks.contains(&observation.landmark_id) {
                continue;
            }
            let point = &map.landmarks.get(&observation.landmark_id)?.position;
            let Some(predicted) = camera.project(&pose.transform_world_point(point)) else {
                continue;
            };
            sum += (predicted - observation.xy).norm();
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}

/// Number of fixed boundary keyframes a solve must carry to satisfy
/// [`fixed_to_optimized_ratio_satisfied`].
///
/// Semantics: `required = ceil(optimized_keyframe_count * ratio)`. A
/// non-positive or non-finite ratio disables the requirement (returns `0`).
pub fn required_fixed_keyframes(optimized_keyframe_count: usize, ratio: f64) -> usize {
    if !ratio.is_finite() || ratio <= 0.0 {
        return 0;
    }
    (optimized_keyframe_count as f64 * ratio).ceil() as usize
}

/// Write-back adequacy predicate for the fixed-anchor ratio gate.
///
/// Returns `true` (adequately anchored) when
/// `fixed_keyframe_count >= ceil(optimized_keyframe_count * ratio)`.
/// This is the *ratio* form of the fixed-boundary requirement; the absolute
/// floor form lives on [`CovisibilityLocalBaConfig::boundary_support_min_fixed_keyframes`].
pub fn fixed_to_optimized_ratio_satisfied(
    optimized_keyframe_count: usize,
    fixed_keyframe_count: usize,
    ratio: f64,
) -> bool {
    fixed_keyframe_count >= required_fixed_keyframes(optimized_keyframe_count, ratio)
}

/// Fraction of the selected optimized landmarks that project behind (or onto)
/// at least one observing optimized camera in `map`.
///
/// Intended to run against the *solved* map (post write-back on a candidate
/// clone) so a degenerate/under-constrained solve — where optimized landmarks
/// collapse behind the optimized cameras — can be detected and its write-back
/// rejected. Uses the same [`Camera::project`] / `Pose::transform_world_point`
/// path as `selected_outlier_keys`, but unlike `mean_reprojection_px`,
/// which silently skips landmarks that fail to project, a non-positive
/// camera-frame depth (or a `None` projection) is counted here as a hard
/// degeneracy.
///
/// Only cameras in [`CovisibilityLocalBaSelection::optimized_keyframe_ids`]
/// (the variable-pose set a bad solve corrupts) are scored; fixed boundary
/// keyframes anchor the gauge and are ignored. A landmark counts as degenerate
/// if it lands behind *any* optimized camera that observes it. Returns `None`
/// when no optimized camera observes any selected landmark (nothing to score).
pub fn behind_camera_optimized_landmark_ratio(
    map: &VisualMap,
    selection: &CovisibilityLocalBaSelection,
) -> Option<f64> {
    let active_kf = map.keyframes.get(&selection.active_keyframe_id)?;
    let camera = map.cameras.get(&active_kf.frame.camera_id)?;

    let mut considered = 0usize;
    let mut degenerate = 0usize;
    for landmark_id in &selection.landmark_ids {
        let Some(landmark) = map.landmarks.get(landmark_id) else {
            continue;
        };
        let mut observed_by_optimized = false;
        let mut behind = false;
        for keyframe_id in &selection.optimized_keyframe_ids {
            let Some(keyframe) = map.keyframes.get(keyframe_id) else {
                continue;
            };
            let Some(pose) = keyframe.frame.pose.as_ref() else {
                continue;
            };
            if !keyframe
                .observations
                .iter()
                .any(|obs| obs.landmark_id == *landmark_id)
            {
                continue;
            }
            observed_by_optimized = true;
            let point_cam = pose.transform_world_point(&landmark.position);
            if point_cam.z <= 0.0 || camera.project(&point_cam).is_none() {
                behind = true;
                break;
            }
        }
        if observed_by_optimized {
            considered += 1;
            if behind {
                degenerate += 1;
            }
        }
    }

    if considered == 0 {
        None
    } else {
        Some(degenerate as f64 / considered as f64)
    }
}

fn selected_outlier_keys(
    map: &VisualMap,
    camera: &Camera,
    selection: &CovisibilityLocalBaSelection,
    threshold_px: f64,
    use_general_stereo_observations: bool,
) -> BTreeSet<ObservationKey> {
    let landmark_ids: BTreeSet<LandmarkId> = selection.landmark_ids.iter().copied().collect();
    let mut outliers = BTreeSet::new();
    for keyframe_id in selection.all_keyframe_ids() {
        let Some(keyframe) = map.keyframes.get(&keyframe_id) else {
            continue;
        };
        let Some(pose) = keyframe.frame.pose.as_ref() else {
            continue;
        };
        for obs in &keyframe.observations {
            if !landmark_ids.contains(&obs.landmark_id) {
                continue;
            }
            let Some(landmark) = map.landmarks.get(&obs.landmark_id) else {
                continue;
            };
            let mut residual = camera
                .project(&pose.transform_world_point(&landmark.position))
                .map(|predicted| (predicted - obs.xy).norm())
                .unwrap_or(f64::INFINITY);
            if let Some((stereo, right_camera)) = use_general_stereo_observations
                .then(|| {
                    map.stereo_observations
                        .iter()
                        .find(|stereo| {
                            stereo.frame_id == obs.frame_id && stereo.landmark_id == obs.landmark_id
                        })
                        .and_then(|stereo| {
                            map.cameras
                                .get(&stereo.right_camera_id)
                                .map(|camera| (stereo, camera))
                        })
                })
                .flatten()
            {
                let point_left = pose.transform_world_point(&landmark.position);
                let right_residual = right_camera
                    .project(&stereo.left_to_right.transform_point(&point_left))
                    .map(|predicted| (predicted - stereo.xy_right).norm())
                    .unwrap_or(f64::INFINITY);
                residual = residual.max(right_residual);
            }
            if !residual.is_finite() || residual > threshold_px {
                outliers.insert(ObservationKey {
                    frame_id: obs.frame_id,
                    landmark_id: obs.landmark_id,
                    keypoint_index: obs.keypoint_index,
                });
            }
        }
    }
    outliers
}

fn remove_observations(map: &mut VisualMap, outliers: &BTreeSet<ObservationKey>) -> usize {
    if outliers.is_empty() {
        return 0;
    }
    let mut removed = 0usize;
    for keyframe in map.keyframes.values_mut() {
        let before = keyframe.observations.len();
        keyframe.observations.retain(|obs| {
            !outliers.contains(&ObservationKey {
                frame_id: obs.frame_id,
                landmark_id: obs.landmark_id,
                keypoint_index: obs.keypoint_index,
            })
        });
        removed += before - keyframe.observations.len();
    }
    for landmark in map.landmarks.values_mut() {
        landmark.observations.retain(|obs| {
            !outliers.contains(&ObservationKey {
                frame_id: obs.frame_id,
                landmark_id: obs.landmark_id,
                keypoint_index: obs.keypoint_index,
            })
        });
    }
    map.stereo_observations.retain(|stereo| {
        !outliers.iter().any(|outlier| {
            outlier.frame_id == stereo.frame_id && outlier.landmark_id == stereo.landmark_id
        })
    });
    removed
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::{Pose, SE3};
    use visloc_core::types::{Camera, Frame, Keyframe, Landmark, StereoObservation};

    use super::*;

    fn pose_at_center(x: f64, y: f64, z: f64) -> Pose {
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-x, -y, -z))
    }

    fn add_keyframe(map: &mut VisualMap, id: FrameId, pose: Pose) {
        let mut frame = Frame::new(id, 1);
        frame.pose = Some(pose);
        map.keyframes.insert(
            id,
            Keyframe {
                frame,
                observations: Vec::new(),
            },
        );
    }

    fn add_observation(map: &mut VisualMap, frame_id: FrameId, landmark_id: LandmarkId) {
        let camera = map.cameras.get(&1).unwrap();
        let pose = map.keyframes[&frame_id].frame.pose.as_ref().unwrap();
        let point = map.landmarks[&landmark_id].position;
        let xy = camera.project(&pose.transform_world_point(&point)).unwrap();
        let keypoint_index = {
            let keyframe = map.keyframes.get_mut(&frame_id).unwrap();
            let idx = keyframe.frame.keypoints.len();
            keyframe.frame.keypoints.push(xy);
            keyframe.frame.descriptors.push(vec![landmark_id as f32]);
            idx
        };
        let obs = Observation {
            frame_id,
            landmark_id,
            keypoint_index,
            xy,
        };
        map.keyframes
            .get_mut(&frame_id)
            .unwrap()
            .observations
            .push(obs.clone());
        map.landmarks
            .get_mut(&landmark_id)
            .unwrap()
            .observations
            .push(obs);
    }

    fn synthetic_map() -> (VisualMap, Vec<Pose>) {
        let camera = Camera::pinhole(1, 640, 480, 320.0, 320.0, 320.0, 240.0);
        let mut map = VisualMap::new();
        map.cameras.insert(1, camera);
        let truth_poses = vec![
            pose_at_center(0.0, 0.0, 0.0),
            pose_at_center(1.0, 0.0, 0.0),
            pose_at_center(2.0, 0.0, 0.0),
            pose_at_center(3.0, 0.0, 0.0),
        ];
        for (id, pose) in truth_poses.iter().enumerate() {
            add_keyframe(&mut map, id as u64, pose.clone());
        }
        for i in 0..24u64 {
            let x = (i % 6) as f64 * 0.35 + 0.5;
            let y = (i / 6) as f64 * 0.12 - 0.18;
            let z = 5.0 + (i % 4) as f64 * 0.4;
            map.landmarks
                .insert(i, Landmark::new(i, Point3::new(x, y, z)));
        }
        for frame_id in 0..4u64 {
            for landmark_id in 0..24u64 {
                add_observation(&mut map, frame_id, landmark_id);
            }
        }
        // Perturb two variable keyframes and the landmarks after measurements
        // have been generated from the true geometry.
        map.keyframes.get_mut(&0).unwrap().frame.pose = Some(pose_at_center(0.25, 0.02, 0.0));
        map.keyframes.get_mut(&2).unwrap().frame.pose = Some(pose_at_center(2.35, -0.04, 0.0));
        for (id, landmark) in map.landmarks.iter_mut() {
            let s = (*id % 3) as f64 - 1.0;
            landmark.position.x += 0.04 * s;
            landmark.position.z += 0.08;
        }
        (map, truth_poses)
    }

    #[test]
    fn selection_picks_neighbors_boundaries_and_local_landmarks() {
        let (map, _) = synthetic_map();
        let config = CovisibilityLocalBaConfig {
            max_neighbor_keyframes: 1,
            min_shared_landmarks: 1,
            max_boundary_keyframes: 2,
            min_boundary_observations: 1,
            min_observations_per_landmark: 2,
            ..CovisibilityLocalBaConfig::default()
        };

        let selection = select_covisibility_local_ba_window(&map, 2, &config).unwrap();

        assert_eq!(selection.active_keyframe_id, 2);
        assert_eq!(selection.optimized_keyframe_ids, vec![2, 0]);
        assert_eq!(selection.fixed_keyframe_ids, vec![1, 3]);
        assert_eq!(selection.landmark_ids.len(), 24);
        assert_eq!(selection.observation_count, 96);
        assert!(!selection.boundary_fallback_used);
    }

    #[test]
    fn bundle_builder_uses_general_stereo_instead_of_double_counting_left_pixel() {
        let (mut map, _) = synthetic_map();
        let right_camera = Camera::pinhole(2, 640, 480, 325.0, 315.0, 318.0, 242.0);
        let left_to_right = SE3::new(
            UnitQuaternion::from_euler_angles(0.01, -0.02, 0.015),
            Vector3::new(-0.11, 0.002, 0.001),
        );
        let pose = map.keyframes[&2].frame.pose.as_ref().unwrap();
        let point_left = pose.transform_world_point(&map.landmarks[&0].position);
        let xy_right = right_camera
            .project(&left_to_right.transform_point(&point_left))
            .unwrap();
        map.cameras.insert(2, right_camera);
        map.stereo_observations.push(StereoObservation {
            frame_id: 2,
            landmark_id: 0,
            right_camera_id: 2,
            xy_right,
            left_to_right,
        });
        let config = CovisibilityLocalBaConfig {
            max_neighbor_keyframes: 1,
            min_shared_landmarks: 1,
            max_boundary_keyframes: 2,
            min_boundary_observations: 1,
            min_observations_per_landmark: 2,
            use_general_stereo_observations: true,
            ..CovisibilityLocalBaConfig::default()
        };
        let selection = select_covisibility_local_ba_window(&map, 2, &config).unwrap();
        let ba = build_ba_from_selection(&map, map.cameras.get(&1).unwrap(), &selection, &config)
            .unwrap();

        assert_eq!(ba.general_stereo_observations.len(), 1);
        assert_eq!(ba.observations.len(), selection.observation_count - 1);

        map.stereo_observations[0].xy_right.x += 100.0;
        let gated =
            build_ba_from_selection(&map, map.cameras.get(&1).unwrap(), &selection, &config)
                .unwrap();
        assert!(gated.general_stereo_observations.is_empty());
        assert_eq!(gated.observations.len(), selection.observation_count);
    }

    #[test]
    fn selection_rejects_active_keyframe_below_observation_floor() {
        let (map, _) = synthetic_map();
        let config = CovisibilityLocalBaConfig {
            max_neighbor_keyframes: 1,
            min_shared_landmarks: 1,
            max_boundary_keyframes: 2,
            min_boundary_observations: 1,
            min_observations_per_landmark: 2,
            min_active_observations: 25,
            ..CovisibilityLocalBaConfig::default()
        };

        let err = select_covisibility_local_ba_window(&map, 2, &config).unwrap_err();

        assert_eq!(
            err,
            CovisibilityLocalBaError::InsufficientActiveObservations {
                keyframe_id: 2,
                observation_count: 24,
                min_observations: 25,
                boundary_fallback_used: false,
            }
        );
    }

    #[test]
    fn selection_can_fallback_to_lower_boundary_observation_floor() {
        let (map, _) = synthetic_map();
        let strict_config = CovisibilityLocalBaConfig {
            max_neighbor_keyframes: 0,
            min_shared_landmarks: 1,
            max_boundary_keyframes: 2,
            min_boundary_observations: 25,
            fallback_min_boundary_observations: None,
            min_observations_per_landmark: 2,
            ..CovisibilityLocalBaConfig::default()
        };

        let strict_err = select_covisibility_local_ba_window(&map, 2, &strict_config).unwrap_err();
        assert_eq!(strict_err, CovisibilityLocalBaError::NoLocalLandmarks);

        let fallback_config = CovisibilityLocalBaConfig {
            fallback_min_boundary_observations: Some(1),
            ..strict_config
        };
        let selection = select_covisibility_local_ba_window(&map, 2, &fallback_config).unwrap();

        assert!(selection.boundary_fallback_used);
        assert_eq!(selection.optimized_keyframe_ids, vec![2]);
        assert_eq!(selection.fixed_keyframe_ids, vec![0, 1]);
        assert_eq!(selection.landmark_ids.len(), 24);
    }

    #[test]
    fn selection_rejects_large_window_with_insufficient_boundary_keyframes() {
        let (map, _) = synthetic_map();
        let config = CovisibilityLocalBaConfig {
            max_neighbor_keyframes: 3,
            min_shared_landmarks: 1,
            max_boundary_keyframes: 1,
            min_boundary_observations: 1,
            min_observations_per_landmark: 2,
            boundary_support_min_optimized_keyframes: Some(3),
            boundary_support_min_fixed_keyframes: 2,
            ..CovisibilityLocalBaConfig::default()
        };

        let err = select_covisibility_local_ba_window(&map, 2, &config).unwrap_err();

        assert_eq!(
            err,
            CovisibilityLocalBaError::InsufficientBoundaryKeyframes {
                optimized_keyframe_count: 4,
                fixed_keyframe_count: 0,
                min_optimized_keyframes: 3,
                min_fixed_keyframes: 2,
                boundary_fallback_used: false,
            }
        );
    }

    #[test]
    fn covisibility_ba_reduces_reprojection_and_updates_active_pose() {
        let (mut map, truth_poses) = synthetic_map();
        let before_center = map.keyframes[&2]
            .frame
            .pose
            .as_ref()
            .unwrap()
            .camera_center_world();
        let before_error = (before_center - truth_poses[2].camera_center_world()).norm();
        let config = CovisibilityLocalBaConfig {
            max_neighbor_keyframes: 1,
            min_shared_landmarks: 1,
            max_boundary_keyframes: 2,
            min_boundary_observations: 1,
            min_observations_per_landmark: 2,
            outlier_reprojection_threshold_px: Some(10.0),
            ..CovisibilityLocalBaConfig::default()
        };

        let result = refine_visual_map_with_covisibility_ba(&mut map, 2, &config).unwrap();

        let after_center = map.keyframes[&2]
            .frame
            .pose
            .as_ref()
            .unwrap()
            .camera_center_world();
        let after_error = (after_center - truth_poses[2].camera_center_world()).norm();

        assert!(result.mean_reprojection_after_px < result.mean_reprojection_before_px);
        assert!(after_error < before_error);
        assert!(result.updated_keyframe_count >= 1);
        assert_eq!(result.updated_landmark_count, 24);
        assert_eq!(result.outlier_observation_count, 0);
    }

    #[test]
    fn covisibility_ba_can_remove_post_ba_outlier_observations() {
        let (mut map, _) = synthetic_map();
        let config = CovisibilityLocalBaConfig {
            max_neighbor_keyframes: 1,
            min_shared_landmarks: 1,
            max_boundary_keyframes: 2,
            min_boundary_observations: 1,
            min_observations_per_landmark: 2,
            outlier_reprojection_threshold_px: Some(0.5),
            remove_outlier_observations: true,
            ..CovisibilityLocalBaConfig::default()
        };
        // Corrupt one selected observation after the synthetic measurements were
        // inserted. The same key is mirrored in the landmark observation list.
        let keypoint_index = map.keyframes[&2].observations[0].keypoint_index;
        map.keyframes.get_mut(&2).unwrap().observations[0].xy.x += 100.0;
        if let Some(obs) = map
            .landmarks
            .get_mut(&0)
            .unwrap()
            .observations
            .iter_mut()
            .find(|obs| obs.frame_id == 2 && obs.keypoint_index == keypoint_index)
        {
            obs.xy.x += 100.0;
        }

        let result = refine_visual_map_with_covisibility_ba(&mut map, 2, &config).unwrap();

        assert!(result.outlier_observation_count >= 1);
        assert!(result.removed_observation_count >= 1);
        assert!(!map.keyframes[&2]
            .observations
            .iter()
            .any(|obs| { obs.landmark_id == 0 && obs.keypoint_index == keypoint_index }));
        assert!(!map.landmarks[&0]
            .observations
            .iter()
            .any(|obs| { obs.frame_id == 2 && obs.keypoint_index == keypoint_index }));
    }

    fn behind_camera_test_config() -> CovisibilityLocalBaConfig {
        CovisibilityLocalBaConfig {
            max_neighbor_keyframes: 1,
            min_shared_landmarks: 1,
            max_boundary_keyframes: 2,
            min_boundary_observations: 1,
            min_observations_per_landmark: 2,
            ..CovisibilityLocalBaConfig::default()
        }
    }

    #[test]
    fn behind_camera_ratio_is_zero_for_healthy_window() {
        let (map, _) = synthetic_map();
        let config = behind_camera_test_config();
        let selection = select_covisibility_local_ba_window(&map, 2, &config).unwrap();

        // All synthetic landmarks sit ~5 m in front of the cameras.
        let ratio = behind_camera_optimized_landmark_ratio(&map, &selection).unwrap();
        assert!(ratio.abs() < 1e-9, "healthy window ratio {ratio}");
    }

    #[test]
    fn behind_camera_ratio_is_high_when_optimized_landmarks_collapse_behind() {
        let (mut map, _) = synthetic_map();
        let config = behind_camera_test_config();
        let selection = select_covisibility_local_ba_window(&map, 2, &config).unwrap();

        // Simulate a degenerate solve: drive every selected landmark behind the
        // (identity-rotation) optimized cameras, whose centres sit near z = 0.
        for landmark_id in &selection.landmark_ids {
            if let Some(landmark) = map.landmarks.get_mut(landmark_id) {
                landmark.position.z = -10.0;
            }
        }

        let ratio = behind_camera_optimized_landmark_ratio(&map, &selection).unwrap();
        assert!(ratio > 0.5, "collapsed window ratio {ratio}");
    }

    #[test]
    fn fixed_to_optimized_ratio_gate_semantics() {
        // required = ceil(optimized * ratio); satisfied = fixed >= required.
        // optimized=7, r=0.34 -> ceil(2.38)=3, so fixed=1 is rejected.
        assert_eq!(required_fixed_keyframes(7, 0.34), 3);
        assert!(!fixed_to_optimized_ratio_satisfied(7, 1, 0.34));
        // optimized=3, r=0.34 -> ceil(1.02)=2, so fixed=2 is accepted.
        assert_eq!(required_fixed_keyframes(3, 0.34), 2);
        assert!(fixed_to_optimized_ratio_satisfied(3, 2, 0.34));
        // A non-positive ratio disables the requirement (always satisfied).
        assert_eq!(required_fixed_keyframes(7, 0.0), 0);
        assert!(fixed_to_optimized_ratio_satisfied(7, 0, 0.0));
    }

    #[test]
    fn pose_anchor_prior_pins_window_gauge() {
        // `synthetic_map` already perturbs keyframes 0 and 2 away from the true
        // geometry before BA runs, so there is real pose motion for an unanchored
        // solve to make and for the anchor prior to resist.
        let (map, _) = synthetic_map();
        let base_config = CovisibilityLocalBaConfig {
            max_neighbor_keyframes: 1,
            min_shared_landmarks: 1,
            max_boundary_keyframes: 2,
            min_boundary_observations: 1,
            min_observations_per_landmark: 2,
            ..CovisibilityLocalBaConfig::default()
        };

        let initial_centers: BTreeMap<FrameId, Point3<f64>> = map
            .keyframes
            .iter()
            .map(|(id, kf)| (*id, kf.frame.pose.as_ref().unwrap().camera_center_world()))
            .collect();

        let displacement = |refined_map: &VisualMap, optimized_ids: &[FrameId]| -> f64 {
            optimized_ids
                .iter()
                .map(|id| {
                    let after = refined_map.keyframes[id]
                        .frame
                        .pose
                        .as_ref()
                        .unwrap()
                        .camera_center_world();
                    (after - initial_centers[id]).norm()
                })
                .sum()
        };

        let mut unanchored_map = map.clone();
        let unanchored_config = CovisibilityLocalBaConfig {
            pose_anchor_prior_weight: None,
            ..base_config.clone()
        };
        let unanchored_result =
            refine_visual_map_with_covisibility_ba(&mut unanchored_map, 2, &unanchored_config)
                .unwrap();
        let unanchored_displacement = displacement(
            &unanchored_map,
            &unanchored_result.selection.optimized_keyframe_ids,
        );

        let mut anchored_map = map.clone();
        let anchored_config = CovisibilityLocalBaConfig {
            pose_anchor_prior_weight: Some(1.0e6),
            ..base_config
        };
        let anchored_result =
            refine_visual_map_with_covisibility_ba(&mut anchored_map, 2, &anchored_config).unwrap();
        let anchored_displacement = displacement(
            &anchored_map,
            &anchored_result.selection.optimized_keyframe_ids,
        );

        assert!(
            unanchored_displacement > 1e-6,
            "expected the unanchored BA to move the window, got {unanchored_displacement}"
        );
        assert!(
            anchored_displacement < unanchored_displacement,
            "anchored displacement {anchored_displacement} should be smaller than unanchored {unanchored_displacement}"
        );
    }
}
