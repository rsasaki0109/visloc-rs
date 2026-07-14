//! Per-frame stereo landmark replenishment.
//!
//! The stereo *bootstrap* seeds the map once at the seed frame. As the
//! camera moves the map is only extended by whatever tracking re-observes,
//! so long runs slowly starve the tracker of fresh landmarks. This module
//! *replenishes* the map every frame: for each cam0 keypoint that tracking
//! did **not** match to an existing landmark, it stereo-matches against the
//! same-instant cam1 image (reusing [`bootstrap_stereo_landmarks`]),
//! triangulates the survivor, and stages a two-observation
//! [`LandmarkCandidate`] — the current frame's real detected pixel plus the
//! matched real keypoint in an already-existing keyframe (the "anchor"). The
//! provisional stereo point is used only to predict and gate that anchor
//! match; the [`crate::LinearTriangulator`] that runs at the next keyframe
//! receives two honest image measurements.
//!
//! The anchor association is the delicate part: naively fabricating it
//! (hardcoded keypoint index, no quality gates, no duplicate suppression) or
//! pairing a real keypoint index with a synthetic pixel pollutes the map with
//! bogus / over-confident landmarks. Every gate below exists to keep that
//! association honest and the triangulation well-conditioned; each is
//! config-gated with permissive defaults so the common case still admits
//! plenty of fresh landmarks.

use std::collections::{HashMap, HashSet};

use nalgebra::Point2;
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Camera, FrameId, VisualMap};
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::BruteForceMatcher;
use visloc_vision::stereo_bootstrap::{bootstrap_stereo_landmarks, StereoBootstrapConfig};

use crate::{LandmarkCandidate, LandmarkCandidateId, LandmarkCandidateObservation};

/// Tunable knobs for [`build_stereo_replenish_candidates`].
///
/// Defaults are deliberately **permissive**: replenishment only helps if it
/// actually admits new landmarks, so the gates are sized to reject
/// clearly-degenerate / duplicate geometry without strangling recall. Values
/// mirror [`StereoBootstrapConfig::default`]'s own gates (2.0 px reprojection,
/// 0.1–50.0 m depth) where the two overlap.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoReplenishConfig {
    /// Cap on accepted candidates returned per call. Survivors are processed
    /// in `left_keypoint_index` order (the deterministic order
    /// [`bootstrap_stereo_landmarks`] returns), so truncation is stable.
    pub max_candidates_per_frame: usize,
    /// Stereo match / triangulate / reprojection gates applied to the
    /// same-instant cam0/cam1 pair. Reused verbatim from the bootstrap path.
    pub bootstrap_config: StereoBootstrapConfig,
    /// Radius (px) around the reprojected anchor pixel within which a *real*
    /// detected anchor-keyframe keypoint must exist for the candidate to be
    /// accepted. The anchor observation uses both THAT keypoint's index and
    /// its measured pixel coordinate; the stereo reprojection remains only a
    /// search prediction. A few pixels of slack absorbs the accumulated
    /// anchor-pose / current-pose error; too tight would strangle recall.
    pub anchor_keypoint_match_radius_px: f64,
    /// Optional descriptor-distance gate between the candidate's cam0
    /// descriptor and the matched anchor keypoint's descriptor (L2, via
    /// [`BruteForceMatcher::l2_distance`]). `None` (default) disables it —
    /// the geometric radius match is usually enough, and requiring
    /// descriptor agreement across the (often large) anchor→current baseline
    /// noticeably cuts recall.
    pub anchor_keypoint_max_descriptor_distance: Option<f32>,
    /// Radius (px) for geometric duplicate suppression. Each of the anchor
    /// keyframe's *own* existing landmark observations is reprojected into
    /// the current frame using the current tracked pose; any stereo survivor
    /// whose raw cam0 pixel lands within this radius of such a reprojection
    /// is dropped as a probable duplicate of an already-mapped point. This
    /// catches both PnP-outlier-but-really-matched keypoints and keypoints
    /// the descriptor matcher missed entirely (the more likely driver of
    /// duplicate landmarks). Kept small so genuinely new nearby structure
    /// still survives.
    pub duplicate_suppression_radius_px: f64,
    /// Minimum parallax angle (degrees) between the ray from the anchor
    /// camera centre to the triangulated world point and the ray from the
    /// current camera centre to the same point. The downstream two-view
    /// [`crate::LinearTriangulator`] re-triangulates the candidate from
    /// exactly these two views, so a near-zero angle (stationary / tiny
    /// baseline) yields a degenerate, depth-unstable solve. Small permissive
    /// default; only rejects the clearly-degenerate tail.
    pub min_parallax_deg: f64,
    /// Minimum triangulated depth (m) in the current cam0 frame. Independent,
    /// tunable re-check of [`StereoBootstrapConfig::min_depth_meters`].
    pub min_depth_meters: f64,
    /// Maximum triangulated depth (m) in the current cam0 frame. Independent,
    /// tunable re-check of [`StereoBootstrapConfig::max_depth_meters`].
    pub max_depth_meters: f64,
}

impl Default for StereoReplenishConfig {
    fn default() -> Self {
        Self {
            // Matches the demo's historical `--stereo-landmark-replenish-max-per-frame`.
            max_candidates_per_frame: 100,
            bootstrap_config: StereoBootstrapConfig::default(),
            // A few px of slack: the reprojected anchor pixel is only as
            // accurate as the anchor + current pose estimates, so a strict
            // 1-2 px match would reject good candidates. 5 px keeps a genuine
            // nearby detection requirement without over-rejecting.
            anchor_keypoint_match_radius_px: 5.0,
            // Off by default (see field doc): descriptor agreement across the
            // anchor→current baseline is a weak, recall-cutting signal.
            anchor_keypoint_max_descriptor_distance: None,
            // 3 px: two detections this close to the same reprojected existing
            // landmark are almost certainly the same physical point; larger
            // would suppress legitimate new structure.
            duplicate_suppression_radius_px: 3.0,
            // 0.5°: rejects only the near-stationary degenerate tail while
            // keeping most genuine motion-parallax points (0.5° at 5 m depth
            // is ~4 cm baseline).
            min_parallax_deg: 0.5,
            // Mirror the bootstrap depth window.
            min_depth_meters: 0.1,
            max_depth_meters: 50.0,
        }
    }
}

/// Triangulate same-instant stereo features into metric cam0-frame points,
/// keyed by cam0 keypoint index. Unlike replenishment candidate construction,
/// this requires neither a tracked pose nor an anchor keyframe and can run
/// before online tracking/loop closure.
pub fn build_stereo_metric_points(
    cam0_camera: &Camera,
    cam1_camera: &Camera,
    cam0_to_cam1: &SE3,
    cam0_features: &FeatureSet,
    cam1_features: &FeatureSet,
    config: &StereoReplenishConfig,
) -> HashMap<usize, nalgebra::Point3<f64>> {
    bootstrap_stereo_landmarks(
        cam0_camera,
        cam1_camera,
        cam0_to_cam1,
        cam0_features,
        cam1_features,
        &config.bootstrap_config,
    )
    .into_iter()
    .filter(|survivor| {
        let depth = survivor.point_left_camera_frame.z;
        depth.is_finite() && depth >= config.min_depth_meters && depth <= config.max_depth_meters
    })
    .map(|survivor| {
        (
            survivor.left_keypoint_index,
            survivor.point_left_camera_frame,
        )
    })
    .collect()
}

/// Build stereo-replenishment [`LandmarkCandidate`]s for the current frame.
///
/// For each cam0 keypoint that tracking did not match this frame
/// (`matched_cam0_indices` are the PnP-inlier query indices to exclude),
/// stereo-match against `cam1_features`, triangulate, gate, and — for
/// survivors — stage a two-observation candidate: the current frame's real
/// detection plus the nearest gated real detection in the
/// `anchor_frame_id` keyframe. The stereo reprojection is a prediction, not a
/// replacement measurement.
///
/// Returns an empty `Vec` (bails cleanly) when the anchor keyframe, its
/// pose, its camera, or its keypoints are missing/empty — mirroring the
/// guard clauses of the original inline demo block.
///
/// Candidate ids are assigned sequentially starting at `next_candidate_id`.
///
/// # Gate order (per survivor)
/// 1. matched-index exclusion (skip keypoints tracking already used),
/// 2. geometric duplicate suppression (skip near an existing landmark's
///    reprojection),
/// 3. per-frame cap,
/// 4. world point (current-pose camera-to-world) + anchor reprojection,
/// 5. anchor in-front + image-bounds gate,
/// 6. current-frame depth window,
/// 7. anchor↔current parallax,
/// 8. nearest real anchor keypoint (excluding already-claimed indices) with
///    optional descriptor gate.
///
/// # Note on exclusion scope
/// `matched_cam0_indices` only carries PnP *inliers* (the pre-RANSAC match
/// set is not threaded out of localization), so a keypoint that really
/// matched an existing landmark but was a PnP outlier is not in the set.
/// The geometric duplicate-suppression gate (step 2) is what actually
/// catches those, plus matcher misses — but only against the anchor
/// keyframe's *own* observations, kept deliberately cheap: duplicates only
/// visible via other nearby keyframes are not suppressed here.
#[allow(clippy::too_many_arguments)]
pub fn build_stereo_replenish_candidates(
    map: &VisualMap,
    anchor_frame_id: FrameId,
    frame_id: FrameId,
    cam1_camera: &Camera,
    cam0_to_cam1: &SE3,
    cam0_features: &FeatureSet,
    cam1_features: &FeatureSet,
    matched_cam0_indices: &HashSet<usize>,
    current_pose: &Pose,
    next_candidate_id: LandmarkCandidateId,
    config: &StereoReplenishConfig,
) -> Vec<LandmarkCandidate> {
    // Guard clauses: bail cleanly if the anchor keyframe context is missing.
    let Some(anchor_keyframe) = map.keyframes.get(&anchor_frame_id) else {
        return Vec::new();
    };
    let Some(anchor_pose) = anchor_keyframe.frame.pose.as_ref() else {
        return Vec::new();
    };
    // The anchor keyframe's camera is the same cam0 model used to triangulate
    // and to reproject into both the anchor and the current frame.
    let Some(cam0_camera) = map.cameras.get(&anchor_keyframe.frame.camera_id) else {
        return Vec::new();
    };
    if anchor_keyframe.frame.keypoints.is_empty() {
        return Vec::new();
    }

    let survivors = bootstrap_stereo_landmarks(
        cam0_camera,
        cam1_camera,
        cam0_to_cam1,
        cam0_features,
        cam1_features,
        &config.bootstrap_config,
    );
    if survivors.is_empty() {
        return Vec::new();
    }

    // Precompute the anchor keyframe's own existing landmarks reprojected into
    // the CURRENT frame (for geometric duplicate suppression), plus the set of
    // anchor keypoint indices already claimed by an existing observation (so a
    // newly associated anchor observation never collides with one). We seed
    // `claimed_anchor_indices` with the existing observations and extend it as
    // we accept candidates, so two survivors in this same call cannot both
    // claim the same anchor keypoint either.
    let mut existing_reprojections: Vec<Point2<f64>> =
        Vec::with_capacity(anchor_keyframe.observations.len());
    let mut claimed_anchor_indices: HashSet<usize> =
        HashSet::with_capacity(anchor_keyframe.observations.len());
    for observation in &anchor_keyframe.observations {
        claimed_anchor_indices.insert(observation.keypoint_index);
        if let Some(landmark) = map.landmarks.get(&observation.landmark_id) {
            let point_current = current_pose.transform_world_point(&landmark.position);
            if let Some(pixel) = cam0_camera.project(&point_current) {
                existing_reprojections.push(pixel);
            }
        }
    }

    let camera_to_world = current_pose.camera_to_world();
    let anchor_center = anchor_pose.camera_center_world();
    let current_center = current_pose.camera_center_world();
    let min_parallax_cos = config.min_parallax_deg.to_radians().cos();
    let width = cam0_camera.width as f64;
    let height = cam0_camera.height as f64;

    let mut candidates: Vec<LandmarkCandidate> = Vec::new();
    let mut next_id = next_candidate_id;

    for survivor in &survivors {
        // (3) Per-frame cap. Break rather than continue: survivors are in a
        // stable order, so the accepted prefix is deterministic.
        if candidates.len() >= config.max_candidates_per_frame {
            break;
        }

        // (1) Skip keypoints tracking already matched to an existing landmark.
        if matched_cam0_indices.contains(&survivor.left_keypoint_index) {
            continue;
        }

        let raw_left_pixel = cam0_features.keypoints[survivor.left_keypoint_index];

        // (2) Geometric duplicate suppression against the anchor keyframe's
        // own landmarks reprojected into this frame.
        if existing_reprojections.iter().any(|reprojection| {
            (reprojection - raw_left_pixel).norm() <= config.duplicate_suppression_radius_px
        }) {
            continue;
        }

        // (4) Lift the triangulated left-frame point to the world, then into
        // the anchor camera.
        let world_point = camera_to_world.transform_point(&survivor.point_left_camera_frame);
        let anchor_camera_point = anchor_pose.transform_world_point(&world_point);

        // (5) In front of the anchor camera. `project` also returns `None` for
        // z <= 0, but we check explicitly so the intent is documented and not
        // an implicit side effect.
        if anchor_camera_point.z <= 0.0 {
            continue;
        }
        let Some(anchor_xy) = cam0_camera.project(&anchor_camera_point) else {
            continue;
        };
        // Anchor reprojection must land inside the anchor image (cheap
        // early-out before the nearest-keypoint search).
        if anchor_xy.x < 0.0 || anchor_xy.x > width || anchor_xy.y < 0.0 || anchor_xy.y > height {
            continue;
        }

        // (6) Current-frame depth window (independent re-check of the bootstrap
        // depth gate).
        let depth = survivor.point_left_camera_frame.z;
        if !depth.is_finite() || depth < config.min_depth_meters || depth > config.max_depth_meters
        {
            continue;
        }

        // (7) Parallax between anchor and current views of this world point.
        let anchor_ray = world_point - anchor_center;
        let current_ray = world_point - current_center;
        let anchor_ray_norm = anchor_ray.norm();
        let current_ray_norm = current_ray.norm();
        if anchor_ray_norm <= f64::EPSILON || current_ray_norm <= f64::EPSILON {
            continue;
        }
        let cos_parallax = anchor_ray.dot(&current_ray) / (anchor_ray_norm * current_ray_norm);
        // cos is monotonically decreasing in angle, so "angle > min" ⇔
        // "cos < cos(min)". Clamp guards against fp overshoot past ±1.
        if cos_parallax.clamp(-1.0, 1.0) >= min_parallax_cos {
            continue;
        }

        // (8) Nearest real anchor keypoint within radius, excluding indices
        // already claimed (existing observations + earlier accepted
        // candidates this call).
        let mut best_index: Option<usize> = None;
        let mut best_distance = config.anchor_keypoint_match_radius_px;
        for (index, keypoint) in anchor_keyframe.frame.keypoints.iter().enumerate() {
            if claimed_anchor_indices.contains(&index) {
                continue;
            }
            let distance = (keypoint - anchor_xy).norm();
            if distance <= best_distance {
                best_distance = distance;
                best_index = Some(index);
            }
        }
        let Some(anchor_keypoint_index) = best_index else {
            // No honest real anchor keypoint nearby: drop rather than fabricate
            // an index (defect (a)).
            continue;
        };
        let anchor_measured_xy = anchor_keyframe.frame.keypoints[anchor_keypoint_index];

        let candidate_descriptor = &cam0_features.descriptors[survivor.left_keypoint_index];

        // Optional descriptor-distance gate on the associated anchor keypoint.
        if let Some(max_descriptor_distance) = config.anchor_keypoint_max_descriptor_distance {
            let Some(anchor_descriptor) =
                anchor_keyframe.frame.descriptors.get(anchor_keypoint_index)
            else {
                continue;
            };
            match BruteForceMatcher::l2_distance(candidate_descriptor, anchor_descriptor) {
                Some(distance) if distance <= max_descriptor_distance => {}
                _ => continue,
            }
        }

        claimed_anchor_indices.insert(anchor_keypoint_index);
        let candidate_id = next_id;
        next_id += 1;
        candidates.push(
            LandmarkCandidate::new(candidate_id)
                .with_observation(LandmarkCandidateObservation::new(
                    frame_id,
                    survivor.left_keypoint_index,
                    raw_left_pixel,
                ))
                .with_observation(LandmarkCandidateObservation::new(
                    anchor_frame_id,
                    anchor_keypoint_index,
                    anchor_measured_xy,
                ))
                .with_descriptor(candidate_descriptor.clone()),
        );
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_core::types::{Frame, Keyframe, Landmark, Observation};

    fn cam0() -> Camera {
        Camera::pinhole(1, 752, 480, 458.0, 457.0, 367.0, 248.0)
    }

    /// cam0→cam1 pure-translation baseline along cam0 +x (EuRoC-style nominal).
    fn cam0_to_cam1() -> SE3 {
        SE3::new(UnitQuaternion::identity(), Vector3::new(-0.11, 0.0, 0.0))
    }

    /// Identity-pose (world == camera) anchor keyframe with `keypoint_count`
    /// keypoints laid out on a horizontal line, no observations.
    fn anchor_keyframe(frame_id: u64, keypoints: Vec<Point2<f64>>) -> Keyframe {
        let mut frame = Frame::new(frame_id, 1);
        frame.descriptors = keypoints.iter().map(|_| vec![1.0_f32, 0.0]).collect();
        frame.keypoints = keypoints;
        frame.pose = Some(Pose::identity());
        Keyframe {
            frame,
            observations: Vec::new(),
        }
    }

    fn base_map(anchor: Keyframe) -> VisualMap {
        let mut map = VisualMap::new();
        map.cameras.insert(1, cam0());
        map.keyframes.insert(anchor.frame.id, anchor);
        map
    }

    /// Project a world point through an identity-pose cam0 to get its pixel.
    fn project_identity(point: &Point3<f64>) -> Point2<f64> {
        cam0().project(point).expect("in front of camera")
    }

    /// Build a synthetic same-instant cam0/cam1 stereo pair for `world_points`
    /// viewed from `current_pose`, with one-hot descriptors so cross-check
    /// matching is unambiguous. Returns `(cam0_features, cam1_features)`.
    fn stereo_features(
        world_points: &[Point3<f64>],
        current_pose: &Pose,
    ) -> (FeatureSet, FeatureSet) {
        let cam = cam0();
        let l2r = cam0_to_cam1();
        let mut cam0_kps = Vec::new();
        let mut cam1_kps = Vec::new();
        let mut cam0_desc = Vec::new();
        let mut cam1_desc = Vec::new();
        for (i, world) in world_points.iter().enumerate() {
            let point_left = current_pose.transform_world_point(world);
            let point_right = l2r.transform_point(&point_left);
            cam0_kps.push(cam.project(&point_left).expect("left in front"));
            cam1_kps.push(cam.project(&point_right).expect("right in front"));
            let mut desc = vec![0.0_f32; world_points.len()];
            desc[i] = 1.0;
            cam0_desc.push(desc.clone());
            cam1_desc.push(desc);
        }
        (
            FeatureSet::new(cam0_kps, cam0_desc).unwrap(),
            FeatureSet::new(cam1_kps, cam1_desc).unwrap(),
        )
    }

    /// Current pose translated `tx` metres along world +x from the anchor
    /// (identity), so anchor↔current has a real baseline.
    fn current_pose_shifted(tx: f64) -> Pose {
        // world_to_camera translation of -tx puts the camera centre at +tx.
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-tx, 0.0, 0.0))
    }

    #[test]
    fn happy_path_builds_candidate_with_real_anchor_keypoint_index() {
        let current = current_pose_shifted(0.3);
        // One world point well in front of both anchor and current cameras.
        let world = Point3::new(0.2, -0.1, 4.0);
        let (cam0_features, cam1_features) = stereo_features(&[world], &current);

        // Anchor keypoint list: put the real detection exactly where `world`
        // reprojects into the identity-pose anchor, at index 2 (NOT 0), plus
        // two decoys far away.
        let anchor_pixel = project_identity(&world);
        let anchor = anchor_keyframe(
            10,
            vec![
                Point2::new(5.0, 5.0),
                Point2::new(700.0, 400.0),
                anchor_pixel,
            ],
        );
        let map = base_map(anchor);

        let candidates = build_stereo_replenish_candidates(
            &map,
            10,
            42,
            &cam0(),
            &cam0_to_cam1(),
            &cam0_features,
            &cam1_features,
            &HashSet::new(),
            &current,
            1000,
            &StereoReplenishConfig::default(),
        );

        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert_eq!(candidate.id, 1000);
        assert_eq!(candidate.observations.len(), 2);
        // Current-frame real observation.
        assert_eq!(candidate.observations[0].frame_id, 42);
        assert_eq!(candidate.observations[0].keypoint_index, 0);
        // Anchor observation uses the REAL nearest detection at index 2.
        assert_eq!(candidate.observations[1].frame_id, 10);
        assert_eq!(candidate.observations[1].keypoint_index, 2);
        assert_eq!(candidate.observations[1].xy, anchor_pixel);
        assert!(candidate.descriptor.is_some());
    }

    #[test]
    fn anchor_observation_uses_measured_pixel_not_stereo_prediction() {
        let current = current_pose_shifted(0.3);
        let world = Point3::new(0.2, -0.1, 4.0);
        let (cam0_features, cam1_features) = stereo_features(&[world], &current);
        let predicted = project_identity(&world);
        // Stay inside the 5 px association radius while making the measured
        // coordinate observably different from the stereo prediction.
        let measured = predicted + nalgebra::Vector2::new(1.25, -0.75);
        let map = base_map(anchor_keyframe(10, vec![measured]));

        let candidates = build_stereo_replenish_candidates(
            &map,
            10,
            42,
            &cam0(),
            &cam0_to_cam1(),
            &cam0_features,
            &cam1_features,
            &HashSet::new(),
            &current,
            1000,
            &StereoReplenishConfig::default(),
        );

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].observations[1].keypoint_index, 0);
        assert_eq!(candidates[0].observations[1].xy, measured);
        assert_ne!(candidates[0].observations[1].xy, predicted);
    }

    #[test]
    fn matched_index_is_excluded() {
        let current = current_pose_shifted(0.3);
        let world = Point3::new(0.2, -0.1, 4.0);
        let (cam0_features, cam1_features) = stereo_features(&[world], &current);
        let anchor_pixel = project_identity(&world);
        let anchor = anchor_keyframe(10, vec![anchor_pixel]);
        let map = base_map(anchor);

        // Mark cam0 keypoint 0 as already tracked → no candidate.
        let mut matched = HashSet::new();
        matched.insert(0usize);

        let candidates = build_stereo_replenish_candidates(
            &map,
            10,
            42,
            &cam0(),
            &cam0_to_cam1(),
            &cam0_features,
            &cam1_features,
            &matched,
            &current,
            1000,
            &StereoReplenishConfig::default(),
        );
        assert!(candidates.is_empty());
    }

    #[test]
    fn duplicate_suppression_drops_survivor_near_existing_landmark() {
        let current = current_pose_shifted(0.3);
        let world = Point3::new(0.2, -0.1, 4.0);
        let (cam0_features, cam1_features) = stereo_features(&[world], &current);
        let anchor_pixel = project_identity(&world);
        let mut anchor = anchor_keyframe(10, vec![anchor_pixel]);

        // Existing landmark at the same world point → its current-frame
        // reprojection coincides with the survivor's raw cam0 pixel.
        let mut map = base_map(anchor.clone());
        map.landmarks.insert(500, Landmark::new(500, world));
        anchor.observations.push(Observation {
            frame_id: 10,
            landmark_id: 500,
            keypoint_index: 0,
            xy: anchor_pixel,
        });
        map.keyframes.insert(10, anchor);

        let candidates = build_stereo_replenish_candidates(
            &map,
            10,
            42,
            &cam0(),
            &cam0_to_cam1(),
            &cam0_features,
            &cam1_features,
            &HashSet::new(),
            &current,
            1000,
            &StereoReplenishConfig::default(),
        );
        assert!(candidates.is_empty());
    }

    #[test]
    fn parallax_gate_drops_zero_baseline_geometry() {
        // Anchor == current pose (both identity) → zero anchor↔current
        // parallax → degenerate two-view geometry → dropped.
        let current = Pose::identity();
        let world = Point3::new(0.2, -0.1, 4.0);
        let (cam0_features, cam1_features) = stereo_features(&[world], &current);
        let anchor_pixel = project_identity(&world);
        let anchor = anchor_keyframe(10, vec![anchor_pixel]);
        let map = base_map(anchor);

        let candidates = build_stereo_replenish_candidates(
            &map,
            10,
            42,
            &cam0(),
            &cam0_to_cam1(),
            &cam0_features,
            &cam1_features,
            &HashSet::new(),
            &current,
            1000,
            &StereoReplenishConfig::default(),
        );
        assert!(
            candidates.is_empty(),
            "zero anchor/current baseline must fail the parallax gate"
        );
    }

    #[test]
    fn anchor_reprojection_out_of_bounds_is_dropped() {
        // World point in front of the current camera but whose reprojection
        // into the anchor camera lands off-image. Achieved by making the
        // anchor look the other way (180° yaw) so the point is behind /
        // projects out of frame.
        let current = current_pose_shifted(0.3);
        let world = Point3::new(0.2, -0.1, 4.0);
        let (cam0_features, cam1_features) = stereo_features(&[world], &current);

        // Anchor rotated 180° about y: the world point is now behind it.
        let mut anchor = anchor_keyframe(10, vec![Point2::new(367.0, 248.0)]);
        anchor.frame.pose = Some(Pose::from_world_to_camera(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::PI),
            Vector3::zeros(),
        ));
        let map = base_map(anchor);

        let candidates = build_stereo_replenish_candidates(
            &map,
            10,
            42,
            &cam0(),
            &cam0_to_cam1(),
            &cam0_features,
            &cam1_features,
            &HashSet::new(),
            &current,
            1000,
            &StereoReplenishConfig::default(),
        );
        assert!(
            candidates.is_empty(),
            "point behind the anchor camera must be dropped"
        );
    }

    #[test]
    fn no_anchor_keypoint_within_radius_is_dropped() {
        let current = current_pose_shifted(0.3);
        let world = Point3::new(0.2, -0.1, 4.0);
        let (cam0_features, cam1_features) = stereo_features(&[world], &current);
        // Anchor's only keypoint is far from where `world` reprojects.
        let anchor = anchor_keyframe(10, vec![Point2::new(5.0, 5.0)]);
        let map = base_map(anchor);

        let candidates = build_stereo_replenish_candidates(
            &map,
            10,
            42,
            &cam0(),
            &cam0_to_cam1(),
            &cam0_features,
            &cam1_features,
            &HashSet::new(),
            &current,
            1000,
            &StereoReplenishConfig::default(),
        );
        assert!(
            candidates.is_empty(),
            "no real anchor keypoint nearby → no fabricated index"
        );
    }

    #[test]
    fn per_frame_cap_truncates_returned_candidates() {
        let current = current_pose_shifted(0.3);
        // Three well-separated world points.
        let worlds = [
            Point3::new(-0.4, 0.1, 4.0),
            Point3::new(0.0, -0.2, 5.0),
            Point3::new(0.5, 0.2, 3.5),
        ];
        let (cam0_features, cam1_features) = stereo_features(&worlds, &current);
        // Anchor keypoints exactly where each world point reprojects.
        let anchor_pixels: Vec<Point2<f64>> = worlds.iter().map(project_identity).collect();
        let anchor = anchor_keyframe(10, anchor_pixels);
        let map = base_map(anchor);

        let config = StereoReplenishConfig {
            max_candidates_per_frame: 2,
            ..StereoReplenishConfig::default()
        };
        let candidates = build_stereo_replenish_candidates(
            &map,
            10,
            42,
            &cam0(),
            &cam0_to_cam1(),
            &cam0_features,
            &cam1_features,
            &HashSet::new(),
            &current,
            2000,
            &config,
        );
        assert_eq!(candidates.len(), 2, "cap must truncate to 2");
        // Sequential ids from the start value.
        assert_eq!(candidates[0].id, 2000);
        assert_eq!(candidates[1].id, 2001);
        // Distinct real anchor keypoint indices (no intra-call collision).
        assert_ne!(
            candidates[0].observations[1].keypoint_index,
            candidates[1].observations[1].keypoint_index
        );
    }

    #[test]
    fn missing_anchor_keyframe_returns_empty() {
        let current = current_pose_shifted(0.3);
        let world = Point3::new(0.2, -0.1, 4.0);
        let (cam0_features, cam1_features) = stereo_features(&[world], &current);
        let map = VisualMap::new(); // no keyframes / cameras

        let candidates = build_stereo_replenish_candidates(
            &map,
            10,
            42,
            &cam0(),
            &cam0_to_cam1(),
            &cam0_features,
            &cam1_features,
            &HashSet::new(),
            &current,
            1000,
            &StereoReplenishConfig::default(),
        );
        assert!(candidates.is_empty());
    }
}
