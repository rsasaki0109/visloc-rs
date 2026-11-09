//! EuRoC calibration conversion, the seed visual map, stereo bootstrap
//! landmarks and map snapshots.

use super::*;

/// Decompose an EuRoC `T_BS` 4×4 body-to-sensor matrix into an [`SE3`].
#[cfg(feature = "image-io")]
pub(super) fn se3_from_t_bs(t_bs: &Matrix4<f64>) -> SE3 {
    let rotation_matrix = t_bs.fixed_view::<3, 3>(0, 0).into_owned();
    let translation = Vector3::new(t_bs[(0, 3)], t_bs[(1, 3)], t_bs[(2, 3)]);
    let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    SE3::new(rotation, translation)
}

/// Build a [`Camera`] from EuRoC's pinhole `(fu, fv, cu, cv)` intrinsics.
/// Distortion coefficients are intentionally ignored (see file docs).
#[cfg(feature = "image-io")]
pub(super) fn camera_from_cam0(cam0: &EurocCameraCalibration, camera_id: u64) -> Camera {
    let (fu, fv, cu, cv) = (
        cam0.intrinsics[0],
        cam0.intrinsics[1],
        cam0.intrinsics[2],
        cam0.intrinsics[3],
    );
    Camera::pinhole(
        camera_id,
        cam0.resolution.0,
        cam0.resolution.1,
        fu,
        fv,
        cu,
        cv,
    )
}

/// Back-project a pixel into the world frame at fixed depth, given the
/// world-to-camera pose. The ray from the camera centre through `pixel`,
/// taken to camera-frame depth `depth_meters`, is mapped into the world
/// frame via the inverse of the supplied pose.
#[cfg(feature = "image-io")]
pub(super) fn back_project_pixel_to_world(
    camera: &Camera,
    pose_world_to_camera: &Pose,
    pixel: &Point2<f64>,
    depth_meters: f64,
) -> Option<Point3<f64>> {
    let normalized = camera.normalize_pixel(pixel)?;
    let p_cam = Point3::new(
        normalized.x * depth_meters,
        normalized.y * depth_meters,
        depth_meters,
    );
    let r_cw = pose_world_to_camera.world_to_camera.rotation;
    let t_cw = pose_world_to_camera.world_to_camera.translation;
    let r_wc = r_cw.inverse();
    // World point = R_cw⁻¹ · (p_cam - t_cw).
    Some(r_wc.transform_point(&Point3::from(p_cam.coords - t_cw)))
}

/// Cam1 camera model, cam0-to-cam1 extrinsic, and cam1 distortion model,
/// computed once and shared by the seed-frame stereo bootstrap and the
/// per-frame stereo landmark replenishment path.
#[cfg(feature = "image-io")]
pub(super) struct Cam1StereoSetup {
    pub(super) camera: Camera,
    pub(super) cam0_to_cam1: SE3,
    pub(super) distortion: RadialTangential,
}

/// Build a [`VisualMap`] by seeding one landmark per keypoint in
/// `features`. When `stereo_world_points[i]` is `Some(world_point)`, that
/// metric-scale 3D point is used directly (typically the output of a
/// stereo-triangulation bootstrap); otherwise the keypoint is
/// back-projected through `pose_world_to_camera` at
/// `bootstrap_depth_meters` as the fixed-depth fall-back. The corner
/// descriptor is attached to every seeded landmark.
///
/// `stereo_world_points` must be either empty (no override) or the same
/// length as `features.keypoints`.
#[cfg(feature = "image-io")]
pub(super) fn bootstrap_map_from_first_frame(
    camera: &Camera,
    pose_world_to_camera: &Pose,
    features: &FeatureSet,
    bootstrap_depth_meters: f64,
    stereo_world_points: &[Option<Point3<f64>>],
    stereo_world_covariances: &[Option<nalgebra::Matrix3<f64>>],
    strict_stereo: bool,
) -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let use_overrides = !stereo_world_points.is_empty();
    if use_overrides {
        debug_assert_eq!(stereo_world_points.len(), features.keypoints.len());
        debug_assert_eq!(stereo_world_covariances.len(), features.keypoints.len());
    }
    for (index, (keypoint, descriptor)) in features
        .keypoints
        .iter()
        .zip(features.descriptors.iter())
        .enumerate()
    {
        let world_point = if use_overrides && stereo_world_points[index].is_some() {
            stereo_world_points[index].unwrap()
        } else if strict_stereo {
            // Phase-23 #2: strict-stereo mode drops keypoints without a
            // triangulated cam0↔cam1 depth, refusing to fall back to
            // the fixed `bootstrap_depth_meters` back-projection. The
            // resulting map is smaller but every landmark has a real
            // metric depth, which empirically lifts recovery-PnP
            // quality (post-cliff recovery, loop-closure verifier) by
            // avoiding the cross-attitude descriptor mismatch that
            // dominates when stale fixed-depth landmarks pollute the
            // matcher.
            continue;
        } else {
            let Some(point) = back_project_pixel_to_world(
                camera,
                pose_world_to_camera,
                keypoint,
                bootstrap_depth_meters,
            ) else {
                continue;
            };
            point
        };
        let mut landmark = Landmark::new(index as u64 + 1, world_point);
        landmark.descriptor = Some(descriptor.clone());
        if use_overrides {
            if let Some(covariance_world) = stereo_world_covariances[index] {
                map.landmark_position_covariances
                    .insert(landmark.id, covariance_world);
            }
        }
        map.landmarks.insert(landmark.id, landmark);
    }
    map
}

/// Mirror of [`OnlineSlamPipeline`]'s private `keyframe_from_tracking_result`.
#[cfg(feature = "image-io")]
pub(super) fn keyframe_from_tracking_result(frame: &Frame, tracking: &TrackingResult) -> Keyframe {
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

/// Preserve the last successfully tracked frame at an Atlas boundary even
/// when the online keyframe policy did not promote it. DROID-style frame
/// graphs retain this overlap; dropping it can leave the next independent
/// submap with only a much older target keyframe.
#[cfg(feature = "image-io")]
pub(super) fn insert_atlas_boundary_keyframe_snapshot(
    map: &mut VisualMap,
    keyframe: &Keyframe,
) -> bool {
    if map.keyframes.contains_key(&keyframe.frame.id) || keyframe.observations.is_empty() {
        return false;
    }

    let mut snapshot = keyframe.clone();
    snapshot
        .observations
        .retain(|observation| map.landmarks.contains_key(&observation.landmark_id));
    if snapshot.observations.is_empty() {
        return false;
    }
    for observation in &snapshot.observations {
        let landmark = map
            .landmarks
            .get_mut(&observation.landmark_id)
            .expect("snapshot observations were filtered against this map");
        if !landmark.observations.iter().any(|existing| {
            existing.frame_id == observation.frame_id
                && existing.landmark_id == observation.landmark_id
        }) {
            landmark.observations.push(observation.clone());
        }
    }
    map.keyframes.insert(snapshot.frame.id, snapshot);
    true
}

/// Append stereo-triangulated landmarks to an existing map. Returns the
/// new landmark ids in `matches` order.
#[cfg(feature = "image-io")]
pub(super) fn append_stereo_bootstrap_landmarks_to_map(
    map: &mut VisualMap,
    pose_world_to_camera: &Pose,
    features: &FeatureSet,
    matches: &[StereoBootstrapLandmark],
    next_landmark_id: &mut u64,
) -> Vec<u64> {
    let cam0_pose_camera_to_world = pose_world_to_camera.camera_to_world();
    let mut landmark_ids = Vec::with_capacity(matches.len());
    for survivor in matches {
        let world_point =
            cam0_pose_camera_to_world.transform_point(&survivor.point_left_camera_frame);
        let id = *next_landmark_id;
        *next_landmark_id += 1;
        let mut landmark = Landmark::new(id, world_point);
        landmark.descriptor = Some(features.descriptors[survivor.left_keypoint_index].clone());
        let rotation_camera_to_world = cam0_pose_camera_to_world
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let covariance_world = rotation_camera_to_world
            * survivor.point_covariance_left_camera_frame
            * rotation_camera_to_world.transpose();
        map.landmark_position_covariances
            .insert(id, covariance_world);
        map.landmarks.insert(id, landmark);
        landmark_ids.push(id);
    }
    landmark_ids
}

/// Build a successful [`TrackingResult`] for a stereo segment restart.
#[cfg(feature = "image-io")]
pub(super) fn build_stereo_segment_restart_tracking_result(
    frame_id: u64,
    seed_pose: Pose,
    matches: &[StereoBootstrapLandmark],
    landmark_ids: &[u64],
    map_stats: MapProviderStats,
) -> TrackingResult {
    let inlier_count = matches.len();
    let inlier_query_indices: Vec<usize> = matches.iter().map(|m| m.left_keypoint_index).collect();
    let inlier_landmark_ids = landmark_ids.to_vec();
    let localization = LocalizationResult {
        success: true,
        pose: Some(seed_pose.clone()),
        failure_reason: None,
        candidate_landmark_count: inlier_count,
        match_count: inlier_count,
        correspondence_count: inlier_count,
        inlier_count,
        outlier_count: 0,
        inlier_ratio: 1.0,
        reprojection_error: Some(0.0),
        median_reprojection_error: Some(0.0),
        max_reprojection_error: Some(0.0),
        inlier_reprojection_errors: vec![0.0; inlier_count],
        inliers: (0..inlier_count).collect(),
        inlier_query_indices,
        inlier_landmark_ids,
        inlier_confidences: vec![None; inlier_count],
        estimator_diagnostics: None,
        pose_failure_diagnostics: None,
    };
    TrackingResult {
        frame_id,
        state: TrackingState::Tracking,
        event: TrackingEvent::Relocalized,
        successive_failures: 0,
        pose_prior: Some(seed_pose),
        used_pose_prior: false,
        used_external_localization_prior: false,
        external_localization_prior_radius: None,
        tracking_failure_reason: None,
        map_landmark_count: map_stats.landmark_count,
        map_stats,
        localization,
        covisibility_local_map_size: None,
    }
}

#[cfg(feature = "image-io")]
pub(super) fn map_provider_stats_from_map(map: &VisualMap) -> MapProviderStats {
    MapProviderStats {
        camera_count: map.cameras.len(),
        landmark_count: map.landmarks.len(),
        keyframe_count: map.keyframes.len(),
        descriptor_count: map
            .landmarks
            .values()
            .filter(|landmark| landmark.descriptor.is_some())
            .count(),
    }
}
