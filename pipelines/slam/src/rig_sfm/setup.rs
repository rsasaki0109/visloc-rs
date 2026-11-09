//! Input validation and the image / track indices, metric seed supports, pose
//! installation and verified stereo links the drivers build.

use super::*;

pub(super) fn validate_inputs(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &RigSfmConfig,
) -> Result<(), RigSfmError> {
    if let Some(frame) = config.seed_frame {
        if frame >= frames.len() {
            return Err(RigSfmError::InvalidSeedFrame {
                frame,
                frame_count: frames.len(),
            });
        }
    }
    if !(1..=rig.sensors().len()).contains(&config.min_pnp_sensors) {
        return Err(RigSfmError::InvalidMinPnpSensors {
            requested: config.min_pnp_sensors,
            sensor_count: rig.sensors().len(),
        });
    }
    if let Some(requested) = config.direct_stereo_min_pnp_sensors {
        if !(1..=rig.sensors().len()).contains(&requested) {
            return Err(RigSfmError::InvalidDirectStereoMinPnpSensors {
                requested,
                sensor_count: rig.sensors().len(),
            });
        }
    }
    if !config.direct_stereo_min_triangulation_angle_deg.is_finite()
        || config.direct_stereo_min_triangulation_angle_deg <= 0.0
    {
        return Err(RigSfmError::InvalidDirectStereoTriangulationAngle(
            config.direct_stereo_min_triangulation_angle_deg,
        ));
    }
    if !config.triangulation_min_inlier_fraction.is_finite()
        || !(0.5..=1.0).contains(&config.triangulation_min_inlier_fraction)
    {
        return Err(RigSfmError::InvalidTriangulationInlierFraction(
            config.triangulation_min_inlier_fraction,
        ));
    }
    for (name, value) in [
        (
            "maximum reprojection error",
            config.conflict_recovery_max_reprojection_error_px,
        ),
        (
            "maximum mean reprojection error",
            config.conflict_recovery_max_mean_reprojection_px,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(RigSfmError::InvalidConflictRecoveryGate { name, value });
        }
    }
    if config.motion_bridge_min_inliers < 8 {
        return Err(RigSfmError::InvalidMotionBridgeMinInliers(
            config.motion_bridge_min_inliers,
        ));
    }
    if !config.motion_bridge_max_rotation_deviation_deg.is_finite()
        || config.motion_bridge_max_rotation_deviation_deg <= 0.0
    {
        return Err(RigSfmError::InvalidMotionBridgeRotationDeviation(
            config.motion_bridge_max_rotation_deviation_deg,
        ));
    }
    if let Some(prefix) = config.deferred_registration_pair_prefix {
        if prefix > pairwise.len() {
            return Err(RigSfmError::DeferredPairPrefix {
                prefix,
                pair_count: pairwise.len(),
            });
        }
    }
    if let Some(prefix) = config.deferred_retriangulation_pair_prefix {
        let registration_prefix = config
            .deferred_registration_pair_prefix
            .unwrap_or(pairwise.len());
        if prefix < registration_prefix || prefix > pairwise.len() {
            return Err(RigSfmError::DeferredRetriangulationPairPrefix {
                prefix,
                registration_prefix,
                pair_count: pairwise.len(),
            });
        }
    }
    if config.deferred_retriangulation_quadrilateral_min_frame_gap == 0
        || config.deferred_retriangulation_quadrilateral_min_frame_gap
            > config.deferred_retriangulation_quadrilateral_max_frame_gap
    {
        return Err(RigSfmError::InvalidDeferredQuadrilateralFrameGap {
            min: config.deferred_retriangulation_quadrilateral_min_frame_gap,
            max: config.deferred_retriangulation_quadrilateral_max_frame_gap,
        });
    }
    if let Some(requested) = config.deferred_registration_min_pnp_sensors {
        if !(1..=rig.sensors().len()).contains(&requested) {
            return Err(RigSfmError::InvalidDeferredMinPnpSensors {
                requested,
                sensor_count: rig.sensors().len(),
            });
        }
    }
    if config.deferred_registration_pnp_max_iterations == Some(0) {
        return Err(RigSfmError::InvalidDeferredPnpIterations);
    }
    if config
        .deferred_registration_min_pnp_inliers
        .is_some_and(|inliers| inliers < 6)
    {
        return Err(RigSfmError::InvalidDeferredMinPnpInliers(
            config
                .deferred_registration_min_pnp_inliers
                .expect("checked as present"),
        ));
    }
    if config.complete_tracks_after_registration {
        if config.track_completion_max_passes == 0 {
            return Err(RigSfmError::InvalidTrackCompletionPasses);
        }
        if !config
            .track_completion_max_reprojection_error_px
            .is_finite()
            || config.track_completion_max_reprojection_error_px <= 0.0
        {
            return Err(RigSfmError::InvalidTrackCompletionReprojectionError(
                config.track_completion_max_reprojection_error_px,
            ));
        }
    }
    if config.repair_isolated_pose_outliers {
        if config.isolated_pose_repair_max_passes == 0 {
            return Err(RigSfmError::InvalidIsolatedPoseRepairPasses);
        }
        if !config.isolated_pose_max_midpoint_error_m.is_finite()
            || config.isolated_pose_max_midpoint_error_m <= 0.0
        {
            return Err(RigSfmError::InvalidIsolatedPoseMidpointError(
                config.isolated_pose_max_midpoint_error_m,
            ));
        }
        if !config.isolated_pose_min_detour_ratio.is_finite()
            || config.isolated_pose_min_detour_ratio <= 1.0
        {
            return Err(RigSfmError::InvalidIsolatedPoseDetourRatio(
                config.isolated_pose_min_detour_ratio,
            ));
        }
    }
    if config.repair_paired_pose_jumps {
        if !config.paired_pose_jump_absolute_step_m.is_finite()
            || config.paired_pose_jump_absolute_step_m <= 0.0
        {
            return Err(RigSfmError::InvalidPairedPoseJumpStep(
                config.paired_pose_jump_absolute_step_m,
            ));
        }
        if !config.paired_pose_jump_min_step_ratio.is_finite()
            || config.paired_pose_jump_min_step_ratio <= 1.0
        {
            return Err(RigSfmError::InvalidPairedPoseJumpStepRatio(
                config.paired_pose_jump_min_step_ratio,
            ));
        }
        if config.paired_pose_jump_max_frame_span == 0 {
            return Err(RigSfmError::InvalidPairedPoseJumpFrameSpan);
        }
        if !config.paired_pose_jump_max_closure_ratio.is_finite()
            || config.paired_pose_jump_max_closure_ratio <= 0.0
            || config.paired_pose_jump_max_closure_ratio >= 1.0
        {
            return Err(RigSfmError::InvalidPairedPoseJumpClosureRatio(
                config.paired_pose_jump_max_closure_ratio,
            ));
        }
    }
    let mut assigned = vec![false; features.len()];
    for (frame_index, frame) in frames.iter().enumerate() {
        if frame.images.is_empty() {
            return Err(RigSfmError::EmptyFrame { frame: frame_index });
        }
        let mut sensors = HashSet::new();
        for image in &frame.images {
            if image.image_index >= features.len() {
                return Err(RigSfmError::ImageIndex {
                    image: image.image_index,
                    feature_count: features.len(),
                });
            }
            if image.sensor_index >= rig.sensors().len() {
                return Err(RigSfmError::SensorIndex {
                    sensor: image.sensor_index,
                    sensor_count: rig.sensors().len(),
                });
            }
            if assigned[image.image_index] {
                return Err(RigSfmError::DuplicateImage {
                    image: image.image_index,
                });
            }
            assigned[image.image_index] = true;
            if !sensors.insert(image.sensor_index) {
                return Err(RigSfmError::DuplicateSensor {
                    frame: frame_index,
                    sensor: image.sensor_index,
                });
            }
        }
    }
    if let Some(image) = assigned.iter().position(|assigned| !assigned) {
        return Err(RigSfmError::UnassignedImage { image });
    }
    for pair in pairwise {
        if pair.image_i >= features.len() || pair.image_j >= features.len() {
            return Err(RigSfmError::PairImageIndex {
                image_i: pair.image_i,
                image_j: pair.image_j,
                feature_count: features.len(),
            });
        }
        if pair.matches.iter().any(|(left, right)| {
            *left >= features[pair.image_i].len() || *right >= features[pair.image_j].len()
        }) {
            return Err(RigSfmError::PairKeypointIndex {
                image_i: pair.image_i,
                image_j: pair.image_j,
            });
        }
    }
    Ok(())
}

pub(super) fn image_assignment(frames: &[RigFrame], image_count: usize) -> Vec<(usize, usize)> {
    let mut assignment = vec![(usize::MAX, usize::MAX); image_count];
    for (frame_index, frame) in frames.iter().enumerate() {
        for image in &frame.images {
            assignment[image.image_index] = (frame_index, image.sensor_index);
        }
    }
    assignment
}

pub(super) fn image_track_index(
    image_count: usize,
    tracks: &[WorkingTrack],
) -> Vec<Vec<(usize, usize)>> {
    let mut by_image = vec![Vec::new(); image_count];
    for (track_index, track) in tracks.iter().enumerate() {
        for &(image, keypoint) in &track.observations {
            by_image[image].push((keypoint, track_index));
        }
    }
    by_image
}

pub(super) fn metric_frame_supports(
    frame_count: usize,
    tracks: &[WorkingTrack],
    image_assignment: &[(usize, usize)],
) -> Vec<usize> {
    let mut supports = vec![0usize; frame_count];
    for track in tracks {
        let mut frame_sensors = track
            .observations
            .iter()
            .map(|(image, _)| image_assignment[*image])
            .collect::<Vec<_>>();
        frame_sensors.sort_unstable();
        let mut offset = 0usize;
        while offset < frame_sensors.len() {
            let frame = frame_sensors[offset].0;
            let first_sensor = frame_sensors[offset].1;
            offset += 1;
            let mut multiple_sensors = false;
            while offset < frame_sensors.len() && frame_sensors[offset].0 == frame {
                multiple_sensors |= frame_sensors[offset].1 != first_sensor;
                offset += 1;
            }
            supports[frame] += usize::from(multiple_sensors);
        }
    }
    supports
}

pub(super) fn metric_seed_candidates(supports: &[usize]) -> Vec<usize> {
    let mut candidates = supports
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, supported)| *supported >= 6)
        .collect::<Vec<_>>();
    candidates
        .sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    candidates.into_iter().map(|(frame, _)| frame).collect()
}

pub(super) fn install_image_poses(
    rig: &GeneralizedCameraRig,
    frame: &RigFrame,
    frame_pose: &Pose,
    image_poses: &mut [Option<Pose>],
) {
    for image in &frame.images {
        let sensor = &rig.sensors()[image.sensor_index];
        image_poses[image.image_index] = Some(Pose {
            world_to_camera: sensor.sensor_from_rig.compose(&frame_pose.world_to_camera),
        });
    }
}

pub(super) fn build_verified_stereo_links(
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
) -> HashMap<(usize, usize), (usize, usize)> {
    let mut links = HashMap::new();
    for pair in pairwise {
        let (frame_i, sensor_i) = image_assignment[pair.image_i];
        let (frame_j, sensor_j) = image_assignment[pair.image_j];
        if frame_i != frame_j || sensor_i == sensor_j {
            continue;
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            links
                .entry((pair.image_i, keypoint_i))
                .or_insert((pair.image_j, keypoint_j));
            links
                .entry((pair.image_j, keypoint_j))
                .or_insert((pair.image_i, keypoint_i));
        }
    }
    links
}

pub(super) fn build_relevant_verified_stereo_links(
    base_pairwise: &[PairwiseMatches],
    deferred_pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
) -> HashMap<(usize, usize), (usize, usize)> {
    let referenced = deferred_pairwise
        .iter()
        .flat_map(|pair| {
            pair.matches.iter().flat_map(|&(keypoint_i, keypoint_j)| {
                [(pair.image_i, keypoint_i), (pair.image_j, keypoint_j)]
            })
        })
        .collect::<HashSet<_>>();
    let mut links = HashMap::new();
    for pair in base_pairwise {
        let (frame_i, sensor_i) = image_assignment[pair.image_i];
        let (frame_j, sensor_j) = image_assignment[pair.image_j];
        if frame_i != frame_j || sensor_i == sensor_j {
            continue;
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            if referenced.contains(&(pair.image_i, keypoint_i)) {
                links
                    .entry((pair.image_i, keypoint_i))
                    .or_insert((pair.image_j, keypoint_j));
            }
            if referenced.contains(&(pair.image_j, keypoint_j)) {
                links
                    .entry((pair.image_j, keypoint_j))
                    .or_insert((pair.image_i, keypoint_i));
            }
        }
    }
    links
}

pub(super) fn build_frame_pair_adjacency(
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    frame_count: usize,
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); frame_count];
    for (pair_index, pair) in pairwise.iter().enumerate() {
        let frame_i = image_assignment[pair.image_i].0;
        let frame_j = image_assignment[pair.image_j].0;
        if frame_i == frame_j {
            continue;
        }
        adjacency[frame_i].push(pair_index);
        adjacency[frame_j].push(pair_index);
    }
    adjacency
}
