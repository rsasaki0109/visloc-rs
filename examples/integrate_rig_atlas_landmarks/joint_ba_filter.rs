//! The joint rig BA observation filter
//! (`--joint-rig-ba-filter-observations`).

use super::*;

pub(super) fn classify_filter_observation(
    key: ObservationKey,
    position: &Point3<f64>,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<f64, FilterObservationReason> {
    if !position.coords.iter().all(|value| value.is_finite()) {
        return Err(FilterObservationReason::NonFinite);
    }
    let image = images
        .get(&key.global_image_id)
        .ok_or(FilterObservationReason::ProjectionFailure)?;
    let camera = cameras
        .get(&image.atlas.camera_id)
        .ok_or(FilterObservationReason::ProjectionFailure)?;
    let observation = observations
        .get(&key)
        .ok_or(FilterObservationReason::ProjectionFailure)?;
    if !observation.xy.coords.iter().all(|value| value.is_finite()) {
        return Err(FilterObservationReason::NonFinite);
    }
    let point_camera = pose_for_image(image, pose_overrides).transform_world_point(position);
    if !point_camera.coords.iter().all(|value| value.is_finite()) {
        return Err(FilterObservationReason::NonFinite);
    }
    if point_camera.z <= 0.0 {
        return Err(FilterObservationReason::BehindCamera);
    }
    let projected = project_allowing_nonpositive_depth(camera, &point_camera)
        .ok_or(FilterObservationReason::ProjectionFailure)?;
    let error = (projected - observation.xy).norm();
    if !error.is_finite() {
        return Err(FilterObservationReason::NonFinite);
    }
    if error > MAX_REPROJECTION_PX {
        return Err(FilterObservationReason::ReprojectionOverMax);
    }
    Ok(error)
}

/// Keep a raw BA point only when filtering left its complete original key set
/// intact and the point passes the same bounded geometric gates as DLT.  The
/// caller falls back to the historical DLT path for every error here; this
/// helper never changes observation ownership or filtering decisions.
pub(super) fn retain_raw_ba_point_if_valid(
    candidate: &LandmarkOutput,
    expected_observations: &[ObservationKey],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<LandmarkOutput, RawPointFallbackReason> {
    if candidate.observations != expected_observations {
        return Err(RawPointFallbackReason::ObservationKeysChanged);
    }
    if candidate.observations.len() < MIN_TRACK_OBSERVATIONS {
        return Err(RawPointFallbackReason::TooFewObservations);
    }
    if !candidate
        .position
        .coords
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(RawPointFallbackReason::NonFinite);
    }
    let sample = bounded_sample(&candidate.observations, MAX_DLT_OBSERVATIONS);
    match has_observable_parallax(&sample, observations, images, cameras, pose_overrides) {
        Ok(true) => {}
        Ok(false) => return Err(RawPointFallbackReason::NoObservableParallax),
        Err(_) => return Err(RawPointFallbackReason::ParallaxCheckFailed),
    }

    let mut errors = Vec::with_capacity(candidate.observations.len());
    for key in &candidate.observations {
        let Some(image) = images.get(&key.global_image_id) else {
            return Err(RawPointFallbackReason::ProjectionFailure);
        };
        let Some(camera) = cameras.get(&image.atlas.camera_id) else {
            return Err(RawPointFallbackReason::ProjectionFailure);
        };
        let Some(observation) = observations.get(key) else {
            return Err(RawPointFallbackReason::ProjectionFailure);
        };
        if !observation.xy.coords.iter().all(|value| value.is_finite()) {
            return Err(RawPointFallbackReason::NonFinite);
        }
        let point_camera =
            pose_for_image(image, pose_overrides).transform_world_point(&candidate.position);
        if !point_camera.coords.iter().all(|value| value.is_finite()) {
            return Err(RawPointFallbackReason::NonFinite);
        }
        if point_camera.z <= 0.0 {
            return Err(RawPointFallbackReason::NonPositiveDepth);
        }
        let Some(projected) = camera.project(&point_camera) else {
            return Err(RawPointFallbackReason::ProjectionFailure);
        };
        let error = (projected - observation.xy).norm();
        if !error.is_finite() {
            return Err(RawPointFallbackReason::NonFinite);
        }
        errors.push(error);
    }
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    let squared_sum = errors.iter().map(|error| error * error).sum::<f64>();
    let rms_error = (squared_sum / errors.len() as f64).sqrt();
    let max_error = errors.iter().copied().fold(0.0, f64::max);
    if !mean_error.is_finite() || !rms_error.is_finite() || !max_error.is_finite() {
        return Err(RawPointFallbackReason::NonFinite);
    }
    if mean_error > MAX_MEAN_REPROJECTION_PX {
        return Err(RawPointFallbackReason::MeanOverMax);
    }
    if max_error > MAX_REPROJECTION_PX {
        return Err(RawPointFallbackReason::MaxOverMax);
    }
    let mut retained = candidate.clone();
    retained.errors = errors;
    retained.mean_error = mean_error;
    retained.rms_error = rms_error;
    retained.max_error = max_error;
    Ok(retained)
}

fn cost_for_landmark_keys(
    position: &Point3<f64>,
    keys: &[ObservationKey],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<f64, String> {
    let mut cost = 0.0;
    for key in keys {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "landmark references unknown image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "landmark references unknown camera".to_owned())?;
        let observation = observations
            .get(key)
            .ok_or_else(|| "landmark references unknown observation".to_owned())?;
        let point_camera = pose_for_image(image, pose_overrides).transform_world_point(position);
        let projected = project_allowing_nonpositive_depth(camera, &point_camera)
            .ok_or_else(|| "landmark cost became non-finite".to_owned())?;
        let error = (projected - observation.xy).norm();
        if !error.is_finite() {
            return Err("landmark cost became non-finite".to_owned());
        }
        cost += error * error;
    }
    if !cost.is_finite() {
        return Err("landmark cost became non-finite".to_owned());
    }
    Ok(cost)
}

pub(super) fn cost_non_increasing(before: f64, after: f64) -> bool {
    before.is_finite() && after.is_finite() && after <= before + 1.0e-8 * before.abs().max(1.0)
}

pub(super) fn validate_filter_costs(
    full_pre_ba_cost: f64,
    full_post_ba_cost: f64,
    retained_pre_ba_cost: f64,
    retained_post_filter_cost: f64,
) -> Result<(), String> {
    if !cost_non_increasing(full_pre_ba_cost, full_post_ba_cost) {
        return Err(format!(
            "full selected cost increased before filtering from {full_pre_ba_cost:.12} to {full_post_ba_cost:.12}"
        ));
    }
    if !cost_non_increasing(retained_pre_ba_cost, retained_post_filter_cost) {
        return Err(format!(
            "retained cost increased after filtering from {retained_pre_ba_cost:.12} to {retained_post_filter_cost:.12}"
        ));
    }
    Ok(())
}

pub(super) fn joint_ba_window_counts(
    landmarks: &[LandmarkOutput],
    images: &BTreeMap<u64, GlobalImage>,
    active_frames: &BTreeSet<u64>,
) -> Result<(usize, usize, usize, usize), String> {
    let selected = selected_landmarks_for_frames(landmarks, active_frames, images)?;
    joint_ba_window_counts_with_selected(landmarks, images, active_frames, &selected)
}

pub(super) fn joint_ba_window_counts_with_selected(
    landmarks: &[LandmarkOutput],
    images: &BTreeMap<u64, GlobalImage>,
    active_frames: &BTreeSet<u64>,
    selected: &[usize],
) -> Result<(usize, usize, usize, usize), String> {
    let selected_observations = selected
        .iter()
        .map(|index| landmarks[*index].observations.len())
        .sum::<usize>();
    let mut referenced_frames = BTreeSet::new();
    for index in selected {
        for key in &landmarks[*index].observations {
            let image = images
                .get(&key.global_image_id)
                .ok_or_else(|| "landmark references unknown image".to_owned())?;
            referenced_frames.insert(image.atlas.frame_id);
        }
    }
    let free_frames = active_frames.intersection(&referenced_frames).count();
    Ok((
        selected.len(),
        selected_observations,
        referenced_frames.len(),
        free_frames,
    ))
}

#[cfg(test)]
pub(super) fn build_joint_rig_ba_filter_candidate(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    baseline_connectivity: &SupportConnectivity,
) -> Result<JointRigBaFilteringCandidate, JointRigBaSkip> {
    build_joint_rig_ba_filter_candidate_with_policy(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        active_frames,
        baseline_connectivity,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn build_joint_rig_ba_filter_candidate_with_policy(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    baseline_connectivity: &SupportConnectivity,
    preserve_optimized_points: bool,
) -> Result<JointRigBaFilteringCandidate, JointRigBaSkip> {
    let selected = selected_landmarks_for_frames(landmarks, active_frames, images)
        .map_err(JointRigBaSkip::Invalid)?;
    build_joint_rig_ba_filter_candidate_with_selected(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        active_frames,
        &selected,
        baseline_connectivity,
        preserve_optimized_points,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_joint_rig_ba_filter_candidate_with_selected(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    selected: &[usize],
    baseline_connectivity: &SupportConnectivity,
    preserve_optimized_points: bool,
) -> Result<JointRigBaFilteringCandidate, JointRigBaSkip> {
    let raw = build_joint_rig_ba_window_raw_with_selected(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        active_frames,
        selected,
    )?;
    let mut timer = AtlasInnerTimer::new(active_frames, "filter_retriangulate_costs");
    if raw.landmark_updates.len() != selected.len() {
        return Err(JointRigBaSkip::Invalid(
            "raw BA candidate omitted a selected landmark".to_owned(),
        ));
    }

    let mut full_pre_ba_cost = 0.0;
    let mut full_post_ba_cost = 0.0;
    for index in selected {
        let old = landmarks.get(*index).ok_or_else(|| {
            JointRigBaSkip::Invalid("selected landmark index is invalid".to_owned())
        })?;
        let candidate = raw.landmark_updates.get(index).ok_or_else(|| {
            JointRigBaSkip::Invalid(format!("raw BA omitted selected landmark {index}"))
        })?;
        if candidate.track_id != old.track_id || candidate.observations != old.observations {
            return Err(JointRigBaSkip::Invalid(format!(
                "raw BA changed identity or observations for landmark {index}"
            )));
        }
        full_pre_ba_cost += cost_for_landmark_keys(
            &old.position,
            &old.observations,
            &store.observations,
            images,
            cameras,
            &BTreeMap::new(),
        )
        .map_err(JointRigBaSkip::Invalid)?;
        full_post_ba_cost += cost_for_landmark_keys(
            &candidate.position,
            &candidate.observations,
            &store.observations,
            images,
            cameras,
            &raw.pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
    }
    let mut landmark_updates = BTreeMap::new();
    let mut removed_observations = BTreeMap::new();
    let mut removed_track_ids = BTreeSet::new();
    let mut retained_pre_ba_cost = 0.0;
    let mut retained_post_filter_cost = 0.0;
    let mut retriangulated_tracks = 0;
    let mut raw_preserved_tracks = 0;
    let mut dlt_attempted_tracks = 0;
    let mut raw_fallback_reason_counts = BTreeMap::new();
    for index in selected {
        let old = landmarks.get(*index).ok_or_else(|| {
            JointRigBaSkip::Invalid("selected landmark index is invalid".to_owned())
        })?;
        let candidate = raw.landmark_updates.get(index).ok_or_else(|| {
            JointRigBaSkip::Invalid(format!("raw BA omitted selected landmark {index}"))
        })?;
        let raw_gate_result = preserve_optimized_points.then(|| {
            retain_raw_ba_point_if_valid(
                candidate,
                &old.observations,
                &store.observations,
                images,
                cameras,
                &raw.pose_overrides,
            )
        });
        let mut retained = Vec::with_capacity(candidate.observations.len());
        for key in &candidate.observations {
            match classify_filter_observation(
                *key,
                &candidate.position,
                &store.observations,
                images,
                cameras,
                &raw.pose_overrides,
            ) {
                Ok(_) => retained.push(*key),
                Err(reason) => {
                    removed_observations.insert(*key, reason);
                }
            }
        }
        if retained.len() < MIN_TRACK_OBSERVATIONS {
            if preserve_optimized_points {
                if let Some(Err(reason)) = raw_gate_result {
                    *raw_fallback_reason_counts.entry(reason).or_default() += 1;
                }
            }
            removed_track_ids.insert(old.track_id);
            for key in &old.observations {
                removed_observations
                    .entry(*key)
                    .or_insert(FilterObservationReason::TrackBelowMinimum);
            }
            continue;
        }

        if preserve_optimized_points {
            let raw_result = raw_gate_result
                .expect("raw gate result is present when optimized points are enabled");
            match (retained == old.observations, raw_result) {
                (true, Ok(raw_landmark)) => {
                    retained_pre_ba_cost += cost_for_landmark_keys(
                        &old.position,
                        &retained,
                        &store.observations,
                        images,
                        cameras,
                        &BTreeMap::new(),
                    )
                    .map_err(JointRigBaSkip::Invalid)?;
                    retained_post_filter_cost += cost_for_landmark_keys(
                        &raw_landmark.position,
                        &retained,
                        &store.observations,
                        images,
                        cameras,
                        &raw.pose_overrides,
                    )
                    .map_err(JointRigBaSkip::Invalid)?;
                    raw_preserved_tracks += 1;
                    landmark_updates.insert(*index, raw_landmark);
                    continue;
                }
                (false, Ok(_)) => {
                    *raw_fallback_reason_counts
                        .entry(RawPointFallbackReason::ObservationKeysChanged)
                        .or_default() += 1;
                }
                (_, Err(reason)) => {
                    *raw_fallback_reason_counts.entry(reason).or_default() += 1;
                }
            }
        }

        dlt_attempted_tracks += 1;
        let refined_track = GlobalTrack {
            observations: retained.clone(),
        };
        let refined = match triangulate_track_with_pose_overrides(
            old.track_id,
            &refined_track,
            &store.observations,
            images,
            cameras,
            &raw.pose_overrides,
        ) {
            Ok(landmark) => landmark,
            Err(_) => {
                removed_track_ids.insert(old.track_id);
                for key in &old.observations {
                    removed_observations
                        .entry(*key)
                        .or_insert(FilterObservationReason::RetriangulationFailed);
                }
                continue;
            }
        };
        retained_pre_ba_cost += cost_for_landmark_keys(
            &old.position,
            &retained,
            &store.observations,
            images,
            cameras,
            &BTreeMap::new(),
        )
        .map_err(JointRigBaSkip::Invalid)?;
        retained_post_filter_cost += cost_for_landmark_keys(
            &refined.position,
            &retained,
            &store.observations,
            images,
            cameras,
            &raw.pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        retriangulated_tracks += 1;
        landmark_updates.insert(*index, refined);
    }
    validate_filter_costs(
        full_pre_ba_cost,
        full_post_ba_cost,
        retained_pre_ba_cost,
        retained_post_filter_cost,
    )
    .map_err(JointRigBaSkip::Invalid)?;

    timer.next("candidate_connectivity");
    let candidate_connectivity = support_connectivity_for_landmarks(
        landmarks,
        &landmark_updates,
        &removed_track_ids,
        images,
    )
    .map_err(JointRigBaSkip::Invalid)?;
    if !connectivity_not_split(baseline_connectivity, &candidate_connectivity) {
        return Err(JointRigBaSkip::Invalid(
            "filter candidate loses supported image/frame support or splits connectivity"
                .to_owned(),
        ));
    }
    timer.next("candidate_pose_validation");
    validate_candidate_pose_overrides(manifest, images, &raw.pose_overrides)
        .map_err(JointRigBaSkip::Invalid)?;
    Ok(JointRigBaFilteringCandidate {
        pose_overrides: raw.pose_overrides,
        landmark_updates,
        removed_observations,
        removed_track_ids,
        selected_landmarks: raw.selected_landmarks,
        selected_observations: raw.selected_observations,
        referenced_frames: raw.referenced_frames,
        free_frames: raw.free_frames,
        full_pre_ba_cost,
        full_post_ba_cost,
        retained_pre_ba_cost,
        retained_post_filter_cost,
        retriangulated_tracks,
        raw_preserved_tracks,
        dlt_attempted_tracks,
        raw_fallback_reason_counts,
        iterations: raw.iterations,
        converged: raw.converged,
    })
}

fn validate_candidate_pose_overrides(
    manifest: &RigManifest,
    images: &BTreeMap<u64, GlobalImage>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<(), String> {
    let mut frame_poses = BTreeMap::<u64, Pose>::new();
    for (image_id, pose) in pose_overrides {
        let image = images
            .get(image_id)
            .ok_or_else(|| "BA pose override references unknown image".to_owned())?;
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| "BA pose override references unknown sensor".to_owned())?;
        if !diagnostic_pose_is_finite(pose) {
            return Err("BA pose override is non-finite".to_owned());
        }
        let rig_pose = Pose {
            world_to_camera: sensor
                .sensor_from_rig
                .inverse()
                .compose(&pose.world_to_camera),
        };
        if let Some(previous) = frame_poses.insert(image.atlas.frame_id, rig_pose.clone()) {
            let center_error =
                (previous.camera_center_world() - rig_pose.camera_center_world()).norm();
            let rotation_error = previous
                .world_to_camera
                .rotation
                .rotation_to(&rig_pose.world_to_camera.rotation)
                .angle()
                .to_degrees();
            if !center_error.is_finite()
                || !rotation_error.is_finite()
                || center_error > MAX_RIG_CENTER_DISAGREEMENT_M
                || rotation_error > MAX_RIG_ROTATION_DISAGREEMENT_DEG
            {
                return Err(format!(
                    "BA pose overrides disagree for frame {}",
                    image.atlas.frame_id
                ));
            }
        }
    }
    Ok(())
}

fn filter_reason_counts(
    removed_observations: &BTreeMap<ObservationKey, FilterObservationReason>,
) -> BTreeMap<FilterObservationReason, usize> {
    let mut counts = BTreeMap::new();
    for reason in removed_observations.values() {
        *counts.entry(*reason).or_default() += 1;
    }
    counts
}

fn format_filter_reason_counts(counts: &BTreeMap<FilterObservationReason, usize>) -> String {
    counts
        .iter()
        .map(|(reason, count)| format!("{reason}={count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_raw_fallback_reason_counts(counts: &BTreeMap<RawPointFallbackReason, usize>) -> String {
    if counts.is_empty() {
        return "none".to_owned();
    }
    counts
        .iter()
        .map(|(reason, count)| format!("{reason}={count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn validate_filter_candidate_for_apply(
    candidate: &JointRigBaFilteringCandidate,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    landmarks: &[LandmarkOutput],
) -> Result<(), String> {
    for (image_id, pose) in &candidate.pose_overrides {
        if !images.contains_key(image_id) {
            return Err(format!(
                "filter candidate pose override references unknown image {image_id}"
            ));
        }
        if !diagnostic_pose_is_finite(pose) {
            return Err(format!(
                "filter candidate pose override for image {image_id} is non-finite"
            ));
        }
    }

    let mut replacement_tracks = BTreeSet::new();
    let mut retained_keys = BTreeSet::new();
    let mut expected_removed_keys = BTreeSet::new();
    for (index, replacement) in &candidate.landmark_updates {
        let old = landmarks
            .get(*index)
            .ok_or_else(|| format!("filter candidate landmark index {index} is invalid"))?;
        if old.track_id != replacement.track_id {
            return Err(format!(
                "filter candidate changed track identity at landmark index {index}"
            ));
        }
        if replacement.observations.len() < MIN_TRACK_OBSERVATIONS {
            return Err(format!(
                "filter candidate replacement track {} has too few observations",
                replacement.track_id
            ));
        }
        if !replacement
            .position
            .coords
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(format!(
                "filter candidate replacement track {} has a non-finite position",
                replacement.track_id
            ));
        }
        if !replacement_tracks.insert(replacement.track_id) {
            return Err(format!(
                "filter candidate updates track {} more than once",
                replacement.track_id
            ));
        }
        let source_track = store.tracks.get(replacement.track_id).ok_or_else(|| {
            format!(
                "filter candidate replacement references unknown track {}",
                replacement.track_id
            )
        })?;
        let old_keys = old.observations.iter().copied().collect::<BTreeSet<_>>();
        let source_keys = source_track
            .observations
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if source_keys != old_keys {
            return Err(format!(
                "filter candidate source track {} disagrees with landmark observations",
                replacement.track_id
            ));
        }
        let mut replacement_keys = BTreeSet::new();
        for key in &replacement.observations {
            if !replacement_keys.insert(*key) {
                return Err(format!(
                    "filter candidate replacement track {} has duplicate observation {:?}",
                    replacement.track_id, key
                ));
            }
            if !old_keys.contains(key) {
                return Err(format!(
                    "filter candidate replacement track {} adds observation {:?}",
                    replacement.track_id, key
                ));
            }
            if !images.contains_key(&key.global_image_id) {
                return Err(format!(
                    "filter candidate replacement references unknown image {}",
                    key.global_image_id
                ));
            }
            let state = store.observations.get(key).ok_or_else(|| {
                format!("filter candidate replacement references unknown observation {key:?}")
            })?;
            if state.owner_track != replacement.track_id {
                return Err(format!(
                    "filter candidate replacement observation {:?} belongs to track {}",
                    key, state.owner_track
                ));
            }
            retained_keys.insert(*key);
        }
        expected_removed_keys.extend(old_keys.difference(&replacement_keys).copied());
        if candidate.removed_track_ids.contains(&replacement.track_id) {
            return Err(format!(
                "filter candidate both replaces and removes track {}",
                replacement.track_id
            ));
        }
    }

    for track_id in &candidate.removed_track_ids {
        let track = store
            .tracks
            .get(*track_id)
            .ok_or_else(|| format!("filter candidate references unknown track {track_id}"))?;
        if replacement_tracks.contains(track_id) {
            return Err(format!(
                "filter candidate both replaces and removes track {track_id}"
            ));
        }
        for key in &track.observations {
            if !images.contains_key(&key.global_image_id) {
                return Err(format!(
                    "source track {track_id} references unknown image {}",
                    key.global_image_id
                ));
            }
            let state = store.observations.get(key).ok_or_else(|| {
                format!("source track {track_id} references unknown observation {key:?}")
            })?;
            if state.owner_track != *track_id {
                return Err(format!(
                    "source track {track_id} observation {:?} has owner {}",
                    key, state.owner_track
                ));
            }
            expected_removed_keys.insert(*key);
        }
    }

    let actual_removed_keys = candidate
        .removed_observations
        .keys()
        .copied()
        .collect::<BTreeSet<_>>();
    if actual_removed_keys != expected_removed_keys {
        return Err(format!(
            "filter candidate removal ledger disagrees: expected {} keys, got {}",
            expected_removed_keys.len(),
            actual_removed_keys.len()
        ));
    }
    for key in candidate.removed_observations.keys() {
        if !images.contains_key(&key.global_image_id) {
            return Err(format!(
                "filter candidate removes observation {:?} from an unknown image",
                key
            ));
        }
        let state = store
            .observations
            .get(key)
            .ok_or_else(|| format!("filter candidate removes unknown observation {key:?}"))?;
        store.tracks.get(state.owner_track).ok_or_else(|| {
            format!(
                "filter candidate observation {:?} has unknown owner track {}",
                key, state.owner_track
            )
        })?;
        if !candidate.removed_track_ids.contains(&state.owner_track)
            && !replacement_tracks.contains(&state.owner_track)
        {
            return Err(format!(
                "filter candidate removes observation {:?} from an untouched track {}",
                key, state.owner_track
            ));
        }
        if retained_keys.contains(key) {
            return Err(format!(
                "filter candidate both removes and retains observation {key:?}"
            ));
        }
    }

    let mut required_landmark_tracks = replacement_tracks.clone();
    required_landmark_tracks.extend(candidate.removed_track_ids.iter().copied());
    let mut found_landmark_tracks = BTreeSet::new();
    for landmark in landmarks {
        if required_landmark_tracks.contains(&landmark.track_id) {
            found_landmark_tracks.insert(landmark.track_id);
        }
    }
    for track_id in &candidate.removed_track_ids {
        if !found_landmark_tracks.contains(track_id) {
            return Err(format!(
                "filter candidate removes track {track_id} without a landmark"
            ));
        }
    }

    Ok(())
}

pub(super) fn apply_joint_rig_ba_filter_candidate(
    candidate: &JointRigBaFilteringCandidate,
    store: &mut TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    landmarks: &mut Vec<LandmarkOutput>,
) -> Result<(), String> {
    // All fallible validation happens before the first mutation.  The actual
    // apply phase below only indexes entries proven to exist here, so a
    // rejected candidate cannot leave a partially updated model behind.
    validate_filter_candidate_for_apply(candidate, store, images, landmarks)?;
    apply_pose_overrides(images, &candidate.pose_overrides)?;
    for (index, replacement) in &candidate.landmark_updates {
        let landmark = landmarks
            .get_mut(*index)
            .expect("filter candidate landmark index was prevalidated");
        *landmark = replacement.clone();
    }
    for key in candidate.removed_observations.keys() {
        store.observations.remove(key);
    }
    for track_id in &candidate.removed_track_ids {
        let track = store
            .tracks
            .get_mut(*track_id)
            .expect("filter candidate removed track was prevalidated");
        track.observations.clear();
    }
    for replacement in candidate.landmark_updates.values() {
        let track_id = replacement.track_id;
        store
            .tracks
            .get_mut(track_id)
            .expect("replacement track was prevalidated")
            .observations = replacement.observations.clone();
    }
    landmarks.retain(|landmark| !candidate.removed_track_ids.contains(&landmark.track_id));
    Ok(())
}

pub(super) fn validate_filtered_model(
    baseline_connectivity: &SupportConnectivity,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
) -> Result<(), String> {
    let current_connectivity =
        support_connectivity_for_landmarks(landmarks, &BTreeMap::new(), &BTreeSet::new(), images)?;
    if !connectivity_not_split(baseline_connectivity, &current_connectivity) {
        return Err("final filtered model loses support or splits connectivity".to_owned());
    }
    for landmark in landmarks {
        if landmark.observations.len() < MIN_TRACK_OBSERVATIONS {
            return Err(format!(
                "final landmark {} has too few observations",
                landmark.track_id
            ));
        }
        let metrics = evaluate_landmark_metrics(
            landmark,
            &store.observations,
            images,
            cameras,
            &BTreeMap::new(),
        )?;
        if metrics.mean_error > MAX_MEAN_REPROJECTION_PX || metrics.max_error > MAX_REPROJECTION_PX
        {
            return Err(format!(
                "final landmark {} failed reprojection gate",
                landmark.track_id
            ));
        }
        let track = store
            .tracks
            .get(landmark.track_id)
            .ok_or_else(|| format!("final landmark {} has no source track", landmark.track_id))?;
        if track.observations != landmark.observations {
            return Err(format!(
                "final landmark {} and source track observations disagree",
                landmark.track_id
            ));
        }
        for key in &landmark.observations {
            let observation = store
                .observations
                .get(key)
                .ok_or_else(|| "final landmark references removed observation".to_owned())?;
            if observation.owner_track != landmark.track_id {
                return Err("final landmark observation owner disagrees".to_owned());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn run_joint_rig_ba_filtering(
    manifest: &RigManifest,
    store: &mut TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut Vec<LandmarkOutput>,
) -> Result<JointRigBaFilteringSummary, String> {
    run_joint_rig_ba_filtering_with_policy(manifest, store, images, cameras, landmarks, false)
}

fn print_joint_rig_ba_filter_log(pass: Option<usize>, line: std::fmt::Arguments<'_>) {
    if let Some(pass) = pass {
        println!("pass={pass} cost_scope=overlapping_window_events {line}");
    } else {
        println!("{line}");
    }
}

pub(super) fn print_joint_rig_ba_filter_summary(
    summary: &JointRigBaFilteringSummary,
    preserve_optimized_points: bool,
    pass: Option<usize>,
) {
    if preserve_optimized_points {
        print_joint_rig_ba_filter_log(pass, format_args!(
            "joint_rig_ba_filter_observations windows_considered={} windows_accepted={} windows_skipped={} selected_landmarks={} selected_observations={} max_referenced_frames={} max_free_frames={} max_iterations={} converged_windows={} removed_observations={} removed_tracks={} retriangulated_tracks={} raw_preserved_tracks={} dlt_attempted_tracks={} raw_fallback_reasons={} raw_counts_are_window_events=true full_pre_ba_cost={:.9} full_post_ba_cost={:.9} retained_pre_ba_cost={:.9} retained_post_filter_cost={:.9}",
            summary.windows_considered,
            summary.windows_accepted,
            summary.windows_skipped,
            summary.selected_landmarks,
            summary.selected_observations,
            summary.max_referenced_frames,
            summary.max_free_frames,
            summary.max_iterations,
            summary.converged_windows,
            summary.removed_observations,
            summary.removed_tracks,
            summary.retriangulated_tracks,
            summary.raw_preserved_tracks,
            summary.dlt_attempted_tracks,
            format_raw_fallback_reason_counts(&summary.raw_fallback_reason_counts),
            summary.full_pre_ba_cost,
            summary.full_post_ba_cost,
            summary.retained_pre_ba_cost,
            summary.retained_post_filter_cost,
        ));
    } else {
        print_joint_rig_ba_filter_log(pass, format_args!(
            "joint_rig_ba_filter_observations windows_considered={} windows_accepted={} windows_skipped={} selected_landmarks={} selected_observations={} max_referenced_frames={} max_free_frames={} max_iterations={} converged_windows={} removed_observations={} removed_tracks={} retriangulated_tracks={} full_pre_ba_cost={:.9} full_post_ba_cost={:.9} retained_pre_ba_cost={:.9} retained_post_filter_cost={:.9}",
            summary.windows_considered,
            summary.windows_accepted,
            summary.windows_skipped,
            summary.selected_landmarks,
            summary.selected_observations,
            summary.max_referenced_frames,
            summary.max_free_frames,
            summary.max_iterations,
            summary.converged_windows,
            summary.removed_observations,
            summary.removed_tracks,
            summary.retriangulated_tracks,
            summary.full_pre_ba_cost,
            summary.full_post_ba_cost,
            summary.retained_pre_ba_cost,
            summary.retained_post_filter_cost,
        ));
    }
}

pub(super) fn print_joint_rig_ba_filter_pass_support(
    pass: usize,
    images: &BTreeMap<u64, GlobalImage>,
    landmarks: &[LandmarkOutput],
) -> Result<(), String> {
    let support =
        support_connectivity_for_landmarks(landmarks, &BTreeMap::new(), &BTreeSet::new(), images)?;
    println!(
        "pass={pass} cost_scope=overlapping_window_events support_images={} support_frames={} components={}",
        support.supported_images.len(),
        support.supported_frames.len(),
        support.component_count
    );
    Ok(())
}

pub(super) fn run_joint_rig_ba_filtering_with_policy(
    manifest: &RigManifest,
    store: &mut TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut Vec<LandmarkOutput>,
    preserve_optimized_points: bool,
) -> Result<JointRigBaFilteringSummary, String> {
    run_joint_rig_ba_filtering_with_policy_and_pass(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        preserve_optimized_points,
        None,
    )
}

pub(super) fn run_joint_rig_ba_filtering_with_policy_and_pass(
    manifest: &RigManifest,
    store: &mut TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut Vec<LandmarkOutput>,
    preserve_optimized_points: bool,
    pass: Option<usize>,
) -> Result<JointRigBaFilteringSummary, String> {
    let trace_timing = std::env::var_os("VISLOC_ATLAS_TRACE_WINDOW_TIMING").is_some();
    let baseline_started = trace_timing.then(std::time::Instant::now);
    let baseline_connectivity =
        support_connectivity_for_landmarks(landmarks, &BTreeMap::new(), &BTreeSet::new(), images)?;
    if let Some(started) = baseline_started {
        eprintln!(
            "atlas-pass-timing: pass={} phase=baseline_connectivity seconds={:.9}",
            pass.unwrap_or(0),
            started.elapsed().as_secs_f64()
        );
    }
    let mut frame_ids = images
        .values()
        .map(|image| image.atlas.frame_id)
        .collect::<Vec<_>>();
    frame_ids.sort_unstable();
    frame_ids.dedup();
    let mut summary = JointRigBaFilteringSummary::default();
    for start in (0..frame_ids.len()).step_by(JOINT_BA_WINDOW_STRIDE) {
        let end = (start + JOINT_BA_WINDOW_LENGTH).min(frame_ids.len());
        let active_frames = frame_ids[start..end]
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        summary.windows_considered += 1;
        let selection_started = trace_timing.then(std::time::Instant::now);
        let selected_result = selected_landmarks_for_frames(landmarks, &active_frames, images);
        if let Some(started) = selection_started {
            eprintln!("atlas-window-timing: pass={} start_frame={} phase=selection seconds={:.9} scanned_landmarks={}",
                pass.unwrap_or(0), frame_ids[start], started.elapsed().as_secs_f64(), landmarks.len());
        }
        let counts_started = trace_timing.then(std::time::Instant::now);
        let window_counts = match selected_result.as_ref() {
            Ok(selected) => {
                joint_ba_window_counts_with_selected(landmarks, images, &active_frames, selected)
            }
            Err(error) => Err(error.clone()),
        };
        if let Ok((selected_count, observation_count, referenced_count, free_count)) =
            window_counts.as_ref()
        {
            summary.selected_landmarks += *selected_count;
            summary.selected_observations += *observation_count;
            summary.max_referenced_frames = summary.max_referenced_frames.max(*referenced_count);
            summary.max_free_frames = summary.max_free_frames.max(*free_count);
        }
        if let Some(started) = counts_started {
            eprintln!(
                "atlas-window-timing: pass={} start_frame={} phase=counts seconds={:.9}",
                pass.unwrap_or(0),
                frame_ids[start],
                started.elapsed().as_secs_f64()
            );
        }
        let candidate_started = trace_timing.then(std::time::Instant::now);
        let candidate_result = match selected_result {
            Ok(selected) => build_joint_rig_ba_filter_candidate_with_selected(
                manifest,
                store,
                images,
                cameras,
                landmarks,
                &active_frames,
                &selected,
                &baseline_connectivity,
                preserve_optimized_points,
            ),
            Err(reason) => Err(JointRigBaSkip::Invalid(reason)),
        };
        if let Some(started) = candidate_started {
            eprintln!("atlas-window-timing: pass={} start_frame={} phase=build_solve_validate seconds={:.9} accepted={}",
                pass.unwrap_or(0), frame_ids[start], started.elapsed().as_secs_f64(), candidate_result.is_ok());
        }
        match candidate_result {
            Ok(candidate) => {
                let reason_counts = filter_reason_counts(&candidate.removed_observations);
                let apply_started = trace_timing.then(std::time::Instant::now);
                apply_joint_rig_ba_filter_candidate(&candidate, store, images, landmarks)?;
                if let Some(started) = apply_started {
                    eprintln!(
                        "atlas-window-timing: pass={} start_frame={} phase=apply seconds={:.9}",
                        pass.unwrap_or(0),
                        frame_ids[start],
                        started.elapsed().as_secs_f64()
                    );
                }
                summary.windows_accepted += 1;
                summary.max_iterations = summary.max_iterations.max(candidate.iterations);
                if candidate.converged {
                    summary.converged_windows += 1;
                }
                summary.removed_observations += candidate.removed_observations.len();
                summary.removed_tracks += candidate.removed_track_ids.len();
                summary.retriangulated_tracks += candidate.retriangulated_tracks;
                summary.raw_preserved_tracks += candidate.raw_preserved_tracks;
                summary.dlt_attempted_tracks += candidate.dlt_attempted_tracks;
                summary.full_pre_ba_cost += candidate.full_pre_ba_cost;
                summary.full_post_ba_cost += candidate.full_post_ba_cost;
                summary.retained_pre_ba_cost += candidate.retained_pre_ba_cost;
                summary.retained_post_filter_cost += candidate.retained_post_filter_cost;
                for (reason, count) in &reason_counts {
                    *summary.removed_reason_counts.entry(*reason).or_default() += *count;
                }
                for (reason, count) in &candidate.raw_fallback_reason_counts {
                    *summary
                        .raw_fallback_reason_counts
                        .entry(*reason)
                        .or_default() += *count;
                }
                if preserve_optimized_points {
                    print_joint_rig_ba_filter_log(pass, format_args!(
                        "joint_rig_ba_filter_window start_frame={} end_frame={} active_frames={} status=accepted selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} removed_observations={} removed_tracks={} retriangulated_tracks={} raw_preserved_tracks={} dlt_attempted_tracks={} raw_fallback_reasons={} raw_counts_are_window_events=true reasons={} full_pre_ba_cost={:.9} full_post_ba_cost={:.9} retained_pre_ba_cost={:.9} retained_post_filter_cost={:.9} iterations={} converged={}",
                        active_frames.first().copied().unwrap_or_default(),
                        active_frames.last().copied().unwrap_or_default(),
                        active_frames.len(),
                        candidate.selected_landmarks,
                        candidate.selected_observations,
                        candidate.referenced_frames,
                        candidate.free_frames,
                        candidate.removed_observations.len(),
                        candidate.removed_track_ids.len(),
                        candidate.retriangulated_tracks,
                        candidate.raw_preserved_tracks,
                        candidate.dlt_attempted_tracks,
                        format_raw_fallback_reason_counts(&candidate.raw_fallback_reason_counts),
                        format_filter_reason_counts(&reason_counts),
                        candidate.full_pre_ba_cost,
                        candidate.full_post_ba_cost,
                        candidate.retained_pre_ba_cost,
                        candidate.retained_post_filter_cost,
                        candidate.iterations,
                        candidate.converged,
                    ));
                } else {
                    print_joint_rig_ba_filter_log(pass, format_args!(
                        "joint_rig_ba_filter_window start_frame={} end_frame={} active_frames={} status=accepted selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} removed_observations={} removed_tracks={} retriangulated_tracks={} reasons={} full_pre_ba_cost={:.9} full_post_ba_cost={:.9} retained_pre_ba_cost={:.9} retained_post_filter_cost={:.9} iterations={} converged={}",
                        active_frames.first().copied().unwrap_or_default(),
                        active_frames.last().copied().unwrap_or_default(),
                        active_frames.len(),
                        candidate.selected_landmarks,
                        candidate.selected_observations,
                        candidate.referenced_frames,
                        candidate.free_frames,
                        candidate.removed_observations.len(),
                        candidate.removed_track_ids.len(),
                        candidate.retriangulated_tracks,
                        format_filter_reason_counts(&reason_counts),
                        candidate.full_pre_ba_cost,
                        candidate.full_post_ba_cost,
                        candidate.retained_pre_ba_cost,
                        candidate.retained_post_filter_cost,
                        candidate.iterations,
                        candidate.converged,
                    ));
                }
            }
            Err(reason) => {
                summary.windows_skipped += 1;
                let (selected_count, observation_count, referenced_count, free_count) =
                    window_counts.unwrap_or_default();
                if preserve_optimized_points {
                    print_joint_rig_ba_filter_log(pass, format_args!(
                        "joint_rig_ba_filter_window start_frame={} end_frame={} active_frames={} status=skipped selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} raw_preserved_tracks=0 dlt_attempted_tracks=0 raw_fallback_reasons=none raw_counts_are_window_events=true reason={reason}",
                        active_frames.first().copied().unwrap_or_default(),
                        active_frames.last().copied().unwrap_or_default(),
                        active_frames.len(),
                        selected_count,
                        observation_count,
                        referenced_count,
                        free_count,
                    ));
                } else {
                    print_joint_rig_ba_filter_log(pass, format_args!(
                        "joint_rig_ba_filter_window start_frame={} end_frame={} active_frames={} status=skipped selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} reason={reason}",
                        active_frames.first().copied().unwrap_or_default(),
                        active_frames.last().copied().unwrap_or_default(),
                        active_frames.len(),
                        selected_count,
                        observation_count,
                        referenced_count,
                        free_count,
                    ));
                }
            }
        }
    }
    let validation_started = trace_timing.then(std::time::Instant::now);
    validate_filtered_model(&baseline_connectivity, store, images, cameras, landmarks)?;
    if let Some(started) = validation_started {
        eprintln!(
            "atlas-pass-timing: pass={} phase=final_validation seconds={:.9}",
            pass.unwrap_or(0),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(summary)
}
