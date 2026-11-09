//! Global images, observation ingestion, track merging and triangulation.

use super::*;

pub(super) fn update_global_images(
    global_images: &mut BTreeMap<u64, GlobalImage>,
    source: &SourceWindow,
    image_name_to_id: &BTreeMap<String, u64>,
    stats: &mut IngestStats,
) -> Result<(), String> {
    for image in &source.images {
        let Some(global_image_id) = image_name_to_id.get(&image.name).copied() else {
            stats.rejected_invalid_source += 1;
            continue;
        };
        let Some(global) = global_images.get_mut(&global_image_id) else {
            return Err(format!("manifest/atlas lost image {:?}", image.name));
        };
        if global.atlas.camera_id != image.camera_id {
            return Err(format!(
                "image {:?} changes camera id across source windows",
                image.name
            ));
        }
        let keypoints = image
            .keypoints
            .iter()
            .map(|keypoint| keypoint.xy)
            .collect::<Vec<_>>();
        if global.keypoints.is_empty() {
            global.keypoints = keypoints;
        } else if global.keypoints.len() != keypoints.len()
            || global
                .keypoints
                .iter()
                .zip(keypoints.iter())
                .any(|(left, right)| !points_close(left, right))
        {
            return Err(format!(
                "image {:?} has inconsistent ordered keypoint coordinates across windows",
                image.name
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_global_images(
    global_images: &BTreeMap<u64, GlobalImage>,
) -> Result<(), String> {
    for image in global_images.values() {
        if image.keypoints.is_empty() {
            return Err(format!(
                "atlas image {:?} has no source keypoint array",
                image.atlas.name
            ));
        }
    }
    Ok(())
}

pub(super) fn ingest_source_points(
    source: &SourceWindow,
    image_name_to_id: &BTreeMap<String, u64>,
    global_images: &BTreeMap<u64, GlobalImage>,
    store: &mut TrackStore,
    stats: &mut IngestStats,
) -> Result<(), String> {
    let local_images = source
        .images
        .iter()
        .map(|image| (image.local_id, image))
        .collect::<BTreeMap<_, _>>();
    for point in &source.points {
        let mut candidate = Vec::with_capacity(point.observations.len());
        let mut filtered_out = 0;
        for observation in &point.observations {
            let image = local_images.get(&observation.image_id).ok_or_else(|| {
                format!(
                    "validated source point {} references missing image",
                    point.id
                )
            })?;
            let Some(global_image_id) = image_name_to_id.get(&image.name).copied() else {
                stats.rejected_missing_atlas_images += 1;
                filtered_out += 1;
                continue;
            };
            let global = global_images
                .get(&global_image_id)
                .expect("global image id came from the atlas manifest");
            if observation.keypoint_index >= global.keypoints.len() {
                return Err(format!(
                    "source point {} keypoint {} exceeds canonical image {:?}",
                    point.id, observation.keypoint_index, image.name
                ));
            }
            candidate.push(ObservationKey {
                global_image_id,
                keypoint_index: observation.keypoint_index,
            });
        }
        stats.filtered_out_of_component_observations += filtered_out;
        if candidate.len() < MIN_TRACK_OBSERVATIONS {
            stats.rejected_short_candidates += 1;
            continue;
        }
        candidate.sort_unstable();
        match merge_candidate(store, &candidate, global_images) {
            Ok(MergeOutcome::New) => stats.accepted_candidates += 1,
            Ok(MergeOutcome::Extended { duplicate_count }) => {
                stats.extended_candidates += 1;
                stats.duplicate_observations += duplicate_count;
            }
            Err(rejection) => {
                stats.rejected_candidates += 1;
                match rejection {
                    MergeRejection::SameImageDifferentKeypoint
                    | MergeRejection::ExistingTrackSameImageConflict => {
                        stats.rejected_same_image += 1;
                        if matches!(rejection, MergeRejection::ExistingTrackSameImageConflict) {
                            stats.rejected_track_conflict += 1;
                        }
                    }
                    MergeRejection::DuplicateCandidateObservation => stats.rejected_xy += 1,
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MergeOutcome {
    New,
    Extended { duplicate_count: usize },
}

pub(super) fn merge_candidate(
    store: &mut TrackStore,
    candidate: &[ObservationKey],
    global_images: &BTreeMap<u64, GlobalImage>,
) -> Result<MergeOutcome, MergeRejection> {
    if candidate.is_empty() {
        return Err(MergeRejection::DuplicateCandidateObservation);
    }
    let mut image_keypoints = BTreeMap::<u64, usize>::new();
    let mut owners = BTreeSet::new();
    for key in candidate {
        if let Some(previous) = image_keypoints.insert(key.global_image_id, key.keypoint_index) {
            if previous != key.keypoint_index {
                return Err(MergeRejection::SameImageDifferentKeypoint);
            }
            return Err(MergeRejection::DuplicateCandidateObservation);
        }
        if let Some(state) = store.observations.get(key) {
            owners.insert(state.owner_track);
        }
        let image = global_images
            .get(&key.global_image_id)
            .expect("candidate was built from atlas images");
        if key.keypoint_index >= image.keypoints.len() {
            return Err(MergeRejection::DuplicateCandidateObservation);
        }
    }
    let owner = owners.iter().next().copied();
    let mut combined = BTreeSet::new();
    for key in candidate {
        combined.insert(*key);
    }
    for owner_id in &owners {
        for key in &store.tracks[*owner_id].observations {
            combined.insert(*key);
        }
    }
    let mut combined_image_keypoints = BTreeMap::<u64, usize>::new();
    for key in &combined {
        if let Some(previous) =
            combined_image_keypoints.insert(key.global_image_id, key.keypoint_index)
        {
            if previous != key.keypoint_index {
                return Err(MergeRejection::ExistingTrackSameImageConflict);
            }
        }
    }
    let target = match owner {
        Some(owner) => owner,
        None => {
            let track_id = store.tracks.len();
            store.tracks.push(GlobalTrack {
                observations: Vec::new(),
            });
            track_id
        }
    };
    let duplicate_count = candidate
        .iter()
        .filter(|key| store.observations.contains_key(key))
        .count();
    if owners.len() > 1 {
        for owner_id in owners
            .iter()
            .copied()
            .filter(|owner_id| *owner_id != target)
        {
            store.tracks[owner_id].observations.clear();
        }
    }
    let combined = combined.into_iter().collect::<Vec<_>>();
    store.tracks[target].observations = combined.clone();
    for key in &combined {
        let image = global_images
            .get(&key.global_image_id)
            .expect("candidate was built from atlas images");
        let xy = image.keypoints[key.keypoint_index];
        match store.observations.get_mut(key) {
            Some(state) => {
                state.owner_track = target;
                state.xy = xy;
            }
            None => {
                store.observations.insert(
                    *key,
                    ObservationState {
                        xy,
                        owner_track: target,
                    },
                );
            }
        }
    }
    if owner.is_some() {
        Ok(MergeOutcome::Extended { duplicate_count })
    } else {
        Ok(MergeOutcome::New)
    }
}

pub(super) fn triangulate_track(
    track_id: usize,
    track: &GlobalTrack,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
) -> Result<LandmarkOutput, String> {
    triangulate_track_with_pose_overrides(
        track_id,
        track,
        observations,
        images,
        cameras,
        &BTreeMap::new(),
    )
}

pub(super) fn triangulate_track_with_pose_overrides(
    track_id: usize,
    track: &GlobalTrack,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<LandmarkOutput, String> {
    if track.observations.len() < MIN_TRACK_OBSERVATIONS {
        return Err("track has too few observations".to_owned());
    }
    let sample = bounded_sample(&track.observations, MAX_DLT_OBSERVATIONS);
    if !has_observable_parallax(&sample, observations, images, cameras, pose_overrides)? {
        return Err("track has no observable camera baseline/parallax".to_owned());
    }
    let mut matrix = DMatrix::<f64>::zeros(sample.len() * 2, 4);
    for (index, key) in sample.iter().enumerate() {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "track references unknown atlas image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "track references unknown camera".to_owned())?;
        let state = observations
            .get(key)
            .ok_or_else(|| "track references unknown observation".to_owned())?;
        let normalized = camera
            .normalize_pixel(&state.xy)
            .ok_or_else(|| "camera model cannot normalize pixel".to_owned())?;
        let pose = pose_for_image(image, pose_overrides);
        let pose_matrix = pose.matrix();
        let row_x = index * 2;
        let row_y = row_x + 1;
        for column in 0..4 {
            matrix[(row_x, column)] =
                normalized.x * pose_matrix[(2, column)] - pose_matrix[(0, column)];
            matrix[(row_y, column)] =
                normalized.y * pose_matrix[(2, column)] - pose_matrix[(1, column)];
        }
    }
    let svd = matrix.svd(false, true);
    let v_t = svd
        .v_t
        .ok_or_else(|| "triangulation SVD has no V^T".to_owned())?;
    let homogeneous = v_t.row(v_t.nrows() - 1);
    let w = homogeneous[3];
    if !w.is_finite() || w.abs() <= MIN_HOMOGENEOUS_SCALE {
        return Err("triangulation is degenerate".to_owned());
    }
    let position = Point3::new(homogeneous[0] / w, homogeneous[1] / w, homogeneous[2] / w);
    if !position.coords.iter().all(|value| value.is_finite()) {
        return Err("triangulated position is non-finite".to_owned());
    }
    let mut errors = Vec::with_capacity(track.observations.len());
    for key in &track.observations {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "track references unknown atlas image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "track references unknown camera".to_owned())?;
        let state = observations
            .get(key)
            .ok_or_else(|| "track references unknown observation".to_owned())?;
        let pose = pose_for_image(image, pose_overrides);
        let point_camera = pose.transform_world_point(&position);
        if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z <= 0.0 {
            return Err("triangulated point is behind a camera".to_owned());
        }
        let projected = camera
            .project(&point_camera)
            .ok_or_else(|| "projection failed".to_owned())?;
        let error = (projected - state.xy).norm();
        if !error.is_finite() {
            return Err("reprojection is non-finite".to_owned());
        }
        errors.push(error);
    }
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    let rms_error =
        (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    let max_error = errors.iter().copied().fold(0.0, f64::max);
    if mean_error > MAX_MEAN_REPROJECTION_PX || max_error > MAX_REPROJECTION_PX {
        return Err(format!(
            "reprojection gate failed mean={mean_error} max={max_error}"
        ));
    }
    Ok(LandmarkOutput {
        track_id,
        position,
        observations: track.observations.clone(),
        errors,
        rms_error,
        mean_error,
        max_error,
        dlt_sample_count: sample.len(),
    })
}

pub(super) fn has_observable_parallax(
    keys: &[ObservationKey],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<bool, String> {
    let mut rays = Vec::with_capacity(keys.len());
    for key in keys {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "track references unknown atlas image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "track references unknown camera".to_owned())?;
        let state = observations
            .get(key)
            .ok_or_else(|| "track references unknown observation".to_owned())?;
        let normalized = camera
            .normalize_pixel(&state.xy)
            .ok_or_else(|| "camera model cannot normalize pixel".to_owned())?;
        let bearing_camera = Vector3::new(normalized.x, normalized.y, 1.0).normalize();
        let pose = pose_for_image(image, pose_overrides);
        let bearing_world = pose
            .camera_to_world()
            .rotation
            .transform_vector(&bearing_camera);
        rays.push((pose.camera_center_world(), bearing_world));
    }
    for (index, (center_a, ray_a)) in rays.iter().enumerate() {
        for (center_b, ray_b) in rays.iter().skip(index + 1) {
            let baseline = (center_a - center_b).norm();
            let ray_cross = ray_a.cross(ray_b).norm();
            if baseline > MIN_BASELINE_M && ray_cross > MIN_RAY_CROSS_NORM {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
