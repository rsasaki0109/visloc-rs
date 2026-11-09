//! Landmark ordering, COLMAP model writing and output validation.

use super::*;

pub(super) fn landmark_observation_count(landmarks: &[LandmarkOutput]) -> usize {
    landmarks
        .iter()
        .map(|landmark| landmark.observations.len())
        .sum()
}

pub(super) fn canonical_landmark_indices(landmarks: &[LandmarkOutput]) -> Vec<usize> {
    let mut indices = (0..landmarks.len()).collect::<Vec<_>>();
    indices.sort_by_key(|index| {
        (
            landmarks[*index]
                .observations
                .first()
                .copied()
                .unwrap_or(ObservationKey {
                    global_image_id: u64::MAX,
                    keypoint_index: usize::MAX,
                }),
            landmarks[*index].track_id,
        )
    });
    indices
}

pub(super) fn write_model(
    out_dir: &Path,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    stats: &IngestStats,
) -> Result<SupportSummary, String> {
    write_model_ordered(out_dir, images, cameras, landmarks, stats, false)
}

pub(super) fn write_model_canonical(
    out_dir: &Path,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    stats: &IngestStats,
) -> Result<SupportSummary, String> {
    write_model_ordered(out_dir, images, cameras, landmarks, stats, true)
}

fn write_model_ordered(
    out_dir: &Path,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    stats: &IngestStats,
    canonical_order: bool,
) -> Result<SupportSummary, String> {
    fs::create_dir_all(out_dir)
        .map_err(|error| format!("create output {}: {error}", out_dir.display()))?;
    let indices = if canonical_order {
        canonical_landmark_indices(landmarks)
    } else {
        (0..landmarks.len()).collect::<Vec<_>>()
    };
    let ordered_landmarks = indices
        .iter()
        .map(|index| &landmarks[*index])
        .collect::<Vec<_>>();
    let used_cameras = images
        .values()
        .map(|image| image.atlas.camera_id)
        .collect::<BTreeSet<_>>();
    let mut cameras_text = String::from("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n");
    for camera_id in used_cameras {
        let camera = cameras
            .get(&camera_id)
            .ok_or_else(|| format!("missing output camera {camera_id}"))?;
        let model = camera_model_name(&camera.model)?;
        write!(
            cameras_text,
            "{} {} {} {}",
            camera.id, model, camera.width, camera.height
        )
        .unwrap();
        for parameter in &camera.params {
            write!(cameras_text, " {}", format_f64(*parameter)).unwrap();
        }
        cameras_text.push('\n');
    }

    let mut point_ids = BTreeMap::<ObservationKey, u64>::new();
    let mut points_text =
        String::from("# POINT3D_ID X Y Z R G B ERROR TRACK[] as IMAGE_ID POINT2D_IDX\n");
    for (index, landmark) in ordered_landmarks.iter().enumerate() {
        let point_id = index as u64 + 1;
        for key in &landmark.observations {
            if point_ids.insert(*key, point_id).is_some() {
                return Err(format!(
                    "observation {:?} belongs to multiple output landmarks",
                    key
                ));
            }
        }
        writeln!(
            points_text,
            "{} {} {} {} 255 255 255 {} {}",
            point_id,
            format_f64(landmark.position.x),
            format_f64(landmark.position.y),
            format_f64(landmark.position.z),
            format_f64(landmark.mean_error),
            landmark
                .observations
                .iter()
                .map(|key| format!("{} {}", key.global_image_id, key.keypoint_index))
                .collect::<Vec<_>>()
                .join(" ")
        )
        .unwrap();
    }

    let mut images_text = String::from(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as X Y POINT3D_ID\n",
    );
    let mut image_support = BTreeMap::<u64, usize>::new();
    for key in point_ids.keys() {
        *image_support.entry(key.global_image_id).or_default() += 1;
    }
    let mut frame_support = BTreeMap::<u64, (usize, usize)>::new();
    for image in images.values() {
        let entry = frame_support.entry(image.atlas.frame_id).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += image_support
            .get(&image.atlas.global_image_id)
            .copied()
            .unwrap_or(0);
        validate_image_name(&image.atlas.name)?;
        let quaternion = image.atlas.pose.world_to_camera.rotation.quaternion();
        let translation = image.atlas.pose.world_to_camera.translation;
        writeln!(
            images_text,
            "{} {} {} {} {} {} {} {} {} {}",
            image.atlas.global_image_id,
            format_f64(quaternion.w),
            format_f64(quaternion.i),
            format_f64(quaternion.j),
            format_f64(quaternion.k),
            format_f64(translation.x),
            format_f64(translation.y),
            format_f64(translation.z),
            image.atlas.camera_id,
            image.atlas.name
        )
        .unwrap();
        let mut tokens = Vec::with_capacity(image.keypoints.len());
        for (keypoint_index, xy) in image.keypoints.iter().enumerate() {
            let key = ObservationKey {
                global_image_id: image.atlas.global_image_id,
                keypoint_index,
            };
            let point_id = point_ids.get(&key).copied().unwrap_or(0);
            let point_text = if point_id == 0 {
                "-1".to_owned()
            } else {
                point_id.to_string()
            };
            tokens.push(format!(
                "{} {} {}",
                format_f64(xy.x),
                format_f64(xy.y),
                point_text
            ));
        }
        images_text.push_str(&tokens.join(" "));
        images_text.push('\n');
    }
    fs::write(out_dir.join("cameras.txt"), cameras_text)
        .map_err(|error| format!("write cameras.txt: {error}"))?;
    fs::write(out_dir.join("images.txt"), images_text)
        .map_err(|error| format!("write images.txt: {error}"))?;
    fs::write(out_dir.join("points3D.txt"), points_text)
        .map_err(|error| format!("write points3D.txt: {error}"))?;
    let mut image_support_text =
        String::from("# GLOBAL_IMAGE_ID FRAME_ID SENSOR_INDEX NAME RETAINED_OBSERVATIONS\n");
    let mut supported_image_count = 0;
    for image in images.values() {
        let support = image_support
            .get(&image.atlas.global_image_id)
            .copied()
            .unwrap_or(0);
        if support > 0 {
            supported_image_count += 1;
        }
        writeln!(
            image_support_text,
            "{} {} {} {} {}",
            image.atlas.global_image_id,
            image.atlas.frame_id,
            image.atlas.sensor_index,
            image.atlas.name,
            support
        )
        .unwrap();
    }
    let mut frame_support_text = String::from("# FRAME_ID IMAGE_COUNT RETAINED_OBSERVATIONS\n");
    let mut supported_frame_count = 0;
    for (frame_id, (image_count, support)) in &frame_support {
        if *support > 0 {
            supported_frame_count += 1;
        }
        writeln!(
            frame_support_text,
            "{} {} {}",
            frame_id, image_count, support
        )
        .unwrap();
    }
    fs::write(out_dir.join("image_support.tsv"), image_support_text)
        .map_err(|error| format!("write image_support.tsv: {error}"))?;
    fs::write(out_dir.join("frame_support.tsv"), frame_support_text)
        .map_err(|error| format!("write frame_support.tsv: {error}"))?;
    let support_summary = SupportSummary {
        atlas_image_count: images.len(),
        supported_image_count,
        zero_support_image_count: images.len() - supported_image_count,
        atlas_frame_count: frame_support.len(),
        supported_frame_count,
        zero_support_frame_count: frame_support.len() - supported_frame_count,
    };
    let summary = format!(
        "source_nodes\t{}\nsource_points\t{}\nsource_observations\t{}\naccepted_candidates\t{}\nextended_candidates\t{}\nduplicate_observations\t{}\nrejected_candidates\t{}\nrejected_missing_atlas_images\t{}\nfiltered_out_of_component_observations\t{}\nrejected_short_candidates\t{}\nrejected_same_image\t{}\nrejected_track_conflict\t{}\nrejected_triangulation\t{}\ntriangulation_sampled_tracks\t{}\ntriangulation_sampled_observations\t{}\natlas_poses\t{}\nsupported_images\t{}\nzero_support_images\t{}\natlas_frames\t{}\nsupported_frames\t{}\nzero_support_frames\t{}\noutput_landmarks\t{}\noutput_observations\t{}\n",
        stats.source_nodes,
        stats.source_points,
        stats.source_observations,
        stats.accepted_candidates,
        stats.extended_candidates,
        stats.duplicate_observations,
        stats.rejected_candidates,
        stats.rejected_missing_atlas_images,
        stats.filtered_out_of_component_observations,
        stats.rejected_short_candidates,
        stats.rejected_same_image,
        stats.rejected_track_conflict,
        stats.rejected_triangulation,
        stats.triangulation_sampled_tracks,
        stats.triangulation_sampled_observations,
        support_summary.atlas_image_count,
        support_summary.supported_image_count,
        support_summary.zero_support_image_count,
        support_summary.atlas_frame_count,
        support_summary.supported_frame_count,
        support_summary.zero_support_frame_count,
        stats.output_landmarks,
        stats.output_observations,
    );
    fs::write(out_dir.join("integration_summary.tsv"), summary)
        .map_err(|error| format!("write integration_summary.tsv: {error}"))?;
    Ok(support_summary)
}

pub(super) fn require_nonempty_landmarks(
    landmarks: &[LandmarkOutput],
    observation_count: usize,
) -> Result<(), String> {
    if landmarks.is_empty() || observation_count == 0 {
        return Err(
            "no landmarks survived conservative ownership, parallax, and reprojection gates; refusing to publish a pose-only model"
                .to_owned(),
        );
    }
    Ok(())
}

pub(super) fn validate_output_cameras(
    manifest: &RigManifest,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
) -> Result<(), String> {
    let mut frame_poses = BTreeMap::<u64, Pose>::new();
    for image in images.values() {
        let assignment = manifest
            .assignments
            .get(&image.atlas.name)
            .expect("atlas images were bound to manifest");
        let sensor = manifest
            .sensors
            .get(&assignment.sensor_index)
            .expect("manifest sensor index was validated");
        if image.atlas.frame_id != assignment.frame_id
            || image.atlas.sensor_index != assignment.sensor_index
        {
            return Err(format!(
                "atlas image {:?} frame/sensor assignment disagrees with manifest",
                image.atlas.name
            ));
        }
        let camera = cameras.get(&image.atlas.camera_id).ok_or_else(|| {
            format!(
                "atlas image {:?} has no camera definition",
                image.atlas.name
            )
        })?;
        if camera.id != sensor.camera_id
            || camera.width != sensor.width
            || camera.height != sensor.height
            || !camera_matches_manifest(camera, sensor)
        {
            return Err(format!(
                "atlas image {:?} camera calibration is not manifest-compatible",
                image.atlas.name
            ));
        }
        if !sensor
            .sensor_from_rig
            .translation
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(format!(
                "sensor {} has non-finite extrinsic",
                assignment.sensor_index
            ));
        }
        camera_model_name(&camera.model)?;
        let rig_pose = Pose {
            world_to_camera: sensor
                .sensor_from_rig
                .inverse()
                .compose(&image.atlas.pose.world_to_camera),
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
                    "atlas frame {} sensor poses disagree: centre={center_error:.9}m rotation={rotation_error:.9}deg",
                    image.atlas.frame_id
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn camera_matches_manifest(camera: &Camera, sensor: &SensorCalibration) -> bool {
    if !matches!(camera.model, CameraModel::Pinhole) || camera.params.len() != 4 {
        return false;
    }
    let expected = [sensor.fx, sensor.fy, sensor.cx, sensor.cy];
    camera.params[..4]
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= INTRINSIC_TOLERANCE)
}
