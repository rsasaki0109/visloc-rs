//! COLMAP-model inputs: pose priors, track membership, per-image calibration, oracle BA probes and guided matching.

use super::*;

/// Essential matrix `E` with `x_jᵀ E x_i = 0` from absolute poses
/// (`X_j = R X_i + t`, `E = [t]× R`).
pub(super) fn essential_from_absolute_poses(pose_i: &Pose, pose_j: &Pose) -> Option<Matrix3<f64>> {
    let rel = pose_j
        .world_to_camera
        .compose(&pose_i.world_to_camera.inverse());
    let t = rel.translation;
    if t.norm() < 1e-9 {
        return None;
    }
    let r = rel.rotation.to_rotation_matrix().into_inner();
    let t_skew = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    Some(t_skew * r)
}

/// Parse COLMAP text `images.txt` into `{stem → Pose}` (world-to-camera).
pub(super) fn poses_from_colmap_images_txt(path: &Path) -> Result<HashMap<String, Pose>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let mut out = HashMap::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        // IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if parts.len() < 10 {
            continue;
        }
        let name = parts[9];
        if Path::new(name).extension().is_none() {
            continue;
        }
        let qw: f64 = parts[1].parse().map_err(|e| format!("{e}"))?;
        let qx: f64 = parts[2].parse().map_err(|e| format!("{e}"))?;
        let qy: f64 = parts[3].parse().map_err(|e| format!("{e}"))?;
        let qz: f64 = parts[4].parse().map_err(|e| format!("{e}"))?;
        let tx: f64 = parts[5].parse().map_err(|e| format!("{e}"))?;
        let ty: f64 = parts[6].parse().map_err(|e| format!("{e}"))?;
        let tz: f64 = parts[7].parse().map_err(|e| format!("{e}"))?;
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));
        let pose = Pose::from_world_to_camera(q, Vector3::new(tx, ty, tz));
        out.insert(
            Path::new(name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(name)
                .to_string(),
            pose,
        );
        // Skip points2D row when present.
        if let Some(nxt) = lines.peek() {
            let n = nxt.trim();
            if !n.is_empty() && !n.starts_with('#') {
                let np: Vec<&str> = n.split_whitespace().collect();
                let looks_image = np.len() >= 10
                    && np[0].parse::<i64>().is_ok()
                    && Path::new(np[9]).extension().is_some();
                if !looks_image {
                    lines.next();
                }
            }
        }
    }
    Ok(out)
}

/// Parse and validate a partial COLMAP pose model for the opt-in staged
/// incremental path. Unlike the diagnostics-only parser above, this helper
/// rejects duplicate/unknown image stems and checks the sibling camera model
/// before returning an index-aligned pose vector.
pub(super) fn initial_poses_from_colmap_images_txt(
    path: &Path,
    image_names: &[String],
    camera: &Camera,
) -> Result<Vec<Option<Pose>>, String> {
    initial_poses_from_colmap_images_txt_with_expected_cameras(path, image_names, camera, None)
}

/// Parse the opt-in initial-pose model while validating each pose's camera
/// against the loaded image calibration.  `expected_cameras` is `None` for
/// the historical shared-camera path; when present it is indexed like
/// `image_names` and permits a pose model to use several COLMAP camera IDs.
pub(super) fn initial_poses_from_colmap_images_txt_with_expected_cameras(
    path: &Path,
    image_names: &[String],
    camera: &Camera,
    expected_cameras: Option<&[Camera]>,
) -> Result<Vec<Option<Pose>>, String> {
    if let Some(expected_cameras) = expected_cameras {
        if expected_cameras.len() != image_names.len() {
            return Err(format!(
                "--initial-poses per-image camera count {} does not match loaded image count {}",
                expected_cameras.len(),
                image_names.len()
            ));
        }
    }
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read --initial-poses model {path:?}: {error}"))?;
    let mut entries: Vec<(String, u64, Pose)> = Vec::new();
    let mut source_stems = HashSet::new();
    for (line_number, line) in text.lines().enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 10 || parts[0].starts_with('#') {
            continue;
        }
        // A COLMAP points2D row can also contain many tokens. The image
        // header is distinguished by its integer image/camera ids and a
        // filename-like final token.
        let Ok(_image_id) = parts[0].parse::<u64>() else {
            continue;
        };
        let Ok(camera_id) = parts[8].parse::<u64>() else {
            continue;
        };
        let name = parts[9];
        if Path::new(name).extension().is_none() {
            continue;
        }
        let values = parts[1..8]
            .iter()
            .map(|value| value.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                format!(
                    "invalid pose values in --initial-poses {path:?} line {}: {error}",
                    line_number + 1
                )
            })?;
        if !values.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "non-finite pose in --initial-poses {path:?} line {}",
                line_number + 1
            ));
        }
        let quaternion_norm = values[0..4].iter().map(|value| value * value).sum::<f64>();
        if quaternion_norm <= 1.0e-24 {
            return Err(format!(
                "zero quaternion in --initial-poses {path:?} line {}",
                line_number + 1
            ));
        }
        let stem = image_stem(name).to_owned();
        if !source_stems.insert(stem.clone()) {
            return Err(format!(
                "duplicate image stem {stem:?} in --initial-poses model {path:?}"
            ));
        }
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            values[0], values[1], values[2], values[3],
        ));
        let pose = Pose::from_world_to_camera(q, Vector3::new(values[4], values[5], values[6]));
        entries.push((stem, camera_id, pose));
    }
    if entries.len() < 2 {
        return Err(format!(
            "--initial-poses model {path:?} contains only {} usable image poses; at least 2 are required",
            entries.len()
        ));
    }

    let mut loaded_by_stem = HashMap::new();
    for (image, name) in image_names.iter().enumerate() {
        let stem = image_stem(name).to_owned();
        if loaded_by_stem.insert(stem.clone(), image).is_some() {
            return Err(format!(
                "loaded images contain duplicate stem {stem:?}; --initial-poses cannot map it uniquely"
            ));
        }
    }
    let mut used_camera_ids = HashSet::new();
    let mut poses = vec![None; image_names.len()];
    let mut camera_id_by_image = vec![None; image_names.len()];
    for (stem, camera_id, pose) in entries {
        let Some(&image) = loaded_by_stem.get(&stem) else {
            return Err(format!(
                "--initial-poses model contains unknown image stem {stem:?}"
            ));
        };
        used_camera_ids.insert(camera_id);
        poses[image] = Some(pose);
        camera_id_by_image[image] = Some(camera_id);
    }
    let seeded = poses.iter().filter(|pose| pose.is_some()).count();
    if seeded < 2 {
        return Err(format!(
            "--initial-poses model overlaps loaded images at only {seeded} pose(s); at least 2 are required"
        ));
    }

    let camera_path = path
        .parent()
        .map(|parent| parent.join("cameras.txt"))
        .ok_or_else(|| format!("--initial-poses path {path:?} has no model directory"))?;
    let camera_text = std::fs::read_to_string(&camera_path).map_err(|error| {
        format!("--initial-poses requires readable sibling cameras.txt at {camera_path:?}: {error}")
    })?;
    if camera.tangential_distortion().is_some() {
        return Err(
            "--initial-poses COLMAP PINHOLE validation does not support nonzero input distortion"
                .into(),
        );
    }
    if let Some((k1, k2)) = camera.radial_distortion() {
        if k1.abs() > 1.0e-12 || k2.abs() > 1.0e-12 {
            return Err(
                "--initial-poses COLMAP PINHOLE validation does not support nonzero input distortion"
                    .into(),
            );
        }
    }
    let mut expected_by_camera_id: HashMap<u64, &Camera> = HashMap::new();
    if let Some(expected_cameras) = expected_cameras {
        for (image, camera_id) in camera_id_by_image.iter().enumerate() {
            let Some(camera_id) = camera_id else {
                continue;
            };
            let expected_camera = &expected_cameras[image];
            if let Some(previous) = expected_by_camera_id.insert(*camera_id, expected_camera) {
                if previous != expected_camera {
                    return Err(format!(
                        "--initial-poses camera id {camera_id} maps to incompatible loaded per-image calibrations"
                    ));
                }
            }
        }
    }
    let mut matched_camera_ids = HashSet::new();
    for (line_number, line) in camera_text.lines().enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 8 || line.trim_start().starts_with('#') {
            continue;
        }
        let Ok(camera_id) = parts[0].parse::<u64>() else {
            continue;
        };
        if !used_camera_ids.contains(&camera_id) {
            continue;
        }
        if parts[1] != "PINHOLE" {
            return Err(format!(
                "--initial-poses camera {camera_id} uses {}, expected PINHOLE (line {})",
                parts[1],
                line_number + 1
            ));
        }
        let expected_camera = expected_by_camera_id
            .get(&camera_id)
            .copied()
            .unwrap_or(camera);
        let expected = expected_camera.intrinsics().ok_or_else(|| {
            format!(
                "--initial-poses expected camera for CAMERA_ID {camera_id} has no finite pinhole intrinsics"
            )
        })?;
        let width = parts[2].parse::<u32>().map_err(|error| {
            format!(
                "invalid width in --initial-poses cameras.txt line {}: {error}",
                line_number + 1
            )
        })?;
        let height = parts[3].parse::<u32>().map_err(|error| {
            format!(
                "invalid height in --initial-poses cameras.txt line {}: {error}",
                line_number + 1
            )
        })?;
        if width != expected_camera.width || height != expected_camera.height {
            return Err(format!(
                "--initial-poses camera {camera_id} dimensions {width}x{height} disagree with input camera {}x{}",
                expected_camera.width, expected_camera.height
            ));
        }
        let params = parts[4..8]
            .iter()
            .map(|value| value.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                format!("invalid intrinsics in --initial-poses cameras.txt: {error}")
            })?;
        for (actual, expected_value) in params
            .iter()
            .zip([expected.0, expected.1, expected.2, expected.3])
        {
            let tolerance = 1.0e-6 * expected_value.abs().max(1.0);
            if !actual.is_finite() || (actual - expected_value).abs() > tolerance {
                return Err(format!(
                    "--initial-poses camera {camera_id} intrinsics disagree with input camera: model={params:?}, input=({:.9},{:.9},{:.9},{:.9})",
                    expected.0, expected.1, expected.2, expected.3
                ));
            }
        }
        if !matched_camera_ids.insert(camera_id) {
            return Err(format!(
                "duplicate camera id {camera_id} in --initial-poses cameras.txt"
            ));
        }
    }
    if matched_camera_ids.len() != used_camera_ids.len() {
        let missing: Vec<u64> = used_camera_ids
            .difference(&matched_camera_ids)
            .copied()
            .collect();
        return Err(format!(
            "--initial-poses cameras.txt is missing camera id(s) used by poses: {missing:?}"
        ));
    }
    Ok(poses)
}

/// Parsed observation-only membership from a COLMAP sparse model.  The
/// exporter deliberately discards point coordinates, colors, reprojection
/// errors, and camera poses: the mapper must re-triangulate this partition
/// using the currently loaded feature pixels and intrinsics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ColmapTrackMembership {
    pub(super) tracks: Vec<Vec<(usize, usize)>>,
    pub(super) source_points: usize,
    pub(super) source_observations: usize,
    pub(super) retained_observations: usize,
    pub(super) skipped_conflicting_points: usize,
    pub(super) skipped_conflicting_observations: usize,
}

/// Read COLMAP `points3D.txt` membership and validate it against the loaded
/// feature manifest.  COLMAP's text model identifies observations by
/// `(IMAGE_ID, POINT2D_IDX)`, so the sibling `images.txt` is required to map
/// image IDs to the loaded image names and to validate the point2D row count.
/// A few historical sparse models contain a point with two observations from
/// one image.  Such a point cannot be represented by the mapper's
/// one-observation-per-image invariant; it is excluded as an invalid source
/// track and counted explicitly instead of silently selecting one row.
pub(super) fn parse_colmap_track_membership(
    points_path: &Path,
    image_names: &[String],
    features: &[FeatureSet],
) -> Result<ColmapTrackMembership, String> {
    if image_names.len() != features.len() {
        return Err(format!(
            "COLMAP track membership manifest mismatch: {} image names vs {} feature sets",
            image_names.len(),
            features.len()
        ));
    }
    let images_path = points_path
        .parent()
        .map(|parent| parent.join("images.txt"))
        .ok_or_else(|| format!("{points_path:?} has no sibling images.txt directory"))?;
    let images_file = std::fs::File::open(&images_path)
        .map_err(|error| format!("cannot read COLMAP track sibling {images_path:?}: {error}"))?;
    let mut image_entries: HashMap<u64, (String, usize)> = HashMap::new();
    let mut image_names_seen = HashSet::new();
    let mut image_lines = BufReader::new(images_file).lines().enumerate();
    while let Some((line_index, line_result)) = image_lines.next() {
        let line = line_result.map_err(|error| {
            format!(
                "cannot read COLMAP images.txt line {}: {error}",
                line_index + 1
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 10 {
            continue;
        }
        let Ok(image_id) = parts[0].parse::<u64>() else {
            continue;
        };
        if parts[8].parse::<u64>().is_err() {
            continue;
        }
        let name = parts[9..].join(" ");
        if Path::new(&name).extension().is_none() {
            continue;
        }
        if image_entries.contains_key(&image_id) {
            return Err(format!(
                "duplicate IMAGE_ID {image_id} in COLMAP images.txt line {}",
                line_index + 1
            ));
        }
        if !image_names_seen.insert(name.clone()) {
            return Err(format!(
                "duplicate image name {name:?} in COLMAP images.txt line {}",
                line_index + 1
            ));
        }
        let points_line = image_lines.next().ok_or_else(|| {
            format!(
                "COLMAP images.txt has no POINTS2D row after IMAGE_ID {image_id} line {}",
                line_index + 1
            )
        })?;
        let points_line_number = points_line.0 + 1;
        let points_line = points_line.1.map_err(|error| {
            format!("cannot read COLMAP images.txt line {points_line_number}: {error}")
        })?;
        let point_tokens: Vec<&str> = points_line.split_whitespace().collect();
        if !point_tokens.len().is_multiple_of(3) {
            return Err(format!(
                "COLMAP images.txt POINTS2D row after IMAGE_ID {image_id} has {} fields, not a multiple of 3",
                point_tokens.len()
            ));
        }
        for chunk in point_tokens.chunks_exact(3) {
            chunk[0].parse::<f64>().map_err(|error| {
                format!(
                    "invalid POINTS2D x in COLMAP images.txt line {points_line_number}: {error}"
                )
            })?;
            chunk[1].parse::<f64>().map_err(|error| {
                format!(
                    "invalid POINTS2D y in COLMAP images.txt line {points_line_number}: {error}"
                )
            })?;
            chunk[2].parse::<i64>().map_err(|error| {
                format!(
                    "invalid POINTS2D point id in COLMAP images.txt line {points_line_number}: {error}"
                )
            })?;
        }
        image_entries.insert(image_id, (name, point_tokens.len() / 3));
    }
    if image_entries.len() != image_names.len() {
        return Err(format!(
            "COLMAP images.txt contains {} usable images, loaded feature manifest contains {}",
            image_entries.len(),
            image_names.len()
        ));
    }
    let mut loaded_by_name = HashMap::new();
    for (image, name) in image_names.iter().enumerate() {
        if loaded_by_name.insert(name.as_str(), image).is_some() {
            return Err(format!(
                "loaded feature manifest repeats image name {name:?}"
            ));
        }
    }
    let source_names: HashSet<&str> = image_entries
        .values()
        .map(|(name, _)| name.as_str())
        .collect();
    let missing: Vec<&str> = image_names
        .iter()
        .filter_map(|name| (!source_names.contains(name.as_str())).then_some(name.as_str()))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "COLMAP image/name manifest does not cover loaded images; missing={missing:?}"
        ));
    }
    let mut image_id_to_index = HashMap::new();
    for (image_id, (name, row_count)) in image_entries {
        let Some(&image) = loaded_by_name.get(name.as_str()) else {
            return Err(format!(
                "COLMAP images.txt image {name:?} (IMAGE_ID {image_id}) is absent from loaded feature manifest"
            ));
        };
        if row_count != features[image].keypoints.len()
            || row_count != features[image].descriptors.len()
        {
            return Err(format!(
                "POINTS2D row count for {name:?} is {row_count}, loaded feature set has {} keypoints / {} descriptors",
                features[image].keypoints.len(),
                features[image].descriptors.len()
            ));
        }
        image_id_to_index.insert(image_id, image);
    }
    debug_assert_eq!(image_id_to_index.len(), image_names.len());

    let points_file = std::fs::File::open(points_path)
        .map_err(|error| format!("cannot read COLMAP points3D file {points_path:?}: {error}"))?;
    let mut result = ColmapTrackMembership::default();
    let mut owned_observations = HashSet::new();
    for (line_index, line_result) in BufReader::new(points_file).lines().enumerate() {
        let line = line_result.map_err(|error| {
            format!(
                "cannot read COLMAP points3D line {}: {error}",
                line_index + 1
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 8 || !(parts.len() - 8).is_multiple_of(2) {
            return Err(format!(
                "COLMAP points3D line {} has malformed TRACK[] fields",
                line_index + 1
            ));
        }
        parts[0].parse::<u64>().map_err(|error| {
            format!(
                "invalid POINT3D_ID in COLMAP points3D line {}: {error}",
                line_index + 1
            )
        })?;
        for (column, value) in parts[1..8].iter().enumerate() {
            value.parse::<f64>().map_err(|error| {
                format!(
                    "invalid point metadata column {} in COLMAP points3D line {}: {error}",
                    column + 1,
                    line_index + 1
                )
            })?;
        }
        result.source_points += 1;
        let mut track = Vec::with_capacity((parts.len() - 8) / 2);
        let mut track_images = HashSet::new();
        let mut conflicting = false;
        for pair in parts[8..].chunks_exact(2) {
            let image_id = pair[0].parse::<u64>().map_err(|error| {
                format!(
                    "invalid IMAGE_ID in COLMAP points3D line {}: {error}",
                    line_index + 1
                )
            })?;
            let keypoint = pair[1].parse::<usize>().map_err(|error| {
                format!(
                    "invalid POINT2D_IDX in COLMAP points3D line {}: {error}",
                    line_index + 1
                )
            })?;
            let Some(&image) = image_id_to_index.get(&image_id) else {
                return Err(format!(
                    "COLMAP points3D line {} references unknown IMAGE_ID {image_id}",
                    line_index + 1
                ));
            };
            if keypoint >= features[image].keypoints.len() {
                return Err(format!(
                    "COLMAP points3D line {} references {name} keypoint {keypoint}, outside loaded feature rows",
                    line_index + 1,
                    name = image_names[image]
                ));
            }
            if !track_images.insert(image) {
                conflicting = true;
            }
            track.push((image, keypoint));
        }
        result.source_observations += track.len();
        if conflicting {
            result.skipped_conflicting_points += 1;
            result.skipped_conflicting_observations += track.len();
            continue;
        }
        for &observation in &track {
            if !owned_observations.insert(observation) {
                return Err(format!(
                    "COLMAP points3D contains observation ({},{}) in more than one point",
                    observation.0, observation.1
                ));
            }
        }
        result.retained_observations += track.len();
        result.tracks.push(track);
    }
    Ok(result)
}

/// Estimate the Sim(3) that maps one camera-centre set into another. This is
/// only used by the opt-in COLMAP-basin BA probe to put our reconstructed
/// landmarks in the injected pose frame; it follows the same Umeyama
/// convention as `scripts/score_umeyama_centers.py`.
fn umeyama_centres(
    source: &[Vector3<f64>],
    target: &[Vector3<f64>],
) -> Result<(f64, Matrix3<f64>, Vector3<f64>), String> {
    if source.len() != target.len() || source.len() < 3 {
        return Err(format!(
            "Sim(3) alignment needs at least 3 paired centres, got {} and {}",
            source.len(),
            target.len()
        ));
    }
    let n = source.len() as f64;
    let source_mean = source.iter().copied().sum::<Vector3<f64>>() / n;
    let target_mean = target.iter().copied().sum::<Vector3<f64>>() / n;
    let source_zero: Vec<Vector3<f64>> = source.iter().map(|p| *p - source_mean).collect();
    let target_zero: Vec<Vector3<f64>> = target.iter().map(|p| *p - target_mean).collect();
    let mut covariance = Matrix3::zeros();
    let mut source_variance = 0.0;
    for (src, dst) in source_zero.iter().zip(&target_zero) {
        covariance += dst * src.transpose();
        source_variance += src.norm_squared();
    }
    covariance /= n;
    source_variance /= n;
    if !source_variance.is_finite() || source_variance <= f64::EPSILON {
        return Err("Sim(3) source centres have zero variance".into());
    }
    let svd = covariance.svd(true, true);
    let u = svd.u.ok_or("Sim(3) SVD did not return U")?;
    let v_t = svd.v_t.ok_or("Sim(3) SVD did not return V^T")?;
    let mut correction = Matrix3::identity();
    if u.determinant() * v_t.determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    let rotation = u * correction * v_t;
    let numerator = svd.singular_values[0] * correction[(0, 0)]
        + svd.singular_values[1] * correction[(1, 1)]
        + svd.singular_values[2] * correction[(2, 2)];
    let scale = numerator / source_variance;
    let translation = target_mean - scale * (rotation * source_mean);
    if !scale.is_finite() || scale <= 0.0 || !translation.iter().all(|v| v.is_finite()) {
        return Err("Sim(3) alignment produced a non-finite or non-positive scale".into());
    }
    Ok((scale, rotation, translation))
}

fn transform_point_by_sim3(
    point: Point3<f64>,
    scale: f64,
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
) -> Point3<f64> {
    Point3::from(scale * (rotation * point.coords) + translation)
}

fn mean_track_reprojection(
    camera: &Camera,
    tracks: &[visloc_rs::slam::SfmTrack],
    poses: &[Option<Pose>],
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for track in tracks {
        for &(image, _, observed) in &track.observations {
            let Some(Some(pose)) = poses.get(image) else {
                continue;
            };
            let Some(projected) = camera.project(&pose.transform_world_point(&track.position))
            else {
                continue;
            };
            let error = (projected - observed).norm();
            if error.is_finite() {
                sum += error;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Replace an already-complete incremental result with an external COLMAP
/// pose basin and run one fixed-support BA solve. The mapper's track topology
/// and observations remain the support under test; only the world-frame point
/// coordinates are carried through the same Sim(3) used to align the two pose
/// sets. This function is called only by `--diagnose-ba-oracle-poses`.
pub(super) fn run_oracle_pose_ba_probe(
    result: &mut visloc_rs::slam::IncrementalSfmResult,
    features: &[FeatureSet],
    image_names: &[String],
    camera: &Camera,
    config: &IncrementalSfmConfig,
    oracle_path: &Path,
) -> Result<(f64, f64, f64, visloc_rs::BaResult), String> {
    let oracle_by_stem = poses_from_colmap_images_txt(oracle_path)?;
    let mut source_centres = Vec::with_capacity(image_names.len());
    let mut target_centres = Vec::with_capacity(image_names.len());
    let mut oracle_poses = Vec::with_capacity(image_names.len());
    for (image, name) in image_names.iter().enumerate() {
        let Some(estimated) = result.poses.get(image).and_then(Option::as_ref) else {
            return Err(format!(
                "oracle BA probe requires every mapper pose; image {image} ({name}) is missing"
            ));
        };
        let stem = image_stem(name);
        let Some(oracle) = oracle_by_stem.get(stem) else {
            return Err(format!("oracle pose is missing image stem {stem:?}"));
        };
        source_centres.push(estimated.camera_center_world().coords);
        target_centres.push(oracle.camera_center_world().coords);
        oracle_poses.push(Some(oracle.clone()));
    }
    let (scale, rotation, translation) = umeyama_centres(&source_centres, &target_centres)?;
    let mut oracle_tracks = result.tracks.clone();
    for track in &mut oracle_tracks {
        track.position = transform_point_by_sim3(track.position, scale, &rotation, &translation);
    }
    let initial_reprojection = mean_track_reprojection(camera, &oracle_tracks, &oracle_poses);
    let mut probe_config = config.clone();
    // The caller wants the same ordinary BA objective/schedule as the mapper;
    // this probe is not allowed to trigger the separate optional polish pass.
    probe_config.final_ba_polish_iterations = 0;
    let (ba_result, refined_camera) = run_fixed_support_bundle_adjustment(
        camera,
        features,
        &mut oracle_tracks,
        &probe_config,
        &mut oracle_poses,
    )
    .map_err(|error| format!("oracle BA probe failed: {error:?}"))?;
    let final_reprojection = mean_track_reprojection(camera, &oracle_tracks, &oracle_poses);
    result.poses = oracle_poses;
    result.tracks = oracle_tracks;
    result.mean_reprojection_px = final_reprojection;
    result.ba_result = Some(ba_result.clone());
    result.refined_camera = refined_camera;
    Ok((scale, initial_reprojection, final_reprojection, ba_result))
}

/// Build rotations for the fixed-rotation BA diagnostic.  A source path is
/// parsed as COLMAP `images.txt`; `current`/`champion` use the completed
/// incremental rotations themselves.  External rotations are right-aligned
/// to the current world gauge using the lowest-index registered image, while
/// the current translations are deliberately retained.
fn fixed_rotation_targets(
    result: &visloc_rs::slam::IncrementalSfmResult,
    image_names: &[String],
    source: &str,
) -> Result<(Vec<Option<Pose>>, String), String> {
    let mut targets = vec![None; result.poses.len()];
    let registered: Vec<usize> = result
        .poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| pose.as_ref().map(|_| image))
        .collect();
    if registered.is_empty() {
        return Err("fixed-rotation BA requires at least one registered pose".into());
    }
    if source == "current" || source == "champion" {
        for &image in &registered {
            targets[image] = result.poses[image].clone();
        }
        return Ok((targets, source.to_owned()));
    }

    let source_by_stem = poses_from_colmap_images_txt(Path::new(source))?;
    let anchor = registered[0];
    let anchor_stem = image_stem(&image_names[anchor]);
    let source_anchor = source_by_stem
        .get(anchor_stem)
        .ok_or_else(|| format!("fixed-rotation source is missing anchor stem {anchor_stem:?}"))?;
    let current_anchor = result.poses[anchor]
        .as_ref()
        .expect("registered pose exists");
    // For world-to-camera rotations, a global world-frame change acts on the
    // right.  Q_inv maps the source frame into the current mapper gauge.
    let q_inv =
        source_anchor.world_to_camera.rotation.inverse() * current_anchor.world_to_camera.rotation;
    for &image in &registered {
        let stem = image_stem(&image_names[image]);
        let source_pose = source_by_stem
            .get(stem)
            .ok_or_else(|| format!("fixed-rotation source is missing stem {stem:?}"))?;
        let current_pose = result.poses[image]
            .as_ref()
            .expect("registered pose exists");
        targets[image] = Some(Pose::from_world_to_camera(
            source_pose.world_to_camera.rotation * q_inv,
            current_pose.world_to_camera.translation,
        ));
    }
    Ok((targets, source.to_owned()))
}

/// Replace the incremental model's rotations by the requested diagnostic
/// targets and run translation/landmark-only BA on its exact support.
pub(super) fn run_fixed_rotation_ba_probe(
    result: &mut visloc_rs::slam::IncrementalSfmResult,
    features: &[FeatureSet],
    image_names: &[String],
    camera: &Camera,
    config: &IncrementalSfmConfig,
    source: &str,
) -> Result<(String, usize, f64, f64, f64, visloc_rs::BaResult), String> {
    let (targets, label) = fixed_rotation_targets(result, image_names, source)?;
    let fixed_count = targets.iter().filter(|pose| pose.is_some()).count();
    let initial_reprojection = mean_track_reprojection(camera, &result.tracks, &targets);
    let mut probe_config = config.clone();
    // The fixed-rotation diagnostic is a pose/structure decomposition.  Keep
    // the intrinsics and separate final polish schedule out of the probe so
    // only the requested pose constraint changes the experiment.
    probe_config.refine_intrinsics = false;
    probe_config.final_ba_polish_iterations = 0;
    probe_config.geometry_weighted_ba = false;
    let (ba_result, refined_camera) = run_fixed_rotation_support_bundle_adjustment(
        camera,
        features,
        &mut result.tracks,
        &probe_config,
        &mut result.poses,
        &targets,
    )
    .map_err(|error| format!("fixed-rotation BA probe failed: {error:?}"))?;
    let final_reprojection = mean_track_reprojection(camera, &result.tracks, &result.poses);
    let mut max_rotation_delta = 0.0f64;
    for (after, target) in result.poses.iter().zip(&targets) {
        let (Some(after), Some(target)) = (after.as_ref(), target.as_ref()) else {
            continue;
        };
        let delta = (target.world_to_camera.rotation.inverse() * after.world_to_camera.rotation)
            .angle()
            .to_degrees();
        if delta.is_finite() {
            max_rotation_delta = max_rotation_delta.max(delta);
        }
    }
    result.mean_reprojection_px = final_reprojection;
    result.ba_result = Some(ba_result.clone());
    result.refined_camera = refined_camera;
    Ok((
        label,
        fixed_count,
        initial_reprojection,
        final_reprojection,
        max_rotation_delta,
        ba_result,
    ))
}

/// COLMAP-style guided matching (`FeaturePairsMatching`'s
/// `FindGuidedMatches`): given the pair's verified essential geometry,
/// rematch descriptors that the initial NN+ratio pass missed under an
/// epipolar constraint. For every not-yet-matched query descriptor the best
/// unused train descriptor is accepted only when **both** the Lowe ratio
/// (`0.9`, looser than the main pass) and the squared Sampson distance
/// (`guided_max_error_px`) pass — pure geometric admission without a ratio
/// gate is what produced M5's false-bridge failure, so this stays
/// deliberately conservative. Conflicts (two queries claiming one train)
/// resolve to the smaller descriptor distance, greedy by distance order.
///
/// When `pose_essential` is `Some`, that matrix is used for the Sampson gate
/// (pose-guided rematch after global); otherwise E is estimated from
/// `inlier_corrs` via normalized eight-point.
pub(super) fn guided_epipolar_matches(
    camera: &Camera,
    features_i: &FeatureSet,
    features_j: &FeatureSet,
    initial: &[DescriptorMatch],
    inlier_corrs: &[TwoViewCorrespondence],
    max_error_px: f64,
    pose_essential: Option<Matrix3<f64>>,
    max_lowe_ratio: f64,
) -> Vec<DescriptorMatch> {
    let essential = if let Some(e) = pose_essential {
        e
    } else {
        let Some(e) = EssentialMatrixEstimator::estimate(
            &EightPointEssentialMatrixEstimator::default(),
            inlier_corrs,
            camera,
        ) else {
            return Vec::new();
        };
        e
    };
    let (fx, fy, _, _) = camera.intrinsics().unwrap_or((1.0, 1.0, 0.0, 0.0));
    let focal = 0.5 * (fx + fy);
    let max_sq_norm = (max_error_px / focal).powi(2);

    let normalize_all = |keypoints: &[Point2<f64>]| -> Vec<Option<[f64; 3]>> {
        keypoints
            .iter()
            .map(|p| camera.normalize_pixel(p).map(|n| [n.x, n.y, 1.0]))
            .collect()
    };
    let norm_i = normalize_all(&features_i.keypoints);
    let norm_j = normalize_all(&features_j.keypoints);
    let sampson_sq = |ni: &[f64; 3], nj: &[f64; 3]| -> Option<f64> {
        let e_ni = essential * nalgebra::Vector3::new(ni[0], ni[1], ni[2]);
        let et_nj = essential.transpose() * nalgebra::Vector3::new(nj[0], nj[1], nj[2]);
        let numerator = nalgebra::Vector3::new(nj[0], nj[1], nj[2])
            .dot(&e_ni)
            .powi(2);
        let denominator = e_ni.x * e_ni.x + e_ni.y * e_ni.y + et_nj.x * et_nj.x + et_nj.y * et_nj.y;
        if denominator < 1e-18 {
            None
        } else {
            Some(numerator / denominator)
        }
    };

    let mut used_query = vec![false; features_i.descriptors.len()];
    let mut used_train = vec![false; features_j.descriptors.len()];
    for m in initial {
        used_query[m.query_index] = true;
        used_train[m.train_index] = true;
    }

    // Descriptor-distance matrix over the full pair (one GEMM), rows =
    // queries, cols = trains.
    let n_q = features_i.descriptors.len();
    let n_t = features_j.descriptors.len();
    if n_q == 0 || n_t == 0 || features_i.descriptors[0].is_empty() {
        return Vec::new();
    }
    let dim = features_i.descriptors[0].len();
    let q = nalgebra::DMatrix::from_fn(n_q, dim, |a, k| features_i.descriptors[a][k] as f64);
    let t = nalgebra::DMatrix::from_fn(n_t, dim, |b, k| features_j.descriptors[b][k] as f64);
    let dist = &q * &t.transpose();

    struct Candidate {
        query: usize,
        train: usize,
        distance: f32,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for qi in 0..n_q {
        if used_query[qi] || norm_i[qi].is_none() {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        let mut second: f64 = f64::INFINITY;
        for tj in 0..n_t {
            if used_train[tj] {
                continue;
            }
            let d = ((dist[(qi, tj)]).max(0.0)).sqrt();
            if d < second {
                if d < best.map_or(f64::INFINITY, |(_, bd)| bd) {
                    second = best.map_or(f64::INFINITY, |(_, bd)| bd);
                    best = Some((tj, d));
                } else {
                    second = d;
                }
            }
        }
        let Some((tj, d)) = best else { continue };
        if d <= 0.0 || d >= second {
            continue;
        }
        if d / second > max_lowe_ratio {
            continue;
        }
        let Some(nj) = norm_j[tj] else { continue };
        let Some(sq) = sampson_sq(&norm_i[qi].unwrap(), &nj) else {
            continue;
        };
        if sq <= max_sq_norm {
            candidates.push(Candidate {
                query: qi,
                train: tj,
                distance: d as f32,
            });
        }
    }
    candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    let mut taken_train = used_train;
    let mut out = Vec::new();
    for c in candidates {
        if taken_train[c.train] {
            continue;
        }
        taken_train[c.train] = true;
        out.push(DescriptorMatch {
            query_index: c.query,
            train_index: c.train,
            distance: c.distance,
            second_best_distance: None,
            ratio: None,
            confidence: None,
        });
    }
    out
}

/// The model that COLMAP's `FindGuidedMatches` selects from a verified
/// `TwoViewGeometry`.  In particular, an `UNCALIBRATED` report means that F
/// won model selection (the report has no camera-specific calibration object),
/// while `CALIBRATED` selects E.  Homography-only configurations use H and
/// unresolved/multiple configurations are deliberately not guided.
#[derive(Debug, Clone, Copy)]
pub(super) enum ColmapGuidedGeometry {
    Essential(Matrix3<f64>),
    Fundamental(Matrix3<f64>),
    Homography(Matrix3<f64>),
}

pub(super) fn colmap_guided_geometry(
    report: &TwoViewGeometryReport,
) -> Option<ColmapGuidedGeometry> {
    match report.config {
        ConfigurationType::Calibrated => report.essential.map(ColmapGuidedGeometry::Essential),
        ConfigurationType::Uncalibrated => {
            report.fundamental.map(ColmapGuidedGeometry::Fundamental)
        }
        ConfigurationType::Planar
        | ConfigurationType::Panoramic
        | ConfigurationType::PlanarOrPanoramic => {
            report.homography.map(ColmapGuidedGeometry::Homography)
        }
        ConfigurationType::Undefined
        | ConfigurationType::Degenerate
        | ConfigurationType::Watermark
        | ConfigurationType::Multiple => None,
    }
}

pub(super) const fn colmap_guided_geometry_name(
    geometry: Option<ColmapGuidedGeometry>,
) -> &'static str {
    match geometry {
        Some(ColmapGuidedGeometry::Essential(_)) => "E",
        Some(ColmapGuidedGeometry::Fundamental(_)) => "F",
        Some(ColmapGuidedGeometry::Homography(_)) => "H",
        None => "none",
    }
}

/// Append-only guided matching with COLMAP's geometry and descriptor rules.
///
/// COLMAP first masks a full descriptor-distance matrix using the selected
/// E/F/H model, then applies the ordinary two-nearest-neighbour matcher in
/// both directions.  The production demo's historical guided path predates
/// this compatibility mode and remains untouched; it has a known dot-product
/// distance quirk and is kept for reproducibility.  This function fixes that
/// mismatch behind `--colmap-guided-matching`, while retaining every endpoint
/// used by the initial match set and never replacing an initial match.
pub(super) fn colmap_guided_matches(
    camera: &Camera,
    features_i: &FeatureSet,
    features_j: &FeatureSet,
    initial: &[DescriptorMatch],
    report: &TwoViewGeometryReport,
    max_error_px: f64,
    max_lowe_ratio: f64,
    cross_check: bool,
) -> Vec<DescriptorMatch> {
    let geometry = colmap_guided_geometry(report);
    let Some(geometry) = geometry else {
        return Vec::new();
    };
    if !max_error_px.is_finite() || max_error_px < 0.0 {
        return Vec::new();
    }
    if !max_lowe_ratio.is_finite() || max_lowe_ratio <= 0.0 {
        return Vec::new();
    }

    let n_q = features_i.keypoints.len().min(features_i.descriptors.len());
    let n_t = features_j.keypoints.len().min(features_j.descriptors.len());
    if n_q == 0 || n_t == 0 {
        return Vec::new();
    }

    let essential_threshold =
        TwoViewGeometryOptions::for_camera(camera, max_error_px).essential_sampson_threshold;
    let essential_threshold_sq = essential_threshold * essential_threshold;
    let pixel_threshold_sq = max_error_px * max_error_px;
    let geometry_accepts = |query: usize, train: usize| -> bool {
        let Some(previous_xy) = features_i.keypoints.get(query).copied() else {
            return false;
        };
        let Some(current_xy) = features_j.keypoints.get(train).copied() else {
            return false;
        };
        let correspondence = TwoViewCorrespondence::new(previous_xy, current_xy);
        match geometry {
            ColmapGuidedGeometry::Essential(essential) => {
                normalized_essential_squared_sampson_error(&essential, &correspondence, camera)
                    .is_some_and(|error| error <= essential_threshold_sq)
            }
            ColmapGuidedGeometry::Fundamental(fundamental) => {
                let error = fundamental_squared_sampson_error(&fundamental, &correspondence);
                error.is_finite() && error <= pixel_threshold_sq
            }
            ColmapGuidedGeometry::Homography(homography) => {
                homography_squared_error(&homography, &correspondence)
                    .is_some_and(|error| error.is_finite() && error <= pixel_threshold_sq)
            }
        }
    };

    let mut used_query = vec![false; n_q];
    let mut used_train = vec![false; n_t];
    for descriptor_match in initial {
        if descriptor_match.query_index < n_q && descriptor_match.train_index < n_t {
            used_query[descriptor_match.query_index] = true;
            used_train[descriptor_match.train_index] = true;
        }
    }

    // COLMAP's SIFT descriptors are byte-equivalent vectors with an L2 norm
    // near 512, and its default max-distance is 0.7 in that scale.  Keep the
    // threshold fixed in this compatibility mode rather than silently tying
    // it to the main pass's Lowe ratio.
    const COLMAP_MAX_DESCRIPTOR_DISTANCE: f64 = 512.0 * 0.7;

    #[derive(Debug, Clone, Copy)]
    struct Candidate {
        query: usize,
        train: usize,
        distance_sq: f64,
    }

    let nearest = |query: usize, train_filter: Option<usize>| -> Option<(usize, f64, f64)> {
        let mut best: Option<(usize, f64)> = None;
        let mut second: Option<(usize, f64)> = None;
        for train in 0..n_t {
            if train_filter == Some(train) {
                continue;
            }
            if !geometry_accepts(query, train) {
                continue;
            }
            let distance_sq = descriptor_squared_distance(
                features_i.descriptors.get(query)?,
                features_j.descriptors.get(train)?,
            );
            if !distance_sq.is_finite() {
                continue;
            }
            if best.is_none_or(|(_, current)| distance_sq < current) {
                second = best;
                best = Some((train, distance_sq));
            } else if second.is_none_or(|(_, current)| distance_sq < current) {
                second = Some((train, distance_sq));
            }
        }
        let (train, best_distance_sq) = best?;
        Some((
            train,
            best_distance_sq,
            second.map_or(f64::INFINITY, |(_, distance_sq)| distance_sq),
        ))
    };

    let passes_ratio_and_distance = |distance_sq: f64, second_sq: f64| -> bool {
        if !distance_sq.is_finite() || distance_sq > COLMAP_MAX_DESCRIPTOR_DISTANCE.powi(2) {
            return false;
        }
        if second_sq.is_finite() {
            let distance = distance_sq.sqrt();
            let second = second_sq.sqrt();
            distance.is_finite() && second.is_finite() && distance < max_lowe_ratio * second
        } else {
            true
        }
    };

    let mut forward = Vec::new();
    for (query, used) in used_query.iter().enumerate().take(n_q) {
        if *used {
            continue;
        }
        let Some((train, distance_sq, second_sq)) = nearest(query, None) else {
            continue;
        };
        if passes_ratio_and_distance(distance_sq, second_sq) {
            forward.push(Candidate {
                query,
                train,
                distance_sq,
            });
        }
    }

    if cross_check {
        // This is the same mutual-NN test as COLMAP's second
        // `FindBestMatchesIndex` call.  A train descriptor's reverse nearest
        // query is found under the same geometry mask and ratio/distance bar.
        let mut reverse_best = vec![None; n_t];
        for (train, reverse_slot) in reverse_best.iter_mut().enumerate().take(n_t) {
            let mut best: Option<(usize, f64)> = None;
            let mut second: Option<(usize, f64)> = None;
            for query in 0..n_q {
                if !geometry_accepts(query, train) {
                    continue;
                }
                let distance_sq = descriptor_squared_distance(
                    features_i.descriptors.get(query).unwrap_or(&Vec::new()),
                    features_j.descriptors.get(train).unwrap_or(&Vec::new()),
                );
                if !distance_sq.is_finite() {
                    continue;
                }
                if best.is_none_or(|(_, current)| distance_sq < current) {
                    second = best;
                    best = Some((query, distance_sq));
                } else if second.is_none_or(|(_, current)| distance_sq < current) {
                    second = Some((query, distance_sq));
                }
            }
            if let Some((query, distance_sq)) = best {
                let second_sq = second.map_or(f64::INFINITY, |(_, value)| value);
                if passes_ratio_and_distance(distance_sq, second_sq) {
                    *reverse_slot = Some(query);
                }
            }
        }
        forward.retain(|candidate| reverse_best[candidate.train] == Some(candidate.query));
    }

    // FindGuidedMatches itself has no append-only conflict stage, but the
    // demo must not replace an existing match.  Resolve any residual conflicts
    // deterministically by distance, then physical row indices.
    forward.sort_by(|lhs, rhs| {
        lhs.distance_sq
            .total_cmp(&rhs.distance_sq)
            .then_with(|| lhs.query.cmp(&rhs.query))
            .then_with(|| lhs.train.cmp(&rhs.train))
    });
    let mut taken_train = used_train;
    let mut out = Vec::with_capacity(forward.len());
    for candidate in forward {
        if taken_train[candidate.train] {
            continue;
        }
        taken_train[candidate.train] = true;
        let distance = candidate.distance_sq.sqrt();
        out.push(DescriptorMatch {
            query_index: candidate.query,
            train_index: candidate.train,
            distance: distance as f32,
            second_best_distance: None,
            ratio: None,
            confidence: None,
        });
    }
    out.sort_by_key(|candidate| candidate.query_index);
    out
}

/// Camera assignment retained alongside the validated, index-aligned rig.
/// The native camera rows are kept for the multi-camera COLMAP exporter while
/// the rig's first camera is used as the internal ray convention.
#[derive(Debug, Clone)]
pub(super) struct LoadedPerImageCalibration {
    pub(super) rig: PerImageCameras,
    pub(super) native_cameras: Vec<Camera>,
}

/// Parse the `IMAGE_ID ... CAMERA_ID NAME` headers from COLMAP text
/// `images.txt`.  The following POINTS2D line is skipped, so a point record
/// whose first token happens to be an integer cannot be mistaken for an image.
pub(super) fn parse_colmap_image_camera_assignments(
    contents: &str,
) -> Result<HashMap<String, u64>, String> {
    let mut assignments = HashMap::new();
    let mut lines = contents.lines();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 10
            || tokens[0].parse::<u64>().is_err()
            || tokens[8].parse::<u64>().is_err()
            || tokens[9].parse::<f64>().is_ok()
        {
            // A valid image header has ten fields.  Invalid non-header lines
            // are ignored for compatibility with COLMAP comments/exports.
            continue;
        }
        let camera_id = tokens[8]
            .parse::<u64>()
            .map_err(|error| format!("invalid CAMERA_ID {:?} in images.txt: {error}", tokens[8]))?;
        let name = tokens[9..].join(" ");
        if name.is_empty() {
            return Err("images.txt contains an image header with an empty NAME".into());
        }
        if assignments.insert(name.clone(), camera_id).is_some() {
            return Err(format!("duplicate image NAME {name:?} in images.txt"));
        }
        // COLMAP always writes the points row, including when it is empty.
        let _ = lines.next();
    }
    if assignments.is_empty() {
        return Err("images.txt contains no usable image headers".into());
    }
    Ok(assignments)
}

fn calibration_name_candidates(name: &str) -> Vec<String> {
    let path = Path::new(name);
    let mut candidates = vec![name.to_owned()];
    if let Some(base) = path.file_name().and_then(|value| value.to_str()) {
        if !candidates.iter().any(|candidate| candidate == base) {
            candidates.push(base.to_owned());
        }
    }
    let stem = image_stem(name);
    if !candidates
        .iter()
        .any(|candidate| image_stem(candidate) == stem)
    {
        candidates.push(stem.to_owned());
    }
    candidates
}

/// Resolve each loaded image name to the camera id declared by COLMAP and
/// validate the supported PINHOLE parameter contract.  Resolution accepts an
/// exact path, basename, or unique stem, which covers feature files that use a
/// different image extension while still rejecting ambiguous mappings.
pub(super) fn resolve_input_colmap_calibration(
    model_dir: &Path,
    image_names: &[String],
) -> Result<LoadedPerImageCalibration, String> {
    let cameras_path = model_dir.join("cameras.txt");
    let images_path = model_dir.join("images.txt");
    let camera_text = std::fs::read_to_string(&cameras_path)
        .map_err(|error| format!("cannot read calibration {cameras_path:?}: {error}"))?;
    let image_text = std::fs::read_to_string(&images_path)
        .map_err(|error| format!("cannot read calibration {images_path:?}: {error}"))?;
    let parsed_cameras = visloc_io::colmap::parse_cameras_txt(&camera_text)
        .map_err(|error| format!("cannot parse calibration cameras.txt: {error}"))?;
    let mut cameras_by_id = HashMap::new();
    for camera in parsed_cameras {
        if cameras_by_id.insert(camera.id, camera).is_some() {
            return Err("cameras.txt contains duplicate CAMERA_ID".into());
        }
    }
    let assignments = parse_colmap_image_camera_assignments(&image_text)?;
    let mut by_basename: HashMap<String, Vec<(String, u64)>> = HashMap::new();
    let mut by_stem: HashMap<String, Vec<(String, u64)>> = HashMap::new();
    for (name, camera_id) in &assignments {
        if let Some(base) = Path::new(name).file_name().and_then(|value| value.to_str()) {
            by_basename
                .entry(base.to_owned())
                .or_default()
                .push((name.clone(), *camera_id));
        }
        by_stem
            .entry(image_stem(name).to_owned())
            .or_default()
            .push((name.clone(), *camera_id));
    }
    let mut native_cameras = Vec::with_capacity(image_names.len());
    for name in image_names {
        let mut resolved = None;
        for candidate in calibration_name_candidates(name) {
            if let Some(&camera_id) = assignments.get(&candidate) {
                resolved = Some((candidate, camera_id));
                break;
            }
            if let Some(entries) = by_basename.get(&candidate) {
                if entries.len() > 1 {
                    return Err(format!(
                        "image {name:?} matches ambiguous calibration basenames {:?}",
                        entries.iter().map(|(entry, _)| entry).collect::<Vec<_>>()
                    ));
                }
                if let Some((entry, camera_id)) = entries.first() {
                    resolved = Some((entry.clone(), *camera_id));
                    break;
                }
            }
            if let Some(entries) = by_stem.get(image_stem(&candidate)) {
                if entries.len() > 1 {
                    return Err(format!(
                        "image {name:?} matches ambiguous calibration stems {:?}",
                        entries.iter().map(|(entry, _)| entry).collect::<Vec<_>>()
                    ));
                }
                if let Some((entry, camera_id)) = entries.first() {
                    resolved = Some((entry.clone(), *camera_id));
                    break;
                }
            }
        }
        let Some((resolved_name, camera_id)) = resolved else {
            return Err(format!(
                "calibration images.txt has no camera assignment for loaded image {name:?}"
            ));
        };
        let camera = cameras_by_id.get(&camera_id).ok_or_else(|| {
            format!("calibration image {resolved_name:?} refers to missing CAMERA_ID {camera_id}")
        })?;
        if camera.model != CameraModel::Pinhole {
            return Err(format!(
                "calibration CAMERA_ID {camera_id} uses {:?}; only PINHOLE is supported",
                camera.model
            ));
        }
        if camera.params.len() != 4 {
            return Err(format!(
                "calibration CAMERA_ID {camera_id} has {} parameters; PINHOLE requires 4",
                camera.params.len()
            ));
        }
        native_cameras.push(camera.clone());
    }
    let rig = PerImageCameras::new(native_cameras.clone()).map_err(|error| error.to_string())?;
    Ok(LoadedPerImageCalibration {
        rig,
        native_cameras,
    })
}

/// Resolve and validate a per-image calibration for an already-loaded feature
/// set.  The feature-file path keeps this wrapper so it can retain the exact
/// historical validation order; streaming extraction uses the resolver above
/// and validates each decoded image before writing its result.
pub(super) fn load_input_colmap_calibration(
    model_dir: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    image_dir: Option<&Path>,
) -> Result<LoadedPerImageCalibration, String> {
    let loaded = resolve_input_colmap_calibration(model_dir, image_names)?;
    loaded
        .rig
        .validate_features(features)
        .map_err(|error| format!("calibration feature validation failed: {error}"))?;
    validate_calibration_image_dimensions(&loaded.rig, image_names, image_dir)?;
    Ok(loaded)
}

#[cfg(feature = "image-io")]
pub(super) fn validate_calibration_image_dimensions(
    rig: &PerImageCameras,
    image_names: &[String],
    image_dir: Option<&Path>,
) -> Result<(), String> {
    let Some(image_dir) = image_dir else {
        return Ok(());
    };
    let mut dimensions = Vec::with_capacity(image_names.len());
    for name in image_names {
        let direct = image_dir.join(name);
        let path = if direct.is_file() {
            direct
        } else {
            let stem = image_stem(name);
            let mut matches = std::fs::read_dir(image_dir)
                .map_err(|error| format!("cannot scan image directory {image_dir:?}: {error}"))?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter(|path| {
                    path.file_stem()
                        .and_then(|value| value.to_str())
                        .is_some_and(|candidate| candidate == stem)
                });
            let Some(path) = matches.next() else {
                return Err(format!(
                    "cannot find source image {name:?} under {image_dir:?} for calibration dimension validation"
                ));
            };
            if matches.next().is_some() {
                return Err(format!(
                    "multiple source images match {name:?} under {image_dir:?}"
                ));
            }
            path
        };
        let image = visloc_io::images::read_common_image(&path)
            .map_err(|error| format!("cannot decode source image {path:?}: {error}"))?;
        dimensions.push((image.width() as u32, image.height() as u32));
    }
    rig.validate_image_dimensions(&dimensions)
        .map_err(|error| error.to_string())
}

#[cfg(not(feature = "image-io"))]
pub(super) fn validate_calibration_image_dimensions(
    _rig: &PerImageCameras,
    _image_names: &[String],
    image_dir: Option<&Path>,
) -> Result<(), String> {
    if image_dir.is_some() {
        return Err(
            "--input-colmap-calibration image-dimension validation requires the image-io feature"
                .into(),
        );
    }
    Ok(())
}
