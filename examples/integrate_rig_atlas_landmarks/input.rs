//! Readers for the node list, the rig manifest, the atlas images and the
//! source windows.

use super::*;

pub(super) fn parse_node_specs(path: &Path) -> Result<Vec<NodeSpec>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read node manifest {}: {error}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut nodes = Vec::new();
    let mut ids = BTreeSet::new();
    let mut starts_paths = BTreeSet::new();
    for (line_index, raw) in contents.lines().enumerate() {
        let line = line_index + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(format!(
                "nodes {}:{} requires node_id<TAB>window_start<TAB>images.txt",
                path.display(),
                line
            ));
        }
        let node_id = parse_u64(fields[0], "node id", path, line)?;
        let window_start = parse_u64(fields[1], "window start", path, line)?;
        let images_txt = resolve_path(base, fields[2]);
        if !ids.insert(node_id) {
            return Err(format!(
                "nodes {}:{} duplicates node id {node_id}",
                path.display(),
                line
            ));
        }
        if !starts_paths.insert((window_start, images_txt.clone())) {
            return Err(format!(
                "nodes {}:{} duplicates ({window_start}, {})",
                path.display(),
                line,
                images_txt.display()
            ));
        }
        if !images_txt.is_file() {
            return Err(format!(
                "nodes {}:{} source images.txt does not exist: {}",
                path.display(),
                line,
                images_txt.display()
            ));
        }
        nodes.push(NodeSpec {
            node_id,
            window_start,
            images_txt,
        });
    }
    if nodes.is_empty() {
        return Err(format!("nodes {} contains no rows", path.display()));
    }
    nodes.sort_by_key(|node| (node.window_start, node.node_id));
    Ok(nodes)
}

pub(super) fn parse_rig_manifest(path: &Path) -> Result<RigManifest, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read rig manifest {}: {error}", path.display()))?;
    let mut sensors = BTreeMap::new();
    let mut frame_rows = Vec::new();
    for (line_index, raw) in contents.lines().enumerate() {
        let line = line_index + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split_whitespace().collect::<Vec<_>>();
        match fields.first().copied() {
            Some("S") => {
                if fields.len() != 16 {
                    return Err(format!(
                        "rig manifest {}:{} sensor row requires 16 fields",
                        path.display(),
                        line
                    ));
                }
                let index = parse_usize(fields[1], "sensor index", path, line)?;
                let camera_id = parse_u64(fields[2], "camera id", path, line)?;
                let width = parse_u32(fields[3], "sensor width", path, line)?;
                let height = parse_u32(fields[4], "sensor height", path, line)?;
                if width == 0 || height == 0 {
                    return Err(format!(
                        "rig manifest {}:{} has zero sensor size",
                        path.display(),
                        line
                    ));
                }
                let values = (5..16)
                    .map(|index| parse_f64(fields[index], "sensor value", path, line))
                    .collect::<Result<Vec<_>, _>>()?;
                let quaternion = Quaternion::new(values[4], values[5], values[6], values[7]);
                let norm = quaternion.norm();
                if !norm.is_finite() || norm <= MIN_HOMOGENEOUS_SCALE {
                    return Err(format!(
                        "rig manifest {}:{} has invalid sensor quaternion",
                        path.display(),
                        line
                    ));
                }
                if sensors
                    .insert(
                        index,
                        SensorCalibration {
                            camera_id,
                            width,
                            height,
                            fx: values[0],
                            fy: values[1],
                            cx: values[2],
                            cy: values[3],
                            sensor_from_rig: SE3::new(
                                UnitQuaternion::new_normalize(quaternion),
                                Vector3::new(values[8], values[9], values[10]),
                            ),
                        },
                    )
                    .is_some()
                {
                    return Err(format!(
                        "rig manifest {}:{} duplicates sensor {index}",
                        path.display(),
                        line
                    ));
                }
            }
            Some("F") => {
                if fields.len() != 4 {
                    return Err(format!(
                        "rig manifest {}:{} frame row requires 4 fields",
                        path.display(),
                        line
                    ));
                }
                let frame_id = parse_u64(fields[1], "frame id", path, line)?;
                let name = fields[2].to_owned();
                if name.is_empty() {
                    return Err(format!(
                        "rig manifest {}:{} has empty image name",
                        path.display(),
                        line
                    ));
                }
                let sensor_index = parse_usize(fields[3], "sensor index", path, line)?;
                frame_rows.push((frame_id, name, sensor_index, line));
            }
            Some(kind) => {
                return Err(format!(
                    "rig manifest {}:{} unknown row kind {kind:?}",
                    path.display(),
                    line
                ));
            }
            None => unreachable!(),
        }
    }
    if sensors.is_empty() || frame_rows.is_empty() {
        return Err(format!(
            "rig manifest {} has no sensors or frame rows",
            path.display()
        ));
    }
    for (expected, actual) in sensors.keys().copied().enumerate() {
        if expected != actual {
            return Err(format!(
                "rig manifest {} sensor indices must be contiguous from zero",
                path.display()
            ));
        }
    }
    let mut assignments = BTreeMap::new();
    let mut frame_sensors = BTreeSet::new();
    for (frame_id, name, sensor_index, line) in frame_rows {
        if !sensors.contains_key(&sensor_index) {
            return Err(format!(
                "rig manifest {}:{} references unknown sensor {sensor_index}",
                path.display(),
                line
            ));
        }
        if !frame_sensors.insert((frame_id, sensor_index)) {
            return Err(format!(
                "rig manifest {}:{} duplicates frame/sensor ({frame_id}, {sensor_index})",
                path.display(),
                line
            ));
        }
        let global_image_id = assignments.len() as u64 + 1;
        if assignments
            .insert(
                name,
                ImageAssignment {
                    frame_id,
                    sensor_index,
                    global_image_id,
                },
            )
            .is_some()
        {
            return Err(format!(
                "rig manifest {}:{} duplicates image name",
                path.display(),
                line
            ));
        }
    }
    Ok(RigManifest {
        sensors,
        assignments,
    })
}

pub(super) fn parse_atlas_images(
    atlas_dir: &Path,
    manifest: &RigManifest,
) -> Result<BTreeMap<String, AtlasImage>, String> {
    let mut paths = Vec::new();
    let direct = atlas_dir.join("images.txt");
    if direct.is_file() {
        paths.push(direct);
    } else {
        let entries = fs::read_dir(atlas_dir)
            .map_err(|error| format!("read atlas directory {}: {error}", atlas_dir.display()))?;
        let mut component_paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| format!("read atlas entry: {error}"))?;
            let path = entry.path();
            if entry
                .file_type()
                .map_err(|error| format!("stat atlas entry: {error}"))?
                .is_dir()
                && entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("component-")
            {
                let image_path = path.join("images.txt");
                if image_path.is_file() {
                    component_paths.push(image_path);
                }
            }
        }
        component_paths.sort();
        match component_paths.as_slice() {
            [] => {}
            [only] => paths.push(only.clone()),
            _ => {
                return Err(format!(
                    "atlas {} contains {} independent component images.txt files; pass one component directory per invocation",
                    atlas_dir.display(),
                    component_paths.len()
                ));
            }
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "atlas {} contains no images.txt",
            atlas_dir.display()
        ));
    }
    let mut images = BTreeMap::new();
    for path in paths {
        for image in parse_atlas_images_file(&path)? {
            let assignment = manifest.assignments.get(&image.name).ok_or_else(|| {
                format!("atlas image {:?} is absent from rig manifest", image.name)
            })?;
            let sensor = manifest
                .sensors
                .get(&assignment.sensor_index)
                .expect("manifest sensor index was validated");
            if image.camera_id != sensor.camera_id {
                return Err(format!(
                    "atlas image {:?} camera {} disagrees with manifest camera {}",
                    image.name, image.camera_id, sensor.camera_id
                ));
            }
            let bound = AtlasImage {
                global_image_id: assignment.global_image_id,
                frame_id: assignment.frame_id,
                sensor_index: assignment.sensor_index,
                name: image.name.clone(),
                camera_id: image.camera_id,
                pose: image.pose,
            };
            if images.insert(image.name.clone(), bound).is_some() {
                return Err(format!(
                    "atlas contains duplicate image name {:?}",
                    image.name
                ));
            }
        }
    }
    Ok(images)
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedPoseImage {
    camera_id: u64,
    name: String,
    pose: Pose,
}

fn parse_atlas_images_file(path: &Path) -> Result<Vec<ParsedPoseImage>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read atlas images {}: {error}", path.display()))?;
    let lines = contents.lines().collect::<Vec<_>>();
    let mut parsed = Vec::new();
    let mut index = 0;
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    while let Some((line_number, header)) = next_data_line(&lines, &mut index) {
        let fields = header.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 10 {
            return Err(format!(
                "atlas images {}:{} pose row requires at least 10 fields",
                path.display(),
                line_number
            ));
        }
        let local_id = parse_u64(fields[0], "image id", path, line_number)?;
        let quaternion = parse_quaternion(&fields[1..5], path, line_number)?;
        let translation = parse_vector(&fields[5..8], path, line_number)?;
        let camera_id = parse_u64(fields[8], "camera id", path, line_number)?;
        let name = fields[9].to_owned();
        if !ids.insert(local_id) || !names.insert(name.clone()) {
            return Err(format!(
                "atlas images {}:{} duplicates image id or name",
                path.display(),
                line_number
            ));
        }
        let points_line = next_required_line(&lines, &mut index).ok_or_else(|| {
            format!(
                "atlas images {}:{} missing POINTS2D row",
                path.display(),
                line_number
            )
        })?;
        // The atlas produced by the trajectory stitcher intentionally has no
        // POINTS2D.  Still parse non-empty rows so malformed input fails closed.
        parse_points2d(points_line.1, path, points_line.0)?;
        parsed.push(ParsedPoseImage {
            camera_id,
            name,
            pose: Pose::from_world_to_camera(quaternion, translation),
        });
    }
    if parsed.is_empty() {
        return Err(format!(
            "atlas images {} contains no images",
            path.display()
        ));
    }
    Ok(parsed)
}

pub(super) fn read_source_window(
    node: &NodeSpec,
    manifest: &RigManifest,
    cameras: &mut BTreeMap<u64, Camera>,
) -> Result<SourceWindow, String> {
    let model_dir = node
        .images_txt
        .parent()
        .ok_or_else(|| format!("node {} images path has no parent", node.node_id))?;
    let cameras_path = model_dir.join("cameras.txt");
    let points_path = model_dir.join("points3D.txt");
    let camera_text = fs::read_to_string(&cameras_path).map_err(|error| {
        format!(
            "node {} read {}: {error}",
            node.node_id,
            cameras_path.display()
        )
    })?;
    let source_cameras = parse_cameras_txt(&camera_text)
        .map_err(|error| format!("node {} parse cameras: {error}", node.node_id))?;
    let mut local_cameras = BTreeMap::new();
    for camera in source_cameras {
        if camera.width == 0
            || camera.height == 0
            || camera.params.iter().any(|value| !value.is_finite())
        {
            return Err(format!(
                "node {} has invalid camera {}",
                node.node_id, camera.id
            ));
        }
        if local_cameras.insert(camera.id, camera.clone()).is_some() {
            return Err(format!(
                "node {} duplicates camera {}",
                node.node_id, camera.id
            ));
        }
        if let Some(previous) = cameras.insert(camera.id, camera.clone()) {
            if previous != camera {
                return Err(format!(
                    "camera {} has incompatible definitions across source windows",
                    camera.id
                ));
            }
        }
    }
    let images = parse_source_images(&node.images_txt)?;
    for image in &images {
        let assignment = manifest.assignments.get(&image.name).ok_or_else(|| {
            format!(
                "node {} image {:?} is absent from rig manifest",
                node.node_id, image.name
            )
        })?;
        let sensor = manifest
            .sensors
            .get(&assignment.sensor_index)
            .expect("manifest sensor index was validated");
        let camera = local_cameras.get(&image.camera_id).ok_or_else(|| {
            format!(
                "node {} image {:?} references missing camera {}",
                node.node_id, image.name, image.camera_id
            )
        })?;
        if image.camera_id != sensor.camera_id
            || camera.width != sensor.width
            || camera.height != sensor.height
            || !camera_matches_manifest(camera, sensor)
        {
            return Err(format!(
                "node {} image {:?} calibration disagrees with manifest sensor {}",
                node.node_id, image.name, assignment.sensor_index
            ));
        }
    }
    let points_text = fs::read_to_string(&points_path).map_err(|error| {
        format!(
            "node {} read {}: {error}",
            node.node_id,
            points_path.display()
        )
    })?;
    let points = parse_source_points(&points_text, &node.images_txt)?;
    validate_source_bidirectional(&images, &points, &node.images_txt)?;
    Ok(SourceWindow { images, points })
}

fn parse_source_images(path: &Path) -> Result<Vec<SourceImage>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read source images {}: {error}", path.display()))?;
    let lines = contents.lines().collect::<Vec<_>>();
    let mut index = 0;
    let mut images = Vec::new();
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    while let Some((line_number, header)) = next_data_line(&lines, &mut index) {
        let fields = header.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 10 {
            return Err(format!(
                "source images {}:{} pose row requires at least 10 fields",
                path.display(),
                line_number
            ));
        }
        let local_id = parse_u64(fields[0], "image id", path, line_number)?;
        let _quaternion = parse_quaternion(&fields[1..5], path, line_number)?;
        let _translation = parse_vector(&fields[5..8], path, line_number)?;
        let camera_id = parse_u64(fields[8], "camera id", path, line_number)?;
        let name = fields[9].to_owned();
        if !ids.insert(local_id) {
            return Err(format!(
                "source images {}:{} duplicate image id",
                path.display(),
                line_number
            ));
        }
        if !names.insert(name.clone()) {
            return Err(format!(
                "source images {}:{} duplicate image name",
                path.display(),
                line_number
            ));
        }
        let points_line = next_required_line(&lines, &mut index).ok_or_else(|| {
            format!(
                "source images {}:{} missing POINTS2D row",
                path.display(),
                line_number
            )
        })?;
        let points = parse_points2d(points_line.1, path, points_line.0)?;
        images.push(SourceImage {
            local_id,
            camera_id,
            name,
            keypoints: points,
        });
    }
    if images.is_empty() {
        return Err(format!(
            "source images {} contains no images",
            path.display()
        ));
    }
    Ok(images)
}

fn parse_source_points(contents: &str, images_path: &Path) -> Result<Vec<SourcePoint>, String> {
    let mut points = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, raw) in contents.lines().enumerate() {
        let line_number = line_index + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 8 || (fields.len() - 8) % 2 != 0 {
            return Err(format!(
                "source points3D {}:{} has malformed fields",
                images_path.display(),
                line_number
            ));
        }
        let id = parse_u64(fields[0], "POINT3D_ID", images_path, line_number)?;
        if !ids.insert(id) {
            return Err(format!(
                "source points3D {}:{} duplicate point {id}",
                images_path.display(),
                line_number
            ));
        }
        let x = parse_f64(fields[1], "point x", images_path, line_number)?;
        let y = parse_f64(fields[2], "point y", images_path, line_number)?;
        let z = parse_f64(fields[3], "point z", images_path, line_number)?;
        let error = parse_f64(fields[7], "point error", images_path, line_number)?;
        if !error.is_finite() {
            return Err(format!(
                "source points3D {}:{} has non-finite error",
                images_path.display(),
                line_number
            ));
        }
        let mut observations = Vec::new();
        for pair in fields[8..].chunks_exact(2) {
            let image_id = parse_u64(pair[0], "track image id", images_path, line_number)?;
            let keypoint_index =
                parse_usize(pair[1], "track keypoint index", images_path, line_number)?;
            observations.push(LocalObservation {
                image_id,
                keypoint_index,
            });
        }
        observations.sort_unstable();
        if observations.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(format!(
                "source points3D {}:{} repeats a track observation",
                images_path.display(),
                line_number
            ));
        }
        if observations
            .windows(2)
            .any(|pair| pair[0].image_id == pair[1].image_id)
        {
            return Err(format!(
                "source points3D {}:{} uses two keypoints from one image",
                images_path.display(),
                line_number
            ));
        }
        points.push(SourcePoint {
            id,
            position: Point3::new(x, y, z),
            observations,
        });
    }
    points.sort_by_key(|point| point.id);
    Ok(points)
}

fn validate_source_bidirectional(
    images: &[SourceImage],
    points: &[SourcePoint],
    path: &Path,
) -> Result<(), String> {
    let by_id = images
        .iter()
        .map(|image| (image.local_id, image))
        .collect::<BTreeMap<_, _>>();
    let mut references = BTreeMap::<LocalObservation, u64>::new();
    for point in points {
        if !point.position.coords.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "source points3D {} has non-finite point",
                path.display()
            ));
        }
        for observation in &point.observations {
            let image = by_id.get(&observation.image_id).ok_or_else(|| {
                format!(
                    "source points3D {} references unknown image {}",
                    path.display(),
                    observation.image_id
                )
            })?;
            let keypoint = image
                .keypoints
                .get(observation.keypoint_index)
                .ok_or_else(|| {
                    format!(
                        "source points3D {} references image {} keypoint {} out of range",
                        path.display(),
                        observation.image_id,
                        observation.keypoint_index
                    )
                })?;
            if keypoint.point3d_id != Some(point.id) {
                return Err(format!(
                    "source points3D {} has non-bidirectional ref image {} keypoint {}",
                    path.display(),
                    observation.image_id,
                    observation.keypoint_index
                ));
            }
            if references.insert(observation.clone(), point.id).is_some() {
                return Err(format!(
                    "source points3D {} repeats image {} keypoint {}",
                    path.display(),
                    observation.image_id,
                    observation.keypoint_index
                ));
            }
        }
    }
    for image in images {
        for (keypoint_index, keypoint) in image.keypoints.iter().enumerate() {
            if let Some(point_id) = keypoint.point3d_id {
                if references.get(&LocalObservation {
                    image_id: image.local_id,
                    keypoint_index,
                }) != Some(&point_id)
                {
                    return Err(format!(
                        "source images {} has POINTS2D ref without matching track",
                        path.display()
                    ));
                }
            }
        }
    }
    Ok(())
}
