//! Build a physical, landmark-bearing COLMAP model from a stitched rig atlas.
//!
//! This is deliberately an opt-in, bounded first integration step.  Source
//! windows are read one at a time; local COLMAP POINT3D_ID values are never
//! used as global identities.  Observations are keyed by the rig-manifest
//! image name (and its deterministic global image id) plus the original
//! keypoint index, then re-triangulated with the final atlas camera poses.
//!
//! The trajectory-only stitcher remains unchanged.  Pass one atlas component
//! directory (the directory containing one `images.txt`) per invocation.  A
//! root containing multiple component gauges is rejected rather than
//! flattening independent gauges into one model.  This example does not
//! release poses or run bundle adjustment; that is a separate, later bounded
//! experiment.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{DMatrix, Point2, Point3, Quaternion, UnitQuaternion, Vector3};
use visloc_rs::io::colmap::parse_cameras_txt;
use visloc_rs::{Camera, CameraModel, Pose, SE3};

const USAGE: &str = "usage: integrate_rig_atlas_landmarks\n    --rig-manifest PATH --nodes-tsv PATH --atlas-dir PATH --out-dir PATH";
const XY_TOLERANCE_PX: f64 = 1.0e-9;
const INTRINSIC_TOLERANCE: f64 = 1.0e-8;
const MIN_TRACK_OBSERVATIONS: usize = 2;
const MAX_DLT_OBSERVATIONS: usize = 64;
const MAX_MEAN_REPROJECTION_PX: f64 = 2.0;
const MAX_REPROJECTION_PX: f64 = 4.0;
const MIN_HOMOGENEOUS_SCALE: f64 = 1.0e-12;
const MIN_BASELINE_M: f64 = 1.0e-9;
const MIN_RAY_CROSS_NORM: f64 = 1.0e-8;
const MAX_RIG_CENTER_DISAGREEMENT_M: f64 = 1.0e-4;
const MAX_RIG_ROTATION_DISAGREEMENT_DEG: f64 = 1.0e-3;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    rig_manifest: PathBuf,
    nodes_tsv: PathBuf,
    atlas_dir: PathBuf,
    out_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NodeSpec {
    node_id: u64,
    window_start: u64,
    images_txt: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
struct SensorCalibration {
    camera_id: u64,
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    sensor_from_rig: SE3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ImageAssignment {
    frame_id: u64,
    sensor_index: usize,
    global_image_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct RigManifest {
    sensors: BTreeMap<usize, SensorCalibration>,
    assignments: BTreeMap<String, ImageAssignment>,
}

#[derive(Debug, Clone, PartialEq)]
struct SourceKeypoint {
    xy: Point2<f64>,
    point3d_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
struct SourceImage {
    local_id: u64,
    camera_id: u64,
    name: String,
    keypoints: Vec<SourceKeypoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LocalObservation {
    image_id: u64,
    keypoint_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct SourcePoint {
    id: u64,
    position: Point3<f64>,
    observations: Vec<LocalObservation>,
}

#[derive(Debug, Clone, PartialEq)]
struct SourceWindow {
    images: Vec<SourceImage>,
    points: Vec<SourcePoint>,
}

#[derive(Debug, Clone, PartialEq)]
struct AtlasImage {
    global_image_id: u64,
    frame_id: u64,
    sensor_index: usize,
    name: String,
    camera_id: u64,
    pose: Pose,
}

#[derive(Debug, Clone, PartialEq)]
struct GlobalImage {
    atlas: AtlasImage,
    keypoints: Vec<Point2<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ObservationKey {
    global_image_id: u64,
    keypoint_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct ObservationState {
    xy: Point2<f64>,
    owner_track: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct GlobalTrack {
    observations: Vec<ObservationKey>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct IngestStats {
    source_nodes: usize,
    source_points: usize,
    source_observations: usize,
    accepted_candidates: usize,
    extended_candidates: usize,
    duplicate_observations: usize,
    rejected_candidates: usize,
    rejected_missing_atlas_images: usize,
    filtered_out_of_component_observations: usize,
    rejected_short_candidates: usize,
    rejected_xy: usize,
    rejected_track_conflict: usize,
    rejected_same_image: usize,
    rejected_invalid_source: usize,
    rejected_triangulation: usize,
    triangulation_sampled_tracks: usize,
    triangulation_sampled_observations: usize,
    output_landmarks: usize,
    output_observations: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct TrackStore {
    observations: BTreeMap<ObservationKey, ObservationState>,
    tracks: Vec<GlobalTrack>,
}

#[derive(Debug, Clone, PartialEq)]
struct LandmarkOutput {
    track_id: usize,
    position: Point3<f64>,
    observations: Vec<ObservationKey>,
    errors: Vec<f64>,
    rms_error: f64,
    mean_error: f64,
    max_error: f64,
    dlt_sample_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SupportSummary {
    atlas_image_count: usize,
    supported_image_count: usize,
    zero_support_image_count: usize,
    atlas_frame_count: usize,
    supported_frame_count: usize,
    zero_support_frame_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeRejection {
    DuplicateCandidateObservation,
    SameImageDifferentKeypoint,
    ExistingTrackSameImageConflict,
}

impl std::fmt::Display for MergeRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateCandidateObservation => write!(f, "duplicate candidate observation"),
            Self::SameImageDifferentKeypoint => write!(f, "same image has different keypoints"),
            Self::ExistingTrackSameImageConflict => {
                write!(f, "existing track already owns another keypoint in image")
            }
        }
    }
}

fn main() {
    let args = parse_args(std::env::args()).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    if let Err(error) = run(&args) {
        eprintln!("integrate_rig_atlas_landmarks: {error}");
        std::process::exit(1);
    }
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = args.into_iter();
    let _program = values.next();
    let mut rig_manifest = None;
    let mut nodes_tsv = None;
    let mut atlas_dir = None;
    let mut out_dir = None;
    while let Some(flag) = values.next() {
        if flag == "-h" || flag == "--help" {
            return Err(USAGE.to_owned());
        }
        let slot = match flag.as_str() {
            "--rig-manifest" => &mut rig_manifest,
            "--nodes-tsv" => &mut nodes_tsv,
            "--atlas-dir" => &mut atlas_dir,
            "--out-dir" => &mut out_dir,
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        };
        if slot.is_some() {
            return Err(format!("duplicate argument {flag}\n{USAGE}"));
        }
        let value = values
            .next()
            .ok_or_else(|| format!("{flag} requires PATH\n{USAGE}"))?;
        if value.starts_with('-') {
            return Err(format!("{flag} requires PATH, got {value:?}\n{USAGE}"));
        }
        *slot = Some(PathBuf::from(value));
    }
    Ok(Args {
        rig_manifest: rig_manifest.ok_or_else(|| format!("--rig-manifest is required\n{USAGE}"))?,
        nodes_tsv: nodes_tsv.ok_or_else(|| format!("--nodes-tsv is required\n{USAGE}"))?,
        atlas_dir: atlas_dir.ok_or_else(|| format!("--atlas-dir is required\n{USAGE}"))?,
        out_dir: out_dir.ok_or_else(|| format!("--out-dir is required\n{USAGE}"))?,
    })
}

fn run(args: &Args) -> Result<(), String> {
    let manifest = parse_rig_manifest(&args.rig_manifest)?;
    let nodes = parse_node_specs(&args.nodes_tsv)?;
    let atlas_images = parse_atlas_images(&args.atlas_dir, &manifest)?;
    let mut cameras = BTreeMap::<u64, Camera>::new();
    let mut global_images = atlas_images
        .values()
        .cloned()
        .map(|atlas| {
            (
                atlas.global_image_id,
                GlobalImage {
                    atlas,
                    keypoints: Vec::new(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut image_name_to_id = BTreeMap::new();
    for image in global_images.values() {
        image_name_to_id.insert(image.atlas.name.clone(), image.atlas.global_image_id);
    }

    let mut store = TrackStore::default();
    let mut stats = IngestStats::default();
    for node in &nodes {
        let source = read_source_window(node, &manifest, &mut cameras)?;
        stats.source_nodes += 1;
        stats.source_points += source.points.len();
        stats.source_observations += source
            .points
            .iter()
            .map(|point| point.observations.len())
            .sum::<usize>();
        update_global_images(&mut global_images, &source, &image_name_to_id, &mut stats)?;
        ingest_source_points(
            &source,
            &image_name_to_id,
            &global_images,
            &mut store,
            &mut stats,
        )?;
    }
    validate_global_images(&global_images)?;
    validate_output_cameras(&manifest, &global_images, &cameras)?;

    let mut landmarks = Vec::new();
    for (track_id, track) in store.tracks.iter_mut().enumerate() {
        track.observations.sort_unstable();
        track.observations.dedup();
        if track.observations.len() < MIN_TRACK_OBSERVATIONS {
            stats.rejected_short_candidates += 1;
            continue;
        }
        match triangulate_track(
            track_id,
            track,
            &store.observations,
            &global_images,
            &cameras,
        ) {
            Ok(landmark) => {
                if landmark.dlt_sample_count < landmark.observations.len() {
                    stats.triangulation_sampled_tracks += 1;
                    stats.triangulation_sampled_observations +=
                        landmark.observations.len() - landmark.dlt_sample_count;
                }
                landmarks.push(landmark)
            }
            Err(_) => stats.rejected_triangulation += 1,
        }
    }
    landmarks.sort_by_key(|landmark| {
        (
            landmark
                .observations
                .first()
                .copied()
                .unwrap_or(ObservationKey {
                    global_image_id: u64::MAX,
                    keypoint_index: usize::MAX,
                }),
            landmark.track_id,
        )
    });
    stats.output_landmarks = landmarks.len();
    stats.output_observations = landmarks
        .iter()
        .map(|landmark| landmark.observations.len())
        .sum();
    require_nonempty_landmarks(&landmarks, stats.output_observations)?;
    let support = write_model(&args.out_dir, &global_images, &cameras, &landmarks, &stats)?;

    let errors = landmarks
        .iter()
        .flat_map(|landmark| landmark.errors.iter().copied())
        .collect::<Vec<_>>();
    let mean = if errors.is_empty() {
        0.0
    } else {
        errors.iter().sum::<f64>() / errors.len() as f64
    };
    let rms = if errors.is_empty() {
        0.0
    } else {
        (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt()
    };
    let p95 = percentile(&errors, 0.95).unwrap_or(0.0);
    let max = errors.iter().copied().fold(0.0, f64::max);
    println!(
        "nodes={} atlas_poses={} supported_images={} zero_support_images={} atlas_frames={} supported_frames={} zero_support_frames={} tracks={} landmarks={} observations={} mean_px={mean:.9} rms_px={rms:.9} p95_px={p95:.9} max_px={max:.9} sampled_tracks={} sampled_observations={} rejected_triangulation={} rejected_collision={}",
        stats.source_nodes,
        global_images.len(),
        support.supported_image_count,
        support.zero_support_image_count,
        support.atlas_frame_count,
        support.supported_frame_count,
        support.zero_support_frame_count,
        store.tracks.len(),
        stats.output_landmarks,
        stats.output_observations,
        stats.triangulation_sampled_tracks,
        stats.triangulation_sampled_observations,
        stats.rejected_triangulation,
        stats.rejected_candidates,
    );
    Ok(())
}

fn parse_node_specs(path: &Path) -> Result<Vec<NodeSpec>, String> {
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

fn parse_rig_manifest(path: &Path) -> Result<RigManifest, String> {
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

fn parse_atlas_images(
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

fn read_source_window(
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

fn update_global_images(
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

fn validate_global_images(global_images: &BTreeMap<u64, GlobalImage>) -> Result<(), String> {
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

fn ingest_source_points(
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
enum MergeOutcome {
    New,
    Extended { duplicate_count: usize },
}

fn merge_candidate(
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

fn triangulate_track(
    track_id: usize,
    track: &GlobalTrack,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
) -> Result<LandmarkOutput, String> {
    if track.observations.len() < MIN_TRACK_OBSERVATIONS {
        return Err("track has too few observations".to_owned());
    }
    let sample = bounded_sample(&track.observations, MAX_DLT_OBSERVATIONS);
    if !has_observable_parallax(&sample, observations, images, cameras)? {
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
        let pose_matrix = image.atlas.pose.matrix();
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
        let point_camera = image.atlas.pose.transform_world_point(&position);
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

fn has_observable_parallax(
    keys: &[ObservationKey],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
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
        let bearing_world = image
            .atlas
            .pose
            .camera_to_world()
            .rotation
            .transform_vector(&bearing_camera);
        rays.push((image.atlas.pose.camera_center_world(), bearing_world));
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

fn write_model(
    out_dir: &Path,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    stats: &IngestStats,
) -> Result<SupportSummary, String> {
    fs::create_dir_all(out_dir)
        .map_err(|error| format!("create output {}: {error}", out_dir.display()))?;
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
    for (index, landmark) in landmarks.iter().enumerate() {
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

fn require_nonempty_landmarks(
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

fn validate_output_cameras(
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

fn camera_matches_manifest(camera: &Camera, sensor: &SensorCalibration) -> bool {
    if !matches!(camera.model, CameraModel::Pinhole) || camera.params.len() != 4 {
        return false;
    }
    let expected = [sensor.fx, sensor.fy, sensor.cx, sensor.cy];
    camera.params[..4]
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= INTRINSIC_TOLERANCE)
}

fn bounded_sample(keys: &[ObservationKey], max_count: usize) -> Vec<ObservationKey> {
    if keys.len() <= max_count {
        return keys.to_vec();
    }
    let mut result = Vec::with_capacity(max_count);
    for index in 0..max_count {
        let source_index = index * (keys.len() - 1) / (max_count - 1);
        if result.last().copied() != Some(keys[source_index]) {
            result.push(keys[source_index]);
        }
    }
    result
}

fn points_close(left: &Point2<f64>, right: &Point2<f64>) -> bool {
    (left - right).norm() <= XY_TOLERANCE_PX
        && left.coords.iter().all(|value| value.is_finite())
        && right.coords.iter().all(|value| value.is_finite())
}

fn parse_points2d(
    line: &str,
    path: &Path,
    line_number: usize,
) -> Result<Vec<SourceKeypoint>, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() % 3 != 0 {
        return Err(format!(
            "COLMAP {}:{} POINTS2D is not triples",
            path.display(),
            line_number
        ));
    }
    let mut points = Vec::with_capacity(fields.len() / 3);
    for triple in fields.chunks_exact(3) {
        let x = parse_f64(triple[0], "POINTS2D x", path, line_number)?;
        let y = parse_f64(triple[1], "POINTS2D y", path, line_number)?;
        let point_id = triple[2].parse::<i64>().map_err(|error| {
            format!(
                "COLMAP {}:{} invalid POINT3D_ID: {error}",
                path.display(),
                line_number
            )
        })?;
        if point_id < -1 {
            return Err(format!(
                "COLMAP {}:{} POINT3D_ID is below -1",
                path.display(),
                line_number
            ));
        }
        points.push(SourceKeypoint {
            xy: Point2::new(x, y),
            point3d_id: (point_id >= 0).then_some(point_id as u64),
        });
    }
    Ok(points)
}

fn next_data_line<'a>(lines: &'a [&'a str], index: &mut usize) -> Option<(usize, &'a str)> {
    while *index < lines.len() {
        let line_number = *index + 1;
        let line = lines[*index].trim();
        *index += 1;
        if !line.is_empty() && !line.starts_with('#') {
            return Some((line_number, line));
        }
    }
    None
}

fn next_required_line<'a>(lines: &'a [&'a str], index: &mut usize) -> Option<(usize, &'a str)> {
    if *index >= lines.len() {
        return None;
    }
    let line_number = *index + 1;
    let line = lines[*index].trim();
    *index += 1;
    if line.starts_with('#') {
        None
    } else {
        Some((line_number, line))
    }
}

fn parse_quaternion(
    fields: &[&str],
    path: &Path,
    line: usize,
) -> Result<UnitQuaternion<f64>, String> {
    if fields.len() != 4 {
        return Err(format!(
            "COLMAP {}:{} quaternion requires four values",
            path.display(),
            line
        ));
    }
    let values = fields
        .iter()
        .map(|field| parse_f64(field, "quaternion", path, line))
        .collect::<Result<Vec<_>, _>>()?;
    let quaternion = Quaternion::new(values[0], values[1], values[2], values[3]);
    let norm = quaternion.norm();
    if !norm.is_finite() || norm <= MIN_HOMOGENEOUS_SCALE {
        return Err(format!(
            "COLMAP {}:{} quaternion is invalid",
            path.display(),
            line
        ));
    }
    Ok(UnitQuaternion::new_normalize(quaternion))
}

fn parse_vector(fields: &[&str], path: &Path, line: usize) -> Result<Vector3<f64>, String> {
    if fields.len() != 3 {
        return Err(format!(
            "COLMAP {}:{} translation requires three values",
            path.display(),
            line
        ));
    }
    Ok(Vector3::new(
        parse_f64(fields[0], "translation", path, line)?,
        parse_f64(fields[1], "translation", path, line)?,
        parse_f64(fields[2], "translation", path, line)?,
    ))
}

fn parse_u64(value: &str, label: &str, path: &Path, line: usize) -> Result<u64, String> {
    value.parse().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })
}

fn parse_u32(value: &str, label: &str, path: &Path, line: usize) -> Result<u32, String> {
    value.parse().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })
}

fn parse_usize(value: &str, label: &str, path: &Path, line: usize) -> Result<usize, String> {
    value.parse().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })
}

fn parse_f64(value: &str, label: &str, path: &Path, line: usize) -> Result<f64, String> {
    let parsed = value.parse::<f64>().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })?;
    if !parsed.is_finite() {
        return Err(format!("{}:{} non-finite {label}", path.display(), line));
    }
    Ok(parsed)
}

fn resolve_path(base: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn camera_model_name(model: &CameraModel) -> Result<&'static str, String> {
    match model {
        CameraModel::Pinhole => Ok("PINHOLE"),
        CameraModel::SimplePinhole => Ok("SIMPLE_PINHOLE"),
        CameraModel::SimpleRadial => Ok("SIMPLE_RADIAL"),
        CameraModel::Radial => Ok("RADIAL"),
        CameraModel::OpenCv => Ok("OPENCV"),
        CameraModel::Unknown(name) => Err(format!("unsupported camera model {name:?}")),
    }
}

fn validate_image_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte == 0)
    {
        return Err(format!(
            "image name is not representable in COLMAP text: {name:?}"
        ));
    }
    Ok(())
}

fn format_f64(value: f64) -> String {
    let formatted = format!("{value:.12}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted.get(index.min(sorted.len() - 1)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image(id: u64, name: &str, keypoints: Vec<Point2<f64>>) -> GlobalImage {
        GlobalImage {
            atlas: AtlasImage {
                global_image_id: id,
                frame_id: id,
                sensor_index: 0,
                name: name.to_owned(),
                camera_id: 1,
                pose: Pose::identity(),
            },
            keypoints,
        }
    }

    fn test_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 100.0, 100.0, 320.0, 240.0)
    }

    #[test]
    fn args_require_all_paths_and_reject_unknown() {
        let error = parse_args(["example".to_owned(), "--rig-manifest".to_owned()]).unwrap_err();
        assert!(error.contains("requires PATH"));
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--bad".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("unknown argument"));
    }

    #[test]
    fn merge_ignores_local_point_ids_and_remaps_by_global_image_key() {
        let mut images = BTreeMap::new();
        images.insert(1, test_image(1, "a", vec![Point2::new(1.0, 2.0)]));
        images.insert(2, test_image(2, "b", vec![Point2::new(3.0, 4.0)]));
        let mut store = TrackStore::default();
        let first = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ];
        assert_eq!(
            merge_candidate(&mut store, &first, &images),
            Ok(MergeOutcome::New)
        );
        let second = first.clone();
        assert_eq!(
            merge_candidate(&mut store, &second, &images),
            Ok(MergeOutcome::Extended { duplicate_count: 2 })
        );
        assert_eq!(store.tracks.len(), 1);
        assert_eq!(store.observations.len(), 2);
    }

    #[test]
    fn repeated_image_with_changed_xy_is_rejected_before_track_merge() {
        let mut images = BTreeMap::new();
        images.insert(1, test_image(1, "shared", vec![Point2::new(10.0, 20.0)]));
        let mut names = BTreeMap::new();
        names.insert("shared".to_owned(), 1);
        let first = SourceWindow {
            images: vec![SourceImage {
                local_id: 7,
                camera_id: 1,
                name: "shared".to_owned(),
                keypoints: vec![SourceKeypoint {
                    xy: Point2::new(10.0, 20.0),
                    point3d_id: None,
                }],
            }],
            points: Vec::new(),
        };
        let second = SourceWindow {
            images: vec![SourceImage {
                local_id: 99,
                camera_id: 1,
                name: "shared".to_owned(),
                keypoints: vec![SourceKeypoint {
                    xy: Point2::new(10.25, 20.0),
                    point3d_id: None,
                }],
            }],
            points: Vec::new(),
        };
        let mut stats = IngestStats::default();
        update_global_images(&mut images, &first, &names, &mut stats).unwrap();
        let error = update_global_images(&mut images, &second, &names, &mut stats).unwrap_err();
        assert!(error.contains("inconsistent ordered keypoint coordinates"));
    }

    #[test]
    fn same_image_different_keypoint_is_rejected_transactionally() {
        let mut images = BTreeMap::new();
        images.insert(
            1,
            test_image(1, "a", vec![Point2::new(1.0, 2.0), Point2::new(3.0, 4.0)]),
        );
        let mut store = TrackStore::default();
        let candidate = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 1,
            },
        ];
        assert_eq!(
            merge_candidate(&mut store, &candidate, &images),
            Err(MergeRejection::SameImageDifferentKeypoint)
        );
        assert!(store.tracks.is_empty());
        assert!(store.observations.is_empty());
    }

    #[test]
    fn conflict_free_crossing_two_existing_tracks_is_unioned_transactionally() {
        let mut images = BTreeMap::new();
        for id in 1..=4 {
            images.insert(
                id,
                test_image(id, &id.to_string(), vec![Point2::new(id as f64, 0.0)]),
            );
        }
        let mut store = TrackStore::default();
        let a = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ];
        let b = vec![
            ObservationKey {
                global_image_id: 3,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 4,
                keypoint_index: 0,
            },
        ];
        assert_eq!(
            merge_candidate(&mut store, &a, &images),
            Ok(MergeOutcome::New)
        );
        assert_eq!(
            merge_candidate(&mut store, &b, &images),
            Ok(MergeOutcome::New)
        );
        let crossing = vec![a[0], b[0]];
        assert_eq!(
            merge_candidate(&mut store, &crossing, &images),
            Ok(MergeOutcome::Extended { duplicate_count: 2 })
        );
        assert_eq!(store.tracks.len(), 2);
        assert_eq!(store.observations.len(), 4);
        assert!(store.tracks[1].observations.is_empty());
        assert_eq!(store.tracks[0].observations.len(), 4);
    }

    #[test]
    fn conflicting_existing_tracks_are_rejected_without_partial_union() {
        let mut images = BTreeMap::new();
        for id in 1..=4 {
            images.insert(
                id,
                test_image(
                    id,
                    &id.to_string(),
                    vec![
                        Point2::new(id as f64, 0.0),
                        Point2::new(id as f64 + 0.5, 0.0),
                    ],
                ),
            );
        }
        let mut store = TrackStore::default();
        let a = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ];
        let b = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 1,
            },
            ObservationKey {
                global_image_id: 3,
                keypoint_index: 0,
            },
        ];
        assert_eq!(
            merge_candidate(&mut store, &a, &images),
            Ok(MergeOutcome::New)
        );
        assert_eq!(
            merge_candidate(&mut store, &b, &images),
            Ok(MergeOutcome::New)
        );
        let bridge = vec![a[1], b[1]];
        assert_eq!(
            merge_candidate(&mut store, &bridge, &images),
            Err(MergeRejection::ExistingTrackSameImageConflict)
        );
        assert_eq!(store.tracks.len(), 2);
        assert_eq!(store.tracks[0].observations.len(), 2);
        assert_eq!(store.tracks[1].observations.len(), 2);
    }

    #[test]
    fn dlt_retriangulates_final_pose_and_reports_actual_error() {
        let camera = test_camera();
        let point = Point3::new(0.2, -0.1, 4.0);
        let pose_a = Pose::identity();
        let pose_b =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-1.0, 0.0, 0.0));
        let mut images = BTreeMap::new();
        images.insert(
            1,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 1,
                    frame_id: 1,
                    sensor_index: 0,
                    name: "a".to_owned(),
                    camera_id: 1,
                    pose: pose_a.clone(),
                },
                keypoints: vec![],
            },
        );
        images.insert(
            2,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 2,
                    frame_id: 2,
                    sensor_index: 0,
                    name: "b".to_owned(),
                    camera_id: 1,
                    pose: pose_b.clone(),
                },
                keypoints: vec![],
            },
        );
        let pose_c =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, -0.75, 0.0));
        images.insert(
            3,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 3,
                    frame_id: 3,
                    sensor_index: 0,
                    name: "c".to_owned(),
                    camera_id: 1,
                    pose: pose_c.clone(),
                },
                keypoints: vec![],
            },
        );
        let xy_a = camera
            .project(&pose_a.transform_world_point(&point))
            .unwrap();
        let xy_b = camera
            .project(&pose_b.transform_world_point(&point))
            .unwrap();
        let xy_c = camera
            .project(&pose_c.transform_world_point(&point))
            .unwrap();
        let keys = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 3,
                keypoint_index: 0,
            },
        ];
        let mut observations = BTreeMap::new();
        observations.insert(
            keys[0],
            ObservationState {
                xy: xy_a,
                owner_track: 0,
            },
        );
        observations.insert(
            keys[1],
            ObservationState {
                xy: Point2::new(xy_b.x + 0.25, xy_b.y),
                owner_track: 0,
            },
        );
        observations.insert(
            keys[2],
            ObservationState {
                xy: xy_c,
                owner_track: 0,
            },
        );
        let mut cameras = BTreeMap::new();
        cameras.insert(1, camera);
        let track = GlobalTrack { observations: keys };
        let landmark = triangulate_track(0, &track, &observations, &images, &cameras).unwrap();
        assert!(landmark.rms_error > 0.0);
        assert!(landmark.max_error < MAX_REPROJECTION_PX);
    }

    #[test]
    fn coincident_camera_centres_are_rejected_even_with_finite_zero_error() {
        let camera = test_camera();
        let point = Point3::new(0.0, 0.0, 4.0);
        let pose = Pose::identity();
        let mut images = BTreeMap::new();
        for id in 1..=2 {
            images.insert(
                id,
                GlobalImage {
                    atlas: AtlasImage {
                        global_image_id: id,
                        frame_id: id,
                        sensor_index: 0,
                        name: id.to_string(),
                        camera_id: 1,
                        pose: pose.clone(),
                    },
                    keypoints: vec![],
                },
            );
        }
        let xy = camera.project(&pose.transform_world_point(&point)).unwrap();
        let keys = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ];
        let observations = keys
            .iter()
            .map(|key| (*key, ObservationState { xy, owner_track: 0 }))
            .collect::<BTreeMap<_, _>>();
        let mut cameras = BTreeMap::new();
        cameras.insert(1, camera);
        let error = triangulate_track(
            0,
            &GlobalTrack { observations: keys },
            &observations,
            &images,
            &cameras,
        )
        .unwrap_err();
        assert!(error.contains("no observable camera baseline"));
    }

    #[test]
    fn mismatched_sensor_poses_for_one_rig_frame_are_rejected() {
        let sensor = |camera_id| SensorCalibration {
            camera_id,
            width: 640,
            height: 480,
            fx: 100.0,
            fy: 100.0,
            cx: 320.0,
            cy: 240.0,
            sensor_from_rig: SE3::identity(),
        };
        let mut sensors = BTreeMap::new();
        sensors.insert(0, sensor(1));
        sensors.insert(1, sensor(1));
        let mut assignments = BTreeMap::new();
        assignments.insert(
            "a".to_owned(),
            ImageAssignment {
                frame_id: 7,
                sensor_index: 0,
                global_image_id: 1,
            },
        );
        assignments.insert(
            "b".to_owned(),
            ImageAssignment {
                frame_id: 7,
                sensor_index: 1,
                global_image_id: 2,
            },
        );
        let manifest = RigManifest {
            sensors,
            assignments,
        };
        let mut images = BTreeMap::new();
        let mut first = test_image(1, "a", vec![Point2::new(320.0, 240.0)]);
        first.atlas.frame_id = 7;
        images.insert(1, first);
        images.insert(
            2,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 2,
                    frame_id: 7,
                    sensor_index: 1,
                    name: "b".to_owned(),
                    camera_id: 1,
                    pose: Pose::from_world_to_camera(
                        UnitQuaternion::identity(),
                        Vector3::new(0.01, 0.0, 0.0),
                    ),
                },
                keypoints: vec![Point2::new(320.0, 240.0)],
            },
        );
        let mut cameras = BTreeMap::new();
        cameras.insert(1, test_camera());
        let error = validate_output_cameras(&manifest, &images, &cameras).unwrap_err();
        assert!(error.contains("sensor poses disagree"));
    }

    #[test]
    fn bounded_sample_is_sorted_endpoint_inclusive_and_deterministic() {
        let keys = (0..100)
            .map(|index| ObservationKey {
                global_image_id: index,
                keypoint_index: 0,
            })
            .collect::<Vec<_>>();
        let sample = bounded_sample(&keys, 7);
        assert_eq!(sample.len(), 7);
        assert_eq!(sample.first(), keys.first());
        assert_eq!(sample.last(), keys.last());
        assert_eq!(sample, bounded_sample(&keys, 7));
    }

    #[test]
    fn multiple_atlas_components_are_rejected_as_independent_gauges() {
        let root = std::env::temp_dir().join(format!(
            "visloc_integrate_landmark_components_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("component-000")).unwrap();
        fs::create_dir_all(root.join("component-001")).unwrap();
        fs::write(root.join("component-000/images.txt"), "").unwrap();
        fs::write(root.join("component-001/images.txt"), "").unwrap();
        let manifest = RigManifest {
            sensors: BTreeMap::new(),
            assignments: BTreeMap::new(),
        };
        let error = parse_atlas_images(&root, &manifest).unwrap_err();
        assert!(error.contains("independent component images.txt"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn empty_landmark_set_cannot_be_published_as_pose_only_model() {
        let error = require_nonempty_landmarks(&[], 0).unwrap_err();
        assert!(error.contains("refusing to publish a pose-only model"));
    }

    #[test]
    fn actual_error_and_bidirectional_track_are_written() {
        let root = std::env::temp_dir().join(format!(
            "visloc_integrate_landmark_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let camera = test_camera();
        let pose = Pose::identity();
        let xy = camera.project(&Point3::new(0.0, 0.0, 4.0)).unwrap();
        let mut images = BTreeMap::new();
        images.insert(
            1,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 1,
                    frame_id: 1,
                    sensor_index: 0,
                    name: "a.png".to_owned(),
                    camera_id: 1,
                    pose,
                },
                keypoints: vec![xy],
            },
        );
        images.insert(
            2,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 2,
                    frame_id: 2,
                    sensor_index: 0,
                    name: "b.png".to_owned(),
                    camera_id: 1,
                    pose: Pose::from_world_to_camera(
                        UnitQuaternion::identity(),
                        Vector3::new(-0.5, 0.0, 0.0),
                    ),
                },
                keypoints: vec![Point2::new(320.0, 240.0)],
            },
        );
        let mut cameras = BTreeMap::new();
        cameras.insert(1, camera);
        let landmark = LandmarkOutput {
            track_id: 0,
            position: Point3::new(0.0, 0.0, 4.0),
            observations: vec![
                ObservationKey {
                    global_image_id: 1,
                    keypoint_index: 0,
                },
                ObservationKey {
                    global_image_id: 2,
                    keypoint_index: 0,
                },
            ],
            errors: vec![0.1, 0.3],
            rms_error: (0.05_f64).sqrt(),
            mean_error: 0.2,
            max_error: 0.3,
            dlt_sample_count: 2,
        };
        let mut stats = IngestStats::default();
        stats.output_landmarks = 1;
        stats.output_observations = 2;
        write_model(&root, &images, &cameras, &[landmark], &stats).unwrap();
        let points = fs::read_to_string(root.join("points3D.txt")).unwrap();
        let image = fs::read_to_string(root.join("images.txt")).unwrap();
        assert!(points.contains(" 0.2 1 0 2 0"));
        assert!(image.lines().any(|line| line.contains("320 240 1")));
        let _ = fs::remove_dir_all(root);
    }
}
