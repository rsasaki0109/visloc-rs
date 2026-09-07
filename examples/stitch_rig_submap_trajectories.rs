//! Build a bounded, deterministic rig-submap trajectory atlas from COLMAP
//! text models. Arms F through L are diagnostic-only extensions of arm E
//! for fixed-rotation fallback/primary experiments.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use visloc_rs::slam::{
    estimate_rig_submap_sim3_constraint, LinearSolver, PoseGraph, PoseGraphEdge, PoseGraphEdgeKind,
    PoseGraphSe3Config, RigSubmapAlignmentConfig, RigSubmapBoundarySampling, RobustKernel,
    SubmapSim3AlignmentConfig,
};
use visloc_rs::{Pose, Sim3, SE3};

const USAGE: &str = "usage: stitch_rig_submap_trajectories\n    --rig-manifest PATH --nodes-tsv PATH --out-dir PATH\n    [--seam-config-arm A|B|C|D|E|F|G|H|I|J|K|L]\n    [--frame-owner-policy newest|interior]\n    [--forest-policy traversal|quality]\n    [--metric-se3]";
const MAX_RIG_CENTER_DISAGREEMENT_M: f64 = 1.0e-4;
const MAX_RIG_ROTATION_DISAGREEMENT_DEG: f64 = 1.0e-3;
const MAX_SEAM_START_GAP: u64 = 750;
const HEALTHY_STEP_RATIO: f64 = 35.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeamConfigArm {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
}

impl SeamConfigArm {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "A" => Ok(Self::A),
            "B" => Ok(Self::B),
            "C" => Ok(Self::C),
            "D" => Ok(Self::D),
            "E" => Ok(Self::E),
            "F" => Ok(Self::F),
            "G" => Ok(Self::G),
            "H" => Ok(Self::H),
            "I" => Ok(Self::I),
            "J" => Ok(Self::J),
            "K" => Ok(Self::K),
            "L" => Ok(Self::L),
            other => Err(format!(
                "--seam-config-arm expects A, B, C, D, E, F, G, H, I, J, K, or L; got {other:?}"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
            Self::E => "E",
            Self::F => "F",
            Self::G => "G",
            Self::H => "H",
            Self::I => "I",
            Self::J => "J",
            Self::K => "K",
            Self::L => "L",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForestPolicy {
    /// Use every accepted edge and the historical deterministic BFS gauge.
    Traversal,
    /// Use the deterministic quality-first Kruskal spanning forest.
    Quality,
}

impl ForestPolicy {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "traversal" => Ok(Self::Traversal),
            "quality" => Ok(Self::Quality),
            other => Err(format!(
                "--forest-policy expects traversal or quality; got {other:?}"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Traversal => "traversal",
            Self::Quality => "quality",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameOwnerPolicy {
    /// Preserve the historical newest-window owner selection.
    Newest,
    /// Prefer the owner for which the frame is deepest inside its registered range.
    Interior,
}

impl FrameOwnerPolicy {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "newest" => Ok(Self::Newest),
            "interior" => Ok(Self::Interior),
            other => Err(format!(
                "--frame-owner-policy expects newest or interior; got {other:?}"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Newest => "newest",
            Self::Interior => "interior",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    rig_manifest: PathBuf,
    nodes_tsv: PathBuf,
    out_dir: PathBuf,
    seam_config_arm: SeamConfigArm,
    frame_owner_policy: FrameOwnerPolicy,
    forest_policy: ForestPolicy,
    metric_se3: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NodeSpec {
    node_id: u64,
    window_start: u64,
    images_txt: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
struct SensorSpec {
    camera_id: u64,
    sensor_from_rig: SE3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameImageAssignment {
    frame_id: u64,
    sensor_index: usize,
    /// Deterministic global output id assigned by F-row appearance order.
    global_image_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct RigManifest {
    sensors: BTreeMap<usize, SensorSpec>,
    assignments: BTreeMap<String, FrameImageAssignment>,
}

#[derive(Debug, Clone, PartialEq)]
struct ColmapImage {
    image_id: u64,
    /// Assigned from the rig manifest; parser-only images use zero until
    /// node extraction binds them to a manifest F row.
    global_image_id: u64,
    camera_id: u64,
    name: String,
    world_to_camera: Pose,
    /// Fixed calibration used to recombine this sensor with its rig pose.
    /// Parser-only images remain unbound until manifest extraction.
    sensor_from_rig: Option<SE3>,
}

#[derive(Debug, Clone, PartialEq)]
struct FrameData {
    pose: Pose,
    images: Vec<ColmapImage>,
}

#[derive(Debug, Clone, PartialEq)]
struct StepStats {
    median_m: f64,
    max_m: f64,
    ratio: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct NodeData {
    spec: NodeSpec,
    images: Vec<ColmapImage>,
    frames: BTreeMap<u64, FrameData>,
    step: StepStats,
}

#[derive(Debug, Clone, PartialEq)]
struct SeamDiagnostic {
    source_node_id: u64,
    target_node_id: u64,
    start_gap: u64,
    accepted: bool,
    common_frames: usize,
    retained_frames: usize,
    reason: String,
    wrapper_rejection_reason: Option<String>,
    sim3_rejection_reason: Option<String>,
    sim3_rejection_inlier_count: Option<usize>,
    sim3_rejection_inlier_ratio: Option<f64>,
    sim3_rejection_mean_residual_ratio: Option<f64>,
    sim3_rejection_rotation_disagreement_deg: Option<f64>,
    sim3_rejection_leave_one_out_log_scale_mad: Option<f64>,
    alignment_method: Option<String>,
    constraint_scale: Option<f64>,
    constraint_rotation_angle_deg: Option<f64>,
    constraint_translation_norm: Option<f64>,
    constraint_inlier_ratio: Option<f64>,
    constraint_mean_residual_ratio: Option<f64>,
    constraint_leave_one_out_log_scale_mad: Option<f64>,
    fallback_attempted: bool,
    fallback_used: bool,
    fallback_rejection_reason: Option<String>,
    fallback_rejection_inlier_count: Option<usize>,
    fallback_rejection_inlier_ratio: Option<f64>,
    fallback_rejection_mean_residual_ratio: Option<f64>,
    fallback_rejection_rotation_disagreement_deg: Option<f64>,
    fallback_rejection_leave_one_out_log_scale_mad: Option<f64>,
    selected_for_transform_graph: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct SeamEdge {
    source_index: usize,
    target_index: usize,
    target_from_source: Sim3,
    common_frames: usize,
    retained_frames: usize,
    inlier_ratio: Option<f64>,
    mean_residual_ratio: Option<f64>,
    start_gap: u64,
    method: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct AtlasComponent {
    anchor_index: usize,
    node_indices: Vec<usize>,
    atlas_from_node: BTreeMap<usize, Sim3>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ComponentSummary {
    component_index: usize,
    anchor_node_id: u64,
    node_ids: Vec<u64>,
    frame_count: usize,
    image_count: usize,
    min_frame_id: Option<u64>,
    max_frame_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
struct StitchResult {
    seam_diagnostics: Vec<SeamDiagnostic>,
    components: Vec<ComponentSummary>,
    metric_se3: Vec<MetricSe3Summary>,
}

#[derive(Debug, Clone, PartialEq)]
struct MetricSe3Summary {
    component_index: usize,
    node_count: usize,
    edge_count: usize,
    initial_cost: f64,
    final_cost: f64,
    iterations: usize,
    converged: bool,
}

fn usage() -> ! {
    eprintln!("{USAGE}");
    std::process::exit(2);
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = args.into_iter();
    let _program = values.next();
    let mut rig_manifest = None;
    let mut nodes_tsv = None;
    let mut out_dir = None;
    let mut seam_config_arm = SeamConfigArm::A;
    let mut seam_config_arm_seen = false;
    let mut frame_owner_policy = FrameOwnerPolicy::Newest;
    let mut frame_owner_policy_seen = false;
    let mut forest_policy = ForestPolicy::Traversal;
    let mut forest_policy_seen = false;
    let mut metric_se3 = false;
    let mut metric_se3_seen = false;
    while let Some(flag) = values.next() {
        if flag == "--seam-config-arm" {
            if seam_config_arm_seen {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            seam_config_arm_seen = true;
            let value = values.next().ok_or_else(|| {
                format!("{flag} requires A, B, C, D, E, F, G, H, I, J, K, or L\n{USAGE}")
            })?;
            if value.starts_with('-') {
                return Err(format!(
                    "{flag} requires A, B, C, D, E, F, G, H, I, J, K, or L, got {value:?}\n{USAGE}"
                ));
            }
            seam_config_arm =
                SeamConfigArm::parse(&value).map_err(|error| format!("{error}\n{USAGE}"))?;
            continue;
        }
        if flag == "--frame-owner-policy" {
            if frame_owner_policy_seen {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            frame_owner_policy_seen = true;
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires newest or interior\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!(
                    "{flag} requires newest or interior, got {value:?}\n{USAGE}"
                ));
            }
            frame_owner_policy =
                FrameOwnerPolicy::parse(&value).map_err(|error| format!("{error}\n{USAGE}"))?;
            continue;
        }
        if flag == "--forest-policy" {
            if forest_policy_seen {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            forest_policy_seen = true;
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires traversal or quality\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!(
                    "{flag} requires traversal or quality, got {value:?}\n{USAGE}"
                ));
            }
            forest_policy =
                ForestPolicy::parse(&value).map_err(|error| format!("{error}\n{USAGE}"))?;
            continue;
        }
        if flag == "--metric-se3" {
            if metric_se3_seen {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            metric_se3_seen = true;
            metric_se3 = true;
            continue;
        }
        let slot = match flag.as_str() {
            "--rig-manifest" => &mut rig_manifest,
            "--nodes-tsv" => &mut nodes_tsv,
            "--out-dir" => &mut out_dir,
            "-h" | "--help" => return Err(USAGE.to_owned()),
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
        out_dir: out_dir.ok_or_else(|| format!("--out-dir is required\n{USAGE}"))?,
        seam_config_arm,
        frame_owner_policy,
        forest_policy,
        metric_se3,
    })
}

fn parse_node_specs(path: &Path) -> Result<Vec<NodeSpec>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("read node manifest {}: {error}", path.display()))?;
    let mut nodes = Vec::new();
    let mut node_ids = BTreeSet::new();
    let mut start_and_path = BTreeSet::new();
    for (zero_line, raw) in contents.lines().enumerate() {
        let line = zero_line + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(format!(
                "node manifest {}:{} requires node_id<TAB>window_start<TAB>images_txt",
                path.display(),
                line
            ));
        }
        let node_id = fields[0].parse::<u64>().map_err(|error| {
            format!(
                "node manifest {}:{} has invalid node id {:?}: {error}",
                path.display(),
                line,
                fields[0]
            )
        })?;
        let window_start = fields[1].parse::<u64>().map_err(|error| {
            format!(
                "node manifest {}:{} has invalid window start {:?}: {error}",
                path.display(),
                line,
                fields[1]
            )
        })?;
        if fields[2].is_empty() {
            return Err(format!(
                "node manifest {}:{} has an empty images.txt path",
                path.display(),
                line
            ));
        }
        let images_txt = PathBuf::from(fields[2]);
        if !node_ids.insert(node_id) {
            return Err(format!(
                "node manifest {}:{} duplicates node id {node_id}",
                path.display(),
                line
            ));
        }
        if !start_and_path.insert((window_start, images_txt.clone())) {
            return Err(format!(
                "node manifest {}:{} duplicates window_start/images_txt ({window_start}, {})",
                path.display(),
                line,
                images_txt.display()
            ));
        }
        if !images_txt.is_file() {
            return Err(format!(
                "node manifest {}:{} images.txt does not exist: {}",
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
        return Err(format!(
            "node manifest {} contains no nodes",
            path.display()
        ));
    }
    nodes.sort_by_key(|node| (node.window_start, node.node_id));
    Ok(nodes)
}

fn parse_rig_manifest(path: &Path) -> Result<RigManifest, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("read rig manifest {}: {error}", path.display()))?;
    let mut sensors = BTreeMap::new();
    let mut frame_rows = Vec::new();
    for (zero_line, raw) in contents.lines().enumerate() {
        let line = zero_line + 1;
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
                let index = parse_usize_field(&fields, 1, path, line, "sensor index")?;
                let camera_id = parse_u64_field(&fields, 2, path, line, "camera id")?;
                let width = parse_usize_field(&fields, 3, path, line, "width")?;
                let height = parse_usize_field(&fields, 4, path, line, "height")?;
                if width == 0 || height == 0 {
                    return Err(format!(
                        "rig manifest {}:{} sensor dimensions must be positive",
                        path.display(),
                        line
                    ));
                }
                for field in 5..16 {
                    let value = parse_f64_field(&fields, field, path, line, "sensor value")?;
                    if !value.is_finite() {
                        return Err(format!(
                            "rig manifest {}:{} sensor value is non-finite",
                            path.display(),
                            line
                        ));
                    }
                }
                let quaternion = Quaternion::new(
                    parse_f64_field(&fields, 9, path, line, "quaternion")?,
                    parse_f64_field(&fields, 10, path, line, "quaternion")?,
                    parse_f64_field(&fields, 11, path, line, "quaternion")?,
                    parse_f64_field(&fields, 12, path, line, "quaternion")?,
                );
                let norm = quaternion.norm();
                if !norm.is_finite() || norm <= 1.0e-12 {
                    return Err(format!(
                        "rig manifest {}:{} sensor quaternion is zero or non-finite",
                        path.display(),
                        line
                    ));
                }
                let sensor_from_rig = SE3::new(
                    UnitQuaternion::new_normalize(quaternion),
                    Vector3::new(
                        parse_f64_field(&fields, 13, path, line, "translation")?,
                        parse_f64_field(&fields, 14, path, line, "translation")?,
                        parse_f64_field(&fields, 15, path, line, "translation")?,
                    ),
                );
                if sensors
                    .insert(
                        index,
                        SensorSpec {
                            camera_id,
                            sensor_from_rig,
                        },
                    )
                    .is_some()
                {
                    return Err(format!(
                        "rig manifest {}:{} duplicates sensor index {index}",
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
                let frame_id = parse_u64_field(&fields, 1, path, line, "frame id")?;
                if fields[2].is_empty() {
                    return Err(format!(
                        "rig manifest {}:{} image name is empty",
                        path.display(),
                        line
                    ));
                }
                let sensor_index = parse_usize_field(&fields, 3, path, line, "frame sensor index")?;
                frame_rows.push((frame_id, fields[2].to_owned(), sensor_index, line));
            }
            Some(kind) => {
                return Err(format!(
                    "rig manifest {}:{} unknown row kind {kind:?}",
                    path.display(),
                    line
                ));
            }
            None => unreachable!("empty rows are skipped above"),
        }
    }
    if sensors.is_empty() {
        return Err(format!(
            "rig manifest {} contains no sensors",
            path.display()
        ));
    }
    if sensors
        .keys()
        .copied()
        .enumerate()
        .any(|(expected, actual)| expected != actual)
    {
        return Err(format!(
            "rig manifest {} sensor indices must be contiguous from zero",
            path.display()
        ));
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
        let global_image_id = assignments.len() as u64 + 1;
        if assignments
            .insert(
                name.clone(),
                FrameImageAssignment {
                    frame_id,
                    sensor_index,
                    global_image_id,
                },
            )
            .is_some()
        {
            return Err(format!(
                "rig manifest {}:{} duplicates image assignment {name:?}",
                path.display(),
                line
            ));
        }
        if !frame_sensors.insert((frame_id, sensor_index)) {
            return Err(format!(
                "rig manifest {}:{} duplicates frame {frame_id} sensor {sensor_index}",
                path.display(),
                line
            ));
        }
    }
    if assignments.is_empty() {
        return Err(format!(
            "rig manifest {} contains no frame rows",
            path.display()
        ));
    }
    Ok(RigManifest {
        sensors,
        assignments,
    })
}

fn parse_u64_field(
    fields: &[&str],
    index: usize,
    path: &Path,
    line: usize,
    label: &str,
) -> Result<u64, String> {
    fields
        .get(index)
        .ok_or_else(|| {
            format!(
                "rig manifest {}:{} missing {label} field {index}",
                path.display(),
                line
            )
        })?
        .parse::<u64>()
        .map_err(|error| {
            format!(
                "rig manifest {}:{} invalid {label} {:?}: {error}",
                path.display(),
                line,
                fields[index]
            )
        })
}

fn parse_usize_field(
    fields: &[&str],
    index: usize,
    path: &Path,
    line: usize,
    label: &str,
) -> Result<usize, String> {
    fields
        .get(index)
        .ok_or_else(|| {
            format!(
                "rig manifest {}:{} missing {label} field {index}",
                path.display(),
                line
            )
        })?
        .parse::<usize>()
        .map_err(|error| {
            format!(
                "rig manifest {}:{} invalid {label} {:?}: {error}",
                path.display(),
                line,
                fields[index]
            )
        })
}

fn parse_f64_field(
    fields: &[&str],
    index: usize,
    path: &Path,
    line: usize,
    label: &str,
) -> Result<f64, String> {
    fields
        .get(index)
        .ok_or_else(|| {
            format!(
                "rig manifest {}:{} missing {label} field {index}",
                path.display(),
                line
            )
        })?
        .parse::<f64>()
        .map_err(|error| {
            format!(
                "rig manifest {}:{} invalid {label} {:?}: {error}",
                path.display(),
                line,
                fields[index]
            )
        })
}

fn parse_colmap_images(path: &Path) -> Result<Vec<ColmapImage>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("read COLMAP images {}: {error}", path.display()))?;
    let lines = contents.lines().collect::<Vec<_>>();
    let mut images = Vec::new();
    let mut image_ids = BTreeSet::new();
    let mut image_names = BTreeSet::new();
    let mut index = 0;
    while index < lines.len() {
        let line_number = index + 1;
        let raw = lines[index];
        index += 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 10 {
            return Err(format!(
                "COLMAP images {}:{} pose row requires 10 fields",
                path.display(),
                line_number
            ));
        }
        let image_id = fields[0].parse::<u64>().map_err(|error| {
            format!(
                "COLMAP images {}:{} invalid image id {:?}: {error}",
                path.display(),
                line_number,
                fields[0]
            )
        })?;
        let values = fields[1..8]
            .iter()
            .map(|field| {
                field.parse::<f64>().map_err(|error| {
                    format!(
                        "COLMAP images {}:{} invalid pose value {:?}: {error}",
                        path.display(),
                        line_number,
                        field
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "COLMAP images {}:{} pose contains a non-finite value",
                path.display(),
                line_number
            ));
        }
        let camera_id = fields[8].parse::<u64>().map_err(|error| {
            format!(
                "COLMAP images {}:{} invalid camera id {:?}: {error}",
                path.display(),
                line_number,
                fields[8]
            )
        })?;
        let name = fields[9].to_owned();
        if name.is_empty() {
            return Err(format!(
                "COLMAP images {}:{} image name is empty",
                path.display(),
                line_number
            ));
        }
        if !image_ids.insert(image_id) {
            return Err(format!(
                "COLMAP images {}:{} duplicates image id {image_id}",
                path.display(),
                line_number
            ));
        }
        if !image_names.insert(name.clone()) {
            return Err(format!(
                "COLMAP images {}:{} duplicates image name {name:?}",
                path.display(),
                line_number
            ));
        }
        let quaternion = Quaternion::new(values[0], values[1], values[2], values[3]);
        let norm = quaternion.norm();
        if !norm.is_finite() || norm <= 1.0e-12 {
            return Err(format!(
                "COLMAP images {}:{} quaternion is zero or non-finite",
                path.display(),
                line_number
            ));
        }
        let pose = Pose::from_world_to_camera(
            UnitQuaternion::new_normalize(quaternion),
            Vector3::new(values[4], values[5], values[6]),
        );
        if index >= lines.len() {
            return Err(format!(
                "COLMAP images {}:{} is missing its POINTS2D row",
                path.display(),
                line_number
            ));
        }
        let points_line_number = index + 1;
        let points = lines[index].trim();
        index += 1;
        validate_points2d_line(points, path, points_line_number)?;
        images.push(ColmapImage {
            image_id,
            global_image_id: 0,
            camera_id,
            name,
            world_to_camera: pose,
            sensor_from_rig: None,
        });
    }
    if images.is_empty() {
        return Err(format!(
            "COLMAP images {} contains no poses",
            path.display()
        ));
    }
    Ok(images)
}

fn validate_points2d_line(points: &str, path: &Path, line: usize) -> Result<(), String> {
    if points.is_empty() {
        return Ok(());
    }
    let fields = points.split_whitespace().collect::<Vec<_>>();
    if fields.len() % 3 != 0 {
        return Err(format!(
            "COLMAP images {}:{} POINTS2D row must contain triples",
            path.display(),
            line
        ));
    }
    for triple in fields.chunks_exact(3) {
        triple[0].parse::<f64>().map_err(|error| {
            format!(
                "COLMAP images {}:{} invalid POINTS2D x {:?}: {error}",
                path.display(),
                line,
                triple[0]
            )
        })?;
        triple[1].parse::<f64>().map_err(|error| {
            format!(
                "COLMAP images {}:{} invalid POINTS2D y {:?}: {error}",
                path.display(),
                line,
                triple[1]
            )
        })?;
        triple[2].parse::<i64>().map_err(|error| {
            format!(
                "COLMAP images {}:{} invalid POINTS2D id {:?}: {error}",
                path.display(),
                line,
                triple[2]
            )
        })?;
    }
    Ok(())
}

fn extract_node_data(spec: NodeSpec, manifest: &RigManifest) -> Result<NodeData, String> {
    let images = parse_colmap_images(&spec.images_txt)?;
    let mut frames = BTreeMap::<u64, FrameData>::new();
    let mut bound_images = Vec::with_capacity(images.len());
    for image in &images {
        let assignment = manifest.assignments.get(&image.name).ok_or_else(|| {
            format!(
                "node {} image {:?} is absent from rig manifest",
                spec.node_id, image.name
            )
        })?;
        let sensor = manifest
            .sensors
            .get(&assignment.sensor_index)
            .expect("manifest assignments are validated against sensors");
        if image.camera_id != sensor.camera_id {
            return Err(format!(
                "node {} image {:?} camera {} disagrees with manifest sensor {} camera {}",
                spec.node_id,
                image.name,
                image.camera_id,
                assignment.sensor_index,
                sensor.camera_id
            ));
        }
        let mut bound_image = image.clone();
        bound_image.global_image_id = assignment.global_image_id;
        bound_image.sensor_from_rig = Some(sensor.sensor_from_rig.clone());
        let rig_pose = Pose {
            world_to_camera: sensor
                .sensor_from_rig
                .inverse()
                .compose(&image.world_to_camera.world_to_camera),
        };
        validate_pose(
            &rig_pose,
            &format!("node {} frame {}", spec.node_id, assignment.frame_id),
        )?;
        let frame = frames
            .entry(assignment.frame_id)
            .or_insert_with(|| FrameData {
                pose: rig_pose.clone(),
                images: Vec::new(),
            });
        if frame
            .images
            .iter()
            .any(|row| row.name == bound_image.name || row.camera_id == bound_image.camera_id)
        {
            return Err(format!(
                "node {} contains duplicate image/frame assignment for frame {}",
                spec.node_id, assignment.frame_id
            ));
        }
        let centre_error =
            (frame.pose.camera_center_world() - rig_pose.camera_center_world()).norm();
        let rotation_error = frame
            .pose
            .world_to_camera
            .rotation
            .rotation_to(&rig_pose.world_to_camera.rotation)
            .angle()
            .to_degrees();
        if !centre_error.is_finite()
            || !rotation_error.is_finite()
            || centre_error > MAX_RIG_CENTER_DISAGREEMENT_M
            || rotation_error > MAX_RIG_ROTATION_DISAGREEMENT_DEG
        {
            return Err(format!(
                "node {} frame {} sensor poses disagree: centre={centre_error:.9}m rotation={rotation_error:.9}deg",
                spec.node_id, assignment.frame_id
            ));
        }
        frame.images.push(bound_image.clone());
        bound_images.push(bound_image);
    }
    for frame in frames.values_mut() {
        frame
            .images
            .sort_by_key(|image| (image.camera_id, image.image_id));
    }
    let step = compute_step_stats(&frames);
    Ok(NodeData {
        spec,
        images: bound_images,
        frames,
        step,
    })
}

fn validate_pose(pose: &Pose, context: &str) -> Result<(), String> {
    if !pose
        .world_to_camera
        .rotation
        .coords
        .iter()
        .all(|value| value.is_finite())
        || !pose
            .world_to_camera
            .translation
            .iter()
            .all(|value| value.is_finite())
    {
        return Err(format!("{context} pose contains a non-finite value"));
    }
    let centre = pose.camera_center_world();
    if !centre.coords.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "{context} camera centre contains a non-finite value"
        ));
    }
    Ok(())
}

fn compute_step_stats(frames: &BTreeMap<u64, FrameData>) -> StepStats {
    let centres = frames
        .values()
        .map(|frame| frame.pose.camera_center_world())
        .collect::<Vec<_>>();
    let mut steps = centres
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).norm())
        .collect::<Vec<_>>();
    steps.retain(|step| step.is_finite());
    steps.sort_by(f64::total_cmp);
    let median_m = median(&steps).unwrap_or(0.0);
    let max_m = steps.last().copied().unwrap_or(0.0);
    let ratio = if median_m > 0.0 {
        max_m / median_m
    } else if max_m == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    StepStats {
        median_m,
        max_m,
        ratio,
    }
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

#[cfg(test)]
fn seam_alignment_config() -> RigSubmapAlignmentConfig {
    seam_alignment_config_for_arm(SeamConfigArm::A)
}

fn seam_alignment_config_for_arm(arm: SeamConfigArm) -> RigSubmapAlignmentConfig {
    let mut alignment = SubmapSim3AlignmentConfig {
        min_correspondences: 8,
        min_inliers: 8,
        ..SubmapSim3AlignmentConfig::default()
    };
    match arm {
        SeamConfigArm::A => {}
        SeamConfigArm::B => {
            alignment.min_second_to_first_singular_ratio = 0.0;
        }
        SeamConfigArm::C => {
            alignment.min_second_to_first_singular_ratio = 0.0;
            alignment.max_inlier_residual_ratio = 0.10;
            alignment.max_mean_residual_ratio = 0.05;
        }
        SeamConfigArm::D => {
            alignment.min_second_to_first_singular_ratio = 0.0;
            alignment.max_inlier_residual_ratio = 0.10;
            alignment.max_mean_residual_ratio = 0.05;
            alignment.min_inlier_ratio = 0.5;
        }
        // Arm E is diagnostic-only for exact shared-frame identities.  The
        // measured gates are singular=0, residual=.10/.06, inlier=.35, and
        // local rotation disagreement=60 degrees; arm A remains the
        // production default.
        SeamConfigArm::E => {
            alignment.min_second_to_first_singular_ratio = 0.0;
            alignment.max_inlier_residual_ratio = 0.10;
            alignment.max_mean_residual_ratio = 0.06;
            alignment.min_inlier_ratio = 0.35;
            alignment.max_rotation_disagreement_deg = 60.0;
        }
        // Arm F is exactly arm E plus the wrapper-only, fixed-consensus
        // orientation fallback.  No residual, ratio, or rotation gate is
        // relaxed beyond the measured diagnostic arm E.
        SeamConfigArm::F => {
            alignment.min_second_to_first_singular_ratio = 0.0;
            alignment.max_inlier_residual_ratio = 0.10;
            alignment.max_mean_residual_ratio = 0.06;
            alignment.min_inlier_ratio = 0.35;
            alignment.max_rotation_disagreement_deg = 60.0;
        }
        // Arm G is exactly arm E with fixed-consensus orientation selected as
        // the primary estimator. It intentionally has no generic fallback.
        SeamConfigArm::G => {
            alignment.min_second_to_first_singular_ratio = 0.0;
            alignment.max_inlier_residual_ratio = 0.10;
            alignment.max_mean_residual_ratio = 0.06;
            alignment.min_inlier_ratio = 0.35;
            alignment.max_rotation_disagreement_deg = 60.0;
        }
        // Arm H is exactly arm G plus a diagnostic generic fallback after a
        // fixed-primary rejection. It does not change any quality threshold.
        // Arm I is H with uniform overlap sampling instead of earliest ids.
        SeamConfigArm::H | SeamConfigArm::I => {
            alignment.min_second_to_first_singular_ratio = 0.0;
            alignment.max_inlier_residual_ratio = 0.10;
            alignment.max_mean_residual_ratio = 0.06;
            alignment.min_inlier_ratio = 0.35;
            alignment.max_rotation_disagreement_deg = 60.0;
        }
        // Arm J is H with only the fixed-rotation mean-residual diagnostic
        // gate widened to the measured 0.07 value.  It remains diagnostic
        // and does not alter arms A through I.
        SeamConfigArm::J => {
            alignment.min_second_to_first_singular_ratio = 0.0;
            alignment.max_inlier_residual_ratio = 0.10;
            alignment.max_mean_residual_ratio = 0.07;
            alignment.min_inlier_ratio = 0.35;
            alignment.max_rotation_disagreement_deg = 60.0;
        }
        // Arm K is J with unit scale forced in the fixed-rotation estimator.
        // Arm L is identical to K except that generic fallback is disabled.
        SeamConfigArm::K | SeamConfigArm::L => {
            alignment.min_second_to_first_singular_ratio = 0.0;
            alignment.max_inlier_residual_ratio = 0.10;
            alignment.max_mean_residual_ratio = 0.07;
            alignment.min_inlier_ratio = 0.35;
            alignment.max_rotation_disagreement_deg = 60.0;
        }
    }
    RigSubmapAlignmentConfig {
        max_boundary_frames: 64,
        boundary_sampling: if arm == SeamConfigArm::I {
            RigSubmapBoundarySampling::UniformAcrossOverlap
        } else {
            RigSubmapBoundarySampling::Earliest
        },
        rotation_consensus_deg: if matches!(
            arm,
            SeamConfigArm::E
                | SeamConfigArm::F
                | SeamConfigArm::G
                | SeamConfigArm::H
                | SeamConfigArm::I
                | SeamConfigArm::J
                | SeamConfigArm::K
                | SeamConfigArm::L
        ) {
            60.0
        } else {
            5.0
        },
        min_rotation_consensus_ratio: if matches!(
            arm,
            SeamConfigArm::E
                | SeamConfigArm::F
                | SeamConfigArm::G
                | SeamConfigArm::H
                | SeamConfigArm::I
                | SeamConfigArm::J
                | SeamConfigArm::K
                | SeamConfigArm::L
        ) {
            0.35
        } else {
            0.6
        },
        alignment,
        allow_fixed_rotation_fallback: arm == SeamConfigArm::F,
        prefer_fixed_rotation_alignment: matches!(
            arm,
            SeamConfigArm::G
                | SeamConfigArm::H
                | SeamConfigArm::I
                | SeamConfigArm::J
                | SeamConfigArm::K
                | SeamConfigArm::L
        ),
        allow_generic_fallback_after_fixed_rotation: matches!(
            arm,
            SeamConfigArm::H | SeamConfigArm::I | SeamConfigArm::J | SeamConfigArm::K
        ),
        force_fixed_rotation_unit_scale: matches!(arm, SeamConfigArm::K | SeamConfigArm::L),
    }
}

#[cfg(test)]
fn build_seam_graph(nodes: &[NodeData]) -> (Vec<SeamEdge>, Vec<SeamDiagnostic>) {
    build_seam_graph_with_config(nodes, &seam_alignment_config())
}

fn build_seam_graph_with_config(
    nodes: &[NodeData],
    config: &RigSubmapAlignmentConfig,
) -> (Vec<SeamEdge>, Vec<SeamDiagnostic>) {
    let frame_poses = nodes
        .iter()
        .map(|node| {
            node.frames
                .iter()
                .map(|(frame_id, frame)| (*frame_id, frame.pose.clone()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut start_groups = BTreeMap::<u64, Vec<usize>>::new();
    for (node_index, node) in nodes.iter().enumerate() {
        start_groups
            .entry(node.spec.window_start)
            .or_default()
            .push(node_index);
    }
    for node_indices in start_groups.values_mut() {
        node_indices.sort_by_key(|index| nodes[*index].spec.node_id);
    }

    let mut edges = Vec::new();
    let mut diagnostics = Vec::new();
    for source_index in 0..nodes.len() {
        let source = &nodes[source_index];
        let lower_start = source.spec.window_start.saturating_sub(MAX_SEAM_START_GAP);
        for (target_start, target_indices) in
            start_groups.range(lower_start..source.spec.window_start)
        {
            let start_gap = source.spec.window_start - *target_start;
            for target_index in target_indices {
                let target = &nodes[*target_index];
                match estimate_rig_submap_sim3_constraint(
                    source.spec.node_id,
                    target.spec.node_id,
                    &frame_poses[source_index],
                    &frame_poses[*target_index],
                    config,
                ) {
                    Ok(result) => {
                        // Keep the primary rejection visible for fixed-primary
                        // H, while retaining the generic-primary field for F
                        // and the default path.
                        let sim3_rejection = result
                            .diagnostics
                            .fixed_rotation_rejection
                            .as_ref()
                            .or(result.diagnostics.generic_sim3_rejection.as_ref());
                        let constraint = &result.constraint;
                        let target_from_source = constraint.target_from_source.clone();
                        diagnostics.push(SeamDiagnostic {
                            source_node_id: source.spec.node_id,
                            target_node_id: target.spec.node_id,
                            start_gap,
                            accepted: true,
                            common_frames: result.diagnostics.common_frame_count,
                            retained_frames: result.diagnostics.retained_frame_count,
                            reason: "accepted".to_owned(),
                            wrapper_rejection_reason: None,
                            sim3_rejection_reason: sim3_rejection
                                .map(|rejection| format!("{:?}", rejection.reason)),
                            sim3_rejection_inlier_count: sim3_rejection
                                .map(|rejection| rejection.inlier_count),
                            sim3_rejection_inlier_ratio: sim3_rejection
                                .map(|rejection| rejection.inlier_ratio),
                            sim3_rejection_mean_residual_ratio: sim3_rejection
                                .and_then(|rejection| rejection.mean_residual_ratio),
                            sim3_rejection_rotation_disagreement_deg: sim3_rejection
                                .and_then(|rejection| rejection.rotation_disagreement_deg),
                            sim3_rejection_leave_one_out_log_scale_mad: sim3_rejection
                                .and_then(|rejection| rejection.leave_one_out_log_scale_mad),
                            alignment_method: result
                                .diagnostics
                                .alignment_method
                                .map(|method| method.as_str().to_owned()),
                            constraint_scale: Some(constraint.target_from_source.scale),
                            constraint_rotation_angle_deg: Some(
                                constraint.target_from_source.rotation.angle().to_degrees(),
                            ),
                            constraint_translation_norm: Some(
                                constraint.target_from_source.translation.norm(),
                            ),
                            constraint_inlier_ratio: Some(constraint.inlier_ratio),
                            constraint_mean_residual_ratio: Some(constraint.mean_residual_ratio),
                            constraint_leave_one_out_log_scale_mad: Some(
                                constraint.leave_one_out_log_scale_mad,
                            ),
                            fallback_attempted: result.diagnostics.fallback_attempted,
                            fallback_used: result.diagnostics.fallback_used,
                            fallback_rejection_reason: None,
                            fallback_rejection_inlier_count: None,
                            fallback_rejection_inlier_ratio: None,
                            fallback_rejection_mean_residual_ratio: None,
                            fallback_rejection_rotation_disagreement_deg: None,
                            fallback_rejection_leave_one_out_log_scale_mad: None,
                            selected_for_transform_graph: false,
                        });
                        edges.push(SeamEdge {
                            source_index,
                            target_index: *target_index,
                            target_from_source,
                            common_frames: result.diagnostics.common_frame_count,
                            retained_frames: result.diagnostics.retained_frame_count,
                            inlier_ratio: Some(constraint.inlier_ratio),
                            mean_residual_ratio: Some(constraint.mean_residual_ratio),
                            start_gap,
                            method: result
                                .diagnostics
                                .alignment_method
                                .map(|method| method.as_str().to_owned()),
                        });
                    }
                    Err(rejection) => {
                        let sim3_rejection = rejection.sim3_rejection.as_ref();
                        let fallback_rejection = rejection.fallback_rejection.as_ref();
                        diagnostics.push(SeamDiagnostic {
                            source_node_id: source.spec.node_id,
                            target_node_id: target.spec.node_id,
                            start_gap,
                            accepted: false,
                            common_frames: rejection.diagnostics.common_frame_count,
                            retained_frames: rejection.diagnostics.retained_frame_count,
                            reason: format!("{:?}", rejection.reason),
                            wrapper_rejection_reason: Some(format!("{:?}", rejection.reason)),
                            sim3_rejection_reason: sim3_rejection
                                .map(|rejection| format!("{:?}", rejection.reason)),
                            sim3_rejection_inlier_count: sim3_rejection
                                .map(|rejection| rejection.inlier_count),
                            sim3_rejection_inlier_ratio: sim3_rejection
                                .map(|rejection| rejection.inlier_ratio),
                            sim3_rejection_mean_residual_ratio: sim3_rejection
                                .and_then(|rejection| rejection.mean_residual_ratio),
                            sim3_rejection_rotation_disagreement_deg: sim3_rejection
                                .and_then(|rejection| rejection.rotation_disagreement_deg),
                            sim3_rejection_leave_one_out_log_scale_mad: sim3_rejection
                                .and_then(|rejection| rejection.leave_one_out_log_scale_mad),
                            alignment_method: rejection
                                .diagnostics
                                .alignment_method
                                .map(|method| method.as_str().to_owned()),
                            constraint_scale: None,
                            constraint_rotation_angle_deg: None,
                            constraint_translation_norm: None,
                            constraint_inlier_ratio: None,
                            constraint_mean_residual_ratio: None,
                            constraint_leave_one_out_log_scale_mad: None,
                            fallback_attempted: rejection.diagnostics.fallback_attempted,
                            fallback_used: rejection.diagnostics.fallback_used,
                            fallback_rejection_reason: fallback_rejection
                                .map(|rejection| format!("{:?}", rejection.reason)),
                            fallback_rejection_inlier_count: fallback_rejection
                                .map(|rejection| rejection.inlier_count),
                            fallback_rejection_inlier_ratio: fallback_rejection
                                .map(|rejection| rejection.inlier_ratio),
                            fallback_rejection_mean_residual_ratio: fallback_rejection
                                .and_then(|rejection| rejection.mean_residual_ratio),
                            fallback_rejection_rotation_disagreement_deg: fallback_rejection
                                .and_then(|rejection| rejection.rotation_disagreement_deg),
                            fallback_rejection_leave_one_out_log_scale_mad: fallback_rejection
                                .and_then(|rejection| rejection.leave_one_out_log_scale_mad),
                            selected_for_transform_graph: false,
                        });
                    }
                }
            }
        }
    }
    diagnostics.sort_by_key(|diagnostic| {
        (
            diagnostic.source_node_id,
            diagnostic.target_node_id,
            diagnostic.start_gap,
        )
    });
    edges.sort_by_key(|edge| {
        (
            nodes[edge.source_index].spec.node_id,
            nodes[edge.target_index].spec.node_id,
        )
    });
    // The candidate graph is already bounded by the predecessor start-group
    // scan above. Keep every accepted candidate here; the caller chooses the
    // traversal or quality transform graph explicitly.
    (edges, diagnostics)
}

/// Select a deterministic spanning forest from accepted seam candidates.
/// Higher-support seams are preferred before residual quality, inlier support,
/// temporal gap, and node-id tie breakers. The input candidate list may be in
/// any order; all ordering that affects the result is explicit here.
fn select_quality_forest(nodes: &[NodeData], edges: &[SeamEdge]) -> Vec<SeamEdge> {
    let mut ordered = edges.to_vec();
    ordered.sort_by(|left, right| compare_edge_quality(nodes, left, right));
    let mut union_find = UnionFind::new(nodes.len());
    let mut forest = Vec::with_capacity(nodes.len().saturating_sub(1));
    for edge in ordered {
        debug_assert!(edge.method.is_some());
        if union_find.union(edge.source_index, edge.target_index) {
            forest.push(edge);
        }
    }
    forest
}

fn compare_edge_quality(nodes: &[NodeData], left: &SeamEdge, right: &SeamEdge) -> Ordering {
    right
        .common_frames
        .cmp(&left.common_frames)
        .then_with(|| right.retained_frames.cmp(&left.retained_frames))
        .then_with(|| {
            compare_optional_f64_ascending(left.mean_residual_ratio, right.mean_residual_ratio)
        })
        .then_with(|| compare_optional_f64_descending(left.inlier_ratio, right.inlier_ratio))
        .then_with(|| left.start_gap.cmp(&right.start_gap))
        .then_with(|| {
            nodes[left.source_index]
                .spec
                .node_id
                .cmp(&nodes[right.source_index].spec.node_id)
        })
        .then_with(|| {
            nodes[left.target_index]
                .spec
                .node_id
                .cmp(&nodes[right.target_index].spec.node_id)
        })
}

fn compare_optional_f64_ascending(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn compare_optional_f64_descending(left: Option<f64>, right: Option<f64>) -> Ordering {
    compare_optional_f64_ascending(right, left)
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            let root = self.find(self.parent[node]);
            self.parent[node] = root;
        }
        self.parent[node]
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return false;
        }
        if self.rank[left_root] < self.rank[right_root]
            || (self.rank[left_root] == self.rank[right_root] && left_root > right_root)
        {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
        true
    }
}

fn connected_components(node_count: usize, edges: &[SeamEdge]) -> Vec<AtlasComponent> {
    let mut adjacency = vec![Vec::<(usize, Sim3)>::new(); node_count];
    for edge in edges {
        adjacency[edge.source_index].push((edge.target_index, edge.target_from_source.inverse()));
        adjacency[edge.target_index].push((edge.source_index, edge.target_from_source.clone()));
    }
    for neighbours in &mut adjacency {
        neighbours.sort_by_key(|(node_index, _)| *node_index);
    }

    let mut remaining = (0..node_count).collect::<BTreeSet<_>>();
    let mut components = Vec::new();
    while let Some(anchor) = remaining.iter().next().copied() {
        remaining.remove(&anchor);
        let mut atlas_from_node = BTreeMap::new();
        atlas_from_node.insert(anchor, Sim3::identity());
        let mut queue = VecDeque::from([anchor]);
        while let Some(current) = queue.pop_front() {
            let atlas_from_current = atlas_from_node
                .get(&current)
                .expect("BFS queue contains an assigned node")
                .clone();
            for (neighbour, neighbour_from_current) in &adjacency[current] {
                if atlas_from_node.contains_key(neighbour) {
                    continue;
                }
                let atlas_from_neighbour = atlas_from_current.compose(neighbour_from_current);
                atlas_from_node.insert(*neighbour, atlas_from_neighbour);
                remaining.remove(neighbour);
                queue.push_back(*neighbour);
            }
        }
        let node_indices = atlas_from_node.keys().copied().collect::<Vec<_>>();
        components.push(AtlasComponent {
            anchor_index: anchor,
            node_indices,
            atlas_from_node,
        });
    }
    components
}

fn healthy_node(node: &NodeData) -> bool {
    node.step.ratio.is_finite() && node.step.ratio <= HEALTHY_STEP_RATIO
}

fn select_frame_owners(
    nodes: &[NodeData],
    component: &AtlasComponent,
    policy: FrameOwnerPolicy,
) -> BTreeMap<u64, usize> {
    match policy {
        FrameOwnerPolicy::Newest => {
            let mut owners = BTreeMap::<u64, usize>::new();
            for node_index in &component.node_indices {
                let node = &nodes[*node_index];
                if !healthy_node(node) {
                    continue;
                }
                for frame_id in node.frames.keys().copied() {
                    let replace = match owners.get(&frame_id) {
                        Some(previous_index) => {
                            prefers_newer_owner(nodes, *node_index, *previous_index)
                        }
                        None => true,
                    };
                    if replace {
                        owners.insert(frame_id, *node_index);
                    }
                }
            }
            owners
        }
        FrameOwnerPolicy::Interior => {
            let mut scored_owners = BTreeMap::<u64, (usize, u64)>::new();
            for node_index in &component.node_indices {
                let node = &nodes[*node_index];
                if !healthy_node(node) {
                    continue;
                }
                let (Some(&min_frame), Some(&max_frame)) =
                    (node.frames.keys().next(), node.frames.keys().next_back())
                else {
                    continue;
                };
                for frame_id in node.frames.keys().copied() {
                    let interior_depth = (frame_id - min_frame).min(max_frame - frame_id);
                    let replace = match scored_owners.get(&frame_id) {
                        Some((previous_index, previous_depth)) => {
                            interior_depth > *previous_depth
                                || (interior_depth == *previous_depth
                                    && prefers_newer_owner(nodes, *node_index, *previous_index))
                        }
                        None => true,
                    };
                    if replace {
                        scored_owners.insert(frame_id, (*node_index, interior_depth));
                    }
                }
            }
            scored_owners
                .into_iter()
                .map(|(frame_id, (node_index, _))| (frame_id, node_index))
                .collect()
        }
    }
}

fn prefers_newer_owner(nodes: &[NodeData], candidate_index: usize, previous_index: usize) -> bool {
    let candidate = &nodes[candidate_index];
    let previous = &nodes[previous_index];
    (candidate.spec.window_start, candidate.spec.node_id)
        > (previous.spec.window_start, previous.spec.node_id)
}

const UNIT_SCALE_TOLERANCE: f64 = 1.0e-12;

fn unit_scale_se3(transform: &Sim3, context: &str) -> Result<SE3, String> {
    if !transform.scale.is_finite() || (transform.scale - 1.0).abs() > UNIT_SCALE_TOLERANCE {
        return Err(format!(
            "{context} has non-unit scale {:.17}",
            transform.scale
        ));
    }
    let rotation_norm = transform.rotation.coords.norm();
    if !transform.translation.iter().all(|value| value.is_finite())
        || !transform
            .rotation
            .coords
            .iter()
            .all(|value| value.is_finite())
        || !rotation_norm.is_finite()
        || (rotation_norm - 1.0).abs() > UNIT_SCALE_TOLERANCE
    {
        return Err(format!(
            "{context} contains an invalid rotation or non-finite transform"
        ));
    }
    Ok(SE3::new(transform.rotation, transform.translation))
}

fn se3_is_finite(transform: &SE3) -> bool {
    transform.translation.iter().all(|value| value.is_finite())
        && transform
            .rotation
            .coords
            .iter()
            .all(|value| value.is_finite())
}

// Reconcile an already-built unit-scale atlas with all accepted L seams.
//
// The initial poses are node_from_atlas = atlas_from_node.inverse(), matching
// PoseGraph's world-to-camera convention. Each accepted edge stores
// target_from_source, so it is inserted as from=source, to=target without
// inversion. This is deliberately opt-in: the historical traversal atlas
// remains byte-identical when --metric-se3 is absent.
//
// The graph is solved one connected component at a time. Candidate edges are
// already bounded by the predecessor scan; sparse factorization can still
// incur graph-dependent fill-in, so this function makes no linear-memory claim.
fn optimize_metric_se3_components(
    nodes: &[NodeData],
    accepted_edges: &[SeamEdge],
    components: &mut [AtlasComponent],
) -> Result<Vec<MetricSe3Summary>, String> {
    let config = PoseGraphSe3Config {
        robust_kernel: RobustKernel::None,
        initial_lambda: Some(1.0e-3),
        linear_solver: LinearSolver::Sparse,
        chordal_init: false,
        ..PoseGraphSe3Config::default()
    };
    let mut summaries = Vec::with_capacity(components.len());

    for (component_index, component) in components.iter_mut().enumerate() {
        let component_nodes = component
            .node_indices
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let anchor_node_id = nodes[component.anchor_index].spec.node_id;
        let mut graph = PoseGraph::new();
        let mut initial_poses = BTreeMap::<u64, SE3>::new();
        for node_index in &component.node_indices {
            let node_id = nodes[*node_index].spec.node_id;
            let atlas_from_node = component
                .atlas_from_node
                .get(node_index)
                .ok_or_else(|| format!("component {component_index} is missing node transform"))?;
            let atlas_from_node = unit_scale_se3(
                atlas_from_node,
                &format!("component {component_index} node {node_id} atlas transform"),
            )?;
            let node_from_atlas = atlas_from_node.inverse();
            if !se3_is_finite(&node_from_atlas) {
                return Err(format!(
                    "component {component_index} node {node_id} initial pose is non-finite"
                ));
            }
            graph.add_pose(
                node_id,
                Pose {
                    world_to_camera: node_from_atlas.clone(),
                },
            );
            initial_poses.insert(node_id, node_from_atlas);
        }
        graph.anchor(anchor_node_id);

        let mut edge_count = 0;
        for edge in accepted_edges.iter().filter(|edge| {
            component_nodes.contains(&edge.source_index)
                && component_nodes.contains(&edge.target_index)
        }) {
            let source_id = nodes[edge.source_index].spec.node_id;
            let target_id = nodes[edge.target_index].spec.node_id;
            let measurement = unit_scale_se3(
                &edge.target_from_source,
                &format!("component {component_index} seam {target_id}<-{source_id}"),
            )?;
            graph.edges.push(PoseGraphEdge {
                from: source_id,
                to: target_id,
                measurement,
                kind: PoseGraphEdgeKind::LoopClosure,
                weight: 1.0,
                information: None,
            });
            edge_count += 1;
        }

        if edge_count == 0 {
            summaries.push(MetricSe3Summary {
                component_index,
                node_count: component.node_indices.len(),
                edge_count,
                initial_cost: 0.0,
                final_cost: 0.0,
                iterations: 0,
                converged: true,
            });
            continue;
        }

        let initial_cost = graph.se3_cost();
        if !initial_cost.is_finite() {
            return Err(format!(
                "component {component_index} metric SE3 initial cost is non-finite"
            ));
        }
        let result = graph
            .optimize_se3_iterative(&config)
            .map_err(|error| format!("component {component_index} metric SE3 failed: {error}"))?;
        if !result.initial_cost.is_finite() || !result.final_cost.is_finite() {
            return Err(format!(
                "component {component_index} metric SE3 returned a non-finite cost"
            ));
        }
        let tolerance = 1.0e-10 * result.initial_cost.abs().max(1.0);
        if result.final_cost > result.initial_cost + tolerance {
            return Err(format!(
                "component {component_index} metric SE3 increased cost {:.17} -> {:.17}",
                result.initial_cost, result.final_cost
            ));
        }
        let mut previous_accepted_cost = result.initial_cost;
        for iteration in &result.iterations {
            if !iteration.cost_before.is_finite()
                || !iteration.cost_after.is_finite()
                || !iteration.max_step_norm.is_finite()
            {
                return Err(format!(
                    "component {component_index} metric SE3 iteration {} is non-finite",
                    iteration.iteration
                ));
            }
            if iteration.step_accepted {
                let tolerance = 1.0e-10 * iteration.cost_before.abs().max(1.0);
                if iteration.cost_after > iteration.cost_before + tolerance
                    || iteration.cost_after > previous_accepted_cost + tolerance
                {
                    return Err(format!(
                        "component {component_index} metric SE3 accepted a non-monotone step {}",
                        iteration.iteration
                    ));
                }
                previous_accepted_cost = iteration.cost_after;
            }
        }
        let final_cost = graph.se3_cost();
        if !final_cost.is_finite() || final_cost > initial_cost + tolerance {
            return Err(format!(
                "component {component_index} metric SE3 final graph cost is invalid: {final_cost:.17}"
            ));
        }

        let anchor_pose = graph
            .poses
            .get(&anchor_node_id)
            .ok_or_else(|| format!("component {component_index} lost its anchor pose"))?;
        let initial_anchor = initial_poses
            .get(&anchor_node_id)
            .expect("anchor was inserted with every component pose");
        if (anchor_pose.world_to_camera.translation - initial_anchor.translation).norm() > 1.0e-10
            || anchor_pose
                .world_to_camera
                .rotation
                .rotation_to(&initial_anchor.rotation)
                .angle()
                > 1.0e-10
        {
            return Err(format!(
                "component {component_index} metric SE3 moved its anchor"
            ));
        }

        for node_index in &component.node_indices {
            let node_id = nodes[*node_index].spec.node_id;
            let node_from_atlas = graph
                .poses
                .get(&node_id)
                .ok_or_else(|| format!("component {component_index} lost node {node_id}"))?
                .world_to_camera
                .clone();
            if !se3_is_finite(&node_from_atlas) {
                return Err(format!(
                    "component {component_index} optimized node {node_id} pose is non-finite"
                ));
            }
            let atlas_from_node = node_from_atlas.inverse();
            if !se3_is_finite(&atlas_from_node) {
                return Err(format!(
                    "component {component_index} optimized node {node_id} atlas pose is non-finite"
                ));
            }
            component.atlas_from_node.insert(
                *node_index,
                Sim3::new(atlas_from_node.rotation, atlas_from_node.translation, 1.0),
            );
        }
        summaries.push(MetricSe3Summary {
            component_index,
            node_count: component.node_indices.len(),
            edge_count,
            initial_cost: result.initial_cost,
            final_cost,
            iterations: result.iterations.len(),
            converged: result.converged,
        });
    }
    Ok(summaries)
}

fn transform_pose_to_atlas(atlas_from_node: &Sim3, pose: &Pose) -> Result<Pose, String> {
    if !atlas_from_node.scale.is_finite() || atlas_from_node.scale <= 0.0 {
        return Err("atlas Sim3 has a non-positive or non-finite scale".to_owned());
    }
    if !atlas_from_node
        .translation
        .iter()
        .all(|value| value.is_finite())
        || !atlas_from_node
            .rotation
            .coords
            .iter()
            .all(|value| value.is_finite())
    {
        return Err("atlas Sim3 contains a non-finite value".to_owned());
    }
    let centre_atlas = atlas_from_node.transform_point(&pose.camera_center_world());
    let rotation = pose.world_to_camera.rotation * atlas_from_node.rotation.inverse();
    let translation = -rotation.transform_vector(&centre_atlas.coords);
    let transformed = Pose::from_world_to_camera(rotation, translation);
    validate_pose(&transformed, "transformed component pose")?;
    Ok(transformed)
}

fn compose_camera_pose_from_rig(
    sensor_from_rig: &SE3,
    rig_world_to_camera: &Pose,
) -> Result<Pose, String> {
    let camera_pose = Pose {
        world_to_camera: sensor_from_rig.compose(&rig_world_to_camera.world_to_camera),
    };
    validate_pose(&camera_pose, "recomposed atlas camera pose")?;
    Ok(camera_pose)
}

fn publish_component(
    component_index: usize,
    nodes: &[NodeData],
    component: &AtlasComponent,
    out_dir: &Path,
    output_image_ids: &mut BTreeSet<u64>,
    owner_policy: FrameOwnerPolicy,
) -> Result<ComponentSummary, String> {
    let owners = select_frame_owners(nodes, component, owner_policy);
    let mut output = String::from(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as X Y POINT3D_ID\n",
    );
    let mut image_count = 0;
    for (frame_id, node_index) in &owners {
        let node = &nodes[*node_index];
        let atlas_from_node = component
            .atlas_from_node
            .get(node_index)
            .expect("component owner has a BFS transform");
        let frame = node
            .frames
            .get(frame_id)
            .expect("owner frame came from the node frame map");
        let atlas_rig_pose = transform_pose_to_atlas(atlas_from_node, &frame.pose)?;
        for image in &frame.images {
            if image.global_image_id == 0 {
                return Err(format!(
                    "component {component_index} image {:?} has no manifest global image id",
                    image.name
                ));
            }
            if !output_image_ids.insert(image.global_image_id) {
                return Err(format!(
                    "component {component_index} duplicates global image id {}",
                    image.global_image_id
                ));
            }
            let sensor_from_rig = image.sensor_from_rig.as_ref().ok_or_else(|| {
                format!(
                    "component {component_index} image {:?} has no bound sensor calibration",
                    image.name
                )
            })?;
            let pose = compose_camera_pose_from_rig(sensor_from_rig, &atlas_rig_pose)?;
            let quaternion = pose.world_to_camera.rotation.quaternion();
            let translation = pose.world_to_camera.translation;
            output.push_str(&format!(
                "{} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {:.15} {} {}\n\n",
                image.global_image_id,
                quaternion.w,
                quaternion.i,
                quaternion.j,
                quaternion.k,
                translation.x,
                translation.y,
                translation.z,
                image.camera_id,
                image.name
            ));
            image_count += 1;
        }
    }
    let component_dir = out_dir.join(format!("component-{component_index:03}"));
    std::fs::create_dir_all(&component_dir).map_err(|error| {
        format!(
            "create component output directory {}: {error}",
            component_dir.display()
        )
    })?;
    let images_path = component_dir.join("images.txt");
    std::fs::write(&images_path, output)
        .map_err(|error| format!("write {}: {error}", images_path.display()))?;

    let node_ids = component
        .node_indices
        .iter()
        .map(|index| nodes[*index].spec.node_id)
        .collect::<Vec<_>>();
    Ok(ComponentSummary {
        component_index,
        anchor_node_id: nodes[component.anchor_index].spec.node_id,
        node_ids,
        frame_count: owners.len(),
        image_count,
        min_frame_id: owners.keys().next().copied(),
        max_frame_id: owners.keys().next_back().copied(),
    })
}

#[cfg(test)]
fn stitch_nodes(nodes: &[NodeData], out_dir: &Path) -> Result<StitchResult, String> {
    stitch_nodes_with_config(nodes, out_dir, &seam_alignment_config())
}

#[cfg(test)]
fn stitch_nodes_with_config(
    nodes: &[NodeData],
    out_dir: &Path,
    config: &RigSubmapAlignmentConfig,
) -> Result<StitchResult, String> {
    stitch_nodes_with_policy(
        nodes,
        out_dir,
        config,
        ForestPolicy::Traversal,
        FrameOwnerPolicy::Newest,
        false,
    )
}

fn stitch_nodes_with_policy(
    nodes: &[NodeData],
    out_dir: &Path,
    config: &RigSubmapAlignmentConfig,
    forest_policy: ForestPolicy,
    owner_policy: FrameOwnerPolicy,
    metric_se3: bool,
) -> Result<StitchResult, String> {
    if nodes.is_empty() {
        return Err("cannot stitch an empty node set".to_owned());
    }
    if metric_se3 && forest_policy == ForestPolicy::Quality {
        return Err("--metric-se3 requires --forest-policy traversal".to_owned());
    }
    let (accepted_edges, mut seam_diagnostics) = build_seam_graph_with_config(nodes, config);
    let transform_edges = match forest_policy {
        ForestPolicy::Traversal => accepted_edges.clone(),
        ForestPolicy::Quality => select_quality_forest(nodes, &accepted_edges),
    };
    let selected_keys = transform_edges
        .iter()
        .map(|edge| {
            (
                nodes[edge.source_index].spec.node_id,
                nodes[edge.target_index].spec.node_id,
                edge.start_gap,
            )
        })
        .collect::<BTreeSet<_>>();
    for diagnostic in &mut seam_diagnostics {
        diagnostic.selected_for_transform_graph = diagnostic.accepted
            && selected_keys.contains(&(
                diagnostic.source_node_id,
                diagnostic.target_node_id,
                diagnostic.start_gap,
            ));
    }
    let mut components = connected_components(nodes.len(), &transform_edges);
    components.sort_by_key(|component| {
        let min_frame_id = component
            .node_indices
            .iter()
            .flat_map(|index| nodes[*index].frames.keys().copied())
            .min()
            .unwrap_or(u64::MAX);
        (
            min_frame_id,
            component
                .node_indices
                .first()
                .copied()
                .unwrap_or(usize::MAX),
        )
    });
    let metric_se3_summaries = if metric_se3 {
        optimize_metric_se3_components(nodes, &accepted_edges, &mut components)?
    } else {
        Vec::new()
    };
    std::fs::create_dir_all(out_dir)
        .map_err(|error| format!("create output directory {}: {error}", out_dir.display()))?;
    let mut output_image_ids = BTreeSet::new();
    let summaries = components
        .iter()
        .enumerate()
        .map(|(component_index, component)| {
            publish_component(
                component_index,
                nodes,
                component,
                out_dir,
                &mut output_image_ids,
                owner_policy,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StitchResult {
        seam_diagnostics,
        components: summaries,
        metric_se3: metric_se3_summaries,
    })
}

fn main() {
    let args = match parse_args(std::env::args()) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            usage();
        }
    };
    let manifest = parse_rig_manifest(&args.rig_manifest).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(1);
    });
    let nodes = parse_node_specs(&args.nodes_tsv).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(1);
    });
    println!(
        "parsed rig_manifest={} nodes={} out_dir={} seam_config_arm={} frame_owner_policy={} forest_policy={}",
        args.rig_manifest.display(),
        nodes.len(),
        args.out_dir.display(),
        args.seam_config_arm.as_str(),
        args.frame_owner_policy.as_str(),
        args.forest_policy.as_str()
    );
    if args.metric_se3 {
        println!("metric_se3=enabled");
    }
    let node_data = nodes
        .into_iter()
        .map(|spec| extract_node_data(spec, &manifest))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|error| {
            eprintln!("error: {error}");
            std::process::exit(1);
        });
    for node in &node_data {
        println!(
            "node id={} window_start={} images={} frames={} step_median_m={:.9} step_max_m={:.9} step_ratio={:.6}",
            node.spec.node_id,
            node.spec.window_start,
            node.images.len(),
            node.frames.len(),
            node.step.median_m,
            node.step.max_m,
            node.step.ratio
        );
    }
    let seam_config = seam_alignment_config_for_arm(args.seam_config_arm);
    let stitched = stitch_nodes_with_policy(
        &node_data,
        &args.out_dir,
        &seam_config,
        args.forest_policy,
        args.frame_owner_policy,
        args.metric_se3,
    )
    .unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(1);
    });
    for diagnostic in &stitched.seam_diagnostics {
        println!(
            "seam source={} target={} start_gap={} verdict={} selected_for_transform_graph={} common_frames={} retained_frames={} reason={} method={:?} constraint_scale={:?} constraint_rotation_angle_deg={:?} constraint_translation_norm={:?} constraint_inlier_ratio={:?} constraint_mean_residual_ratio={:?} constraint_leave_one_out_log_scale_mad={:?} wrapper_reason={:?} primary_rejection_reason={:?} primary_rejection_inlier_count={:?} primary_rejection_inlier_ratio={:?} primary_rejection_mean_residual_ratio={:?} primary_rejection_rotation_disagreement_deg={:?} primary_rejection_leave_one_out_log_scale_mad={:?} fallback_attempted={} fallback_used={} fallback_rejection_reason={:?} fallback_rejection_inlier_count={:?} fallback_rejection_inlier_ratio={:?} fallback_rejection_mean_residual_ratio={:?} fallback_rejection_rotation_disagreement_deg={:?} fallback_rejection_leave_one_out_log_scale_mad={:?}",
            diagnostic.source_node_id,
            diagnostic.target_node_id,
            diagnostic.start_gap,
            if diagnostic.accepted {
                "accepted"
            } else {
                "rejected"
            },
            diagnostic.selected_for_transform_graph,
            diagnostic.common_frames,
            diagnostic.retained_frames,
            diagnostic.reason,
            diagnostic.alignment_method,
            diagnostic.constraint_scale,
            diagnostic.constraint_rotation_angle_deg,
            diagnostic.constraint_translation_norm,
            diagnostic.constraint_inlier_ratio,
            diagnostic.constraint_mean_residual_ratio,
            diagnostic.constraint_leave_one_out_log_scale_mad,
            diagnostic.wrapper_rejection_reason,
            diagnostic.sim3_rejection_reason,
            diagnostic.sim3_rejection_inlier_count,
            diagnostic.sim3_rejection_inlier_ratio,
            diagnostic.sim3_rejection_mean_residual_ratio,
            diagnostic.sim3_rejection_rotation_disagreement_deg,
            diagnostic.sim3_rejection_leave_one_out_log_scale_mad,
            diagnostic.fallback_attempted,
            diagnostic.fallback_used,
            diagnostic.fallback_rejection_reason,
            diagnostic.fallback_rejection_inlier_count,
            diagnostic.fallback_rejection_inlier_ratio,
            diagnostic.fallback_rejection_mean_residual_ratio,
            diagnostic.fallback_rejection_rotation_disagreement_deg,
            diagnostic.fallback_rejection_leave_one_out_log_scale_mad
        );
    }
    let accepted_count = stitched
        .seam_diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.accepted)
        .count();
    let rejected_count = stitched
        .seam_diagnostics
        .iter()
        .filter(|diagnostic| !diagnostic.accepted)
        .count();
    println!(
        "summary accepted_seams={} rejected_seams={} components={}",
        accepted_count,
        rejected_count,
        stitched.components.len()
    );
    for component in &stitched.components {
        let frame_range = match (component.min_frame_id, component.max_frame_id) {
            (Some(min_frame_id), Some(max_frame_id)) => {
                format!("{min_frame_id}..{max_frame_id}")
            }
            _ => "empty".to_owned(),
        };
        println!(
            "component index={} anchor_node={} nodes={:?} frames={} images={} frame_range={}",
            component.component_index,
            component.anchor_node_id,
            component.node_ids,
            component.frame_count,
            component.image_count,
            frame_range
        );
    }
    for metric in &stitched.metric_se3 {
        println!(
            "metric_se3 component={} nodes={} edges={} initial_cost={:.12} final_cost={:.12} iterations={} converged={}",
            metric.component_index,
            metric.node_count,
            metric.edge_count,
            metric.initial_cost,
            metric.final_cost,
            metric.iterations,
            metric.converged
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;
    use std::fs;

    fn fixture_root(test_name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "visloc_stitch_rig_submap_{test_name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_fixture(root: &Path, contents: &str) -> PathBuf {
        let images = root.join("model").join("images.txt");
        fs::create_dir_all(images.parent().unwrap()).unwrap();
        fs::write(&images, "# empty fixture\n").unwrap();
        let nodes = root.join("nodes.tsv");
        fs::write(&nodes, contents).unwrap();
        nodes
    }

    #[test]
    fn parses_required_arguments() {
        let args = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(args.rig_manifest, PathBuf::from("rig.txt"));
        assert_eq!(args.nodes_tsv, PathBuf::from("nodes.tsv"));
        assert_eq!(args.out_dir, PathBuf::from("out"));
        assert_eq!(args.seam_config_arm, SeamConfigArm::A);
        assert_eq!(args.frame_owner_policy, FrameOwnerPolicy::Newest);
        assert_eq!(args.forest_policy, ForestPolicy::Traversal);
        assert!(!args.metric_se3);
    }

    #[test]
    fn parses_frame_owner_policy_and_rejects_unknown_or_duplicate_values() {
        assert_eq!(
            FrameOwnerPolicy::parse("newest"),
            Ok(FrameOwnerPolicy::Newest)
        );
        assert_eq!(
            FrameOwnerPolicy::parse("interior"),
            Ok(FrameOwnerPolicy::Interior)
        );
        assert!(FrameOwnerPolicy::parse("other")
            .unwrap_err()
            .contains("expects newest or interior"));

        let args = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--frame-owner-policy",
                "interior",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(args.frame_owner_policy, FrameOwnerPolicy::Interior);
        assert!(USAGE.contains("[--frame-owner-policy newest|interior]"));

        let unknown = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--frame-owner-policy",
                "other",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(unknown.contains("expects newest or interior"));

        let duplicate = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--frame-owner-policy",
                "newest",
                "--frame-owner-policy",
                "interior",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(duplicate.contains("duplicate argument --frame-owner-policy"));
    }

    #[test]
    fn parses_quality_forest_policy_and_rejects_unknown_values() {
        assert_eq!(
            ForestPolicy::parse("traversal"),
            Ok(ForestPolicy::Traversal)
        );
        assert_eq!(ForestPolicy::parse("quality"), Ok(ForestPolicy::Quality));
        assert!(ForestPolicy::parse("other")
            .unwrap_err()
            .contains("expects traversal or quality"));

        let args = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--forest-policy",
                "quality",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(args.forest_policy, ForestPolicy::Quality);

        let duplicate = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--forest-policy",
                "traversal",
                "--forest-policy",
                "quality",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(duplicate.contains("duplicate argument --forest-policy"));
    }

    #[test]
    fn parses_metric_se3_opt_in_and_rejects_duplicates() {
        let args = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--metric-se3",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert!(args.metric_se3);
        assert!(USAGE.contains("[--metric-se3]"));

        let duplicate = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--metric-se3",
                "--metric-se3",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(duplicate.contains("duplicate argument --metric-se3"));
    }

    #[test]
    fn parses_and_applies_selected_seam_config_arm() {
        let args = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--seam-config-arm",
                "D",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(args.seam_config_arm, SeamConfigArm::D);
        assert_eq!(SeamConfigArm::parse("E"), Ok(SeamConfigArm::E));
        let config = seam_alignment_config_for_arm(args.seam_config_arm);
        assert_eq!(config.max_boundary_frames, 64);
        assert_eq!(config.alignment.min_correspondences, 8);
        assert_eq!(config.alignment.min_inliers, 8);
        assert_eq!(config.rotation_consensus_deg, 5.0);
        assert_eq!(config.min_rotation_consensus_ratio, 0.6);
        assert_eq!(config.alignment.min_second_to_first_singular_ratio, 0.0);
        assert_eq!(config.alignment.max_inlier_residual_ratio, 0.10);
        assert_eq!(config.alignment.max_mean_residual_ratio, 0.05);
        assert_eq!(config.alignment.min_inlier_ratio, 0.5);

        let e_config = seam_alignment_config_for_arm(SeamConfigArm::E);
        assert_eq!(e_config.max_boundary_frames, 64);
        assert_eq!(e_config.alignment.min_correspondences, 8);
        assert_eq!(e_config.alignment.min_inliers, 8);
        assert_eq!(e_config.rotation_consensus_deg, 60.0);
        assert_eq!(e_config.min_rotation_consensus_ratio, 0.35);
        assert_eq!(e_config.alignment.min_second_to_first_singular_ratio, 0.0);
        assert_eq!(e_config.alignment.max_inlier_residual_ratio, 0.10);
        assert_eq!(e_config.alignment.max_mean_residual_ratio, 0.06);
        assert_eq!(e_config.alignment.min_inlier_ratio, 0.35);
        assert_eq!(e_config.alignment.max_rotation_disagreement_deg, 60.0);
        assert!(!e_config.allow_fixed_rotation_fallback);
        assert!(!e_config.prefer_fixed_rotation_alignment);
        assert!(!e_config.allow_generic_fallback_after_fixed_rotation);

        assert_eq!(SeamConfigArm::parse("F"), Ok(SeamConfigArm::F));
        let f_config = seam_alignment_config_for_arm(SeamConfigArm::F);
        assert_eq!(f_config.rotation_consensus_deg, 60.0);
        assert_eq!(f_config.min_rotation_consensus_ratio, 0.35);
        assert_eq!(f_config.alignment.min_second_to_first_singular_ratio, 0.0);
        assert_eq!(f_config.alignment.max_inlier_residual_ratio, 0.10);
        assert_eq!(f_config.alignment.max_mean_residual_ratio, 0.06);
        assert_eq!(f_config.alignment.min_inlier_ratio, 0.35);
        assert_eq!(f_config.alignment.max_rotation_disagreement_deg, 60.0);
        assert!(f_config.allow_fixed_rotation_fallback);
        assert!(!f_config.prefer_fixed_rotation_alignment);
        assert!(!f_config.allow_generic_fallback_after_fixed_rotation);

        assert_eq!(SeamConfigArm::parse("G"), Ok(SeamConfigArm::G));
        let g_config = seam_alignment_config_for_arm(SeamConfigArm::G);
        assert_eq!(g_config.rotation_consensus_deg, 60.0);
        assert_eq!(g_config.min_rotation_consensus_ratio, 0.35);
        assert_eq!(g_config.alignment.min_second_to_first_singular_ratio, 0.0);
        assert_eq!(g_config.alignment.max_inlier_residual_ratio, 0.10);
        assert_eq!(g_config.alignment.max_mean_residual_ratio, 0.06);
        assert_eq!(g_config.alignment.min_inlier_ratio, 0.35);
        assert_eq!(g_config.alignment.max_rotation_disagreement_deg, 60.0);
        assert!(!g_config.allow_fixed_rotation_fallback);
        assert!(g_config.prefer_fixed_rotation_alignment);
        assert!(!g_config.allow_generic_fallback_after_fixed_rotation);

        assert_eq!(SeamConfigArm::parse("H"), Ok(SeamConfigArm::H));
        let h_config = seam_alignment_config_for_arm(SeamConfigArm::H);
        assert_eq!(h_config.rotation_consensus_deg, 60.0);
        assert_eq!(h_config.min_rotation_consensus_ratio, 0.35);
        assert_eq!(h_config.alignment.min_second_to_first_singular_ratio, 0.0);
        assert_eq!(h_config.alignment.max_inlier_residual_ratio, 0.10);
        assert_eq!(h_config.alignment.max_mean_residual_ratio, 0.06);
        assert_eq!(h_config.alignment.min_inlier_ratio, 0.35);
        assert_eq!(h_config.alignment.max_rotation_disagreement_deg, 60.0);
        assert!(!h_config.allow_fixed_rotation_fallback);
        assert!(h_config.prefer_fixed_rotation_alignment);
        assert!(h_config.allow_generic_fallback_after_fixed_rotation);

        assert_eq!(SeamConfigArm::parse("I"), Ok(SeamConfigArm::I));
        let i_config = seam_alignment_config_for_arm(SeamConfigArm::I);
        assert_eq!(
            i_config.boundary_sampling,
            RigSubmapBoundarySampling::UniformAcrossOverlap
        );
        assert_eq!(i_config.rotation_consensus_deg, 60.0);
        assert_eq!(i_config.min_rotation_consensus_ratio, 0.35);
        assert_eq!(i_config.alignment.min_second_to_first_singular_ratio, 0.0);
        assert_eq!(i_config.alignment.max_inlier_residual_ratio, 0.10);
        assert_eq!(i_config.alignment.max_mean_residual_ratio, 0.06);
        assert_eq!(i_config.alignment.min_inlier_ratio, 0.35);
        assert_eq!(i_config.alignment.max_rotation_disagreement_deg, 60.0);
        assert!(!i_config.allow_fixed_rotation_fallback);
        assert!(i_config.prefer_fixed_rotation_alignment);
        assert!(i_config.allow_generic_fallback_after_fixed_rotation);

        assert_eq!(SeamConfigArm::parse("J"), Ok(SeamConfigArm::J));
        let j_config = seam_alignment_config_for_arm(SeamConfigArm::J);
        assert_eq!(
            j_config.boundary_sampling,
            RigSubmapBoundarySampling::Earliest
        );
        assert_eq!(j_config.rotation_consensus_deg, 60.0);
        assert_eq!(j_config.min_rotation_consensus_ratio, 0.35);
        assert_eq!(j_config.alignment.min_second_to_first_singular_ratio, 0.0);
        assert_eq!(j_config.alignment.max_inlier_residual_ratio, 0.10);
        assert_eq!(j_config.alignment.max_mean_residual_ratio, 0.07);
        assert_eq!(j_config.alignment.min_inlier_ratio, 0.35);
        assert_eq!(j_config.alignment.max_rotation_disagreement_deg, 60.0);
        assert!(!j_config.allow_fixed_rotation_fallback);
        assert!(j_config.prefer_fixed_rotation_alignment);
        assert!(j_config.allow_generic_fallback_after_fixed_rotation);

        assert_eq!(SeamConfigArm::parse("K"), Ok(SeamConfigArm::K));
        let k_config = seam_alignment_config_for_arm(SeamConfigArm::K);
        assert_eq!(
            k_config.boundary_sampling,
            RigSubmapBoundarySampling::Earliest
        );
        assert_eq!(k_config.rotation_consensus_deg, 60.0);
        assert_eq!(k_config.min_rotation_consensus_ratio, 0.35);
        assert_eq!(k_config.alignment.min_second_to_first_singular_ratio, 0.0);
        assert_eq!(k_config.alignment.max_inlier_residual_ratio, 0.10);
        assert_eq!(k_config.alignment.max_mean_residual_ratio, 0.07);
        assert_eq!(k_config.alignment.min_inlier_ratio, 0.35);
        assert_eq!(k_config.alignment.max_rotation_disagreement_deg, 60.0);
        assert!(!k_config.allow_fixed_rotation_fallback);
        assert!(k_config.prefer_fixed_rotation_alignment);
        assert!(k_config.allow_generic_fallback_after_fixed_rotation);
        assert!(k_config.force_fixed_rotation_unit_scale);

        assert_eq!(SeamConfigArm::parse("L"), Ok(SeamConfigArm::L));
        let l_config = seam_alignment_config_for_arm(SeamConfigArm::L);
        assert_eq!(
            l_config.boundary_sampling,
            RigSubmapBoundarySampling::Earliest
        );
        assert_eq!(
            l_config.rotation_consensus_deg,
            k_config.rotation_consensus_deg
        );
        assert_eq!(
            l_config.min_rotation_consensus_ratio,
            k_config.min_rotation_consensus_ratio
        );
        assert_eq!(l_config.alignment, k_config.alignment);
        assert!(!l_config.allow_fixed_rotation_fallback);
        assert!(l_config.prefer_fixed_rotation_alignment);
        assert!(!l_config.allow_generic_fallback_after_fixed_rotation);
        assert!(l_config.force_fixed_rotation_unit_scale);
        assert!(USAGE.contains("[--seam-config-arm A|B|C|D|E|F|G|H|I|J|K|L]"));
    }

    #[test]
    fn rejects_unknown_or_duplicate_seam_config_arm() {
        let unknown = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--seam-config-arm",
                "Z",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(unknown.contains("expects A, B, C, D, E, F, G, H, I, J, K, or L"));

        let duplicate = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
                "--out-dir",
                "out",
                "--seam-config-arm",
                "B",
                "--seam-config-arm",
                "C",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(duplicate.contains("duplicate argument --seam-config-arm"));
    }

    #[test]
    fn rejects_missing_required_argument() {
        let error = parse_args(
            [
                "stitch_rig_submap_trajectories",
                "--rig-manifest",
                "rig.txt",
                "--nodes-tsv",
                "nodes.tsv",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(error.contains("--out-dir is required"));
    }

    #[test]
    fn parses_and_sorts_nodes_by_start_then_id() {
        let root = fixture_root("sort");
        let image_a = root.join("a-images.txt");
        let image_b = root.join("b-images.txt");
        fs::write(&image_a, "# a\n").unwrap();
        fs::write(&image_b, "# b\n").unwrap();
        let nodes_path = root.join("nodes.tsv");
        fs::write(
            &nodes_path,
            format!(
                "# node_id\twindow_start\timages_txt\n2\t250\t{}\n1\t250\t{}\n",
                image_b.display(),
                image_a.display()
            ),
        )
        .unwrap();
        let nodes = parse_node_specs(&nodes_path).unwrap();
        assert_eq!(
            nodes,
            vec![
                NodeSpec {
                    node_id: 1,
                    window_start: 250,
                    images_txt: image_a,
                },
                NodeSpec {
                    node_id: 2,
                    window_start: 250,
                    images_txt: image_b,
                },
            ]
        );
    }

    #[test]
    fn rejects_duplicate_node_ids() {
        let root = fixture_root("duplicate-id");
        let image_a = root.join("a-images.txt");
        let image_b = root.join("b-images.txt");
        fs::write(&image_a, "# a\n").unwrap();
        fs::write(&image_b, "# b\n").unwrap();
        let nodes_path = root.join("nodes.tsv");
        fs::write(
            &nodes_path,
            format!(
                "1\t0\t{}\n1\t250\t{}\n",
                image_a.display(),
                image_b.display()
            ),
        )
        .unwrap();
        let error = parse_node_specs(&nodes_path).unwrap_err();
        assert!(error.contains("duplicates node id 1"));
    }

    #[test]
    fn rejects_duplicate_start_and_path() {
        let root = fixture_root("duplicate-start-path");
        let image = root.join("images.txt");
        fs::write(&image, "# image\n").unwrap();
        let nodes_path = root.join("nodes.tsv");
        fs::write(
            &nodes_path,
            format!("1\t250\t{}\n2\t250\t{}\n", image.display(), image.display()),
        )
        .unwrap();
        let error = parse_node_specs(&nodes_path).unwrap_err();
        assert!(error.contains("duplicates window_start/images_txt"));
    }

    #[test]
    fn rejects_missing_images_file() {
        let root = fixture_root("missing-file");
        let nodes_path = write_fixture(&root, "1\t0\tmissing/images.txt\n");
        let error = parse_node_specs(&nodes_path).unwrap_err();
        assert!(error.contains("images.txt does not exist"));
    }

    #[test]
    fn rejects_malformed_node_rows() {
        let root = fixture_root("malformed");
        let nodes_path = write_fixture(&root, "1\t0\n");
        let error = parse_node_specs(&nodes_path).unwrap_err();
        assert!(error.contains("requires node_id<TAB>window_start<TAB>images_txt"));
    }

    fn manifest_text(rows: &str) -> String {
        format!(
            "# generalized-rig-manifest-v1\n\
             # S index camera_id width height fx fy cx cy qw qx qy qz tx ty tz\n\
             S 0 1 848 800 284.0 286.0 425.0 399.0 1 0 0 0 0 0 0\n\
             S 1 2 848 800 284.0 286.0 425.0 399.0 1 0 0 0 -0.1 0 0\n\
             {rows}"
        )
    }

    fn colmap_pose(
        image_id: u64,
        camera_id: u64,
        name: &str,
        translation_x: f64,
        points: &str,
    ) -> String {
        format!("{image_id} 1 0 0 0 {translation_x} 0 0 {camera_id} {name}\n{points}\n")
    }

    #[test]
    fn parses_rig_manifest_sensors_and_assignments() {
        let root = fixture_root("rig-manifest");
        let path = root.join("rig.txt");
        fs::write(&path, manifest_text("F 0 cam1.png 0\nF 0 cam2.png 1\n")).unwrap();
        let manifest = parse_rig_manifest(&path).unwrap();
        assert_eq!(manifest.sensors.len(), 2);
        assert_eq!(manifest.sensors[&1].camera_id, 2);
        assert_eq!(manifest.assignments["cam1.png"].global_image_id, 1);
        assert_eq!(
            manifest.assignments["cam2.png"],
            FrameImageAssignment {
                frame_id: 0,
                sensor_index: 1,
                global_image_id: 2,
            }
        );
        assert!((manifest.sensors[&1].sensor_from_rig.translation.x + 0.1).abs() < 1e-12);
    }

    #[test]
    fn rejects_unknown_sensor_and_duplicate_frame_sensor() {
        let root = fixture_root("rig-manifest-errors");
        let unknown = root.join("unknown.txt");
        fs::write(&unknown, manifest_text("F 0 cam1.png 2\n")).unwrap();
        let error = parse_rig_manifest(&unknown).unwrap_err();
        assert!(error.contains("unknown sensor 2"));

        let duplicate = root.join("duplicate.txt");
        fs::write(
            &duplicate,
            manifest_text("F 0 cam1.png 0\nF 0 cam1-other.png 0\n"),
        )
        .unwrap();
        let error = parse_rig_manifest(&duplicate).unwrap_err();
        assert!(error.contains("duplicates frame 0 sensor 0"));
    }

    #[test]
    fn parses_empty_and_nonempty_points2d_rows() {
        let root = fixture_root("colmap-points");
        let path = root.join("images.txt");
        fs::write(
            &path,
            format!(
                "{}{}",
                colmap_pose(1, 1, "cam1.png", 0.0, ""),
                colmap_pose(2, 2, "cam2.png", -0.1, "10 20 -1 30 40 7")
            ),
        )
        .unwrap();
        let images = parse_colmap_images(&path).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].name, "cam1.png");
        assert_eq!(images[1].name, "cam2.png");
        assert_eq!(images[1].image_id, 2);
    }

    #[test]
    fn rejects_duplicate_and_nonfinite_colmap_poses() {
        let root = fixture_root("colmap-errors");
        let duplicate = root.join("duplicate.txt");
        fs::write(
            &duplicate,
            format!(
                "{}{}",
                colmap_pose(1, 1, "cam1.png", 0.0, ""),
                colmap_pose(1, 1, "cam1-other.png", 0.0, "")
            ),
        )
        .unwrap();
        let error = parse_colmap_images(&duplicate).unwrap_err();
        assert!(error.contains("duplicates image id 1"));

        let nonfinite = root.join("nonfinite.txt");
        fs::write(&nonfinite, colmap_pose(1, 1, "cam1.png", f64::NAN, "")).unwrap();
        let error = parse_colmap_images(&nonfinite).unwrap_err();
        assert!(error.contains("non-finite"));
    }

    #[test]
    fn extracts_one_consistent_rig_pose_per_frame() {
        let root = fixture_root("node-data");
        let manifest_path = root.join("rig.txt");
        fs::write(
            &manifest_path,
            manifest_text("F 0 cam1.png 0\nF 0 cam2.png 1\nF 1 cam1b.png 0\nF 1 cam2b.png 1\n"),
        )
        .unwrap();
        let images_path = root.join("images.txt");
        fs::write(
            &images_path,
            format!(
                "{}{}{}{}",
                colmap_pose(1, 1, "cam1.png", 0.0, ""),
                colmap_pose(2, 2, "cam2.png", -0.1, ""),
                colmap_pose(3, 1, "cam1b.png", -1.0, ""),
                colmap_pose(4, 2, "cam2b.png", -1.1, "")
            ),
        )
        .unwrap();
        let manifest = parse_rig_manifest(&manifest_path).unwrap();
        let node = extract_node_data(
            NodeSpec {
                node_id: 7,
                window_start: 0,
                images_txt: images_path,
            },
            &manifest,
        )
        .unwrap();
        assert_eq!(node.frames.len(), 2);
        assert_eq!(node.frames[&0].images.len(), 2);
        assert_eq!(node.frames[&1].images.len(), 2);
        assert!((node.frames[&0].pose.camera_center_world().x - 0.0).abs() < 1e-12);
        assert!((node.frames[&1].pose.camera_center_world().x - 1.0).abs() < 1e-12);
        assert!((node.step.median_m - 1.0).abs() < 1e-12);
        assert!((node.step.max_m - 1.0).abs() < 1e-12);
        assert!((node.step.ratio - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_inconsistent_sensor_poses_and_unknown_images() {
        let root = fixture_root("node-errors");
        let manifest_path = root.join("rig.txt");
        fs::write(
            &manifest_path,
            manifest_text("F 0 cam1.png 0\nF 0 cam2.png 1\n"),
        )
        .unwrap();
        let manifest = parse_rig_manifest(&manifest_path).unwrap();

        let inconsistent_path = root.join("inconsistent.txt");
        fs::write(
            &inconsistent_path,
            format!(
                "{}{}",
                colmap_pose(1, 1, "cam1.png", 0.0, ""),
                colmap_pose(2, 2, "cam2.png", -0.2, "")
            ),
        )
        .unwrap();
        let error = extract_node_data(
            NodeSpec {
                node_id: 1,
                window_start: 0,
                images_txt: inconsistent_path,
            },
            &manifest,
        )
        .unwrap_err();
        assert!(error.contains("sensor poses disagree"));

        let unknown_path = root.join("unknown.txt");
        fs::write(
            &unknown_path,
            colmap_pose(1, 1, "not-in-manifest.png", 0.0, ""),
        )
        .unwrap();
        let error = extract_node_data(
            NodeSpec {
                node_id: 2,
                window_start: 0,
                images_txt: unknown_path,
            },
            &manifest,
        )
        .unwrap_err();
        assert!(error.contains("absent from rig manifest"));
    }

    fn identity_camera_pose(centre: Point3<f64>) -> Pose {
        Pose::from_world_to_camera(UnitQuaternion::identity(), -centre.coords)
    }

    fn trajectory_centres(last_frame_id: u64) -> BTreeMap<u64, Point3<f64>> {
        (0..=last_frame_id)
            .map(|frame_id| {
                let index = frame_id as f64;
                (
                    frame_id,
                    Point3::new(
                        index * 0.35,
                        ((frame_id * 13) % 17) as f64 * 0.23,
                        ((frame_id * 7) % 11) as f64 * 0.31 + index * 0.02,
                    ),
                )
            })
            .collect()
    }

    fn synthetic_node(
        node_id: u64,
        window_start: u64,
        frame_ids: impl IntoIterator<Item = u64>,
        atlas_from_node: &Sim3,
        centres: &BTreeMap<u64, Point3<f64>>,
        step: Option<StepStats>,
    ) -> NodeData {
        let local_from_atlas = atlas_from_node.inverse();
        let mut frames = BTreeMap::new();
        let mut images = Vec::new();
        for frame_id in frame_ids {
            let local_centre = local_from_atlas.transform_point(&centres[&frame_id]);
            let pose = identity_camera_pose(local_centre);
            let image = ColmapImage {
                // Each node owns a local COLMAP model, so local image ids can
                // legitimately repeat across windows.
                image_id: frame_id,
                global_image_id: node_id * 100_000 + frame_id + 1,
                camera_id: 1,
                name: format!("node{node_id}_frame{frame_id}.png"),
                world_to_camera: pose.clone(),
                sensor_from_rig: Some(SE3::identity()),
            };
            frames.insert(
                frame_id,
                FrameData {
                    pose,
                    images: vec![image.clone()],
                },
            );
            images.push(image);
        }
        let step = step.unwrap_or_else(|| compute_step_stats(&frames));
        NodeData {
            spec: NodeSpec {
                node_id,
                window_start,
                images_txt: PathBuf::from(format!("synthetic-{node_id}.txt")),
            },
            images,
            frames,
            step,
        }
    }

    fn synthetic_stereo_node(
        node_id: u64,
        window_start: u64,
        frame_ids: impl IntoIterator<Item = u64>,
        centres: &BTreeMap<u64, Point3<f64>>,
        sensor_a: &SE3,
        sensor_b: &SE3,
    ) -> NodeData {
        let mut frames = BTreeMap::new();
        let mut images = Vec::new();
        for frame_id in frame_ids {
            let rig_pose = identity_camera_pose(centres[&frame_id]);
            let mut frame_images = Vec::new();
            for (sensor_index, sensor_from_rig) in [sensor_a, sensor_b].iter().enumerate() {
                let image = ColmapImage {
                    image_id: frame_id * 2 + sensor_index as u64 + 1,
                    global_image_id: node_id * 100_000 + frame_id * 2 + sensor_index as u64 + 1,
                    camera_id: sensor_index as u64 + 1,
                    name: format!("node{node_id}_frame{frame_id}_sensor{sensor_index}.png"),
                    world_to_camera: compose_camera_pose_from_rig(sensor_from_rig, &rig_pose)
                        .unwrap(),
                    sensor_from_rig: Some((*sensor_from_rig).clone()),
                };
                frame_images.push(image.clone());
                images.push(image);
            }
            frames.insert(
                frame_id,
                FrameData {
                    pose: rig_pose,
                    images: frame_images,
                },
            );
        }
        NodeData {
            spec: NodeSpec {
                node_id,
                window_start,
                images_txt: PathBuf::from(format!("synthetic-stereo-{node_id}.txt")),
            },
            images,
            frames,
            step: healthy_step(),
        }
    }

    fn healthy_step() -> StepStats {
        StepStats {
            median_m: 1.0,
            max_m: 1.0,
            ratio: 1.0,
        }
    }

    fn bad_step() -> StepStats {
        StepStats {
            median_m: 1.0,
            max_m: 100.0,
            ratio: 100.0,
        }
    }

    fn component_for_node_count(node_count: usize) -> AtlasComponent {
        AtlasComponent {
            anchor_index: 0,
            node_indices: (0..node_count).collect(),
            atlas_from_node: (0..node_count)
                .map(|node_index| (node_index, Sim3::identity()))
                .collect(),
        }
    }

    fn assert_sim3_close(actual: &Sim3, expected: &Sim3) {
        assert!((actual.scale - expected.scale).abs() < 1.0e-12);
        assert!((actual.translation - expected.translation).norm() < 1.0e-12);
        assert!(actual.rotation.rotation_to(&expected.rotation).angle() < 1.0e-12);
    }

    fn assert_se3_close(actual: &SE3, expected: &SE3) {
        assert!((actual.translation - expected.translation).norm() < 1.0e-12);
        assert!(actual.rotation.rotation_to(&expected.rotation).angle() < 1.0e-12);
    }

    #[test]
    fn interior_frame_owner_prefers_the_deepest_registered_overlap() {
        let centres = trajectory_centres(20);
        let nodes = vec![
            synthetic_node(
                10,
                0,
                0..7,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                20,
                250,
                0..11,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                30,
                500,
                4..7,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
        ];
        let component = component_for_node_count(nodes.len());

        let newest = select_frame_owners(&nodes, &component, FrameOwnerPolicy::Newest);
        assert_eq!(newest[&5], 2);
        let interior = select_frame_owners(&nodes, &component, FrameOwnerPolicy::Interior);
        assert_eq!(interior[&5], 1);
    }

    #[test]
    fn interior_frame_owner_uses_newest_start_then_node_id_for_ties() {
        let centres = trajectory_centres(12);
        let nodes = vec![
            synthetic_node(
                10,
                0,
                0..11,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                20,
                250,
                0..11,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                21,
                250,
                0..11,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
        ];
        let owners = select_frame_owners(
            &nodes,
            &component_for_node_count(nodes.len()),
            FrameOwnerPolicy::Interior,
        );
        assert_eq!(owners[&5], 2);
    }

    #[test]
    fn interior_frame_owner_ignores_holes_and_unhealthy_nodes() {
        let centres = trajectory_centres(20);
        let nodes = vec![
            synthetic_node(
                10,
                0,
                0..11,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                20,
                250,
                (0..5).chain(6..11),
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                30,
                500,
                0..21,
                &Sim3::identity(),
                &centres,
                Some(bad_step()),
            ),
        ];
        let owners = select_frame_owners(
            &nodes,
            &component_for_node_count(nodes.len()),
            FrameOwnerPolicy::Interior,
        );
        assert_eq!(owners[&5], 0, "the hole is not an owner candidate");
        assert_eq!(owners[&10], 1, "unhealthy nodes are excluded");
    }

    #[test]
    fn recomposes_calibrated_cameras_after_rig_sim3_without_scaling_baseline() {
        let rig_local = Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.21, -0.17, 0.31),
            Vector3::new(-0.8, 0.45, -1.2),
        );
        let sensor_a = SE3::new(
            UnitQuaternion::from_euler_angles(0.04, -0.08, 0.12),
            Vector3::new(0.11, -0.03, 0.02),
        );
        let sensor_b = SE3::new(
            UnitQuaternion::from_euler_angles(-0.13, 0.06, -0.05),
            Vector3::new(-0.27, 0.08, 0.04),
        );
        let input_baseline = (sensor_a.inverse().transform_point(&Point3::origin())
            - sensor_b.inverse().transform_point(&Point3::origin()))
        .norm();
        assert!(sensor_a.rotation.rotation_to(&sensor_b.rotation).angle() > 1.0e-3);
        let rig_local_centre = rig_local.camera_center_world();

        for scale in [0.7, 1.3] {
            let atlas_from_node = Sim3::new(
                UnitQuaternion::from_euler_angles(-0.19, 0.27, -0.14),
                Vector3::new(2.4, -1.1, 0.8),
                scale,
            );
            let atlas_rig_pose = transform_pose_to_atlas(&atlas_from_node, &rig_local).unwrap();
            let camera_a = compose_camera_pose_from_rig(&sensor_a, &atlas_rig_pose).unwrap();
            let camera_b = compose_camera_pose_from_rig(&sensor_b, &atlas_rig_pose).unwrap();

            assert_se3_close(
                &camera_a
                    .world_to_camera
                    .compose(&atlas_rig_pose.world_to_camera.inverse()),
                &sensor_a,
            );
            assert_se3_close(
                &camera_b
                    .world_to_camera
                    .compose(&atlas_rig_pose.world_to_camera.inverse()),
                &sensor_b,
            );
            let expected_rig_centre = atlas_from_node.transform_point(&rig_local_centre);
            assert!((atlas_rig_pose.camera_center_world() - expected_rig_centre).norm() < 1.0e-12);
            let output_baseline =
                (camera_a.camera_center_world() - camera_b.camera_center_world()).norm();
            assert!((output_baseline - input_baseline).abs() < 1.0e-12);
        }
    }

    fn test_edge(
        source_index: usize,
        target_index: usize,
        target_from_source: Sim3,
        common_frames: usize,
        retained_frames: usize,
        inlier_ratio: Option<f64>,
        mean_residual_ratio: Option<f64>,
        start_gap: u64,
    ) -> SeamEdge {
        SeamEdge {
            source_index,
            target_index,
            target_from_source,
            common_frames,
            retained_frames,
            inlier_ratio,
            mean_residual_ratio,
            start_gap,
            method: Some("test".to_owned()),
        }
    }

    fn stitch_fixture_nodes() -> Vec<NodeData> {
        let centres = trajectory_centres(35);
        vec![
            synthetic_node(
                10,
                0,
                0..16,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                11,
                250,
                8..24,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                12,
                500,
                20..36,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
        ]
    }

    #[test]
    fn composes_noncommuting_sim3_chain_from_oldest_anchor() {
        let first = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.2)),
            Vector3::new(1.0, -0.5, 0.3),
            1.2,
        );
        let second = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.15, 0.1, 0.0)),
            Vector3::new(-0.7, 0.4, 1.1),
            0.8,
        );
        let edges = vec![
            SeamEdge {
                source_index: 1,
                target_index: 0,
                target_from_source: first.clone(),
                common_frames: 10,
                retained_frames: 10,
                inlier_ratio: Some(1.0),
                mean_residual_ratio: Some(0.01),
                start_gap: 1,
                method: Some("test".to_owned()),
            },
            SeamEdge {
                source_index: 2,
                target_index: 1,
                target_from_source: second.clone(),
                common_frames: 10,
                retained_frames: 10,
                inlier_ratio: Some(1.0),
                mean_residual_ratio: Some(0.01),
                start_gap: 1,
                method: Some("test".to_owned()),
            },
        ];
        let components = connected_components(3, &edges);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].anchor_index, 0);
        assert_eq!(components[0].node_indices, vec![0, 1, 2]);
        assert_sim3_close(&components[0].atlas_from_node[&1], &first);
        assert_sim3_close(&components[0].atlas_from_node[&2], &first.compose(&second));
        assert!(
            (first.compose(&second).translation - second.compose(&first).translation).norm()
                > 1.0e-3
        );
    }

    #[test]
    fn metric_se3_preserves_exact_noncommuting_unit_cycle() {
        let centres = trajectory_centres(2);
        let nodes = vec![
            synthetic_node(10, 0, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(20, 250, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(30, 500, 0..2, &Sim3::identity(), &centres, None),
        ];
        let first = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.2)),
            Vector3::new(1.0, -0.5, 0.3),
            1.0,
        );
        let second = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.15, 0.1, 0.0)),
            Vector3::new(-0.7, 0.4, 1.1),
            1.0,
        );
        let expected = first.compose(&second);
        let edges = vec![
            test_edge(1, 0, first.clone(), 10, 10, Some(1.0), Some(0.01), 250),
            test_edge(2, 1, second.clone(), 10, 10, Some(1.0), Some(0.01), 250),
            test_edge(2, 0, expected.clone(), 10, 10, Some(1.0), Some(0.01), 500),
        ];
        let mut components = connected_components(nodes.len(), &edges);
        let reports = optimize_metric_se3_components(&nodes, &edges, &mut components).unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].edge_count, 3);
        assert!(reports[0].initial_cost < 1.0e-20);
        assert!(reports[0].final_cost < 1.0e-20);
        assert_sim3_close(&components[0].atlas_from_node[&1], &first);
        assert_sim3_close(&components[0].atlas_from_node[&2], &expected);
        assert!(components
            .iter()
            .flat_map(|component| component.atlas_from_node.values())
            .all(|transform| (transform.scale - 1.0).abs() < UNIT_SCALE_TOLERANCE));
    }

    #[test]
    fn metric_se3_reduces_inconsistent_cycle_cost_deterministically() {
        let centres = trajectory_centres(2);
        let nodes = vec![
            synthetic_node(10, 0, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(20, 250, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(30, 500, 0..2, &Sim3::identity(), &centres, None),
        ];
        let first = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.2)),
            Vector3::new(1.0, -0.5, 0.3),
            1.0,
        );
        let second = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.15, 0.1, 0.0)),
            Vector3::new(-0.7, 0.4, 1.1),
            1.0,
        );
        let expected = first.compose(&second);
        let perturbation = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.04)),
            Vector3::new(0.08, -0.03, 0.02),
            1.0,
        );
        let edges = vec![
            test_edge(1, 0, first, 10, 10, Some(1.0), Some(0.01), 250),
            test_edge(2, 1, second, 10, 10, Some(1.0), Some(0.01), 250),
            test_edge(
                2,
                0,
                expected.compose(&perturbation),
                10,
                10,
                Some(1.0),
                Some(0.01),
                500,
            ),
        ];
        let mut first_components = connected_components(nodes.len(), &edges);
        let first_report =
            optimize_metric_se3_components(&nodes, &edges, &mut first_components).unwrap();
        let mut second_components = connected_components(nodes.len(), &edges);
        let second_report =
            optimize_metric_se3_components(&nodes, &edges, &mut second_components).unwrap();
        assert!(first_report[0].initial_cost > 1.0e-6);
        assert!(first_report[0].final_cost < first_report[0].initial_cost);
        assert_eq!(first_report, second_report);
        assert_eq!(first_components, second_components);
    }

    #[test]
    fn metric_se3_rejects_nonunit_and_handles_singleton_components() {
        let centres = trajectory_centres(1);
        let nodes = vec![
            synthetic_node(10, 0, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(20, 250, 0..2, &Sim3::identity(), &centres, None),
        ];
        let nonunit = vec![test_edge(
            1,
            0,
            Sim3::new(UnitQuaternion::identity(), Vector3::zeros(), 1.2),
            10,
            10,
            Some(1.0),
            Some(0.01),
            250,
        )];
        let mut nonunit_components = connected_components(nodes.len(), &nonunit);
        let error =
            optimize_metric_se3_components(&nodes, &nonunit, &mut nonunit_components).unwrap_err();
        assert!(error.contains("non-unit scale"));

        let mut singleton_components = connected_components(nodes.len(), &[]);
        let reports =
            optimize_metric_se3_components(&nodes, &[], &mut singleton_components).unwrap();
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|report| {
            report.edge_count == 0
                && report.initial_cost == 0.0
                && report.final_cost == 0.0
                && report.iterations == 0
        }));
    }

    #[test]
    fn metric_se3_rejects_quality_transform_graph() {
        let nodes = stitch_fixture_nodes();
        let root = fixture_root("metric-quality-policy");
        let error = stitch_nodes_with_policy(
            &nodes,
            &root,
            &seam_alignment_config(),
            ForestPolicy::Quality,
            FrameOwnerPolicy::Newest,
            true,
        )
        .unwrap_err();
        assert!(error.contains("requires --forest-policy traversal"));
    }

    #[test]
    fn metric_se3_write_reparse_preserves_calibrated_stereo_baseline() {
        let root = fixture_root("metric-stereo-roundtrip");
        let centres = trajectory_centres(4);
        let sensor_a = SE3::new(
            UnitQuaternion::from_euler_angles(0.04, -0.08, 0.12),
            Vector3::new(0.11, -0.03, 0.02),
        );
        let sensor_b = SE3::new(
            UnitQuaternion::from_euler_angles(-0.13, 0.06, -0.05),
            Vector3::new(-0.27, 0.08, 0.04),
        );
        let nodes = vec![
            synthetic_stereo_node(10, 0, 0..5, &centres, &sensor_a, &sensor_b),
            synthetic_stereo_node(20, 250, 0..5, &centres, &sensor_a, &sensor_b),
            synthetic_stereo_node(30, 500, 0..5, &centres, &sensor_a, &sensor_b),
        ];
        let first = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.2)),
            Vector3::new(1.0, -0.5, 0.3),
            1.0,
        );
        let second = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.15, 0.1, 0.0)),
            Vector3::new(-0.7, 0.4, 1.1),
            1.0,
        );
        let perturbation = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.04)),
            Vector3::new(0.08, -0.03, 0.02),
            1.0,
        );
        let edges = vec![
            test_edge(1, 0, first.clone(), 10, 10, Some(1.0), Some(0.01), 250),
            test_edge(2, 1, second.clone(), 10, 10, Some(1.0), Some(0.01), 250),
            test_edge(
                2,
                0,
                first.compose(&second).compose(&perturbation),
                10,
                10,
                Some(1.0),
                Some(0.01),
                500,
            ),
        ];
        let mut components = connected_components(nodes.len(), &edges);
        let initial_node2 = components[0].atlas_from_node[&2].clone();
        optimize_metric_se3_components(&nodes, &edges, &mut components).unwrap();
        let optimized_node2 = components[0].atlas_from_node[&2].clone();
        assert!(
            (optimized_node2.translation - initial_node2.translation).norm() > 1.0e-6
                || optimized_node2
                    .rotation
                    .rotation_to(&initial_node2.rotation)
                    .angle()
                    > 1.0e-6
        );
        let mut output_ids = BTreeSet::new();
        let summary = publish_component(
            0,
            &nodes,
            &components[0],
            &root,
            &mut output_ids,
            FrameOwnerPolicy::Newest,
        )
        .unwrap();
        assert_eq!(summary.frame_count, 5);
        assert_eq!(summary.image_count, 10);
        let parsed = parse_colmap_images(&root.join("component-000/images.txt")).unwrap();
        assert_eq!(parsed.len(), 10);
        for frame_id in 0..5 {
            let expected_rig =
                transform_pose_to_atlas(&optimized_node2, &nodes[2].frames[&frame_id].pose)
                    .unwrap();
            let expected_a = compose_camera_pose_from_rig(&sensor_a, &expected_rig).unwrap();
            let expected_b = compose_camera_pose_from_rig(&sensor_b, &expected_rig).unwrap();
            let output_a = parsed
                .iter()
                .find(|image| image.name == format!("node30_frame{frame_id}_sensor0.png"))
                .unwrap();
            let output_b = parsed
                .iter()
                .find(|image| image.name == format!("node30_frame{frame_id}_sensor1.png"))
                .unwrap();
            assert_se3_close(
                &output_a.world_to_camera.world_to_camera,
                &expected_a.world_to_camera,
            );
            assert_se3_close(
                &output_b.world_to_camera.world_to_camera,
                &expected_b.world_to_camera,
            );
            let input_baseline =
                (expected_a.camera_center_world() - expected_b.camera_center_world()).norm();
            let output_baseline = (output_a.world_to_camera.camera_center_world()
                - output_b.world_to_camera.camera_center_world())
            .norm();
            assert!((output_baseline - input_baseline).abs() < 1.0e-12);
        }
    }

    #[test]
    fn quality_first_forest_prefers_high_support_cycle_edge_and_is_deterministic() {
        let centres = trajectory_centres(2);
        let nodes = vec![
            synthetic_node(10, 0, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(20, 250, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(30, 500, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(40, 1_000, 0..2, &Sim3::identity(), &centres, None),
        ];
        let edges = vec![
            // Lower support must lose even though its residual is smaller.
            test_edge(1, 0, Sim3::identity(), 10, 8, Some(1.0), Some(0.001), 250),
            test_edge(2, 1, Sim3::identity(), 9, 8, Some(0.8), Some(0.020), 250),
            // The high-support cycle edge is selected first.
            test_edge(2, 0, Sim3::identity(), 20, 8, Some(0.7), Some(0.800), 500),
        ];
        let all_edges = edges.clone();
        let forest = select_quality_forest(&nodes, &edges);
        assert_eq!(forest.len(), 2);
        assert_eq!(
            forest
                .iter()
                .map(|edge| (edge.source_index, edge.target_index))
                .collect::<Vec<_>>(),
            vec![(2, 0), (1, 0)]
        );
        let reversed = edges.into_iter().rev().collect::<Vec<_>>();
        assert_eq!(forest, select_quality_forest(&nodes, &reversed));

        let all_components = connected_components(nodes.len(), &all_edges);
        let forest_components = connected_components(nodes.len(), &forest);
        let component_nodes = |components: &[AtlasComponent]| {
            components
                .iter()
                .map(|component| component.node_indices.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            component_nodes(&forest_components),
            component_nodes(&all_components)
        );
        assert_eq!(forest_components.len(), 2);
    }

    #[test]
    fn quality_forest_preserves_noncommuting_transform_direction() {
        let centres = trajectory_centres(2);
        let nodes = vec![
            synthetic_node(10, 0, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(20, 250, 0..2, &Sim3::identity(), &centres, None),
            synthetic_node(30, 500, 0..2, &Sim3::identity(), &centres, None),
        ];
        let first = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.2)),
            Vector3::new(1.0, -0.5, 0.3),
            1.2,
        );
        let second = Sim3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.15, 0.1, 0.0)),
            Vector3::new(-0.7, 0.4, 1.1),
            0.8,
        );
        let edges = vec![
            test_edge(1, 0, first.clone(), 10, 10, Some(1.0), Some(0.01), 250),
            test_edge(2, 1, second.clone(), 10, 10, Some(1.0), Some(0.01), 250),
        ];
        let forest = select_quality_forest(&nodes, &edges);
        let components = connected_components(3, &forest);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].anchor_index, 0);
        assert_sim3_close(&components[0].atlas_from_node[&1], &first);
        assert_sim3_close(&components[0].atlas_from_node[&2], &first.compose(&second));
    }

    #[test]
    fn publishes_two_components_in_minimum_frame_order() {
        let root = fixture_root("two-components");
        let centres = trajectory_centres(111);
        let nodes = vec![
            synthetic_node(
                10,
                0,
                0..16,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                11,
                250,
                8..24,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                12,
                1_000,
                100..112,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
        ];
        assert_eq!(
            nodes[0].frames[&8].images[0].image_id, nodes[1].frames[&8].images[0].image_id,
            "the fixture must exercise repeated local COLMAP ids"
        );
        let result = stitch_nodes(&nodes, &root).unwrap();
        assert_eq!(result.components.len(), 2);
        assert_eq!(result.components[0].min_frame_id, Some(0));
        assert_eq!(result.components[0].max_frame_id, Some(23));
        assert_eq!(result.components[0].frame_count, 24);
        assert_eq!(result.components[1].min_frame_id, Some(100));
        assert_eq!(result.components[1].max_frame_id, Some(111));
        assert_eq!(result.components[1].frame_count, 12);
        assert_eq!(
            fs::read_to_string(root.join("component-000/images.txt"))
                .unwrap()
                .matches("\n\n")
                .count(),
            24
        );
        assert_eq!(
            fs::read_to_string(root.join("component-001/images.txt"))
                .unwrap()
                .matches("\n\n")
                .count(),
            12
        );
        let mut output_ids = BTreeSet::new();
        for component_index in 0..result.components.len() {
            let output =
                fs::read_to_string(root.join(format!("component-{component_index:03}/images.txt")))
                    .unwrap();
            for line in output
                .lines()
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
            {
                let image_id = line
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();
                assert!(
                    output_ids.insert(image_id),
                    "global image id must be unique"
                );
            }
        }
        assert_eq!(
            output_ids.len(),
            result
                .components
                .iter()
                .map(|component| component.image_count)
                .sum()
        );
    }

    #[test]
    fn bad_bridge_connects_components_but_its_frames_are_not_published() {
        let root = fixture_root("bad-bridge");
        let centres = trajectory_centres(35);
        let nodes = vec![
            synthetic_node(
                10,
                0,
                0..16,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                11,
                250,
                8..28,
                &Sim3::identity(),
                &centres,
                Some(bad_step()),
            ),
            synthetic_node(
                12,
                500,
                20..36,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
        ];
        let result = stitch_nodes(&nodes, &root).unwrap();
        assert_eq!(result.components.len(), 1);
        assert_eq!(result.components[0].frame_count, 32);
        assert!(result.seam_diagnostics.iter().any(|diagnostic| {
            diagnostic.source_node_id == 11
                && diagnostic.target_node_id == 10
                && diagnostic.accepted
                && diagnostic.selected_for_transform_graph
        }));
        assert!(result.seam_diagnostics.iter().any(|diagnostic| {
            diagnostic.source_node_id == 12
                && diagnostic.target_node_id == 11
                && diagnostic.accepted
                && diagnostic.selected_for_transform_graph
        }));
        let output = fs::read_to_string(root.join("component-000/images.txt")).unwrap();
        for frame_id in 16..20 {
            assert!(!output.contains(&format!("node11_frame{frame_id}.png")));
        }
        assert!(output.contains("node10_frame15.png"));
        assert!(output.contains("node12_frame20.png"));
    }

    #[test]
    fn stitching_and_output_are_deterministic() {
        let nodes = stitch_fixture_nodes();
        let first_root = fixture_root("determinism-first");
        let second_root = fixture_root("determinism-second");
        let first = stitch_nodes(&nodes, &first_root).unwrap();
        let second = stitch_nodes(&nodes, &second_root).unwrap();
        assert_eq!(first, second);
        for component_index in 0..first.components.len() {
            let relative = format!("component-{component_index:03}/images.txt");
            assert_eq!(
                fs::read(first_root.join(&relative)).unwrap(),
                fs::read(second_root.join(&relative)).unwrap()
            );
        }
    }

    #[test]
    fn seam_diagnostics_distinguish_wrapper_and_nested_rejections() {
        let collinear_centres = (0..=10)
            .map(|frame_id| (frame_id, Point3::new(frame_id as f64, 0.0, 0.0)))
            .collect::<BTreeMap<_, _>>();
        let nested = build_seam_graph(&[
            synthetic_node(
                10,
                0,
                0..11,
                &Sim3::identity(),
                &collinear_centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                11,
                250,
                0..11,
                &Sim3::identity(),
                &collinear_centres,
                Some(healthy_step()),
            ),
        ])
        .1;
        assert_eq!(nested.len(), 1);
        assert!(!nested[0].selected_for_transform_graph);
        assert_eq!(nested[0].reason, "Sim3Rejected");
        assert_eq!(
            nested[0].wrapper_rejection_reason.as_deref(),
            Some("Sim3Rejected")
        );
        assert_eq!(
            nested[0].sim3_rejection_reason.as_deref(),
            Some("DegenerateSourceGeometry")
        );
        assert_eq!(nested[0].sim3_rejection_inlier_count, Some(0));
        assert_eq!(nested[0].sim3_rejection_inlier_ratio, Some(0.0));
        assert_eq!(nested[0].sim3_rejection_mean_residual_ratio, None);

        let centres = trajectory_centres(15);
        let wrapper = build_seam_graph(&[
            synthetic_node(
                20,
                0,
                0..8,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
            synthetic_node(
                21,
                250,
                8..16,
                &Sim3::identity(),
                &centres,
                Some(healthy_step()),
            ),
        ])
        .1;
        assert_eq!(wrapper.len(), 1);
        assert_eq!(wrapper[0].reason, "InsufficientCommonFrames");
        assert_eq!(
            wrapper[0].wrapper_rejection_reason.as_deref(),
            Some("InsufficientCommonFrames")
        );
        assert_eq!(wrapper[0].sim3_rejection_reason, None);
        assert_eq!(wrapper[0].sim3_rejection_inlier_count, None);
        assert!(!wrapper[0].selected_for_transform_graph);
    }
}
