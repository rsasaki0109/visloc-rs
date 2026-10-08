use super::*;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const MAX_VARIABLE_POSES: usize = 512;
const MAX_LANDMARKS: usize = 8_192;
const MAX_OBSERVATIONS: usize = 262_144;
const MAX_SCHUR_SCALARS: usize = 3_072;
const MAX_CROSS_PAIR_WORK: usize = 64_000_000;
const MAX_CAMERAS: usize = 64;
const ORACLE_PC_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
struct OracleFixture {
    source_hashes: BTreeMap<String, String>,
    initial_cost: f64,
    initial_cost_bits: u64,
    ba: BundleAdjustment,
}

#[derive(Debug, Clone, PartialEq)]
struct PredictedDecrease {
    half_damped: f64,
    squared_damped: f64,
    squared_undamped: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Feasibility {
    finite: bool,
    pose_true_residual: f64,
    implicit_pose_true_residual: Option<f64>,
    max_landmark_backsub_residual: f64,
    geometry_observation_count: usize,
    geometry_invalid_observations: usize,
    geometry_nonpositive_depth: usize,
    geometry_cost: f64,
    geometry_rms_error: f64,
    geometry_max_error: f64,
    geometry_feasible: bool,
    feasible: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct OracleCaseReport {
    lambda: f64,
    pcg_max_iterations: usize,
    pose_blocks: usize,
    landmark_count: usize,
    observation_count: usize,
    schur_dimension: usize,
    singular_landmarks: usize,
    raw_schur_asymmetry: Option<f64>,
    schur_base_action_norm: Option<f64>,
    schur_eliminated_action_norm: Option<f64>,
    schur_arithmetic_scale: Option<f64>,
    operator_raw_action_error: Option<f64>,
    operator_raw_action_relative_error: Option<f64>,
    operator_lower_action_error: Option<f64>,
    operator_lower_action_relative_error: Option<f64>,
    operator_rhs_error: Option<f64>,
    direct_explicit_pose_error: Option<f64>,
    direct_explicit_landmark_error: Option<f64>,
    direct_feasibility: Option<Feasibility>,
    explicit_feasibility: Option<Feasibility>,
    direct_prediction: Option<PredictedDecrease>,
    explicit_prediction: Option<PredictedDecrease>,
    pcg_status: String,
    pcg_iterations: Option<usize>,
    pcg_true_residual: Option<f64>,
    pcg_target: Option<f64>,
    matrix_free_pose_error: Option<f64>,
    matrix_free_landmark_error: Option<f64>,
    matrix_free_feasibility: Option<Feasibility>,
    matrix_free_prediction: Option<PredictedDecrease>,
}

/// Result of the test-only PCG recurrence when the explicitly materialized
/// lower-mirrored Schur matrix is used as the action.  A failed solve does
/// not retain its last iterate: in particular, no failed trial is passed
/// to the geometry or prediction checks below.
#[derive(Debug, Clone, PartialEq)]
struct ExplicitPcgIsolationReport {
    lambda: f64,
    pcg_max_iterations: usize,
    status: String,
    iterations: Option<usize>,
    recursive_residual: Option<f64>,
    lower_true_residual: Option<f64>,
    implicit_true_residual: Option<f64>,
    target: Option<f64>,
    dense_lower_true_residual: Option<f64>,
    dense_implicit_true_residual: Option<f64>,
    pose_error_vs_dense: Option<f64>,
    landmark_error_vs_dense: Option<f64>,
    implicit_pcg_status: String,
    implicit_pcg_iterations: Option<usize>,
    implicit_pcg_true_residual: Option<f64>,
    implicit_pcg_target: Option<f64>,
    explicit_vs_implicit_pose_error: Option<f64>,
    explicit_vs_implicit_landmark_error: Option<f64>,
    feasibility: Option<Feasibility>,
    prediction: Option<PredictedDecrease>,
}

/// Diagnostics collected before choosing the landmark-block factorization.
/// The asymmetry and scale are deliberately reported separately: a large
/// block scale can make a small absolute difference look alarming, while
/// a small-looking difference can still matter after Schur cancellation.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LandmarkFactorMetrics {
    h_ll_max_asymmetry: f64,
    h_ll_max_scale: f64,
    inverse_max_asymmetry: Option<f64>,
    inverse_max_scale: Option<f64>,
    h_ll_inverse_identity_residual: Option<f64>,
}

impl Default for LandmarkFactorMetrics {
    fn default() -> Self {
        Self {
            h_ll_max_asymmetry: 0.0,
            h_ll_max_scale: 0.0,
            inverse_max_asymmetry: None,
            inverse_max_scale: None,
            h_ll_inverse_identity_residual: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CholeskyPcgArmReport {
    status: String,
    iterations: Option<usize>,
    recursive_residual: Option<f64>,
    action_true_residual: Option<f64>,
    target: Option<f64>,
    lower_true_residual: Option<f64>,
    dense_reference_lower_true_residual: Option<f64>,
    dense_reference_action_true_residual: Option<f64>,
    dense_reference_original_pose_equation_residual: Option<f64>,
    original_pose_equation_residual: Option<f64>,
    pose_error_vs_dense: Option<f64>,
    landmark_error_vs_dense: Option<f64>,
    feasibility: Option<Feasibility>,
    prediction: Option<PredictedDecrease>,
}

#[derive(Debug, Clone, PartialEq)]
struct CholeskyPcgIsolationReport {
    lambda: f64,
    pcg_max_iterations: usize,
    general_metrics: LandmarkFactorMetrics,
    cholesky_metrics: LandmarkFactorMetrics,
    general: CholeskyPcgArmReport,
    cholesky: CholeskyPcgArmReport,
    general_vs_cholesky_pose_error: Option<f64>,
    general_vs_cholesky_landmark_error: Option<f64>,
    rhs_error: Option<f64>,
    probe_action_error: Option<f64>,
    probe_preconditioner_error: Option<f64>,
}

fn parse_f64(token: &str, context: &str) -> Result<f64, String> {
    let value = token
        .parse::<f64>()
        .map_err(|error| format!("{context}: {error}"))?;
    if !value.is_finite() {
        return Err(format!("{context}: non-finite value"));
    }
    Ok(value)
}

fn parse_usize(token: &str, context: &str) -> Result<usize, String> {
    token
        .parse::<usize>()
        .map_err(|error| format!("{context}: {error}"))
}

fn parse_u64(token: &str, context: &str) -> Result<u64, String> {
    token
        .parse::<u64>()
        .map_err(|error| format!("{context}: {error}"))
}

fn parse_se3(fields: &[&str], start: usize, context: &str) -> Result<SE3, String> {
    if fields.len() < start + 7 {
        return Err(format!("{context}: expected quaternion and translation"));
    }
    let values = (0..7)
        .map(|index| parse_f64(fields[start + index], context))
        .collect::<Result<Vec<_>, _>>()?;
    let quaternion = nalgebra::Quaternion::new(values[0], values[1], values[2], values[3]);
    let norm = quaternion.norm();
    if !norm.is_finite() || norm <= 1.0e-12 || (norm - 1.0).abs() > 1.0e-6 {
        return Err(format!("{context}: quaternion norm is not one"));
    }
    Ok(SE3::new(
        // The exporter already validated and emitted a unit quaternion.
        // Preserve its f64 components exactly; normalizing here would
        // change the initial normal system by a rounding-dependent amount.
        // `from_quaternion` normalizes.  The fixture has already checked
        // the norm, so use the unchecked constructor to preserve every
        // serialized f64 bit and keep the normal system identical.
        nalgebra::UnitQuaternion::new_unchecked(quaternion),
        Vector3::new(values[4], values[5], values[6]),
    ))
}

fn parse_fixture(path: &Path) -> Result<OracleFixture, String> {
    let file = File::open(path).map_err(|error| format!("open fixture: {error}"))?;
    let reader = BufReader::new(file);
    let mut source_hashes = BTreeMap::new();
    let mut expected_initial_cost = None;
    let mut expected_initial_cost_bits = None;
    let mut expected_counts = BTreeMap::<String, usize>::new();
    let mut cameras = BTreeMap::<u64, Camera>::new();
    let mut poses = BTreeMap::<u64, Pose>::new();
    let mut landmarks = BTreeMap::<u64, Point3<f64>>::new();
    let mut fixed_poses = BTreeSet::new();
    let mut rig_observations = Vec::new();
    let mut saw_header = false;
    let mut saw_end = false;

    for (line_index, line_result) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line_result.map_err(|error| format!("fixture line {line_number}: {error}"))?;
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.is_empty() {
            continue;
        }
        if saw_end && fields[0] != "END" {
            return Err(format!(
                "fixture line {line_number}: records after END are not allowed"
            ));
        }
        if !saw_header && fields[0] != "VISLOC_BA_ORACLE_FIXTURE" {
            return Err(format!(
                "fixture line {line_number}: header must be the first record"
            ));
        }
        match fields[0] {
            "VISLOC_BA_ORACLE_FIXTURE" => {
                if fields != ["VISLOC_BA_ORACLE_FIXTURE", "1"] {
                    return Err(format!("fixture line {line_number}: unsupported version"));
                }
                if saw_header {
                    return Err(format!("fixture line {line_number}: duplicate header"));
                }
                saw_header = true;
            }
            "SOURCE_SHA256"
            | "SOURCE_SHA256_CAMERAS"
            | "SOURCE_SHA256_IMAGES"
            | "SOURCE_SHA256_POINTS"
            | "SOURCE_SHA256_MANIFEST" => {
                if fields.len() != 2
                    || fields[1].len() != 64
                    || !fields[1].bytes().all(|byte| byte.is_ascii_hexdigit())
                {
                    return Err(format!("fixture line {line_number}: invalid source hash"));
                }
                if source_hashes
                    .insert(fields[0].to_owned(), fields[1].to_owned())
                    .is_some()
                {
                    return Err(format!("fixture line {line_number}: duplicate source hash"));
                }
            }
            "INITIAL_COST" => {
                if fields.len() != 2 || expected_initial_cost.is_some() {
                    return Err(format!("fixture line {line_number}: invalid INITIAL_COST"));
                }
                expected_initial_cost = Some(parse_f64(fields[1], "initial cost")?);
            }
            "INITIAL_COST_BITS" => {
                if fields.len() != 2 || expected_initial_cost_bits.is_some() {
                    return Err(format!(
                        "fixture line {line_number}: invalid INITIAL_COST_BITS"
                    ));
                }
                expected_initial_cost_bits = Some(parse_u64(fields[1], "initial cost bits")?);
            }
            "CAMERA_COUNT" | "POSE_COUNT" | "LANDMARK_COUNT" | "OBSERVATION_COUNT" => {
                if fields.len() != 2 {
                    return Err(format!("fixture line {line_number}: invalid count"));
                }
                let count = parse_usize(fields[1], "fixture count")?;
                let cap = match fields[0] {
                    "CAMERA_COUNT" => MAX_CAMERAS,
                    "POSE_COUNT" => MAX_VARIABLE_POSES + 1,
                    "LANDMARK_COUNT" => MAX_LANDMARKS,
                    "OBSERVATION_COUNT" => MAX_OBSERVATIONS,
                    _ => unreachable!(),
                };
                if count > cap {
                    return Err(format!(
                        "fixture line {line_number}: {} exceeds cap {cap}",
                        fields[0]
                    ));
                }
                if expected_counts
                    .insert(fields[0].to_owned(), count)
                    .is_some()
                {
                    return Err(format!("fixture line {line_number}: duplicate count"));
                }
            }
            "CAMERA" => {
                if fields.len() < 6 {
                    return Err(format!("fixture line {line_number}: short camera"));
                }
                if cameras.len() >= MAX_CAMERAS {
                    return Err(format!("fixture line {line_number}: camera cap exceeded"));
                }
                let id = parse_u64(fields[1], "camera id")?;
                let model = CameraModel::from_colmap_name(fields[2]);
                if matches!(model, CameraModel::Unknown(_)) {
                    return Err(format!("fixture line {line_number}: unknown camera model"));
                }
                let width = fields[3]
                    .parse::<u32>()
                    .map_err(|error| format!("fixture line {line_number}: width: {error}"))?;
                let height = fields[4]
                    .parse::<u32>()
                    .map_err(|error| format!("fixture line {line_number}: height: {error}"))?;
                let parameter_count = parse_usize(fields[5], "camera parameter count")?;
                if parameter_count > 16 {
                    return Err(format!(
                        "fixture line {line_number}: camera parameter cap exceeded"
                    ));
                }
                if fields.len() != 6 + parameter_count {
                    return Err(format!(
                        "fixture line {line_number}: camera parameter count mismatch"
                    ));
                }
                let params = fields[6..]
                    .iter()
                    .map(|token| parse_f64(token, "camera parameter"))
                    .collect::<Result<Vec<_>, _>>()?;
                if cameras
                    .insert(
                        id,
                        Camera {
                            id,
                            model,
                            width,
                            height,
                            params,
                        },
                    )
                    .is_some()
                {
                    return Err(format!("fixture line {line_number}: duplicate camera"));
                }
            }
            "POSE" => {
                if fields.len() != 9 {
                    return Err(format!("fixture line {line_number}: invalid pose"));
                }
                if poses.len() > MAX_VARIABLE_POSES {
                    return Err(format!("fixture line {line_number}: pose cap exceeded"));
                }
                let id = parse_u64(fields[1], "pose id")?;
                let pose = Pose {
                    world_to_camera: parse_se3(&fields, 2, "pose")?,
                };
                if poses.insert(id, pose).is_some() {
                    return Err(format!("fixture line {line_number}: duplicate pose"));
                }
            }
            "FIXED_POSE" => {
                if fields.len() != 2 {
                    return Err(format!("fixture line {line_number}: invalid fixed pose"));
                }
                if !fixed_poses.insert(parse_u64(fields[1], "fixed pose id")?) {
                    return Err(format!("fixture line {line_number}: duplicate fixed pose"));
                }
            }
            "LANDMARK" => {
                if fields.len() != 5 {
                    return Err(format!("fixture line {line_number}: invalid landmark"));
                }
                if landmarks.len() >= MAX_LANDMARKS {
                    return Err(format!("fixture line {line_number}: landmark cap exceeded"));
                }
                let id = parse_u64(fields[1], "landmark id")?;
                let point = Point3::new(
                    parse_f64(fields[2], "landmark x")?,
                    parse_f64(fields[3], "landmark y")?,
                    parse_f64(fields[4], "landmark z")?,
                );
                if landmarks.insert(id, point).is_some() {
                    return Err(format!("fixture line {line_number}: duplicate landmark"));
                }
            }
            "RIG_OBSERVATION" => {
                if fields.len() != 13 {
                    return Err(format!(
                        "fixture line {line_number}: invalid rig observation"
                    ));
                }
                if rig_observations.len() >= MAX_OBSERVATIONS {
                    return Err(format!(
                        "fixture line {line_number}: observation cap exceeded"
                    ));
                }
                let frame_id = parse_u64(fields[1], "observation frame id")?;
                let landmark_id = parse_u64(fields[2], "observation landmark id")?;
                let xy = Point2::new(
                    parse_f64(fields[3], "observation x")?,
                    parse_f64(fields[4], "observation y")?,
                );
                let camera_id = parse_u64(fields[5], "observation camera id")?;
                let sensor_from_rig = parse_se3(&fields, 6, "sensor extrinsic")?;
                rig_observations.push((frame_id, landmark_id, xy, camera_id, sensor_from_rig));
            }
            "END" => {
                if fields.len() != 1 || saw_end {
                    return Err(format!("fixture line {line_number}: invalid END"));
                }
                saw_end = true;
            }
            other => {
                return Err(format!(
                    "fixture line {line_number}: unknown record {other:?}"
                ));
            }
        }
    }
    if !saw_header || !saw_end {
        return Err("fixture is missing header or END".to_owned());
    }
    let expected_initial_cost =
        expected_initial_cost.ok_or_else(|| "fixture is missing INITIAL_COST".to_owned())?;
    let expected_initial_cost_bits = expected_initial_cost_bits
        .ok_or_else(|| "fixture is missing INITIAL_COST_BITS".to_owned())?;
    for key in [
        "SOURCE_SHA256",
        "SOURCE_SHA256_CAMERAS",
        "SOURCE_SHA256_IMAGES",
        "SOURCE_SHA256_POINTS",
        "SOURCE_SHA256_MANIFEST",
    ] {
        if !source_hashes.contains_key(key) {
            return Err(format!("fixture is missing {key}"));
        }
    }
    if cameras.is_empty() || poses.is_empty() || landmarks.is_empty() || rig_observations.is_empty()
    {
        return Err("fixture has no BA records".to_owned());
    }
    if fixed_poses.len() != 1 || !fixed_poses.iter().all(|id| poses.contains_key(id)) {
        return Err("fixture fixed pose set is invalid".to_owned());
    }
    let first_camera = cameras
        .values()
        .next()
        .cloned()
        .ok_or_else(|| "fixture has no camera".to_owned())?;
    let mut ba = BundleAdjustment::new(first_camera);
    for (id, pose) in poses {
        ba.add_pose(id, pose);
    }
    for id in fixed_poses {
        ba.fix_pose(id);
    }
    for (id, point) in landmarks {
        ba.add_landmark(id, point);
    }
    for (frame_id, landmark_id, xy, camera_id, sensor_from_rig) in rig_observations {
        let camera = cameras
            .get(&camera_id)
            .cloned()
            .ok_or_else(|| format!("observation references unknown camera {camera_id}"))?;
        if !ba.poses.contains_key(&frame_id) {
            return Err(format!("observation references unknown pose {frame_id}"));
        }
        if !ba.landmarks.contains_key(&landmark_id) {
            return Err(format!(
                "observation references unknown landmark {landmark_id}"
            ));
        }
        ba.add_rig_observation(BaRigObservation {
            keyframe_id: frame_id,
            landmark_id,
            xy,
            camera,
            sensor_from_rig,
        });
    }
    let expected = [
        ("CAMERA_COUNT", cameras.len()),
        ("POSE_COUNT", ba.poses.len()),
        ("LANDMARK_COUNT", ba.landmarks.len()),
        ("OBSERVATION_COUNT", ba.rig_observations.len()),
    ];
    for (key, actual) in expected {
        if expected_counts.get(key).copied() != Some(actual) {
            return Err(format!("fixture {key} does not match records"));
        }
    }
    let actual_initial_cost = ba.cost();
    if !actual_initial_cost.is_finite()
        || actual_initial_cost.to_bits() != expected_initial_cost_bits
        || actual_initial_cost.to_bits() != expected_initial_cost.to_bits()
    {
        return Err(format!(
            "fixture initial cost bit mismatch: expected {} ({expected_initial_cost:.17e}), actual {} ({actual_initial_cost:.17e})",
            expected_initial_cost_bits,
            actual_initial_cost.to_bits(),
        ));
    }
    Ok(OracleFixture {
        source_hashes,
        initial_cost: expected_initial_cost,
        initial_cost_bits: expected_initial_cost_bits,
        ba,
    })
}

fn variable_indices(ba: &BundleAdjustment) -> (BTreeMap<u64, usize>, BTreeMap<u64, usize>) {
    let pose_index = ba
        .poses
        .keys()
        .copied()
        .filter(|id| !ba.fixed_poses.contains(id))
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect();
    let landmark_index = ba
        .landmarks
        .keys()
        .copied()
        .filter(|id| !ba.fixed_landmarks.contains(id))
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect();
    (pose_index, landmark_index)
}

fn observation_count(ba: &BundleAdjustment) -> usize {
    ba.observations.len()
        + ba.stereo_observations.len()
        + ba.general_stereo_observations.len()
        + ba.rig_observations.len()
}

fn cross_pair_work(system: &NormalEquationsBa) -> Result<usize, String> {
    system
        .landmarks
        .iter()
        .try_fold(0_usize, |total, landmark| {
            let pairs = landmark
                .cross
                .len()
                .checked_mul(landmark.cross.len())
                .ok_or_else(|| "cross-pair work overflows".to_owned())?;
            total
                .checked_add(pairs)
                .ok_or_else(|| "cross-pair work overflows".to_owned())
        })
}

fn check_caps(
    pose_blocks: usize,
    landmarks: usize,
    observations: usize,
    cross_pairs: usize,
) -> Result<usize, String> {
    if pose_blocks > MAX_VARIABLE_POSES {
        return Err(format!("oracle variable-pose cap exceeded: {pose_blocks}"));
    }
    if landmarks > MAX_LANDMARKS {
        return Err(format!("oracle landmark cap exceeded: {landmarks}"));
    }
    if observations > MAX_OBSERVATIONS {
        return Err(format!("oracle observation cap exceeded: {observations}"));
    }
    if cross_pairs > MAX_CROSS_PAIR_WORK {
        return Err(format!("oracle cross-pair cap exceeded: {cross_pairs}"));
    }
    let dimension = pose_blocks
        .checked_mul(6)
        .ok_or_else(|| "oracle Schur dimension overflows".to_owned())?;
    if dimension > MAX_SCHUR_SCALARS {
        return Err(format!("oracle Schur dimension cap exceeded: {dimension}"));
    }
    Ok(dimension)
}

fn build_oracle_system(
    ba: &BundleAdjustment,
) -> Result<(NormalEquationsBa, usize, usize, usize), String> {
    if ba.observations.is_empty()
        && ba.stereo_observations.is_empty()
        && ba.general_stereo_observations.is_empty()
        && ba.rig_observations.is_empty()
    {
        return Err("oracle requires visual observations".to_owned());
    }
    if ba.velocities.is_empty()
        && ba.biases.is_empty()
        && ba.imu_factors.is_empty()
        && ba.bias_random_walk_factors.is_empty()
        && ba.gravity_prior.is_none()
        && ba.per_pose_gravity_prior.is_none()
        && ba.position_prior.is_none()
        && ba.pairwise_pose_factors.is_empty()
        && ba.navigation_state_prior.is_none()
    {
        // Pure-visual eligibility is intentionally explicit.  The empty
        // branch is only a guard; all actual work follows below.
    } else {
        return Err("oracle only accepts pure visual input".to_owned());
    }
    let se3_is_finite = |transform: &SE3| {
        transform
            .rotation
            .quaternion()
            .coords
            .iter()
            .all(|value| value.is_finite())
            && transform.translation.iter().all(|value| value.is_finite())
    };
    if !ba
        .poses
        .values()
        .all(|pose| se3_is_finite(&pose.world_to_camera))
        || !ba
            .landmarks
            .values()
            .all(|point| point.coords.iter().all(|v| v.is_finite()))
    {
        return Err("oracle input pose or landmark is non-finite".to_owned());
    }
    for observation in &ba.rig_observations {
        if !matches!(
            observation.camera.model,
            CameraModel::Pinhole | CameraModel::SimplePinhole
        ) {
            return Err("oracle requires distortion-free pinhole cameras".to_owned());
        }
        if observation
            .camera
            .radial_distortion()
            .is_some_and(|(k1, k2)| k1 != 0.0 || k2 != 0.0)
        {
            return Err("oracle rejects nonzero camera distortion".to_owned());
        }
        let Some(intrinsics) = observation.camera.intrinsics() else {
            return Err("oracle camera has no pinhole intrinsics".to_owned());
        };
        if ![intrinsics.0, intrinsics.1, intrinsics.2, intrinsics.3]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("oracle camera intrinsics are non-finite".to_owned());
        }
        if !se3_is_finite(&observation.sensor_from_rig) {
            return Err("oracle sensor extrinsic is non-finite".to_owned());
        }
    }
    let intrinsics = ba
        .camera
        .intrinsics()
        .ok_or_else(|| "oracle camera has no pinhole intrinsics".to_owned())?;
    let (pose_index, landmark_index) = variable_indices(ba);
    if pose_index.is_empty() {
        return Err("oracle requires at least one variable pose".to_owned());
    }
    let mut system = build_normal_equations(
        ba,
        &intrinsics,
        &pose_index,
        &landmark_index,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &RobustKernel::None,
        None,
        false,
        true,
    );
    constrain_fixed_pose_rotations(&ba.fixed_pose_rotations, &pose_index, &mut system);
    let observations = observation_count(ba);
    let pairs = cross_pair_work(&system)?;
    check_caps(pose_index.len(), landmark_index.len(), observations, pairs)?;
    Ok((system, pose_index.len(), landmark_index.len(), observations))
}

fn explicit_schur_rhs(
    system: &NormalEquationsBa,
    lambda: f64,
) -> Result<(DMatrix<f64>, DVector<f64>, usize), String> {
    let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
        return Err("oracle requires pose-diagonal system".to_owned());
    };
    let dimension = diagonal.len() * 6;
    let mut schur = DMatrix::zeros(dimension, dimension);
    for (pose, block) in diagonal.iter().enumerate() {
        let mut damped = *block;
        for component in 0..6 {
            damped[(component, component)] += lambda;
        }
        schur
            .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
            .copy_from(&damped);
    }
    let mut rhs = -&system.b_p;
    let mut singular_landmarks = 0;
    for landmark in &system.landmarks {
        let mut h_ll = landmark.h_ll;
        for component in 0..3 {
            h_ll[(component, component)] += lambda;
        }
        let Some(inverse) = h_ll.try_inverse() else {
            singular_landmarks += 1;
            continue;
        };
        for (pose, cross) in &landmark.cross {
            let update: Vector6<f64> = cross * inverse * landmark.b_l;
            for component in 0..6 {
                rhs[pose * 6 + component] += update[component];
            }
            for (other_pose, other_cross) in &landmark.cross {
                let block: Matrix6<f64> = cross * inverse * other_cross.transpose();
                for row in 0..6 {
                    for column in 0..6 {
                        schur[(pose * 6 + row, other_pose * 6 + column)] -= block[(row, column)];
                    }
                }
            }
        }
    }
    if !schur.iter().all(|value| value.is_finite()) || !rhs.iter().all(|value| value.is_finite()) {
        return Err("oracle explicit Schur is non-finite".to_owned());
    }
    Ok((schur, rhs, singular_landmarks))
}

type LandmarkCholesky = nalgebra::Cholesky<f64, nalgebra::Const<3>>;

const CHOLESKY_HLL_SYMMETRY_RELATIVE_TOLERANCE: f64 = 1.0e-12;

fn damped_landmark_hessian(landmark: &LandmarkBlock, lambda: f64) -> Result<Matrix3<f64>, String> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(format!("invalid landmark damping {lambda}"));
    }
    let mut h_ll = landmark.h_ll;
    for component in 0..3 {
        h_ll[(component, component)] += lambda;
    }
    if !h_ll.iter().all(|value| value.is_finite()) {
        return Err("damped landmark Hessian is non-finite".to_owned());
    }
    Ok(h_ll)
}

fn matrix3_max_abs(matrix: &Matrix3<f64>) -> f64 {
    matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
}

fn matrix3_max_asymmetry(matrix: &Matrix3<f64>) -> f64 {
    let mut max_difference = 0.0_f64;
    for row in 0..3 {
        for column in (row + 1)..3 {
            max_difference =
                max_difference.max((matrix[(row, column)] - matrix[(column, row)]).abs());
        }
    }
    max_difference
}

fn update_landmark_factor_metrics(
    metrics: &mut LandmarkFactorMetrics,
    h_ll: &Matrix3<f64>,
    inverse: Option<&Matrix3<f64>>,
) -> Result<(), String> {
    let h_ll_asymmetry = matrix3_max_asymmetry(h_ll);
    let h_ll_scale = matrix3_max_abs(h_ll);
    if !h_ll_asymmetry.is_finite() || !h_ll_scale.is_finite() {
        return Err("landmark Hessian metrics are non-finite".to_owned());
    }
    metrics.h_ll_max_asymmetry = metrics.h_ll_max_asymmetry.max(h_ll_asymmetry);
    metrics.h_ll_max_scale = metrics.h_ll_max_scale.max(h_ll_scale);
    let Some(inverse) = inverse else {
        return Ok(());
    };
    let inverse_asymmetry = matrix3_max_asymmetry(inverse);
    let inverse_scale = matrix3_max_abs(inverse);
    let identity_residual = (h_ll * inverse - Matrix3::identity()).norm();
    if !inverse_asymmetry.is_finite()
        || !inverse_scale.is_finite()
        || !identity_residual.is_finite()
    {
        return Err("landmark inverse metrics are non-finite".to_owned());
    }
    metrics.inverse_max_asymmetry = Some(
        metrics
            .inverse_max_asymmetry
            .unwrap_or(0.0)
            .max(inverse_asymmetry),
    );
    metrics.inverse_max_scale = Some(metrics.inverse_max_scale.unwrap_or(0.0).max(inverse_scale));
    metrics.h_ll_inverse_identity_residual = Some(
        metrics
            .h_ll_inverse_identity_residual
            .unwrap_or(0.0)
            .max(identity_residual),
    );
    Ok(())
}

fn general_landmark_factor_metrics(
    system: &NormalEquationsBa,
    lambda: f64,
) -> Result<LandmarkFactorMetrics, String> {
    let mut metrics = LandmarkFactorMetrics::default();
    for landmark in &system.landmarks {
        let h_ll = damped_landmark_hessian(landmark, lambda)?;
        let inverse = h_ll.try_inverse();
        if let Some(inverse) = inverse {
            if !inverse.iter().all(|value| value.is_finite()) {
                return Err("general landmark inverse is non-finite".to_owned());
            }
            update_landmark_factor_metrics(&mut metrics, &h_ll, Some(&inverse))?;
        } else {
            update_landmark_factor_metrics(&mut metrics, &h_ll, None)?;
        }
    }
    Ok(metrics)
}

/// Cholesky consumes the lower triangle, but only after this numerical
/// symmetry audit.  The upper triangle is then replaced from the lower
/// triangle so every Cholesky operation (RHS, Schur action, preconditioner
/// and back-substitution) sees the same explicitly audited matrix.
fn lower_symmetric_landmark_hessian(h_ll: &Matrix3<f64>) -> Result<Matrix3<f64>, String> {
    let asymmetry = matrix3_max_asymmetry(h_ll);
    let scale = matrix3_max_abs(h_ll).max(1.0);
    if asymmetry > CHOLESKY_HLL_SYMMETRY_RELATIVE_TOLERANCE * scale {
        return Err(format!(
            "landmark Hessian is not symmetric: asymmetry={asymmetry:.17e} scale={scale:.17e}"
        ));
    }
    let mut symmetric = *h_ll;
    for row in 0..3 {
        for column in (row + 1)..3 {
            symmetric[(row, column)] = symmetric[(column, row)];
        }
    }
    Ok(symmetric)
}

#[derive(Debug, Clone)]
struct CholeskyBuildFailure {
    metrics: LandmarkFactorMetrics,
    reason: String,
}

/// Test-only Schur operator whose landmark elimination uses a validated
/// lower-triangle Cholesky factor and triangular solves.  The production
/// `ImplicitSchurOperator` remains untouched; this operator exists only to
/// compare the complete elimination path under the same PCG recurrence.
struct CholeskySchurOperator<'a> {
    diagonal: Vec<Matrix6<f64>>,
    landmarks: &'a [LandmarkBlock],
    factors: Vec<LandmarkCholesky>,
    preconditioner_inverse: Vec<Matrix6<f64>>,
    rhs: DVector<f64>,
}

impl<'a> CholeskySchurOperator<'a> {
    fn new(
        system: &'a NormalEquationsBa,
        lambda: f64,
    ) -> Result<(Self, LandmarkFactorMetrics), CholeskyBuildFailure> {
        let mut metrics = LandmarkFactorMetrics::default();
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            return Err(CholeskyBuildFailure {
                metrics,
                reason: "oracle requires pose-diagonal system".to_owned(),
            });
        };
        if diagonal.is_empty() {
            return Err(CholeskyBuildFailure {
                metrics,
                reason: "oracle has no variable pose blocks".to_owned(),
            });
        }
        let dimension = diagonal.len() * 6;
        if system.b_p.len() != dimension {
            return Err(CholeskyBuildFailure {
                metrics,
                reason: format!(
                    "pose RHS dimension mismatch: expected {dimension}, actual {}",
                    system.b_p.len()
                ),
            });
        }
        if !system.b_p.iter().all(|value| value.is_finite()) {
            return Err(CholeskyBuildFailure {
                metrics,
                reason: "pose RHS is non-finite".to_owned(),
            });
        }

        let mut damped_diagonal = diagonal.clone();
        for block in &mut damped_diagonal {
            for component in 0..6 {
                block[(component, component)] += lambda;
            }
        }
        if !damped_diagonal
            .iter()
            .flat_map(|block| block.iter())
            .all(|value| value.is_finite())
        {
            return Err(CholeskyBuildFailure {
                metrics,
                reason: "damped pose Hessian is non-finite".to_owned(),
            });
        }

        let mut factors = Vec::with_capacity(system.landmarks.len());
        for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
            let h_ll = match damped_landmark_hessian(landmark, lambda) {
                Ok(h_ll) => h_ll,
                Err(reason) => {
                    return Err(CholeskyBuildFailure { metrics, reason });
                }
            };
            for (pose, cross) in &landmark.cross {
                if *pose >= diagonal.len() {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: format!(
                            "landmark {landmark_index} references pose {pose} outside {} blocks",
                            diagonal.len()
                        ),
                    });
                }
                if !cross.iter().all(|value| value.is_finite()) {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: format!("landmark {landmark_index} cross is non-finite"),
                    });
                }
            }
            if !landmark.b_l.iter().all(|value| value.is_finite()) {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: format!("landmark {landmark_index} RHS is non-finite"),
                });
            }
            let symmetric_h_ll = match lower_symmetric_landmark_hessian(&h_ll) {
                Ok(h_ll) => h_ll,
                Err(reason) => {
                    if let Err(metric_reason) =
                        update_landmark_factor_metrics(&mut metrics, &h_ll, None)
                    {
                        return Err(CholeskyBuildFailure {
                            metrics,
                            reason: metric_reason,
                        });
                    }
                    return Err(CholeskyBuildFailure { metrics, reason });
                }
            };
            let Some(cholesky) = symmetric_h_ll.cholesky() else {
                if let Err(metric_reason) =
                    update_landmark_factor_metrics(&mut metrics, &h_ll, None)
                {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: metric_reason,
                    });
                }
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: format!("landmark {landmark_index} damped Hessian is not SPD"),
                });
            };
            let inverse = cholesky.inverse();
            if !inverse.iter().all(|value| value.is_finite()) {
                if let Err(metric_reason) =
                    update_landmark_factor_metrics(&mut metrics, &h_ll, None)
                {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: metric_reason,
                    });
                }
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: format!("landmark {landmark_index} Cholesky inverse is non-finite"),
                });
            }
            if let Err(metric_reason) =
                update_landmark_factor_metrics(&mut metrics, &h_ll, Some(&inverse))
            {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: metric_reason,
                });
            }
            factors.push(cholesky);
        }

        let mut rhs = -&system.b_p;
        for (landmark, factor) in system.landmarks.iter().zip(&factors) {
            let solved = factor.solve(&landmark.b_l);
            if !solved.iter().all(|value| value.is_finite()) {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: "Cholesky RHS solve is non-finite".to_owned(),
                });
            }
            for (pose, cross) in &landmark.cross {
                let update: Vector6<f64> = cross * solved;
                for component in 0..6 {
                    rhs[pose * 6 + component] += update[component];
                }
            }
        }
        if !rhs.iter().all(|value| value.is_finite()) {
            return Err(CholeskyBuildFailure {
                metrics,
                reason: "Cholesky reduced RHS is non-finite".to_owned(),
            });
        }

        let mut preconditioner = damped_diagonal.clone();
        for (landmark, factor) in system.landmarks.iter().zip(&factors) {
            let mut grouped: BTreeMap<usize, Matrix6x3<f64>> = BTreeMap::new();
            for (pose, cross) in &landmark.cross {
                *grouped.entry(*pose).or_insert_with(Matrix6x3::zeros) += cross;
            }
            for (pose, cross) in grouped {
                let cross_transpose = cross.transpose();
                let solved = factor.solve(&cross_transpose);
                if !solved.iter().all(|value| value.is_finite()) {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: "Cholesky preconditioner solve is non-finite".to_owned(),
                    });
                }
                preconditioner[pose] -= cross * solved;
            }
        }
        if !preconditioner
            .iter()
            .flat_map(|block| block.iter())
            .all(|value| value.is_finite())
        {
            return Err(CholeskyBuildFailure {
                metrics,
                reason: "Cholesky Schur preconditioner is non-finite".to_owned(),
            });
        }
        let mut preconditioner_inverse = Vec::with_capacity(preconditioner.len());
        for (pose, block) in preconditioner.iter().enumerate() {
            let Some(cholesky) = block.cholesky() else {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: format!("Cholesky Schur preconditioner block {pose} is not SPD"),
                });
            };
            let inverse = cholesky.inverse();
            if !inverse.iter().all(|value| value.is_finite()) {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: format!("Cholesky Schur preconditioner inverse {pose} is non-finite"),
                });
            }
            preconditioner_inverse.push(inverse);
        }

        Ok((
            Self {
                diagonal: damped_diagonal,
                landmarks: &system.landmarks,
                factors,
                preconditioner_inverse,
                rhs,
            },
            metrics,
        ))
    }

    fn dimension(&self) -> usize {
        self.diagonal.len() * 6
    }

    fn rhs(&self) -> &DVector<f64> {
        &self.rhs
    }

    fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError> {
        if x.len() != self.dimension() {
            return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                expected: self.dimension(),
                actual: x.len(),
            });
        }
        if !x.iter().all(|value| value.is_finite()) {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "Cholesky operator input",
            ));
        }
        let mut out = DVector::zeros(self.dimension());
        for (pose, block) in self.diagonal.iter().enumerate() {
            let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
            let value = block * x_pose;
            for component in 0..6 {
                out[pose * 6 + component] = value[component];
            }
        }
        for (landmark, factor) in self.landmarks.iter().zip(&self.factors) {
            let mut projected = Vector3::zeros();
            for (pose, cross) in &landmark.cross {
                let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
                projected += cross.transpose() * x_pose;
            }
            let reduced = factor.solve(&projected);
            if !reduced.iter().all(|value| value.is_finite()) {
                return Err(implicit_schur::ImplicitSchurError::NonFinite(
                    "Cholesky operator landmark solve",
                ));
            }
            for (pose, cross) in &landmark.cross {
                let value: Vector6<f64> = cross * reduced;
                for component in 0..6 {
                    out[pose * 6 + component] -= value[component];
                }
            }
        }
        if !out.iter().all(|value| value.is_finite()) {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "Cholesky operator output",
            ));
        }
        Ok(out)
    }

    fn apply_preconditioner(
        &self,
        residual: &DVector<f64>,
    ) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError> {
        if residual.len() != self.dimension() {
            return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                expected: self.dimension(),
                actual: residual.len(),
            });
        }
        if !residual.iter().all(|value| value.is_finite()) {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "Cholesky preconditioner input",
            ));
        }
        let mut out = DVector::zeros(self.dimension());
        for (pose, inverse) in self.preconditioner_inverse.iter().enumerate() {
            let residual_pose: Vector6<f64> = residual.fixed_rows::<6>(pose * 6).into_owned();
            let value = inverse * residual_pose;
            for component in 0..6 {
                out[pose * 6 + component] = value[component];
            }
        }
        if !out.iter().all(|value| value.is_finite()) {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "Cholesky preconditioner output",
            ));
        }
        Ok(out)
    }

    fn complete_delta(
        &self,
        delta_pose: &DVector<f64>,
    ) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError> {
        if delta_pose.len() != self.dimension() {
            return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                expected: self.dimension(),
                actual: delta_pose.len(),
            });
        }
        if !delta_pose.iter().all(|value| value.is_finite()) {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "Cholesky pose delta",
            ));
        }
        let mut delta_landmarks = DVector::zeros(self.landmarks.len() * 3);
        for (index, (landmark, factor)) in self.landmarks.iter().zip(&self.factors).enumerate() {
            let mut accumulated = -landmark.b_l;
            for (pose, cross) in &landmark.cross {
                let delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
                accumulated -= cross.transpose() * delta;
            }
            let value = factor.solve(&accumulated);
            if !value.iter().all(|entry| entry.is_finite()) {
                return Err(implicit_schur::ImplicitSchurError::NonFinite(
                    "Cholesky landmark delta",
                ));
            }
            for component in 0..3 {
                delta_landmarks[index * 3 + component] = value[component];
            }
        }
        Ok(delta_landmarks)
    }
}

fn explicit_cholesky_schur_rhs(
    system: &NormalEquationsBa,
    operator: &CholeskySchurOperator<'_>,
) -> Result<(DMatrix<f64>, DVector<f64>), String> {
    let dimension = operator.dimension();
    let mut schur = DMatrix::zeros(dimension, dimension);
    for (pose, block) in operator.diagonal.iter().enumerate() {
        schur
            .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
            .copy_from(block);
    }
    let rhs = operator.rhs.clone();
    for (landmark, factor) in system.landmarks.iter().zip(&operator.factors) {
        // Solve each 3x6 cross block once per landmark.  The nested
        // pose-pair accumulation below is O(B^2), but repeated triangular
        // solves would add an unnecessary O(B^2) factorization cost.
        let mut solved_crosses = Vec::with_capacity(landmark.cross.len());
        for (other_pose, other_cross) in &landmark.cross {
            let solved = factor.solve(&other_cross.transpose());
            if !solved.iter().all(|value| value.is_finite()) {
                return Err("Cholesky explicit Schur block is non-finite".to_owned());
            }
            solved_crosses.push((*other_pose, solved));
        }
        for (pose, cross) in &landmark.cross {
            for (other_pose, solved) in &solved_crosses {
                let block: Matrix6<f64> = cross * solved;
                for row in 0..6 {
                    for column in 0..6 {
                        schur[(pose * 6 + row, other_pose * 6 + column)] -= block[(row, column)];
                    }
                }
            }
        }
    }
    if !schur.iter().all(|value| value.is_finite()) || !rhs.iter().all(|value| value.is_finite()) {
        return Err("Cholesky explicit Schur or RHS is non-finite".to_owned());
    }
    Ok((schur, rhs))
}

fn explicit_lower_action(
    schur: &DMatrix<f64>,
    x: &DVector<f64>,
) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError> {
    if schur.nrows() != schur.ncols() {
        return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
            expected: schur.nrows(),
            actual: schur.ncols(),
        });
    }
    if x.len() != schur.nrows() {
        return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
            expected: schur.nrows(),
            actual: x.len(),
        });
    }
    if !x.iter().all(|value| value.is_finite()) {
        return Err(implicit_schur::ImplicitSchurError::NonFinite(
            "operator input",
        ));
    }
    let output = schur * x;
    if !output.iter().all(|value| value.is_finite()) {
        return Err(implicit_schur::ImplicitSchurError::NonFinite(
            "operator output",
        ));
    }
    Ok(output)
}

use super::solve_qr_pcg as solve_test_pcg;

struct CholeskyPcgArmOutcome {
    report: CholeskyPcgArmReport,
    solution: Option<DVector<f64>>,
    landmark_delta: Option<DVector<f64>>,
}

#[allow(clippy::too_many_arguments)]
fn run_cholesky_pcg_arm<Apply, Preconditioner, Complete>(
    rhs: &DVector<f64>,
    dimension: usize,
    options: implicit_schur::PcgOptions,
    lower_schur: &DMatrix<f64>,
    dense_solution: Option<&DVector<f64>>,
    dense_landmark_delta: Option<&DVector<f64>>,
    dense_reference_action_true_residual: Option<f64>,
    dense_reference_original_pose_equation_residual: Option<f64>,
    apply: Apply,
    apply_preconditioner: Preconditioner,
    mut complete_delta: Complete,
    success_label: &str,
) -> CholeskyPcgArmOutcome
where
    Apply: FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
    Preconditioner:
        FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
    Complete: FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
{
    match solve_test_pcg(rhs, dimension, options, apply, apply_preconditioner) {
        Ok(result) => {
            let lower_true_residual = explicit_lower_action(lower_schur, &result.solution)
                .ok()
                .map(|applied| (rhs - applied).norm());
            let landmark_delta = complete_delta(&result.solution).ok();
            let dense_reference_lower_true_residual = dense_solution.and_then(|solution| {
                explicit_lower_action(lower_schur, solution)
                    .ok()
                    .map(|applied| (rhs - applied).norm())
            });
            let pose_error_vs_dense =
                dense_solution.map(|solution| (&result.solution - solution).norm());
            let landmark_error_vs_dense = match (landmark_delta.as_ref(), dense_landmark_delta) {
                (Some(actual), Some(expected)) => Some((actual - expected).norm()),
                _ => None,
            };
            CholeskyPcgArmOutcome {
                report: CholeskyPcgArmReport {
                    status: success_label.to_owned(),
                    iterations: Some(result.iterations),
                    recursive_residual: None,
                    action_true_residual: Some(result.residual_norm),
                    target: Some(result.target),
                    lower_true_residual,
                    dense_reference_lower_true_residual,
                    dense_reference_action_true_residual,
                    dense_reference_original_pose_equation_residual,
                    original_pose_equation_residual: None,
                    pose_error_vs_dense,
                    landmark_error_vs_dense,
                    feasibility: None,
                    prediction: None,
                },
                solution: Some(result.solution),
                landmark_delta,
            }
        }
        Err(error) => {
            let (iterations, recursive_residual, action_true_residual, target) =
                pcg_error_details(&error);
            CholeskyPcgArmOutcome {
                report: CholeskyPcgArmReport {
                    status: format!("failure:{error:?}"),
                    iterations,
                    recursive_residual,
                    action_true_residual,
                    target,
                    lower_true_residual: None,
                    dense_reference_lower_true_residual: dense_solution.and_then(|solution| {
                        explicit_lower_action(lower_schur, solution)
                            .ok()
                            .map(|applied| (rhs - applied).norm())
                    }),
                    dense_reference_action_true_residual,
                    dense_reference_original_pose_equation_residual,
                    original_pose_equation_residual: None,
                    pose_error_vs_dense: None,
                    landmark_error_vs_dense: None,
                    feasibility: None,
                    prediction: None,
                },
                solution: None,
                landmark_delta: None,
            }
        }
    }
}

fn failed_cholesky_pcg_arm(
    status: String,
    rhs: &DVector<f64>,
    lower_schur: Option<&DMatrix<f64>>,
    dense_solution: Option<&DVector<f64>>,
    dense_reference_action_true_residual: Option<f64>,
    dense_reference_original_pose_equation_residual: Option<f64>,
) -> CholeskyPcgArmOutcome {
    let dense_reference_lower_true_residual = dense_solution.and_then(|solution| {
        lower_schur.and_then(|lower| {
            explicit_lower_action(lower, solution)
                .ok()
                .map(|applied| (rhs - applied).norm())
        })
    });
    CholeskyPcgArmOutcome {
        report: CholeskyPcgArmReport {
            status,
            iterations: None,
            recursive_residual: None,
            action_true_residual: None,
            target: None,
            lower_true_residual: None,
            dense_reference_lower_true_residual,
            dense_reference_action_true_residual,
            dense_reference_original_pose_equation_residual,
            original_pose_equation_residual: None,
            pose_error_vs_dense: None,
            landmark_error_vs_dense: None,
            feasibility: None,
            prediction: None,
        },
        solution: None,
        landmark_delta: None,
    }
}

fn populate_cholesky_arm_diagnostics(
    outcome: &mut CholeskyPcgArmOutcome,
    ba: &BundleAdjustment,
    system: &NormalEquationsBa,
    lambda: f64,
    lower_schur: &DMatrix<f64>,
    rhs: &DVector<f64>,
    factors: Option<&[LandmarkCholesky]>,
) {
    let (Some(solution), Some(landmark_delta)) =
        (outcome.solution.as_ref(), outcome.landmark_delta.as_ref())
    else {
        return;
    };
    outcome.report.original_pose_equation_residual =
        original_pose_equation_residual(system, lambda, solution, landmark_delta);
    outcome.report.prediction = predicted_decrease(system, lambda, solution, landmark_delta).ok();
    outcome.report.feasibility = Some(match factors {
        Some(factors) => feasibility_with_cholesky(
            ba,
            system,
            lambda,
            lower_schur,
            rhs,
            solution,
            landmark_delta,
            outcome.report.action_true_residual,
            factors,
        ),
        None => feasibility(
            ba,
            system,
            lambda,
            lower_schur,
            rhs,
            solution,
            landmark_delta,
            outcome.report.action_true_residual,
        ),
    });
}

fn pcg_error_details(
    error: &implicit_schur::ImplicitSchurError,
) -> (Option<usize>, Option<f64>, Option<f64>, Option<f64>) {
    match *error {
        implicit_schur::ImplicitSchurError::ResidualCheckFailed {
            iterations,
            recursive_norm,
            true_norm,
            target,
        }
        | implicit_schur::ImplicitSchurError::MaxIterations {
            iterations,
            recursive_norm,
            residual_norm: true_norm,
            target,
        } => (
            Some(iterations),
            Some(recursive_norm),
            Some(true_norm),
            Some(target),
        ),
        _ => (None, None, None, None),
    }
}

fn failed_explicit_pcg_reports(lambda: f64, status: String) -> Vec<ExplicitPcgIsolationReport> {
    [128_usize, 512_usize]
        .into_iter()
        .map(|pcg_max_iterations| ExplicitPcgIsolationReport {
            lambda,
            pcg_max_iterations,
            status: status.clone(),
            iterations: None,
            recursive_residual: None,
            lower_true_residual: None,
            implicit_true_residual: None,
            target: None,
            dense_lower_true_residual: None,
            dense_implicit_true_residual: None,
            pose_error_vs_dense: None,
            landmark_error_vs_dense: None,
            implicit_pcg_status: "not_run".to_owned(),
            implicit_pcg_iterations: None,
            implicit_pcg_true_residual: None,
            implicit_pcg_target: None,
            explicit_vs_implicit_pose_error: None,
            explicit_vs_implicit_landmark_error: None,
            feasibility: None,
            prediction: None,
        })
        .collect()
}

fn run_explicit_pcg_isolation(
    ba: &BundleAdjustment,
) -> Result<Vec<ExplicitPcgIsolationReport>, String> {
    let (system, pose_blocks, _landmark_count, _observation_count) = build_oracle_system(ba)?;
    let CameraHessian::PoseDiagonal(_diagonal) = &system.h_pp else {
        return Err("explicit PCG oracle requires pose blocks".to_owned());
    };
    let dimension = pose_blocks * 6;
    let mut reports = Vec::new();

    // Keep the two PR #80 damping cases and add the measured practical
    // acceptance-region case without changing the legacy four reports.
    for lambda in [1.0e-4, 1.0e5, 1.0e10] {
        let (raw_schur, rhs, _singular_landmarks) = match explicit_schur_rhs(&system, lambda) {
            Ok(value) => value,
            Err(error) => {
                reports.extend(failed_explicit_pcg_reports(
                    lambda,
                    format!("failure:explicit_schur:{error}"),
                ));
                continue;
            }
        };
        let mut lower_schur = raw_schur;
        mirror_lower_triangle(&mut lower_schur);
        let operator = match implicit_schur::ImplicitSchurOperator::new(&system, lambda) {
            Ok(operator) => operator,
            Err(error) => {
                reports.extend(failed_explicit_pcg_reports(
                    lambda,
                    format!("failure:implicit_operator:{error:?}"),
                ));
                continue;
            }
        };

        let explicit_pose = solve_normal_equations(&lower_schur, &rhs).ok();
        let explicit_landmarks = explicit_pose
            .as_ref()
            .and_then(|pose| operator.complete_delta(pose).ok());
        let dense_lower_true_residual = explicit_pose.as_ref().and_then(|pose| {
            explicit_lower_action(&lower_schur, pose)
                .ok()
                .map(|applied| (&rhs - applied).norm())
        });
        let dense_implicit_true_residual = explicit_pose.as_ref().and_then(|pose| {
            operator
                .apply(pose)
                .ok()
                .map(|applied| (&rhs - applied).norm())
        });

        for pcg_max_iterations in [128_usize, 512_usize] {
            let options = implicit_schur::PcgOptions {
                max_iterations: pcg_max_iterations,
                relative_tolerance: ORACLE_PC_TOLERANCE,
                absolute_tolerance: ORACLE_PC_TOLERANCE,
            };

            // Both arms consume this same rhs and preconditioner from one
            // assembled normal system.  The production arm is retained as
            // the recurrence/diagnostic comparison for the generic test
            // recurrence used by the explicit action.
            let implicit_result = operator.solve_pcg(&rhs, options);
            let (
                implicit_pcg_status,
                implicit_pcg_iterations,
                implicit_pcg_true_residual,
                implicit_pcg_target,
                implicit_solution,
            ) = match implicit_result {
                Ok(result) => (
                    "success".to_owned(),
                    Some(result.iterations),
                    Some(result.residual_norm),
                    Some(result.target),
                    Some(result.solution),
                ),
                Err(error) => {
                    let (iterations, _recursive, residual, target) = pcg_error_details(&error);
                    (
                        format!("failure:{error:?}"),
                        iterations,
                        residual,
                        target,
                        None,
                    )
                }
            };

            let explicit_result = solve_test_pcg(
                &rhs,
                dimension,
                options,
                |x| explicit_lower_action(&lower_schur, x),
                |residual| operator.apply_preconditioner(residual),
            );
            let (
                explicit_status,
                explicit_iterations,
                explicit_recursive_residual,
                explicit_lower_true_residual,
                explicit_implicit_true_residual,
                explicit_target,
                explicit_solution,
            ) = match explicit_result {
                Ok(result) => {
                    let lower_true_residual = explicit_lower_action(&lower_schur, &result.solution)
                        .ok()
                        .map(|applied| (&rhs - applied).norm());
                    let implicit_true_residual = operator
                        .apply(&result.solution)
                        .ok()
                        .map(|applied| (&rhs - applied).norm());
                    (
                        "success_lower_pcg".to_owned(),
                        Some(result.iterations),
                        None,
                        lower_true_residual,
                        implicit_true_residual,
                        Some(result.target),
                        Some(result.solution),
                    )
                }
                Err(error) => {
                    let (iterations, recursive_residual, lower_true_residual, target) =
                        pcg_error_details(&error);
                    (
                        format!("failure:{error:?}"),
                        iterations,
                        recursive_residual,
                        lower_true_residual,
                        None,
                        target,
                        None,
                    )
                }
            };

            let delta_landmarks = explicit_solution
                .as_ref()
                .and_then(|pose| operator.complete_delta(pose).ok());
            let feasibility = explicit_solution.as_ref().and_then(|pose| {
                delta_landmarks.as_ref().map(|landmarks| {
                    feasibility(
                        ba,
                        &system,
                        lambda,
                        &lower_schur,
                        &rhs,
                        pose,
                        landmarks,
                        explicit_implicit_true_residual,
                    )
                })
            });
            let prediction = explicit_solution.as_ref().and_then(|pose| {
                delta_landmarks
                    .as_ref()
                    .and_then(|landmarks| predicted_decrease(&system, lambda, pose, landmarks).ok())
            });
            let pose_error_vs_dense = explicit_solution
                .as_ref()
                .and_then(|pose| explicit_pose.as_ref().map(|dense| (pose - dense).norm()));
            let landmark_error_vs_dense = delta_landmarks.as_ref().and_then(|landmarks| {
                explicit_landmarks
                    .as_ref()
                    .map(|dense| (landmarks - dense).norm())
            });
            let explicit_vs_implicit_pose_error = explicit_solution.as_ref().and_then(|explicit| {
                implicit_solution
                    .as_ref()
                    .map(|implicit| (explicit - implicit).norm())
            });
            let implicit_landmarks = implicit_solution
                .as_ref()
                .and_then(|pose| operator.complete_delta(pose).ok());
            let explicit_vs_implicit_landmark_error =
                delta_landmarks.as_ref().and_then(|explicit| {
                    implicit_landmarks
                        .as_ref()
                        .map(|implicit| (explicit - implicit).norm())
                });
            reports.push(ExplicitPcgIsolationReport {
                lambda,
                pcg_max_iterations,
                status: explicit_status,
                iterations: explicit_iterations,
                recursive_residual: explicit_recursive_residual,
                lower_true_residual: explicit_lower_true_residual,
                implicit_true_residual: explicit_implicit_true_residual,
                target: explicit_target,
                dense_lower_true_residual,
                dense_implicit_true_residual,
                pose_error_vs_dense,
                landmark_error_vs_dense,
                implicit_pcg_status,
                implicit_pcg_iterations,
                implicit_pcg_true_residual,
                implicit_pcg_target,
                explicit_vs_implicit_pose_error,
                explicit_vs_implicit_landmark_error,
                feasibility,
                prediction,
            });
        }
    }
    Ok(reports)
}

fn run_cholesky_pcg_isolation(
    ba: &BundleAdjustment,
) -> Result<Vec<CholeskyPcgIsolationReport>, String> {
    let (system, pose_blocks, _landmark_count, _observation_count) = build_oracle_system(ba)?;
    let CameraHessian::PoseDiagonal(_diagonal) = &system.h_pp else {
        return Err("Cholesky PCG oracle requires pose blocks".to_owned());
    };
    let dimension = pose_blocks * 6;
    let mut reports = Vec::with_capacity(6);

    for lambda in [1.0e-4, 1.0e5, 1.0e10] {
        let general_metrics = general_landmark_factor_metrics(&system, lambda)?;
        let (general_raw_schur, general_rhs, _singular_landmarks) =
            explicit_schur_rhs(&system, lambda)?;
        let mut general_lower_schur = general_raw_schur;
        mirror_lower_triangle(&mut general_lower_schur);
        let general_operator = implicit_schur::ImplicitSchurOperator::new(&system, lambda)
            .map_err(|error| format!("general operator: {error:?}"))?;
        let general_dense_solution =
            solve_normal_equations(&general_lower_schur, &general_rhs).ok();
        let general_dense_landmark_delta = general_dense_solution
            .as_ref()
            .and_then(|solution| general_operator.complete_delta(solution).ok());
        let general_dense_action_true_residual =
            general_dense_solution.as_ref().and_then(|solution| {
                general_operator
                    .apply(solution)
                    .ok()
                    .map(|applied| (general_operator.rhs() - applied).norm())
            });
        let general_dense_original_pose_equation_residual =
            match (&general_dense_solution, &general_dense_landmark_delta) {
                (Some(solution), Some(landmarks)) => {
                    original_pose_equation_residual(&system, lambda, solution, landmarks)
                }
                _ => None,
            };

        let cholesky_build = CholeskySchurOperator::new(&system, lambda);
        let (cholesky_operator, cholesky_metrics) = match cholesky_build {
            Ok(value) => value,
            Err(failure) => {
                for pcg_max_iterations in [128_usize, 512_usize] {
                    let options = implicit_schur::PcgOptions {
                        max_iterations: pcg_max_iterations,
                        relative_tolerance: ORACLE_PC_TOLERANCE,
                        absolute_tolerance: ORACLE_PC_TOLERANCE,
                    };
                    let mut general = run_cholesky_pcg_arm(
                        &general_rhs,
                        dimension,
                        options,
                        &general_lower_schur,
                        general_dense_solution.as_ref(),
                        general_dense_landmark_delta.as_ref(),
                        general_dense_action_true_residual,
                        general_dense_original_pose_equation_residual,
                        |x| general_operator.apply(x),
                        |residual| general_operator.apply_preconditioner(residual),
                        |pose| general_operator.complete_delta(pose),
                        "general_pcg",
                    );
                    populate_cholesky_arm_diagnostics(
                        &mut general,
                        ba,
                        &system,
                        lambda,
                        &general_lower_schur,
                        &general_rhs,
                        None,
                    );
                    let cholesky = failed_cholesky_pcg_arm(
                        format!("failure:cholesky_build:{}", failure.reason),
                        &general_rhs,
                        None,
                        None,
                        None,
                        None,
                    );
                    reports.push(CholeskyPcgIsolationReport {
                        lambda,
                        pcg_max_iterations,
                        general_metrics,
                        cholesky_metrics: failure.metrics,
                        general: general.report,
                        cholesky: cholesky.report,
                        general_vs_cholesky_pose_error: None,
                        general_vs_cholesky_landmark_error: None,
                        rhs_error: None,
                        probe_action_error: None,
                        probe_preconditioner_error: None,
                    });
                }
                continue;
            }
        };

        let (cholesky_raw_schur, cholesky_rhs) =
            match explicit_cholesky_schur_rhs(&system, &cholesky_operator) {
                Ok(value) => value,
                Err(error) => {
                    for pcg_max_iterations in [128_usize, 512_usize] {
                        let options = implicit_schur::PcgOptions {
                            max_iterations: pcg_max_iterations,
                            relative_tolerance: ORACLE_PC_TOLERANCE,
                            absolute_tolerance: ORACLE_PC_TOLERANCE,
                        };
                        let mut general = run_cholesky_pcg_arm(
                            &general_rhs,
                            dimension,
                            options,
                            &general_lower_schur,
                            general_dense_solution.as_ref(),
                            general_dense_landmark_delta.as_ref(),
                            general_dense_action_true_residual,
                            general_dense_original_pose_equation_residual,
                            |x| general_operator.apply(x),
                            |residual| general_operator.apply_preconditioner(residual),
                            |pose| general_operator.complete_delta(pose),
                            "general_pcg",
                        );
                        populate_cholesky_arm_diagnostics(
                            &mut general,
                            ba,
                            &system,
                            lambda,
                            &general_lower_schur,
                            &general_rhs,
                            None,
                        );
                        let cholesky = failed_cholesky_pcg_arm(
                            format!("failure:cholesky_explicit_schur:{error}"),
                            &general_rhs,
                            None,
                            None,
                            None,
                            None,
                        );
                        reports.push(CholeskyPcgIsolationReport {
                            lambda,
                            pcg_max_iterations,
                            general_metrics,
                            cholesky_metrics,
                            general: general.report,
                            cholesky: cholesky.report,
                            general_vs_cholesky_pose_error: None,
                            general_vs_cholesky_landmark_error: None,
                            rhs_error: None,
                            probe_action_error: None,
                            probe_preconditioner_error: None,
                        });
                    }
                    continue;
                }
            };
        let mut cholesky_lower_schur = cholesky_raw_schur;
        mirror_lower_triangle(&mut cholesky_lower_schur);
        let cholesky_dense_solution =
            solve_normal_equations(&cholesky_lower_schur, &cholesky_rhs).ok();
        let cholesky_dense_landmark_delta = cholesky_dense_solution
            .as_ref()
            .and_then(|solution| cholesky_operator.complete_delta(solution).ok());
        let cholesky_dense_action_true_residual =
            cholesky_dense_solution.as_ref().and_then(|solution| {
                cholesky_operator
                    .apply(solution)
                    .ok()
                    .map(|applied| (cholesky_operator.rhs() - applied).norm())
            });
        let cholesky_dense_original_pose_equation_residual =
            match (&cholesky_dense_solution, &cholesky_dense_landmark_delta) {
                (Some(solution), Some(landmarks)) => {
                    original_pose_equation_residual(&system, lambda, solution, landmarks)
                }
                _ => None,
            };

        let probe = DVector::from_iterator(
            dimension,
            (0..dimension)
                .map(|index| 0.001 * ((index % 17) as f64 - 8.0) + 0.00001 * index as f64),
        );
        let rhs_error = (&general_rhs - cholesky_operator.rhs()).norm();
        let probe_action_error = match (
            general_operator.apply(&probe),
            cholesky_operator.apply(&probe),
        ) {
            (Ok(general), Ok(cholesky)) => Some((general - cholesky).norm()),
            _ => None,
        };
        let probe_preconditioner_error = match (
            general_operator.apply_preconditioner(&probe),
            cholesky_operator.apply_preconditioner(&probe),
        ) {
            (Ok(general), Ok(cholesky)) => Some((general - cholesky).norm()),
            _ => None,
        };

        for pcg_max_iterations in [128_usize, 512_usize] {
            let options = implicit_schur::PcgOptions {
                max_iterations: pcg_max_iterations,
                relative_tolerance: ORACLE_PC_TOLERANCE,
                absolute_tolerance: ORACLE_PC_TOLERANCE,
            };
            let mut general = run_cholesky_pcg_arm(
                &general_rhs,
                dimension,
                options,
                &general_lower_schur,
                general_dense_solution.as_ref(),
                general_dense_landmark_delta.as_ref(),
                general_dense_action_true_residual,
                general_dense_original_pose_equation_residual,
                |x| general_operator.apply(x),
                |residual| general_operator.apply_preconditioner(residual),
                |pose| general_operator.complete_delta(pose),
                "general_pcg",
            );
            populate_cholesky_arm_diagnostics(
                &mut general,
                ba,
                &system,
                lambda,
                &general_lower_schur,
                &general_rhs,
                None,
            );
            let mut cholesky = run_cholesky_pcg_arm(
                cholesky_operator.rhs(),
                dimension,
                options,
                &cholesky_lower_schur,
                cholesky_dense_solution.as_ref(),
                cholesky_dense_landmark_delta.as_ref(),
                cholesky_dense_action_true_residual,
                cholesky_dense_original_pose_equation_residual,
                |x| cholesky_operator.apply(x),
                |residual| cholesky_operator.apply_preconditioner(residual),
                |pose| cholesky_operator.complete_delta(pose),
                "cholesky_pcg",
            );
            populate_cholesky_arm_diagnostics(
                &mut cholesky,
                ba,
                &system,
                lambda,
                &cholesky_lower_schur,
                cholesky_operator.rhs(),
                Some(&cholesky_operator.factors),
            );
            let general_vs_cholesky_pose_error =
                match (general.solution.as_ref(), cholesky.solution.as_ref()) {
                    (Some(general), Some(cholesky)) => Some((general - cholesky).norm()),
                    _ => None,
                };
            let general_vs_cholesky_landmark_error = match (
                general.landmark_delta.as_ref(),
                cholesky.landmark_delta.as_ref(),
            ) {
                (Some(general), Some(cholesky)) => Some((general - cholesky).norm()),
                _ => None,
            };
            reports.push(CholeskyPcgIsolationReport {
                lambda,
                pcg_max_iterations,
                general_metrics,
                cholesky_metrics,
                general: general.report,
                cholesky: cholesky.report,
                general_vs_cholesky_pose_error,
                general_vs_cholesky_landmark_error,
                rhs_error: Some(rhs_error),
                probe_action_error,
                probe_preconditioner_error,
            });
        }
    }
    Ok(reports)
}

fn schur_asymmetry(matrix: &DMatrix<f64>) -> f64 {
    let mut max_difference: f64 = 0.0;
    for row in 0..matrix.nrows() {
        for column in (row + 1)..matrix.ncols() {
            max_difference =
                max_difference.max((matrix[(row, column)] - matrix[(column, row)]).abs());
        }
    }
    max_difference
}

fn mirror_lower_triangle(matrix: &mut DMatrix<f64>) {
    for row in 0..matrix.nrows() {
        for column in (row + 1)..matrix.ncols() {
            matrix[(row, column)] = matrix[(column, row)];
        }
    }
}

fn predicted_decrease(
    system: &NormalEquationsBa,
    lambda: f64,
    delta_pose: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
) -> Result<PredictedDecrease, String> {
    let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
        return Err("predicted-decrease oracle requires pose blocks".to_owned());
    };
    if delta_pose.len() != diagonal.len() * 6 || delta_landmarks.len() != system.landmarks.len() * 3
    {
        return Err("predicted-decrease delta dimensions mismatch".to_owned());
    }
    let mut gradient_dot = system.b_p.dot(delta_pose);
    let mut hessian_quadratic = 0.0;
    let mut delta_squared = delta_pose.norm_squared();
    for (pose, block) in diagonal.iter().enumerate() {
        let delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
        hessian_quadratic += delta.dot(&(block * delta));
    }
    for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
        let delta: Vector3<f64> = delta_landmarks
            .fixed_rows::<3>(landmark_index * 3)
            .into_owned();
        gradient_dot += landmark.b_l.dot(&delta);
        hessian_quadratic += delta.dot(&(landmark.h_ll * delta));
        delta_squared += delta.norm_squared();
        for (pose, cross) in &landmark.cross {
            let pose_delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
            hessian_quadratic += 2.0 * pose_delta.dot(&(cross * delta));
        }
    }
    let damped_quadratic = hessian_quadratic + lambda * delta_squared;
    let half_damped = -gradient_dot - 0.5 * damped_quadratic;
    let squared_damped = -2.0 * gradient_dot - damped_quadratic;
    let squared_undamped = -2.0 * gradient_dot - hessian_quadratic;
    let values = [half_damped, squared_damped, squared_undamped];
    if !values.iter().all(|value| value.is_finite()) {
        return Err("predicted decrease is non-finite".to_owned());
    }
    Ok(PredictedDecrease {
        half_damped,
        squared_damped,
        squared_undamped,
    })
}

#[allow(clippy::too_many_arguments)]
fn feasibility(
    ba: &BundleAdjustment,
    system: &NormalEquationsBa,
    lambda: f64,
    schur: &DMatrix<f64>,
    rhs: &DVector<f64>,
    delta_pose: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
    implicit_pose_true_residual: Option<f64>,
) -> Feasibility {
    let finite = delta_pose.iter().all(|value| value.is_finite())
        && delta_landmarks.iter().all(|value| value.is_finite());
    let pose_residual = rhs - schur * delta_pose;
    let pose_true_residual = pose_residual.norm();
    let mut max_landmark = 0.0_f64;
    for (index, landmark) in system.landmarks.iter().enumerate() {
        let delta_l: Vector3<f64> = delta_landmarks.fixed_rows::<3>(index * 3).into_owned();
        let mut h_ll = landmark.h_ll;
        for component in 0..3 {
            h_ll[(component, component)] += lambda;
        }
        if h_ll.try_inverse().is_none() {
            continue;
        }
        let mut residual = h_ll * delta_l + landmark.b_l;
        for (pose, cross) in &landmark.cross {
            let delta_p: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
            residual += cross.transpose() * delta_p;
        }
        max_landmark = max_landmark.max(residual.norm());
    }
    let (pose_index, landmark_index) = variable_indices(ba);
    let mut geometry_cost = 0.0_f64;
    let mut geometry_max_error = 0.0_f64;
    let mut geometry_observation_count = 0_usize;
    let mut geometry_invalid_observations = 0_usize;
    let mut geometry_nonpositive_depth = 0_usize;
    for observation in &ba.rig_observations {
        let Some(pose) = ba.poses.get(&observation.keyframe_id) else {
            geometry_invalid_observations += 1;
            continue;
        };
        let Some(point) = ba.landmarks.get(&observation.landmark_id) else {
            geometry_invalid_observations += 1;
            continue;
        };
        let mut updated_pose = pose.world_to_camera.clone();
        if let Some(&index) = pose_index.get(&observation.keyframe_id) {
            let xi: Vector6<f64> = delta_pose.fixed_rows::<6>(index * 6).into_owned();
            if xi.iter().all(|value| value.is_finite()) {
                updated_pose = updated_pose.compose(&SE3::exp(&xi));
            } else {
                geometry_invalid_observations += 1;
                continue;
            }
        }
        let mut updated_point = *point;
        if let Some(&index) = landmark_index.get(&observation.landmark_id) {
            let delta: Vector3<f64> = delta_landmarks.fixed_rows::<3>(index * 3).into_owned();
            if delta.iter().all(|value| value.is_finite()) {
                updated_point = Point3::from(point.coords + delta);
            } else {
                geometry_invalid_observations += 1;
                continue;
            }
        }
        let point_camera = observation
            .sensor_from_rig
            .compose(&updated_pose)
            .transform_point(&updated_point);
        if !point_camera.z.is_finite() || point_camera.z <= 0.0 {
            geometry_nonpositive_depth += 1;
            geometry_invalid_observations += 1;
            continue;
        }
        let Some(projected) = observation.camera.project(&point_camera) else {
            geometry_invalid_observations += 1;
            continue;
        };
        let residual = projected - observation.xy;
        let squared = residual.norm_squared();
        if !point_camera.coords.iter().all(|value| value.is_finite())
            || !projected.coords.iter().all(|value| value.is_finite())
            || !squared.is_finite()
        {
            geometry_invalid_observations += 1;
            continue;
        }
        let error = squared.sqrt();
        if !error.is_finite() {
            geometry_invalid_observations += 1;
            continue;
        }
        geometry_observation_count += 1;
        geometry_cost += squared;
        geometry_max_error = geometry_max_error.max(error);
    }
    let geometry_rms_error = if geometry_observation_count == 0 {
        f64::NAN
    } else {
        (geometry_cost / geometry_observation_count as f64).sqrt()
    };
    let geometry_feasible = geometry_observation_count > 0
        && geometry_invalid_observations == 0
        && geometry_cost.is_finite()
        && geometry_rms_error.is_finite()
        && geometry_max_error.is_finite();
    Feasibility {
        finite,
        pose_true_residual,
        implicit_pose_true_residual,
        max_landmark_backsub_residual: max_landmark,
        geometry_observation_count,
        geometry_invalid_observations,
        geometry_nonpositive_depth,
        geometry_cost,
        geometry_rms_error,
        geometry_max_error,
        geometry_feasible,
        feasible: finite
            && pose_true_residual.is_finite()
            && max_landmark.is_finite()
            && geometry_feasible,
    }
}

fn cholesky_backsub_residual(
    system: &NormalEquationsBa,
    lambda: f64,
    _factors: &[LandmarkCholesky],
    delta_pose: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
) -> f64 {
    let mut max_residual = 0.0_f64;
    for (index, landmark) in system.landmarks.iter().enumerate() {
        let delta_l: Vector3<f64> = delta_landmarks.fixed_rows::<3>(index * 3).into_owned();
        let mut h_ll = landmark.h_ll;
        for component in 0..3 {
            h_ll[(component, component)] += lambda;
        }
        let mut residual = h_ll * delta_l + landmark.b_l;
        for (pose, cross) in &landmark.cross {
            let delta_p: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
            residual += cross.transpose() * delta_p;
        }
        let norm = residual.norm();
        if !norm.is_finite() {
            return f64::NAN;
        }
        max_residual = max_residual.max(norm);
    }
    max_residual
}

#[allow(clippy::too_many_arguments)]
fn feasibility_with_cholesky(
    ba: &BundleAdjustment,
    system: &NormalEquationsBa,
    lambda: f64,
    schur: &DMatrix<f64>,
    rhs: &DVector<f64>,
    delta_pose: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
    implicit_pose_true_residual: Option<f64>,
    factors: &[LandmarkCholesky],
) -> Feasibility {
    let mut result = feasibility(
        ba,
        system,
        lambda,
        schur,
        rhs,
        delta_pose,
        delta_landmarks,
        implicit_pose_true_residual,
    );
    result.max_landmark_backsub_residual =
        cholesky_backsub_residual(system, lambda, factors, delta_pose, delta_landmarks);
    result.feasible = result.finite
        && result.pose_true_residual.is_finite()
        && result.max_landmark_backsub_residual.is_finite()
        && result.geometry_feasible;
    result
}

fn original_pose_equation_residual(
    system: &NormalEquationsBa,
    lambda: f64,
    delta_pose: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
) -> Option<f64> {
    let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
        return None;
    };
    if delta_pose.len() != diagonal.len() * 6 || delta_landmarks.len() != system.landmarks.len() * 3
    {
        return None;
    }
    let mut residual = system.b_p.clone();
    for (pose, block) in diagonal.iter().enumerate() {
        let mut damped = *block;
        for component in 0..6 {
            damped[(component, component)] += lambda;
        }
        let delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
        let value = damped * delta;
        for component in 0..6 {
            residual[pose * 6 + component] += value[component];
        }
    }
    for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
        let delta: Vector3<f64> = delta_landmarks
            .fixed_rows::<3>(landmark_index * 3)
            .into_owned();
        for (pose, cross) in &landmark.cross {
            let value: Vector6<f64> = cross * delta;
            for component in 0..6 {
                residual[pose * 6 + component] += value[component];
            }
        }
    }
    let norm = residual.norm();
    norm.is_finite().then_some(norm)
}

fn schur_action_scales(
    system: &NormalEquationsBa,
    lambda: f64,
    x: &DVector<f64>,
) -> Result<(f64, f64, f64), String> {
    let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
        return Err("Schur scale oracle requires pose blocks".to_owned());
    };
    if x.len() != diagonal.len() * 6 {
        return Err("Schur scale probe has the wrong dimension".to_owned());
    }
    let mut base = DVector::<f64>::zeros(x.len());
    for (pose, block) in diagonal.iter().enumerate() {
        let mut damped = *block;
        for component in 0..6 {
            damped[(component, component)] += lambda;
        }
        let value: Vector6<f64> = damped * x.fixed_rows::<6>(pose * 6).into_owned();
        base.fixed_rows_mut::<6>(pose * 6).copy_from(&value);
    }
    let mut eliminated = DVector::<f64>::zeros(x.len());
    for landmark in &system.landmarks {
        let mut h_ll = landmark.h_ll;
        for component in 0..3 {
            h_ll[(component, component)] += lambda;
        }
        let Some(inverse) = h_ll.try_inverse() else {
            continue;
        };
        let mut projected = Vector3::zeros();
        for (pose, cross) in &landmark.cross {
            let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
            projected += cross.transpose() * x_pose;
        }
        let reduced = inverse * projected;
        for (pose, cross) in &landmark.cross {
            let value: Vector6<f64> = cross * reduced;
            for component in 0..6 {
                eliminated[pose * 6 + component] += value[component];
            }
        }
    }
    let base_norm = base.norm();
    let eliminated_norm = eliminated.norm();
    let arithmetic_scale = (base_norm + eliminated_norm).max(1.0);
    if ![base_norm, eliminated_norm, arithmetic_scale]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err("Schur arithmetic scale is non-finite".to_owned());
    }
    Ok((base_norm, eliminated_norm, arithmetic_scale))
}

#[allow(clippy::too_many_arguments)]
fn oracle_failure_reports(
    lambda: f64,
    pose_blocks: usize,
    landmark_count: usize,
    observation_count: usize,
    dimension: usize,
    singular_landmarks: usize,
    raw_schur_asymmetry: Option<f64>,
    schur_base_action_norm: Option<f64>,
    schur_eliminated_action_norm: Option<f64>,
    schur_arithmetic_scale: Option<f64>,
    status: String,
) -> Vec<OracleCaseReport> {
    [128_usize, 512_usize]
        .into_iter()
        .map(|pcg_max_iterations| OracleCaseReport {
            lambda,
            pcg_max_iterations,
            pose_blocks,
            landmark_count,
            observation_count,
            schur_dimension: dimension,
            singular_landmarks,
            raw_schur_asymmetry,
            schur_base_action_norm,
            schur_eliminated_action_norm,
            schur_arithmetic_scale,
            operator_raw_action_error: None,
            operator_raw_action_relative_error: None,
            operator_lower_action_error: None,
            operator_lower_action_relative_error: None,
            operator_rhs_error: None,
            direct_explicit_pose_error: None,
            direct_explicit_landmark_error: None,
            direct_feasibility: None,
            explicit_feasibility: None,
            direct_prediction: None,
            explicit_prediction: None,
            pcg_status: status.clone(),
            pcg_iterations: None,
            pcg_true_residual: None,
            pcg_target: None,
            matrix_free_pose_error: None,
            matrix_free_landmark_error: None,
            matrix_free_feasibility: None,
            matrix_free_prediction: None,
        })
        .collect()
}

fn run_oracle(ba: &BundleAdjustment) -> Result<Vec<OracleCaseReport>, String> {
    let (system, pose_blocks, landmark_count, observation_count) = build_oracle_system(ba)?;
    let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
        return Err("oracle system did not retain pose blocks".to_owned());
    };
    let mut reports = Vec::new();
    for lambda in [1.0e-4, 1.0e10] {
        let (raw_schur, explicit_rhs, singular_landmarks) =
            match explicit_schur_rhs(&system, lambda) {
                Ok(value) => value,
                Err(error) => {
                    reports.extend(oracle_failure_reports(
                        lambda,
                        pose_blocks,
                        landmark_count,
                        observation_count,
                        pose_blocks * 6,
                        0,
                        None,
                        None,
                        None,
                        None,
                        format!("oracle_failure:explicit_schur:{error}"),
                    ));
                    continue;
                }
            };
        let raw_asymmetry = schur_asymmetry(&raw_schur);
        let dimension = raw_schur.nrows();
        let probe = DVector::from_iterator(
            dimension,
            (0..dimension)
                .map(|index| 0.001 * ((index % 17) as f64 - 8.0) + 0.00001 * index as f64),
        );
        let mut lower_schur = raw_schur.clone();
        mirror_lower_triangle(&mut lower_schur);
        let lower_action = &lower_schur * &probe;
        let raw_action = &raw_schur * &probe;
        let scale_metrics = schur_action_scales(&system, lambda, &probe).ok();
        let schur_base_action_norm = scale_metrics.map(|metrics| metrics.0);
        let schur_eliminated_action_norm = scale_metrics.map(|metrics| metrics.1);
        let schur_arithmetic_scale = scale_metrics.map(|metrics| metrics.2);
        let mut solve_failures = Vec::new();
        let explicit_pose = match solve_normal_equations(&lower_schur, &explicit_rhs) {
            Ok(solution) => Some(solution),
            Err(error) => {
                solve_failures.push(format!("explicit_schur_factor:{error:?}"));
                None
            }
        };
        let explicit_operator = match implicit_schur::ImplicitSchurOperator::new(&system, lambda) {
            Ok(operator) => operator,
            Err(error) => {
                reports.extend(oracle_failure_reports(
                    lambda,
                    pose_blocks,
                    landmark_count,
                    observation_count,
                    dimension,
                    singular_landmarks,
                    Some(raw_asymmetry),
                    schur_base_action_norm,
                    schur_eliminated_action_norm,
                    schur_arithmetic_scale,
                    format!("oracle_failure:implicit_operator:{error:?}"),
                ));
                continue;
            }
        };
        let operator_action = match explicit_operator.apply(&probe) {
            Ok(action) => action,
            Err(error) => {
                reports.extend(oracle_failure_reports(
                    lambda,
                    pose_blocks,
                    landmark_count,
                    observation_count,
                    dimension,
                    singular_landmarks,
                    Some(raw_asymmetry),
                    schur_base_action_norm,
                    schur_eliminated_action_norm,
                    schur_arithmetic_scale,
                    format!("oracle_failure:implicit_apply:{error:?}"),
                ));
                continue;
            }
        };
        let operator_rhs_error = (explicit_operator.rhs() - &explicit_rhs).norm();
        let operator_raw_action_error = (&operator_action - &raw_action).norm();
        let operator_lower_action_error = (&operator_action - &lower_action).norm();
        let operator_raw_action_relative_error =
            schur_arithmetic_scale.map(|scale| operator_raw_action_error / scale);
        let operator_lower_action_relative_error =
            schur_arithmetic_scale.map(|scale| operator_lower_action_error / scale);
        let mut direct_cache = None;
        let direct_solution = match solve_step_pose_blocks(
            &system,
            diagonal.clone(),
            pose_blocks,
            landmark_count,
            lambda,
            &mut direct_cache,
        ) {
            Ok(solution) => Some(solution),
            Err(error) => {
                solve_failures.push(format!("direct_schur_factor:{error:?}"));
                None
            }
        };
        let direct_pose = direct_solution.as_ref().map(|solution| &solution.0);
        let direct_landmarks = direct_solution.as_ref().map(|solution| &solution.1);
        let explicit_landmarks = explicit_pose
            .as_ref()
            .and_then(|pose| explicit_operator.complete_delta(pose).ok());
        let direct_implicit_true_residual = direct_pose.and_then(|pose| {
            explicit_operator
                .apply(pose)
                .ok()
                .map(|applied| (explicit_operator.rhs() - applied).norm())
        });
        let explicit_implicit_true_residual = explicit_pose.as_ref().and_then(|pose| {
            explicit_operator
                .apply(pose)
                .ok()
                .map(|applied| (explicit_operator.rhs() - applied).norm())
        });
        let direct_feasibility = match (direct_pose, direct_landmarks) {
            (Some(pose), Some(landmarks)) => Some(feasibility(
                ba,
                &system,
                lambda,
                &lower_schur,
                &explicit_rhs,
                pose,
                landmarks,
                direct_implicit_true_residual,
            )),
            _ => None,
        };
        let explicit_feasibility = match (explicit_pose.as_ref(), explicit_landmarks.as_ref()) {
            (Some(pose), Some(landmarks)) => Some(feasibility(
                ba,
                &system,
                lambda,
                &lower_schur,
                &explicit_rhs,
                pose,
                landmarks,
                explicit_implicit_true_residual,
            )),
            _ => None,
        };
        let direct_prediction = match (direct_pose, direct_landmarks) {
            (Some(pose), Some(landmarks)) => {
                predicted_decrease(&system, lambda, pose, landmarks).ok()
            }
            _ => None,
        };
        let explicit_prediction = match (explicit_pose.as_ref(), explicit_landmarks.as_ref()) {
            (Some(pose), Some(landmarks)) => {
                predicted_decrease(&system, lambda, pose, landmarks).ok()
            }
            _ => None,
        };
        let direct_explicit_pose_error = match (direct_pose, explicit_pose.as_ref()) {
            (Some(direct), Some(explicit)) => Some((direct - explicit).norm()),
            _ => None,
        };
        let direct_explicit_landmark_error = match (direct_landmarks, explicit_landmarks.as_ref()) {
            (Some(direct), Some(explicit)) => Some((direct - explicit).norm()),
            _ => None,
        };
        for pcg_max_iterations in [128_usize, 512_usize] {
            let pcg_result = explicit_operator.solve_pcg(
                explicit_operator.rhs(),
                implicit_schur::PcgOptions {
                    max_iterations: pcg_max_iterations,
                    relative_tolerance: ORACLE_PC_TOLERANCE,
                    absolute_tolerance: ORACLE_PC_TOLERANCE,
                },
            );
            let (
                pcg_status,
                pcg_iterations,
                pcg_true_residual,
                pcg_target,
                matrix_free_pose_error,
                matrix_free_landmark_error,
                matrix_free_feasibility,
                matrix_free_prediction,
            ) = match pcg_result {
                Ok(result) => match explicit_operator.apply(&result.solution) {
                    Err(error) => (
                        format!("failure:recheck_true_residual:{error:?}"),
                        Some(result.iterations),
                        None,
                        Some(result.target),
                        None,
                        None,
                        None,
                        None,
                    ),
                    Ok(applied) => {
                        let true_residual = explicit_operator.rhs() - applied;
                        match explicit_operator.complete_delta(&result.solution) {
                            Err(error) => (
                                format!("failure:landmark_backsub:{error:?}"),
                                Some(result.iterations),
                                Some(true_residual.norm()),
                                Some(result.target),
                                None,
                                None,
                                None,
                                None,
                            ),
                            Ok(matrix_free_landmarks) => {
                                let feasibility = feasibility(
                                    ba,
                                    &system,
                                    lambda,
                                    &lower_schur,
                                    &explicit_rhs,
                                    &result.solution,
                                    &matrix_free_landmarks,
                                    Some(true_residual.norm()),
                                );
                                match predicted_decrease(
                                    &system,
                                    lambda,
                                    &result.solution,
                                    &matrix_free_landmarks,
                                ) {
                                    Err(error) => (
                                        format!("failure:prediction:{error}"),
                                        Some(result.iterations),
                                        Some(true_residual.norm()),
                                        Some(result.target),
                                        explicit_pose
                                            .as_ref()
                                            .map(|pose| (&result.solution - pose).norm()),
                                        explicit_landmarks.as_ref().map(|landmarks| {
                                            (&matrix_free_landmarks - landmarks).norm()
                                        }),
                                        Some(feasibility),
                                        None,
                                    ),
                                    Ok(prediction) => (
                                        "success".to_owned(),
                                        Some(result.iterations),
                                        Some(true_residual.norm()),
                                        Some(result.target),
                                        explicit_pose
                                            .as_ref()
                                            .map(|pose| (&result.solution - pose).norm()),
                                        explicit_landmarks.as_ref().map(|landmarks| {
                                            (&matrix_free_landmarks - landmarks).norm()
                                        }),
                                        Some(feasibility),
                                        Some(prediction),
                                    ),
                                }
                            }
                        }
                    }
                },
                Err(error) => {
                    let (iterations, residual, target) = error.diagnostics();
                    (
                        format!("failure:{error:?}"),
                        iterations,
                        residual,
                        target,
                        None,
                        None,
                        None,
                        None,
                    )
                }
            };
            let pcg_status = if solve_failures.is_empty() {
                pcg_status
            } else {
                format!("{};pcg={}", solve_failures.join(","), pcg_status)
            };
            reports.push(OracleCaseReport {
                lambda,
                pcg_max_iterations,
                pose_blocks,
                landmark_count,
                observation_count,
                schur_dimension: dimension,
                singular_landmarks,
                raw_schur_asymmetry: Some(raw_asymmetry),
                schur_base_action_norm,
                schur_eliminated_action_norm,
                schur_arithmetic_scale,
                operator_raw_action_error: Some(operator_raw_action_error),
                operator_raw_action_relative_error,
                operator_lower_action_error: Some(operator_lower_action_error),
                operator_lower_action_relative_error,
                operator_rhs_error: Some(operator_rhs_error),
                direct_explicit_pose_error,
                direct_explicit_landmark_error,
                direct_feasibility: direct_feasibility.clone(),
                explicit_feasibility: explicit_feasibility.clone(),
                direct_prediction: direct_prediction.clone(),
                explicit_prediction: explicit_prediction.clone(),
                pcg_status,
                pcg_iterations,
                pcg_true_residual,
                pcg_target,
                matrix_free_pose_error,
                matrix_free_landmark_error,
                matrix_free_feasibility,
                matrix_free_prediction,
            });
        }
    }
    Ok(reports)
}

fn synthetic_rig_problem() -> BundleAdjustment {
    let camera = Camera::pinhole(1, 640, 480, 420.0, 418.0, 320.0, 240.0);
    let sensor_one = SE3::new(
        nalgebra::UnitQuaternion::from_euler_angles(0.01, -0.02, 0.03),
        Vector3::new(0.2, -0.01, 0.02),
    );
    let truth_poses = [
        Pose::identity(),
        Pose::from_world_to_camera(
            nalgebra::UnitQuaternion::from_euler_angles(0.01, -0.015, 0.02),
            Vector3::new(-0.18, 0.01, 0.03),
        ),
    ];
    let mut ba = BundleAdjustment::new(camera.clone());
    ba.add_pose(0, truth_poses[0].clone());
    ba.add_pose(
        1,
        Pose::from_world_to_camera(
            nalgebra::UnitQuaternion::from_euler_angles(0.013, -0.012, 0.018),
            Vector3::new(-0.20, 0.02, 0.04),
        ),
    );
    ba.fix_pose(0);
    for id in 0..8_u64 {
        let truth = Point3::new(
            -0.8 + 0.23 * id as f64,
            -0.4 + 0.11 * (id % 4) as f64,
            4.0 + 0.3 * id as f64,
        );
        ba.add_landmark(
            id,
            Point3::from(truth.coords + Vector3::new(0.004, -0.003, 0.006)),
        );
        for (frame_id, pose) in truth_poses.iter().enumerate() {
            for sensor_from_rig in [SE3::identity(), sensor_one.clone()] {
                let sensor_pose = sensor_from_rig.compose(&pose.world_to_camera);
                let xy = camera
                    .project(&sensor_pose.transform_point(&truth))
                    .expect("synthetic rig point must project");
                ba.add_rig_observation(BaRigObservation {
                    keyframe_id: frame_id as u64,
                    landmark_id: id,
                    xy,
                    camera: camera.clone(),
                    sensor_from_rig,
                });
            }
        }
    }
    ba
}

fn non_diagonal_landmark_system() -> NormalEquationsBa {
    let h_pp = Matrix6::identity() * 20.0;
    let cross = Matrix6x3::from_row_slice(&[
        0.8, -0.2, 0.1, 0.0, 0.4, -0.3, 0.2, 0.1, 0.5, -0.1, 0.3, 0.2, 0.6, -0.4, 0.2, 0.1, 0.2,
        0.7,
    ]);
    NormalEquationsBa {
        h_pp: CameraHessian::PoseDiagonal(vec![h_pp]),
        b_p: DVector::from_iterator(6, (0..6).map(|index| 0.2 + index as f64 * 0.03)),
        landmarks: vec![LandmarkBlock {
            h_ll: Matrix3::new(8.0, 1.0, 0.5, 1.0, 6.0, -0.3, 0.5, -0.3, 7.0),
            b_l: Vector3::new(0.4, -0.7, 0.2),
            cross: vec![(0, cross)],
        }],
    }
}

#[test]
fn cholesky_landmark_arm_matches_general_inverse_for_nondiagonal_block() {
    let system = non_diagonal_landmark_system();
    for lambda in [0.0, 0.5, 1.0e5] {
        let general = implicit_schur::ImplicitSchurOperator::new(&system, lambda).unwrap();
        let (cholesky, metrics) = CholeskySchurOperator::new(&system, lambda).unwrap();
        assert!(metrics.h_ll_max_asymmetry < 1.0e-12);
        assert!(metrics.inverse_max_asymmetry.unwrap() < 1.0e-12);
        assert!(metrics.h_ll_inverse_identity_residual.unwrap().is_finite());
        assert!((general.rhs() - cholesky.rhs()).norm() < 1.0e-10);
        let probe = DVector::from_iterator(6, (0..6).map(|index| 0.1 + index as f64 * 0.07));
        assert!(
            (general.apply(&probe).unwrap() - cholesky.apply(&probe).unwrap()).norm() < 1.0e-10
        );
        assert!(
            (general.apply_preconditioner(&probe).unwrap()
                - cholesky.apply_preconditioner(&probe).unwrap())
            .norm()
                < 1.0e-10
        );
        assert!(
            (general.complete_delta(&probe).unwrap() - cholesky.complete_delta(&probe).unwrap())
                .norm()
                < 1.0e-10
        );

        let (general_raw, general_rhs, _) = explicit_schur_rhs(&system, lambda).unwrap();
        let (cholesky_raw, cholesky_rhs) = explicit_cholesky_schur_rhs(&system, &cholesky).unwrap();
        let mut general_lower = general_raw;
        let mut cholesky_lower = cholesky_raw;
        mirror_lower_triangle(&mut general_lower);
        mirror_lower_triangle(&mut cholesky_lower);
        assert!((general_rhs - cholesky_rhs).norm() < 1.0e-10);
        assert!((general_lower - cholesky_lower).norm() < 1.0e-10);
    }
}

#[test]
fn cholesky_landmark_arm_rejects_asymmetric_input_without_fallback() {
    let mut system = non_diagonal_landmark_system();
    if let Some(landmark) = system.landmarks.first_mut() {
        landmark.h_ll[(0, 1)] += 1.0e-6;
    }
    let error = match CholeskySchurOperator::new(&system, 0.5) {
        Ok(_) => panic!("asymmetric landmark block must not use a Cholesky fallback"),
        Err(error) => error,
    };
    assert!(error.reason.contains("not symmetric"));
}

#[test]
fn cholesky_landmark_arm_rejects_symmetric_non_spd_input_without_general_fallback() {
    let mut system = non_diagonal_landmark_system();
    if let Some(landmark) = system.landmarks.first_mut() {
        landmark.h_ll = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        landmark.cross = vec![(0, Matrix6x3::zeros())];
    }
    assert!(implicit_schur::ImplicitSchurOperator::new(&system, 0.1).is_ok());
    let error = match CholeskySchurOperator::new(&system, 0.1) {
        Ok(_) => panic!("symmetric non-SPD block must not use a general fallback"),
        Err(error) => error,
    };
    assert!(error.reason.contains("not SPD"));
}

#[test]
fn synthetic_oracle_compares_both_damping_values_and_coefficients() {
    let reports = run_oracle(&synthetic_rig_problem()).unwrap();
    assert_eq!(reports.len(), 4);
    for report in reports {
        assert!(report.raw_schur_asymmetry.unwrap().is_finite());
        assert!(report.operator_rhs_error.unwrap() < 1.0e-8);
        assert!(report.operator_raw_action_relative_error.unwrap() < 1.0e-8);
        assert!(
            report.operator_lower_action_error.unwrap().is_finite(),
            "lower-mirrored operator comparison must remain finite: {report:?}"
        );
        assert!(report.direct_explicit_pose_error.unwrap() < 1.0e-7);
        assert!(report.direct_explicit_landmark_error.unwrap() < 1.0e-7);
        let direct_feasibility = report
            .direct_feasibility
            .as_ref()
            .expect("synthetic direct arm should succeed");
        let explicit_feasibility = report
            .explicit_feasibility
            .as_ref()
            .expect("synthetic explicit arm should succeed");
        assert!(direct_feasibility.feasible);
        assert!(explicit_feasibility.feasible);
        assert!(direct_feasibility.geometry_feasible);
        assert!(explicit_feasibility.geometry_feasible);
        assert_eq!(direct_feasibility.geometry_invalid_observations, 0);
        assert_eq!(explicit_feasibility.geometry_invalid_observations, 0);
        assert!(direct_feasibility
            .implicit_pose_true_residual
            .is_some_and(f64::is_finite));
        assert!(explicit_feasibility
            .implicit_pose_true_residual
            .is_some_and(f64::is_finite));
        let direct_prediction = report
            .direct_prediction
            .as_ref()
            .expect("synthetic direct prediction should succeed");
        let explicit_prediction = report
            .explicit_prediction
            .as_ref()
            .expect("synthetic explicit prediction should succeed");
        assert!(
            (direct_prediction.squared_damped - 2.0 * direct_prediction.half_damped).abs() < 1.0e-9
        );
        assert!(
            (explicit_prediction.squared_damped - 2.0 * explicit_prediction.half_damped).abs()
                < 1.0e-9
        );
        if let Some(prediction) = report.matrix_free_prediction {
            assert!((prediction.squared_damped - 2.0 * prediction.half_damped).abs() < 1.0e-9);
        }
    }

    let isolation_reports = run_explicit_pcg_isolation(&synthetic_rig_problem()).unwrap();
    assert_eq!(isolation_reports.len(), 6);
    let mut successful_explicit_arms = 0;
    for report in isolation_reports {
        assert!(report.target.is_some_and(f64::is_finite));
        assert!(report.implicit_pcg_target.is_some_and(f64::is_finite));
        if report.status == "success_lower_pcg" {
            successful_explicit_arms += 1;
            assert!(report.lower_true_residual.is_some_and(f64::is_finite));
            assert!(report.implicit_true_residual.is_some_and(f64::is_finite));
            assert!(report.feasibility.is_some());
            assert!(report.prediction.is_some());
        }
        if report.implicit_pcg_status == "success" {
            assert!(report
                .implicit_pcg_true_residual
                .is_some_and(f64::is_finite));
        }
        if report.lambda == 1.0e10 {
            assert_eq!(report.status, "success_lower_pcg");
            assert_eq!(report.implicit_pcg_status, "success");
            let explicit_target = report.target.expect("high-lambda explicit target");
            let explicit_lower_residual = report
                .lower_true_residual
                .expect("high-lambda explicit lower residual");
            assert!(explicit_lower_residual <= explicit_target);
            assert!(report
                .implicit_true_residual
                .expect("high-lambda explicit implicit residual")
                .is_finite());
            assert!(report
                .dense_lower_true_residual
                .expect("high-lambda dense lower residual")
                .is_finite());
            assert!(report
                .dense_implicit_true_residual
                .expect("high-lambda dense implicit residual")
                .is_finite());
            let implicit_target = report
                .implicit_pcg_target
                .expect("high-lambda implicit target");
            let implicit_residual = report
                .implicit_pcg_true_residual
                .expect("high-lambda implicit residual");
            assert!(implicit_residual <= implicit_target);
            assert!(
                report
                    .pose_error_vs_dense
                    .expect("high-lambda explicit dense pose delta")
                    < 1.0e-7
            );
            assert!(
                report
                    .landmark_error_vs_dense
                    .expect("high-lambda explicit dense landmark delta")
                    < 1.0e-7
            );
            assert!(
                report
                    .explicit_vs_implicit_pose_error
                    .expect("high-lambda pose arm comparison")
                    < 1.0e-7
            );
            assert!(
                report
                    .explicit_vs_implicit_landmark_error
                    .expect("high-lambda landmark arm comparison")
                    < 1.0e-7
            );
            assert!(
                report
                    .feasibility
                    .as_ref()
                    .expect("high-lambda explicit feasibility")
                    .feasible
            );
            assert!(report
                .prediction
                .as_ref()
                .expect("high-lambda explicit prediction")
                .squared_damped
                .is_finite());
        }
    }
    assert!(successful_explicit_arms > 0);

    let cholesky_reports = run_cholesky_pcg_isolation(&synthetic_rig_problem()).unwrap();
    assert_eq!(cholesky_reports.len(), 6);
    for report in cholesky_reports {
        assert!(report.general_metrics.h_ll_max_scale.is_finite());
        assert!(report.cholesky_metrics.h_ll_max_scale.is_finite());
        assert!(report.cholesky_metrics.h_ll_max_asymmetry < 1.0e-12);
        assert!(report.rhs_error.is_some_and(|error| error < 1.0e-8));
        assert!(report
            .probe_action_error
            .is_some_and(|error| error < 1.0e-8));
        assert!(report
            .probe_preconditioner_error
            .is_some_and(|error| error < 1.0e-8));
        if report.lambda == 1.0e10 {
            assert_eq!(report.general.status, "general_pcg");
            assert_eq!(report.cholesky.status, "cholesky_pcg");
            assert!(report
                .general
                .pose_error_vs_dense
                .is_some_and(|error| error < 1.0e-7));
            assert!(report
                .cholesky
                .pose_error_vs_dense
                .is_some_and(|error| error < 1.0e-7));
            assert!(report
                .general
                .landmark_error_vs_dense
                .is_some_and(|error| error < 1.0e-7));
            assert!(report
                .cholesky
                .landmark_error_vs_dense
                .is_some_and(|error| error < 1.0e-7));
            assert!(report
                .general_vs_cholesky_pose_error
                .is_some_and(|error| error < 1.0e-7));
            assert!(report
                .general_vs_cholesky_landmark_error
                .is_some_and(|error| error < 1.0e-7));
            assert!(report
                .general
                .original_pose_equation_residual
                .is_some_and(f64::is_finite));
            assert!(report
                .cholesky
                .original_pose_equation_residual
                .is_some_and(f64::is_finite));
            assert!(report
                .general
                .dense_reference_action_true_residual
                .is_some_and(f64::is_finite));
            assert!(report
                .cholesky
                .dense_reference_action_true_residual
                .is_some_and(f64::is_finite));
            assert!(report
                .general
                .dense_reference_original_pose_equation_residual
                .is_some_and(f64::is_finite));
            assert!(report
                .cholesky
                .dense_reference_original_pose_equation_residual
                .is_some_and(f64::is_finite));
            assert!(report
                .general
                .feasibility
                .as_ref()
                .is_some_and(|value| value.feasible));
            assert!(report
                .cholesky
                .feasibility
                .as_ref()
                .is_some_and(|value| value.feasible));
            assert!(report.general.prediction.is_some());
            assert!(report.cholesky.prediction.is_some());
        }
    }
}

#[test]
fn predicted_decrease_matches_independent_squared_cost_fixture() {
    let mut cross = Matrix6x3::zeros();
    for index in 0..3 {
        cross[(index, index)] = 0.5;
    }
    let mut h_pp = Matrix6::identity();
    h_pp *= 2.0;
    let mut h_ll = Matrix3::identity();
    h_ll *= 3.0;
    let system = NormalEquationsBa {
        h_pp: CameraHessian::PoseDiagonal(vec![h_pp]),
        b_p: DVector::from_element(6, 1.0),
        landmarks: vec![LandmarkBlock {
            h_ll,
            b_l: Vector3::from_element(2.0),
            cross: vec![(0, cross)],
        }],
    };
    let delta_pose = DVector::from_element(6, 0.1);
    let delta_landmark = DVector::from_element(3, -0.2);
    let prediction = predicted_decrease(&system, 0.5, &delta_pose, &delta_landmark).unwrap();
    assert!((prediction.squared_undamped - 0.78).abs() < 1.0e-12);
    assert!((prediction.half_damped - 0.345).abs() < 1.0e-12);
    assert!((prediction.squared_damped - 0.69).abs() < 1.0e-12);
}

#[test]
fn fixture_quaternion_reader_preserves_serialized_bits() {
    let values = [
        0.9238795325112867_f64,
        0.0_f64,
        0.3826834323650898_f64,
        0.0_f64,
        0.125_f64,
        -0.25_f64,
        0.5_f64,
    ];
    let serialized = values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>();
    let fields = serialized.iter().map(String::as_str).collect::<Vec<_>>();
    let parsed = parse_se3(&fields, 0, "bit-roundtrip").unwrap();
    let quaternion = parsed.rotation.quaternion();
    assert_eq!(quaternion.w.to_bits(), values[0].to_bits());
    assert_eq!(quaternion.i.to_bits(), values[1].to_bits());
    assert_eq!(quaternion.j.to_bits(), values[2].to_bits());
    assert_eq!(quaternion.k.to_bits(), values[3].to_bits());
    assert_eq!(parsed.translation.x.to_bits(), values[4].to_bits());
    assert_eq!(parsed.translation.y.to_bits(), values[5].to_bits());
    assert_eq!(parsed.translation.z.to_bits(), values[6].to_bits());

    let near_unit = [
        1.0 + f64::EPSILON,
        -0.0_f64,
        0.0_f64,
        0.0_f64,
        0.0_f64,
        0.0_f64,
        0.0_f64,
    ];
    let serialized = near_unit
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>();
    let fields = serialized.iter().map(String::as_str).collect::<Vec<_>>();
    let parsed = parse_se3(&fields, 0, "near-unit-bit-roundtrip").unwrap();
    let quaternion = parsed.rotation.quaternion();
    assert_eq!(quaternion.w.to_bits(), near_unit[0].to_bits());
    assert_eq!(quaternion.i.to_bits(), near_unit[1].to_bits());
}

#[test]
fn fixture_reader_rejects_bad_order_and_oversized_header() {
    let base = std::env::temp_dir().join(format!("visloc-ba-oracle-parser-{}", std::process::id()));
    let _ = std::fs::remove_file(&base);
    std::fs::write(&base, "END\n").unwrap();
    let error = parse_fixture(&base).unwrap_err();
    assert!(error.contains("header must be the first record"));
    std::fs::write(&base, "VISLOC_BA_ORACLE_FIXTURE 1\nPOSE_COUNT 514\nEND\n").unwrap();
    let error = parse_fixture(&base).unwrap_err();
    assert!(error.contains("exceeds cap"));
    std::fs::write(&base, "VISLOC_BA_ORACLE_FIXTURE 1\nEND\nPOSE_COUNT 1\n").unwrap();
    let error = parse_fixture(&base).unwrap_err();
    assert!(error.contains("records after END"));
    let _ = std::fs::remove_file(base);
}

#[test]
fn fixture_reader_roundtrips_initial_cost_bits() {
    let path = std::env::temp_dir().join(format!(
        "visloc-ba-oracle-valid-parser-{}",
        std::process::id()
    ));
    let hash = "0".repeat(64);
    let fixture = format!(
        "VISLOC_BA_ORACLE_FIXTURE 1\nSOURCE_SHA256 {hash}\nSOURCE_SHA256_CAMERAS {hash}\nSOURCE_SHA256_IMAGES {hash}\nSOURCE_SHA256_POINTS {hash}\nSOURCE_SHA256_MANIFEST {hash}\nINITIAL_COST 0.00000000000000000e+00\nINITIAL_COST_BITS 0\nCAMERA_COUNT 1\nPOSE_COUNT 1\nLANDMARK_COUNT 1\nOBSERVATION_COUNT 1\nCAMERA 1 PINHOLE 10 10 4 2 2 0 0\nPOSE 0 1 0 0 0 0 0 0\nFIXED_POSE 0\nLANDMARK 0 0 0 2\nRIG_OBSERVATION 0 0 0 0 1 1 0 0 0 0 0 0\nEND\n"
    );
    std::fs::write(&path, fixture).unwrap();
    let parsed = parse_fixture(&path).unwrap();
    assert_eq!(parsed.initial_cost.to_bits(), 0);
    assert_eq!(parsed.initial_cost_bits, 0);
    assert_eq!(parsed.ba.cost().to_bits(), 0);
    let _ = std::fs::remove_file(path);
}

#[test]
fn oracle_caps_reject_before_dense_dimension_allocation() {
    assert!(check_caps(512, 8_192, 262_144, MAX_CROSS_PAIR_WORK).is_ok());
    assert!(check_caps(513, 1, 1, 1).is_err());
    assert!(check_caps(1, 8_193, 1, 1).is_err());
    assert!(check_caps(1, 1, 262_145, 1).is_err());
    assert!(check_caps(1, 1, 1, MAX_CROSS_PAIR_WORK + 1).is_err());
    assert!(check_caps(513, 1, 1, 1).is_err());
}

#[test]
fn generic_test_pcg_matches_implicit_recurrence_and_failure_diagnostics() {
    let diagonal = vec![
        Matrix6::from_diagonal(&Vector6::from_element(20.0)),
        Matrix6::from_diagonal(&Vector6::from_element(22.0)),
    ];
    let mut cross = Matrix6x3::zeros();
    for component in 0..3 {
        cross[(component, component)] = 0.25;
    }
    let system = NormalEquationsBa {
        h_pp: CameraHessian::PoseDiagonal(diagonal),
        b_p: DVector::from_iterator(12, (0..12).map(|index| 0.03 * (index + 1) as f64)),
        landmarks: vec![LandmarkBlock {
            h_ll: Matrix3::from_diagonal(&Vector3::new(7.0, 8.0, 9.0)),
            b_l: Vector3::new(0.2, -0.1, 0.3),
            cross: vec![(0, cross), (1, -cross)],
        }],
    };
    let operator = implicit_schur::ImplicitSchurOperator::new(&system, 0.25).unwrap();
    let options = [
        implicit_schur::PcgOptions::default(),
        implicit_schur::PcgOptions {
            max_iterations: 1,
            relative_tolerance: 0.0,
            absolute_tolerance: 1.0e-30,
        },
    ];
    let mut saw_success = false;
    let mut saw_failure = false;
    for options in options {
        let expected = operator.solve_pcg(operator.rhs(), options);
        let actual = solve_test_pcg(
            operator.rhs(),
            operator.dimension(),
            options,
            |x| operator.apply(x),
            |residual| operator.apply_preconditioner(residual),
        );
        assert_eq!(actual, expected);
        match expected {
            Ok(_) => saw_success = true,
            Err(implicit_schur::ImplicitSchurError::MaxIterations { .. })
            | Err(implicit_schur::ImplicitSchurError::ResidualCheckFailed { .. }) => {
                saw_failure = true;
            }
            Err(error) => panic!("unexpected PCG diagnostic: {error:?}"),
        }
    }
    assert!(saw_success);
    assert!(saw_failure);
}

#[test]
#[ignore = "requires an explicitly exported frozen 1k fixture and source hash"]
fn ignored_real_fixture_runs_bounded_damping_oracle() {
    let path = env::var_os("VISLOC_MATRIX_FREE_ORACLE_FIXTURE")
        .expect("VISLOC_MATRIX_FREE_ORACLE_FIXTURE is required; fixture absence is failure");
    let fixture = parse_fixture(Path::new(&path)).unwrap();
    let expected_hash = env::var("VISLOC_MATRIX_FREE_ORACLE_EXPECTED_SOURCE_SHA256")
        .expect("VISLOC_MATRIX_FREE_ORACLE_EXPECTED_SOURCE_SHA256 is required");
    assert_eq!(
        fixture.source_hashes["SOURCE_SHA256"], expected_hash,
        "fixture source hash does not match the requested frozen input"
    );
    let reports = run_oracle(&fixture.ba).unwrap();
    assert_eq!(reports.len(), 4);
    println!(
        "matrix_free_oracle_input_cost={:.17e} bits={}",
        fixture.initial_cost, fixture.initial_cost_bits
    );
    for report in reports {
        println!("matrix_free_oracle {report:?}");
    }
    let isolation_reports = run_explicit_pcg_isolation(&fixture.ba).unwrap();
    assert_eq!(isolation_reports.len(), 6);
    for report in isolation_reports {
        println!("explicit_pcg_isolation {report:?}");
    }
    let cholesky_reports = run_cholesky_pcg_isolation(&fixture.ba).unwrap();
    assert_eq!(cholesky_reports.len(), 6);
    for report in cholesky_reports {
        println!("cholesky_pcg_isolation {report:?}");
    }
}
