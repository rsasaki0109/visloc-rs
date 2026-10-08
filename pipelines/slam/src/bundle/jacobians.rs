//! Monocular projection Jacobians and the visual-Jacobian audit.

use super::*;

/// How a monocular observation is projected by the pose/structure BA
/// (`optimize_weighted` and the solvers it dispatches to).
///
/// The residual and its Jacobian must use the same camera model as the cost
/// (`robust_cost_weighted`, which calls the distortion-aware
/// [`Camera::project`]); otherwise LM steps are computed for a lens that the
/// accept/reject test does not evaluate. Pinhole-family cameras whose only lens
/// terms are radial `[k1, k2]` (`Pinhole`, `SimplePinhole`, `SimpleRadial`,
/// `Radial`, and `OpenCv` without tangential terms) keep the two historical
/// closed forms bit-for-bit; every other model the BA admits (see
/// `BundleAdjustment::intrinsics`) goes through the camera's own analytic
/// Jacobian, [`Camera::project_with_point_jacobian`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum MonoProjection<'a> {
    /// Distortion-free pinhole: the historical closed form, bit-identical.
    Pinhole,
    /// Radial `1 + k1·r² + k2·r⁴` with an analytic Jacobian.
    Radial { k1: f64, k2: f64 },
    /// Any other lens (tangential Brown-Conrady, rational, equidistant
    /// fisheye, FOV, Double Sphere): the residual is [`Camera::project`]
    /// itself and the Jacobian its analytic derivative.
    Lens(&'a Camera),
}

impl<'a> MonoProjection<'a> {
    pub(super) fn for_camera(camera: &'a Camera) -> Self {
        if !is_radial_pinhole_family(camera) {
            return Self::Lens(camera);
        }
        match camera
            .radial_distortion()
            .filter(|&(k1, k2)| k1 != 0.0 || k2 != 0.0)
        {
            None => Self::Pinhole,
            Some((k1, k2)) => Self::Radial { k1, k2 },
        }
    }
}

/// Whether `camera` is a pinhole-family model whose [`Camera::project`] is the
/// plain pinhole optionally followed by radial `1 + k1·r² + k2·r⁴` (so
/// `project_pinhole` / [`MonoProjection::Radial`] reproduce it exactly, and the
/// rectified / general stereo terms, which use the plain pinhole, stay
/// meaningful).
fn is_radial_pinhole_family(camera: &Camera) -> bool {
    match camera.model {
        CameraModel::Pinhole
        | CameraModel::SimplePinhole
        | CameraModel::SimpleRadial
        | CameraModel::Radial => true,
        CameraModel::OpenCv => camera.tangential_distortion().is_none(),
        _ => false,
    }
}

/// Predicted pixel of a camera-frame point and the 2×3 Jacobian of that
/// prediction w.r.t. the point, for the lens model selected by `projection`
/// (every branch reproduces [`Camera::project`] exactly).
pub(super) fn mono_project_with_jacobian(
    projection: MonoProjection<'_>,
    intrinsics: &(f64, f64, f64, f64),
    xc: &Point3<f64>,
) -> Option<(Point2<f64>, Matrix2x3<f64>)> {
    if xc.z <= 0.0 {
        return None;
    }
    let (fx, fy, cx, cy) = *intrinsics;
    match projection {
        MonoProjection::Pinhole => {
            let predicted = project_pinhole(intrinsics, xc)?;
            let z_inv = 1.0 / xc.z;
            let mut j_pi = Matrix2x3::<f64>::zeros();
            j_pi[(0, 0)] = fx * z_inv;
            j_pi[(0, 1)] = 0.0;
            j_pi[(0, 2)] = -fx * xc.x * z_inv * z_inv;
            j_pi[(1, 0)] = 0.0;
            j_pi[(1, 1)] = fy * z_inv;
            j_pi[(1, 2)] = -fy * xc.y * z_inv * z_inv;
            Some((predicted, j_pi))
        }
        MonoProjection::Radial { k1, k2 } => {
            // Same arithmetic as `Camera::project`'s radial branch, so the
            // residual equals the one the cost evaluates.
            let x = xc.x / xc.z;
            let y = xc.y / xc.z;
            let r2 = x * x + y * y;
            let d = 1.0 + k1 * r2 + k2 * r2 * r2;
            let predicted = Point2::new(fx * (x * d) + cx, fy * (y * d) + cy);
            // J = diag(fx, fy) · D · ∂(x, y)/∂X_c with
            // D = [[d + 2x²g, 2xyg], [2xyg, d + 2y²g]], g = k1 + 2·k2·r²
            // (the formula of the joint intrinsics solve).
            let g = k1 + 2.0 * k2 * r2;
            let d11 = d + 2.0 * x * x * g;
            let d12 = 2.0 * x * y * g;
            let d22 = d + 2.0 * y * y * g;
            let z_inv = 1.0 / xc.z;
            let mut j_pi = Matrix2x3::<f64>::zeros();
            j_pi[(0, 0)] = fx * d11 * z_inv;
            j_pi[(0, 1)] = fx * d12 * z_inv;
            j_pi[(0, 2)] = -fx * (d11 * x + d12 * y) * z_inv;
            j_pi[(1, 0)] = fy * d12 * z_inv;
            j_pi[(1, 1)] = fy * d22 * z_inv;
            j_pi[(1, 2)] = -fy * (d12 * x + d22 * y) * z_inv;
            (predicted.coords.iter().all(|v| v.is_finite()) && j_pi.iter().all(|v| v.is_finite()))
                .then_some((predicted, j_pi))
        }
        MonoProjection::Lens(camera) => camera.project_with_point_jacobian(xc),
    }
}

pub(super) fn project_pinhole(
    intrinsics: &(f64, f64, f64, f64),
    xc: &Point3<f64>,
) -> Option<Point2<f64>> {
    if xc.z <= 0.0 {
        return None;
    }
    let (fx, fy, cx, cy) = *intrinsics;
    Some(Point2::new(fx * xc.x / xc.z + cx, fy * xc.y / xc.z + cy))
}

/// Maximum absolute and relative (symmetric, Frobenius-normalized) error
/// between two small Jacobians.  This is intentionally a diagnostic helper,
/// not part of the optimizer: it reports both an absolute error (important
/// when a derivative is close to zero) and a scale-free error (important when
/// comparing translation, rotation, point, and intrinsics columns).
fn jacobian_error<I>(pairs: I) -> (f64, f64)
where
    I: IntoIterator<Item = (f64, f64)>,
{
    let mut max_abs = 0.0_f64;
    let mut diff_squared = 0.0_f64;
    let mut scale_squared = 0.0_f64;
    for (analytic, numerical) in pairs {
        if !analytic.is_finite() || !numerical.is_finite() {
            return (f64::INFINITY, f64::INFINITY);
        }
        let diff = analytic - numerical;
        max_abs = max_abs.max(diff.abs());
        diff_squared += diff * diff;
        let scale = analytic.abs().max(numerical.abs());
        scale_squared += scale * scale;
    }
    let scale = scale_squared.sqrt().max(1.0e-15);
    (max_abs, diff_squared.sqrt() / scale)
}

/// Finite-difference audit of the visual pinhole residual Jacobians for one
/// observation.  The production assembly has two deliberately duplicated
/// fast paths (serial and rayon), so this helper mirrors their formulas while
/// evaluating the residual through the public `Camera::project` API.  It is
/// only called by the explicit `VISLOC_SFM_DEBUG_BA_JACOBIANS` diagnostic and
/// by focused unit tests; it never participates in normal BA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BaVisualJacobianCase {
    pub residual_norm: f64,
    pub depth: f64,
    pub pose_max_abs: f64,
    pub pose_relative: f64,
    pub pose_translation_max_abs: f64,
    pub pose_translation_relative: f64,
    pub pose_rotation_max_abs: f64,
    pub pose_rotation_relative: f64,
    pub landmark_max_abs: f64,
    pub landmark_relative: f64,
    pub intrinsics_max_abs: f64,
    pub intrinsics_relative: f64,
}

/// Compare the analytic right-pose, world-landmark, and pinhole-intrinsics
/// Jacobians to central differences at one state.  Intrinsics are reported for
/// the four-parameter pinhole/OpenCV layout; radial distortion is deliberately
/// rejected because the ordinary BA path uses a separate distortion-aware
/// Jacobian in its joint-intrinsics solver.
pub(crate) fn audit_visual_jacobian_case(
    camera: &Camera,
    pose: &Pose,
    point: &Point3<f64>,
    measured: &Point2<f64>,
    epsilon: f64,
) -> Option<BaVisualJacobianCase> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    if !matches!(camera.model, CameraModel::Pinhole | CameraModel::OpenCv)
        || camera.params.len() < 4
        || camera
            .radial_distortion()
            .is_some_and(|(k1, k2)| k1 != 0.0 || k2 != 0.0)
        || camera.tangential_distortion().is_some()
    {
        return None;
    }
    let intrinsics = camera.intrinsics()?;
    let point_camera = pose.transform_world_point(point);
    let predicted = camera.project(&point_camera)?;
    let residual = predicted - *measured;
    let j_projection = pinhole_projection_jacobian(&intrinsics, &point_camera)?;
    let rotation = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let mut dpoint_dpose = Matrix3x6::<f64>::zeros();
    dpoint_dpose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation);
    dpoint_dpose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation * skew(&point.coords)));
    let analytic_pose = j_projection * dpoint_dpose;
    let analytic_landmark = j_projection * rotation;
    let x = point_camera.x / point_camera.z;
    let y = point_camera.y / point_camera.z;
    let analytic_intrinsics = Matrix2x4::new(x, 0.0, 1.0, 0.0, 0.0, y, 0.0, 1.0);

    let mut numerical_pose = Matrix2x6::<f64>::zeros();
    for axis in 0..6 {
        let mut plus = pose.clone();
        let mut minus = pose.clone();
        let mut delta = Vector6::<f64>::zeros();
        delta[axis] = epsilon;
        plus.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&delta));
        delta[axis] = -epsilon;
        minus.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&delta));
        let plus = camera.project(&plus.transform_world_point(point))?;
        let minus = camera.project(&minus.transform_world_point(point))?;
        let derivative = (plus - minus) / (2.0 * epsilon);
        numerical_pose[(0, axis)] = derivative.x;
        numerical_pose[(1, axis)] = derivative.y;
    }

    let mut numerical_landmark = Matrix2x3::<f64>::zeros();
    for axis in 0..3 {
        let mut plus = point.coords;
        let mut minus = point.coords;
        plus[axis] += epsilon;
        minus[axis] -= epsilon;
        let plus = camera.project(&pose.transform_world_point(&Point3::from(plus)))?;
        let minus = camera.project(&pose.transform_world_point(&Point3::from(minus)))?;
        let derivative = (plus - minus) / (2.0 * epsilon);
        numerical_landmark[(0, axis)] = derivative.x;
        numerical_landmark[(1, axis)] = derivative.y;
    }

    let mut numerical_intrinsics = Matrix2x4::<f64>::zeros();
    for axis in 0..4 {
        let parameter_epsilon = epsilon * camera.params[axis].abs().max(1.0);
        let mut plus = camera.clone();
        let mut minus = camera.clone();
        plus.params[axis] += parameter_epsilon;
        minus.params[axis] -= parameter_epsilon;
        let plus = plus.project(&point_camera)?;
        let minus = minus.project(&point_camera)?;
        let derivative = (plus - minus) / (2.0 * parameter_epsilon);
        numerical_intrinsics[(0, axis)] = derivative.x;
        numerical_intrinsics[(1, axis)] = derivative.y;
    }

    let (pose_max_abs, pose_relative) = jacobian_error(
        analytic_pose
            .iter()
            .copied()
            .zip(numerical_pose.iter().copied()),
    );
    let pose_error = |columns: std::ops::Range<usize>| {
        jacobian_error((0..2).flat_map(|row| {
            columns
                .clone()
                .map(move |column| (analytic_pose[(row, column)], numerical_pose[(row, column)]))
        }))
    };
    let (pose_translation_max_abs, pose_translation_relative) = pose_error(0..3);
    let (pose_rotation_max_abs, pose_rotation_relative) = pose_error(3..6);
    let (landmark_max_abs, landmark_relative) = jacobian_error(
        analytic_landmark
            .iter()
            .copied()
            .zip(numerical_landmark.iter().copied()),
    );
    let (intrinsics_max_abs, intrinsics_relative) = jacobian_error(
        analytic_intrinsics
            .iter()
            .copied()
            .zip(numerical_intrinsics.iter().copied()),
    );
    Some(BaVisualJacobianCase {
        residual_norm: residual.norm(),
        depth: point_camera.z,
        pose_max_abs,
        pose_relative,
        pose_translation_max_abs,
        pose_translation_relative,
        pose_rotation_max_abs,
        pose_rotation_relative,
        landmark_max_abs,
        landmark_relative,
        intrinsics_max_abs,
        intrinsics_relative,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct BaVisualJacobianBucket {
    pub samples: usize,
    pub pose_max_abs: f64,
    pub pose_relative_max: f64,
    pub pose_translation_max_abs: f64,
    pub pose_translation_relative_max: f64,
    pub pose_rotation_max_abs: f64,
    pub pose_rotation_relative_max: f64,
    pub landmark_max_abs: f64,
    pub landmark_relative_max: f64,
    pub intrinsics_max_abs: f64,
    pub intrinsics_relative_max: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct BaVisualJacobianAudit {
    pub observations_seen: usize,
    pub samples_audited: usize,
    pub invalid_samples: usize,
    pub normal: BaVisualJacobianBucket,
    pub far_depth: BaVisualJacobianBucket,
    pub low_parallax: BaVisualJacobianBucket,
    pub high_residual: BaVisualJacobianBucket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BaJacobianBucketKind {
    Normal,
    FarDepth,
    LowParallax,
    HighResidual,
}

fn update_jacobian_bucket(bucket: &mut BaVisualJacobianBucket, case: &BaVisualJacobianCase) {
    bucket.samples += 1;
    bucket.pose_max_abs = bucket.pose_max_abs.max(case.pose_max_abs);
    bucket.pose_relative_max = bucket.pose_relative_max.max(case.pose_relative);
    bucket.pose_translation_max_abs = bucket
        .pose_translation_max_abs
        .max(case.pose_translation_max_abs);
    bucket.pose_translation_relative_max = bucket
        .pose_translation_relative_max
        .max(case.pose_translation_relative);
    bucket.pose_rotation_max_abs = bucket.pose_rotation_max_abs.max(case.pose_rotation_max_abs);
    bucket.pose_rotation_relative_max = bucket
        .pose_rotation_relative_max
        .max(case.pose_rotation_relative);
    bucket.landmark_max_abs = bucket.landmark_max_abs.max(case.landmark_max_abs);
    bucket.landmark_relative_max = bucket.landmark_relative_max.max(case.landmark_relative);
    bucket.intrinsics_max_abs = bucket.intrinsics_max_abs.max(case.intrinsics_max_abs);
    bucket.intrinsics_relative_max = bucket.intrinsics_relative_max.max(case.intrinsics_relative);
}

fn observation_jacobian_bucket(
    ba: &BundleAdjustment,
    obs_idx: usize,
    point_camera: &Point3<f64>,
    residual_norm: f64,
) -> BaJacobianBucketKind {
    // A large residual is the most useful disjoint bucket for checking the
    // robust-weighting path.  For geometric conditioning, estimate the widest
    // ray angle to another observation of the same landmark.  This is a
    // deterministic, diagnostic-only proxy; no optimizer decision uses it.
    if residual_norm > 10.0 {
        return BaJacobianBucketKind::HighResidual;
    }
    let observation = &ba.observations[obs_idx];
    let Some(anchor_pose) = ba.poses.get(&observation.keyframe_id) else {
        return BaJacobianBucketKind::Normal;
    };
    let point = &ba.landmarks[&observation.landmark_id];
    let anchor_ray = point.coords - anchor_pose.camera_center_world().coords;
    let mut max_angle = None;
    for (other_idx, other) in ba.observations.iter().enumerate() {
        if other_idx == obs_idx || other.landmark_id != observation.landmark_id {
            continue;
        }
        let Some(other_pose) = ba.poses.get(&other.keyframe_id) else {
            continue;
        };
        let other_ray = point.coords - other_pose.camera_center_world().coords;
        let (Some(a), Some(b)) = (
            anchor_ray.try_normalize(1.0e-15),
            other_ray.try_normalize(1.0e-15),
        ) else {
            continue;
        };
        let cosine = a.dot(&b).clamp(-1.0, 1.0);
        let angle = cosine.acos();
        if angle.is_finite() {
            max_angle = Some(max_angle.map_or(angle, |current: f64| current.max(angle)));
        }
    }
    if max_angle.is_some_and(|angle| angle.to_degrees() < 1.0) {
        BaJacobianBucketKind::LowParallax
    } else if point_camera.z > 100.0 {
        BaJacobianBucketKind::FarDepth
    } else {
        BaJacobianBucketKind::Normal
    }
}

/// Audit a deterministic, small sample from a live BA state.  The function is
/// intentionally `pub(crate)` so the incremental SFM diagnostic can invoke it
/// without exposing a new public solver API.  It returns no result used by the
/// optimizer and is never called unless the explicit debug environment flag is
/// enabled by the caller.
pub(crate) fn audit_bundle_visual_jacobians(
    ba: &BundleAdjustment,
    max_samples: usize,
) -> BaVisualJacobianAudit {
    let mut report = BaVisualJacobianAudit {
        observations_seen: ba.observations.len(),
        ..BaVisualJacobianAudit::default()
    };
    if max_samples == 0 {
        return report;
    }

    // Reserve an equal deterministic quota for each conditioning bucket, so a
    // long track ordered entirely by one region cannot hide the other cases.
    let quota = max_samples.div_ceil(4);
    let mut candidates: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::new());
    for (obs_idx, observation) in ba.observations.iter().enumerate() {
        let (Some(pose), Some(point)) = (
            ba.poses.get(&observation.keyframe_id),
            ba.landmarks.get(&observation.landmark_id),
        ) else {
            continue;
        };
        let point_camera = pose.transform_world_point(point);
        let Some(predicted) = ba.camera.project(&point_camera) else {
            continue;
        };
        let residual_norm = (predicted - observation.xy).norm();
        let kind = observation_jacobian_bucket(ba, obs_idx, &point_camera, residual_norm);
        let slot = match kind {
            BaJacobianBucketKind::Normal => 0,
            BaJacobianBucketKind::FarDepth => 1,
            BaJacobianBucketKind::LowParallax => 2,
            BaJacobianBucketKind::HighResidual => 3,
        };
        if candidates[slot].len() < quota {
            candidates[slot].push(obs_idx);
        }
    }

    for indices in candidates {
        for obs_idx in indices {
            if report.samples_audited >= max_samples {
                break;
            }
            let observation = &ba.observations[obs_idx];
            let (Some(pose), Some(point)) = (
                ba.poses.get(&observation.keyframe_id),
                ba.landmarks.get(&observation.landmark_id),
            ) else {
                report.invalid_samples += 1;
                continue;
            };
            let Some(case) =
                audit_visual_jacobian_case(&ba.camera, pose, point, &observation.xy, 1.0e-6)
            else {
                report.invalid_samples += 1;
                continue;
            };
            let point_camera = pose.transform_world_point(point);
            let kind = observation_jacobian_bucket(ba, obs_idx, &point_camera, case.residual_norm);
            match kind {
                BaJacobianBucketKind::Normal => update_jacobian_bucket(&mut report.normal, &case),
                BaJacobianBucketKind::FarDepth => {
                    update_jacobian_bucket(&mut report.far_depth, &case)
                }
                BaJacobianBucketKind::LowParallax => {
                    update_jacobian_bucket(&mut report.low_parallax, &case)
                }
                BaJacobianBucketKind::HighResidual => {
                    update_jacobian_bucket(&mut report.high_residual, &case)
                }
            }
            report.samples_audited += 1;
        }
    }
    report
}
