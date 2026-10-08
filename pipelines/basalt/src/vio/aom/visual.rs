//! Anchored visual reprojection factors: the f64 path and the f32 Sophus / Eigen bit-parity chain.

use super::*;

pub type Matrix2x6 = SMatrix<f64, 2, 6>;

/// Opt-in f32 visual-chain snapshot used to compare the current-value pose
/// path with the pinned Eigen/Sophus reference after an accepted LM step.
/// These fields are populated only when `VISLOC_BASALT_VISUAL_CHAIN_TRACE`
/// is set and never participate in the normal factor arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VisualChainF32 {
    /// `[qx, qy, qz, qw, tx, ty, tz]` for the intermediate rigid transforms.
    pub target_camera_from_imu: [f32; 7],
    pub target_imu_from_anchor_imu: [f32; 7],
    pub target_camera_from_anchor_imu: [f32; 7],
    /// FEJ relative camera transform used exclusively by d_rel/d_h.
    pub target_camera_from_anchor_imu_fej: [f32; 7],
    pub target_camera_from_anchor_camera: [f32; 7],
    /// Reconstructed with the same matrix builder as the production GEMV.
    pub target_camera_matrix: [f32; 16],
    /// `d_rel_d_h` in nalgebra's column-major storage order. This is only
    /// emitted by the opt-in chain trace while auditing the FEJ Jacobian.
    pub relative_wrt_anchor: [f32; 36],
    pub point4_target: [f32; 4],
    pub point_target: [f32; 3],
    pub projection: [f32; 2],
    /// Projection Jacobian in row-major order.
    pub projection_jacobian: [f32; 6],
}

/// Eigen/GCC's pinned scalar schedule for the float visual residual.  The
/// native `LandmarkBlockAbsDynamic<float, 6>` Double-Sphere path computes
/// `x*x` first, then contracts `y*y + x2` (`vfmadd231ss`); spelling that
/// reduction explicitly keeps the robust Huber boundary at native f32 bits
/// without changing the f64 compatibility path or IMU factors.
#[inline]
pub(super) fn eigen_vector2_squared_norm_f32(x: f32, y: f32) -> f32 {
    y.mul_add(y, x * x)
}

/// Literal visual row before it is scattered into the global AOM columns.
/// Upstream stores a relative-pose Jacobian and multiplies it by the host and
/// target absolute-pose Jacobians. Keeping those blocks explicit here makes
/// the host contribution (missing from the former world-XYZ factor) auditable.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchoredVisualFactor {
    /// The raw pixel residual (`projection - observation`) before robust
    /// whitening.  This is retained for the frame-4 parity audit; the solver
    /// still consumes `residual` below.
    pub raw_residual: Vector2<f64>,
    /// Projection at the linearization point, in pixel coordinates.
    pub projection: Vector2<f64>,
    /// Unsquared Huber weight used for the row stack.
    pub huber_weight: f64,
    /// Scalar multiplying raw residual/Jacobians (`sqrt(huber_weight) /
    /// observation_stddev`).
    pub sqrt_weight: f64,
    pub residual: Vector2<f64>,
    pub anchor_pose_jacobian: Matrix2x6,
    pub target_pose_jacobian: Matrix2x6,
    /// Columns are `[stereographic_u, stereographic_v, inverse_distance]`.
    pub landmark_jacobian: Matrix2x3<f64>,
    /// The robust objective contribution in Basalt's `computeError`.
    ///
    /// This is intentionally kept separate from `residual.norm_squared()`:
    /// the latter is the quadratic row-stack used by ABS_QR, while Basalt
    /// evaluates Huber rows with `0.5 * (2 - w) * w * ||r||²`.
    pub objective_cost: f64,
    /// Optional current-value f32 chain snapshot for a production-vs-oracle
    /// visual boundary audit.  This is `None` unless the opt-in environment
    /// variable is present.
    pub(crate) debug_chain: Option<VisualChainF32>,
}

/// Port of `ba_utils.h::linearizePoint` plus the absolute host/target pose
/// chain used by `LandmarkBlockAbsDynamic` at the pinned upstream revision.
///
/// Poses are `T_w_i`, extrinsics are `T_i_c`, and Basalt's pose increment is
/// `[world_translation; left_rotation]`. The residual sign is
/// `projection - observation`, exactly as upstream.
#[allow(clippy::too_many_arguments)]
pub fn anchored_visual_reprojection_factor(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    same_time_camera: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    anchored_visual_reprojection_factor_with_time_cam(
        camera,
        anchor_imu_pose,
        anchor_t_imu_cam,
        target_imu_pose,
        target_t_imu_cam,
        landmark,
        observation,
        same_time_camera,
        same_time_camera,
        config,
    )
}

/// Absolute visual factor with the two upstream identity predicates kept
/// separate.  Basalt's absolute landmark block receives a `TimeCamId` for the
/// host and target.  Only equal frame *and* camera takes the exact identity
/// branch; a same-timestamp stereo pair still calls `computeRelPose` and
/// therefore carries both relative pose Jacobian blocks.  The window
/// scatterer later combines those two blocks when both TimeCamIds share one
/// navigation-state column block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn anchored_visual_reprojection_factor_with_time_cam(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    _same_timestamp: bool,
    same_time_cam_id: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    if !config.observation_stddev.is_finite() || config.observation_stddev <= 0.0 {
        return None;
    }

    let target_camera_from_imu = target_t_imu_cam.inverse();
    let target_imu_from_anchor_imu = target_imu_pose.inverse().compose(anchor_imu_pose);
    let target_camera_from_anchor_imu = target_camera_from_imu.compose(&target_imu_from_anchor_imu);
    let target_camera_from_anchor_camera = if same_time_cam_id {
        // `TimeCamId == TimeCamId` is an explicit source branch in
        // linearization_abs_qr.cpp.  Do not compose the four constituent
        // SE(3)s here: even mathematically cancelling f64 operations can
        // leave a tiny rotation/translation that rounds into the landmark
        // Jacobian.  The identity is also independent of the current state
        // and calibration values, as upstream's branch is.
        SE3::identity()
    } else {
        target_camera_from_anchor_imu.compose(anchor_t_imu_cam)
    };

    let bearing = landmark.direction.bearing();
    let point_target = target_camera_from_anchor_camera
        .rotation
        .transform_vector(&bearing)
        + target_camera_from_anchor_camera.translation * landmark.inverse_distance;
    let (predicted, projection_jacobian) =
        project_double_sphere_with_jacobian(camera, point_target)?;
    let raw = predicted - observation.coords;

    // LandmarkBlock applies the Huber norm in raw pixel units, then divides
    // both residual and Jacobians by obs_std_dev.
    let residual_squared = raw.norm_squared();
    let huber_weight =
        if config.huber_delta > 0.0 && residual_squared > config.huber_delta * config.huber_delta {
            config.huber_delta / residual_squared.sqrt()
        } else {
            1.0
        };
    let sqrt_weight = huber_weight.sqrt() / config.observation_stddev;
    let objective_cost =
        robust_objective_from_raw(residual_squared, huber_weight, config.observation_stddev);

    let mut point_wrt_relative_pose = Matrix3x6::zeros();
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(Matrix3::identity() * landmark.inverse_distance));
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew3(point_target)));
    let residual_wrt_relative_pose = projection_jacobian * point_wrt_relative_pose;

    let (relative_wrt_anchor, relative_wrt_target) = if same_time_cam_id {
        // The complete TimeCamId equality branch bypasses computeRelPose and
        // leaves both output Jacobians zero. Keep this explicit and
        // independent of numerical SE(3) cancellation.
        (Matrix6::zeros(), Matrix6::zeros())
    } else {
        let mut anchor_rotation_blocks = Matrix6::zeros();
        let r_w_i_anchor_inv = anchor_imu_pose
            .rotation
            .inverse()
            .to_rotation_matrix()
            .into_inner();
        anchor_rotation_blocks
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&r_w_i_anchor_inv);
        anchor_rotation_blocks
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&r_w_i_anchor_inv);

        let mut target_rotation_blocks = Matrix6::zeros();
        let r_w_i_target_inv = target_imu_pose
            .rotation
            .inverse()
            .to_rotation_matrix()
            .into_inner();
        target_rotation_blocks
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&r_w_i_target_inv);
        target_rotation_blocks
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&r_w_i_target_inv);

        (
            target_camera_from_anchor_imu.adjoint() * anchor_rotation_blocks,
            -(target_camera_from_imu.adjoint() * target_rotation_blocks),
        )
    };

    let direction_jacobian: Matrix3x2<f64> = landmark.direction.bearing_jacobian();
    let mut point_wrt_landmark = Matrix3::zeros();
    point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
        &(target_camera_from_anchor_camera
            .rotation
            .to_rotation_matrix()
            .into_inner()
            * direction_jacobian),
    );
    point_wrt_landmark.set_column(2, &target_camera_from_anchor_camera.translation);

    Some(AnchoredVisualFactor {
        raw_residual: raw,
        projection: predicted,
        huber_weight,
        sqrt_weight,
        residual: raw * sqrt_weight,
        anchor_pose_jacobian: residual_wrt_relative_pose * relative_wrt_anchor * sqrt_weight,
        target_pose_jacobian: residual_wrt_relative_pose * relative_wrt_target * sqrt_weight,
        landmark_jacobian: projection_jacobian * point_wrt_landmark * sqrt_weight,
        objective_cost,
        debug_chain: None,
    })
}

type Matrix3x6 = SMatrix<f64, 3, 6>;

fn skew3(value: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -value.z, value.y, value.z, 0.0, -value.x, -value.y, value.x, 0.0,
    )
}

fn project_double_sphere_with_jacobian(
    camera: &DoubleSphereCamera,
    point: Vector3<f64>,
) -> Option<(Vector2<f64>, Matrix2x3<f64>)> {
    if !point.iter().all(|value| value.is_finite()) {
        return None;
    }
    let d1 = point.norm();
    if d1 <= 1e-12 {
        return None;
    }
    let zeta = camera.xi * d1 + point.z;
    let d2 = (point.x * point.x + point.y * point.y + zeta * zeta).sqrt();
    if d2 <= 1e-12 {
        return None;
    }
    let denominator = camera.alpha * d2 + (1.0 - camera.alpha) * zeta;
    if !denominator.is_finite() || denominator <= 1e-12 {
        return None;
    }

    let d_zeta = Vector3::new(
        camera.xi * point.x / d1,
        camera.xi * point.y / d1,
        camera.xi * point.z / d1 + 1.0,
    );
    let d_d2 = (Vector3::new(point.x, point.y, 0.0) + d_zeta * zeta) / d2;
    let d_denominator = d_d2 * camera.alpha + d_zeta * (1.0 - camera.alpha);
    let denominator_sq = denominator * denominator;
    let mut jacobian = Matrix2x3::zeros();
    for column in 0..3 {
        jacobian[(0, column)] = camera.fx
            * ((if column == 0 { denominator } else { 0.0 }) - point.x * d_denominator[column])
            / denominator_sq;
        jacobian[(1, column)] = camera.fy
            * ((if column == 1 { denominator } else { 0.0 }) - point.y * d_denominator[column])
            / denominator_sq;
    }
    Some((
        Vector2::new(
            camera.fx * point.x / denominator + camera.cx,
            camera.fy * point.y / denominator + camera.cy,
        ),
        jacobian,
    ))
}

/// A minimal float-owned pose used by the upstream compatibility path.  The
/// public `SE3` type is intentionally f64 for API stability; this local pose
/// keeps composition, inverse, adjoint, and point transforms in f32 until the
/// factor crosses back into the f64 row-stack representation.
#[derive(Clone, Copy)]
pub(super) struct F32Pose {
    pub(super) rotation: UnitQuaternion<f32>,
    pub(super) translation: Vector3<f32>,
}

impl F32Pose {
    pub(super) fn from_se3(pose: &SE3) -> Self {
        Self {
            rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                pose.rotation.w as f32,
                pose.rotation.i as f32,
                pose.rotation.j as f32,
                pose.rotation.k as f32,
            )),
            translation: pose.translation.map(|value| value as f32),
        }
    }

    pub(super) fn compose(self, other: Self) -> Self {
        Self {
            // Sophus::SO3 multiplication constructs a fresh SO3 and
            // normalizes its quaternion. nalgebra's UnitQuaternion product
            // is intentionally unchecked, which otherwise leaks norm drift
            // into every f32 visual factor.
            rotation: sophus_quat_product_f32(self.rotation, other.rotation),
            // Sophus::SO3::operator*(Vector3) uses the explicit
            // `uv = q.vec().cross(p); p + q.w() * (2*uv) + q.vec().cross(2*uv)`
            // action. Eigen's Quaternion::transformVector is algebraically
            // equivalent but has a different f32 association.
            translation: sophus_rotate_f32(self.rotation, other.translation) + self.translation,
        }
    }

    pub(super) fn inverse(self) -> Self {
        // Sophus::SO3::inverse() constructs a fresh SO3 from the conjugate;
        // that constructor normalizes the f32 quaternion.  A nalgebra
        // UnitQuaternion inverse only conjugates its unchecked coefficients,
        // which leaks the norm error introduced when an f64 state is cast to
        // f32 into the subsequent camera chain.
        let rotation = sophus_so3_inverse(self.rotation);
        Self {
            rotation,
            translation: -sophus_rotate_f32(rotation, self.translation),
        }
    }
}

#[inline]
pub(super) fn visual_chain_pose_snapshot(pose: F32Pose) -> [f32; 7] {
    let q = pose.rotation.quaternion();
    [
        q.i,
        q.j,
        q.k,
        q.w,
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
    ]
}

pub(super) fn sophus_relative_imu_f32(target: F32Pose, host: F32Pose) -> F32Pose {
    let target_inverse_rotation = sophus_so3_inverse(target.rotation);
    F32Pose {
        rotation: sophus_quat_product_f32(target_inverse_rotation, host.rotation),
        translation: sophus_rotate_difference_f32(
            target_inverse_rotation,
            host.translation,
            target.translation,
        ),
    }
}

pub(super) fn sophus_quat_product_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    sophus_so3_product(first, second)
}

/// Packet quaternion product used by the inlined current-pose branch of
/// `computeRelPose` (libbasalt 0x305543..0x3055b8).
///
/// The current relative rotation is formed as `target.inverse() * host`.
/// Eigen does not evaluate that product with the scalar Sophus expression in
/// [`sophus_quat_product_f32`]: it keeps the two signed packet partials in
/// separate registers, blends the lane-3 accumulator after each partial, and
/// only then applies the final cross-term packet.  Keeping this call-site
/// schedule explicit is important because the final normalized quaternion is
/// consumed by the f32 visual chain.
#[inline(never)]
pub(super) fn sophus_quat_product_current_packet_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let a = first.quaternion();
    let b = second.quaternion();
    let ax = a.i;
    let ay = a.j;
    let az = a.k;
    let aw = a.w;
    let bx = b.i;
    let by = b.j;
    let bz = b.k;
    let bw = b.w;

    // 0x305585..0x30559e: a.vec()*b.w, then the signed b.vec()*a.vec
    // partials.  The arrays are the exact Packet4f lane permutations from
    // the native `vpermilps` instructions.
    let a24 = [ax, ay, az, ax];
    let b3f = [bw, bw, bw, bx];
    let mut product = [
        a24[0] * b3f[0],
        a24[1] * b3f[1],
        a24[2] * b3f[2],
        a24[3] * b3f[3],
    ];
    let a_ff = [aw; 4];
    let b_vec = [bx, by, bz, bw];
    let mut negative = [0.0_f32; 4];
    for lane in 0..4 {
        negative[lane] = b_vec[lane].mul_add(a_ff[lane], -product[lane]);
        product[lane] = b_vec[lane].mul_add(a_ff[lane], product[lane]);
    }
    product[3] = negative[3];

    // 0x3055a8..0x3055b2: second signed packet partial and lane-3 blend.
    let b52 = [bz, bx, by, by];
    let a49 = [ay, az, ax, ay];
    let mut second_negative = [0.0_f32; 4];
    let mut second_positive = [0.0_f32; 4];
    for lane in 0..4 {
        second_negative[lane] = (-b52[lane]).mul_add(a49[lane], product[lane]);
        second_positive[lane] = a49[lane].mul_add(b52[lane], product[lane]);
    }
    product = second_positive;
    product[3] = second_negative[3];

    // 0x3055b8: final signed packet cross-term.
    let b89 = [by, bz, bx, bz];
    let a92 = [az, ax, ay, az];
    for lane in 0..4 {
        product[lane] = (-a92[lane]).mul_add(b89[lane], product[lane]);
    }

    // 0x3055bd..0x3055e4: vmulps followed by the Packet4f pair reduction,
    // scalar sqrt, broadcast, and vdivps.  Spell the reduction as the two
    // packet pairs so it cannot be reassociated with the product lanes.
    let squares = [
        product[0] * product[0],
        product[1] * product[1],
        product[2] * product[2],
        product[3] * product[3],
    ];
    let pair0 = squares[0] + squares[2];
    let pair1 = squares[1] + squares[3];
    let norm = (pair0 + pair1).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        product[3] / norm,
        product[0] / norm,
        product[1] / norm,
        product[2] / norm,
    ))
}

/// Packet inverse/normalization used by the inlined current visual branch
/// before its relative quaternion product.  Keep this separate from the
/// shared Sophus inverse so the current-chain fixture can audit the complete
/// native boundary (the pinned pass-3 input is sensitive to the pairwise
/// reduction's final ulp).
#[inline(never)]
pub(super) fn sophus_quat_inverse_current_packet_f32(
    rotation: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let q = rotation.quaternion();
    let product = [-q.i, -q.j, -q.k, q.w];
    let squares = [
        product[0] * product[0],
        product[1] * product[1],
        product[2] * product[2],
        product[3] * product[3],
    ];
    let pair0 = squares[0] + squares[2];
    let pair1 = squares[1] + squares[3];
    let norm = (pair0 + pair1).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        product[3] / norm,
        product[0] / norm,
        product[1] / norm,
        product[2] / norm,
    ))
}

/// Packet quaternion product emitted by the out-of-line `computeRelPose<float>`
/// body for `tmp = tmp2 * T_t_i_h_i` (2bcbc1..2bcc25).  This is kept separate
/// from the inlined visual-chain product because the native body materializes
/// `b * a.w`, then performs the two signed packet partials and their lane-3
/// blends before the final signed partial.  `mul_add` is the scalar spelling
/// of each packet FMA lane.
#[inline(never)]
fn sophus_quat_product_relpose_out_of_line_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let a = first.quaternion();
    let b = second.quaternion();
    let a24 = [a.i, a.j, a.k, a.i];
    let a49 = [a.j, a.k, a.i, a.j];
    let a92 = [a.k, a.i, a.j, a.k];
    let b3f = [b.w, b.w, b.w, b.i];
    let b52 = [b.k, b.i, b.j, b.j];
    let b89 = [b.j, b.k, b.i, b.k];

    // 2bcb c1: vmulps(relative_q, broadcast(tmp2.w), relative_q).
    let b_times_aw = [b.i * a.w, b.j * a.w, b.k * a.w, b.w * a.w];

    // 2bcbdf..2bcbfb: signed first partial and lane-3 blend.
    let mut first_negative = [0.0_f32; 4];
    let mut first_positive = [0.0_f32; 4];
    for lane in 0..4 {
        first_negative[lane] = (-b3f[lane]).mul_add(a24[lane], b_times_aw[lane]);
        first_positive[lane] = b3f[lane].mul_add(a24[lane], b_times_aw[lane]);
    }
    first_positive[3] = first_negative[3];

    // 2bcc01..2bcc18: second signed partial and lane-3 blend.
    let mut second_negative = [0.0_f32; 4];
    let mut second_positive = [0.0_f32; 4];
    for lane in 0..4 {
        second_negative[lane] = (-b52[lane]).mul_add(a49[lane], first_positive[lane]);
        second_positive[lane] = b52[lane].mul_add(a49[lane], first_positive[lane]);
    }
    second_positive[3] = second_negative[3];

    // 2bcc1e: final signed partial using the saved 0x89 permutation.
    let mut product = [0.0_f32; 4];
    for lane in 0..4 {
        product[lane] = (-a92[lane]).mul_add(b89[lane], second_positive[lane]);
    }

    let x2 = product[0] * product[0];
    let z2 = product[2] * product[2];
    let y2 = product[1] * product[1];
    let w2 = product[3] * product[3];
    let norm = (x2 + z2 + (y2 + w2)).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        product[3] / norm,
        product[0] / norm,
        product[1] / norm,
        product[2] / norm,
    ))
}

/// Scalar lanes interleaved with the out-of-line quaternion packet in
/// `computeRelPose<float>` (2bcb10..2bcc67).  This is Sophus's explicit
/// `q * p` action, but with the exact temporary reuse and FMA association seen
/// in the clean disassembly.  In particular, the first `vunpckhps` leaves
/// `q.z` in the scalar lane used for the initial cross product.
#[inline(never)]
pub(super) fn sophus_rotate_relpose_out_of_line_f32(
    rotation: UnitQuaternion<f32>,
    point: Vector3<f32>,
) -> Vector3<f32> {
    let q = rotation.quaternion();
    let vx = point.x;
    let vy = point.y;
    let vz = point.z;

    let qz = q.k;
    let mut x5 = qz * vy;
    let mut x4 = vz * q.i;
    let mut x3 = vx * q.j;

    // First cross product, doubled (2bcb34..2bcb77).
    x5 = q.j.mul_add(vz, -x5);
    x4 = qz.mul_add(vx, -x4);
    x3 = vy.mul_add(q.i, -x3);
    x5 += x5;
    x4 += x4;
    x3 += x3;

    // q.vec().cross(2 * uv), using the same scalar temporaries and FMA order.
    let mut x11 = x5 * q.j;
    x11 = x4.mul_add(q.i, -x11);
    let mut x2 = x3 * q.i;
    x2 = qz.mul_add(x5, -x2);
    let qz_x4 = qz * x4;
    let x1 = x3.mul_add(q.j, -qz_x4);

    // Add p + q.w() * (2 * uv), then contract the cross terms.
    x4 = x4.mul_add(q.w, vy);
    x5 = x5.mul_add(q.w, vx);
    x3 = x3.mul_add(q.w, vz);
    let y = x2 + x4;
    let x = x1 + x5;
    let z = x3 + x11;
    Vector3::new(x, y, z)
}

/// Packet schedule used by the step-only native `SO3::operator*(Matrix3x1)`
/// body (libbasalt offset 0x2f0050).  Eigen forms three redundant cross
/// products in packed lanes; the duplicate lanes intentionally retain their
/// own FMA accumulator, so algebraically identical terms can differ by one
/// ulp.  This helper keeps those six scalar lanes separate before the final
/// packet-style cross/add stage.
#[inline(never)]
pub(crate) fn sophus_rotate_step_packet_f32(
    rotation: UnitQuaternion<f32>,
    point: Vector3<f32>,
) -> Vector3<f32> {
    let q = rotation.quaternion();
    let px = point.x;
    let py = point.y;
    let pz = point.z;

    // 0x2f0094..0x2f00b9: three packed partial products.  The duplicated
    // lanes below correspond to different source packets and therefore use
    // different FMA accumulators in the native body.
    let uv_x_a = (-q.k).mul_add(py, q.j * pz);
    let uv_y_a = (-q.i).mul_add(pz, q.k * px);
    let uv_y_b = q.k.mul_add(px, -(q.i * pz));
    let uv_z_b = q.i.mul_add(py, -(q.j * px));
    let uv_z_c = q.i.mul_add(py, -(px * q.j));
    let uv_x_c = q.j.mul_add(pz, -(py * q.k));

    let uv2_x_a = uv_x_a + uv_x_a;
    let uv2_y_a = uv_y_a + uv_y_a;
    let uv2_y_b = uv_y_b + uv_y_b;
    let uv2_z_b = uv_z_b + uv_z_b;
    let uv2_z_c = uv_z_c + uv_z_c;
    let uv2_x_c = uv_x_c + uv_x_c;

    // 0x2f00ca..0x2f00e8: packed x/y cross and scalar z cross.
    let cross_x = q.j.mul_add(uv2_z_c, -(q.k * uv2_y_b));
    let cross_y = q.k.mul_add(uv2_x_c, -(q.i * uv2_z_b));
    let cross_z = q.i.mul_add(uv2_y_b, -(q.j * uv2_x_c));

    // 0x2f00fa..0x2f0108: packet q.w action plus the second cross.
    Vector3::new(
        q.w.mul_add(uv2_x_a, px) + cross_x,
        q.w.mul_add(uv2_y_a, py) + cross_y,
        q.w.mul_add(uv2_z_c, pz) + cross_z,
    )
}

/// Source-faithful FEJ boundary for the out-of-line `computeRelPose<float>`:
/// `tmp2 * (T_w_i_t.inverse() * T_w_i_h)`.  The production FEJ path calls
/// this with the frozen target/anchor IMU poses so no current-pose value-chain
/// rounding can enter the relative Jacobian boundary.
#[inline(never)]
pub(super) fn sophus_compute_relpose_tmp_out_of_line_f32(
    tmp2: F32Pose,
    target_imu: F32Pose,
    host_imu: F32Pose,
) -> F32Pose {
    let target_inverse_rotation = sophus_so3_inverse(target_imu.rotation);
    let relative_rotation = sophus_quat_product_f32(target_inverse_rotation, host_imu.rotation);
    let relative_translation = sophus_rotate_difference_f32(
        target_inverse_rotation,
        host_imu.translation,
        target_imu.translation,
    );
    F32Pose {
        rotation: sophus_quat_product_relpose_out_of_line_f32(tmp2.rotation, relative_rotation),
        translation: sophus_rotate_relpose_out_of_line_f32(tmp2.rotation, relative_translation)
            + tmp2.translation,
    }
}

/// Packet product used by the inlined `computeRelPose` camera prefix.  Eigen's
/// fixed `Packet4f` product keeps the first scalar product as the accumulator,
/// performs each following term as an FMA, and blends the w lane after the
/// two signed/unsigned partial products.  This is intentionally separate from
/// the generic Sophus product: the native compiler emits a different packet
/// schedule at this call site.
pub(super) fn sophus_quat_product_camera_prefix_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let a = first.quaternion();
    let b = second.quaternion();
    let ax = a.i;
    let ay = a.j;
    let az = a.k;
    let aw = a.w;
    let bx = b.i;
    let by = b.j;
    let bz = b.k;
    let bw = b.w;
    let c24 = [ax, ay, az, ax];
    let p3f = [bw, bw, bw, bx];
    let mut x5 = [0.0_f32; 4];
    let mut x8 = [aw * bx, aw * by, aw * bz, aw * bw];
    for lane in 0..4 {
        x5[lane] = (-p3f[lane]).mul_add(c24[lane], x8[lane]);
        x8[lane] = p3f[lane].mul_add(c24[lane], x8[lane]);
    }
    x8[3] = x5[3];
    let p52 = [bz, bx, by, by];
    let c49 = [ay, az, ax, ay];
    let mut x2 = [0.0_f32; 4];
    for lane in 0..4 {
        x2[lane] = (-p52[lane]).mul_add(c49[lane], x8[lane]);
        x8[lane] = p52[lane].mul_add(c49[lane], x8[lane]);
    }
    x8[3] = x2[3];
    let p89 = [by, bz, bx, bz];
    let c92 = [az, ax, ay, az];
    for lane in 0..4 {
        x8[lane] = (-p89[lane]).mul_add(c92[lane], x8[lane]);
    }
    let pair0 = x8[0] * x8[0] + x8[2] * x8[2];
    let pair1 = x8[1] * x8[1] + x8[3] * x8[3];
    let norm = (pair0 + pair1).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        x8[3] / norm,
        x8[0] / norm,
        x8[1] / norm,
        x8[2] / norm,
    ))
}

/// Packet product used by the inlined host-extrinsic suffix of
/// `computeRelPose`.  This is the corresponding Eigen register schedule for
/// `q_prefix * q_host`; it has a distinct first partial product from the
/// camera prefix above.
pub(super) fn sophus_quat_product_camera_suffix_f32(
    first: UnitQuaternion<f32>,
    second: UnitQuaternion<f32>,
) -> UnitQuaternion<f32> {
    let a = first.quaternion();
    let b = second.quaternion();
    let ax = a.i;
    let ay = a.j;
    let az = a.k;
    let aw = a.w;
    let bx = b.i;
    let by = b.j;
    let bz = b.k;
    let bw = b.w;

    let a24 = [ax, ay, az, ax];
    let b3f = [bw, bw, bw, bx];
    let mut x1 = [
        a24[0] * b3f[0],
        a24[1] * b3f[1],
        a24[2] * b3f[2],
        a24[3] * b3f[3],
    ];
    let a_ff = [aw; 4];
    let b_vec = [bx, by, bz, bw];
    let mut x11 = [0.0_f32; 4];
    let mut x8 = [0.0_f32; 4];
    for lane in 0..4 {
        x11[lane] = b_vec[lane].mul_add(a_ff[lane], -x1[lane]);
        x8[lane] = b_vec[lane].mul_add(a_ff[lane], x1[lane]);
    }
    x8[3] = x11[3];

    let b52 = [bz, bx, by, by];
    let a49 = [ay, az, ax, ay];
    for lane in 0..4 {
        x1[lane] = (-b52[lane]).mul_add(a49[lane], x8[lane]);
        x8[lane] = a49[lane].mul_add(b52[lane], x8[lane]);
    }
    x8[3] = x1[3];

    let b89 = [by, bz, bx, bz];
    let a92 = [az, ax, ay, az];
    for lane in 0..4 {
        x8[lane] = (-b89[lane]).mul_add(a92[lane], x8[lane]);
    }
    let pair0 = x8[0] * x8[0] + x8[2] * x8[2];
    let pair1 = x8[1] * x8[1] + x8[3] * x8[3];
    let norm = (pair0 + pair1).sqrt();
    UnitQuaternion::new_unchecked(Quaternion::new(
        x8[3] / norm,
        x8[0] / norm,
        x8[1] / norm,
        x8[2] / norm,
    ))
}

pub(super) fn skew3_f32(value: Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(
        0.0, -value.z, value.y, value.z, 0.0, -value.x, -value.y, value.x, 0.0,
    )
}

/// Fixed-size `3x3 * 3x3` reduction used by Eigen for Sophus adjoint blocks.
/// Keep the first product as the accumulator and contract the next two terms
/// in source order; nalgebra's generic product can choose a different f32
/// association for these small blocks.
pub(super) fn eigen_matrix_product_3x3_f32(
    left: Matrix3<f32>,
    right: Matrix3<f32>,
) -> Matrix3<f32> {
    let mut result = Matrix3::<f32>::zeros();
    for row in 0..3 {
        for column in 0..3 {
            let value = left[(row, 1)] * right[(1, column)];
            let value = left[(row, 2)].mul_add(right[(2, column)], value);
            result[(row, column)] = left[(row, 0)].mul_add(right[(0, column)], value);
        }
    }
    result
}

/// Eigen's fixed-size six-term packet reduction.  The native `6x6` product
/// processes four output lanes together and contracts the six terms in
/// increasing k order; even when a block-diagonal right operand makes terms
/// zero, it does not collapse to a smaller `3x3` product.
pub(super) fn eigen_matrix_product_6x6_f32(
    left: SMatrix<f32, 6, 6>,
    right: SMatrix<f32, 6, 6>,
) -> SMatrix<f32, 6, 6> {
    let mut result = SMatrix::<f32, 6, 6>::zeros();
    for row in 0..6 {
        for column in 0..6 {
            let value = left[(row, 0)] * right[(0, column)];
            let value = left[(row, 1)].mul_add(right[(1, column)], value);
            let value = left[(row, 2)].mul_add(right[(2, column)], value);
            let value = left[(row, 3)].mul_add(right[(3, column)], value);
            let value = left[(row, 4)].mul_add(right[(4, column)], value);
            result[(row, column)] = left[(row, 5)].mul_add(right[(5, column)], value);
        }
    }
    result
}

/// Evaluate the fixed-size Eigen product used to chain a visual residual
/// Jacobian (`2x6`) into an absolute-pose Jacobian (`6x6`).
///
/// The pinned x86 Eigen assignment kernel does not reduce the six terms in
/// increasing-k order.  Its two-row packet/tail schedule forms the two
/// three-term groups `(k4 + k5) + k3` and `(k1 + k2) + k0`, then adds those
/// groups.  Although the two trees are algebraically equivalent, their f32
/// rounding (and signed-zero propagation) is observable in the ABS_QR rows.
/// Keep this helper separate from the ordinary nalgebra product so the source
/// boundary remains explicit and testable.
pub(super) fn eigen_matrix_product_2x6_f32(
    left: SMatrix<f32, 2, 6>,
    right: SMatrix<f32, 6, 6>,
) -> SMatrix<f32, 2, 6> {
    let mut result = SMatrix::<f32, 2, 6>::zeros();
    for row in 0..2 {
        for column in 0..6 {
            // This is the exact scalar spelling of Eigen's `2x6 * 6x6`
            // packet assignment observed in the pinned clean libbasalt:
            // (k4 + k5) + k3, then (k1 + k2) + k0, then the pair add.
            let high = left[(row, 4)] * right[(4, column)];
            let high = left[(row, 5)].mul_add(right[(5, column)], high);
            let high = left[(row, 3)].mul_add(right[(3, column)], high);
            let low = left[(row, 1)] * right[(1, column)];
            let low = left[(row, 2)].mul_add(right[(2, column)], low);
            let low = left[(row, 0)].mul_add(right[(0, column)], low);
            result[(row, column)] = high + low;
        }
    }
    result
}

/// Source-faithful absolute-pose chain for one whitened visual row.
///
/// `LandmarkBlockAbsDynamic` first scales its relative residual Jacobian in
/// place and only then evaluates the fixed `2x6 * 6x6` product.  Keeping the
/// scalar multiplication before [`eigen_matrix_product_2x6_f32`] matters for
/// the final binary32 lanes; multiplying the completed product instead is a
/// different rounding path.
pub(super) fn eigen_weighted_pose_jacobian_f32(
    relative_pose_jacobian: SMatrix<f32, 2, 6>,
    absolute_pose_jacobian: SMatrix<f32, 6, 6>,
    sqrt_weight: f32,
) -> SMatrix<f32, 2, 6> {
    let mut weighted_relative = relative_pose_jacobian;
    for value in weighted_relative.as_mut_slice() {
        *value *= sqrt_weight;
    }
    eigen_matrix_product_2x6_f32(weighted_relative, absolute_pose_jacobian)
}

/// Compute a signed Sophus `Adj() * diag(R, R)` while retaining the source
/// fixed block/product boundaries used by `computeRelPose` for both absolute
/// pose derivatives.
///
/// The sign belongs to the left operand in the upstream expression.  In
/// particular, `-tmp2.Adj() * RR_t` is not equivalent at the binary32 bit
/// level to negating the completed product: the latter changes the signs of
/// zeros produced by the six-term Eigen reduction.  Keeping the sign here
/// makes that ordering explicit without introducing a factor- or
/// observation-specific path.
pub(super) fn eigen_adjoint_times_rotation_blocks_f32(
    pose: F32Pose,
    rotation: Matrix3<f32>,
    left_sign: f32,
) -> SMatrix<f32, 6, 6> {
    let pose_rotation = eigen_quaternion_matrix_f32(pose.rotation);
    let cross_rotation = eigen_matrix_product_3x3_f32(skew3_f32(pose.translation), pose_rotation);
    let mut adjoint = SMatrix::<f32, 6, 6>::zeros();
    adjoint
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&pose_rotation);
    adjoint
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&cross_rotation);
    adjoint
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&pose_rotation);
    let mut rotation_blocks = SMatrix::<f32, 6, 6>::zeros();
    rotation_blocks
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation);
    rotation_blocks
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&rotation);
    eigen_matrix_product_6x6_f32(adjoint * left_sign, rotation_blocks)
}

pub(super) fn sophus_rotate_f32(
    rotation: UnitQuaternion<f32>,
    point: Vector3<f32>,
) -> Vector3<f32> {
    let q = rotation.quaternion();
    // Sophus `SO3Base::operator*(Point)` is `p + q.w() * uv +
    // q.vec().cross(uv)` after `uv = q.vec().cross(p); uv += uv`.
    // Spell the pinned native cross/FMA lane order explicitly.
    let uv_x = q.j.mul_add(point.z, -(q.k * point.y));
    let uv_y = q.k.mul_add(point.x, -(q.i * point.z));
    let uv_z = q.i.mul_add(point.y, -(q.j * point.x));
    let uv2_x = uv_x + uv_x;
    let uv2_y = uv_y + uv_y;
    let uv2_z = uv_z + uv_z;
    let cross_x = q.j.mul_add(uv2_z, -(q.k * uv2_y));
    let cross_y = q.k.mul_add(uv2_x, -(q.i * uv2_z));
    let cross_z = q.i.mul_add(uv2_y, -(q.j * uv2_x));
    Vector3::new(
        q.w.mul_add(uv2_x, point.x) + cross_x,
        q.w.mul_add(uv2_y, point.y) + cross_y,
        q.w.mul_add(uv2_z, point.z) + cross_z,
    )
}

/// Eigen's packetized `SO3::operator*(host_translation - target_translation)`
/// path used by `computeRelPose`.  This differs from the scalar `Vector3`
/// overload above at the intermediate cross-product boundary: Eigen keeps
/// three independently packed products, then contracts each signed pair with
/// a distinct FMA.  Keep the difference fused into this schedule instead of
/// materializing a `Vector3` before calling the generic action.
pub(super) fn sophus_rotate_difference_f32(
    rotation: UnitQuaternion<f32>,
    first: Vector3<f32>,
    second: Vector3<f32>,
) -> Vector3<f32> {
    let q = rotation.quaternion();
    let dx = first.x - second.x;
    let dy = first.y - second.y;
    let dz = first.z - second.z;

    // The packet layout is [x,y], [y,z], and [z,x].  Keep each independently
    // formed lane: Eigen's three products are not common-subexpression
    // eliminated, and their duplicate cross terms can round differently.
    let uv_a_x = (-q.k).mul_add(dy, q.j * dz);
    let uv_a_y = (-q.i).mul_add(dz, q.k * dx);
    let uv_b_y = q.k.mul_add(dx, -(q.i * dz));
    let uv_b_z = q.i.mul_add(dy, -(q.j * dx));
    let uv_c_z = q.i.mul_add(dy, -(q.j * dx));
    let uv_c_x = q.j.mul_add(dz, -(q.k * dy));

    let uv2_a_x = uv_a_x + uv_a_x;
    let uv2_a_y = uv_a_y + uv_a_y;
    let uv2_b_y = uv_b_y + uv_b_y;
    let uv2_b_z = uv_b_z + uv_b_z;
    let uv2_c_z = uv_c_z + uv_c_z;
    let uv2_c_x = uv_c_x + uv_c_x;

    // Second cross packet: [cross.x, cross.y].
    let cross_x = q.j.mul_add(uv2_c_z, -(q.k * uv2_b_y));
    let cross_y = q.k.mul_add(uv2_c_x, -(q.i * uv2_b_z));
    let cross_z = q.i.mul_add(uv2_b_y, -(q.j * uv2_c_x));

    Vector3::new(
        q.w.mul_add(uv2_a_x, dx) + cross_x,
        q.w.mul_add(uv2_a_y, dy) + cross_y,
        q.w.mul_add(uv2_c_z, dz) + cross_z,
    )
}

/// Current-pose visual relinearization uses a distinct inline Eigen path from
/// the FEJ/relative-Jacobian `computeRelPose` call.  The inline path first
/// materializes `T_w_i_h.translation() - T_w_i_t.translation()` and then
/// applies the quaternion action.  Keeping that difference as a concrete
/// vector is observable at the f32 boundary (frame 4's prefix `t.y` is two
/// ulps below the packetized FEJ helper above), so do not reuse
/// [`sophus_rotate_difference_f32`] here.
pub(super) fn sophus_rotate_difference_visual_f32(
    rotation: UnitQuaternion<f32>,
    first: Vector3<f32>,
    second: Vector3<f32>,
) -> Vector3<f32> {
    let difference = first - second;
    sophus_rotate_f32(rotation, difference)
}

/// Eigen::QuaternionBase::toRotationMatrix operation order.
///
/// nalgebra's `UnitQuaternion::to_rotation_matrix` is algebraically the same
/// conversion, but its expression tree is not the one used by the pinned
/// Eigen implementation at the f32 boundary.  In particular, the native
/// fixed-size packet path contracts the two x-diagonal products before the
/// subtraction from one.  Keep that operation boundary explicit for every
/// upstream quaternion-to-matrix call, including the d_rel rotation blocks.
pub(super) fn eigen_quaternion_matrix_f32(rotation: UnitQuaternion<f32>) -> Matrix3<f32> {
    let q = rotation.quaternion();
    let tx = 2.0_f32 * q.i;
    let ty = 2.0_f32 * q.j;
    let tz = 2.0_f32 * q.k;
    let txy = ty * q.i;
    let txz = tz * q.i;
    let tyy = ty * q.j;
    let tyz = tz * q.j;
    let tzz = tz * q.k;
    Matrix3::new(
        1.0_f32 - (tyy + tzz),
        (-tz).mul_add(q.w, txy),
        ty.mul_add(q.w, txz),
        tz.mul_add(q.w, txy),
        1.0_f32 - q.i.mul_add(tx, tzz),
        (-tx).mul_add(q.w, tyz),
        (-ty).mul_add(q.w, txz),
        tx.mul_add(q.w, tyz),
        1.0_f32 - q.i.mul_add(tx, tyy),
    )
}

/// Eigen/Sophus `QuaternionBase::toRotationMatrix` scalar schedule used by
/// the native relative-pose temporary.  Eigen materializes the doubled
/// quaternion products, then the compiler contracts each signed cross term
/// as one multiply-add.  Keep the sign on the multiplicand (rather than
/// materializing `tw*` and subtracting afterward): that is the f32 boundary
/// exposed by the pinned `T_t_h` packets.
pub(super) fn eigen_quaternion_matrix_native_f32(rotation: UnitQuaternion<f32>) -> Matrix3<f32> {
    let q = rotation.quaternion();
    let tx = 2.0_f32 * q.i;
    let ty = 2.0_f32 * q.j;
    let tz = 2.0_f32 * q.k;
    let txy = ty * q.i;
    let txz = tz * q.i;
    let tyy = ty * q.j;
    let tyz = tz * q.j;
    let tzz = tz * q.k;
    Matrix3::new(
        1.0_f32 - (tyy + tzz),
        (-tz).mul_add(q.w, txy),
        ty.mul_add(q.w, txz),
        tz.mul_add(q.w, txy),
        1.0_f32 - q.i.mul_add(tx, tzz),
        (-tx).mul_add(q.w, tyz),
        (-ty).mul_add(q.w, txz),
        tx.mul_add(q.w, tyz),
        1.0_f32 - q.i.mul_add(tx, tyy),
    )
}

/// Reproduce the fixed-size `3x4 * 4x2` reduction used for the stereographic
/// part of `linearizePoint`'s homogeneous landmark block.  `source_jup` is
/// deliberately four rows wide: the fourth row is homogeneous and zero, but
/// Eigen still evaluates the four-term product.  Its scalar assignment kernel
/// starts the `(k2,k3)` and `(k0,k1)` pairs separately, contracts the second
/// term of each pair, and then adds the pair sums.  A `3x3 * 3x2` nalgebra
/// product has a different f32 association and can move a single Jpp lane.
pub(super) fn eigen_homogeneous_landmark_direction_f32(
    transform_top_left: SMatrix<f32, 3, 4>,
    source_jup: SMatrix<f32, 4, 2>,
) -> SMatrix<f32, 3, 2> {
    let mut result = SMatrix::<f32, 3, 2>::zeros();
    for row in 0..3 {
        for column in 0..2 {
            let pair23 = transform_top_left[(row, 2)] * source_jup[(2, column)];
            let pair23 = transform_top_left[(row, 3)].mul_add(source_jup[(3, column)], pair23);
            let pair01 = transform_top_left[(row, 0)] * source_jup[(0, column)];
            let pair01 = transform_top_left[(row, 1)].mul_add(source_jup[(1, column)], pair01);
            result[(row, column)] = pair23 + pair01;
        }
    }
    result
}

/// Apply the pinned `ba_utils.h::linearizePoint` homogeneous action:
/// `T_t_h * [bearing; inverse_distance]`.  Keeping the 4x4 product explicit
/// matters because the Eigen matrix path rounds differently from the
/// algebraically equivalent `R * bearing + t * rho` spelling.
#[cfg(test)]
pub(super) fn sophus_homogeneous_point_f32(
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
    bearing: Vector3<f32>,
    inverse_distance: f32,
) -> Vector3<f32> {
    let rotation = eigen_quaternion_matrix_native_f32(rotation);
    let mut transform = SMatrix::<f32, 4, 4>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);
    let mut homogeneous = SMatrix::<f32, 4, 1>::zeros();
    homogeneous.fixed_view_mut::<3, 1>(0, 0).copy_from(&bearing);
    homogeneous[(3, 0)] = inverse_distance;
    (transform * homogeneous).fixed_rows::<3>(0).into_owned()
}

/// `SE3::matrix()` consumes the unit quaternion already held by Sophus.  Keep
/// the stored `UnitQuaternion` unchanged at this boundary: renormalizing its
/// coefficients here would create a second f32 rounding point that is absent
/// from the native matrix conversion.
#[cfg(test)]
pub(super) fn sophus_homogeneous_point_normalized_rotation_f32(
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
    bearing: Vector3<f32>,
    inverse_distance: f32,
) -> Vector3<f32> {
    eigen_homogeneous_point_gemv_f32(rotation, translation, bearing, inverse_distance)
        .fixed_rows::<3>(0)
        .into_owned()
}

/// Evaluate the source-shaped Eigen product used by
/// `linearizePoint` for the target point:
///
/// ```text
/// T_t_h.matrix() * [bearing.x, bearing.y, bearing.z, inverse_distance]
/// ```
///
/// The native code materializes the homogeneous `4x4` transform and invokes
/// Eigen's fixed-size `4x4 * 4x1` product.  Keep that boundary explicit here:
/// Keep the fixed-size product materialized rather than spelling it as
/// `R * bearing + t * rho`.  In particular, the `k3` term is the homogeneous
/// translation-times-rho lane, and the fourth output row is retained as part
/// of the product even though callers generally consume only xyz.
pub(super) fn eigen_homogeneous_point_gemv_f32(
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
    bearing: Vector3<f32>,
    inverse_distance: f32,
) -> SMatrix<f32, 4, 1> {
    let transform = eigen_homogeneous_transform_f32(rotation, translation);
    eigen_homogeneous_point_product_f32(transform, bearing, inverse_distance)
}

pub(super) fn eigen_homogeneous_transform_f32(
    rotation: UnitQuaternion<f32>,
    translation: Vector3<f32>,
) -> SMatrix<f32, 4, 4> {
    let rotation = eigen_quaternion_matrix_native_f32(rotation);
    let mut transform = SMatrix::<f32, 4, 4>::identity();
    transform.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    transform
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&translation);

    transform
}

pub(super) fn eigen_homogeneous_point_product_f32(
    transform: SMatrix<f32, 4, 4>,
    bearing: Vector3<f32>,
    inverse_distance: f32,
) -> SMatrix<f32, 4, 1> {
    // Eigen's column-major Packet4 kernel keeps one accumulator lane per
    // output row. It seeds all four lanes from column zero, then folds the
    // remaining columns with one FMA per lane. This is not the horizontal
    // even/odd reduction used by Eigen's row-dot kernels.
    let x = bearing.x;
    let y = bearing.y;
    let z = bearing.z;
    let rho = inverse_distance;
    let mut accumulator = [
        transform[(0, 0)] * x,
        transform[(1, 0)] * x,
        transform[(2, 0)] * x,
        transform[(3, 0)] * x,
    ];
    for row in 0..4 {
        accumulator[row] = transform[(row, 1)].mul_add(y, accumulator[row]);
        accumulator[row] = transform[(row, 2)].mul_add(z, accumulator[row]);
        accumulator[row] = transform[(row, 3)].mul_add(rho, accumulator[row]);
    }
    SMatrix::<f32, 4, 1>::from_column_slice(&accumulator)
}

pub(super) fn project_double_sphere_with_jacobian_f32(
    camera: &DoubleSphereCamera,
    point: Vector3<f32>,
) -> Option<(Vector2<f32>, SMatrix<f32, 2, 3>)> {
    if !point.iter().all(|value| value.is_finite()) {
        return None;
    }
    let x = point.x;
    let y = point.y;
    let z = point.z;
    let xx = x * x;
    let yy = y * y;
    let r2 = xx + yy;
    // Keep the native DoubleSphere `d1_2` sequence materialized: Eigen's
    // scalar path rounds `z*z` and then adds it to `r2` separately, rather
    // than contracting the sum into one FMA.  This boundary feeds both the
    // projection and every camera-Jacobian derivative below.
    let zz = z * z;
    let d1_2 = r2 + zz;
    let d1 = d1_2.sqrt();
    if d1 <= f32::EPSILON {
        return None;
    }
    let xi = camera.xi as f32;
    let alpha = camera.alpha as f32;
    let fx = camera.fx as f32;
    let fy = camera.fy as f32;
    let cx = camera.cx as f32;
    let cy = camera.cy as f32;
    let w1 = if alpha > 0.5_f32 {
        (1.0_f32 - alpha) / alpha
    } else {
        alpha / (1.0_f32 - alpha)
    };
    let w2_denominator = 2.0_f32 * w1 * xi + xi * xi + 1.0_f32;
    let w2 = (w1 + xi) / w2_denominator.sqrt();
    let valid = z > -w2 * d1;

    let k = xi.mul_add(d1, z);
    let d2 = k.mul_add(k, r2).sqrt();
    let one_minus_alpha = 1.0_f32 - alpha;
    let norm = alpha.mul_add(d2, one_minus_alpha * k);
    if !valid || !norm.is_finite() || norm <= f32::EPSILON {
        return None;
    }
    let mx = x / norm;
    let my = y / norm;
    let norm_sq = norm * norm;
    let xy = x * y;
    let tt2 = xi * z / d1 + 1.0_f32;
    let d_norm_d_r2 =
        (xi * (1.0_f32 - alpha) / d1 + alpha * (xi * k / d1 + 1.0_f32) / d2) / norm_sq;
    let tmp2_numerator = tt2.mul_add(one_minus_alpha, alpha * k * tt2 / d2);
    let tmp2 = tmp2_numerator / norm_sq;
    let mut jacobian = SMatrix::<f32, 2, 3>::zeros();
    jacobian[(0, 0)] = (-xx).mul_add(d_norm_d_r2, 1.0_f32 / norm) * fx;
    jacobian[(1, 0)] = -fy * xy * d_norm_d_r2;
    jacobian[(0, 1)] = -fx * xy * d_norm_d_r2;
    jacobian[(1, 1)] = (-yy).mul_add(d_norm_d_r2, 1.0_f32 / norm) * fy;
    jacobian[(0, 2)] = -fx * x * tmp2;
    jacobian[(1, 2)] = -fy * y * tmp2;
    Some((
        Vector2::new(fx.mul_add(mx, cx), fy.mul_add(my, cy)),
        jacobian,
    ))
}

/// Reproduce Eigen's fixed-size `2x4 * 4x3` reduction used by
/// `linearizePoint` for the landmark block.
///
/// Eigen evaluates the first four column-major output lanes as one Packet4f:
/// `[row0-col0, row1-col0, row0-col1, row1-col1]`.  Its fixed-size product
/// computes the two pair reductions `(k0 + k1)` and `(k2 + k3)` with the
/// second term contracted, then adds those pair results.  The final column is
/// the scalar tail with the same two FMA/add pair reductions.  Keep the
/// homogeneous fourth row and column in the inputs even when they are zero;
/// collapsing this to a `2x3 * 3x3` product changes f32 lanes.
pub(super) fn eigen_landmark_jacobian_f32(
    projection_jacobian: SMatrix<f32, 2, 4>,
    point_wrt_landmark: SMatrix<f32, 4, 3>,
) -> SMatrix<f32, 2, 3> {
    // This is the scalar spelling of Eigen's Packet4f operation.  Arrays are
    // deliberately used instead of target-specific intrinsics so the helper
    // remains portable while preserving the packet lane order and operation
    // grouping in source.
    let pair01 = [
        projection_jacobian[(0, 1)].mul_add(
            point_wrt_landmark[(1, 0)],
            projection_jacobian[(0, 0)] * point_wrt_landmark[(0, 0)],
        ),
        projection_jacobian[(1, 1)].mul_add(
            point_wrt_landmark[(1, 0)],
            projection_jacobian[(1, 0)] * point_wrt_landmark[(0, 0)],
        ),
        projection_jacobian[(0, 1)].mul_add(
            point_wrt_landmark[(1, 1)],
            projection_jacobian[(0, 0)] * point_wrt_landmark[(0, 1)],
        ),
        projection_jacobian[(1, 1)].mul_add(
            point_wrt_landmark[(1, 1)],
            projection_jacobian[(1, 0)] * point_wrt_landmark[(0, 1)],
        ),
    ];
    let pair23 = [
        projection_jacobian[(0, 3)].mul_add(
            point_wrt_landmark[(3, 0)],
            projection_jacobian[(0, 2)] * point_wrt_landmark[(2, 0)],
        ),
        projection_jacobian[(1, 3)].mul_add(
            point_wrt_landmark[(3, 0)],
            projection_jacobian[(1, 2)] * point_wrt_landmark[(2, 0)],
        ),
        projection_jacobian[(0, 3)].mul_add(
            point_wrt_landmark[(3, 1)],
            projection_jacobian[(0, 2)] * point_wrt_landmark[(2, 1)],
        ),
        projection_jacobian[(1, 3)].mul_add(
            point_wrt_landmark[(3, 1)],
            projection_jacobian[(1, 2)] * point_wrt_landmark[(2, 1)],
        ),
    ];
    let packet = [
        pair01[0] + pair23[0],
        pair01[1] + pair23[1],
        pair01[2] + pair23[2],
        pair01[3] + pair23[3],
    ];

    let tail_pair01 = projection_jacobian[(0, 1)].mul_add(
        point_wrt_landmark[(1, 2)],
        projection_jacobian[(0, 0)] * point_wrt_landmark[(0, 2)],
    );
    let tail_pair23 = projection_jacobian[(0, 3)].mul_add(
        point_wrt_landmark[(3, 2)],
        projection_jacobian[(0, 2)] * point_wrt_landmark[(2, 2)],
    );
    let tail_row0 = tail_pair01 + tail_pair23;
    let tail_pair01 = projection_jacobian[(1, 1)].mul_add(
        point_wrt_landmark[(1, 2)],
        projection_jacobian[(1, 0)] * point_wrt_landmark[(0, 2)],
    );
    let tail_pair23 = projection_jacobian[(1, 3)].mul_add(
        point_wrt_landmark[(3, 2)],
        projection_jacobian[(1, 2)] * point_wrt_landmark[(2, 2)],
    );
    let tail_row1 = tail_pair01 + tail_pair23;

    let mut result = SMatrix::<f32, 2, 3>::zeros();
    result[(0, 0)] = packet[0];
    result[(1, 0)] = packet[1];
    result[(0, 1)] = packet[2];
    result[(1, 1)] = packet[3];
    result[(0, 2)] = tail_row0;
    result[(1, 2)] = tail_row1;
    result
}

/// Reproduce the same Eigen `2x4 * 4x6` reduction used for the relative-pose
/// block returned by `linearizePoint`.  The fourth projection/point row is
/// homogeneous and zero, but Eigen still keeps it in the fixed-size product's
/// reduction tree.  The pairwise `(k0 + k1) + (k2 + k3)` order is observably
/// different from nalgebra's left-associated `2x3 * 3x6` product in f32.
pub(super) fn eigen_relative_pose_jacobian_f32(
    projection_jacobian: SMatrix<f32, 2, 3>,
    point_wrt_relative_pose: SMatrix<f32, 3, 6>,
) -> SMatrix<f32, 2, 6> {
    let mut result = SMatrix::<f32, 2, 6>::zeros();
    for row in 0..2 {
        for column in 0..6 {
            let pair01 = projection_jacobian[(row, 0)] * point_wrt_relative_pose[(0, column)];
            let pair01 =
                projection_jacobian[(row, 1)].mul_add(point_wrt_relative_pose[(1, column)], pair01);
            let pair23 = projection_jacobian[(row, 2)] * point_wrt_relative_pose[(2, column)];
            result[(row, column)] = pair01 + pair23;
        }
    }
    result
}

/// Float-owned counterpart of [`anchored_visual_reprojection_factor`].  This
/// is deliberately separate from the f64 API so ExtendedF64 retains its
/// historical path and unit-test tolerances.
#[allow(clippy::too_many_arguments)]
pub fn anchored_visual_reprojection_factor_f32(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    same_time_camera: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    anchored_visual_reprojection_factor_f32_with_time_cam(
        camera,
        anchor_imu_pose,
        anchor_t_imu_cam,
        target_imu_pose,
        target_t_imu_cam,
        landmark,
        observation,
        same_time_camera,
        same_time_camera,
        config,
    )
}

/// Float-owned counterpart of
/// [`anchored_visual_reprojection_factor_with_time_cam`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn anchored_visual_reprojection_factor_f32_with_time_cam(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    _same_timestamp: bool,
    same_time_cam_id: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    anchored_visual_reprojection_factor_f32_with_time_cam_fej(
        camera,
        anchor_imu_pose,
        anchor_imu_pose,
        anchor_t_imu_cam,
        target_imu_pose,
        target_imu_pose,
        target_t_imu_cam,
        landmark,
        observation,
        _same_timestamp,
        same_time_cam_id,
        config,
    )
}

/// Float visual factor with the upstream split between the current value
/// state and the FEJ state used for pose derivatives.
///
/// The value-side camera chain remains the source for the residual and
/// landmark/value Jacobian.  The anchor absolute-pose block uses the separate
/// FEJ endpoint together with the FEJ host inverse rotation, while the target
/// block retains its existing target-side FEJ rotation input.  This keeps the
/// schedule distinction explicit: the single-host/target endpoint fixture
/// does not contain a distinct FEJ state, so it cannot by itself select this
/// production endpoint semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn anchored_visual_reprojection_factor_f32_with_time_cam_fej(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_imu_pose_fej: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_imu_pose_fej: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    _same_timestamp: bool,
    same_time_cam_id: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    // Compatibility callers have no lifecycle flags; retain their explicit
    // current-value contract. Window callers use the lifecycle-aware entry.
    anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
        camera,
        anchor_imu_pose,
        anchor_imu_pose_fej,
        anchor_t_imu_cam,
        target_imu_pose,
        target_imu_pose_fej,
        target_t_imu_cam,
        landmark,
        observation,
        _same_timestamp,
        same_time_cam_id,
        true,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
    camera: &DoubleSphereCamera,
    anchor_imu_pose: &SE3,
    anchor_imu_pose_fej: &SE3,
    anchor_t_imu_cam: &SE3,
    target_imu_pose: &SE3,
    target_imu_pose_fej: &SE3,
    target_t_imu_cam: &SE3,
    landmark: &InverseDistanceLandmark,
    observation: Point2<f64>,
    _same_timestamp: bool,
    same_time_cam_id: bool,
    either_endpoint_linearized: bool,
    config: FactorConfig,
) -> Option<AnchoredVisualFactor> {
    let observation_stddev = config.observation_stddev as f32;
    if !observation_stddev.is_finite() || observation_stddev <= 0.0 {
        return None;
    }
    let anchor_imu_pose = F32Pose::from_se3(anchor_imu_pose);
    let anchor_imu_pose_fej = F32Pose::from_se3(anchor_imu_pose_fej);
    let anchor_t_imu_cam = F32Pose::from_se3(anchor_t_imu_cam);
    let target_imu_pose = F32Pose::from_se3(target_imu_pose);
    let target_imu_pose_fej = F32Pose::from_se3(target_imu_pose_fej);
    let target_t_imu_cam = F32Pose::from_se3(target_t_imu_cam);
    let target_camera_from_imu = target_t_imu_cam.inverse();
    // Keep the translation expression in the same source order as
    // `computeRelPose`: form the world-frame difference first, then apply the
    // target inverse rotation.  Building `target_pose.inverse()` and composing
    // the host pose is algebraically equivalent, but its two point actions
    // expose a different f32 rounding path for small cross-time translations.
    let target_inverse_rotation = sophus_so3_inverse(target_imu_pose.rotation);
    let target_imu_from_anchor_imu = F32Pose {
        // The current value-side computeRelPose branch is inlined into
        // linearizePoint in the pinned Eigen build.  Its relative quaternion
        // product has a distinct Packet4f/FMA lane schedule from the
        // out-of-line FEJ product; keep this substitution local to the
        // current chain so d_rel and all other products retain their source
        // paths.
        rotation: sophus_quat_product_current_packet_f32(
            target_inverse_rotation,
            anchor_imu_pose.rotation,
        ),
        translation: sophus_rotate_difference_visual_f32(
            target_inverse_rotation,
            anchor_imu_pose.translation,
            target_imu_pose.translation,
        ),
    };
    let target_camera_from_anchor_imu = F32Pose {
        rotation: sophus_quat_product_camera_prefix_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.rotation,
        ),
        translation: sophus_rotate_f32(
            target_camera_from_imu.rotation,
            target_imu_from_anchor_imu.translation,
        ) + target_camera_from_imu.translation,
    };
    // Keep the value-side camera chain above for the residual, landmark
    // projection, and target-camera endpoint.  The separately reconstructed
    // FEJ endpoint below is the production source for the anchor absolute-pose
    // block.  Its inputs are frozen FEJ poses, but its arithmetic intentionally
    // retains the prepatch packet-quaternion/camera-prefix schedule so this
    // control isolates endpoint inputs from endpoint rounding order.
    let target_camera_from_anchor_imu_fej = if same_time_cam_id {
        F32Pose {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    } else {
        let target_inverse_rotation_fej = sophus_so3_inverse(target_imu_pose_fej.rotation);
        let target_imu_from_anchor_imu_fej = F32Pose {
            rotation: sophus_quat_product_current_packet_f32(
                target_inverse_rotation_fej,
                anchor_imu_pose_fej.rotation,
            ),
            translation: sophus_rotate_difference_f32(
                target_inverse_rotation_fej,
                anchor_imu_pose_fej.translation,
                target_imu_pose_fej.translation,
            ),
        };
        F32Pose {
            rotation: sophus_quat_product_camera_prefix_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu_fej.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_imu.rotation,
                target_imu_from_anchor_imu_fej.translation,
            ) + target_camera_from_imu.translation,
        }
    };
    // Upstream linearizeProblem retains the poseLin value unless at least
    // one endpoint is linearized. Equal pose bits do not determine this flag.
    let target_camera_from_anchor_imu = if either_endpoint_linearized {
        target_camera_from_anchor_imu
    } else {
        target_camera_from_anchor_imu_fej
    };
    let target_camera_from_anchor_camera = if same_time_cam_id {
        // Preserve Basalt's exact `TimeCamId` equality branch.  In
        // particular, do not let f32 inverse/compose roundoff turn an
        // identity same-camera observation into a tiny relative transform.
        F32Pose {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    } else {
        F32Pose {
            rotation: sophus_quat_product_camera_suffix_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_t_imu_cam.rotation,
            ),
            translation: sophus_rotate_f32(
                target_camera_from_anchor_imu.rotation,
                anchor_t_imu_cam.translation,
            ) + target_camera_from_anchor_imu.translation,
        }
    };

    let bearing = landmark.direction.bearing_f32();
    let point4_target = eigen_homogeneous_point_gemv_f32(
        target_camera_from_anchor_camera.rotation,
        target_camera_from_anchor_camera.translation,
        bearing,
        landmark.inverse_distance as f32,
    );
    let point_target = point4_target.fixed_rows::<3>(0).into_owned();
    let (predicted, projection_jacobian) =
        project_double_sphere_with_jacobian_f32(camera, point_target)?;
    let raw = predicted - Vector2::new(observation.x as f32, observation.y as f32);
    let residual_squared = eigen_vector2_squared_norm_f32(raw.x, raw.y);
    let huber_delta = config.huber_delta as f32;
    let huber_weight = if huber_delta > 0.0 && residual_squared > huber_delta * huber_delta {
        huber_delta / residual_squared.sqrt()
    } else {
        1.0_f32
    };
    let sqrt_weight = huber_weight.sqrt() / observation_stddev;
    let objective_cost = 0.5_f32 * (2.0_f32 - huber_weight) * huber_weight * residual_squared
        / (observation_stddev * observation_stddev);

    let mut point_wrt_relative_pose = SMatrix::<f32, 3, 6>::zeros();
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(SMatrix::<f32, 3, 3>::identity() * landmark.inverse_distance as f32));
    point_wrt_relative_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew3_f32(point_target)));
    let residual_wrt_relative_pose =
        eigen_relative_pose_jacobian_f32(projection_jacobian, point_wrt_relative_pose);

    let (relative_wrt_anchor, relative_wrt_target) = if same_time_cam_id {
        // Exact TimeCamId identity is the only branch that skips
        // computeRelPose and supplies zero pose Jacobians. Same-timestamp
        // stereo uses the ordinary relative camera chain below.
        (SMatrix::<f32, 6, 6>::zeros(), SMatrix::<f32, 6, 6>::zeros())
    } else {
        let r_w_i_anchor_inv =
            eigen_quaternion_matrix_f32(sophus_so3_inverse(anchor_imu_pose_fej.rotation));
        let r_w_i_target_inv =
            eigen_quaternion_matrix_f32(sophus_so3_inverse(target_imu_pose_fej.rotation));
        (
            eigen_adjoint_times_rotation_blocks_f32(
                target_camera_from_anchor_imu_fej,
                r_w_i_anchor_inv,
                1.0,
            ),
            eigen_adjoint_times_rotation_blocks_f32(target_camera_from_imu, r_w_i_target_inv, -1.0),
        )
    };

    let direction_jacobian = landmark.direction.bearing_jacobian_f32();
    let rotation = eigen_quaternion_matrix_f32(target_camera_from_anchor_camera.rotation);
    let mut transform_top_left = SMatrix::<f32, 3, 4>::zeros();
    transform_top_left
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation);
    let mut source_jup = SMatrix::<f32, 4, 2>::zeros();
    source_jup
        .fixed_view_mut::<3, 2>(0, 0)
        .copy_from(&direction_jacobian);
    let mut point_wrt_landmark = SMatrix::<f32, 3, 3>::zeros();
    point_wrt_landmark.fixed_view_mut::<3, 2>(0, 0).copy_from(
        &eigen_homogeneous_landmark_direction_f32(transform_top_left, source_jup),
    );
    point_wrt_landmark.set_column(2, &target_camera_from_anchor_camera.translation);
    let anchor = eigen_weighted_pose_jacobian_f32(
        residual_wrt_relative_pose,
        relative_wrt_anchor,
        sqrt_weight,
    );
    let target = eigen_weighted_pose_jacobian_f32(
        residual_wrt_relative_pose,
        relative_wrt_target,
        sqrt_weight,
    );
    let mut camera_jacobian = SMatrix::<f32, 2, 4>::zeros();
    camera_jacobian
        .fixed_view_mut::<2, 3>(0, 0)
        .copy_from(&projection_jacobian);
    let mut homogeneous_point_wrt_landmark = SMatrix::<f32, 4, 3>::zeros();
    homogeneous_point_wrt_landmark
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&point_wrt_landmark);
    // `Jpp.col(2)` is the full homogeneous transform's fourth column.  Its
    // final row is the exact affine homogeneous scale, even though the
    // camera's fourth Jacobian column is zero.
    homogeneous_point_wrt_landmark[(3, 2)] = 1.0_f32;
    let landmark_jacobian =
        eigen_landmark_jacobian_f32(camera_jacobian, homogeneous_point_wrt_landmark) * sqrt_weight;
    let debug_chain = if crate::vio::window::diagnostic_env_snapshot().visual_chain_trace {
        Some(VisualChainF32 {
            target_camera_from_imu: visual_chain_pose_snapshot(target_camera_from_imu),
            target_imu_from_anchor_imu: visual_chain_pose_snapshot(target_imu_from_anchor_imu),
            target_camera_from_anchor_imu: visual_chain_pose_snapshot(
                target_camera_from_anchor_imu,
            ),
            target_camera_from_anchor_imu_fej: visual_chain_pose_snapshot(
                target_camera_from_anchor_imu_fej,
            ),
            target_camera_from_anchor_camera: visual_chain_pose_snapshot(
                target_camera_from_anchor_camera,
            ),
            target_camera_matrix: {
                let transform = eigen_homogeneous_transform_f32(
                    target_camera_from_anchor_camera.rotation,
                    target_camera_from_anchor_camera.translation,
                );
                let mut values = [0.0_f32; 16];
                values.copy_from_slice(transform.as_slice());
                values
            },
            // Copy the matrix used by production above so the opt-in sidecar
            // cannot silently drift to a separately reconstructed endpoint.
            relative_wrt_anchor: {
                let mut values = [0.0_f32; 36];
                values.copy_from_slice(relative_wrt_anchor.as_slice());
                values
            },
            point4_target: [
                point4_target[0],
                point4_target[1],
                point4_target[2],
                point4_target[3],
            ],
            point_target: [point_target.x, point_target.y, point_target.z],
            projection: [predicted.x, predicted.y],
            projection_jacobian: [
                projection_jacobian[(0, 0)],
                projection_jacobian[(0, 1)],
                projection_jacobian[(0, 2)],
                projection_jacobian[(1, 0)],
                projection_jacobian[(1, 1)],
                projection_jacobian[(1, 2)],
            ],
        })
    } else {
        None
    };
    Some(AnchoredVisualFactor {
        raw_residual: raw.map(|value| value as f64),
        projection: predicted.map(|value| value as f64),
        huber_weight: huber_weight as f64,
        sqrt_weight: sqrt_weight as f64,
        residual: raw.map(|value| (value * sqrt_weight) as f64),
        anchor_pose_jacobian: anchor.map(|value| value as f64),
        target_pose_jacobian: target.map(|value| value as f64),
        landmark_jacobian: landmark_jacobian.map(|value| value as f64),
        objective_cost: objective_cost as f64,
        debug_chain,
    })
}
