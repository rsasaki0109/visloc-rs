//! Flatten / decode helpers between window states and the global state
//! vector.

use super::*;

pub fn flatten_states(states: &[WindowState]) -> DVector<f64> {
    let mut out = DVector::zeros(states.len() * NAV_STATE_DOF);
    for (index, state) in states.iter().enumerate() {
        out.rows_mut(index * NAV_STATE_DOF, NAV_STATE_DOF)
            .copy_from(&flatten_nav(&state.nav));
    }
    out
}

/// Flatten the mixed absolute order used by the square-root VIO system:
/// pose-only keyframes first, followed by full navigation states.
pub fn flatten_blocks(poses: &[WindowPose], states: &[WindowState]) -> DVector<f64> {
    flatten_blocks_with_mode(poses, states, ScalarMode::ExtendedF64)
}

/// Flatten the state in the scalar representation owned by the active
/// estimator.  The upstream float estimator keeps an SO3 object as the state
/// block and applies local rotation increments to that object; it does not
/// round-trip the initial quaternion through a double-precision log/exp.
/// Keep the three stored rotation lanes as the f32 quaternion xyz lanes in
/// this internal vector representation so decoding preserves the source f32
/// components exactly.  The LM step still remains a local scaled-axis
/// increment (see `apply_step`).
pub fn flatten_blocks_with_mode(
    poses: &[WindowPose],
    states: &[WindowState],
    scalar_mode: ScalarMode,
) -> DVector<f64> {
    let mut out = DVector::zeros(poses.len() * POSE_DOF + states.len() * NAV_STATE_DOF);
    for (index, pose) in poses.iter().enumerate() {
        out.rows_mut(index * POSE_DOF, POSE_DOF)
            .copy_from(&flatten_pose_with_mode(&pose.pose, scalar_mode));
    }
    let state_offset = poses.len() * POSE_DOF;
    for (index, state) in states.iter().enumerate() {
        out.rows_mut(state_offset + index * NAV_STATE_DOF, NAV_STATE_DOF)
            .copy_from(&flatten_nav_with_mode(&state.nav, scalar_mode));
    }
    out
}

pub fn write_states(states: &mut [WindowState], vector: &DVector<f64>) {
    for (index, state) in states.iter_mut().enumerate() {
        state.nav = decode_state(vector.rows(index * NAV_STATE_DOF, NAV_STATE_DOF));
        if !state.linearized {
            state.linearized_nav = state.nav.clone();
            state.linearized_delta.fill(0.0);
        } else {
            state.stored_current_nav = state.nav.clone();
        }
    }
}

pub fn write_blocks(poses: &mut [WindowPose], states: &mut [WindowState], vector: &DVector<f64>) {
    write_blocks_with_mode(poses, states, vector, ScalarMode::ExtendedF64);
}

pub fn write_blocks_with_mode(
    poses: &mut [WindowPose],
    states: &mut [WindowState],
    vector: &DVector<f64>,
    scalar_mode: ScalarMode,
) {
    for (index, pose) in poses.iter_mut().enumerate() {
        let block = vector.rows(index * POSE_DOF, POSE_DOF);
        let mut decoded = decode_pose_with_mode(&block, scalar_mode);
        // A solved UpstreamF32 block is paired with the full pose sidecar
        // maintained by WindowProblem.  Preserve its propagated w component
        // when the compact xyz lanes identify the same pose; arbitrary
        // callers that provide a different chart still get the historical
        // reconstruction behavior.
        if scalar_mode == ScalarMode::UpstreamF32
            && quaternion_xyz_matches(&[block[3], block[4], block[5]], &pose.pose.rotation)
        {
            decoded.rotation = pose.pose.rotation.clone();
        }
        pose.pose = decoded;
        pose.stored_current_pose = pose.pose.clone();
    }
    let state_offset = poses.len() * POSE_DOF;
    for (index, state) in states.iter_mut().enumerate() {
        let block = vector.rows(state_offset + index * NAV_STATE_DOF, NAV_STATE_DOF);
        let mut decoded = decode_state_with_mode(&block, scalar_mode);
        if scalar_mode == ScalarMode::UpstreamF32
            && quaternion_xyz_matches(
                &[block[3], block[4], block[5]],
                &state.nav.imu_to_world.rotation,
            )
        {
            decoded.imu_to_world.rotation = state.nav.imu_to_world.rotation.clone();
        }
        state.nav = decoded;
        // Upstream `applyInc` mutates `state_linearized` while a block is
        // still non-linearized.  Keep the Rust FEJ sidecar at that moving
        // point; frozen blocks retain their original linearization point.
        if !state.linearized {
            state.linearized_nav = state.nav.clone();
            state.linearized_delta.fill(0.0);
        } else {
            state.stored_current_nav = state.nav.clone();
        }
    }
}

pub fn decode_states(vector: &DVector<f64>, count: usize) -> Vec<BasaltNavState> {
    (0..count)
        .map(|index| decode_state(vector.rows(index * NAV_STATE_DOF, NAV_STATE_DOF)))
        .collect()
}

pub fn flatten_nav(nav: &BasaltNavState) -> DVector<f64> {
    flatten_nav_with_mode(nav, ScalarMode::ExtendedF64)
}

pub fn flatten_nav_with_mode(nav: &BasaltNavState, scalar_mode: ScalarMode) -> DVector<f64> {
    let rotation = if scalar_mode == ScalarMode::UpstreamF32 {
        let quaternion = quat_f32_unchecked_from_f64(&nav.imu_to_world.rotation);
        Vector3::new(
            quaternion.i as f64,
            quaternion.j as f64,
            quaternion.k as f64,
        )
    } else {
        nav.imu_to_world.rotation.scaled_axis()
    };
    let cast = |value: f64| {
        if scalar_mode == ScalarMode::UpstreamF32 {
            (value as f32) as f64
        } else {
            value
        }
    };
    DVector::from_iterator(
        NAV_STATE_DOF,
        nav.imu_to_world
            .translation
            .iter()
            .map(|value| cast(*value))
            .chain(rotation.iter().copied())
            .chain(nav.velocity_world_m_s.iter().copied())
            .chain(nav.gyro_bias_rad_s.iter().copied())
            .chain(nav.accel_bias_m_s2.iter().copied())
            .map(cast),
    )
}

pub fn flatten_pose(pose: &SE3) -> DVector<f64> {
    flatten_pose_with_mode(pose, ScalarMode::ExtendedF64)
}

pub(crate) fn flatten_pose_with_mode(pose: &SE3, scalar_mode: ScalarMode) -> DVector<f64> {
    let rotation = if scalar_mode == ScalarMode::UpstreamF32 {
        let quaternion = quat_f32_unchecked_from_f64(&pose.rotation);
        Vector3::new(
            quaternion.i as f64,
            quaternion.j as f64,
            quaternion.k as f64,
        )
    } else {
        pose.rotation.scaled_axis()
    };
    let cast = |value: f64| {
        if scalar_mode == ScalarMode::UpstreamF32 {
            (value as f32) as f64
        } else {
            value
        }
    };
    DVector::from_iterator(
        POSE_DOF,
        pose.translation
            .iter()
            .map(|value| cast(*value))
            .chain(rotation.iter().copied())
            .map(cast),
    )
}

pub fn decode_pose(vector: impl AsRef<[f64]>) -> SE3 {
    decode_pose_with_mode(vector, ScalarMode::ExtendedF64)
}

pub(super) fn decode_pose_with_mode(vector: impl AsRef<[f64]>, scalar_mode: ScalarMode) -> SE3 {
    let vector = vector.as_ref();
    SE3::new(
        decode_rotation_with_mode(&vector[3..6], scalar_mode),
        Vector3::new(
            cast_component(vector[0], scalar_mode),
            cast_component(vector[1], scalar_mode),
            cast_component(vector[2], scalar_mode),
        ),
    )
}

pub fn local_pose_difference(current: &SE3, reference: &SE3) -> DVector<f64> {
    // Basalt's PoseState::incPose is not an SE(3) exponential update: the
    // translation is additive in world coordinates and the rotation is a
    // left SO(3) increment.  FEJ must use the inverse of that exact chart,
    // rather than SE3::log(), whose translational component contains the
    // SO(3) left-Jacobian coupling.
    let rotation = (current.rotation * reference.rotation.inverse()).scaled_axis();
    DVector::from_iterator(
        POSE_DOF,
        current
            .translation
            .iter()
            .zip(reference.translation.iter())
            .map(|(current, reference)| current - reference)
            .chain(rotation.iter().copied()),
    )
}

pub(super) const fn block_dof(kind: WindowBlockKind) -> usize {
    match kind {
        WindowBlockKind::Pose => POSE_DOF,
        WindowBlockKind::StatePose => POSE_DOF,
        WindowBlockKind::State => NAV_STATE_DOF,
    }
}

/// Left-local SE3 boxminus plus additive navigation components. This is the
/// coordinate chart used by visual/IMU pose Jacobians and by FEJ priors.
pub fn local_state_difference(
    current: &BasaltNavState,
    reference: &BasaltNavState,
) -> DVector<f64> {
    let rotation =
        (current.imu_to_world.rotation * reference.imu_to_world.rotation.inverse()).scaled_axis();
    DVector::from_iterator(
        NAV_STATE_DOF,
        current
            .imu_to_world
            .translation
            .iter()
            .zip(reference.imu_to_world.translation.iter())
            .map(|(current, reference)| current - reference)
            .chain(rotation.iter().copied())
            .chain(
                (current.velocity_world_m_s - reference.velocity_world_m_s)
                    .iter()
                    .copied(),
            )
            .chain(
                (current.gyro_bias_rad_s - reference.gyro_bias_rad_s)
                    .iter()
                    .copied(),
            )
            .chain(
                (current.accel_bias_m_s2 - reference.accel_bias_m_s2)
                    .iter()
                    .copied(),
            ),
    )
}

/// Accumulate the tangent exactly at the solver's scalar boundary.  Upstream
/// stores `delta += inc` in `float`; reconstructing it from two poses would
/// replace the source's sum of local rotations with a nonlinear box-minus.
pub(super) fn accumulate_linearized_delta(
    delta: &mut DVector<f64>,
    increment: &DVector<f64>,
    scalar_mode: ScalarMode,
) {
    if delta.len() != increment.len() {
        return;
    }
    if scalar_mode == ScalarMode::UpstreamF32 {
        for (lhs, rhs) in delta.iter_mut().zip(increment.iter()) {
            *lhs = ((*lhs as f32) + (*rhs as f32)) as f64;
        }
    } else {
        *delta += increment;
    }
}

pub fn decode_state(vector: impl AsRef<[f64]>) -> BasaltNavState {
    decode_state_with_mode(vector, ScalarMode::ExtendedF64)
}

pub(super) fn decode_state_with_mode(
    vector: impl AsRef<[f64]>,
    scalar_mode: ScalarMode,
) -> BasaltNavState {
    let vector = vector.as_ref();
    let translation = Vector3::new(
        cast_component(vector[0], scalar_mode),
        cast_component(vector[1], scalar_mode),
        cast_component(vector[2], scalar_mode),
    );
    let rotation = decode_rotation_with_mode(&vector[3..6], scalar_mode);
    BasaltNavState {
        imu_to_world: SE3::new(rotation, translation),
        velocity_world_m_s: Vector3::new(
            cast_component(vector[6], scalar_mode),
            cast_component(vector[7], scalar_mode),
            cast_component(vector[8], scalar_mode),
        ),
        gyro_bias_rad_s: Vector3::new(
            cast_component(vector[9], scalar_mode),
            cast_component(vector[10], scalar_mode),
            cast_component(vector[11], scalar_mode),
        ),
        accel_bias_m_s2: Vector3::new(
            cast_component(vector[12], scalar_mode),
            cast_component(vector[13], scalar_mode),
            cast_component(vector[14], scalar_mode),
        ),
    }
}

fn cast_component(value: f64, scalar_mode: ScalarMode) -> f64 {
    if scalar_mode == ScalarMode::UpstreamF32 {
        (value as f32) as f64
    } else {
        value
    }
}

fn quat_f32_unchecked_from_f64(rotation: &UnitQuaternion<f64>) -> UnitQuaternion<f32> {
    let q = rotation.quaternion();
    UnitQuaternion::new_unchecked(Quaternion::new(
        q.w as f32, q.i as f32, q.j as f32, q.k as f32,
    ))
}

fn quat_f64_unchecked_from_f32(rotation: &UnitQuaternion<f32>) -> UnitQuaternion<f64> {
    let q = rotation.quaternion();
    UnitQuaternion::new_unchecked(Quaternion::new(
        q.w as f64, q.i as f64, q.j as f64, q.k as f64,
    ))
}

/// Return whether a compact UpstreamF32 block still refers to the quaternion
/// carried by the full state sidecar.  The comparison is deliberately at the
/// source scalar boundary: no norm, log/exp, or epsilon chart can recover the
/// fourth lane once it has been discarded.
pub(super) fn quaternion_xyz_matches(vector: &[f64], rotation: &UnitQuaternion<f64>) -> bool {
    if vector.len() != 3 {
        return false;
    }
    let quaternion = quat_f32_unchecked_from_f64(rotation);
    vector[0] as f32 == quaternion.i
        && vector[1] as f32 == quaternion.j
        && vector[2] as f32 == quaternion.k
}

fn decode_rotation_with_mode(vector: &[f64], scalar_mode: ScalarMode) -> UnitQuaternion<f64> {
    if scalar_mode != ScalarMode::UpstreamF32 {
        return UnitQuaternion::from_scaled_axis(Vector3::new(vector[0], vector[1], vector[2]));
    }
    let x = vector[0] as f32;
    let y = vector[1] as f32;
    let z = vector[2] as f32;
    let norm_sq = x * x + y * y + z * z;
    let w = (1.0_f32 - norm_sq).max(0.0).sqrt();
    let rotation = UnitQuaternion::new_unchecked(Quaternion::new(w, x, y, z));
    quat_f64_unchecked_from_f32(&rotation)
}
