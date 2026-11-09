//! FNV fingerprints that bind an LM trial token to the window generation and
//! factor plan.

use super::*;

const LM_TOKEN_FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const LM_TOKEN_FNV_PRIME: u64 = 1_099_511_628_211;

#[inline]
const fn lm_token_hash_u64(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(LM_TOKEN_FNV_PRIME);
}

#[inline]
const fn lm_token_hash_usize(hash: &mut u64, value: usize) {
    lm_token_hash_u64(hash, value as u64);
}

#[inline]
const fn lm_token_hash_bool(hash: &mut u64, value: bool) {
    lm_token_hash_u64(hash, value as u64);
}

#[inline]
const fn lm_token_hash_f64(hash: &mut u64, value: f64) {
    lm_token_hash_u64(hash, value.to_bits());
}

#[inline]
fn lm_token_hash_se3(hash: &mut u64, value: &SE3) {
    for component in value.rotation.coords.iter() {
        lm_token_hash_f64(hash, *component);
    }
    for component in value.translation.iter() {
        lm_token_hash_f64(hash, *component);
    }
}

#[inline]
fn lm_token_hash_nav(hash: &mut u64, value: &BasaltNavState) {
    lm_token_hash_se3(hash, &value.imu_to_world);
    for component in value.velocity_world_m_s.iter() {
        lm_token_hash_f64(hash, *component);
    }
    for component in value.gyro_bias_rad_s.iter() {
        lm_token_hash_f64(hash, *component);
    }
    for component in value.accel_bias_m_s2.iter() {
        lm_token_hash_f64(hash, *component);
    }
}

#[inline]
fn lm_token_hash_vector(hash: &mut u64, value: &DVector<f64>) {
    lm_token_hash_usize(hash, value.len());
    for component in value.iter() {
        lm_token_hash_f64(hash, *component);
    }
}

fn lm_token_camera_fingerprint(hash: &mut u64, camera: &DoubleSphereCamera) {
    lm_token_hash_f64(hash, camera.fx);
    lm_token_hash_f64(hash, camera.fy);
    lm_token_hash_f64(hash, camera.cx);
    lm_token_hash_f64(hash, camera.cy);
    lm_token_hash_f64(hash, camera.xi);
    lm_token_hash_f64(hash, camera.alpha);
    lm_token_hash_u64(hash, camera.width as u64);
    lm_token_hash_u64(hash, camera.height as u64);
}

/// Fingerprint the current window identity/topology and mutable sidecars that
/// can make a previously recovered landmark increment stale.  This is a
/// deterministic, allocation-free FNV-1a pass; the object address is kept as
/// a separate binding field so a clone cannot consume another window's token.
pub(super) fn lm_token_window_generation(problem: &WindowProblem) -> u64 {
    let mut hash = LM_TOKEN_FNV_OFFSET;
    lm_token_hash_usize(&mut hash, problem.cameras.len());
    lm_token_hash_usize(&mut hash, problem.t_imu_cam.len());
    lm_token_camera_fingerprint(&mut hash, &problem.camera);
    for camera in &problem.cameras {
        lm_token_camera_fingerprint(&mut hash, camera);
    }
    for transform in &problem.t_imu_cam {
        lm_token_hash_se3(&mut hash, transform);
    }

    lm_token_hash_usize(&mut hash, problem.poses.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, pose.frame_id);
        lm_token_hash_u64(&mut hash, pose.timestamp_ns as u64);
        lm_token_hash_bool(&mut hash, pose.is_keyframe);
        lm_token_hash_se3(&mut hash, &pose.pose);
        lm_token_hash_se3(&mut hash, &pose.stored_current_pose);
        lm_token_hash_se3(&mut hash, &pose.linearized_pose);
        lm_token_hash_vector(&mut hash, &pose.linearized_delta);
    }

    lm_token_hash_usize(&mut hash, problem.states.len());
    for (index, state) in problem.states.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, state.frame_id);
        lm_token_hash_u64(&mut hash, state.timestamp_ns as u64);
        lm_token_hash_bool(&mut hash, state.is_keyframe);
        lm_token_hash_bool(&mut hash, state.is_latest);
        lm_token_hash_bool(&mut hash, state.linearized);
        lm_token_hash_nav(&mut hash, &state.nav);
        lm_token_hash_nav(&mut hash, &state.stored_current_nav);
        lm_token_hash_nav(&mut hash, &state.linearized_nav);
        lm_token_hash_vector(&mut hash, &state.linearized_delta);
    }

    lm_token_hash_usize(&mut hash, problem.trial_host_order.len());
    for &(timestamp, camera) in &problem.trial_host_order {
        lm_token_hash_u64(&mut hash, timestamp);
        lm_token_hash_u64(&mut hash, camera as u64);
    }
    lm_token_hash_usize(&mut hash, problem.landmarks.len());
    for (index, landmark) in problem.landmarks.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, landmark.track_id);
        lm_token_hash_usize(&mut hash, landmark.anchor_state_index);
        lm_token_hash_u64(&mut hash, landmark.anchor_camera_id as u64);
        lm_token_hash_f64(&mut hash, landmark.direction.xy.x);
        lm_token_hash_f64(&mut hash, landmark.direction.xy.y);
        lm_token_hash_f64(&mut hash, landmark.inverse_distance);
        lm_token_hash_usize(&mut hash, landmark.observations.len());
        for observation in &landmark.observations {
            lm_token_hash_usize(&mut hash, observation.state_index);
            lm_token_hash_u64(&mut hash, observation.camera_id as u64);
            lm_token_hash_f64(&mut hash, observation.pixel.x);
            lm_token_hash_f64(&mut hash, observation.pixel.y);
        }
    }

    lm_token_hash_usize(&mut hash, problem.imu_links.len());
    for link in &problem.imu_links {
        lm_token_hash_usize(&mut hash, link.from_index);
        lm_token_hash_usize(&mut hash, link.to_index);
        lm_token_hash_f64(&mut hash, link.delta.delta_time);
    }

    match &problem.prior {
        None => lm_token_hash_u64(&mut hash, 0),
        Some(prior) => {
            lm_token_hash_u64(&mut hash, 1);
            lm_token_hash_usize(&mut hash, prior.frame_ids.len());
            for frame_id in &prior.frame_ids {
                lm_token_hash_u64(&mut hash, *frame_id);
            }
            lm_token_hash_usize(&mut hash, prior.block_kinds.len());
            for kind in &prior.block_kinds {
                lm_token_hash_u64(
                    &mut hash,
                    match kind {
                        WindowBlockKind::Pose => 1,
                        WindowBlockKind::StatePose => 3,
                        WindowBlockKind::State => 2,
                    },
                );
            }
            lm_token_hash_usize(&mut hash, prior.jacobian.nrows());
            lm_token_hash_usize(&mut hash, prior.jacobian.ncols());
            lm_token_hash_usize(&mut hash, prior.rhs.len());
            lm_token_hash_usize(&mut hash, prior.fej_point.len());
        }
    }
    match &problem.anchor_point {
        None => lm_token_hash_u64(&mut hash, 0),
        Some(point) => {
            lm_token_hash_u64(&mut hash, 1);
            lm_token_hash_vector(&mut hash, point);
        }
    }
    hash
}

#[inline]
fn lm_token_hash_f64_iter<'a, I>(hash: &mut u64, values: I)
where
    I: IntoIterator<Item = &'a f64>,
{
    for value in values {
        lm_token_hash_f64(hash, *value);
    }
}

/// Hash the immutable numerical inputs consumed by the active factor plan.
///
/// `lm_token_window_generation` deliberately emphasizes mutable window
/// sidecars used by the accepted-step token.  A preparation also needs a
/// separate factor-plan binding: prior values, noise, preintegrated IMU
/// derivatives, and factor-order metadata can change the rows while the
/// solved state/step fingerprints remain unchanged.  This pass is still
/// allocation-free and does not call `linearize`, so binding cannot alter the
/// solver's arithmetic or factor schedule.
fn lm_token_hash_imu_delta(hash: &mut u64, delta: &ImuPreintegratedDelta) {
    lm_token_hash_f64_iter(hash, delta.delta_rotation.coords.iter());
    lm_token_hash_f64_iter(hash, delta.delta_velocity.iter());
    lm_token_hash_f64_iter(hash, delta.delta_position.iter());
    lm_token_hash_f64(hash, delta.delta_time);
    lm_token_hash_f64_iter(hash, delta.bias_gyro.iter());
    lm_token_hash_f64_iter(hash, delta.bias_accel.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_rotation_gyro_bias.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_velocity_gyro_bias.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_velocity_accel_bias.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_position_gyro_bias.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_position_accel_bias.iter());
    lm_token_hash_f64_iter(hash, delta.covariance.iter());
}

pub(super) fn lm_token_factor_plan_fingerprint(problem: &WindowProblem) -> u64 {
    let mut hash = LM_TOKEN_FNV_OFFSET;
    // Version/tag keeps future additions from silently sharing an old binding
    // domain if the factor construction contract changes.
    lm_token_hash_u64(&mut hash, 0x5142_504c_414e_5631);
    lm_token_hash_u64(
        &mut hash,
        match problem.scalar_mode {
            ScalarMode::UpstreamF32 => 1,
            ScalarMode::ExtendedF64 => 2,
        },
    );
    lm_token_hash_camera_plan(&mut hash, problem);

    lm_token_hash_usize(&mut hash, problem.poses.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, pose.frame_id);
        lm_token_hash_u64(&mut hash, pose.timestamp_ns as u64);
        lm_token_hash_bool(&mut hash, pose.is_keyframe);
    }
    lm_token_hash_usize(&mut hash, problem.states.len());
    for (index, state) in problem.states.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, state.frame_id);
        lm_token_hash_u64(&mut hash, state.timestamp_ns as u64);
        lm_token_hash_bool(&mut hash, state.is_keyframe);
        lm_token_hash_bool(&mut hash, state.is_latest);
        lm_token_hash_bool(&mut hash, state.linearized);
    }

    // Visual factor order is landmark-index order, with observation rows in
    // their stored order.  Include both identities and the source geometry so
    // a same-state/step preparation cannot cross a changed visual plan.
    lm_token_hash_usize(&mut hash, problem.trial_host_order.len());
    for &(timestamp, camera) in &problem.trial_host_order {
        lm_token_hash_u64(&mut hash, timestamp);
        lm_token_hash_u64(&mut hash, camera as u64);
    }
    lm_token_hash_usize(&mut hash, problem.landmarks.len());
    for (index, landmark) in problem.landmarks.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, landmark.track_id);
        lm_token_hash_usize(&mut hash, landmark.anchor_state_index);
        lm_token_hash_u64(&mut hash, landmark.anchor_camera_id as u64);
        lm_token_hash_f64_iter(&mut hash, landmark.direction.xy.coords.iter());
        lm_token_hash_f64(&mut hash, landmark.inverse_distance);
        lm_token_hash_usize(&mut hash, landmark.observations.len());
        for (observation_index, observation) in landmark.observations.iter().enumerate() {
            lm_token_hash_usize(&mut hash, observation_index);
            lm_token_hash_usize(&mut hash, observation.state_index);
            lm_token_hash_u64(&mut hash, observation.camera_id as u64);
            lm_token_hash_f64(&mut hash, observation.pixel.x);
            lm_token_hash_f64(&mut hash, observation.pixel.y);
        }
    }

    // Each IMU link contributes a contiguous [Imu9, Bias6] pair.  Hash the
    // complete preintegrated numerical payload (excluding update_trace, which
    // is diagnostic provenance and is not consumed by factor construction).
    lm_token_hash_usize(&mut hash, problem.imu_links.len());
    for (index, link) in problem.imu_links.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_usize(&mut hash, link.from_index);
        lm_token_hash_usize(&mut hash, link.to_index);
        lm_token_hash_imu_delta(&mut hash, &link.delta);
    }
    lm_token_hash_f64(&mut hash, problem.imu_noise.gyro_density);
    lm_token_hash_f64(&mut hash, problem.imu_noise.accel_density);
    lm_token_hash_f64(&mut hash, problem.bias_walk_noise.gyro_density);
    lm_token_hash_f64(&mut hash, problem.bias_walk_noise.accel_density);
    lm_token_hash_f64(&mut hash, problem.initial_pose_weight);
    lm_token_hash_f64(&mut hash, problem.initial_accel_bias_weight);
    lm_token_hash_f64(&mut hash, problem.initial_gyro_bias_weight);
    lm_token_hash_f64_iter(&mut hash, problem.gravity_world.iter());

    match &problem.prior {
        None => lm_token_hash_u64(&mut hash, 0),
        Some(prior) => {
            lm_token_hash_u64(&mut hash, 1);
            lm_token_hash_usize(&mut hash, prior.frame_ids.len());
            for frame_id in &prior.frame_ids {
                lm_token_hash_u64(&mut hash, *frame_id);
            }
            let explicit_kinds = prior.block_kinds.len() == prior.frame_ids.len();
            lm_token_hash_bool(&mut hash, explicit_kinds);
            if explicit_kinds {
                for kind in &prior.block_kinds {
                    lm_token_hash_u64(
                        &mut hash,
                        match kind {
                            WindowBlockKind::Pose => 1,
                            WindowBlockKind::StatePose => 3,
                            WindowBlockKind::State => 2,
                        },
                    );
                }
            } else {
                for _ in &prior.frame_ids {
                    lm_token_hash_u64(&mut hash, 2);
                }
            }
            lm_token_hash_usize(&mut hash, prior.jacobian.nrows());
            lm_token_hash_usize(&mut hash, prior.jacobian.ncols());
            lm_token_hash_f64_iter(&mut hash, prior.jacobian.iter());
            lm_token_hash_usize(&mut hash, prior.rhs.len());
            lm_token_hash_vector(&mut hash, &prior.rhs);
            lm_token_hash_usize(&mut hash, prior.fej_point.len());
            lm_token_hash_vector(&mut hash, &prior.fej_point);
        }
    }
    match &problem.anchor_point {
        None => lm_token_hash_u64(&mut hash, 0),
        Some(point) => {
            lm_token_hash_u64(&mut hash, 1);
            lm_token_hash_vector(&mut hash, point);
        }
    }
    hash
}

fn lm_token_hash_camera_plan(hash: &mut u64, problem: &WindowProblem) {
    lm_token_hash_camera(hash, &problem.camera);
    lm_token_hash_usize(hash, problem.cameras.len());
    for camera in &problem.cameras {
        lm_token_hash_camera(hash, camera);
    }
    lm_token_hash_usize(hash, problem.t_imu_cam.len());
    for transform in &problem.t_imu_cam {
        lm_token_hash_se3(hash, transform);
    }
}

#[inline]
fn lm_token_hash_camera(hash: &mut u64, camera: &DoubleSphereCamera) {
    lm_token_camera_fingerprint(hash, camera);
}
