//! On-manifold IMU pre-integration (Forster et al., T-RO 2017).
//!
//! Given a stream of body-frame angular velocity and linear acceleration
//! samples between two keyframes, this module accumulates the relative
//! rotation, velocity, and position deltas in keyframe-`i`'s body frame,
//! all free of gravity (so the resulting `(ΔR, Δv, Δp)` triplet is what
//! the BA-side factor will compare against `R_iᵀ · R_j`,
//! `R_iᵀ · (v_j − v_i − g · Δt)`, and
//! `R_iᵀ · (p_j − p_i − v_i · Δt − 0.5 · g · Δt²)` respectively).
//!
//! Bias state `(b_a, b_g)` is held at the linearisation point passed to
//! [`ImuPreintegrator::new_with_bias`]; the integrator additionally
//! propagates the first-order Jacobians of `(ΔR, Δv, Δp)` with respect
//! to that linearisation point (Forster eq. 35-39). At residual time,
//! [`ImuPreintegrationFactor::residual_with_bias_correction`] applies
//! the first-order correction `(δb_g, δb_a) = b_now − b_lin` instead of
//! requiring a full re-integration when the bias estimate moves.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use visloc_core::geometry::SO3;

/// Output of pre-integrating a window of IMU samples between two keyframes.
/// Quantities are in keyframe-`i`'s body frame and gravity-free; the
/// BA-side residual combines them with the world-frame gravity vector and
/// the global `(R, v, p)` of both keyframes.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuPreintegratedDelta {
    /// Relative rotation `R_iᵀ · R_j` on SO(3).
    pub delta_rotation: SO3,
    /// Relative velocity `R_iᵀ · (v_j − v_i − g · Δt_{ij})` in `i`'s body frame.
    pub delta_velocity: Vector3<f64>,
    /// Relative position `R_iᵀ · (p_j − p_i − v_i · Δt_{ij} − 0.5 · g · Δt_{ij}²)`.
    pub delta_position: Vector3<f64>,
    /// Total elapsed time `Δt_{ij}` between the two keyframes.
    pub delta_time: f64,
    /// Linearisation point of the gyro bias used during integration. The
    /// bias-correction term in
    /// [`ImuPreintegrationFactor::residual_with_bias_correction`] is
    /// `δb_g = b_g_now − bias_gyro_linearisation`.
    pub bias_gyro_linearisation: Vector3<f64>,
    /// Linearisation point of the accelerometer bias used during integration.
    pub bias_acc_linearisation: Vector3<f64>,
    /// `∂(Log ΔR) / ∂b_g` accumulated through propagation (Forster eq. 35).
    pub j_rotation_bg: Matrix3<f64>,
    /// `∂Δv / ∂b_a` (Forster eq. 36).
    pub j_velocity_ba: Matrix3<f64>,
    /// `∂Δv / ∂b_g` (Forster eq. 37).
    pub j_velocity_bg: Matrix3<f64>,
    /// `∂Δp / ∂b_a` (Forster eq. 38).
    pub j_position_ba: Matrix3<f64>,
    /// `∂Δp / ∂b_g` (Forster eq. 39).
    pub j_position_bg: Matrix3<f64>,
}

impl ImuPreintegratedDelta {
    /// The zero delta: identity rotation, zero translation / velocity, dt=0,
    /// zero bias linearisation, and zero bias Jacobians.
    pub fn identity() -> Self {
        Self {
            delta_rotation: SO3::identity(),
            delta_velocity: Vector3::zeros(),
            delta_position: Vector3::zeros(),
            delta_time: 0.0,
            bias_gyro_linearisation: Vector3::zeros(),
            bias_acc_linearisation: Vector3::zeros(),
            j_rotation_bg: Matrix3::zeros(),
            j_velocity_ba: Matrix3::zeros(),
            j_velocity_bg: Matrix3::zeros(),
            j_position_ba: Matrix3::zeros(),
            j_position_bg: Matrix3::zeros(),
        }
    }

    /// Apply the first-order bias correction (Forster eq. 44 / sec. C):
    /// returns `(ΔR_corr, Δv_corr, Δp_corr)` at the bias point
    /// `(bias_gyro, bias_acc)`, taking `δb = bias_now − bias_linearisation`
    /// against the stored Jacobians. Suitable for residual evaluation
    /// without re-running the integrator.
    pub fn corrected(
        &self,
        bias_gyro: &Vector3<f64>,
        bias_acc: &Vector3<f64>,
    ) -> (SO3, Vector3<f64>, Vector3<f64>) {
        let delta_bg = bias_gyro - self.bias_gyro_linearisation;
        let delta_ba = bias_acc - self.bias_acc_linearisation;
        let phi_corr: Vector3<f64> = self.j_rotation_bg * delta_bg;
        let rotation_corr = UnitQuaternion::from_scaled_axis(phi_corr);
        let delta_rot = SO3::from_quaternion(self.delta_rotation.quaternion() * rotation_corr);
        let delta_vel =
            self.delta_velocity + self.j_velocity_ba * delta_ba + self.j_velocity_bg * delta_bg;
        let delta_pos =
            self.delta_position + self.j_position_ba * delta_ba + self.j_position_bg * delta_bg;
        (delta_rot, delta_vel, delta_pos)
    }

    /// Re-linearise the stored `(ΔR, Δv, Δp)` and bias linearisation
    /// point at the new bias estimate. Bakes the first-order bias
    /// correction `(δb = new_bias − old_linearisation)` into the deltas
    /// and resets `bias_*_linearisation = new_bias`, so subsequent
    /// `residual_with_bias_correction` evaluations at biases near
    /// `new_bias` see `δb ≈ 0` and stay inside the first-order regime.
    ///
    /// Jacobians (`j_*`) are preserved unchanged — this is the standard
    /// ORB-SLAM3 trick: avoid the O(N) re-integration cost by accepting
    /// that the bias-Jacobians don't move much for small bias shifts.
    /// Use periodically (e.g. on VI-init promotion when the bias jumps
    /// from a placeholder zero, or when the BA-refined bias drifts more
    /// than a configured threshold away from the stored linearisation
    /// point); for very large jumps a full re-integration would be more
    /// accurate.
    pub fn relinearise_at(&mut self, new_bias_gyro: &Vector3<f64>, new_bias_acc: &Vector3<f64>) {
        let (delta_rot, delta_vel, delta_pos) = self.corrected(new_bias_gyro, new_bias_acc);
        self.delta_rotation = delta_rot;
        self.delta_velocity = delta_vel;
        self.delta_position = delta_pos;
        self.bias_gyro_linearisation = *new_bias_gyro;
        self.bias_acc_linearisation = *new_bias_acc;
    }
}

/// Accumulator that ingests body-frame IMU samples one at a time and
/// maintains a running [`ImuPreintegratedDelta`]. Construct with
/// [`ImuPreintegrator::new`] (no bias) or
/// [`ImuPreintegrator::new_with_bias`] when the IMU-side bias estimate is
/// known; in either case, the accumulator subtracts the supplied bias from
/// every gyro / accel sample before integrating, and accumulates the
/// first-order bias-Jacobians of the running delta (Forster eq. 35-39).
#[derive(Debug, Clone, PartialEq)]
pub struct ImuPreintegrator {
    delta: ImuPreintegratedDelta,
    bias_gyro: Vector3<f64>,
    bias_acc: Vector3<f64>,
}

impl ImuPreintegrator {
    /// Start a fresh integrator with zero gyro and accelerometer bias. The
    /// resulting [`ImuPreintegratedDelta`] is the identity delta until at
    /// least one sample is folded in via [`Self::integrate_sample`].
    pub fn new() -> Self {
        Self::new_with_bias(Vector3::zeros(), Vector3::zeros())
    }

    /// Start a fresh integrator with non-zero biases. The biases are the
    /// linearisation point used for the first-order bias-Jacobian
    /// propagation stored on the running [`ImuPreintegratedDelta`].
    pub fn new_with_bias(bias_gyro: Vector3<f64>, bias_acc: Vector3<f64>) -> Self {
        let mut delta = ImuPreintegratedDelta::identity();
        delta.bias_gyro_linearisation = bias_gyro;
        delta.bias_acc_linearisation = bias_acc;
        Self {
            delta,
            bias_gyro,
            bias_acc,
        }
    }

    /// Fold one body-frame IMU sample into the running delta. `dt` is the
    /// time elapsed since the previous sample (or since the start of the
    /// integration window for the first sample). Sign convention follows
    /// Forster 2017: `gyro` and `accel` are measured in the body frame at
    /// the current orientation, gravity is NOT pre-subtracted from `accel`
    /// (the BA-side residual handles gravity separately).
    pub fn integrate_sample(&mut self, gyro: Vector3<f64>, accel: Vector3<f64>, dt: f64) {
        debug_assert!(dt > 0.0, "dt must be positive, got {dt}");
        let omega = gyro - self.bias_gyro;
        let alpha = accel - self.bias_acc;

        // Capture pre-update state for Forster eq. 35-39.
        let rot_pre_mat: Matrix3<f64> = self.delta.delta_rotation.matrix();
        let j_v_a_pre = self.delta.j_velocity_ba;
        let j_v_g_pre = self.delta.j_velocity_bg;
        let j_r_pre = self.delta.j_rotation_bg;
        let alpha_skew = skew(&alpha);

        // First update Δp and Δv using the rotation BEFORE this step.
        let alpha_in_i: Vector3<f64> = rot_pre_mat * alpha;
        self.delta.delta_position += self.delta.delta_velocity * dt + 0.5 * alpha_in_i * dt * dt;
        self.delta.delta_velocity += alpha_in_i * dt;

        // Position bias-Jacobians (Forster eq. 38-39) — read J_v BEFORE
        // its update at this step.
        let half_dt_sq = 0.5 * dt * dt;
        self.delta.j_position_ba += j_v_a_pre * dt - rot_pre_mat * half_dt_sq;
        self.delta.j_position_bg +=
            j_v_g_pre * dt - rot_pre_mat * alpha_skew * j_r_pre * half_dt_sq;

        // Velocity bias-Jacobians (Forster eq. 36-37).
        self.delta.j_velocity_ba = j_v_a_pre - rot_pre_mat * dt;
        self.delta.j_velocity_bg = j_v_g_pre - rot_pre_mat * alpha_skew * j_r_pre * dt;

        // Then advance the rotation by exp(ω · dt).
        let omega_dt = omega * dt;
        let delta_q = UnitQuaternion::from_scaled_axis(omega_dt);
        let new_q = self.delta.delta_rotation.quaternion() * delta_q;
        self.delta.delta_rotation = SO3::from_quaternion(new_q);

        // Rotation bias-Jacobian (Forster eq. 35):
        //   J_R^{k+1} = exp(ω·dt)^T · J_R^k − Jr(ω·dt) · dt
        let step_rot_mat: Matrix3<f64> = nalgebra::Rotation3::from_scaled_axis(omega_dt).into();
        let jr_omega_dt = right_jacobian_so3(&omega_dt);
        self.delta.j_rotation_bg = step_rot_mat.transpose() * j_r_pre - jr_omega_dt * dt;

        self.delta.delta_time += dt;
    }

    /// Snapshot the current accumulated delta. Idempotent — does not reset
    /// or modify internal state, so the integrator can keep ingesting more
    /// samples after the snapshot.
    pub fn delta(&self) -> ImuPreintegratedDelta {
        self.delta.clone()
    }

    /// Reset the accumulator back to the identity delta (preserving the
    /// configured bias linearisation point). Useful when reusing an
    /// integrator across many inter-keyframe windows in a streaming loop.
    pub fn reset(&mut self) {
        self.delta = ImuPreintegratedDelta::identity();
        self.delta.bias_gyro_linearisation = self.bias_gyro;
        self.delta.bias_acc_linearisation = self.bias_acc;
    }
}

impl Default for ImuPreintegrator {
    fn default() -> Self {
        Self::new()
    }
}

/// BA-side pre-integration factor connecting two keyframes with the
/// gravity-compensated `(ΔR, Δv, Δp)` produced by [`ImuPreintegrator`].
/// The optional bias-correction path
/// ([`Self::residual_with_bias_correction`]) lets a BA-side solver shift
/// the integrator's linearisation point without re-running the pre-
/// integration, using the first-order Jacobians stored on
/// [`ImuPreintegratedDelta`].
#[derive(Debug, Clone, PartialEq)]
pub struct ImuPreintegrationFactor {
    /// "From" keyframe id (the integration window's left endpoint).
    pub keyframe_id_from: u64,
    /// "To" keyframe id (the integration window's right endpoint).
    pub keyframe_id_to: u64,
    /// Pre-integrated relative measurement.
    pub delta: ImuPreintegratedDelta,
    /// World-frame gravity vector. KITTI y-down: `(0, 9.81, 0)`.
    pub gravity_world: Vector3<f64>,
    /// 3-vector position residual weight `1/σ_p²`.
    pub weight_position: f64,
    /// 3-vector velocity residual weight `1/σ_v²`.
    pub weight_velocity: f64,
    /// 3-vector rotation residual weight `1/σ_R²` (on the SO(3) log).
    pub weight_rotation: f64,
}

impl ImuPreintegrationFactor {
    /// Compute the gravity-compensated `(r_R, r_v, r_p)` residual triplet
    /// against world-frame state `(R_i, p_i, v_i)` and `(R_j, p_j, v_j)`,
    /// returning the 9-vector stacked as `[r_R; r_v; r_p]`. Sign and
    /// ordering follow Forster 2017 eq. 45-47:
    ///
    /// - `r_R = log(ΔR.transpose() · R_iᵀ · R_j)`
    /// - `r_v = R_iᵀ · (v_j − v_i − g · Δt) − Δv`
    /// - `r_p = R_iᵀ · (p_j − p_i − v_i · Δt − 0.5 · g · Δt²) − Δp`
    ///
    /// Bias is *not* applied here: the residual uses the pre-integrated
    /// delta as-is. Pass [`Self::residual_with_bias_correction`] the
    /// current bias estimate to get the bias-corrected residual instead.
    pub fn residual(
        &self,
        r_i: &SO3,
        p_i: &Vector3<f64>,
        v_i: &Vector3<f64>,
        r_j: &SO3,
        p_j: &Vector3<f64>,
        v_j: &Vector3<f64>,
    ) -> [Vector3<f64>; 3] {
        self.residual_corrected_internal(
            r_i,
            p_i,
            v_i,
            r_j,
            p_j,
            v_j,
            &self.delta.delta_rotation,
            &self.delta.delta_velocity,
            &self.delta.delta_position,
        )
    }

    /// Like [`Self::residual`] but applies a first-order bias correction:
    /// `δb = bias_now − delta.bias_*_linearisation` against the
    /// `j_*` Jacobians stored on `self.delta`. Falls back to the un-
    /// corrected residual when both biases equal the linearisation
    /// point. Used by the BA-side IMU factor when biases are part of the
    /// optimisation state.
    pub fn residual_with_bias_correction(
        &self,
        r_i: &SO3,
        p_i: &Vector3<f64>,
        v_i: &Vector3<f64>,
        r_j: &SO3,
        p_j: &Vector3<f64>,
        v_j: &Vector3<f64>,
        bias_gyro: &Vector3<f64>,
        bias_acc: &Vector3<f64>,
    ) -> [Vector3<f64>; 3] {
        let (delta_rot, delta_vel, delta_pos) = self.delta.corrected(bias_gyro, bias_acc);
        self.residual_corrected_internal(
            r_i, p_i, v_i, r_j, p_j, v_j, &delta_rot, &delta_vel, &delta_pos,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn residual_corrected_internal(
        &self,
        r_i: &SO3,
        p_i: &Vector3<f64>,
        v_i: &Vector3<f64>,
        r_j: &SO3,
        p_j: &Vector3<f64>,
        v_j: &Vector3<f64>,
        delta_rot: &SO3,
        delta_vel: &Vector3<f64>,
        delta_pos: &Vector3<f64>,
    ) -> [Vector3<f64>; 3] {
        let dt = self.delta.delta_time;
        let g = self.gravity_world;

        let r_i_inv = r_i.inverse();
        let r_i_inv_mat: Matrix3<f64> = r_i_inv.matrix();
        let q_rel = delta_rot.quaternion().inverse() * r_i_inv.quaternion() * r_j.quaternion();
        let r_rot = q_rel.scaled_axis();

        let r_vel = r_i_inv_mat * (v_j - v_i - g * dt) - delta_vel;
        let r_pos = r_i_inv_mat * (p_j - p_i - v_i * dt - 0.5 * g * dt * dt) - delta_pos;

        [r_rot, r_vel, r_pos]
    }
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// SO(3) right Jacobian `Jr(φ) = I − (1−cos θ)/θ² · [φ]× + (θ−sin θ)/θ³ · [φ]×²`.
/// For small `θ`, expands to `I − ½[φ]× + (1/6)[φ]×²`.
fn right_jacobian_so3(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta_sq = phi.norm_squared();
    let phi_skew = skew(phi);
    if theta_sq < 1e-10 {
        Matrix3::identity() - 0.5 * phi_skew + (1.0 / 6.0) * (phi_skew * phi_skew)
    } else {
        let theta = theta_sq.sqrt();
        let c1 = (1.0 - theta.cos()) / theta_sq;
        let c2 = (theta - theta.sin()) / (theta * theta_sq);
        Matrix3::identity() - c1 * phi_skew + c2 * (phi_skew * phi_skew)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn zero_motion_yields_identity_delta() {
        let mut pre = ImuPreintegrator::new();
        for _ in 0..10 {
            pre.integrate_sample(Vector3::zeros(), Vector3::zeros(), 0.01);
        }
        let d = pre.delta();
        assert!((d.delta_velocity).norm() < 1.0e-12);
        assert!((d.delta_position).norm() < 1.0e-12);
        let angle = d
            .delta_rotation
            .quaternion()
            .angle_to(&UnitQuaternion::identity());
        assert!(
            angle < 1.0e-12,
            "expected identity rotation, got angle {angle}"
        );
        assert!(approx_eq(d.delta_time, 0.10, 1.0e-12));
    }

    #[test]
    fn constant_linear_acceleration_matches_closed_form() {
        // Pure x-direction body-frame acceleration of 2 m/s² for 1 second
        // → Δv = (2, 0, 0), Δp = (1, 0, 0).
        let mut pre = ImuPreintegrator::new();
        let dt = 0.001;
        let steps = 1000;
        let acc = Vector3::new(2.0, 0.0, 0.0);
        for _ in 0..steps {
            pre.integrate_sample(Vector3::zeros(), acc, dt);
        }
        let d = pre.delta();
        assert!(
            (d.delta_velocity - Vector3::new(2.0, 0.0, 0.0)).norm() < 1.0e-9,
            "Δv = {:?}",
            d.delta_velocity
        );
        assert!(
            (d.delta_position - Vector3::new(1.0, 0.0, 0.0)).norm() < 1.0e-6,
            "Δp = {:?}",
            d.delta_position
        );
        let angle = d
            .delta_rotation
            .quaternion()
            .angle_to(&UnitQuaternion::identity());
        assert!(angle < 1.0e-12);
    }

    #[test]
    fn constant_yaw_rate_matches_closed_form_rotation() {
        // ω_y = π/2 rad/s for 1 s → ΔR is +π/2 yaw, Δv / Δp stay zero.
        let mut pre = ImuPreintegrator::new();
        let dt = 0.001;
        let steps = 1000;
        let gyro = Vector3::new(0.0, FRAC_PI_2, 0.0);
        for _ in 0..steps {
            pre.integrate_sample(gyro, Vector3::zeros(), dt);
        }
        let d = pre.delta();
        let expected = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), FRAC_PI_2);
        let dq = d.delta_rotation.quaternion().rotation_to(&expected);
        assert!(dq.angle() < 1.0e-6, "ΔR off by {} rad", dq.angle());
        assert!(d.delta_velocity.norm() < 1.0e-12);
        assert!(d.delta_position.norm() < 1.0e-12);
    }

    #[test]
    fn bias_subtraction_removes_constant_bias() {
        // Inject a constant +1 m/s² accelerometer bias; cancel it via
        // `new_with_bias`. The result must look like zero acceleration.
        let bias_acc = Vector3::new(1.0, 0.0, 0.0);
        let mut pre = ImuPreintegrator::new_with_bias(Vector3::zeros(), bias_acc);
        let dt = 0.01;
        for _ in 0..100 {
            pre.integrate_sample(Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0), dt);
        }
        let d = pre.delta();
        assert!(d.delta_velocity.norm() < 1.0e-12);
        assert!(d.delta_position.norm() < 1.0e-12);
    }

    #[test]
    fn rotated_acceleration_rotates_with_body_frame() {
        // First half of the window: rotate +π/2 yaw at constant rate.
        // Second half: apply 1 m/s² in body-frame +x. After the rotation,
        // body-frame +x points along world-frame +z (with KITTI y-down,
        // right-handed). The integrated Δv direction must reflect that.
        let mut pre = ImuPreintegrator::new();
        let dt = 0.001;
        // Rotate over 1 second to π/2 rad.
        let gyro = Vector3::new(0.0, FRAC_PI_2, 0.0);
        for _ in 0..1000 {
            pre.integrate_sample(gyro, Vector3::zeros(), dt);
        }
        // Then push +x for 1 second at 1 m/s².
        for _ in 0..1000 {
            pre.integrate_sample(Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0), dt);
        }
        let d = pre.delta();
        // Body-frame +x after π/2 yaw around y points along world -z in a
        // right-handed frame (rotation matrix carries +x → -z direction).
        // The integrated Δv must therefore lie on the world z axis.
        assert!(d.delta_velocity.x.abs() < 1.0e-2);
        assert!(d.delta_velocity.y.abs() < 1.0e-9);
        // |Δv| ≈ a · Δt_acc = 1 · 1 = 1 m/s.
        assert!(
            (d.delta_velocity.norm() - 1.0).abs() < 1.0e-2,
            "|Δv| = {}",
            d.delta_velocity.norm()
        );
    }

    #[test]
    fn residual_zero_at_consistent_state() {
        // Build a synthetic 1-second window where the body undergoes a
        // constant +x acceleration (no rotation, no gravity in this test).
        // Construct the matching (R_i, p_i, v_i), (R_j, p_j, v_j) by hand
        // and check that the factor residual is the zero 9-vector.
        let mut pre = ImuPreintegrator::new();
        let dt = 0.001;
        let acc = Vector3::new(2.0, 0.0, 0.0);
        for _ in 0..1000 {
            pre.integrate_sample(Vector3::zeros(), acc, dt);
        }
        let delta = pre.delta();
        let factor = ImuPreintegrationFactor {
            keyframe_id_from: 0,
            keyframe_id_to: 1,
            delta,
            gravity_world: Vector3::zeros(),
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        };

        let r_i = SO3::identity();
        let r_j = SO3::identity();
        let p_i = Vector3::new(0.0, 0.0, 0.0);
        let p_j = Vector3::new(1.0, 0.0, 0.0);
        let v_i = Vector3::zeros();
        let v_j = Vector3::new(2.0, 0.0, 0.0);

        let [r_rot, r_vel, r_pos] = factor.residual(&r_i, &p_i, &v_i, &r_j, &p_j, &v_j);
        assert!(r_rot.norm() < 1.0e-9, "r_R = {:?}", r_rot);
        assert!(r_vel.norm() < 1.0e-9, "r_v = {:?}", r_vel);
        assert!(r_pos.norm() < 1.0e-6, "r_p = {:?}", r_pos);
    }

    #[test]
    fn residual_handles_gravity_compensation() {
        // No body-frame acceleration; gravity in world = +y · 9.81. After
        // 1 second, world `v_j = -g · 1` and `p_j = -0.5 · g`. Pre-integrated
        // delta is identity (no accel, no rotation), so the residual must
        // still vanish thanks to the gravity-compensation terms in
        // `r_v` and `r_p`.
        let pre = ImuPreintegrator::new();
        // Pretend we ingested 1 second worth of zero samples without bothering
        // to step the accumulator; just hand-build the dt=1 delta.
        let mut delta = pre.delta();
        delta.delta_time = 1.0;

        let factor = ImuPreintegrationFactor {
            keyframe_id_from: 0,
            keyframe_id_to: 1,
            delta,
            gravity_world: Vector3::new(0.0, 9.81, 0.0),
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        };

        let r_i = SO3::identity();
        let r_j = SO3::identity();
        let p_i = Vector3::zeros();
        let v_i = Vector3::zeros();
        let v_j = Vector3::new(0.0, 9.81, 0.0); // accumulated under gravity
        let p_j = Vector3::new(0.0, 0.5 * 9.81, 0.0);

        let [r_rot, r_vel, r_pos] = factor.residual(&r_i, &p_i, &v_i, &r_j, &p_j, &v_j);
        assert!(r_rot.norm() < 1.0e-9);
        assert!(r_vel.norm() < 1.0e-9, "r_v = {:?}", r_vel);
        assert!(r_pos.norm() < 1.0e-9, "r_p = {:?}", r_pos);
    }

    #[test]
    fn bias_jacobians_match_finite_difference() {
        // Integrate a non-trivial body-frame motion (gyro + accel both
        // non-zero) at a linearisation bias point, then again at a small
        // perturbation `δb`. The first-order corrected delta must agree
        // with the directly re-integrated delta to within `|δb|²` (the
        // second-order remainder of the Taylor expansion).
        let dt = 0.005;
        let steps = 200; // 1 s window
        let gyro_raw = Vector3::new(0.10, 0.05, -0.03);
        let accel_raw = Vector3::new(0.50, -0.20, 9.81);

        let b_g_lin = Vector3::new(0.010, -0.020, 0.005);
        let b_a_lin = Vector3::new(0.050, -0.100, 0.020);

        let mut pre_lin = ImuPreintegrator::new_with_bias(b_g_lin, b_a_lin);
        for _ in 0..steps {
            pre_lin.integrate_sample(gyro_raw, accel_raw, dt);
        }
        let delta_lin = pre_lin.delta();

        // Small perturbation: |δb| ~ 1e-3, so |δb|² ~ 1e-6.
        let db_g = Vector3::new(0.001, 0.0005, -0.0008);
        let db_a = Vector3::new(0.005, -0.002, 0.003);

        let mut pre_pert = ImuPreintegrator::new_with_bias(b_g_lin + db_g, b_a_lin + db_a);
        for _ in 0..steps {
            pre_pert.integrate_sample(gyro_raw, accel_raw, dt);
        }
        let delta_pert = pre_pert.delta();

        let (rot_corr, vel_corr, pos_corr) =
            delta_lin.corrected(&(b_g_lin + db_g), &(b_a_lin + db_a));

        // First-order accuracy: the gap should be O(|δb|²) ≈ 1e-5. We
        // allow 1e-4 (10× headroom) to absorb integrator higher-order
        // coupling and the 200-step integration error itself.
        let err_v = (vel_corr - delta_pert.delta_velocity).norm();
        let err_p = (pos_corr - delta_pert.delta_position).norm();
        let rot_err = rot_corr
            .quaternion()
            .rotation_to(delta_pert.delta_rotation.quaternion())
            .angle();
        assert!(err_v < 1.0e-4, "Δv err = {}", err_v);
        assert!(err_p < 1.0e-4, "Δp err = {}", err_p);
        assert!(rot_err < 1.0e-4, "ΔR err = {} rad", rot_err);

        // And the uncorrected delta (i.e. naive use of the linearisation
        // delta at the perturbed bias) should be visibly worse, so the
        // correction is actually pulling its weight.
        let raw_err_v = (delta_lin.delta_velocity - delta_pert.delta_velocity).norm();
        let raw_err_p = (delta_lin.delta_position - delta_pert.delta_position).norm();
        assert!(
            raw_err_v > err_v * 10.0,
            "raw {} should be much worse than corrected {}",
            raw_err_v,
            err_v
        );
        assert!(
            raw_err_p > err_p * 10.0,
            "raw {} should be much worse than corrected {}",
            raw_err_p,
            err_p
        );
    }

    #[test]
    fn residual_with_bias_correction_matches_residual_at_linearisation_point() {
        // When `bias_now == bias_linearisation`, the corrected residual
        // and the uncorrected residual must be bit-identical.
        let mut pre = ImuPreintegrator::new_with_bias(
            Vector3::new(0.01, -0.02, 0.005),
            Vector3::new(0.05, -0.10, 0.02),
        );
        for _ in 0..200 {
            pre.integrate_sample(
                Vector3::new(0.1, 0.05, -0.03),
                Vector3::new(0.5, -0.2, 9.81),
                0.005,
            );
        }
        let delta = pre.delta();
        let factor = ImuPreintegrationFactor {
            keyframe_id_from: 0,
            keyframe_id_to: 1,
            delta: delta.clone(),
            gravity_world: Vector3::new(0.0, 9.81, 0.0),
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        };

        let r_i = SO3::identity();
        let r_j = SO3::identity();
        let p_i = Vector3::zeros();
        let v_i = Vector3::zeros();
        let p_j = Vector3::new(0.4, 4.7, -0.05);
        let v_j = Vector3::new(0.6, 9.6, -0.1);

        let a = factor.residual(&r_i, &p_i, &v_i, &r_j, &p_j, &v_j);
        let b = factor.residual_with_bias_correction(
            &r_i,
            &p_i,
            &v_i,
            &r_j,
            &p_j,
            &v_j,
            &delta.bias_gyro_linearisation,
            &delta.bias_acc_linearisation,
        );
        for k in 0..3 {
            assert!((a[k] - b[k]).norm() < 1.0e-12);
        }
    }

    #[test]
    fn relinearise_at_updates_linearisation_point_and_bakes_correction() {
        // Build a non-trivial delta at bias linearisation `b_lin = 0`, then
        // re-linearise at a new bias `b_new`. After re-linearisation the
        // stored `bias_*_linearisation` must equal `b_new`, and evaluating
        // `corrected(b_new, b_new)` must reproduce the original
        // `corrected(b_new, b_new)` exactly: the deltas are just bookkeeping.
        let mut pre = ImuPreintegrator::new();
        for _ in 0..200 {
            pre.integrate_sample(
                Vector3::new(0.05, -0.02, 0.01),
                Vector3::new(0.3, -0.1, 9.7),
                0.005,
            );
        }
        let delta_at_zero = pre.delta();
        let b_new_gyro = Vector3::new(0.012, -0.008, 0.005);
        let b_new_acc = Vector3::new(0.04, -0.02, 0.01);

        // Reference: deltas evaluated by the existing `corrected()` at b_new
        // *from the zero linearisation*. This is the value any downstream
        // residual sees today.
        let (ref_rot, ref_vel, ref_pos) = delta_at_zero.corrected(&b_new_gyro, &b_new_acc);

        // Re-linearise in-place. New evaluation at the SAME b_new must be
        // identity in the small-δb limit (`δb = b_new - b_new = 0`).
        let mut relinearised = delta_at_zero.clone();
        relinearised.relinearise_at(&b_new_gyro, &b_new_acc);

        assert!(
            (relinearised.bias_gyro_linearisation - b_new_gyro).norm() < 1.0e-15,
            "linearisation point must move to the new gyro bias",
        );
        assert!(
            (relinearised.bias_acc_linearisation - b_new_acc).norm() < 1.0e-15,
            "linearisation point must move to the new accel bias",
        );

        // The baked deltas must match what `corrected()` produced from the
        // old linearisation — the trick is identity at the re-linearisation
        // boundary, by construction.
        let q_baked = relinearised.delta_rotation.quaternion();
        let q_ref = ref_rot.quaternion();
        let q_diff = q_ref.inverse() * q_baked;
        let angle_diff = q_diff.scaled_axis().norm();
        assert!(
            angle_diff < 1.0e-12,
            "baked ΔR must match reference within 1e-12 rad, got {angle_diff}",
        );
        assert!((relinearised.delta_velocity - ref_vel).norm() < 1.0e-12);
        assert!((relinearised.delta_position - ref_pos).norm() < 1.0e-12);

        // Evaluating corrected() at the new linearisation point gives the
        // baked deltas verbatim (δb = 0).
        let (back_rot, back_vel, back_pos) = relinearised.corrected(&b_new_gyro, &b_new_acc);
        let q_back = back_rot.quaternion();
        let q_back_diff = q_baked.inverse() * q_back;
        assert!(q_back_diff.scaled_axis().norm() < 1.0e-15);
        assert!((back_vel - relinearised.delta_velocity).norm() < 1.0e-15);
        assert!((back_pos - relinearised.delta_position).norm() < 1.0e-15);

        // Jacobians are preserved unchanged — first-order assumption.
        assert_eq!(relinearised.j_rotation_bg, delta_at_zero.j_rotation_bg);
        assert_eq!(relinearised.j_velocity_ba, delta_at_zero.j_velocity_ba);
        assert_eq!(relinearised.j_velocity_bg, delta_at_zero.j_velocity_bg);
        assert_eq!(relinearised.j_position_ba, delta_at_zero.j_position_ba);
        assert_eq!(relinearised.j_position_bg, delta_at_zero.j_position_bg);
    }

    #[test]
    fn relinearise_at_zero_delta_is_no_op_on_identity() {
        // A factor that was constructed at b_lin = (0, 0, 0) and is being
        // re-linearised at (0, 0, 0) again must be a perfect no-op — the
        // deltas, linearisation point, and Jacobians all stay put.
        let mut pre = ImuPreintegrator::new();
        for _ in 0..50 {
            pre.integrate_sample(
                Vector3::new(0.02, 0.01, -0.005),
                Vector3::new(0.1, -0.05, 9.85),
                0.01,
            );
        }
        let before = pre.delta();
        let mut after = before.clone();
        after.relinearise_at(&Vector3::zeros(), &Vector3::zeros());
        let q_before = before.delta_rotation.quaternion();
        let q_after = after.delta_rotation.quaternion();
        let q_diff = q_before.inverse() * q_after;
        assert!(q_diff.scaled_axis().norm() < 1.0e-15);
        assert!((after.delta_velocity - before.delta_velocity).norm() < 1.0e-15);
        assert!((after.delta_position - before.delta_position).norm() < 1.0e-15);
        assert_eq!(
            after.bias_gyro_linearisation,
            before.bias_gyro_linearisation
        );
        assert_eq!(after.bias_acc_linearisation, before.bias_acc_linearisation);
    }
}
