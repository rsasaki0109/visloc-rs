//! Factors from outside the VIO: absolute pose priors (e.g. localizations
//! against a prebuilt map) and body-velocity measurements (e.g. a wheeled
//! robot's wheel odometry).
//!
//! Not part of upstream Basalt. With no external factor queued, the window
//! builds exactly the factor list it always did, so the faithful-port
//! trajectory is unchanged. The factors enter the active window as generic
//! rows after the IMU factors; they are not carried into the marginal prior
//! when their state leaves the window (their information on the remaining
//! states has already moved the estimate, and the gauge they fix is the
//! map's, which the next prior re-fixes).

use super::*;
use nalgebra::{Matrix3, Vector6};

/// What one external factor measures.
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalMeasurement {
    /// The IMU's pose in the VIO world frame, with the square-root
    /// information of its translation (xyz) and rotation (left tangent xyz),
    /// and a Huber threshold on the whitened residual norm.
    PosePrior {
        world_from_imu: SE3,
        sqrt_information: Vector6<f64>,
        huber: f64,
    },
    /// The velocity of a body rigidly mounted on the IMU (`imu_from_body`),
    /// in that body's own frame, e.g. `(v, 0, 0)` for a differential-drive
    /// base: forward speed from the wheels, no sideways or vertical motion.
    /// `imu_angular_velocity` is the IMU's angular velocity (rad/s, IMU
    /// frame, bias-corrected), for the lever arm.
    BodyVelocity {
        imu_from_body: SE3,
        imu_angular_velocity: Vector3<f64>,
        body_velocity: Vector3<f64>,
        sqrt_information: Vector3<f64>,
    },
}

/// An external measurement queued on the estimator for one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalFactor {
    pub frame_id: u64,
    pub measurement: ExternalMeasurement,
}

/// An external measurement bound to a block of the active window
/// (poses first, then states, as in the AOM order).
#[derive(Debug, Clone, PartialEq)]
pub struct WindowExternalFactor {
    pub block_index: usize,
    pub measurement: ExternalMeasurement,
}

impl WindowProblem {
    fn external_block_offset(&self, block_index: usize) -> Option<(usize, bool)> {
        let layout = self.layout();
        if block_index < self.poses.len() {
            layout.offset_for_pose(block_index).map(|o| (o, false))
        } else {
            layout
                .offset_for_state(block_index - self.poses.len())
                .map(|o| (o, true))
        }
    }

    /// The external factors at the values `nav` gives each block, as
    /// generic whitened row stacks over the whole state width.
    pub(super) fn external_factors(
        &self,
        nav: impl Fn(usize) -> Option<BasaltNavState>,
    ) -> Vec<WhitenedFactorRowStack> {
        let state_dof = self.state_dof();
        let mut factors = Vec::with_capacity(self.external.len());
        for external in &self.external {
            let Some((offset, has_velocity)) = self.external_block_offset(external.block_index)
            else {
                continue;
            };
            let Some(current) = nav(external.block_index) else {
                continue;
            };
            let rotation = current
                .imu_to_world
                .rotation
                .to_rotation_matrix()
                .into_inner();
            match &external.measurement {
                ExternalMeasurement::PosePrior {
                    world_from_imu,
                    sqrt_information,
                    huber,
                } => {
                    let dt = current.imu_to_world.translation - world_from_imu.translation;
                    let dr = (current.imu_to_world.rotation * world_from_imu.rotation.inverse())
                        .scaled_axis();
                    let mut residual = DVector::zeros(6);
                    let mut jacobian = DMatrix::zeros(6, state_dof);
                    let jr = so3_left_jacobian_inverse(&dr);
                    for axis in 0..3 {
                        residual[axis] = sqrt_information[axis] * dt[axis];
                        jacobian[(axis, offset + axis)] = sqrt_information[axis];
                    }
                    for row in 0..3 {
                        residual[3 + row] = sqrt_information[3 + row] * dr[row];
                        for column in 0..3 {
                            jacobian[(3 + row, offset + 3 + column)] =
                                sqrt_information[3 + row] * jr[(row, column)];
                        }
                    }
                    let norm = residual.norm();
                    let (weight, cost) = if norm > *huber {
                        (*huber / norm, huber * (norm - 0.5 * huber))
                    } else {
                        (1.0, 0.5 * norm * norm)
                    };
                    let scale = weight.sqrt();
                    if let Some(factor) = WhitenedFactorRowStack::with_objective_cost(
                        jacobian * scale,
                        DMatrix::zeros(6, 0),
                        residual * scale,
                        cost,
                    ) {
                        factors.push(factor);
                    }
                }
                ExternalMeasurement::BodyVelocity {
                    imu_from_body,
                    imu_angular_velocity,
                    body_velocity,
                    sqrt_information,
                } => {
                    if !has_velocity {
                        continue;
                    }
                    let r_bi = imu_from_body
                        .rotation
                        .inverse()
                        .to_rotation_matrix()
                        .into_inner();
                    let v_world = current.velocity_world_m_s;
                    // Body velocity in the body frame:
                    // R_bi (R_wi^T v_w + w_i x t_ib).
                    let predicted = r_bi
                        * (rotation.transpose() * v_world
                            + imu_angular_velocity.cross(&imu_from_body.translation));
                    let error = predicted - body_velocity;
                    let d_velocity = r_bi * rotation.transpose();
                    // (exp(d) R)^T v = R^T v + R^T [v]x d.
                    let d_rotation = d_velocity * skew(&v_world);
                    let mut residual = DVector::zeros(3);
                    let mut jacobian = DMatrix::zeros(3, state_dof);
                    for row in 0..3 {
                        residual[row] = sqrt_information[row] * error[row];
                        for column in 0..3 {
                            jacobian[(row, offset + 3 + column)] =
                                sqrt_information[row] * d_rotation[(row, column)];
                            jacobian[(row, offset + 6 + column)] =
                                sqrt_information[row] * d_velocity[(row, column)];
                        }
                    }
                    if let Some(factor) =
                        WhitenedFactorRowStack::new(jacobian, DMatrix::zeros(3, 0), residual)
                    {
                        factors.push(factor);
                    }
                }
            }
        }
        factors
    }

    /// The objective of the external factors at the values `nav` gives.
    pub(super) fn external_cost(&self, nav: impl Fn(usize) -> Option<BasaltNavState>) -> f64 {
        if self.external.is_empty() {
            return 0.0;
        }
        self.external_factors(nav)
            .iter()
            .map(|factor| factor.objective_cost)
            .sum()
    }
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Inverse of the SO(3) left Jacobian.
fn so3_left_jacobian_inverse(omega: &Vector3<f64>) -> Matrix3<f64> {
    let theta_sq = omega.norm_squared();
    let omega_skew = skew(omega);
    if theta_sq < 1e-10 {
        return Matrix3::identity() - 0.5 * omega_skew + (1.0 / 12.0) * omega_skew * omega_skew;
    }
    let theta = theta_sq.sqrt();
    let half = 0.5 * theta;
    let c = (1.0 - half * half.cos() / half.sin()) / theta_sq;
    Matrix3::identity() - 0.5 * omega_skew + c * omega_skew * omega_skew
}
