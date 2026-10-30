use nalgebra::{Matrix3, SMatrix, UnitQuaternion, Vector3};

pub type Matrix9 = SMatrix<f64, 9, 9>;
type Matrix9x3 = SMatrix<f64, 9, 3>;

#[derive(Debug, Clone, PartialEq)]
pub struct ImuDStateUpdateTrace {
    pub t_ns: i64,
    pub f: Matrix9,
    pub g: Matrix9x3,
    pub f_old_bg: Matrix9x3,
    pub d_state_d_bg: Matrix9x3,
    pub d_state_d_ba: Matrix9x3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuNoiseModel {
    pub gyro_density: f64,
    pub accel_density: f64,
}

impl ImuNoiseModel {
    pub fn is_valid(self) -> bool {
        self.gyro_density.is_finite()
            && self.gyro_density >= 0.0
            && self.accel_density.is_finite()
            && self.accel_density >= 0.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImuPreintegratedDelta {
    pub delta_rotation: UnitQuaternion<f64>,
    pub delta_velocity: Vector3<f64>,
    pub delta_position: Vector3<f64>,
    pub delta_time: f64,
    pub bias_gyro: Vector3<f64>,
    pub bias_accel: Vector3<f64>,
    pub jacobian_rotation_gyro_bias: Matrix3<f64>,
    pub jacobian_velocity_gyro_bias: Matrix3<f64>,
    pub jacobian_velocity_accel_bias: Matrix3<f64>,
    pub jacobian_position_gyro_bias: Matrix3<f64>,
    pub jacobian_position_accel_bias: Matrix3<f64>,
    pub covariance: Matrix9,
    pub update_trace: Vec<ImuDStateUpdateTrace>,
}

impl ImuPreintegratedDelta {
    pub fn identity(bias_gyro: Vector3<f64>, bias_accel: Vector3<f64>) -> Self {
        Self {
            delta_rotation: UnitQuaternion::identity(),
            delta_velocity: Vector3::zeros(),
            delta_position: Vector3::zeros(),
            delta_time: 0.0,
            bias_gyro,
            bias_accel,
            jacobian_rotation_gyro_bias: Matrix3::zeros(),
            jacobian_velocity_gyro_bias: Matrix3::zeros(),
            jacobian_velocity_accel_bias: Matrix3::zeros(),
            jacobian_position_gyro_bias: Matrix3::zeros(),
            jacobian_position_accel_bias: Matrix3::zeros(),
            covariance: Matrix9::zeros(),
            update_trace: Vec::new(),
        }
    }

    /// First-order correction around the stored bias linearisation point.
    pub fn corrected(
        &self,
        gyro_bias: Vector3<f64>,
        accel_bias: Vector3<f64>,
    ) -> (UnitQuaternion<f64>, Vector3<f64>, Vector3<f64>) {
        let dr = UnitQuaternion::from_scaled_axis(
            self.jacobian_rotation_gyro_bias * (gyro_bias - self.bias_gyro),
        );
        (
            // IntegratedImuMeasurement stores the orientation bias
            // derivative in the same left-perturbation convention as
            // PoseVelState::applyInc: exp(dtheta) * R.  Keeping this on the
            // left is important once the accumulated delta is no longer
            // close to identity.
            dr * self.delta_rotation,
            self.delta_velocity
                + self.jacobian_velocity_gyro_bias * (gyro_bias - self.bias_gyro)
                + self.jacobian_velocity_accel_bias * (accel_bias - self.bias_accel),
            self.delta_position
                + self.jacobian_position_gyro_bias * (gyro_bias - self.bias_gyro)
                + self.jacobian_position_accel_bias * (accel_bias - self.bias_accel),
        )
    }
}

#[derive(Debug, Clone)]
pub struct ImuPreintegrator {
    delta: ImuPreintegratedDelta,
    noise: Option<ImuNoiseModel>,
    /// The noise densities are continuous and each step gets variance for
    /// its own length (see `EstimatorConfig::imu_noise_per_step`).
    per_step: bool,
}

impl ImuPreintegrator {
    pub fn new(bias_gyro: Vector3<f64>, bias_accel: Vector3<f64>) -> Self {
        Self {
            delta: ImuPreintegratedDelta::identity(bias_gyro, bias_accel),
            noise: None,
            per_step: false,
        }
    }
    /// Continuous noise densities, discretized by each step's length (see
    /// `EstimatorConfig::imu_noise_per_step`); `false` keeps the per-sample
    /// densities of the pinned model.
    #[must_use]
    pub const fn with_per_step_noise(mut self, per_step: bool) -> Self {
        self.per_step = per_step;
        self
    }
    pub fn with_noise(mut self, noise: ImuNoiseModel) -> Option<Self> {
        if noise.is_valid() {
            self.noise = Some(noise);
            Some(self)
        } else {
            None
        }
    }
    pub const fn delta(&self) -> &ImuPreintegratedDelta {
        &self.delta
    }

    /// Integrate one interval using a constant (midpoint/trapezoidal) sample.
    pub fn integrate_sample(&mut self, gyro: Vector3<f64>, accel: Vector3<f64>, dt: f64) {
        self.integrate_sample_at(gyro, accel, dt, None);
    }

    pub fn integrate_sample_at(
        &mut self,
        gyro: Vector3<f64>,
        accel: Vector3<f64>,
        dt: f64,
        t_ns: Option<i64>,
    ) {
        assert!(
            dt.is_finite() && dt > 0.0,
            "IMU dt must be finite and positive"
        );
        let omega = gyro - self.delta.bias_gyro;
        let specific_force = accel - self.delta.bias_accel;
        // This is a direct translation of IntegratedImuMeasurement::
        // propagateState.  Its state order is [position, rotation, velocity]
        // and its orientation perturbations are left-multiplicative.
        let r_half_q =
            self.delta.delta_rotation * UnitQuaternion::from_scaled_axis(omega * (0.5 * dt));
        let r_half = r_half_q.to_rotation_matrix().into_inner();
        let accel_world = r_half * specific_force;
        let old_v = self.delta.delta_velocity;
        self.delta.delta_position += old_v * dt + accel_world * (0.5 * dt * dt);
        self.delta.delta_velocity += accel_world * dt;
        let dtheta = omega * dt;
        self.delta.delta_rotation *= UnitQuaternion::from_scaled_axis(dtheta);
        self.delta.delta_time += dt;

        // The three matrices below are the exact F/A/G matrices from the
        // upstream header.  In particular, G uses SO3's right Jacobian for
        // the finite gyro increment; replacing it with +/-I*dt is only a
        // first-order approximation and produces visibly wrong bias blocks.
        let mut f = Matrix9::identity();
        f.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(Matrix3::identity() * dt));
        let f_rot = skew(-accel_world * dt);
        f.fixed_view_mut::<3, 3>(6, 3).copy_from(&f_rot);
        f.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(f_rot * (0.5 * dt)));

        let mut a = Matrix9x3::zeros();
        a.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(r_half * (0.5 * dt * dt)));
        a.fixed_view_mut::<3, 3>(6, 0).copy_from(&(r_half * dt));

        let mut g = Matrix9x3::zeros();
        let dtheta = omega * dt;
        let dtheta_half = omega * (0.5 * dt);
        let jr = right_jacobian_so3(dtheta);
        let jr2 = right_jacobian_so3(dtheta_half);
        let r_new = self.delta.delta_rotation.to_rotation_matrix().into_inner();
        g.fixed_view_mut::<3, 3>(3, 0).copy_from(&(r_new * jr * dt));
        g.fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(f_rot * r_half * jr2 * (0.5 * dt)));
        let g_position = g.fixed_view::<3, 3>(6, 0).into_owned() * (0.5 * dt);
        g.fixed_view_mut::<3, 3>(0, 0).copy_from(&g_position);

        // Sophus/Eigen and nalgebra use opposite matrix representatives for
        // the same quaternion tangent (nalgebra's `to_rotation_matrix()` is
        // the transpose of Sophus's).  Keep the state integration in the
        // crate's existing nalgebra convention, but transpose the internal
        // header Jacobian blocks at the API boundary.  This preserves the
        // upstream d_state_d_b* numeric contract without changing delta q.
        let old_d_ba = bias_jacobian(
            self.delta.jacobian_position_accel_bias.transpose(),
            self.delta.jacobian_velocity_accel_bias.transpose(),
            None,
        );
        let old_d_bg = bias_jacobian(
            self.delta.jacobian_position_gyro_bias.transpose(),
            self.delta.jacobian_velocity_gyro_bias.transpose(),
            Some(self.delta.jacobian_rotation_gyro_bias.transpose()),
        );
        let new_d_ba = -a + f * old_d_ba;
        let new_d_bg = -g + f * old_d_bg;
        self.delta.jacobian_position_accel_bias =
            new_d_ba.fixed_rows::<3>(0).into_owned().transpose();
        self.delta.jacobian_velocity_accel_bias =
            new_d_ba.fixed_rows::<3>(6).into_owned().transpose();
        self.delta.jacobian_position_gyro_bias =
            new_d_bg.fixed_rows::<3>(0).into_owned().transpose();
        self.delta.jacobian_rotation_gyro_bias =
            new_d_bg.fixed_rows::<3>(3).into_owned().transpose();
        self.delta.jacobian_velocity_gyro_bias =
            new_d_bg.fixed_rows::<3>(6).into_owned().transpose();

        if let Some(t_ns) = t_ns {
            self.delta.update_trace.push(ImuDStateUpdateTrace {
                t_ns,
                f,
                g,
                f_old_bg: f * old_d_bg,
                d_state_d_bg: new_d_bg,
                d_state_d_ba: new_d_ba,
            });
        }

        if let Some(n) = self.noise {
            // Keep the established covariance contract: bias recursion uses
            // the full F/A/G blocks above, while covariance injection follows
            // Basalt's legacy measurement-noise model.
            let mut cov_f = Matrix9::identity();
            cov_f
                .fixed_view_mut::<3, 3>(0, 6)
                .copy_from(&(Matrix3::identity() * dt));
            let rot_accel = -r_half * skew(specific_force);
            cov_f
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(rot_accel * dt));
            cov_f
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(rot_accel * (0.5 * dt * dt)));

            let mut accel_noise = Matrix9x3::zeros();
            accel_noise
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(r_half * (0.5 * dt * dt)));
            accel_noise
                .fixed_view_mut::<3, 3>(6, 0)
                .copy_from(&(r_half * dt));
            let mut gyro_noise = Matrix9x3::zeros();
            gyro_noise
                .fixed_view_mut::<3, 3>(3, 0)
                .copy_from(&(Matrix3::identity() * dt));
            let step = if self.per_step { 1.0 / dt } else { 1.0 };
            self.delta.covariance = cov_f * self.delta.covariance * cov_f.transpose()
                + (accel_noise * accel_noise.transpose())
                    * (n.accel_density * n.accel_density * step)
                + (gyro_noise * gyro_noise.transpose()) * (n.gyro_density * n.gyro_density * step);
            self.delta.covariance =
                (self.delta.covariance + self.delta.covariance.transpose()) * 0.5;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_noise_covariance_is_diagonal_and_uses_interval_dt() {
        let noise = ImuNoiseModel {
            gyro_density: 2.0,
            accel_density: 3.0,
        };
        let mut preintegrator = ImuPreintegrator::new(Vector3::zeros(), Vector3::zeros())
            .with_noise(noise)
            .unwrap();
        preintegrator.integrate_sample(Vector3::zeros(), Vector3::zeros(), 0.1);
        let covariance = &preintegrator.delta().covariance;
        assert!((covariance[(0, 0)] - 0.000225).abs() < 1e-12);
        assert!((covariance[(3, 3)] - 0.04).abs() < 1e-12);
        assert!((covariance[(6, 6)] - 0.09).abs() < 1e-12);
        assert!(covariance[(0, 1)].abs() < 1e-15);
        assert!(covariance[(3, 4)].abs() < 1e-15);
        assert!(covariance[(6, 7)].abs() < 1e-15);
    }

    #[test]
    fn upstream_float_frame1_2_bias_jacobians_golden() {
        // d_state_d_b* exported by IntegratedImuMeasurement<float> from the
        // pinned Basalt header (aa441...; ten MH_01 samples, frame 1 -> 2).
        // The Rust API exposes the same matrix in named [p,R,v] blocks.
        let gyros = [
            [
                -0.11789180338382721,
                -0.023220349103212357,
                -0.043173350393772125,
            ],
            [
                -0.11509927362203598,
                -0.03997550904750824,
                -0.05085279792547226,
            ],
            [
                -0.1081179603934288,
                -0.0637119859457016,
                -0.061324771493673325,
            ],
            [
                -0.10532543063163757,
                -0.08535407483577728,
                -0.07040048390626907,
            ],
            [
                -0.09624972194433212,
                -0.10210923105478287,
                -0.07598553597927094,
            ],
            [
                -0.08717400580644608,
                -0.11397746950387955,
                -0.07807993143796921,
            ],
            [
                -0.07600390166044235,
                -0.11956252157688141,
                -0.08017432689666748,
            ],
            [
                -0.06902258098125458,
                -0.1279401034116745,
                -0.07528740912675858,
            ],
            [
                -0.06343752890825272,
                -0.14190274477005005,
                -0.06690982729196548,
            ],
            [
                -0.060645006597042084,
                -0.1509784460067749,
                -0.058532245457172394,
            ],
        ];
        let accels = [
            [7.7339344024658203, -0.5776441693305969, -3.000910997390747],
            [7.840173244476318, -0.5694719552993774, -2.9191887378692627],
            [8.03630542755127, -0.5449553728103638, -2.8456389904022217],
            [8.13437271118164, -0.5204387307167053, -2.780261278152466],
            [8.25695514678955, -0.4877498745918274, -2.723055839538574],
            [8.355022430419922, -0.5367831587791443, -2.6903669834136963],
            [8.297816276550293, -0.4714055061340332, -2.6249892711639404],
            [8.216094017028809, -0.5040943026542664, -2.674022674560547],
            [8.248783111572266, -0.5286109447479248, -2.7394001483917236],
            [8.118027687072754, -0.5612998008728027, -2.8456389904022217],
        ];
        let times = [
            0.004999936,
            0.005000192,
            0.004999936,
            0.004999936,
            0.004999936,
            0.005000192,
            0.004999936,
            0.004999936,
            0.004999936,
            0.005000192,
        ];
        let mut integrator = ImuPreintegrator::new(Vector3::zeros(), Vector3::zeros());
        for i in 0..10 {
            integrator.integrate_sample(
                Vector3::from_row_slice(&gyros[i]),
                Vector3::from_row_slice(&accels[i]),
                times[i],
            );
        }
        let d = integrator.delta();
        let expected_ba = [
            -0.001250003930181265,
            -0.0000012298474985073,
            0.0000012757516287820,
            0.0000012261144775039,
            -0.001250002300366759,
            -0.0000022456572423835,
            -0.0000012790949313057,
            0.0000022433607682615,
            -0.001250001951120794,
        ];
        let expected_pbg = [
            -0.0000000465239544667,
            0.000057232231483795,
            -0.000011211098353670,
            -0.000057329965784447,
            0.0000001985938098414,
            -0.000171313280588947,
            0.000011169478057127,
            0.000171357940416783,
            0.0000002305991557705,
        ];
        let expected_rbg = [
            -0.0499999001622200,
            -0.000078299490269274,
            0.000092247035354376,
            0.000077972566941753,
            -0.0499998293817043,
            -0.000127114821225405,
            -0.000092511290858965,
            0.000126896542496979,
            -0.0499997846782207,
        ];
        let expected_vba = [
            -0.0499999001622200,
            -0.000078299082815647,
            0.000092247348220553,
            0.000077972988947295,
            -0.0499998368322849,
            -0.000127114471979439,
            -0.000092510992544703,
            0.000126896906294860,
            -0.0499997846782207,
        ];
        let expected_vbg = [
            -0.00000400684575652,
            0.00340186059474945,
            -0.000680765893775970,
            -0.00341101665981114,
            0.000015016544239188,
            -0.0102565130218863,
            0.000676828320138156,
            0.0102606182917953,
            0.000017642001694185,
        ];
        let check = |name: &str, actual: &Matrix3<f64>, expected: &[f64; 9]| {
            let mut max_error: f64 = 0.0;
            let mut max_row = 0;
            let mut max_col = 0;
            for row in 0..3 {
                for col in 0..3 {
                    // nalgebra's fixed matrices expose their column-major
                    // storage through iteration/indexed views; the golden
                    // constants above are written in Eigen's row-major
                    // print order.
                    let error = (actual[(row, col)] - expected[col * 3 + row]).abs();
                    if error > max_error {
                        max_error = error;
                        max_row = row;
                        max_col = col;
                    }
                }
            }
            assert!(
                max_error < 2.0e-7,
                "{name} max error {max_error:e} at ({max_row},{max_col}): actual={} expected={}",
                actual[(max_row, max_col)],
                expected[max_col * 3 + max_row]
            );
        };
        check("p_ba", &d.jacobian_position_accel_bias, &expected_ba);
        check("p_bg", &d.jacobian_position_gyro_bias, &expected_pbg);
        check("r_bg", &d.jacobian_rotation_gyro_bias, &expected_rbg);
        check("v_ba", &d.jacobian_velocity_accel_bias, &expected_vba);
        check("v_bg", &d.jacobian_velocity_gyro_bias, &expected_vbg);
    }
}

fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Matrix form of the upstream 9x3 bias state Jacobian.  Keeping this helper
/// private lets the public delta retain its stable, named 3x3 blocks while
/// still applying the exact `-A + F*d_state_d_b*` update.
fn bias_jacobian(
    position: Matrix3<f64>,
    velocity: Matrix3<f64>,
    rotation: Option<Matrix3<f64>>,
) -> Matrix9x3 {
    let mut result = Matrix9x3::zeros();
    result.fixed_view_mut::<3, 3>(0, 0).copy_from(&position);
    if let Some(rotation) = rotation {
        result.fixed_view_mut::<3, 3>(3, 0).copy_from(&rotation);
    }
    result.fixed_view_mut::<3, 3>(6, 0).copy_from(&velocity);
    result
}

/// Sophus-compatible right Jacobian for SO(3).
fn right_jacobian_so3(phi: Vector3<f64>) -> Matrix3<f64> {
    let phi_norm2 = phi.norm_squared();
    let phi_hat = skew(phi);
    let phi_hat2 = phi_hat * phi_hat;
    let mut result = Matrix3::identity();
    if phi_norm2 > f64::EPSILON {
        let phi_norm = phi_norm2.sqrt();
        let phi_norm3 = phi_norm2 * phi_norm;
        result -= phi_hat * ((1.0 - phi_norm.cos()) / phi_norm2);
        result += phi_hat2 * ((phi_norm - phi_norm.sin()) / phi_norm3);
    } else {
        result -= phi_hat * 0.5;
        result += phi_hat2 / 6.0;
    }
    result
}
