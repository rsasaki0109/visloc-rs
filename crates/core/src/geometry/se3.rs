use nalgebra::{Matrix3, Matrix4, Matrix6, Point3, UnitQuaternion, Vector3, Vector6};

#[derive(Debug, Clone, PartialEq)]
pub struct SE3 {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
}

impl SE3 {
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }

    pub fn new(rotation: UnitQuaternion<f64>, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        Point3::from(self.rotation.transform_point(point).coords + self.translation)
    }

    pub fn transform_vector(&self, vector: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.transform_vector(vector)
    }

    pub fn compose(&self, other: &SE3) -> Self {
        Self::new(
            self.rotation * other.rotation,
            self.rotation.transform_vector(&other.translation) + self.translation,
        )
    }

    pub fn inverse(&self) -> Self {
        let rotation_inv = self.rotation.inverse();
        let translation_inv = -(rotation_inv.transform_vector(&self.translation));
        Self::new(rotation_inv, translation_inv)
    }

    pub fn matrix(&self) -> Matrix4<f64> {
        let mut matrix = Matrix4::identity();
        matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation.to_rotation_matrix().into_inner());
        matrix
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&self.translation);
        matrix
    }

    /// SE(3) logarithm map. Returns ξ = [ρ; ω] ∈ R^6 such that `exp(ξ) = self`.
    pub fn log(&self) -> Vector6<f64> {
        let omega = self.rotation.scaled_axis();
        let v_inv = so3_left_jacobian_inverse(&omega);
        let rho = v_inv * self.translation;
        let mut tangent = Vector6::zeros();
        tangent[0] = rho.x;
        tangent[1] = rho.y;
        tangent[2] = rho.z;
        tangent[3] = omega.x;
        tangent[4] = omega.y;
        tangent[5] = omega.z;
        tangent
    }

    /// SE(3) exponential map. Tangent layout is `[ρ; ω]` (translation first, then rotation).
    pub fn exp(tangent: &Vector6<f64>) -> Self {
        let rho = Vector3::new(tangent[0], tangent[1], tangent[2]);
        let omega = Vector3::new(tangent[3], tangent[4], tangent[5]);
        let rotation = UnitQuaternion::from_scaled_axis(omega);
        let v = so3_left_jacobian(&omega);
        let translation = v * rho;
        Self {
            rotation,
            translation,
        }
    }

    /// SE(3) adjoint matrix: `Ad(T) = [[R, [t]× R], [0, R]]`.
    /// For ξ = [ρ; ω], `T * exp(ξ̂) * T⁻¹ = exp((Ad(T) ξ)^∧)` to first order in ξ.
    pub fn adjoint(&self) -> Matrix6<f64> {
        let r = self.rotation.to_rotation_matrix().into_inner();
        let t_skew = skew(&self.translation);
        let mut ad = Matrix6::zeros();
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        ad.fixed_view_mut::<3, 3>(0, 3).copy_from(&(t_skew * r));
        ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        ad
    }
}

impl Default for SE3 {
    fn default() -> Self {
        Self::identity()
    }
}

/// Skew-symmetric matrix [v]× such that [v]× w = v × w.
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// SO(3) left Jacobian: V(ω) = I + (1−cos θ)/θ² [ω]× + (θ−sin θ)/θ³ [ω]×².
pub fn so3_left_jacobian(omega: &Vector3<f64>) -> Matrix3<f64> {
    let theta_sq = omega.norm_squared();
    let omega_skew = skew(omega);
    if theta_sq < 1e-10 {
        Matrix3::identity() + 0.5 * omega_skew + (1.0 / 6.0) * omega_skew * omega_skew
    } else {
        let theta = theta_sq.sqrt();
        let a = (1.0 - theta.cos()) / theta_sq;
        let b = (theta - theta.sin()) / (theta * theta_sq);
        Matrix3::identity() + a * omega_skew + b * omega_skew * omega_skew
    }
}

/// Inverse of the SO(3) left Jacobian: V⁻¹(ω) = I − ½[ω]× + (1/θ²)(1 − (θ/2) cot(θ/2)) [ω]×².
pub fn so3_left_jacobian_inverse(omega: &Vector3<f64>) -> Matrix3<f64> {
    let theta_sq = omega.norm_squared();
    let omega_skew = skew(omega);
    if theta_sq < 1e-10 {
        Matrix3::identity() - 0.5 * omega_skew + (1.0 / 12.0) * omega_skew * omega_skew
    } else {
        let theta = theta_sq.sqrt();
        let half_theta = 0.5 * theta;
        let c = (1.0 - half_theta * half_theta.cos() / half_theta.sin()) / theta_sq;
        Matrix3::identity() - 0.5 * omega_skew + c * omega_skew * omega_skew
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_se3(omega: Vector3<f64>, translation: Vector3<f64>) -> SE3 {
        SE3::new(UnitQuaternion::from_scaled_axis(omega), translation)
    }

    fn assert_se3_close(a: &SE3, b: &SE3, eps: f64) {
        let dt = (a.translation - b.translation).norm();
        let r1 = a.rotation.to_rotation_matrix().into_inner();
        let r2 = b.rotation.to_rotation_matrix().into_inner();
        let dr = (r1 - r2).norm();
        assert!(dt < eps, "translation diff {dt} >= {eps}");
        assert!(dr < eps, "rotation diff {dr} >= {eps}");
    }

    #[test]
    fn log_of_identity_is_zero() {
        let xi = SE3::identity().log();
        assert!(xi.norm() < 1e-12);
    }

    #[test]
    fn exp_log_round_trip_general() {
        let t = make_se3(Vector3::new(0.3, -0.2, 0.5), Vector3::new(1.0, -0.4, 2.1));
        let t2 = SE3::exp(&t.log());
        assert_se3_close(&t, &t2, 1e-10);
    }

    #[test]
    fn exp_log_round_trip_small_angle() {
        let t = make_se3(
            Vector3::new(1e-7, -2e-7, 3e-7),
            Vector3::new(0.01, -0.02, 0.03),
        );
        let t2 = SE3::exp(&t.log());
        assert_se3_close(&t, &t2, 1e-12);
    }

    #[test]
    fn log_exp_round_trip_tangent() {
        let mut xi = Vector6::zeros();
        xi[0] = 0.5;
        xi[1] = -0.3;
        xi[2] = 0.8;
        xi[3] = 0.1;
        xi[4] = -0.2;
        xi[5] = 0.4;
        let t = SE3::exp(&xi);
        let xi2 = t.log();
        assert!((xi - xi2).norm() < 1e-10);
    }

    #[test]
    fn adjoint_consistency_with_conjugation() {
        // T * exp(ξ̂) * T⁻¹ ≈ exp((Ad(T) ξ)^∧).
        let t = make_se3(Vector3::new(0.2, -0.1, 0.3), Vector3::new(0.5, -0.2, 0.7));
        let mut xi = Vector6::zeros();
        xi[0] = 1e-3;
        xi[1] = -2e-3;
        xi[2] = 3e-3;
        xi[3] = 5e-4;
        xi[4] = -1e-3;
        xi[5] = 2e-3;
        let lhs = t.compose(&SE3::exp(&xi)).compose(&t.inverse());
        let rhs = SE3::exp(&(t.adjoint() * xi));
        assert_se3_close(&lhs, &rhs, 1e-7);
    }

    #[test]
    fn adjoint_inverse_relation() {
        let t = make_se3(Vector3::new(0.4, -0.3, 0.2), Vector3::new(1.0, 0.5, -0.7));
        let prod = t.adjoint() * t.inverse().adjoint();
        assert!((prod - Matrix6::identity()).norm() < 1e-10);
    }
}
