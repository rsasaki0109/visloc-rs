use nalgebra::{Matrix3, Matrix4, Point3, SVector, UnitQuaternion, Vector3};

/// A similarity transform in 3D: `Sim(3) = (s, R, t)`, acting on a point as
/// `S · p = s · R · p + t`.
///
/// This is the 7-DOF group used to correct **scale drift** in monocular SLAM:
/// the extra scale degree of freedom lets a loop closure rescale an entire
/// trajectory segment, which a rigid `SE3` cannot.
#[derive(Debug, Clone, PartialEq)]
pub struct Sim3 {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
    /// Positive scale factor `s`.
    pub scale: f64,
}

/// Tangent vector `ξ = [ρ; ω; σ]` of `Sim(3)`: translation part `ρ` (0..3),
/// rotation part `ω` (3..6), and log-scale `σ` (6).
pub type Sim3Tangent = SVector<f64, 7>;

impl Sim3 {
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
            scale: 1.0,
        }
    }

    pub fn new(rotation: UnitQuaternion<f64>, translation: Vector3<f64>, scale: f64) -> Self {
        Self {
            rotation,
            translation,
            scale,
        }
    }

    /// `S · p = s · R · p + t`.
    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        Point3::from(self.scale * self.rotation.transform_point(point).coords + self.translation)
    }

    /// `(S1 ∘ S2)(p) = S1(S2(p))`, i.e.
    /// `(s1 s2, R1 R2, s1 R1 t2 + t1)`.
    pub fn compose(&self, other: &Sim3) -> Self {
        Self {
            rotation: self.rotation * other.rotation,
            translation: self.scale * self.rotation.transform_vector(&other.translation)
                + self.translation,
            scale: self.scale * other.scale,
        }
    }

    /// `S⁻¹ = (1/s, R⁻¹, -(1/s) R⁻¹ t)`.
    pub fn inverse(&self) -> Self {
        let inverse_scale = 1.0 / self.scale;
        let inverse_rotation = self.rotation.inverse();
        Self {
            rotation: inverse_rotation,
            translation: -inverse_scale * inverse_rotation.transform_vector(&self.translation),
            scale: inverse_scale,
        }
    }

    /// Homogeneous matrix `[[sR, t], [0, 1]]`.
    pub fn matrix(&self) -> Matrix4<f64> {
        let mut matrix = Matrix4::identity();
        matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(self.scale * self.rotation.to_rotation_matrix().into_inner()));
        matrix
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&self.translation);
        matrix
    }

    /// `Sim(3)` exponential map. Tangent layout is `[ρ; ω; σ]`.
    pub fn exp(tangent: &Sim3Tangent) -> Self {
        let rho = Vector3::new(tangent[0], tangent[1], tangent[2]);
        let omega = Vector3::new(tangent[3], tangent[4], tangent[5]);
        let sigma = tangent[6];
        let rotation = UnitQuaternion::from_scaled_axis(omega);
        let scale = sigma.exp();
        let w = sim3_w_matrix(&omega, sigma);
        Self {
            rotation,
            translation: w * rho,
            scale,
        }
    }

    /// `Sim(3)` logarithm map. Returns `ξ = [ρ; ω; σ]` such that
    /// `exp(ξ) = self`.
    pub fn log(&self) -> Sim3Tangent {
        let omega = self.rotation.scaled_axis();
        let sigma = self.scale.ln();
        let w = sim3_w_matrix(&omega, sigma);
        let rho = w
            .try_inverse()
            .map(|w_inv| w_inv * self.translation)
            .unwrap_or(self.translation);
        let mut tangent = Sim3Tangent::zeros();
        tangent[0] = rho.x;
        tangent[1] = rho.y;
        tangent[2] = rho.z;
        tangent[3] = omega.x;
        tangent[4] = omega.y;
        tangent[5] = omega.z;
        tangent[6] = sigma;
        tangent
    }
}

impl Default for Sim3 {
    fn default() -> Self {
        Self::identity()
    }
}

/// Skew-symmetric matrix `[v]×` such that `[v]× w = v × w`.
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// `(e^σ − 1) / σ`, the scale-only translation coefficient, evaluated with a
/// series near `σ = 0` to avoid cancellation.
fn expm1_over(sigma: f64) -> f64 {
    if sigma.abs() < 1.0e-4 {
        1.0 + sigma / 2.0 + sigma * sigma / 6.0 + sigma * sigma * sigma / 24.0
    } else {
        (sigma.exp() - 1.0) / sigma
    }
}

/// The `Sim(3)` translation Jacobian
/// `W(ω, σ) = ∫₀¹ e^{τσ} · exp(τ[ω]×) dτ = A·I + B·[ω]× + C·[ω]×²`,
/// the similarity-group analogue of the `SE(3)` left Jacobian. With `σ = 0` it
/// reduces to the `SE(3)` left Jacobian; with `ω = 0` it reduces to
/// `(e^σ − 1)/σ · I`.
fn sim3_w_matrix(omega: &Vector3<f64>, sigma: f64) -> Matrix3<f64> {
    let theta_sq = omega.norm_squared();
    let omega_skew = skew(omega);
    let a = expm1_over(sigma);

    let (b, c) = if theta_sq < 1.0e-10 {
        // ω → 0: the trig coefficients reduce to their no-rotation limits
        // B0(σ) = ∫₀¹ τ e^{τσ} dτ, C0(σ) = ∫₀¹ (τ²/2) e^{τσ} dτ.
        if sigma.abs() < 1.0e-4 {
            (
                0.5 + sigma / 3.0 + sigma * sigma / 8.0,
                1.0 / 6.0 + sigma / 8.0 + sigma * sigma / 20.0,
            )
        } else {
            let s = sigma.exp();
            let sigma_sq = sigma * sigma;
            (
                (s * (sigma - 1.0) + 1.0) / sigma_sq,
                (s * (sigma_sq - 2.0 * sigma + 2.0) - 2.0) / (2.0 * sigma_sq * sigma),
            )
        }
    } else {
        // General closed forms of the integrals; `denom = σ² + θ²` is safely
        // non-zero whenever θ is non-zero, so no σ branch is needed here.
        let theta = theta_sq.sqrt();
        let s = sigma.exp();
        let denom = sigma * sigma + theta_sq;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let b = (s * (sigma * sin_theta - theta * cos_theta) + theta) / (theta * denom);
        let c = (a - (s * (sigma * cos_theta + theta * sin_theta) - sigma) / denom) / theta_sq;
        (b, c)
    };

    Matrix3::identity() * a + b * omega_skew + c * omega_skew * omega_skew
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sim3(omega: Vector3<f64>, translation: Vector3<f64>, scale: f64) -> Sim3 {
        Sim3::new(UnitQuaternion::from_scaled_axis(omega), translation, scale)
    }

    fn assert_sim3_close(a: &Sim3, b: &Sim3, eps: f64) {
        let dt = (a.translation - b.translation).norm();
        let dr = (a.rotation.to_rotation_matrix().into_inner()
            - b.rotation.to_rotation_matrix().into_inner())
        .norm();
        let ds = (a.scale - b.scale).abs();
        assert!(dt < eps, "translation diff {dt} >= {eps}");
        assert!(dr < eps, "rotation diff {dr} >= {eps}");
        assert!(ds < eps, "scale diff {ds} >= {eps}");
    }

    #[test]
    fn transform_point_applies_scale_rotation_translation() {
        let s = make_sim3(
            Vector3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            Vector3::new(1.0, 2.0, 3.0),
            2.0,
        );
        // 90° about z maps (1,0,0) -> (0,1,0); scale 2 -> (0,2,0); +t -> (1,4,3).
        let p = s.transform_point(&Point3::new(1.0, 0.0, 0.0));
        assert!((p - Point3::new(1.0, 4.0, 3.0)).norm() < 1e-12, "{p:?}");
    }

    #[test]
    fn compose_matches_matrix_product() {
        let a = make_sim3(
            Vector3::new(0.2, -0.1, 0.3),
            Vector3::new(0.5, -0.2, 0.7),
            1.5,
        );
        let b = make_sim3(
            Vector3::new(-0.3, 0.4, 0.1),
            Vector3::new(-1.0, 0.3, 0.2),
            0.7,
        );
        let composed = a.compose(&b);
        let matrix_product = a.matrix() * b.matrix();
        assert!((composed.matrix() - matrix_product).norm() < 1e-12);
    }

    #[test]
    fn inverse_composes_to_identity() {
        let s = make_sim3(
            Vector3::new(0.4, -0.3, 0.2),
            Vector3::new(1.0, 0.5, -0.7),
            2.5,
        );
        assert_sim3_close(&s.compose(&s.inverse()), &Sim3::identity(), 1e-12);
        assert_sim3_close(&s.inverse().compose(&s), &Sim3::identity(), 1e-12);
    }

    #[test]
    fn log_of_identity_is_zero() {
        assert!(Sim3::identity().log().norm() < 1e-12);
    }

    #[test]
    fn exp_log_round_trip_general() {
        let s = make_sim3(
            Vector3::new(0.3, -0.2, 0.5),
            Vector3::new(1.0, -0.4, 2.1),
            1.8,
        );
        assert_sim3_close(&Sim3::exp(&s.log()), &s, 1e-10);
    }

    #[test]
    fn exp_log_round_trip_pure_scale() {
        let s = make_sim3(Vector3::zeros(), Vector3::new(1.0, -0.4, 2.1), 3.0);
        assert_sim3_close(&Sim3::exp(&s.log()), &s, 1e-10);
    }

    #[test]
    fn exp_log_round_trip_no_scale_matches_se3_regime() {
        // σ = 0 (scale 1): W reduces to the SE(3) left Jacobian.
        let s = make_sim3(
            Vector3::new(0.6, -0.1, 0.2),
            Vector3::new(-0.5, 0.9, 0.3),
            1.0,
        );
        assert_sim3_close(&Sim3::exp(&s.log()), &s, 1e-10);
    }

    #[test]
    fn exp_log_round_trip_small_rotation() {
        let s = make_sim3(
            Vector3::new(1e-7, -2e-7, 3e-7),
            Vector3::new(0.01, -0.02, 0.03),
            1.4,
        );
        assert_sim3_close(&Sim3::exp(&s.log()), &s, 1e-10);
    }

    #[test]
    fn exp_log_round_trip_small_rotation_and_small_scale() {
        let s = make_sim3(
            Vector3::new(1e-7, -2e-7, 3e-7),
            Vector3::new(0.01, -0.02, 0.03),
            (1e-6_f64).exp(),
        );
        assert_sim3_close(&Sim3::exp(&s.log()), &s, 1e-10);
    }

    #[test]
    fn log_exp_round_trip_tangent() {
        let mut xi = Sim3Tangent::zeros();
        xi[0] = 0.5;
        xi[1] = -0.3;
        xi[2] = 0.8;
        xi[3] = 0.1;
        xi[4] = -0.2;
        xi[5] = 0.4;
        xi[6] = 0.25;
        let xi2 = Sim3::exp(&xi).log();
        assert!((xi - xi2).norm() < 1e-10, "{:?}", xi - xi2);
    }

    #[test]
    fn exp_of_zero_is_identity() {
        assert_sim3_close(&Sim3::exp(&Sim3Tangent::zeros()), &Sim3::identity(), 1e-12);
    }
}
