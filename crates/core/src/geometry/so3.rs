use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

/// Iteration cap for [`rotation_from_matrix`]. A converging projection takes
/// a handful of iterations; the cap only stops a cycle.
pub const ROTATION_FROM_MATRIX_MAX_ITERATIONS: usize = 1_000;

/// The rotation nearest to `matrix`: `UnitQuaternion::from_matrix(matrix)`,
/// without its hang.
///
/// nalgebra's `from_matrix` iterates Müller et al.'s projection from the
/// identity with no iteration limit. On some inputs it never stops: a
/// rotation of about 180° (KITTI 01's stereo VO hung on
/// `diag(-0.995, 0.995, -1)` from a bad RANSAC sample) sends it into its
/// "stuck in a maximum" branch, which cycles, and so does a non-finite
/// input. This runs the same iteration (same epsilon, same start) with a cap
/// and checks that the result is a fixed point, i.e. that one more iteration
/// leaves it unchanged, which is how the unbounded loop stops. A converged
/// result therefore has the same bits as `from_matrix`. Otherwise the
/// nearest rotation comes from the SVD (`U diag(1, 1, ±1) Vᵀ`), and for a
/// non-finite input the capped iterate (some rotation) is returned.
pub fn rotation_from_matrix(matrix: &Matrix3<f64>) -> UnitQuaternion<f64> {
    let capped = Rotation3::from_matrix_eps(
        matrix,
        f64::EPSILON,
        ROTATION_FROM_MATRIX_MAX_ITERATIONS,
        Rotation3::identity(),
    );
    let next = Rotation3::from_matrix_eps(matrix, f64::EPSILON, 1, capped);
    if next == capped {
        return capped.into();
    }
    nearest_rotation_svd(matrix).unwrap_or(capped).into()
}

fn nearest_rotation_svd(matrix: &Matrix3<f64>) -> Option<Rotation3<f64>> {
    if !matrix.iter().all(|value| value.is_finite()) {
        return None;
    }
    let svd = matrix.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut sign = Matrix3::identity();
    if (u * v_t).determinant() < 0.0 {
        sign[(2, 2)] = -1.0;
    }
    Some(Rotation3::from_matrix_unchecked(u * sign * v_t))
}

#[derive(Debug, Clone, PartialEq)]
pub struct SO3 {
    rotation: UnitQuaternion<f64>,
}

impl SO3 {
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
        }
    }

    pub const fn from_quaternion(rotation: UnitQuaternion<f64>) -> Self {
        Self { rotation }
    }

    pub fn from_matrix(matrix: &Matrix3<f64>) -> Self {
        Self {
            rotation: rotation_from_matrix(matrix),
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            rotation: self.rotation.inverse(),
        }
    }

    pub fn transform_vector(&self, vector: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.transform_vector(vector)
    }

    pub fn matrix(&self) -> Matrix3<f64> {
        self.rotation.to_rotation_matrix().into_inner()
    }

    pub const fn quaternion(&self) -> &UnitQuaternion<f64> {
        &self.rotation
    }
}

impl Default for SO3 {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_from_matrix_matches_nalgebra_wherever_it_converges() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        };
        for case in 0..200 {
            let rotation =
                UnitQuaternion::from_euler_angles(next() * 3.1, next() * 1.5, next() * 3.1)
                    .to_rotation_matrix()
                    .into_inner();
            // Exact rotations, and general matrices to be projected.
            let matrix = if case % 2 == 0 {
                rotation
            } else {
                rotation + Matrix3::from_fn(|_, _| 0.2 * next())
            };
            let expected = UnitQuaternion::from_matrix(&matrix);
            let actual = rotation_from_matrix(&matrix);
            assert_eq!(
                actual.coords.as_slice(),
                expected.coords.as_slice(),
                "case {case}"
            );
        }
    }

    #[test]
    fn rotation_from_matrix_returns_on_inputs_where_nalgebra_cycles() {
        // The half turn KITTI 01's stereo VO hung on (column-major).
        let half_turn = Matrix3::from_column_slice(&[
            -0.9954361201834596,
            0.0954302396208945,
            0.0,
            0.09543023962089417,
            0.9954361201834596,
            0.0,
            0.0,
            0.0,
            -1.0,
        ]);
        let rotation = rotation_from_matrix(&half_turn);
        assert!((rotation.to_rotation_matrix().into_inner() - half_turn).norm() < 1.0e-12);

        // A non-finite input returns some rotation instead of looping.
        let nan = Matrix3::from_element(f64::NAN);
        assert!(rotation_from_matrix(&nan)
            .coords
            .iter()
            .all(|v| v.is_finite()));
    }
}
