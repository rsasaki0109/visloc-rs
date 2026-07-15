//! Small, dependency-free finite-difference utilities shared by SLAM back ends.

use nalgebra::{Point2, SMatrix, SVector};

/// Evaluate a two-dimensional projection Jacobian with a central difference.
///
/// Returning `None` for an invalid step, a failed projection, or a non-finite
/// result keeps numerical failures explicit at the caller's geometry gate.
pub(crate) fn central_difference_projection_jacobian<const N: usize>(
    center: &SVector<f64, N>,
    step: f64,
    mut project: impl FnMut(&SVector<f64, N>) -> Option<Point2<f64>>,
) -> Option<SMatrix<f64, 2, N>> {
    if !step.is_finite() || step <= 0.0 || !center.iter().all(|value| value.is_finite()) {
        return None;
    }

    let mut jacobian = SMatrix::<f64, 2, N>::zeros();
    for axis in 0..N {
        let mut plus = *center;
        let mut minus = *center;
        plus[axis] += step;
        minus[axis] -= step;
        let projected_plus = project(&plus)?;
        let projected_minus = project(&minus)?;
        if !projected_plus.coords.iter().all(|value| value.is_finite())
            || !projected_minus.coords.iter().all(|value| value.is_finite())
        {
            return None;
        }
        jacobian[(0, axis)] = (projected_plus.x - projected_minus.x) / (2.0 * step);
        jacobian[(1, axis)] = (projected_plus.y - projected_minus.y) / (2.0 * step);
    }

    jacobian
        .iter()
        .all(|value| value.is_finite())
        .then_some(jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_linear_projection_jacobian() {
        let center = SVector::<f64, 3>::new(2.0, -1.0, 0.5);
        let jacobian = central_difference_projection_jacobian(&center, 1.0e-5, |value| {
            Some(Point2::new(
                2.0 * value[0] - value[1] + 0.5 * value[2],
                -3.0 * value[0] + 4.0 * value[2],
            ))
        })
        .expect("linear projection should be differentiable");

        let expected = SMatrix::<f64, 2, 3>::from_row_slice(&[2.0, -1.0, 0.5, -3.0, 0.0, 4.0]);
        assert!((jacobian - expected).norm() < 1.0e-9);
    }

    #[test]
    fn rejects_invalid_step_and_failed_projection() {
        let center = SVector::<f64, 2>::zeros();
        assert!(central_difference_projection_jacobian(&center, 0.0, |_| {
            Some(Point2::origin())
        })
        .is_none());
        assert!(central_difference_projection_jacobian(&center, 1.0e-5, |_| None).is_none());
    }

    #[test]
    fn rejects_non_finite_input_and_output() {
        let non_finite_center = SVector::<f64, 2>::new(f64::NAN, 0.0);
        assert!(
            central_difference_projection_jacobian(&non_finite_center, 1.0e-5, |_| Some(
                Point2::origin()
            ),)
            .is_none()
        );

        let center = SVector::<f64, 2>::zeros();
        assert!(
            central_difference_projection_jacobian(&center, 1.0e-5, |_| {
                Some(Point2::new(f64::INFINITY, 0.0))
            })
            .is_none()
        );
    }
}
