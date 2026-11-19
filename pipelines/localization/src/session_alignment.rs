//! Joint rigid alignment of gravity-aligned session maps.

use std::collections::VecDeque;
use std::fmt;

use nalgebra::{DMatrix, DVector, Matrix4, Vector3, Vector4};
use visloc_core::geometry::SE3;

use crate::odom_correction::{wrap_angle, yaw_of, Offset};

/// A measured transform from `other` to `base`, in their original frames.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionConstraint {
    pub base: usize,
    pub other: usize,
    pub base_from_other: SE3,
}

/// Scales of the translation and yaw residuals for the Huber loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SessionAlignmentConfig {
    pub translation_scale_m: f64,
    pub yaw_scale_rad: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionAlignment {
    /// Each original session's transform into session zero's frame.
    pub reference_from_session: Vec<SE3>,
    pub cost_before: f64,
    pub cost_after: f64,
    pub iterations: usize,
    /// Final residuals in the same order as the input constraints.
    pub residuals: Vec<SessionAlignmentResidual>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SessionAlignmentResidual {
    pub translation_m: f64,
    pub yaw_rad: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionAlignmentError {
    InvalidConfig,
    InvalidConstraint(usize),
    Disconnected(Vec<usize>),
    NumericalFailure,
}

impl fmt::Display for SessionAlignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => {
                write!(f, "expected sessions and positive finite residual scales")
            }
            Self::InvalidConstraint(index) => write!(f, "invalid session constraint {index}"),
            Self::Disconnected(ids) => {
                write!(f, "sessions {ids:?} are disconnected from session 0")
            }
            Self::NumericalFailure => {
                write!(f, "session alignment could not solve its normal equations")
            }
        }
    }
}

impl std::error::Error for SessionAlignmentError {}

/// Aligns a connected graph of metric, gravity-aligned sessions. Session zero
/// fixes the gauge. The other poses are initialized along a spanning tree and
/// jointly refined with one Huber-weighted translation/yaw residual per pair.
pub fn align_sessions(
    session_count: usize,
    constraints: &[SessionConstraint],
    config: SessionAlignmentConfig,
) -> Result<SessionAlignment, SessionAlignmentError> {
    let scales = Vector4::new(
        config.translation_scale_m,
        config.translation_scale_m,
        config.translation_scale_m,
        config.yaw_scale_rad,
    );
    if session_count == 0 || !scales.iter().all(|scale| scale.is_finite() && *scale > 0.0) {
        return Err(SessionAlignmentError::InvalidConfig);
    }

    let mut neighbors = vec![Vec::new(); session_count];
    for (index, edge) in constraints.iter().enumerate() {
        let transform = &edge.base_from_other;
        if edge.base >= session_count
            || edge.other >= session_count
            || edge.base == edge.other
            || !transform.translation.iter().all(|value| value.is_finite())
            || !transform
                .rotation
                .coords
                .iter()
                .all(|value| value.is_finite())
            || (transform.rotation * Vector3::z() - Vector3::z()).norm() > 1e-8
        {
            return Err(SessionAlignmentError::InvalidConstraint(index));
        }
        neighbors[edge.base].push((edge.other, transform.clone()));
        neighbors[edge.other].push((edge.base, transform.inverse()));
    }

    let mut initial = vec![None; session_count];
    initial[0] = Some(SE3::identity());
    let mut queue = VecDeque::from([0]);
    while let Some(index) = queue.pop_front() {
        let reference_from_current = initial[index].clone().expect("queued pose is initialized");
        for (neighbor, current_from_neighbor) in &neighbors[index] {
            if initial[*neighbor].is_none() {
                initial[*neighbor] = Some(reference_from_current.compose(current_from_neighbor));
                queue.push_back(*neighbor);
            }
        }
    }
    let disconnected = initial
        .iter()
        .enumerate()
        .filter_map(|(index, pose)| pose.is_none().then_some(index))
        .collect::<Vec<_>>();
    if !disconnected.is_empty() {
        return Err(SessionAlignmentError::Disconnected(disconnected));
    }
    let mut poses = initial
        .into_iter()
        .flatten()
        .map(|pose| Offset {
            yaw: yaw_of(&pose.rotation),
            translation: pose.translation,
        })
        .collect::<Vec<_>>();

    let objective = |poses: &[Offset]| {
        constraints
            .iter()
            .map(|edge| {
                let (_, residual) = edge_residual(poses, edge);
                let norm = residual.component_div(&scales).norm();
                if norm <= 1.0 {
                    0.5 * norm * norm
                } else {
                    norm - 0.5
                }
            })
            .sum::<f64>()
    };
    let cost_before = objective(&poses);
    if !cost_before.is_finite() {
        return Err(SessionAlignmentError::NumericalFailure);
    }
    let mut cost = cost_before;
    let mut iterations = 0;
    for _ in 0..40 {
        if cost <= 1e-24 {
            break;
        }
        let step = alignment_step(&poses, constraints, &scales)
            .ok_or(SessionAlignmentError::NumericalFailure)?;
        if step.norm() <= 1e-10 {
            break;
        }
        let mut accepted = false;
        let mut step_scale = 1.0;
        for _ in 0..12 {
            let mut trial = poses.clone();
            for (pose, delta) in trial[1..].iter_mut().zip(step.as_slice().chunks_exact(4)) {
                pose.translation += step_scale * Vector3::new(delta[0], delta[1], delta[2]);
                pose.yaw = wrap_angle(pose.yaw + step_scale * delta[3]);
            }
            let trial_cost = objective(&trial);
            if trial_cost < cost {
                poses = trial;
                cost = trial_cost;
                iterations += 1;
                accepted = true;
                break;
            }
            step_scale *= 0.5;
        }
        if !accepted {
            break;
        }
    }

    let residuals = constraints
        .iter()
        .map(|edge| {
            let (_, residual) = edge_residual(&poses, edge);
            SessionAlignmentResidual {
                translation_m: residual.fixed_rows::<3>(0).norm(),
                yaw_rad: residual[3].abs(),
                weight: huber_weight(residual.component_div(&scales).norm()),
            }
        })
        .collect();
    Ok(SessionAlignment {
        reference_from_session: poses.iter().map(Offset::se3).collect(),
        cost_before,
        cost_after: cost,
        iterations,
        residuals,
    })
}

fn edge_residual(poses: &[Offset], edge: &SessionConstraint) -> (Vector3<f64>, Vector4<f64>) {
    let base = poses[edge.base];
    let other = poses[edge.other];
    let translated = base.rotation() * edge.base_from_other.translation;
    let error = other.translation - base.translation - translated;
    let yaw = wrap_angle(other.yaw - base.yaw - yaw_of(&edge.base_from_other.rotation));
    (translated, Vector4::new(error.x, error.y, error.z, yaw))
}

fn huber_weight(norm: f64) -> f64 {
    1.0 / norm.max(1.0)
}

fn alignment_step(
    poses: &[Offset],
    constraints: &[SessionConstraint],
    scales: &Vector4<f64>,
) -> Option<DVector<f64>> {
    let dimensions = (poses.len() - 1) * 4;
    let mut hessian = DMatrix::<f64>::zeros(dimensions, dimensions);
    let mut rhs = DVector::<f64>::zeros(dimensions);
    let whiten = Matrix4::from_diagonal(&scales.map(f64::recip));
    for edge in constraints {
        let (translated, residual) = edge_residual(poses, edge);
        let residual = whiten * residual;
        let weight = huber_weight(residual.norm());
        // d(Rz * t)/d(yaw) = z cross (Rz * t).
        let mut base_jacobian = -whiten;
        base_jacobian[(0, 3)] = translated.y / scales[0];
        base_jacobian[(1, 3)] = -translated.x / scales[1];
        let blocks = [(edge.base, base_jacobian), (edge.other, whiten)];
        for (a, jac_a) in &blocks {
            if *a == 0 {
                continue;
            }
            let row = (a - 1) * 4;
            let jt = weight * jac_a.transpose();
            let mut rhs_block = rhs.fixed_rows_mut::<4>(row);
            rhs_block -= jt * residual;
            for (b, jac_b) in &blocks {
                if *b != 0 {
                    let mut block = hessian.fixed_view_mut::<4, 4>(row, (b - 1) * 4);
                    block += jt * jac_b;
                }
            }
        }
    }
    let step = hessian.cholesky()?.solve(&rhs);
    step.iter().all(|value| value.is_finite()).then_some(step)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: SessionAlignmentConfig = SessionAlignmentConfig {
        translation_scale_m: 0.25,
        yaw_scale_rad: 0.035,
    };

    fn pose(x: f64, y: f64, z: f64, yaw: f64) -> SE3 {
        Offset {
            yaw,
            translation: Vector3::new(x, y, z),
        }
        .se3()
    }

    fn edge(poses: &[SE3], base: usize, other: usize) -> SessionConstraint {
        SessionConstraint {
            base,
            other,
            base_from_other: poses[base].inverse().compose(&poses[other]),
        }
    }

    #[test]
    fn aligns_through_an_inverted_bridge_and_reports_disconnection() {
        let truth = [
            SE3::identity(),
            pose(2.0, 1.0, 0.4, 3.10),
            pose(3.5, 2.0, -0.2, -3.05),
        ];
        let first = edge(&truth, 0, 1);
        let result = align_sessions(3, &[first.clone(), edge(&truth, 2, 1)], CONFIG).unwrap();
        assert_eq!(result.reference_from_session[0], SE3::identity());
        for (actual, expected) in result.reference_from_session.iter().zip(&truth) {
            let error = expected.inverse().compose(actual);
            assert!(error.translation.norm() < 1e-10);
            assert!(error.rotation.angle() < 1e-10);
        }
        assert_eq!(
            align_sessions(3, &[first], CONFIG),
            Err(SessionAlignmentError::Disconnected(vec![2]))
        );
    }

    #[test]
    fn redundant_constraints_reduce_the_effect_of_a_bad_direct_registration() {
        let truth = [
            SE3::identity(),
            pose(1.0, 2.0, 0.4, 0.3),
            pose(4.0, 2.0, -0.1, -0.2),
            pose(-2.0, 3.0, 0.2, 0.5),
            pose(2.0, 5.0, 0.3, -0.5),
        ];
        let mut edges = (0..truth.len())
            .flat_map(|base| (base + 1..truth.len()).map(move |other| (base, other)))
            .map(|(base, other)| edge(&truth, base, other))
            .collect::<Vec<_>>();
        edges[0].base_from_other = pose(3.0, 4.0, 0.8, 0.9);
        let result = align_sessions(truth.len(), &edges, CONFIG).unwrap();
        assert!(result.cost_after < result.cost_before);
        assert!(result.residuals[0].weight < 0.1);
        let error = truth[1]
            .inverse()
            .compose(&result.reference_from_session[1]);
        assert!(error.translation.norm() < 0.2, "{error:?}");
        assert!(error.rotation.angle() < 0.03, "{error:?}");
        assert_eq!(result.reference_from_session[0], SE3::identity());
    }
}
