//! Sim(3) pose-graph optimization for scale-drift correction.
//!
//! Monocular SLAM reconstructs a trajectory only up to an unknown, slowly
//! varying scale, so accumulated **scale drift** leaves a loop that should
//! close geometrically offset by a scale factor. A rigid `SE(3)` pose graph
//! cannot absorb that error — it has no scale degree of freedom — so the
//! standard fix (Strasdat et al., 2010) is to optimize a [`Sim3`] pose graph,
//! whose 7-DOF nodes let a loop closure redistribute the scale error across the
//! trajectory.
//!
//! This optimizer mirrors the `SE(3)` [`crate::PoseGraph`] loop (Levenberg-
//! Marquardt with one fixed anchor) but works in the `Sim(3)` tangent and uses
//! central-difference Jacobians, which are simple and robust for the modest
//! graphs a loop-closure scale correction produces. The dense normal equations
//! are solved with the same Cholesky/LU helper as the `SE(3)` path.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector, SMatrix};
use visloc_core::geometry::{Sim3, Sim3Tangent};

use crate::{solve_normal_equations, PoseGraphError};

/// Degrees of freedom of a `Sim(3)` node (`[ρ; ω; σ]`).
const DOF: usize = 7;

/// A `7×7` `Sim(3)` information matrix `Ω` (inverse measurement covariance).
pub type Sim3Information = SMatrix<f64, 7, 7>;

/// Edge in a [`Sim3PoseGraph`]: a measured relative similarity `measurement`
/// such that, at the solution, `pose_to ≈ measurement ∘ pose_from`.
///
/// `information`, when `Some`, is the full `7×7` Mahalanobis weight `Ω` and the
/// scalar `weight` is ignored; when `None`, the edge contributes the isotropic
/// cost `weight · ‖r‖²`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sim3Edge {
    pub from: u64,
    pub to: u64,
    pub measurement: Sim3,
    pub weight: f64,
    pub information: Option<Sim3Information>,
}

/// Per-iteration diagnostics for [`Sim3PoseGraph::optimize`].
#[derive(Debug, Clone, PartialEq)]
pub struct Sim3PoseGraphIterationStats {
    pub iteration: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub max_step_norm: f64,
    /// LM damping `λ` used for this iteration (`0.0` for pure Gauss-Newton).
    pub lambda: f64,
    pub step_accepted: bool,
}

/// Result of a full [`Sim3PoseGraph::optimize`] run.
#[derive(Debug, Clone, PartialEq)]
pub struct Sim3PoseGraphResult {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: Vec<Sim3PoseGraphIterationStats>,
    pub converged: bool,
}

/// Configuration for [`Sim3PoseGraph::optimize`].
#[derive(Debug, Clone, PartialEq)]
pub struct Sim3PoseGraphConfig {
    pub max_iterations: usize,
    /// Convergence threshold on the largest per-node 7-vector update.
    pub step_tolerance: f64,
    /// Convergence threshold on the absolute cost change between accepted steps.
    pub cost_tolerance: f64,
    /// Initial LM damping `λ`. `None` runs pure Gauss-Newton (every step is
    /// accepted unconditionally).
    pub initial_lambda: Option<f64>,
    pub lambda_increase_factor: f64,
    pub lambda_decrease_factor: f64,
    pub max_lambda: f64,
    pub min_lambda: f64,
}

impl Default for Sim3PoseGraphConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            step_tolerance: 1e-8,
            cost_tolerance: 1e-12,
            initial_lambda: Some(1e-3),
            lambda_increase_factor: 10.0,
            lambda_decrease_factor: 0.1,
            max_lambda: 1e12,
            min_lambda: 1e-9,
        }
    }
}

/// Sparse `Sim(3)` pose graph keyed by keyframe id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sim3PoseGraph {
    /// Keyframe id → `Sim(3)` node estimate. `BTreeMap` keeps the variable
    /// layout deterministic.
    pub poses: BTreeMap<u64, Sim3>,
    pub edges: Vec<Sim3Edge>,
    /// Anchor keyframe id; its node is held fixed (fixes the gauge freedom).
    pub anchor: Option<u64>,
}

impl Sim3PoseGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace the `Sim(3)` estimate for `keyframe_id`.
    pub fn add_pose(&mut self, keyframe_id: u64, pose: Sim3) {
        self.poses.insert(keyframe_id, pose);
    }

    /// Hold `keyframe_id` fixed during optimization.
    pub fn anchor(&mut self, keyframe_id: u64) {
        self.anchor = Some(keyframe_id);
    }

    /// Add an isotropic edge with the given scalar weight.
    pub fn add_edge(&mut self, from: u64, to: u64, measurement: Sim3, weight: f64) {
        self.edges.push(Sim3Edge {
            from,
            to,
            measurement,
            weight,
            information: None,
        });
    }

    /// Add an edge carrying a full `7×7` information matrix `Ω`.
    pub fn add_edge_with_information(
        &mut self,
        from: u64,
        to: u64,
        measurement: Sim3,
        information: Sim3Information,
    ) {
        self.edges.push(Sim3Edge {
            from,
            to,
            measurement,
            weight: 1.0,
            information: Some(information),
        });
    }

    /// Sum of (optionally Mahalanobis-weighted) squared edge residuals.
    pub fn cost(&self) -> f64 {
        self.edges
            .iter()
            .filter_map(|edge| {
                let pose_from = self.poses.get(&edge.from)?;
                let pose_to = self.poses.get(&edge.to)?;
                let r = edge_residual(&edge.measurement, pose_from, pose_to);
                Some(match &edge.information {
                    Some(omega) => (r.transpose() * omega * r)[(0, 0)],
                    None => edge.weight * r.norm_squared(),
                })
            })
            .sum()
    }

    /// Optimize all non-anchor nodes with Levenberg-Marquardt, correcting the
    /// accumulated scale (and pose) drift so the graph's loop closures are
    /// satisfied in the `Sim(3)` sense.
    pub fn optimize(
        &mut self,
        config: &Sim3PoseGraphConfig,
    ) -> Result<Sim3PoseGraphResult, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if !self.poses.contains_key(&anchor_id) {
            return Err(PoseGraphError::MissingNode(anchor_id));
        }
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }
        for edge in &self.edges {
            if !self.poses.contains_key(&edge.from) {
                return Err(PoseGraphError::MissingNode(edge.from));
            }
            if !self.poses.contains_key(&edge.to) {
                return Err(PoseGraphError::MissingNode(edge.to));
            }
        }

        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if id == anchor_id {
                continue;
            }
            let next = node_index.len();
            node_index.insert(id, next);
        }
        let variable_count = node_index.len();
        if variable_count == 0 {
            return Err(PoseGraphError::NoVariables);
        }

        let dim = variable_count * DOF;
        let initial_cost = self.cost();
        let mut current_cost = initial_cost;
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let mut iterations = Vec::with_capacity(config.max_iterations);
        let mut converged = false;

        for iteration in 0..config.max_iterations {
            let mut h = DMatrix::<f64>::zeros(dim, dim);
            let mut g = DVector::<f64>::zeros(dim);

            for edge in &self.edges {
                let pose_from = &self.poses[&edge.from];
                let pose_to = &self.poses[&edge.to];
                let r = edge_residual(&edge.measurement, pose_from, pose_to);
                let omega = match &edge.information {
                    Some(omega) => *omega,
                    None => Sim3Information::identity() * edge.weight,
                };

                let i_from = node_index.get(&edge.from).copied();
                let i_to = node_index.get(&edge.to).copied();

                // Central-difference Jacobians of the residual w.r.t. each
                // node's Sim(3) tangent perturbation.
                let j_from =
                    i_from.map(|_| numerical_jacobian(&edge.measurement, pose_from, pose_to, true));
                let j_to =
                    i_to.map(|_| numerical_jacobian(&edge.measurement, pose_from, pose_to, false));

                if let (Some(i), Some(jf)) = (i_from, &j_from) {
                    let ot = jf.transpose() * omega;
                    accumulate_block(&mut h, i * DOF, i * DOF, &(ot * jf));
                    accumulate_segment(&mut g, i * DOF, &(ot * r));
                }
                if let (Some(j), Some(jt)) = (i_to, &j_to) {
                    let ot = jt.transpose() * omega;
                    accumulate_block(&mut h, j * DOF, j * DOF, &(ot * jt));
                    accumulate_segment(&mut g, j * DOF, &(ot * r));
                }
                if let (Some(i), Some(jf), Some(j), Some(jt)) = (i_from, &j_from, i_to, &j_to) {
                    let cross = jf.transpose() * omega * jt;
                    accumulate_block(&mut h, i * DOF, j * DOF, &cross);
                    accumulate_block(&mut h, j * DOF, i * DOF, &cross.transpose());
                }
            }

            // Damp and solve (H + λI) δ = -g.
            let mut damped = h.clone();
            for d in 0..dim {
                damped[(d, d)] += lambda;
            }
            let delta = solve_normal_equations(&damped, &(-&g))?;

            let cost_before = current_cost;
            let saved_poses = config.initial_lambda.is_some().then(|| self.poses.clone());

            let mut max_step_norm: f64 = 0.0;
            for (&id, &index) in &node_index {
                let mut xi = Sim3Tangent::zeros();
                for k in 0..DOF {
                    xi[k] = delta[index * DOF + k];
                }
                max_step_norm = max_step_norm.max(xi.norm());
                let pose = self
                    .poses
                    .get_mut(&id)
                    .ok_or(PoseGraphError::MissingNode(id))?;
                *pose = pose.compose(&Sim3::exp(&xi));
            }

            let cost_after = self.cost();
            let step_accepted = config.initial_lambda.is_none() || cost_after < cost_before;

            if !step_accepted {
                if let Some(saved) = saved_poses {
                    self.poses = saved;
                }
                lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                iterations.push(Sim3PoseGraphIterationStats {
                    iteration,
                    cost_before,
                    cost_after,
                    max_step_norm,
                    lambda,
                    step_accepted: false,
                });
                if lambda >= config.max_lambda {
                    break;
                }
                continue;
            }

            iterations.push(Sim3PoseGraphIterationStats {
                iteration,
                cost_before,
                cost_after,
                max_step_norm,
                lambda,
                step_accepted: true,
            });
            current_cost = cost_after;
            if config.initial_lambda.is_some() {
                lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
            }

            if max_step_norm < config.step_tolerance
                || (cost_before - cost_after).abs() < config.cost_tolerance
            {
                converged = true;
                break;
            }
        }

        Ok(Sim3PoseGraphResult {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }
}

/// Residual `r = log(Z⁻¹ · (S_to · S_from⁻¹)) ∈ R⁷` for one edge, zero when the
/// node estimates exactly satisfy the measured relative similarity `Z`.
fn edge_residual(measurement: &Sim3, pose_from: &Sim3, pose_to: &Sim3) -> Sim3Tangent {
    let predicted = pose_to.compose(&pose_from.inverse());
    measurement.inverse().compose(&predicted).log()
}

/// Central-difference Jacobian `∂r/∂δ` (`7×7`) of the edge residual with
/// respect to a right `Sim(3)` perturbation of either the `from` or `to` node.
fn numerical_jacobian(
    measurement: &Sim3,
    pose_from: &Sim3,
    pose_to: &Sim3,
    perturb_from: bool,
) -> SMatrix<f64, 7, 7> {
    const EPS: f64 = 1e-6;
    let mut jacobian = SMatrix::<f64, 7, 7>::zeros();
    for k in 0..DOF {
        let mut plus = Sim3Tangent::zeros();
        plus[k] = EPS;
        let mut minus = Sim3Tangent::zeros();
        minus[k] = -EPS;
        let (rp, rm) = if perturb_from {
            (
                edge_residual(measurement, &pose_from.compose(&Sim3::exp(&plus)), pose_to),
                edge_residual(measurement, &pose_from.compose(&Sim3::exp(&minus)), pose_to),
            )
        } else {
            (
                edge_residual(measurement, pose_from, &pose_to.compose(&Sim3::exp(&plus))),
                edge_residual(measurement, pose_from, &pose_to.compose(&Sim3::exp(&minus))),
            )
        };
        let column = (rp - rm) / (2.0 * EPS);
        jacobian.set_column(k, &column);
    }
    jacobian
}

fn accumulate_block(h: &mut DMatrix<f64>, row: usize, col: usize, block: &SMatrix<f64, 7, 7>) {
    for r in 0..DOF {
        for c in 0..DOF {
            h[(row + r, col + c)] += block[(r, c)];
        }
    }
}

fn accumulate_segment(g: &mut DVector<f64>, offset: usize, segment: &Sim3Tangent) {
    for r in 0..DOF {
        g[offset + r] += segment[r];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    /// A ground-truth circular loop at unit (metric) scale.
    fn ground_truth(node_count: u64) -> Vec<(u64, Sim3)> {
        (0..node_count)
            .map(|i| {
                let angle = i as f64 / node_count as f64 * std::f64::consts::TAU;
                let pose = Sim3::new(
                    UnitQuaternion::from_euler_angles(0.0, 0.0, angle),
                    Vector3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 0.3 * i as f64),
                    1.0,
                );
                (i, pose)
            })
            .collect()
    }

    /// Build a graph from ground-truth nodes: sequential edges + one loop
    /// closure, every measurement the *true* relative similarity.
    fn graph_with_true_measurements(truth: &[(u64, Sim3)]) -> Sim3PoseGraph {
        let mut graph = Sim3PoseGraph::new();
        for (id, pose) in truth {
            graph.add_pose(*id, pose.clone());
        }
        let relative = |from: usize, to: usize| truth[to].1.compose(&truth[from].1.inverse());
        for w in 1..truth.len() {
            graph.add_edge(truth[w - 1].0, truth[w].0, relative(w - 1, w), 1.0);
        }
        let last = truth.len() - 1;
        graph.add_edge(truth[last].0, truth[0].0, relative(last, 0), 10.0);
        graph
    }

    #[test]
    fn ground_truth_is_a_fixed_point() {
        let truth = ground_truth(8);
        let mut graph = graph_with_true_measurements(&truth);
        graph.anchor(0);
        let result = graph.optimize(&Sim3PoseGraphConfig::default()).unwrap();
        assert!(
            result.initial_cost < 1e-18,
            "init cost {}",
            result.initial_cost
        );
        for (id, pose) in &truth {
            assert!((graph.poses[id].scale - pose.scale).abs() < 1e-9);
            assert!((graph.poses[id].translation - pose.translation).norm() < 1e-9);
        }
    }

    #[test]
    fn corrects_accumulated_scale_drift() {
        let truth = ground_truth(8);
        let mut graph = graph_with_true_measurements(&truth);
        graph.anchor(0);

        // Drift the initial estimates: a cumulative scale error plus a small
        // rotation/translation wobble that grows along the trajectory. The
        // anchor (node 0) stays at ground truth.
        for (id, pose) in &truth {
            if *id == 0 {
                continue;
            }
            let i = *id as f64;
            let mut drift = Sim3Tangent::zeros();
            drift[0] = 0.05 * i; // ρx
            drift[5] = 0.02 * i; // ωz
            drift[6] = 0.06 * i; // σ (log-scale): node 7 ≈ e^0.42 ≈ 1.52x off
            graph.add_pose(*id, pose.compose(&Sim3::exp(&drift)));
        }

        let drifted_cost = graph.cost();
        let worst_scale_error = truth
            .iter()
            .map(|(id, _)| (graph.poses[id].scale - 1.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst_scale_error > 0.4,
            "expected large initial scale drift"
        );

        let result = graph.optimize(&Sim3PoseGraphConfig::default()).unwrap();

        assert!(
            result.final_cost < drifted_cost * 1e-6,
            "final cost {}",
            result.final_cost
        );
        assert!(
            result.converged,
            "expected convergence: {:?}",
            result.iterations.len()
        );
        // Every node's scale and pose return to ground truth (the unique
        // minimum given exact measurements and a fixed anchor).
        for (id, pose) in &truth {
            assert!(
                (graph.poses[id].scale - pose.scale).abs() < 1e-6,
                "node {id} scale {} != 1",
                graph.poses[id].scale
            );
            assert!(
                (graph.poses[id].translation - pose.translation).norm() < 1e-5,
                "node {id} translation off by {}",
                (graph.poses[id].translation - pose.translation).norm()
            );
        }
    }

    #[test]
    fn errors_on_degenerate_graphs() {
        let mut empty = Sim3PoseGraph::new();
        empty.add_pose(0, Sim3::identity());
        assert_eq!(
            empty.optimize(&Sim3PoseGraphConfig::default()),
            Err(PoseGraphError::NoAnchor)
        );
        empty.anchor(0);
        assert_eq!(
            empty.optimize(&Sim3PoseGraphConfig::default()),
            Err(PoseGraphError::NoEdges)
        );

        let mut only_anchor = Sim3PoseGraph::new();
        only_anchor.add_pose(0, Sim3::identity());
        only_anchor.anchor(0);
        only_anchor.add_edge(0, 0, Sim3::identity(), 1.0);
        assert_eq!(
            only_anchor.optimize(&Sim3PoseGraphConfig::default()),
            Err(PoseGraphError::NoVariables)
        );

        let mut missing = Sim3PoseGraph::new();
        missing.add_pose(0, Sim3::identity());
        missing.anchor(0);
        missing.add_edge(0, 9, Sim3::identity(), 1.0);
        assert_eq!(
            missing.optimize(&Sim3PoseGraphConfig::default()),
            Err(PoseGraphError::MissingNode(9))
        );
    }
}
