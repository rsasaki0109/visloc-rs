//! Sparse pose-graph optimizer: SE(3) edges, robust kernels (incl. GNC),
//! chordal initialization, and the dense/sparse normal-equation solvers.

use super::*;

/// Kind of an edge inside a [`PoseGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoseGraphEdgeKind {
    /// Sequential odometry edge between consecutive keyframes.
    Sequential,
    /// Loop-closure edge backed by a verified [`LoopClosureConstraint`].
    LoopClosure,
}

/// Edge in a sparse [`PoseGraph`]. Encodes a measured `previous_to_current`
/// SE3 between two keyframes plus a positive weight used by translation-only
/// least squares.
///
/// `information` optionally carries a full 6×6 information matrix `Ω`, i.e. the
/// inverse measurement covariance, ordered `[ρ; ω]` (translation block first,
/// then rotation) to match [`SE3::log`] and the `.g2o` `EDGE_SE3:QUAT`
/// convention. When `Some`, the SE(3) solver minimizes the anisotropic
/// Mahalanobis cost `rᵀ Ω r` for this edge and the scalar `weight` is ignored;
/// when `None`, the edge falls back to the isotropic `weight · ‖r‖²` behavior.
/// This lets the graph ingest external constraints (e.g. `.g2o`
/// `EDGE_SE3:QUAT`) whose blocks couple rotation and translation, without
/// changing the meaning of the internally-built sequential / loop-closure
/// edges.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphEdge {
    pub from: u64,
    pub to: u64,
    pub measurement: SE3,
    pub kind: PoseGraphEdgeKind,
    pub weight: f64,
    pub information: Option<Matrix6<f64>>,
}

/// A dense Gaussian *prior* over a set of poses — the factor a fixed-lag /
/// sliding-window smoother re-adds when it marginalizes states out of the graph
/// (see [`crate::marginalization`] and [`PoseGraph::marginalize_pose`]).
///
/// It penalizes the quadratic `½ eᵀ Ω e + bᵀ e` where `e` stacks each pose's
/// right-tangent error from its linearization point, `eᵢ = log(T₀ᵢ⁻¹ ∘ Tᵢ)` (so
/// `∂eᵢ/∂δᵢ = I` under the solver's `T ← T ∘ exp(δ)` update). The prior therefore
/// contributes `Ω` to `H` and `Ω·e + b` to `g` with an identity Jacobian. `Ω`
/// (`information`) is the `6k × 6k` curvature and `b` (`gradient`) the linear
/// term, both in the `[ρ; ω]` SE(3)-tangent basis the edges use; `linearization[i]`
/// is `T₀ᵢ`, pose `ids[i]`'s estimate when the prior was formed. The linear term
/// `b` is the gradient of the marginalized factors at `T₀` and is what keeps the
/// linearization point a stationary point of the reduced problem — a pure
/// quadratic (`b = 0`) would wrongly assert the prior's mean is `T₀`. A prior id
/// that is the gauge-fixed anchor (or absent) contributes a zero tangent error
/// and no variable block, dropping out cleanly.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianPrior {
    /// The poses this prior constrains, indexing `information`'s `6×6` blocks,
    /// `gradient`'s `6`-segments, and `linearization` in the same order.
    pub ids: Vec<u64>,
    /// The `6k × 6k` information (inverse-covariance / curvature) matrix `Ω` over
    /// the stacked tangent error, `k = ids.len()`.
    pub information: DMatrix<f64>,
    /// The `6k` linear term `b` (the marginalized factors' gradient at `T₀`).
    /// Zero for a pure quadratic prior pinned at its linearization point.
    pub gradient: DVector<f64>,
    /// Linearization point `T₀ᵢ` (world-to-camera) per id, the estimate at which
    /// the prior was formed; the residual is measured relative to it.
    pub linearization: Vec<SE3>,
}

/// Single Gauss-Newton step diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphOptimizationStep {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub mean_translation_correction: f64,
    pub max_translation_correction: f64,
}

/// Robust kernel applied to each pose-graph edge's residual norm-squared.
/// Down-weights edges whose squared residual exceeds the kernel threshold so
/// outlier loop closures cannot dominate the least-squares solve.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RobustKernel {
    /// Standard squared-error cost (`ρ(s) = s`).
    #[default]
    None,
    /// Huber kernel: quadratic for `s ≤ δ²`, linear in `√s` beyond.
    /// `delta` is the threshold on residual norm where the kernel switches
    /// from quadratic to linear.
    Huber { delta: f64 },
    /// Cauchy / Lorentzian kernel: `ρ(s) = c² · log(1 + s / c²)`.
    /// `c` is the soft-saturation scale on residual norm.
    Cauchy { c: f64 },
}

impl RobustKernel {
    /// Applied cost `ρ(s)` for `s = ||r||²`.
    pub fn cost(&self, s: f64) -> f64 {
        match *self {
            RobustKernel::None => s,
            RobustKernel::Huber { delta } => {
                let delta_sq = delta * delta;
                if s <= delta_sq {
                    s
                } else {
                    2.0 * delta * s.sqrt() - delta_sq
                }
            }
            RobustKernel::Cauchy { c } => {
                let c_sq = c * c;
                c_sq * (1.0 + s / c_sq).ln()
            }
        }
    }

    /// Influence weight `ρ'(s)` used as a multiplier on each edge's normal-equation
    /// contribution (a.k.a. IRLS weight).
    pub fn weight(&self, s: f64) -> f64 {
        match *self {
            RobustKernel::None => 1.0,
            RobustKernel::Huber { delta } => {
                let delta_sq = delta * delta;
                if s <= delta_sq {
                    1.0
                } else {
                    delta / s.sqrt()
                }
            }
            RobustKernel::Cauchy { c } => {
                let c_sq = c * c;
                1.0 / (1.0 + s / c_sq)
            }
        }
    }
}

/// Linear solver backend used by pose-graph optimizers when the normal
/// equations `(H + λI) δ = -g` (or the translation-only analogue) are solved.
///
/// `Dense` materializes the full SPD matrix as a [`DMatrix`] and uses
/// nalgebra's dense Cholesky (LU fallback). `Sparse` assembles the same
/// system from edge triplets and solves it with the block Cholesky (the
/// `block_cholesky` module) in a fill-reducing order. The two paths produce
/// numerically equivalent solutions on connected, well-conditioned graphs but
/// the sparse path scales to thousands of keyframes where the dense path
/// becomes infeasible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinearSolver {
    /// Dense Cholesky / LU on a [`DMatrix`].
    #[default]
    Dense,
    /// Sparse block Cholesky on a triplet-assembled system.
    Sparse,
}

/// How the Levenberg-Marquardt damping `λ` enters the normal matrix `H`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DampingMode {
    /// `H + λI` — add `λ` uniformly to every diagonal entry. Simple, but
    /// scale-blind: the rotation and translation blocks of `H` differ by orders
    /// of magnitude, so a single `λ` over-damps the well-conditioned directions
    /// while under-damping the ill-conditioned ones. As `λ` grows on a hard graph
    /// the step degenerates and the solve stalls at a poor optimum (observed on
    /// sphere2500 / cubicle vs GTSAM).
    #[default]
    Identity,
    /// `H + λ·diag(H)` — Marquardt's scale-invariant damping: each variable is
    /// damped in proportion to its own curvature, so the trust region is an
    /// ellipsoid matched to the local geometry rather than a sphere. This is the
    /// default in mature LM solvers (GTSAM, Ceres) and reaches a markedly lower
    /// optimum than `Identity` on graphs with mixed rotation/translation
    /// curvature.
    Diagonal,
}

/// Configuration for [`PoseGraph::optimize_se3_iterative`].
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSe3Config {
    /// Hard cap on iterations (including rejected LM steps).
    pub max_iterations: usize,
    /// Convergence threshold on the largest per-node 6-vector update of the
    /// most recent accepted step.
    pub step_tolerance: f64,
    /// Convergence threshold on the absolute cost change between two
    /// successive accepted steps.
    pub cost_tolerance: f64,
    /// Robust kernel applied to each edge's squared residual norm.
    pub robust_kernel: RobustKernel,
    /// Initial Levenberg-Marquardt damping `λ`. `None` runs pure
    /// Gauss-Newton (every step is accepted unconditionally). `Some(λ₀)`
    /// enables LM: solve `(H + λI) δ = -g`, accept if cost decreases (and
    /// scale `λ` down by `lambda_decrease_factor`), otherwise reject and
    /// scale `λ` up by `lambda_increase_factor`.
    pub initial_lambda: Option<f64>,
    /// Multiplier applied to `λ` after a rejected LM step.
    pub lambda_increase_factor: f64,
    /// Multiplier applied to `λ` after an accepted LM step.
    pub lambda_decrease_factor: f64,
    /// How `λ` enters `H`. Defaults to [`DampingMode::Identity`] (`H + λI`, leaves
    /// every solve bit-identical). Switch to [`DampingMode::Diagonal`]
    /// (`H + λ·diag(H)`) for graphs with mixed rotation/translation curvature.
    pub damping: DampingMode,
    /// Upper bound on `λ`. When a step is rejected and `λ * factor > max_lambda`,
    /// the optimizer gives up and returns `converged: false`.
    pub max_lambda: f64,
    /// Lower bound on `λ`. Decreases stop here so `λ` cannot collapse to zero.
    pub min_lambda: f64,
    /// Linear-solver backend. Defaults to dense Cholesky for parity with the
    /// pre-sparse solver. Switch to [`LinearSolver::Sparse`] when the graph
    /// has more than a few hundred nodes so the optimizer scales linearly in
    /// non-zero edges instead of cubically in node count.
    pub linear_solver: LinearSolver,
    /// Seed the solve with a chordal rotation initialization
    /// ([`PoseGraph::initialize_rotations_chordal`]) before the first
    /// Gauss-Newton step. Defaults to `true`: it is strictly beneficial —
    /// the rotation optimum is a fixed point of the relaxation, so on an
    /// already-consistent graph it leaves the estimate essentially unchanged
    /// (a cheap extra factorization), while on a hard, odometry-initialized 3D
    /// graph it rescues the solve from a poor basin. The seeding is best-effort:
    /// if its relaxed system is singular it is silently skipped and the solve
    /// proceeds from the unmodified estimate, so enabling it can never turn a
    /// previously-successful optimization into a failure.
    pub chordal_init: bool,
}

impl Default for PoseGraphSe3Config {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            step_tolerance: 1e-6,
            cost_tolerance: 1e-9,
            robust_kernel: RobustKernel::None,
            initial_lambda: None,
            lambda_increase_factor: 10.0,
            lambda_decrease_factor: 0.1,
            damping: DampingMode::Identity,
            max_lambda: 1e12,
            min_lambda: 1e-9,
            linear_solver: LinearSolver::Dense,
            chordal_init: true,
        }
    }
}

/// Per-iteration diagnostics for [`PoseGraph::optimize_se3_iterative`].
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSe3IterationStats {
    pub iteration: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub max_step_norm: f64,
    /// LM damping `λ` used for this iteration (`0.0` for pure Gauss-Newton).
    pub lambda: f64,
    /// `true` when the trial step was kept; only false for rejected LM steps.
    pub step_accepted: bool,
}

/// Diagnostics from [`PoseGraph::initialize_rotations_chordal`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChordalRotationInit {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    /// Chordal rotation cost (see [`PoseGraph::chordal_rotation_cost`]) before
    /// the relaxation was solved.
    pub cost_before: f64,
    /// Chordal rotation cost after replacing every node rotation with the
    /// SVD-projected chordal solution.
    pub cost_after: f64,
    /// Largest per-node geodesic rotation change (degrees) applied by the init.
    pub max_rotation_update_deg: f64,
}

/// Result of a full SE(3) Gauss-Newton run.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSe3Result {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: Vec<PoseGraphSe3IterationStats>,
    pub converged: bool,
}

/// Outcome of [`PoseGraph::optimize_se3_gnc`], the outlier-robust SE(3)
/// pose-graph solve driven by Graduated Non-Convexity (see [`crate::gnc`]).
///
/// Beyond the usual cost/convergence summary it reports the **final per-edge
/// GNC weight** (`edge_weights`, indexed by edge position, each in `[0, 1]`):
/// an edge annealed to a weight near zero was rejected as an outlier, so this
/// vector doubles as a loop-closure inlier/outlier classification.
#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphGncResult {
    pub anchor_id: u64,
    pub edge_count: usize,
    pub variable_count: usize,
    /// Plain (non-robust) least-squares cost at the seeded starting point.
    pub initial_cost: f64,
    /// Plain (non-robust) least-squares cost over all edges at the final
    /// estimate. Outlier edges still contribute their (large) residual here;
    /// use [`Self::inlier_cost`] for the cost over retained edges only.
    pub final_cost: f64,
    /// Plain least-squares cost summed over edges whose final weight is at or
    /// above the inlier `threshold` passed to the solve — the cost GNC actually
    /// drove down once outliers were rejected.
    pub inlier_cost: f64,
    /// The inlier scale `c` the solve actually used: the configured
    /// [`gnc::GncConfig::c`] verbatim, or — under
    /// [`gnc::GncConfig::auto_scale`] — the MAD estimate (floored at the
    /// configured `c`).
    pub inlier_scale: f64,
    /// Number of outer `μ` levels executed.
    pub outer_iterations: usize,
    /// Whether the `μ` schedule reached the true robust cost (terminal `μ`).
    pub converged: bool,
    /// Final GNC weight per edge, indexed by edge position, in `[0, 1]`.
    pub edge_weights: Vec<f64>,
}

impl PoseGraphGncResult {
    /// Number of edges GNC kept as inliers: final weight `≥ threshold`.
    pub fn inlier_count(&self, threshold: f64) -> usize {
        self.edge_weights
            .iter()
            .filter(|&&w| w >= threshold)
            .count()
    }

    /// Number of edges GNC rejected as outliers: final weight `< threshold`.
    pub fn outlier_count(&self, threshold: f64) -> usize {
        self.edge_count - self.inlier_count(threshold)
    }
}

/// Errors returned by [`PoseGraph::optimize_translations_once`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoseGraphError {
    /// No anchor was specified before optimization.
    NoAnchor,
    /// An edge or anchor referenced a node that is missing from the graph.
    MissingNode(u64),
    /// The graph contains no edges, so there is nothing to optimize.
    NoEdges,
    /// The graph contains no non-anchor nodes (all variables are fixed).
    NoVariables,
    /// The Gauss-Newton normal equations were singular, e.g., because the
    /// graph has disconnected components or rank-deficient constraints.
    SingularSystem,
}

impl std::fmt::Display for PoseGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoseGraphError::NoAnchor => write!(f, "pose graph has no anchor"),
            PoseGraphError::MissingNode(id) => write!(f, "pose graph is missing node {id}"),
            PoseGraphError::NoEdges => write!(f, "pose graph has no edges"),
            PoseGraphError::NoVariables => write!(f, "pose graph has no non-anchor nodes"),
            PoseGraphError::SingularSystem => {
                write!(f, "pose graph translation Gauss-Newton system was singular")
            }
        }
    }
}

impl std::error::Error for PoseGraphError {}

/// Sparse pose graph keyed by keyframe id. Stores per-node poses plus a flat
/// list of sequential and loop-closure edges, and provides a single
/// translation-only Gauss-Newton step that keeps node rotations fixed.
///
/// This is intentionally a skeleton: rotations are not optimized, the solver
/// is a single linear least-squares step rather than an iterative SE3 solver,
/// and there is no incremental incremental map update. Future milestones can
/// extend the same data type with full SE3 Jacobians, robust kernels, or a
/// production solver.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseGraph {
    /// Keyframe id → pose. `BTreeMap` keeps the iteration order deterministic
    /// so the variable layout in the linear system is reproducible.
    pub poses: BTreeMap<u64, Pose>,
    /// Edges in insertion order.
    pub edges: Vec<PoseGraphEdge>,
    /// Dense Gaussian priors (marginalization factors). Empty for a plain graph,
    /// so the solver and cost are bit-identical until a [`Self::marginalize_pose`]
    /// folds a state into one.
    pub priors: Vec<GaussianPrior>,
    /// Anchor keyframe id; its pose is held fixed during optimization.
    pub anchor: Option<u64>,
}

impl PoseGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a pose for `keyframe_id`.
    pub fn add_pose(&mut self, keyframe_id: u64, pose: Pose) {
        self.poses.insert(keyframe_id, pose);
    }

    /// Designate `keyframe_id` as the anchor whose pose stays fixed during
    /// translation optimization. Replaces any previously selected anchor.
    pub fn anchor(&mut self, keyframe_id: u64) {
        self.anchor = Some(keyframe_id);
    }

    /// Add a sequential odometry edge with weight `1.0`.
    pub fn add_sequential_edge(&mut self, from: u64, to: u64, measurement: SE3) {
        self.edges.push(PoseGraphEdge {
            from,
            to,
            measurement,
            kind: PoseGraphEdgeKind::Sequential,
            weight: 1.0,
            information: None,
        });
    }

    /// Append a loop-closure constraint as a graph edge. The verifier-derived
    /// inlier count is reused as the edge weight (clamped to a minimum of
    /// `1.0`) so loops with more inliers carry more pull on the solver.
    pub fn add_loop_closure_constraint(&mut self, constraint: &LoopClosureConstraint) {
        let weight = (constraint.inlier_count as f64).max(1.0);
        self.add_loop_closure_constraint_with_weight(constraint, weight);
    }

    /// Append a loop-closure constraint with an explicit isotropic scalar
    /// weight. This is the controlled alternative to interpreting a verifier's
    /// raw inlier count as inverse covariance: correspondence count and pose
    /// uncertainty do not share units, and a large count can otherwise make one
    /// loop hundreds of times stiffer than every sequential edge. Invalid
    /// weights are conservatively replaced by `1.0` so a malformed runtime
    /// configuration cannot inject NaNs into the graph.
    pub fn add_loop_closure_constraint_with_weight(
        &mut self,
        constraint: &LoopClosureConstraint,
        weight: f64,
    ) {
        let weight = if weight.is_finite() && weight > 0.0 {
            weight
        } else {
            1.0
        };
        self.edges.push(PoseGraphEdge {
            from: constraint.from_keyframe_id,
            to: constraint.to_keyframe_id,
            measurement: constraint.relative_pose.clone(),
            kind: PoseGraphEdgeKind::LoopClosure,
            weight,
            information: None,
        });
    }

    /// Add an edge carrying a full 6×6 information matrix `Ω`, ordered `[ρ; ω]`
    /// (translation block first, then rotation — the [`SE3::log`] / `.g2o`
    /// `EDGE_SE3:QUAT` convention). The SE(3) solver minimizes the anisotropic
    /// `rᵀ Ω r` for this edge; the scalar `weight` is left at `1.0` and unused
    /// while `information` is `Some`.
    pub fn add_edge_with_information(
        &mut self,
        from: u64,
        to: u64,
        measurement: SE3,
        kind: PoseGraphEdgeKind,
        information: Matrix6<f64>,
    ) {
        self.edges.push(PoseGraphEdge {
            from,
            to,
            measurement,
            kind,
            weight: 1.0,
            information: Some(information),
        });
    }

    /// Sum of squared edge translation residuals in world coordinates.
    /// Rotation residuals are ignored — this is a translation-only metric
    /// that matches what [`Self::optimize_translations_once`] minimizes.
    pub fn translation_cost(&self) -> f64 {
        let mut total = 0.0;
        for edge in &self.edges {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let displacement = expected_world_displacement(to, &edge.measurement);
            let actual = to.camera_center_world() - from.camera_center_world();
            let residual = actual - displacement;
            total += edge.weight * residual.norm_squared();
        }
        total
    }

    /// Solve a single Gauss-Newton step on the translation residuals while
    /// holding rotations fixed. With linear-in-translation residuals the
    /// "single step" is the exact least-squares optimum of the underlying
    /// linear system, not a Newton iteration that needs to be repeated.
    ///
    /// Equivalent to [`Self::optimize_translations_once_with`] called with
    /// [`LinearSolver::Dense`].
    pub fn optimize_translations_once(
        &mut self,
    ) -> Result<PoseGraphOptimizationStep, PoseGraphError> {
        self.optimize_translations_once_with(LinearSolver::Dense)
    }

    /// Variant of [`Self::optimize_translations_once`] that selects the
    /// linear-solver backend. Use [`LinearSolver::Sparse`] for graphs with
    /// hundreds-to-thousands of keyframes — the normal-equations matrix is
    /// block-banded with at most four `3×3` blocks per edge, so the sparse
    /// block Cholesky is much faster than the dense path and uses dramatically
    /// less memory.
    pub fn optimize_translations_once_with(
        &mut self,
        linear_solver: LinearSolver,
    ) -> Result<PoseGraphOptimizationStep, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        let anchor_pose = self
            .poses
            .get(&anchor_id)
            .ok_or(PoseGraphError::MissingNode(anchor_id))?
            .clone();
        let anchor_center = anchor_pose.camera_center_world();
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
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

        for edge in &self.edges {
            if !self.poses.contains_key(&edge.from) {
                return Err(PoseGraphError::MissingNode(edge.from));
            }
            if !self.poses.contains_key(&edge.to) {
                return Err(PoseGraphError::MissingNode(edge.to));
            }
        }

        let cost_before = self.translation_cost();

        // Assemble the normal equations `A^T A x = A^T b` directly. Each row
        // of `A` (per edge) has at most two nonzero 3×3 identity-shaped
        // blocks (`+w · I` at `to`, `-w · I` at `from`), so the contribution
        // to `A^T A` per edge is also block-structured: `w² · I` at the
        // `(to, to)` and `(from, from)` diagonal blocks and `-w² · I` at
        // both off-diagonal blocks. Anchor-incident edges only push to one
        // diagonal and shift `A^T b` by the anchor center.
        let dim = variable_count * 3;
        let mut h_dense = match linear_solver {
            LinearSolver::Dense => Some(DMatrix::<f64>::zeros(dim, dim)),
            LinearSolver::Sparse => None,
        };
        let mut triplets: Vec<(usize, usize, f64)> = match linear_solver {
            LinearSolver::Dense => Vec::new(),
            LinearSolver::Sparse => Vec::with_capacity(self.edges.len() * 36),
        };
        let mut atb = DVector::<f64>::zeros(dim);

        for edge in &self.edges {
            let to_pose = &self.poses[&edge.to];
            let displacement = expected_world_displacement(to_pose, &edge.measurement);
            let mut rhs = displacement;
            let w2 = edge.weight;

            let i_to = node_index.get(&edge.to).copied();
            let i_from = node_index.get(&edge.from).copied();
            if i_to.is_none() {
                rhs -= anchor_center.coords;
            }
            if i_from.is_none() {
                rhs += anchor_center.coords;
            }

            if let Some(j) = i_to {
                add_diag_block3(&mut h_dense, &mut triplets, j * 3, w2);
                for k in 0..3 {
                    atb[j * 3 + k] += w2 * rhs[k];
                }
            }
            if let Some(i) = i_from {
                add_diag_block3(&mut h_dense, &mut triplets, i * 3, w2);
                for k in 0..3 {
                    atb[i * 3 + k] -= w2 * rhs[k];
                }
            }
            if let (Some(j), Some(i)) = (i_to, i_from) {
                add_offdiag_block3(&mut h_dense, &mut triplets, j * 3, i * 3, -w2);
                add_offdiag_block3(&mut h_dense, &mut triplets, i * 3, j * 3, -w2);
            }
        }

        let solution = match linear_solver {
            LinearSolver::Dense => {
                let h = h_dense.expect("dense matrix initialized when LinearSolver::Dense");
                solve_normal_equations(&h, &atb)?
            }
            LinearSolver::Sparse => {
                let order = reordering::Reordering::fill_reducing(dim, 3, &triplets);
                // One-shot solve (no LM loop), so the symbolic analysis is not reused.
                solve_normal_equations_sparse(&triplets, dim, 3, &atb, 0.0, &order, &mut None)?
            }
        };

        let mut total_correction = 0.0;
        let mut max_correction: f64 = 0.0;
        for (&id, &i) in &node_index {
            let new_center = Point3::new(solution[i * 3], solution[i * 3 + 1], solution[i * 3 + 2]);
            let pose = self
                .poses
                .get_mut(&id)
                .ok_or(PoseGraphError::MissingNode(id))?;
            let old_center = pose.camera_center_world();
            let correction_norm = (new_center - old_center).norm();
            total_correction += correction_norm;
            if correction_norm > max_correction {
                max_correction = correction_norm;
            }
            let rotation_matrix = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            pose.world_to_camera.translation = -(rotation_matrix * new_center.coords);
        }

        let cost_after = self.translation_cost();
        let mean_translation_correction = if variable_count > 0 {
            total_correction / variable_count as f64
        } else {
            0.0
        };

        Ok(PoseGraphOptimizationStep {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            cost_before,
            cost_after,
            mean_translation_correction,
            max_translation_correction: max_correction,
        })
    }

    /// Sum of squared SE(3) residuals: r_e = log(meas_e⁻¹ · T_to · T_from⁻¹),
    /// weighted by `edge.weight`. Unlike [`Self::translation_cost`], this
    /// includes both the translation and rotation components of every edge.
    pub fn se3_cost(&self) -> f64 {
        self.robust_se3_cost(&RobustKernel::None)
    }

    /// Robust SE(3) cost: `Σ_e edge.weight · ρ(||r_e||²)` where `ρ` is the
    /// supplied [`RobustKernel`]. With [`RobustKernel::None`] this matches
    /// [`Self::se3_cost`].
    pub fn robust_se3_cost(&self, kernel: &RobustKernel) -> f64 {
        let mut total = 0.0;
        for edge in &self.edges {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
            let r = edge.measurement.inverse().compose(&predicted).log();
            total += match &edge.information {
                // Anisotropic: robust kernel operates on the Mahalanobis
                // distance rᵀΩr; the scalar weight is folded into Ω.
                Some(omega) => kernel.cost((r.transpose() * omega * r)[(0, 0)]),
                // Isotropic: weight scales the kernel output, kernel sees ‖r‖².
                None => edge.weight * kernel.cost(r.norm_squared()),
            };
        }
        total + self.prior_cost()
    }

    /// Stacked right-tangent error of a prior's poses from their linearization:
    /// `eᵢ = log(T₀ᵢ⁻¹ ∘ Tᵢ)`, one 6-vector per id, in `ids` order (`[ρ; ω]`
    /// SE(3) tangent). A pose absent from the graph — or pinned at its
    /// linearization, like the anchor — contributes a zero block. Length
    /// `6 · ids.len()`.
    fn prior_tangent_error(&self, prior: &GaussianPrior) -> DVector<f64> {
        let mut e = DVector::<f64>::zeros(prior.ids.len() * 6);
        for (i, (&id, t0)) in prior.ids.iter().zip(&prior.linearization).enumerate() {
            if let Some(pose) = self.poses.get(&id) {
                let err = t0.inverse().compose(&pose.world_to_camera).log();
                for k in 0..6 {
                    e[i * 6 + k] = err[k];
                }
            }
        }
        e
    }

    /// Total Gaussian-prior cost `Σ (eᵀ Ω e + 2 bᵀ e)` (the marginalization
    /// factors), on the same `2×` scale as the edge cost `rᵀΩr` so the LM accept
    /// test sees one consistent objective (the dropped constant cancels in
    /// `cost_after − cost_before`). Zero when the graph carries no priors.
    fn prior_cost(&self) -> f64 {
        self.priors
            .iter()
            .map(|prior| {
                let e = self.prior_tangent_error(prior);
                (e.transpose() * &prior.information * &e)[(0, 0)] + 2.0 * prior.gradient.dot(&e)
            })
            .sum()
    }

    /// Robust SE(3) cost with an optional per-edge multiplier (`gnc_weights`,
    /// indexed by edge position). Identical to [`Self::robust_se3_cost`] when
    /// `gnc_weights` is `None`; with weights it is the *weighted* objective
    /// `Σ wᵢ · ρ(sᵢ)` that the Graduated Non-Convexity driver minimizes at a
    /// fixed `μ` level (the Black-Rangarajan inner problem), where `wᵢ` is the
    /// closed-form GNC weight and the kernel is [`RobustKernel::None`] (GNC
    /// supersedes the M-estimator). See [`crate::gnc`] and
    /// [`Self::optimize_se3_gnc`].
    fn robust_se3_cost_weighted(&self, kernel: &RobustKernel, gnc_weights: Option<&[f64]>) -> f64 {
        let mut total = 0.0;
        for (idx, edge) in self.edges.iter().enumerate() {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
            let r = edge.measurement.inverse().compose(&predicted).log();
            let gw = gnc_weights.map_or(1.0, |w| w[idx]);
            total += gw
                * match &edge.information {
                    Some(omega) => kernel.cost((r.transpose() * omega * r)[(0, 0)]),
                    None => edge.weight * kernel.cost(r.norm_squared()),
                };
        }
        total + self.prior_cost()
    }

    /// Per-edge (whitened) squared residual `sᵢ`, indexed by edge position —
    /// the same quantity the [`RobustKernel`] sees: the Mahalanobis distance
    /// `rᵀΩr` for an edge carrying a full information matrix, else `‖r‖²`.
    /// Edges referencing a missing node contribute `0.0` so the vector stays
    /// aligned with [`Self::edges`]. Used by the GNC driver to reweight edges.
    fn edge_squared_residuals(&self) -> Vec<f64> {
        self.edges
            .iter()
            .map(|edge| {
                let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
                else {
                    return 0.0;
                };
                let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
                let r = edge.measurement.inverse().compose(&predicted).log();
                match &edge.information {
                    Some(omega) => (r.transpose() * omega * r)[(0, 0)],
                    None => r.norm_squared(),
                }
            })
            .collect()
    }

    /// Assemble the Gauss-Newton normal equations `(H, g)` for the SE(3) pose
    /// graph: the robust-weighted `H = Σ Jᵀ Ω J` and gradient `g = Σ Jᵀ Ω r`
    /// over all edges, in the `node_index` variable layout (`dim = 6 · #vars`).
    ///
    /// Extracted from [`Self::optimize_se3_iterative`] so both the plain
    /// optimizer and the GNC driver share one assembly. `gnc_weights` is an
    /// optional per-edge multiplier (indexed by edge position): `None`
    /// reproduces the plain assembly bit-for-bit; `Some(w)` scales each edge's
    /// contribution by `wᵢ ∈ [0, 1]` for the GNC inner solve (see
    /// [`crate::gnc`]). The isotropic path keeps the legacy semantics
    /// (`weight` outside the kernel, kernel on `‖r‖²`); the anisotropic path
    /// folds `Ω` into both `JᵀJ` and `Jᵀr` and lets the kernel see the
    /// Mahalanobis distance `rᵀΩr`.
    fn assemble_se3_system(
        &self,
        node_index: &BTreeMap<u64, usize>,
        dim: usize,
        kernel: &RobustKernel,
        gnc_weights: Option<&[f64]>,
        linear_solver: LinearSolver,
    ) -> (NormalEquations6, DVector<f64>) {
        let mut builder = NormalEquations6::new(dim, linear_solver, self.edges.len());
        let mut g = DVector::<f64>::zeros(dim);

        for (idx, edge) in self.edges.iter().enumerate() {
            let t_from = &self.poses[&edge.from].world_to_camera;
            let t_to = &self.poses[&edge.to].world_to_camera;
            let predicted = t_to.compose(&t_from.inverse());
            let r = edge.measurement.inverse().compose(&predicted).log();
            let ad_from = t_from.adjoint();
            let (weight, ata, atr) = match &edge.information {
                Some(omega) => {
                    let robust_weight = kernel.weight((r.transpose() * omega * r)[(0, 0)]);
                    let oa = ad_from.transpose() * omega;
                    (robust_weight, oa * ad_from, oa * r)
                }
                None => {
                    let robust_weight = kernel.weight(r.norm_squared());
                    (
                        edge.weight * robust_weight,
                        ad_from.transpose() * ad_from,
                        ad_from.transpose() * r,
                    )
                }
            };
            // GNC reweighting: a multiplier in [0, 1] (1.0 when not running
            // GNC) that scales the whole edge contribution, rejecting outliers
            // as `wᵢ → 0`.
            let weight = weight * gnc_weights.map_or(1.0, |w| w[idx]);

            let i_from = node_index.get(&edge.from).copied();
            let i_to = node_index.get(&edge.to).copied();

            if let Some(j) = i_to {
                builder.add_block6(j * 6, j * 6, weight, &ata);
                add_segment6(&mut g, j * 6, weight, &atr);
            }
            if let Some(i) = i_from {
                builder.add_block6(i * 6, i * 6, weight, &ata);
                add_segment6(&mut g, i * 6, -weight, &atr);
            }
            if let (Some(j), Some(i)) = (i_to, i_from) {
                let cross = -ata;
                let cross_t = cross.transpose();
                builder.add_block6(j * 6, i * 6, weight, &cross);
                builder.add_block6(i * 6, j * 6, weight, &cross_t);
            }
        }

        // Gaussian (marginalization) priors: the factor `½ eᵀ Ω e` with
        // `eᵢ = log(T₀ᵢ⁻¹ ∘ Tᵢ)` and identity Jacobian (`∂eᵢ/∂δᵢ = I`), so it
        // contributes `Ω` to `H` and `Ω·e` to `g` on the variable blocks. A
        // prior id that is the anchor (or absent) has `eᵢ ≈ 0` and no variable
        // block, so it drops out — leaving the kept poses' coupling intact.
        for prior in &self.priors {
            let e = self.prior_tangent_error(prior);
            let ge = &prior.information * &e + &prior.gradient; // Ω·e + b  (length 6k)
            for (pi, &id_i) in prior.ids.iter().enumerate() {
                let Some(vi) = node_index.get(&id_i).copied() else {
                    continue;
                };
                add_segment6(
                    &mut g,
                    vi * 6,
                    1.0,
                    &Vector6::from_fn(|k, _| ge[pi * 6 + k]),
                );
                for (pj, &id_j) in prior.ids.iter().enumerate() {
                    let Some(vj) = node_index.get(&id_j).copied() else {
                        continue;
                    };
                    let block =
                        Matrix6::from_fn(|r, c| prior.information[(pi * 6 + r, pj * 6 + c)]);
                    builder.add_block6(vi * 6, vj * 6, 1.0, &block);
                }
            }
        }

        (builder, g)
    }

    /// Chordal (Frobenius-relaxed) cost of the current node rotations:
    /// `Σ_e w_e · ‖R_to − R_meas_e · R_from‖_F²`, where `R_*` is the rotation
    /// of each node's `world_to_camera` and `R_meas_e` the rotation of the
    /// edge measurement. This is the objective minimized by
    /// [`Self::initialize_rotations_chordal`]; unlike [`Self::se3_cost`] it
    /// ignores translation and uses the chordal (embedded-Euclidean) metric on
    /// SO(3) rather than the geodesic one, so it is a convex function of the
    /// relaxed (unconstrained 3×3) rotation variables.
    pub fn chordal_rotation_cost(&self) -> f64 {
        let mut total = 0.0;
        for edge in &self.edges {
            let (Some(from), Some(to)) = (self.poses.get(&edge.from), self.poses.get(&edge.to))
            else {
                continue;
            };
            let r_from = from
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let r_to = to
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let r_meas = edge.measurement.rotation.to_rotation_matrix().into_inner();
            total += chordal_rotation_weight(edge) * (r_to - r_meas * r_from).norm_squared();
        }
        total
    }

    /// Initialize node rotations by solving the *chordal relaxation* of the
    /// rotation-only sub-problem (Carlone et al., "Initialization Techniques
    /// for 3D SLAM", ICRA 2015). On hard 3D datasets the SE(3) cost surface is
    /// strongly non-convex in rotation, so a full solve started from raw
    /// odometry stalls in a poor basin; seeding it with the chordal solution
    /// lands it near the global optimum.
    ///
    /// Each edge contributes the residual `R_to − R_meas · R_from` measured in
    /// the embedded-Euclidean (Frobenius) metric. Relaxing every `R_i` from
    /// `SO(3)` to an unconstrained `3×3` matrix makes the objective a single
    /// linear least-squares problem; the per-node `9`-vector splits into three
    /// independent `3`-vector systems (one per rotation column) that share the
    /// *same* `3n × 3n` normal matrix — so this factors once and solves three
    /// right-hand sides. Each relaxed `3×3` block is then projected back onto
    /// `SO(3)` with an SVD (`R = U·diag(1,1,det(UVᵀ))·Vᵀ`).
    ///
    /// The anchor's rotation is held fixed (it fixes the global gauge). Each
    /// node's camera *center* is preserved — only its orientation is replaced —
    /// so this is safe to call standalone, though the intended flow is
    /// chordal-rotation → [`Self::optimize_translations_once_with`] →
    /// [`Self::optimize_se3_iterative`].
    pub fn initialize_rotations_chordal(
        &mut self,
        linear_solver: LinearSolver,
    ) -> Result<ChordalRotationInit, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        let anchor_pose = self
            .poses
            .get(&anchor_id)
            .ok_or(PoseGraphError::MissingNode(anchor_id))?;
        let r_anchor = anchor_pose
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
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

        let cost_before = self.chordal_rotation_cost();

        // Assemble the shared 3n×3n normal matrix and three right-hand sides
        // (the three columns of the stacked rotation matrices). The matrix is
        // identical across columns because it depends only on the (orthonormal)
        // measured rotations and the edge weights; each anchor-incident edge
        // shifts one column's right-hand side by that column of R_anchor.
        let dim = variable_count * 3;
        let mut h_dense = match linear_solver {
            LinearSolver::Dense => Some(DMatrix::<f64>::zeros(dim, dim)),
            LinearSolver::Sparse => None,
        };
        let mut triplets: Vec<(usize, usize, f64)> = match linear_solver {
            LinearSolver::Dense => Vec::new(),
            LinearSolver::Sparse => Vec::with_capacity(self.edges.len() * 36),
        };
        let mut rhs = DMatrix::<f64>::zeros(dim, 3);

        for edge in &self.edges {
            let r_meas = edge.measurement.rotation.to_rotation_matrix().into_inner();
            let w = chordal_rotation_weight(edge);

            let i_to = node_index.get(&edge.to).copied();
            let i_from = node_index.get(&edge.from).copied();

            if let Some(j) = i_to {
                add_diag_block3(&mut h_dense, &mut triplets, j * 3, w);
            }
            if let Some(i) = i_from {
                add_diag_block3(&mut h_dense, &mut triplets, i * 3, w);
            }
            match (i_to, i_from) {
                (Some(j), Some(i)) => {
                    // Both endpoints free: off-diagonal coupling -w·R_meas.
                    add_dense_block3(&mut h_dense, &mut triplets, j * 3, i * 3, &(-w * r_meas));
                    add_dense_block3(
                        &mut h_dense,
                        &mut triplets,
                        i * 3,
                        j * 3,
                        &(-w * r_meas.transpose()),
                    );
                }
                (Some(j), None) => {
                    // `from` is the anchor: g_to += w · R_meas · col(R_anchor).
                    let contribution = w * r_meas * r_anchor;
                    for c in 0..3 {
                        for k in 0..3 {
                            rhs[(j * 3 + k, c)] += contribution[(k, c)];
                        }
                    }
                }
                (None, Some(i)) => {
                    // `to` is the anchor: g_from += w · R_measᵀ · col(R_anchor).
                    let contribution = w * r_meas.transpose() * r_anchor;
                    for c in 0..3 {
                        for k in 0..3 {
                            rhs[(i * 3 + k, c)] += contribution[(k, c)];
                        }
                    }
                }
                (None, None) => {}
            }
        }

        let solution = match linear_solver {
            LinearSolver::Dense => {
                let h = h_dense.expect("dense matrix initialized when LinearSolver::Dense");
                let chol = h.clone().cholesky().ok_or(PoseGraphError::SingularSystem)?;
                chol.solve(&rhs)
            }
            LinearSolver::Sparse => {
                let order = reordering::Reordering::fill_reducing(dim, 3, &triplets);
                solve_normal_equations_sparse_multi(&triplets, dim, 3, &rhs, &order)?
            }
        };

        // Reshape each node's solved columns into a 3×3, project onto SO(3),
        // and replace the node's orientation while keeping its camera center.
        let mut max_rotation_update_rad: f64 = 0.0;
        for (&id, &i) in &node_index {
            let mut block = Matrix3::<f64>::zeros();
            for c in 0..3 {
                for k in 0..3 {
                    block[(k, c)] = solution[(i * 3 + k, c)];
                }
            }
            let projected = project_to_so3(&block).ok_or(PoseGraphError::SingularSystem)?;
            let new_rotation =
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(projected));

            let pose = self
                .poses
                .get_mut(&id)
                .ok_or(PoseGraphError::MissingNode(id))?;
            let center = pose.camera_center_world();
            let delta = new_rotation
                .rotation_to(&pose.world_to_camera.rotation)
                .angle();
            max_rotation_update_rad = max_rotation_update_rad.max(delta);
            pose.world_to_camera.rotation = new_rotation;
            // Re-derive translation from the preserved center: t = -R·center.
            pose.world_to_camera.translation = -(projected * center.coords);
        }

        let cost_after = self.chordal_rotation_cost();
        Ok(ChordalRotationInit {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            cost_before,
            cost_after,
            max_rotation_update_deg: max_rotation_update_rad.to_degrees(),
        })
    }

    /// Run a full SE(3) Gauss-Newton optimization with right-perturbation
    /// updates `T_i ← T_i · Exp(δ_i)`. Uses the first-order BCH approximation
    /// `J_r⁻¹(r) ≈ I`, so each edge contributes:
    ///
    /// - residual: `r_e = log(meas_e⁻¹ · T_to · T_from⁻¹)` (6-vector),
    /// - Jacobians: `∂r/∂δ_to = Ad(T_from)`, `∂r/∂δ_from = -Ad(T_from)`.
    ///
    /// The anchor pose is held fixed; all other poses are updated. Returns the
    /// per-iteration cost trace plus a `converged` flag derived from the
    /// configured tolerances.
    pub fn optimize_se3_iterative(
        &mut self,
        config: &PoseGraphSe3Config,
    ) -> Result<PoseGraphSe3Result, PoseGraphError> {
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

        let kernel = config.robust_kernel;
        // `initial_cost` records the true starting point — measured before any
        // seeding — so the reported reduction reflects the full improvement,
        // including the (often large) drop the chordal step front-loads.
        let initial_cost = self.robust_se3_cost(&kernel);

        // Optional chordal rotation seeding: solve the relaxed rotation
        // sub-problem to a globally-consistent orientation and re-derive
        // translations before the non-convex SE(3) solve. Best-effort — a
        // singular relaxation is skipped, leaving the unmodified estimate, so
        // seeding can never turn a solvable problem into a failure. The
        // rotation re-derivation already restores translations from the
        // preserved camera centers, so the translation LS is a further refine
        // whose failure is also harmless.
        if config.chordal_init
            && self
                .initialize_rotations_chordal(config.linear_solver)
                .is_ok()
        {
            let _ = self.optimize_translations_once_with(config.linear_solver);
        }

        let mut iterations: Vec<PoseGraphSe3IterationStats> =
            Vec::with_capacity(config.max_iterations);
        let mut converged = false;
        // The LM accept test compares against the *seeded* cost, so the loop
        // starts from the post-seed state rather than the pre-seed `initial_cost`.
        let mut current_cost = self.robust_se3_cost(&kernel);
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let dim = variable_count * 6;
        // The fill-reducing ordering depends only on the (iteration-invariant)
        // sparsity pattern, so compute it lazily on the first sparse solve and
        // reuse it for the rest of the optimization.
        let mut order_cache: Option<reordering::Reordering> = None;
        // The normal-equations sparsity pattern is iteration-invariant, so the
        // block-Cholesky symbolic factorization is analyzed once and reused
        // alongside the fill-reducing order across all LM iterations.
        let mut symbolic_cache: Option<block_cholesky::BlockSymbolic> = None;

        for iteration in 0..config.max_iterations {
            let (builder, g) =
                self.assemble_se3_system(&node_index, dim, &kernel, None, config.linear_solver);

            // Marquardt damping scales by the (undamped) curvature, so capture the
            // diagonal of H before it is consumed by the solve.
            let diag = match config.damping {
                DampingMode::Diagonal => Some(builder.diagonal()),
                DampingMode::Identity => None,
            };
            let neg_g = -&g;
            let delta = builder.solve(
                lambda,
                diag.as_deref(),
                &neg_g,
                &mut order_cache,
                &mut symbolic_cache,
            )?;

            // Tentatively apply the step so we can evaluate the new cost.
            let mut max_step_norm: f64 = 0.0;
            let cost_before = current_cost;
            let saved_poses = if config.initial_lambda.is_some() {
                Some(self.poses.clone())
            } else {
                None
            };
            for (&id, &i) in &node_index {
                let block = i * 6;
                let mut xi = Vector6::<f64>::zeros();
                for k in 0..6 {
                    xi[k] = delta[block + k];
                }
                let step = xi.norm();
                if step > max_step_norm {
                    max_step_norm = step;
                }
                let pose = self
                    .poses
                    .get_mut(&id)
                    .ok_or(PoseGraphError::MissingNode(id))?;
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
            }

            let cost_after = self.robust_se3_cost(&kernel);
            let step_accepted = match config.initial_lambda {
                None => true,
                Some(_) => cost_after < cost_before,
            };

            if !step_accepted {
                if let Some(saved) = saved_poses {
                    self.poses = saved;
                }
                lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                iterations.push(PoseGraphSe3IterationStats {
                    iteration,
                    cost_before,
                    cost_after,
                    max_step_norm,
                    lambda,
                    step_accepted: false,
                });
                if lambda >= config.max_lambda {
                    // λ saturated without finding a downhill step → bail.
                    break;
                }
                continue;
            }

            iterations.push(PoseGraphSe3IterationStats {
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

            if max_step_norm < config.step_tolerance {
                converged = true;
                break;
            }
            if (cost_before - cost_after).abs() < config.cost_tolerance {
                converged = true;
                break;
            }
        }

        Ok(PoseGraphSe3Result {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }

    /// Outlier-robust SE(3) pose-graph optimization via Graduated Non-Convexity
    /// (GNC; see [`crate::gnc`]). Use this instead of [`Self::optimize_se3_iterative`]
    /// when loop closures may be **wrong** (perceptual aliasing, place-recognition
    /// false positives): a single bad loop closure pulls a plain least-squares —
    /// or even a Huber/Cauchy IRLS — solve into a corrupted basin, whereas GNC
    /// anneals from a convex surrogate that trusts every edge to the true robust
    /// cost that rejects outliers, recovering the inlier trajectory.
    ///
    /// `config` supplies the same SE(3) LM settings as
    /// [`Self::optimize_se3_iterative`] (linear solver, `λ` schedule,
    /// tolerances, chordal seeding); its `robust_kernel` is ignored — GNC
    /// supersedes the M-estimator and the inner solve runs on the GNC-weighted
    /// least-squares cost. `gnc` selects the surrogate family, the inlier scale
    /// `c`, the `μ` annealing factor, the outer-level cap, and the number of
    /// inner LM iterations per level.
    ///
    /// Each outer level reweights every edge by its closed-form GNC weight at
    /// the current `μ`, runs a bounded weighted-LS solve, then sharpens `μ` one
    /// geometric step. The fill-reducing order and block-Cholesky symbolic
    /// factorization are `μ`-invariant and reused across all levels. The
    /// returned [`PoseGraphGncResult::edge_weights`] is the final per-edge
    /// inlier/outlier classification.
    pub fn optimize_se3_gnc(
        &mut self,
        config: &PoseGraphSe3Config,
        gnc: &gnc::GncConfig,
    ) -> Result<PoseGraphGncResult, PoseGraphError> {
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
        let dim = variable_count * 6;

        // GNC replaces the M-estimator, so the inner solve is plain weighted
        // least squares (kernel `None`); the GNC weights carry the robustness.
        let kernel = RobustKernel::None;
        // Report the pre-seed plain L2 cost so the reduction reflects the full
        // improvement, mirroring `optimize_se3_iterative`.
        let initial_cost = self.robust_se3_cost(&kernel);

        // Optional chordal rotation seeding (best-effort). GNC's first surrogate
        // is convex (trusts every edge), so seeding from the all-edge rotation
        // least squares is consistent with it; as μ sharpens, outliers are
        // down-weighted and the estimate moves off any seed they corrupted.
        if config.chordal_init
            && self
                .initialize_rotations_chordal(config.linear_solver)
                .is_ok()
        {
            let _ = self.optimize_translations_once_with(config.linear_solver);
        }

        // Convex first surrogate: initialize μ from the largest seeded residual.
        // The same seeded residuals optionally drive the MAD auto-estimate of
        // the inlier scale `c` (with `gnc.c` as a floor), so the inlier/outlier
        // boundary tracks this graph's noise level instead of a hand-set value.
        let squared_residuals = self.edge_squared_residuals();
        let s_max = squared_residuals.iter().copied().fold(0.0_f64, f64::max);
        let effective_gnc = match gnc.auto_scale {
            Some(k) => {
                let c = gnc::estimate_scale_mad(&squared_residuals, k)
                    .map_or(gnc.c, |est| est.max(gnc.c));
                gnc::GncConfig { c, ..*gnc }
            }
            None => *gnc,
        };
        let mut inlier_scale = effective_gnc.c;
        let mut state = gnc::GncState::new(&effective_gnc, s_max);
        let mut gnc_weights = vec![1.0_f64; self.edges.len()];

        // The sparsity pattern is μ-invariant, so the fill-reducing order and
        // block-Cholesky symbolic factorization are analyzed once and reused.
        let mut order_cache: Option<reordering::Reordering> = None;
        let mut symbolic_cache: Option<block_cholesky::BlockSymbolic> = None;

        let mut converged = false;
        let mut outer_iterations = 0usize;
        let max_outer = gnc.max_outer.max(1);

        for _ in 0..max_outer {
            outer_iterations += 1;
            // A level entered already-terminal is solving at the true robust
            // cost — run it, then stop (guarantees one solve at terminal μ).
            let terminal_level = state.is_terminal();

            // Black-Rangarajan weight update at the current μ.
            let residuals = self.edge_squared_residuals();
            // Adaptive inlier scale: re-derive `c` from the current residuals
            // each level (configured `c` as a floor). Level 0 reproduces the
            // one-shot estimate; later levels tighten as outliers are
            // suppressed and inlier residuals shrink.
            if gnc.auto_scale_readapt {
                if let Some(k) = gnc.auto_scale {
                    if let Some(est) = gnc::estimate_scale_mad(&residuals, k) {
                        let c = est.max(gnc.c);
                        state.set_inlier_scale(c);
                        inlier_scale = c;
                    }
                }
            }
            for (i, &s) in residuals.iter().enumerate() {
                gnc_weights[i] = state.weight(s);
            }

            // Inner weighted-LS solve at fixed weights (a few LM steps).
            let mut lambda = config.initial_lambda.unwrap_or(0.0);
            let mut current_cost = self.robust_se3_cost_weighted(&kernel, Some(&gnc_weights));
            for _ in 0..gnc.inner_iterations.max(1) {
                let (builder, g) = self.assemble_se3_system(
                    &node_index,
                    dim,
                    &kernel,
                    Some(&gnc_weights),
                    config.linear_solver,
                );
                // Marquardt damping scales by the (undamped) curvature, so capture
                // the diagonal of H before it is consumed by the solve.
                let diag = match config.damping {
                    DampingMode::Diagonal => Some(builder.diagonal()),
                    DampingMode::Identity => None,
                };
                let neg_g = -&g;
                let delta = builder.solve(
                    lambda,
                    diag.as_deref(),
                    &neg_g,
                    &mut order_cache,
                    &mut symbolic_cache,
                )?;

                let saved_poses = if config.initial_lambda.is_some() {
                    Some(self.poses.clone())
                } else {
                    None
                };
                let mut max_step_norm: f64 = 0.0;
                for (&id, &i) in &node_index {
                    let block = i * 6;
                    let mut xi = Vector6::<f64>::zeros();
                    for k in 0..6 {
                        xi[k] = delta[block + k];
                    }
                    max_step_norm = max_step_norm.max(xi.norm());
                    let pose = self
                        .poses
                        .get_mut(&id)
                        .ok_or(PoseGraphError::MissingNode(id))?;
                    pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
                }

                let cost_after = self.robust_se3_cost_weighted(&kernel, Some(&gnc_weights));
                let accepted = match config.initial_lambda {
                    None => true,
                    Some(_) => cost_after < current_cost,
                };
                if !accepted {
                    if let Some(saved) = saved_poses {
                        self.poses = saved;
                    }
                    lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                    if lambda >= config.max_lambda {
                        break;
                    }
                    continue;
                }
                current_cost = cost_after;
                if config.initial_lambda.is_some() {
                    lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
                }
                if max_step_norm < config.step_tolerance {
                    break;
                }
            }

            if terminal_level {
                converged = true;
                break;
            }
            state.anneal();
        }

        // Final per-edge classification at the converged estimate and μ.
        let residuals = self.edge_squared_residuals();
        for (i, &s) in residuals.iter().enumerate() {
            gnc_weights[i] = state.weight(s);
        }
        let final_cost = self.robust_se3_cost(&kernel);
        // Plain L2 over retained inliers: binarize the weights at the inlier
        // cutoff and reuse the weighted cost.
        const INLIER_THRESHOLD: f64 = 0.5;
        let inlier_mask: Vec<f64> = gnc_weights
            .iter()
            .map(|&w| if w >= INLIER_THRESHOLD { 1.0 } else { 0.0 })
            .collect();
        let inlier_cost = self.robust_se3_cost_weighted(&kernel, Some(&inlier_mask));

        Ok(PoseGraphGncResult {
            anchor_id,
            edge_count: self.edges.len(),
            variable_count,
            initial_cost,
            final_cost,
            inlier_cost,
            inlier_scale,
            outer_iterations,
            converged,
            edge_weights: gnc_weights,
        })
    }

    /// Recover the marginal covariance of every non-anchor pose from the
    /// information matrix `Λ = JᵀΩJ` assembled at the *current* estimate (run a
    /// solve first so this is the covariance at the optimum). Uses the
    /// Takahashi sparse-inverse recursion ([`crate::covariance`]) so the dense
    /// `Λ⁻¹` is never formed. Each `Matrix6` is the covariance of that pose in
    /// its local SE(3) tangent (the `[ω | ρ]` ordering of
    /// [`SE3::log`](visloc_core::geometry::SE3::log)); the gauge-fixed anchor
    /// has no free covariance and is omitted from the result.
    ///
    /// Useful for loop-closure gating (gate a candidate on the relative
    /// uncertainty between its endpoints) and uncertainty-aware fusion. Errors
    /// mirror the solvers: [`PoseGraphError::NoAnchor`] / `NoEdges` /
    /// `NoVariables`, and [`PoseGraphError::SingularSystem`] when `Λ` is not
    /// positive-definite (a rank-deficient / disconnected graph).
    pub fn pose_marginal_covariances(&self) -> Result<BTreeMap<u64, Matrix6<f64>>, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if !self.poses.contains_key(&anchor_id) {
            return Err(PoseGraphError::MissingNode(anchor_id));
        }
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }
        // Free-variable indexing (anchor excluded), identical to the solvers.
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
        let dim = variable_count * 6;

        // Assemble the *sparse* information matrix at the current estimate (plain
        // L2 — kernel `None`, no GNC weights) as COO triplets, in the same
        // free-variable order as `node_index`.
        let (builder, _g) = self.assemble_se3_system(
            &node_index,
            dim,
            &RobustKernel::None,
            None,
            LinearSolver::Sparse,
        );
        let (triplets, dim) = match builder {
            NormalEquations6::Sparse { triplets, dim } => (triplets, dim),
            NormalEquations6::Dense(_) => unreachable!("forced the sparse backend above"),
        };

        // Factor `Λ = L Lᵀ` at 6×6 block granularity in natural order (so block
        // column `idx` maps straight back to `node_index`), then recover the
        // per-pose marginal blocks with the block Takahashi recursion — `O(nnz)`,
        // no dense `Λ` or `Λ⁻¹` ([`crate::covariance::block_takahashi_diagonals`]).
        let (col_rows, col_vals, diag_inv) = block_cholesky::factor_blocks(&triplets, dim, 6)
            .map_err(|_| PoseGraphError::SingularSystem)?;
        let blocks = covariance::block_takahashi_diagonals(&col_rows, &col_vals, &diag_inv)
            .ok_or(PoseGraphError::SingularSystem)?;
        let mut out = BTreeMap::new();
        for (&id, &idx) in &node_index {
            let block = &blocks[idx];
            out.insert(id, Matrix6::from_fn(|r, c| block[(r, c)]));
        }
        Ok(out)
    }

    /// Covariance of the *relative* pose `a → b` implied by the current estimate
    /// — the joint marginal of the two pose blocks reduced to their difference
    /// (`Σ_aa + Σ_bb − Σ_ab − Σ_abᵀ`, the first-order tangent approximation; see
    /// [`covariance::relative_covariance`]). A gauge-fixed anchor endpoint
    /// contributes a zero block (its frame is certain), so the relative
    /// covariance to the anchor is just the other pose's marginal.
    ///
    /// This is the prediction covariance a loop-closure innovation is gated
    /// against: a candidate asserting a relative pose far outside this
    /// uncertainty (a confident-but-wrong place recognition between two
    /// well-localized frames) is statistically implausible. Recovers only the
    /// two block-columns `Σ[:, a]` and `Σ[:, b]` it needs by solving
    /// `Λ Y = [E_a | E_b]` with the sparse block Cholesky (back-substitution,
    /// `O(nnz(L))` per column) — the dense `Λ⁻¹` is never formed, so it stays
    /// cheap enough for the online gate. Errors mirror
    /// [`Self::pose_marginal_covariances`]; also
    /// [`PoseGraphError::MissingNode`] when `a` or `b` is absent.
    pub fn relative_pose_covariance(&self, a: u64, b: u64) -> Result<Matrix6<f64>, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if !self.poses.contains_key(&anchor_id) {
            return Err(PoseGraphError::MissingNode(anchor_id));
        }
        if !self.poses.contains_key(&a) {
            return Err(PoseGraphError::MissingNode(a));
        }
        if !self.poses.contains_key(&b) {
            return Err(PoseGraphError::MissingNode(b));
        }
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
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
        let dim = variable_count * 6;

        // Assemble the *sparse* normal equations (COO triplets) at the current
        // estimate, in the same free-variable order the solvers use.
        let (builder, _g) = self.assemble_se3_system(
            &node_index,
            dim,
            &RobustKernel::None,
            None,
            LinearSolver::Sparse,
        );
        let (triplets, dim) = match builder {
            NormalEquations6::Sparse { triplets, dim } => (triplets, dim),
            NormalEquations6::Dense(_) => unreachable!("forced the sparse backend above"),
        };

        // Recover only the two block-columns of `Σ = Λ⁻¹` we actually need —
        // columns `a` and `b` — by solving `Λ Y = [E_a | E_b]` with the block
        // Cholesky (back-substitution is `O(nnz(L))` per right-hand side),
        // instead of forming the dense inverse. `Y[:, 0..6] = Σ[:, a]` and
        // `Y[:, 6..12] = Σ[:, b]`, so every block we need (`Σaa`, `Σbb`, `Σab`,
        // `Σba`) is a slice of `Y`. A gauge-fixed anchor endpoint is absent from
        // the free variables, so its selector columns stay zero → a zero
        // covariance block (its frame is certain), reproducing the dense path.
        let ia = node_index.get(&a).copied();
        let ib = node_index.get(&b).copied();
        let mut selector = DMatrix::<f64>::zeros(dim, 12);
        if let Some(idx) = ia {
            for k in 0..6 {
                selector[(idx * 6 + k, k)] = 1.0;
            }
        }
        if let Some(idx) = ib {
            for k in 0..6 {
                selector[(idx * 6 + k, 6 + k)] = 1.0;
            }
        }
        let order = reordering::Reordering::fill_reducing(dim, 6, &triplets);
        let sigma_cols = solve_normal_equations_sparse_multi(&triplets, dim, 6, &selector, &order)?;

        // Read the four 6×6 blocks (anchor endpoint → zero block) and reduce to
        // the relative covariance `Σ_rel = Σaa + Σbb − Σab − Σba`.
        let read = |rows: Option<usize>, col0: usize| -> Matrix6<f64> {
            match rows {
                Some(idx) => Matrix6::from_fn(|r, c| sigma_cols[(idx * 6 + r, col0 + c)]),
                None => Matrix6::zeros(),
            }
        };
        let saa = read(ia, 0);
        let sba = read(ib, 0);
        let sab = read(ia, 6);
        let sbb = read(ib, 6);
        Ok(saa + sbb - sab - sba)
    }

    /// Marginal *information* (inverse-covariance) of the pose set `keep_ids`:
    /// the joint Gaussian information `Λ'` over those poses after marginalizing
    /// every other non-anchor pose out of the full information matrix
    /// `Λ = JᵀΩJ` (Schur complement, [`crate::marginalization::marginalize`]).
    /// The blocks are ordered as `keep_ids`, each a 6-DOF SE(3) tangent block
    /// in the same right-perturbation basis the solver uses
    /// (`T ← T ∘ exp(δ)`), so `Λ'` is exactly the prior a fixed-lag / sliding-
    /// window smoother re-adds when it drops the marginalized states. The
    /// gauge-fixed anchor is held (excluded from the variables), so `Λ'`
    /// encodes the kept poses' information relative to it.
    ///
    /// This is the information-form dual of [`Self::pose_marginal_covariances`]:
    /// `marginal_information([id])⁻¹` equals pose `id`'s marginal covariance,
    /// and the two-pose block inverts to the pair's joint covariance. (Assembles
    /// `Λ` densely and Schur-complements it, `O(n³)` — a building block for the
    /// windowed smoother, not a per-edge inner loop; the smoother itself will
    /// marginalize only the *leaving* states' Markov blanket to stay sparse.)
    /// Errors mirror [`Self::pose_marginal_covariances`], plus
    /// [`PoseGraphError::MissingNode`] for an absent or anchor `keep` id, and
    /// [`PoseGraphError::NoVariables`] when `keep_ids` is empty.
    pub fn marginal_information(&self, keep_ids: &[u64]) -> Result<DMatrix<f64>, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if !self.poses.contains_key(&anchor_id) {
            return Err(PoseGraphError::MissingNode(anchor_id));
        }
        if self.edges.is_empty() {
            return Err(PoseGraphError::NoEdges);
        }
        if keep_ids.is_empty() {
            return Err(PoseGraphError::NoVariables);
        }
        // Free-variable indexing (anchor excluded), identical to the solvers.
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
        let dim = variable_count * 6;

        // Scalar dimensions of the kept poses (each pose → its 6 tangent dims),
        // in `keep_ids` order — an anchor or absent id is an error.
        let mut keep_dims: Vec<usize> = Vec::with_capacity(keep_ids.len() * 6);
        for &id in keep_ids {
            let idx = node_index
                .get(&id)
                .copied()
                .ok_or(PoseGraphError::MissingNode(id))?;
            for k in 0..6 {
                keep_dims.push(idx * 6 + k);
            }
        }

        // Dense `Λ` at the current estimate (plain L2), then Schur-complement
        // the complement of `keep_dims` out. `η` is unused for the information
        // block, so pass zeros.
        let (builder, _g) = self.assemble_se3_system(
            &node_index,
            dim,
            &RobustKernel::None,
            None,
            LinearSolver::Dense,
        );
        let lambda = match builder {
            NormalEquations6::Dense(h) => h,
            NormalEquations6::Sparse { .. } => unreachable!("forced the dense backend above"),
        };
        let eta = DVector::<f64>::zeros(dim);
        let (lambda_prime, _eta_prime) = marginalization::marginalize(&lambda, &eta, &keep_dims)
            .ok_or(PoseGraphError::SingularSystem)?;
        Ok(lambda_prime)
    }

    /// Marginalize pose `id` out of the graph, replacing it with a dense
    /// [`GaussianPrior`] over its Markov blanket — the bounded-cost step of a
    /// fixed-lag / sliding-window smoother. The blanket `B` is the set of
    /// non-anchor poses sharing an edge or an existing prior with `id`. The new
    /// prior is the Schur complement of `id` out of the sub-system built from
    /// **only the `id`-incident factors** (edges and priors), so the factors
    /// *not* incident to `id` stay in the graph and their information is never
    /// double-counted; the prior carries both the curvature `Λ'` and the linear
    /// term `b'` (the incident factors' gradient at the current estimate), so the
    /// current estimate remains a stationary point of the reduced problem.
    ///
    /// Linearizes at the *current* estimate, so run a solve first (marginalizing
    /// at a converged estimate is exact to first order). Removes `id`'s pose and
    /// every edge/prior incident to it. A pose whose only neighbour is the anchor
    /// (empty blanket) is simply dropped (its information was purely relative to
    /// the fixed gauge). Errors: [`PoseGraphError::NoAnchor`];
    /// [`PoseGraphError::MissingNode`] if `id` is absent or is the anchor (not a
    /// free variable); [`PoseGraphError::SingularSystem`] if `id`'s information
    /// block is rank-deficient (an unconstrained pose cannot be marginalized).
    pub fn marginalize_pose(&mut self, id: u64) -> Result<(), PoseGraphError> {
        self.marginalize_pose_impl(id, false)
    }

    /// Like [`Self::marginalize_pose`], but **sparsify** the resulting blanket
    /// prior with the KL-optimal Chow-Liu tree
    /// ([`crate::sparsification::sparsify_chow_liu`]) instead of keeping the dense
    /// clique. The Schur complement couples every blanket pose to every other; the
    /// tree approximation keeps only the `N−1` strongest (highest mutual
    /// information) couplings, preserving each pose's marginal and the tree-edge
    /// pairwise marginals exactly, so a window that marginalizes repeatedly does
    /// not accumulate dense priors. Identical to the dense prior when the blanket
    /// has ≤ 2 poses (a 2-clique already *is* a tree). Same errors as
    /// [`Self::marginalize_pose`].
    pub fn marginalize_pose_sparsified(&mut self, id: u64) -> Result<(), PoseGraphError> {
        self.marginalize_pose_impl(id, true)
    }

    fn marginalize_pose_impl(&mut self, id: u64, sparsify: bool) -> Result<(), PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        if id == anchor_id || !self.poses.contains_key(&id) {
            return Err(PoseGraphError::MissingNode(id));
        }

        // Markov blanket: non-anchor poses sharing an edge or a prior with `id`.
        let mut blanket: HashSet<u64> = HashSet::new();
        for e in &self.edges {
            if e.from == id && e.to != anchor_id {
                blanket.insert(e.to);
            }
            if e.to == id && e.from != anchor_id {
                blanket.insert(e.from);
            }
        }
        for p in &self.priors {
            if p.ids.contains(&id) {
                for &pid in &p.ids {
                    if pid != id && pid != anchor_id {
                        blanket.insert(pid);
                    }
                }
            }
        }
        blanket.remove(&id);
        let mut b: Vec<u64> = blanket.into_iter().collect();
        b.sort_unstable();

        // Sub-system over {anchor, id} ∪ B carrying ONLY the id-incident factors,
        // assembled at the current estimate; `id` indexed first (its 6 dims are
        // the ones marginalized), B after (the kept dims).
        let mut sub = PoseGraph::new();
        sub.add_pose(anchor_id, self.poses[&anchor_id].clone());
        sub.anchor(anchor_id);
        sub.add_pose(id, self.poses[&id].clone());
        for &bid in &b {
            sub.add_pose(bid, self.poses[&bid].clone());
        }
        for e in &self.edges {
            if e.from == id || e.to == id {
                sub.edges.push(e.clone());
            }
        }
        for p in &self.priors {
            if p.ids.contains(&id) {
                sub.priors.push(p.clone());
            }
        }
        let mut node_index: BTreeMap<u64, usize> = BTreeMap::new();
        node_index.insert(id, 0);
        for (k, &bid) in b.iter().enumerate() {
            node_index.insert(bid, k + 1);
        }
        let dim = (1 + b.len()) * 6;
        let (builder, g_sub) = sub.assemble_se3_system(
            &node_index,
            dim,
            &RobustKernel::None,
            None,
            LinearSolver::Dense,
        );
        let h_sub = match builder {
            NormalEquations6::Dense(h) => h,
            NormalEquations6::Sparse { .. } => unreachable!("forced the dense backend above"),
        };

        // Drop `id` and its incident factors before re-attaching the prior.
        self.poses.remove(&id);
        self.edges.retain(|e| e.from != id && e.to != id);
        self.priors.retain(|p| !p.ids.contains(&id));

        if b.is_empty() {
            // The state only touched the fixed anchor — nothing to preserve.
            return Ok(());
        }

        // Keep B's dims (indices 6..), marginalize `id`'s dims (0..6).
        let keep_dims: Vec<usize> = (0..b.len())
            .flat_map(|k| (0..6).map(move |d| (k + 1) * 6 + d))
            .collect();
        let (lambda_prime, b_prime) = marginalization::marginalize(&h_sub, &g_sub, &keep_dims)
            .ok_or(PoseGraphError::SingularSystem)?;
        // Optionally sparsify the dense blanket clique to its Chow-Liu tree. The
        // prior is in the right-perturbation tangent basis at `linearization`;
        // sparsification is a pure transform of (Λ', b') that preserves the
        // minimizer e* = −Λ'⁻¹b' (so the linearization is unchanged). Falls back to
        // the dense prior if the tree build fails (a well-formed SPD prior won't).
        let (information, gradient) = if sparsify {
            match crate::sparsification::sparsify_chow_liu(&lambda_prime, &b_prime, 6) {
                Some(sp) => (sp.lambda, sp.eta),
                None => (lambda_prime, b_prime),
            }
        } else {
            (lambda_prime, b_prime)
        };
        let linearization: Vec<SE3> = b
            .iter()
            .map(|bid| self.poses[bid].world_to_camera.clone())
            .collect();
        self.priors.push(GaussianPrior {
            ids: b,
            information,
            gradient,
            linearization,
        });
        Ok(())
    }

    /// Sliding-window / fixed-lag driver: [`Self::marginalize_pose`] the oldest
    /// non-anchor poses (lowest ids) until at most `window_size` poses remain
    /// (the gauge-fixed anchor always among them), keeping the graph — and so the
    /// per-solve cost — bounded as keyframes accumulate. Returns the ids
    /// marginalized, in removal order (oldest first).
    ///
    /// Marginalize at a converged estimate (run a solve first): each step is
    /// then exact, and because the optimum stays stationary after each removal
    /// the chained marginalizations compose without drift — a windowed re-solve
    /// reproduces the batch estimate on the retained poses (a test asserts this).
    /// No-op when already within the window. Stops early if only the anchor is
    /// left. `window_size` is clamped to at least `1` (the anchor). Errors mirror
    /// [`Self::marginalize_pose`].
    pub fn marginalize_oldest(&mut self, window_size: usize) -> Result<Vec<u64>, PoseGraphError> {
        self.marginalize_oldest_impl(window_size, false)
    }

    /// Like [`Self::marginalize_oldest`], but **sparsify** each blanket prior with
    /// its Chow-Liu tree ([`Self::marginalize_pose_sparsified`]) so a long-running
    /// window does not accumulate dense priors as it slides — the bounded *and*
    /// sparse fixed-lag smoother. Same removal order and errors.
    pub fn marginalize_oldest_sparsified(
        &mut self,
        window_size: usize,
    ) -> Result<Vec<u64>, PoseGraphError> {
        self.marginalize_oldest_impl(window_size, true)
    }

    fn marginalize_oldest_impl(
        &mut self,
        window_size: usize,
        sparsify: bool,
    ) -> Result<Vec<u64>, PoseGraphError> {
        let anchor_id = self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        let target = window_size.max(1);
        let mut removed = Vec::new();
        while self.poses.len() > target {
            // Oldest non-anchor pose: the lowest id that is not the anchor
            // (`poses` is a `BTreeMap`, so `keys()` ascends).
            let Some(&oldest) = self.poses.keys().find(|&&id| id != anchor_id) else {
                break; // only the anchor remains
            };
            self.marginalize_pose_impl(oldest, sparsify)?;
            removed.push(oldest);
        }
        Ok(removed)
    }

    /// Merge another session's pose graph into this one, welding the two
    /// trajectories at a cross-session `bridge` constraint — the core of
    /// multi-session / multi-map SLAM (an ORB-SLAM3-Atlas-style weld). `other`'s
    /// nodes are relabeled by `id_offset` (which must exceed this graph's largest
    /// id so the two id spaces stay disjoint) and rigidly transformed into this
    /// graph's world frame, then its edges are spliced in and the `bridge` is
    /// added as a loop-closure edge. A joint [`Self::optimize_se3_iterative`]
    /// afterwards re-welds the seam.
    ///
    /// `bridge.from_keyframe_id` is a node of `self`; `bridge.to_keyframe_id` is a
    /// node of `other` (its *original*, pre-offset id); `bridge.relative_pose` is
    /// the measured relative pose `a → b` (the same `relative_world_to_camera`
    /// convention as every edge), e.g. from a cross-session place-recognition
    /// match. The alignment is computed so `other`'s bridge node lands exactly at
    /// the bridge prediction: each `other` pose `Tₙ` becomes `Tₙ ∘ (T_b⁻¹ ∘ z ∘
    /// T_a)`, a right-multiply under which every relative edge measurement is
    /// invariant (so `other`'s edges carry over unchanged, only relabeled).
    ///
    /// This graph's anchor is kept as the merged gauge. Errors:
    /// [`PoseGraphError::NoAnchor`] if either graph lacks an anchor;
    /// [`PoseGraphError::MissingNode`] if a bridge endpoint is absent or a
    /// relabeled id would collide with an existing node. `other` must carry no
    /// [`GaussianPrior`]s (a fresh session); priors are not transformed.
    pub fn merge_session(
        &mut self,
        other: &PoseGraph,
        id_offset: u64,
        bridge: &LoopClosureConstraint,
    ) -> Result<(), PoseGraphError> {
        self.anchor.ok_or(PoseGraphError::NoAnchor)?;
        other.anchor.ok_or(PoseGraphError::NoAnchor)?;
        let a = bridge.from_keyframe_id;
        let b = bridge.to_keyframe_id;
        let t_a = self
            .poses
            .get(&a)
            .ok_or(PoseGraphError::MissingNode(a))?
            .world_to_camera
            .clone();
        let t_b = other
            .poses
            .get(&b)
            .ok_or(PoseGraphError::MissingNode(b))?
            .world_to_camera
            .clone();

        // Right-multiply alignment: Tₙ_new = Tₙ ∘ (T_b⁻¹ ∘ z ∘ T_a). For the
        // bridge node b this gives z ∘ T_a, i.e. exactly the bridge prediction.
        let rhs = t_b.inverse().compose(&bridge.relative_pose.compose(&t_a));

        // Splice the relabeled, transformed nodes (id collision is an error).
        for (&id, pose) in &other.poses {
            let new_id = id + id_offset;
            if self.poses.contains_key(&new_id) {
                return Err(PoseGraphError::MissingNode(new_id));
            }
            self.poses.insert(
                new_id,
                Pose {
                    world_to_camera: pose.world_to_camera.compose(&rhs),
                },
            );
        }
        // Edge measurements are invariant under the right-multiply, so carry them
        // over verbatim, only relabeling endpoints.
        for edge in &other.edges {
            self.edges.push(PoseGraphEdge {
                from: edge.from + id_offset,
                to: edge.to + id_offset,
                measurement: edge.measurement.clone(),
                kind: edge.kind,
                weight: edge.weight,
                information: edge.information,
            });
        }
        // The weld: a loop-closure edge a → (b + offset) carrying the bridge.
        self.edges.push(PoseGraphEdge {
            from: a,
            to: b + id_offset,
            measurement: bridge.relative_pose.clone(),
            kind: PoseGraphEdgeKind::LoopClosure,
            weight: (bridge.inlier_count as f64).max(1.0),
            information: None,
        });
        Ok(())
    }

    /// Screen a set of candidate cross-session bridges for mutual consistency
    /// before merging, returning the indices (into `candidates`) of the maximum
    /// mutually-consistent subset — PCM ([`crate::pcm::maximum_consistent_set`])
    /// applied across two sessions, the front-end guard against a wrong
    /// cross-session place-recognition match.
    ///
    /// `candidates[i].from_keyframe_id` is a node of `self`,
    /// `candidates[i].to_keyframe_id` a node of `other` (its original, pre-offset
    /// id), `relative_pose` the measured `a → b`. Consistency is the PCM cycle
    /// `a_k →z_k→ b_k →(other odometry)→ b_l →z_l⁻¹→ a_l →(self odometry)→ a_k ≈
    /// I`, evaluated over a *combined* odometry map (this graph's poses ∪
    /// `other`'s relabeled by `id_offset`); each leg is a frame-invariant relative
    /// pose, so the cycle is well-defined even though the two sessions live in
    /// different world frames.
    ///
    /// Use `cfg.require_individual = false`: the individual odometry self-check is
    /// meaningless across sessions (a self node and an other node have no
    /// single-session relative pose), so screening rests on pairwise consistency
    /// and needs ≥ 2 candidates to bite. A lone candidate is always kept.
    pub fn consistent_session_bridges(
        &self,
        other: &PoseGraph,
        id_offset: u64,
        candidates: &[LoopClosureConstraint],
        cfg: &pcm::PcmConfig,
    ) -> Vec<usize> {
        if candidates.len() <= 1 {
            return (0..candidates.len()).collect();
        }
        let mut odometry: BTreeMap<u64, SE3> = BTreeMap::new();
        for (&id, p) in &self.poses {
            odometry.insert(id, p.world_to_camera.clone());
        }
        for (&id, p) in &other.poses {
            odometry.insert(id + id_offset, p.world_to_camera.clone());
        }
        let measurements: Vec<pcm::LoopMeasurement> = candidates
            .iter()
            .map(|c| pcm::LoopMeasurement {
                from: c.from_keyframe_id,
                to: c.to_keyframe_id + id_offset,
                relative: c.relative_pose.clone(),
            })
            .collect();
        pcm::maximum_consistent_set(&measurements, &odometry, cfg)
    }

    /// Serialize this pose graph to a plain-text format. The format is
    /// line-oriented and human-readable so it doubles as a debug dump:
    ///
    /// ```text
    /// # visloc-rs PoseGraph v1
    /// P <id> <qw> <qx> <qy> <qz> <tx> <ty> <tz>
    /// ...
    /// A <id>
    /// E <from> <to> <kind:0|1> <weight> <qw> <qx> <qy> <qz> <tx> <ty> <tz>
    /// ...
    /// ```
    ///
    /// `kind = 0` is `Sequential`, `kind = 1` is `LoopClosure`. Lines
    /// starting with `#` and blank lines are ignored on read. Round-trips
    /// through [`Self::load_text`] without precision loss within `f64`'s
    /// `{:.17e}` representation.
    pub fn save_text(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let mut text = String::from("# visloc-rs PoseGraph v1\n");
        for (id, pose) in self.poses.iter() {
            let q = pose.world_to_camera.rotation.into_inner();
            let t = &pose.world_to_camera.translation;
            text.push_str(&format!(
                "P {id} {qw:.17e} {qx:.17e} {qy:.17e} {qz:.17e} {tx:.17e} {ty:.17e} {tz:.17e}\n",
                qw = q.w,
                qx = q.i,
                qy = q.j,
                qz = q.k,
                tx = t.x,
                ty = t.y,
                tz = t.z,
            ));
        }
        if let Some(anchor) = self.anchor {
            text.push_str(&format!("A {anchor}\n"));
        }
        for edge in &self.edges {
            let kind: u8 = match edge.kind {
                PoseGraphEdgeKind::Sequential => 0,
                PoseGraphEdgeKind::LoopClosure => 1,
            };
            let q = edge.measurement.rotation.into_inner();
            let t = &edge.measurement.translation;
            text.push_str(&format!(
                "E {from} {to} {kind} {weight:.17e} {qw:.17e} {qx:.17e} {qy:.17e} {qz:.17e} {tx:.17e} {ty:.17e} {tz:.17e}\n",
                from = edge.from, to = edge.to, weight = edge.weight,
                qw = q.w, qx = q.i, qy = q.j, qz = q.k, tx = t.x, ty = t.y, tz = t.z,
            ));
        }
        std::fs::write(path, text)
    }

    /// Inverse of [`Self::save_text`]. Returns `PoseGraphParseError`
    /// on syntactic problems (unknown line tag, missing column, bad
    /// number, unrecognised kind tag).
    pub fn load_text(path: impl AsRef<std::path::Path>) -> Result<Self, PoseGraphParseError> {
        let text = std::fs::read_to_string(path).map_err(PoseGraphParseError::Io)?;
        let mut graph = PoseGraph::new();
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut tok = line.split_ascii_whitespace();
            let tag = tok.next().ok_or_else(|| PoseGraphParseError::Syntax {
                line: lineno + 1,
                reason: "empty line tag".to_string(),
            })?;
            match tag {
                "P" => {
                    let id = parse_field::<u64>(tok.next(), lineno, "id")?;
                    let qw = parse_field::<f64>(tok.next(), lineno, "qw")?;
                    let qx = parse_field::<f64>(tok.next(), lineno, "qx")?;
                    let qy = parse_field::<f64>(tok.next(), lineno, "qy")?;
                    let qz = parse_field::<f64>(tok.next(), lineno, "qz")?;
                    let tx = parse_field::<f64>(tok.next(), lineno, "tx")?;
                    let ty = parse_field::<f64>(tok.next(), lineno, "ty")?;
                    let tz = parse_field::<f64>(tok.next(), lineno, "tz")?;
                    let rot = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
                    let pose = Pose::from_world_to_camera(rot, Vector3::new(tx, ty, tz));
                    graph.poses.insert(id, pose);
                }
                "A" => {
                    let id = parse_field::<u64>(tok.next(), lineno, "anchor id")?;
                    graph.anchor = Some(id);
                }
                "E" => {
                    let from = parse_field::<u64>(tok.next(), lineno, "from")?;
                    let to = parse_field::<u64>(tok.next(), lineno, "to")?;
                    let kind_tag = parse_field::<u8>(tok.next(), lineno, "kind")?;
                    let kind = match kind_tag {
                        0 => PoseGraphEdgeKind::Sequential,
                        1 => PoseGraphEdgeKind::LoopClosure,
                        other => {
                            return Err(PoseGraphParseError::Syntax {
                                line: lineno + 1,
                                reason: format!("unrecognised edge kind tag {other}"),
                            });
                        }
                    };
                    let weight = parse_field::<f64>(tok.next(), lineno, "weight")?;
                    let qw = parse_field::<f64>(tok.next(), lineno, "qw")?;
                    let qx = parse_field::<f64>(tok.next(), lineno, "qx")?;
                    let qy = parse_field::<f64>(tok.next(), lineno, "qy")?;
                    let qz = parse_field::<f64>(tok.next(), lineno, "qz")?;
                    let tx = parse_field::<f64>(tok.next(), lineno, "tx")?;
                    let ty = parse_field::<f64>(tok.next(), lineno, "ty")?;
                    let tz = parse_field::<f64>(tok.next(), lineno, "tz")?;
                    let rot = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
                    graph.edges.push(PoseGraphEdge {
                        from,
                        to,
                        measurement: SE3::new(rot, Vector3::new(tx, ty, tz)),
                        kind,
                        weight,
                        information: None,
                    });
                }
                other => {
                    return Err(PoseGraphParseError::Syntax {
                        line: lineno + 1,
                        reason: format!("unknown line tag '{other}'"),
                    });
                }
            }
        }
        Ok(graph)
    }
}

#[derive(Debug)]
pub enum PoseGraphParseError {
    Io(std::io::Error),
    Syntax { line: usize, reason: String },
}

impl std::fmt::Display for PoseGraphParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoseGraphParseError::Io(e) => write!(f, "I/O error reading pose graph: {e}"),
            PoseGraphParseError::Syntax { line, reason } => {
                write!(f, "pose graph parse error at line {line}: {reason}")
            }
        }
    }
}

impl std::error::Error for PoseGraphParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PoseGraphParseError::Io(e) => Some(e),
            PoseGraphParseError::Syntax { .. } => None,
        }
    }
}

fn parse_field<T: std::str::FromStr>(
    field: Option<&str>,
    lineno: usize,
    name: &str,
) -> Result<T, PoseGraphParseError>
where
    T::Err: std::fmt::Display,
{
    let s = field.ok_or_else(|| PoseGraphParseError::Syntax {
        line: lineno + 1,
        reason: format!("missing field '{name}'"),
    })?;
    s.parse::<T>().map_err(|e| PoseGraphParseError::Syntax {
        line: lineno + 1,
        reason: format!("bad {name}: {e}"),
    })
}

/// Solve `H · x = b` preferring Cholesky (SPD path) and falling back to LU
/// for ill-conditioned or rank-deficient systems.
pub(crate) fn solve_normal_equations(
    h: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, PoseGraphError> {
    if let Some(chol) = h.clone().cholesky() {
        return Ok(chol.solve(b));
    }
    h.clone()
        .lu()
        .solve(b)
        .ok_or(PoseGraphError::SingularSystem)
}

/// Sparse Cholesky solve of `(H + λI) · x = b` where `H` is supplied as
/// COO triplets and assumed SPD by construction (it is a sum of `wᵀw` block
/// outer products from edge Jacobians). Triplets may contain duplicates;
/// they are summed during the COO → CSC conversion.
///
/// The system is solved in the fill-reducing variable order carried by `order`
/// (see the `reordering` module), applied as a symmetric permutation. That keeps
/// the Cholesky factor near-banded and prevents the catastrophic fill-in that
/// makes poorly-ordered or intrinsically wide 3D pose graphs (e.g.
/// `torus`/`sphere`) intractable. The permutation is purely structural and
/// deterministic, so the returned solution is unchanged up to floating-point
/// summation order. The ordering depends only on the sparsity pattern, so
/// callers compute it once and reuse it across iterations.
///
/// The factorization itself is the block Cholesky (see [`block_cholesky`]):
/// `block_size` is the variable-block dimension (6 for SE(3) poses), and the
/// permuted system keeps those blocks contiguous, so the factor runs on dense
/// `B×B` kernels instead of scalar columns.
///
/// Returns [`PoseGraphError::SingularSystem`] when the factorization fails
/// (e.g., disconnected graph). The damping term `λ` is added to the diagonal
/// before factoring, matching the dense LM path's `H + λI` formulation.
fn solve_normal_equations_sparse(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    block_size: usize,
    b: &DVector<f64>,
    lambda: f64,
    order: &reordering::Reordering,
    symbolic_cache: &mut Option<block_cholesky::BlockSymbolic>,
) -> Result<DVector<f64>, PoseGraphError> {
    let permuted = order.permute_triplets(triplets);
    let rhs_permuted = order.permute_rhs(b);
    let rhs = DMatrix::from_column_slice(dim, 1, rhs_permuted.as_slice());
    // The permuted sparsity pattern is identical across LM iterations, so the
    // block-Cholesky symbolic analysis (and the COO→block assembly) is cached
    // and only the numeric refactorization runs after the first solve.
    let solution = block_cholesky::solve_spd_block_cached(
        symbolic_cache,
        &permuted,
        dim,
        block_size,
        &rhs,
        lambda,
    )
    .map_err(|_| PoseGraphError::SingularSystem)?;
    let solution_permuted = DVector::from_column_slice(solution.as_slice());
    Ok(order.restore_solution(&solution_permuted))
}

/// Multi-right-hand-side variant of [`solve_normal_equations_sparse`]: factor
/// the SPD matrix once and solve every column of `rhs` against it. The chordal
/// rotation initializer assembles one normal matrix shared by all three
/// rotation columns (`block_size` 3), so a single block factorization amortizes
/// over the three solves. The fill-reducing `order` is applied as a symmetric
/// permutation exactly as in the single-RHS path; each column is permuted in,
/// solved, and restored.
fn solve_normal_equations_sparse_multi(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    block_size: usize,
    rhs: &DMatrix<f64>,
    order: &reordering::Reordering,
) -> Result<DMatrix<f64>, PoseGraphError> {
    let permuted = order.permute_triplets(triplets);

    // Permute every right-hand side into the reordered space, factor once via
    // the block Cholesky, and restore each solved column.
    let cols = rhs.ncols();
    let mut rhs_permuted = DMatrix::<f64>::zeros(dim, cols);
    for c in 0..cols {
        let column = DVector::from_column_slice(rhs.column(c).as_slice());
        rhs_permuted.set_column(c, &order.permute_rhs(&column));
    }
    let solved = block_cholesky::solve_spd_block(&permuted, dim, block_size, &rhs_permuted, 0.0)
        .map_err(|_| PoseGraphError::SingularSystem)?;

    let mut out = DMatrix::<f64>::zeros(dim, cols);
    for c in 0..cols {
        let solved_permuted = DVector::from_column_slice(solved.column(c).as_slice());
        out.set_column(c, &order.restore_solution(&solved_permuted));
    }
    Ok(out)
}

fn add_block6(h: &mut DMatrix<f64>, row: usize, col: usize, weight: f64, block: &Matrix6<f64>) {
    for r in 0..6 {
        for c in 0..6 {
            h[(row + r, col + c)] += weight * block[(r, c)];
        }
    }
}

fn add_segment6(g: &mut DVector<f64>, start: usize, weight: f64, v: &Vector6<f64>) {
    for k in 0..6 {
        g[start + k] += weight * v[k];
    }
}

/// Add `value · I_3` to the `(start, start)` diagonal block of either a
/// dense `H` matrix or a triplet vector. Used by the translation-only
/// optimizer where the per-edge contribution to `A^T A` is a scaled
/// identity on the diagonal blocks.
fn add_diag_block3(
    h_dense: &mut Option<DMatrix<f64>>,
    triplets: &mut Vec<(usize, usize, f64)>,
    start: usize,
    value: f64,
) {
    if let Some(h) = h_dense {
        for k in 0..3 {
            h[(start + k, start + k)] += value;
        }
    } else {
        for k in 0..3 {
            triplets.push((start + k, start + k, value));
        }
    }
}

/// Add `value · I_3` to the `(row_start, col_start)` off-diagonal block
/// (off-diagonal in the block-of-3 sense — also used when row != col).
fn add_offdiag_block3(
    h_dense: &mut Option<DMatrix<f64>>,
    triplets: &mut Vec<(usize, usize, f64)>,
    row_start: usize,
    col_start: usize,
    value: f64,
) {
    if let Some(h) = h_dense {
        for k in 0..3 {
            h[(row_start + k, col_start + k)] += value;
        }
    } else {
        for k in 0..3 {
            triplets.push((row_start + k, col_start + k, value));
        }
    }
}

/// Add a full (possibly dense) 3×3 `block` at the `(row_start, col_start)`
/// position of either a dense `H` or a triplet vector. Used by the chordal
/// rotation initializer, whose off-diagonal coupling `-w·R_meas` is a dense
/// rotation matrix rather than a scaled identity. Zero entries are skipped in
/// the sparse path so the rotation matrices contribute only their nonzeros.
fn add_dense_block3(
    h_dense: &mut Option<DMatrix<f64>>,
    triplets: &mut Vec<(usize, usize, f64)>,
    row_start: usize,
    col_start: usize,
    block: &Matrix3<f64>,
) {
    if let Some(h) = h_dense {
        for r in 0..3 {
            for c in 0..3 {
                h[(row_start + r, col_start + c)] += block[(r, c)];
            }
        }
    } else {
        for r in 0..3 {
            for c in 0..3 {
                let v = block[(r, c)];
                if v != 0.0 {
                    triplets.push((row_start + r, col_start + c, v));
                }
            }
        }
    }
}

/// Scalar edge weight used by the chordal rotation initializer. For an
/// isotropic edge this is just `edge.weight`; for an edge carrying a full 6×6
/// information matrix `Ω` (ordered `[ρ; ω]`) it is the mean of the rotational
/// diagonal `(Ω₃₃ + Ω₄₄ + Ω₅₅)/3`, i.e. the confidence g2o assigned to the
/// rotation block. Negative or non-finite results are clamped to a tiny
/// positive weight so the relaxed normal matrix stays positive definite.
fn chordal_rotation_weight(edge: &PoseGraphEdge) -> f64 {
    let w = match &edge.information {
        Some(omega) => (omega[(3, 3)] + omega[(4, 4)] + omega[(5, 5)]) / 3.0,
        None => edge.weight,
    };
    if w.is_finite() && w > 0.0 {
        w
    } else {
        1e-9
    }
}

/// Project a 3×3 matrix onto the nearest rotation in `SO(3)` (Frobenius sense)
/// via its SVD: `R = U·diag(1, 1, det(UVᵀ))·Vᵀ`. The determinant correction
/// guarantees `det(R) = +1` (a proper rotation, never a reflection). Returns
/// `None` only when the SVD fails to converge.
fn project_to_so3(m: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let svd = m.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        // Flip the sign of the smallest singular direction (the last column of
        // U) to turn the reflection into a proper rotation.
        let mut u_fixed = u;
        for k in 0..3 {
            u_fixed[(k, 2)] = -u_fixed[(k, 2)];
        }
        r = u_fixed * v_t;
    }
    Some(r)
}

/// Storage for the SE(3) Gauss-Newton normal-equations matrix `H` that
/// dispatches to either a dense [`DMatrix`] or a COO triplet vector backing
/// a sparse Cholesky solve. The right-hand side `g` is assembled separately
/// (see callers) so this builder stays focused on `H`.
enum NormalEquations6 {
    Dense(DMatrix<f64>),
    Sparse {
        triplets: Vec<(usize, usize, f64)>,
        dim: usize,
    },
}

impl NormalEquations6 {
    fn new(dim: usize, solver: LinearSolver, edge_hint: usize) -> Self {
        match solver {
            LinearSolver::Dense => Self::Dense(DMatrix::zeros(dim, dim)),
            // Each edge contributes up to four 6×6 blocks = 4·36 = 144 entries.
            LinearSolver::Sparse => Self::Sparse {
                triplets: Vec::with_capacity(edge_hint * 144),
                dim,
            },
        }
    }

    fn add_block6(&mut self, row: usize, col: usize, weight: f64, block: &Matrix6<f64>) {
        match self {
            Self::Dense(h) => add_block6(h, row, col, weight, block),
            Self::Sparse { triplets, .. } => {
                for r in 0..6 {
                    for c in 0..6 {
                        triplets.push((row + r, col + c, weight * block[(r, c)]));
                    }
                }
            }
        }
    }

    /// Solve the assembled system. For the sparse backend the fill-reducing
    /// ordering is computed once into `order_cache` (the sparsity pattern is
    /// identical across LM iterations) and reused on subsequent calls.
    /// The diagonal of `H` (length `dim`), in the natural (unpermuted) variable
    /// order. Used to build the Marquardt damping `λ·diag(H)` and the gain-ratio
    /// predicted reduction.
    fn diagonal(&self) -> Vec<f64> {
        match self {
            Self::Dense(h) => (0..h.nrows()).map(|k| h[(k, k)]).collect(),
            Self::Sparse { triplets, dim } => {
                let mut diag = vec![0.0; *dim];
                for &(r, c, v) in triplets {
                    if r == c {
                        diag[r] += v;
                    }
                }
                diag
            }
        }
    }

    /// Solve `(H + D) δ = -g` where the damping `D` is either `λI`
    /// (`diag_scale = None`) or `λ·diag(H)` (`diag_scale = Some(diag)`, the
    /// per-variable curvature from [`Self::diagonal`]). With `diag_scale = None`
    /// and a given `lambda` this is bit-identical to the original solve.
    fn solve(
        self,
        lambda: f64,
        diag_scale: Option<&[f64]>,
        neg_g: &DVector<f64>,
        order_cache: &mut Option<reordering::Reordering>,
        symbolic_cache: &mut Option<block_cholesky::BlockSymbolic>,
    ) -> Result<DVector<f64>, PoseGraphError> {
        match self {
            Self::Dense(mut h) => {
                let dim = h.nrows();
                match diag_scale {
                    // Marquardt: H + λ·diag(H). Read the original diagonal from
                    // `diag` (captured before this in-place update) so the scaling
                    // is by the undamped curvature.
                    Some(diag) => {
                        for k in 0..dim {
                            h[(k, k)] += lambda * diag[k];
                        }
                    }
                    None if lambda > 0.0 => {
                        for k in 0..dim {
                            h[(k, k)] += lambda;
                        }
                    }
                    None => {}
                }
                solve_normal_equations(&h, neg_g)
            }
            Self::Sparse { mut triplets, dim } => {
                // For Marquardt damping, fold λ·diag(H) into the triplets as extra
                // diagonal entries and let the factorizer run undamped (λ = 0); the
                // sparsity pattern is unchanged (diagonal blocks are always
                // present), so the cached symbolic analysis still applies.
                let factor_lambda = match diag_scale {
                    Some(diag) => {
                        triplets.reserve(dim);
                        for (k, &d) in diag.iter().enumerate().take(dim) {
                            triplets.push((k, k, lambda * d));
                        }
                        0.0
                    }
                    None => lambda,
                };
                let order = order_cache.get_or_insert_with(|| {
                    reordering::Reordering::fill_reducing(dim, 6, &triplets)
                });
                solve_normal_equations_sparse(
                    &triplets,
                    dim,
                    6,
                    neg_g,
                    factor_lambda,
                    order,
                    symbolic_cache,
                )
            }
        }
    }
}

/// Compute the relative SE3 `previous_to_current` such that
/// `to_pose.world_to_camera == relative * from_pose.world_to_camera`. This is
/// the same convention used by [`PoseGraphEdge::measurement`].
pub fn relative_world_to_camera(from_pose: &Pose, to_pose: &Pose) -> SE3 {
    to_pose
        .world_to_camera
        .compose(&from_pose.world_to_camera.inverse())
}
/// Translation-only constraint on camera centers in world coordinates implied
/// by `measurement` together with `to_pose`'s rotation: `c_to - c_from`
/// equals this displacement.
fn expected_world_displacement(to_pose: &Pose, measurement: &SE3) -> nalgebra::Vector3<f64> {
    let rotation_matrix = to_pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    -(rotation_matrix.transpose() * measurement.translation)
}
