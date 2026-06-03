//! iSAM-style incremental SE(3) pose-graph smoother.
//!
//! The batch [`PoseGraph::optimize_se3_iterative`](crate::PoseGraph::optimize_se3_iterative)
//! re-analyzes and refactors the *whole* normal matrix every time the graph
//! changes — `O(n)` per keyframe, `O(n²)` to build a trajectory online. This
//! smoother instead maintains the factorization across keyframes: each new
//! keyframe grows the factor by one block variable
//! ([`BlockSymbolic::append_variable`](crate::block_cholesky)) without re-analyzing,
//! and each Gauss–Newton relinearization re-factors only the columns that changed
//! and their elimination-tree ancestor paths
//! ([`refactor_incremental`](crate::block_cholesky)). It is the driver on top of
//! [`BlockIncrementalSolver`].
//!
//! Natural variable order (no fill-reducing permutation), so a new keyframe is
//! always the highest index — a cheap top-of-tree edit. Gauss–Newton, not
//! Levenberg–Marquardt: an `λ` that changed every iteration would perturb every
//! diagonal block and defeat the incrementality (a good odometry initialization
//! makes GN well-behaved here). The smoother therefore suits the online VO/SLAM
//! regime — a growing trajectory with occasional loop closures — where a chain
//! append touches only the new pose while a loop closure's correction propagates
//! over the span it spans; the anchor pins the gauge exactly as in the batch path.
//! Validated on real KITTI 00: the incremental trajectory matches a from-scratch
//! batch Gauss–Newton pose-for-pose.
//!
//! ## Relinearization and its limit
//!
//! After a Gauss–Newton step the columns whose normal-matrix block changed — the
//! poses that moved plus their edge neighbours (an edge's curvature block depends
//! on its source pose and lands on both endpoints) — are recomputed at the new
//! estimate and their ancestor paths refactored; the rest of the factor is reused.
//!
//! [`IncrementalSmootherConfig::relin_threshold`] can in principle skip poses that
//! moved less than it, to keep a far, settled span out of the refactor. **But this
//! is only sound at (or very near) `0`**: the linear system is assembled at the
//! *current* poses for every variable, so reusing a column's old factor while its
//! pose has meaningfully moved mixes two linearization points and can make a
//! Schur-complement block indefinite (a `SingularSystem` on real data). A
//! genuinely large, safe threshold needs per-variable linearization points (assemble
//! each edge at its endpoints' *frozen* linearization poses, relinearize only the
//! chosen variables) — the iSAM2 mechanism, not yet built here. The default is
//! therefore `0.0` (exact: every moved column relinearizes, matching batch), and a
//! positive value is an experimental approximation guarded by this caveat.

use std::collections::BTreeMap;

use nalgebra::{DVector, Matrix6, Vector6};
use visloc_core::geometry::{Pose, SE3};

use crate::block_cholesky::BlockIncrementalSolver;
use crate::{PoseGraph, PoseGraphEdge, PoseGraphError};

/// Tuning for the incremental smoother's inner Gauss–Newton loop.
#[derive(Debug, Clone, Copy)]
pub struct IncrementalSmootherConfig {
    /// Skip relinearizing a pose whose step tangent norm is below this. **Keep at
    /// (or very near) `0.0`** — see the module docs: the system is assembled at the
    /// current poses, so skipping a meaningfully-moved pose mixes linearization
    /// points and can yield an indefinite block. A large, safe value needs
    /// per-variable linearization points (not yet implemented).
    pub relin_threshold: f64,
    /// Inner loop converges when the largest pose step falls below this.
    pub step_tolerance: f64,
    /// Hard cap on inner Gauss–Newton iterations per keyframe.
    pub max_inner_iters: usize,
}

impl Default for IncrementalSmootherConfig {
    fn default() -> Self {
        Self {
            relin_threshold: 0.0,
            step_tolerance: 1e-6,
            max_inner_iters: 10,
        }
    }
}

/// What one [`IncrementalPoseGraph::add_keyframe`] did — for measurement.
#[derive(Debug, Clone, Default)]
pub struct IncrementalUpdateStats {
    /// Inner Gauss–Newton iterations run.
    pub inner_iters: usize,
    /// Incremental refactors performed (one per relinearization).
    pub refactors: usize,
    /// Sum of relinearized (moved) columns across the inner iterations — the
    /// work the fluid-relinearization gate admitted versus the full `O(n)`.
    pub relinearized_columns: usize,
    /// Whether the inner loop reached the step tolerance.
    pub converged: bool,
    /// Seconds spent assembling the normal matrix / gradient (profiling).
    pub assemble_secs: f64,
    /// Seconds spent in the linear solve / back-substitution (profiling).
    pub solve_secs: f64,
    /// Seconds spent growing/refactoring the block factor (profiling).
    pub factor_secs: f64,
}

/// An SE(3) pose graph whose factorization is maintained incrementally as
/// keyframes are appended. See the module docs.
pub struct IncrementalPoseGraph {
    /// The underlying graph (poses, edges, anchor) — public for inspection,
    /// metric evaluation, and handoff to the batch solver.
    pub graph: PoseGraph,
    config: IncrementalSmootherConfig,
    /// Variable index per pose id (natural order, anchor excluded).
    node_index: BTreeMap<u64, usize>,
    /// Pose id per variable index (the inverse of `node_index`), so a column can
    /// be re-assembled from its pose's incident edges.
    column_id: Vec<u64>,
    /// `incident[pid]`: indices into `graph.edges` of every edge touching pose
    /// `pid` — the edges a column re-assembles from.
    incident: BTreeMap<u64, Vec<usize>>,
    /// `col_h[c]`: column `c`'s lower-triangular normal-matrix block values
    /// (diagonal + below-diagonal cross blocks), maintained incrementally so only
    /// the columns whose pose moved are re-assembled rather than the whole system.
    col_h: Vec<Vec<(usize, usize, f64)>>,
    /// The maintained gradient `g` (one 6-block per variable, flat), updated only
    /// on the columns that changed.
    g: Vec<f64>,
    /// The maintained block factor (`None` until the first keyframe).
    solver: Option<BlockIncrementalSolver<6>>,
}

impl IncrementalPoseGraph {
    /// Start a smoother anchored at `anchor_id` (gauge-fixed at `anchor_pose`).
    pub fn new(anchor_id: u64, anchor_pose: Pose, config: IncrementalSmootherConfig) -> Self {
        let mut graph = PoseGraph::new();
        graph.add_pose(anchor_id, anchor_pose);
        graph.anchor(anchor_id);
        Self {
            graph,
            config,
            node_index: BTreeMap::new(),
            column_id: Vec::new(),
            incident: BTreeMap::new(),
            col_h: Vec::new(),
            g: Vec::new(),
            solver: None,
        }
    }

    /// Number of optimized (non-anchor) variables.
    pub fn variable_count(&self) -> usize {
        self.node_index.len()
    }

    /// Append a new keyframe `id` (must be the highest id so far) at initial
    /// estimate `pose`, coupled to existing poses by `edges` (each edge has one
    /// endpoint `== id`, the other an existing pose — the anchor or a variable),
    /// then incrementally re-optimize. Returns what the update did.
    pub fn add_keyframe(
        &mut self,
        id: u64,
        pose: Pose,
        edges: Vec<PoseGraphEdge>,
    ) -> Result<IncrementalUpdateStats, PoseGraphError> {
        if self.graph.poses.contains_key(&id) {
            return Err(PoseGraphError::MissingNode(id));
        }
        for e in &edges {
            let other = if e.from == id {
                e.to
            } else if e.to == id {
                e.from
            } else {
                // Neither endpoint is the new keyframe.
                return Err(PoseGraphError::MissingNode(id));
            };
            if other != id && !self.graph.poses.contains_key(&other) {
                return Err(PoseGraphError::MissingNode(other));
            }
        }

        let new_index = self.node_index.len();
        self.graph.poses.insert(id, pose);
        self.node_index.insert(id, new_index);
        self.column_id.push(id);
        self.col_h.push(Vec::new());
        self.g.extend_from_slice(&[0.0; 6]);

        // Register the new edges in the incidence index, collect the new variable's
        // couplings, and note which existing columns the new constraints dirty (the
        // new pose plus each direct neighbour, whose H/g blocks the new edges touch).
        let edge_start = self.graph.edges.len();
        self.graph.edges.extend(edges);
        let mut edges_to: Vec<usize> = Vec::new();
        let mut touched: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        touched.insert(id);
        for ei in edge_start..self.graph.edges.len() {
            let (from, to) = {
                let e = &self.graph.edges[ei];
                (e.from, e.to)
            };
            self.incident.entry(from).or_default().push(ei);
            self.incident.entry(to).or_default().push(ei);
            let other = if from == id { to } else { from };
            touched.insert(other);
            if let Some(&oi) = self.node_index.get(&other) {
                if oi != new_index {
                    edges_to.push(oi);
                }
            }
        }
        edges_to.sort_unstable();
        edges_to.dedup();

        let mut stats = IncrementalUpdateStats::default();

        // Assemble only the new column and its direct neighbours (the only blocks
        // the new edges changed) — not the whole system.
        let t_asm = std::time::Instant::now();
        let rebuild: Vec<usize> = touched
            .iter()
            .filter_map(|pid| self.node_index.get(pid).copied())
            .collect();
        self.rebuild_columns(&rebuild);
        stats.assemble_secs += t_asm.elapsed().as_secs_f64();

        // Grow (or initialize) the factor at the current linearization point.
        let dim = (new_index + 1) * 6;
        let t_fac = std::time::Instant::now();
        match &mut self.solver {
            None => {
                let flat: Vec<(usize, usize, f64)> = self.col_h.iter().flatten().copied().collect();
                self.solver = Some(
                    BlockIncrementalSolver::factor(&flat, dim)
                        .map_err(|_| PoseGraphError::SingularSystem)?,
                );
            }
            Some(solver) => {
                solver
                    .append_variable(&edges_to, &self.col_h)
                    .map_err(|_| PoseGraphError::SingularSystem)?;
            }
        }
        stats.factor_secs += t_fac.elapsed().as_secs_f64();

        for _ in 0..self.config.max_inner_iters {
            let neg_g = DVector::from_fn(self.g.len(), |i, _| -self.g[i]);
            let t_solve = std::time::Instant::now();
            let delta = self.solver.as_ref().unwrap().solve(&neg_g);
            stats.solve_secs += t_solve.elapsed().as_secs_f64();
            stats.inner_iters += 1;

            // Snapshot (id, index, step) so the pose mutation below doesn't alias
            // the node_index borrow.
            let steps: Vec<(u64, usize, Vector6<f64>)> = self
                .node_index
                .iter()
                .map(|(&pid, &vi)| (pid, vi, Vector6::from_fn(|k, _| delta[vi * 6 + k])))
                .collect();

            let mut max_step = 0.0f64;
            let mut moved_ids: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
            for (pid, _vi, xi) in &steps {
                let s = xi.norm();
                max_step = max_step.max(s);
                if s > 0.0 {
                    let p = self.graph.poses.get_mut(pid).unwrap();
                    p.world_to_camera = p.world_to_camera.compose(&SE3::exp(xi));
                }
                if s > self.config.relin_threshold {
                    moved_ids.insert(*pid);
                }
            }

            if max_step < self.config.step_tolerance {
                stats.converged = true;
                break;
            }

            // Dirty H columns: a moved pose dirties itself and every pose it shares
            // an edge with (the curvature block depends on the source pose and lands
            // on both endpoints). See the module docs for why skipping sub-threshold
            // moved poses is unsound.
            let mut dirty: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
            for (_pid, vi, xi) in &steps {
                if xi.norm() > self.config.relin_threshold {
                    dirty.insert(*vi);
                }
            }
            for edge in &self.graph.edges {
                if moved_ids.contains(&edge.from) || moved_ids.contains(&edge.to) {
                    if let Some(&i) = self.node_index.get(&edge.from) {
                        dirty.insert(i);
                    }
                    if let Some(&i) = self.node_index.get(&edge.to) {
                        dirty.insert(i);
                    }
                }
            }
            let moved: Vec<usize> = dirty.into_iter().collect();
            stats.relinearized_columns += moved.len();

            // Re-assemble only the dirty columns' H + g, then refactor along their
            // elimination-tree ancestor paths.
            let t_asm = std::time::Instant::now();
            self.rebuild_columns(&moved);
            stats.assemble_secs += t_asm.elapsed().as_secs_f64();
            let t_fac = std::time::Instant::now();
            let solver = self.solver.as_mut().unwrap();
            solver
                .update_columns(&moved, &self.col_h)
                .map_err(|_| PoseGraphError::SingularSystem)?;
            stats.factor_secs += t_fac.elapsed().as_secs_f64();
            stats.refactors += 1;
        }

        Ok(stats)
    }

    /// Re-assemble block column `c`'s lower-triangular normal-matrix blocks
    /// (diagonal + below-diagonal cross blocks to higher-indexed neighbours) and
    /// its gradient block, from the pose's incident edges. Mirrors the batch
    /// [`PoseGraph::assemble_se3_system`] per-edge math (no robust kernel, no GNC)
    /// in natural variable order. `O(degree)`, not `O(n)`.
    fn assemble_column(&self, c: usize) -> (Vec<(usize, usize, f64)>, Vector6<f64>) {
        let pid = self.column_id[c];
        let mut diag = Matrix6::<f64>::zeros();
        let mut g_block = Vector6::<f64>::zeros();
        let mut crosses: Vec<(usize, Matrix6<f64>)> = Vec::new();
        if let Some(edge_idxs) = self.incident.get(&pid) {
            for &ei in edge_idxs {
                let edge = &self.graph.edges[ei];
                let (w, ata, atr) = edge_gn_terms(edge, &self.graph.poses);
                diag += w * ata;
                // Gradient sign: +AᵀΩr at the `to` endpoint, −AᵀΩr at the `from`.
                if edge.to == pid {
                    g_block += w * atr;
                }
                if edge.from == pid {
                    g_block -= w * atr;
                }
                // Below-diagonal cross block to a higher-indexed neighbour. The
                // curvature block AᵀΩA is symmetric, so both endpoint orderings of
                // the off-diagonal coupling are −w·AᵀΩA.
                let other = if edge.from == pid { edge.to } else { edge.from };
                if let Some(&oi) = self.node_index.get(&other) {
                    if oi > c {
                        crosses.push((oi, -(w * ata)));
                    }
                }
            }
        }
        let mut col: Vec<(usize, usize, f64)> = Vec::with_capacity(36 * (crosses.len() + 1));
        push_block(&mut col, c, c, 1.0, &diag);
        for (oi, cross) in &crosses {
            push_block(&mut col, *oi, c, 1.0, cross);
        }
        (col, g_block)
    }

    /// Re-assemble the given columns' `col_h` triplets and gradient blocks in place.
    fn rebuild_columns(&mut self, cols: &[usize]) {
        for &c in cols {
            let (col, g_block) = self.assemble_column(c);
            self.col_h[c] = col;
            for k in 0..6 {
                self.g[c * 6 + k] = g_block[k];
            }
        }
    }
}

/// Per-edge Gauss–Newton terms `(weight, AᵀΩA, AᵀΩr)` at the current estimate,
/// with the approximate adjoint Jacobian `A = Ad(T_from)` the batch solver uses.
fn edge_gn_terms(
    edge: &PoseGraphEdge,
    poses: &BTreeMap<u64, Pose>,
) -> (f64, Matrix6<f64>, Vector6<f64>) {
    let t_from = &poses[&edge.from].world_to_camera;
    let t_to = &poses[&edge.to].world_to_camera;
    let predicted = t_to.compose(&t_from.inverse());
    let r = edge.measurement.inverse().compose(&predicted).log();
    let ad_from = t_from.adjoint();
    match &edge.information {
        Some(omega) => {
            let oa = ad_from.transpose() * omega;
            (1.0, oa * ad_from, oa * r)
        }
        None => (
            edge.weight,
            ad_from.transpose() * ad_from,
            ad_from.transpose() * r,
        ),
    }
}

/// Scatter a weighted `6×6` block into block position `(bi, bj)` of the scalar
/// COO triplet list (all 36 entries, so the block is always structurally present).
fn push_block(
    triplets: &mut Vec<(usize, usize, f64)>,
    bi: usize,
    bj: usize,
    w: f64,
    m: &Matrix6<f64>,
) {
    for r in 0..6 {
        for c in 0..6 {
            triplets.push((bi * 6 + r, bj * 6 + c, w * m[(r, c)]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{relative_world_to_camera, LinearSolver, PoseGraphEdgeKind, PoseGraphSe3Config};
    use nalgebra::{UnitQuaternion, Vector3};

    /// Deterministic splitmix64 for reproducible measurement noise.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * 2.0 - 1.0
        }
    }

    fn truth_c2w(i: u64, n: u64) -> SE3 {
        let t = i as f64 / n as f64;
        let angle = t * std::f64::consts::TAU;
        SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, angle),
            Vector3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 3.0 * t),
        )
    }
    fn truth_pose(i: u64, n: u64) -> Pose {
        Pose {
            world_to_camera: truth_c2w(i, n).inverse(),
        }
    }

    /// Driving the smoother keyframe by keyframe (relinearizing every moved pose,
    /// `relin_threshold = 0`) must reach the same optimum as a from-scratch batch
    /// Gauss–Newton on the identical graph: an exact incremental Gauss–Newton.
    /// The graph is a 3D loop with *noisy* sequential measurements (so the optimum
    /// is non-trivial) plus a closing loop edge added with the last keyframe.
    #[test]
    fn incremental_matches_batch_on_a_noisy_loop() {
        const N: u64 = 24;

        // Drifted initial estimates: integrate the true relative motion plus a
        // constant yaw bias each step (accumulated odometry drift).
        let drift = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.02),
            Vector3::zeros(),
        );
        let mut est = vec![truth_c2w(0, N)];
        for i in 1..N {
            let true_rel = truth_c2w(i - 1, N).inverse().compose(&truth_c2w(i, N));
            let prev = est[(i - 1) as usize].clone();
            est.push(prev.compose(&true_rel).compose(&drift));
        }
        let est_pose = |i: u64| Pose {
            world_to_camera: est[i as usize].inverse(),
        };

        // Deterministic small noise on each measurement so the optimum is a real
        // balance, not exact truth (which both solvers would reach trivially).
        let mut rng = Rng(0xc0ffee);
        let noisy_measurement = |from: u64, to: u64, rng: &mut Rng| -> SE3 {
            let base = relative_world_to_camera(&truth_pose(from, N), &truth_pose(to, N));
            let noise = Vector6::from_fn(|_, _| 0.01 * rng.next_f64());
            base.compose(&SE3::exp(&noise))
        };

        // Pre-generate the measurements so incremental and batch see the same ones.
        let mut seq: Vec<(u64, u64, SE3)> = Vec::new();
        for i in 1..N {
            seq.push((i - 1, i, noisy_measurement(i - 1, i, &mut rng)));
        }
        let loop_meas = noisy_measurement(N - 1, 0, &mut rng);

        let edge =
            |from: u64, to: u64, m: SE3, kind: PoseGraphEdgeKind, weight: f64| PoseGraphEdge {
                from,
                to,
                measurement: m,
                kind,
                weight,
                information: None,
            };

        // --- incremental ---
        let cfg = IncrementalSmootherConfig {
            relin_threshold: 0.0, // exact Gauss–Newton
            step_tolerance: 1e-10,
            max_inner_iters: 40,
        };
        let mut inc = IncrementalPoseGraph::new(0, est_pose(0), cfg);
        for i in 1..N {
            let mut edges = vec![edge(
                i - 1,
                i,
                seq[(i - 1) as usize].2.clone(),
                PoseGraphEdgeKind::Sequential,
                1.0,
            )];
            if i == N - 1 {
                edges.push(edge(
                    N - 1,
                    0,
                    loop_meas.clone(),
                    PoseGraphEdgeKind::LoopClosure,
                    10.0,
                ));
            }
            inc.add_keyframe(i, est_pose(i), edges)
                .expect("add_keyframe");
        }
        let inc_cost = inc.graph.se3_cost();

        // --- batch reference: identical graph, batch Gauss–Newton ---
        let mut batch = PoseGraph::new();
        for i in 0..N {
            batch.add_pose(i, est_pose(i));
        }
        batch.anchor(0);
        for (from, to, m) in &seq {
            batch.add_sequential_edge(*from, *to, m.clone());
        }
        batch.edges.push(edge(
            N - 1,
            0,
            loop_meas.clone(),
            PoseGraphEdgeKind::LoopClosure,
            10.0,
        ));
        let config = PoseGraphSe3Config {
            max_iterations: 100,
            initial_lambda: None, // Gauss–Newton
            chordal_init: false,
            ..PoseGraphSe3Config::default()
        };
        let result = batch.optimize_se3_iterative(&config).expect("batch solve");
        let batch_cost = result.final_cost;

        // Same optimum: both Gauss–Newton on the same graph from a good init.
        assert!(
            (inc_cost - batch_cost).abs() / batch_cost.max(1e-9) < 1e-4,
            "incremental cost {inc_cost:.6e} disagrees with batch {batch_cost:.6e}"
        );
        // And the per-pose poses must agree (gauge fixed by the shared anchor).
        for i in 1..N {
            let pi = inc.graph.poses[&i].world_to_camera.clone();
            let pb = batch.poses[&i].world_to_camera.clone();
            let diff = pi.inverse().compose(&pb).log().norm();
            assert!(
                diff < 1e-4,
                "pose {i} disagrees between incremental and batch: {diff:.3e}"
            );
        }
    }

    /// A/B timing of the online step: building a long trajectory keyframe by
    /// keyframe with the incremental smoother (fluid relinearization, so most
    /// keyframes touch only the new pose and any loop's path), vs. the cost of a
    /// *single* full batch Gauss–Newton at the final size — the work the naive
    /// online loop (re-solve everything each keyframe) would pay `N` times. The
    /// trajectory is a noisy 3D loop with a place-recognition loop closure every
    /// 50 keyframes. Run with `cargo test -p visloc-slam --release -- --ignored
    /// --nocapture bench_incremental_vs_batch_per_step`.
    #[test]
    #[ignore]
    fn bench_incremental_vs_batch_per_step() {
        use std::time::Instant;
        const N: u64 = 600;
        const LOOP_GAP: u64 = 50;

        let mut rng = Rng(0x5a1ad);
        let noisy = |from: u64, to: u64, rng: &mut Rng| -> SE3 {
            let base = relative_world_to_camera(&truth_pose(from, N), &truth_pose(to, N));
            base.compose(&SE3::exp(&Vector6::from_fn(|_, _| 0.01 * rng.next_f64())))
        };
        // Drifted odometry init.
        let drift = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.01),
            Vector3::zeros(),
        );
        let mut est = vec![truth_c2w(0, N)];
        for i in 1..N {
            let true_rel = truth_c2w(i - 1, N).inverse().compose(&truth_c2w(i, N));
            let prev = est[(i - 1) as usize].clone();
            est.push(prev.compose(&true_rel).compose(&drift));
        }
        let est_pose = |i: u64| Pose {
            world_to_camera: est[i as usize].inverse(),
        };
        let edge = |from, to, m, kind, weight| PoseGraphEdge {
            from,
            to,
            measurement: m,
            kind,
            weight,
            information: None,
        };

        // Pre-generate measurements (same for both paths).
        let seq: Vec<SE3> = (1..N).map(|i| noisy(i - 1, i, &mut rng)).collect();
        let loops: Vec<(u64, u64, SE3)> = (1..N)
            .filter(|&i| i >= LOOP_GAP && i % LOOP_GAP == 0)
            .map(|i| (i, i - LOOP_GAP, noisy(i, i - LOOP_GAP, &mut rng)))
            .collect();

        // --- incremental, keyframe by keyframe ---
        let cfg = IncrementalSmootherConfig {
            relin_threshold: 0.0,
            step_tolerance: 1e-7,
            max_inner_iters: 10,
        };
        let mut inc = IncrementalPoseGraph::new(0, est_pose(0), cfg);
        let mut relin_total = 0usize;
        let (mut asm_s, mut solve_s, mut fac_s) = (0.0, 0.0, 0.0);
        let t0 = Instant::now();
        for i in 1..N {
            let mut edges = vec![edge(
                i - 1,
                i,
                seq[(i - 1) as usize].clone(),
                PoseGraphEdgeKind::Sequential,
                1.0,
            )];
            for (lf, lt, lm) in loops.iter().filter(|(lf, _, _)| *lf == i) {
                edges.push(edge(
                    *lf,
                    *lt,
                    lm.clone(),
                    PoseGraphEdgeKind::LoopClosure,
                    10.0,
                ));
            }
            let stats = inc.add_keyframe(i, est_pose(i), edges).unwrap();
            relin_total += stats.relinearized_columns;
            asm_s += stats.assemble_secs;
            solve_s += stats.solve_secs;
            fac_s += stats.factor_secs;
        }
        let inc_total = t0.elapsed().as_secs_f64() * 1e3;
        let inc_cost = inc.graph.se3_cost();
        println!(
            "  breakdown: assemble {:.0}ms / solve {:.0}ms / factor {:.0}ms",
            asm_s * 1e3,
            solve_s * 1e3,
            fac_s * 1e3,
        );

        // --- one full batch Gauss–Newton at the final size ---
        let mut batch = PoseGraph::new();
        for i in 0..N {
            batch.add_pose(i, est_pose(i));
        }
        batch.anchor(0);
        for i in 1..N {
            batch.add_sequential_edge(i - 1, i, seq[(i - 1) as usize].clone());
        }
        for (lf, lt, lm) in &loops {
            batch.edges.push(edge(
                *lf,
                *lt,
                lm.clone(),
                PoseGraphEdgeKind::LoopClosure,
                10.0,
            ));
        }
        let config = PoseGraphSe3Config {
            max_iterations: 100,
            initial_lambda: None,
            chordal_init: false,
            linear_solver: LinearSolver::Sparse,
            ..PoseGraphSe3Config::default()
        };
        let reps = 5;
        let t1 = Instant::now();
        let mut batch_cost = 0.0;
        for _ in 0..reps {
            let mut b = batch.clone();
            batch_cost = b.optimize_se3_iterative(&config).unwrap().final_cost;
        }
        let batch_one = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;

        println!(
            "incremental build of {N} keyframes ({} loop closures): {inc_total:.1} ms total ({:.1} us/kf), {} relinearized columns total, final cost {inc_cost:.4e}",
            loops.len(),
            inc_total * 1e3 / (N - 1) as f64,
            relin_total,
        );
        println!(
            "  one full batch GN @ {N}: {batch_one:.1} ms (final cost {batch_cost:.4e}); naive online (re-solve every keyframe) ~= {} x that = ~{:.0} ms",
            N - 1,
            batch_one * (N - 1) as f64,
        );
    }
}
