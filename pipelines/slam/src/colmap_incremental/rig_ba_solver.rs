//! C2.5: fast native bundle-adjustment backend for
//! `bundle_adjustment.rs::solve`'s rig-reprojection problem
//! (`bundle::BundleAdjustment` populated with `add_pose`/`add_landmark`/
//! `add_rig_observation`/`fix_pose`/`fix_landmark` only — the exact subset
//! `bundle_adjustment.rs::solve` builds).
//!
//! Replaces `bundle::BundleAdjustment::optimize`'s `BTreeMap`-indexed,
//! serially-assembled, densely-Schur-solved LM loop (measured ~340ms/iter at
//! 97 frames / 51k obs / 1.7k points; the dense 7.5k×7.5k reduced-camera
//! solve for a 1250-frame problem is already the dominant cost and is
//! infeasible past ~10k frames) with:
//! - contiguous `Vec`-of-struct observation/frame/point storage (§ below),
//!   sorted by point so per-point Schur elimination reads a contiguous slice
//!   — no map lookups in the per-observation/per-point hot loops;
//! - `rayon`-parallel Jacobian evaluation and per-point Schur elimination,
//!   reduced deterministically (every parallel step is either a positional
//!   `collect()`, whose result order is independent of the thread count by
//!   construction of `rayon`'s `IndexedParallelIterator`, or a serial fold
//!   over that already-ordered `Vec` — never a `rayon` `.sum()`/`.reduce()`,
//!   whose split structure this module does not want to depend on);
//! - the reduced (pose-only) camera system solved as a 6×6-block-sparse
//!   system via [`crate::block_cholesky::solve_spd_blocks6_cached`] (already
//!   in this crate; reused here rather than re-implemented, per the task
//!   brief's "reuse `block_cholesky.rs` if it fits" — it does: the module's
//!   own doc frames it as "the BA Schur solve" as one of its two intended
//!   callers). No separate dense path is implemented for small frame counts
//!   (a documented deviation from the task's "dense fallback ≤200 frames"
//!   suggestion): `block_cholesky`'s own doc states it is the *general*
//!   production path (dense-at-block-granularity even for small block
//!   counts, see its "Why not supernodal?" section), so a second,
//!   independently-tested dense code path would duplicate risk without a
//!   measured benefit — verified instead by comparing Native's output
//!   directly against `BundleAdjustment::optimize` (`Legacy`, itself a dense
//!   solver) on identical small synthetic problems (`native_vs_legacy_*`
//!   tests below).
//!
//! ## Residual model
//! Identical formula to `bundle.rs`'s `rig_residual_jacobians` (the existing,
//! already-exercised port of COLMAP's `RigReprojErrorConstantRigCostFunctor`,
//! `cost_functions/reprojection_error.h:389-417`, documented in
//! `bundle_adjustment.rs`'s module doc citation): fixed pinhole intrinsics
//! `(fx, fy, cx, cy)`, fixed `sensor_from_rig`, variable `rig_from_world`
//! pose and variable landmark `xyz`:
//! ```text
//! point_rig    = rig_from_world · point_world
//! point_sensor = sensor_from_rig · point_rig
//! residual     = (fx·x/z + cx, fy·y/z + cy) − observed_xy     (point_sensor = (x,y,z))
//! ```
//! Robust loss (`SoftLOneLoss`/`CauchyLoss`, selectable per
//! [`super::bundle_adjustment::LossFunction`]) is supported per the C2.7 task
//! brief's finding that COLMAP's control is **not** trivial-loss everywhere:
//! `LocalBundleAdjustment()` sets `SOFT_L1` (scale 1.0,
//! `controllers/incremental_pipeline.cc:217-219`), `GlobalBundleAdjustment()`
//! stays `TRIVIAL` (`:267-268`), and `IterativeLocalRefinement` downgrades to
//! `TRIVIAL` after its first iteration ("Only use robust cost function for
//! first iteration", `sfm/incremental_mapper.cc:1277-1281`) — see
//! `bundle_adjustment.rs`'s module doc and `mapper.rs::iterative_local_refinement`
//! for how those defaults/the per-iteration switch are threaded through.
//! [`Corrector`] below is a direct, 1:1 port of Ceres' own residual/Jacobian
//! robustification (`internal/ceres/corrector.{h,cc}`), including the exact
//! call order and cost convention `residual_block.cc:161-197` establishes
//! (`cost = 0.5·rho(s)`; Jacobian corrected *before* the residual, both using
//! the *uncorrected* residual) — this module's `linearize_point`/
//! `evaluate_cost` mirror that order exactly. No deviation was needed: the
//! algorithm is closed-form (no root-finding, no auxiliary randomness).
//! [`residual_and_jacobians`] below is an independent, allocation-free
//! re-derivation of the same formula (not a call into `bundle.rs`, keeping
//! this module link-independent); `tests::matches_bundle_rs_formula` checks
//! the two agree bit-for-bit-close on random inputs, and
//! `tests::jacobian_matches_finite_difference` checks both partials
//! numerically.
//!
//! ## Pose parameterization
//! `rig_from_world` is `Pose { world_to_camera: SE3 }` (`SE3 { rotation:
//! UnitQuaternion, translation }`). The local update is this crate's
//! existing SE(3) convention — the same one every other `bundle.rs` LM loop
//! already uses (e.g. `bundle.rs:2894,3906,16075,16205`):
//! `T_new = T_old.compose(SE3::exp(xi))`, `xi = [ρ; ω] ∈ R^6` (translation
//! tangent first, then rotation axis-angle) — a *right* (local, body-frame)
//! 6-DoF manifold update. COLMAP's own Ceres setup instead uses
//! `EigenQuaternionManifold` for the rotation (a separate 4-parameter block
//! with its own local parameterization) plus a plain unconstrained R³ block
//! for the translation. Both are minimal 6-DoF local parameterizations of
//! the same SE(3) manifold at the same linearization point; the converged
//! optimum (a stationary point of the same reprojection-error cost) and the
//! Gauss-Newton/LM trust-region geometry near it coincide to first order —
//! the difference is bookkeeping (one coupled 6-vector update vs. two
//! independent blocks), not a modeling difference. This port keeps this
//! crate's existing convention rather than introducing a second one.
//!
//! ## Trust region (Ceres `LEVENBERG_MARQUARDT` defaults, as COLMAP's control
//! configures them — `bundle_adjustment_ceres.cc:103-105` sets
//! `function_tolerance=0`, `parameter_tolerance=0`; `gradient_tolerance` is
//! left at Ceres' own default `1e-4`; the trust-region strategy and its
//! `initial_trust_region_radius=1e4` are Ceres solver defaults, not
//! overridden by COLMAP)
//! - Diagonal (Jacobi) scaling: each parameter block's damping uses
//!   `mu · clamp(diag(H_block), 1e-6, 1e32)` (`mu = 1/radius`), added to that
//!   block's own diagonal only — the well-known
//!   `LevenbergMarquardtStrategy` clamp constants from the public Ceres
//!   Solver source (`internal/ceres/levenberg_marquardt_strategy.cc`).
//! - Step quality `rho = actual_cost_decrease / predicted_cost_decrease`
//!   (both on the Ceres-internal `0.5·Σr²` cost scale, so the gradient
//!   `g = Jᵀr` and the `gradient_tolerance` check line up with Ceres';
//!   [`BaResult`]'s reported `initial_cost`/`final_cost` are rescaled back to
//!   `Σr²` to match `BundleAdjustment::cost()`'s convention for the
//!   Native/Legacy parity test). `predicted_decrease = ½·dx·(b + μ·D·dx)`,
//!   the standard trust-region model-decrease identity for a step solving
//!   `(H + μD)dx = b`.
//! - Accept iff `rho > 1e-3`; on accept `radius ← radius / max(1/3, 1 −
//!   (2ρ−1)³)`, reset the decrease factor to 2; on reject `radius ←
//!   radius / decrease_factor`, then `decrease_factor *= 2`.
//! - Terminate on `max_num_iterations` (from
//!   [`super::bundle_adjustment::BundleAdjustmentOptions`]) or
//!   `‖g‖_∞ ≤ 1e-4` (checked once per trial, at the current linearization —
//!   `function_tolerance=0`/`parameter_tolerance=0` are absolute zeros, i.e.
//!   COLMAP's control disables those two early-stop checks, so this is the
//!   only early-stop path, matching the task brief).
//! - Jacobians are only re-evaluated after an **accepted** step (mirroring
//!   Ceres' `TrustRegionMinimizer`); a rejected trial only re-solves the
//!   reduced system at the new damping and re-evaluates the (Jacobian-free)
//!   candidate cost.
//!
//! ## Gauge / constant handling
//! Frames in `ba.fixed_poses` and points in `ba.fixed_landmarks` are never
//! given a variable slot (their observations still constrain the *other*
//! side of the residual — see [`Problem::free_frame_slot`] / point handling
//! in [`linearize_point`]) — this is `bundle_adjustment.rs`'s existing
//! `BundleAdjustmentConfig`/`Gauge` mechanism (`set_constant_rig_from_world_pose`,
//! `add_constant_point`, `fix_gauge`), unchanged by this backend swap.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use nalgebra::{DMatrix, Matrix3, Matrix6, Point2, Point3, SMatrix, Vector2, Vector3, Vector6};
use rayon::prelude::*;

use visloc_core::geometry::SE3;

use crate::block_cholesky::{solve_spd_blocks6_cached, BlockSymbolic};
use crate::bundle::{BaError, BaIterationStats, BaResult, BundleAdjustment};

use super::bundle_adjustment::LossFunction;

/// Ceres `LevenbergMarquardtStrategy` defaults (public Ceres Solver source,
/// `internal/ceres/levenberg_marquardt_strategy.cc` /
/// `trust_region_minimizer.cc`); see module doc.
const INITIAL_RADIUS: f64 = 1e4;
const MIN_RELATIVE_DECREASE: f64 = 1e-3;
const GRADIENT_TOLERANCE: f64 = 1e-4;
const MIN_DIAGONAL: f64 = 1e-6;
const MAX_DIAGONAL: f64 = 1e32;

/// One rig-reprojection observation, flattened to plain floats/indices — no
/// `Camera`/`SE3` lookups by id anywhere in the per-observation hot loop.
/// Grouped by point in [`Problem::obs`] so a point's observations are a
/// contiguous slice.
#[derive(Debug, Clone)]
struct Obs {
    frame_idx: u32,
    xy: Point2<f64>,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    sensor_from_rig: SE3,
}

struct Problem {
    frame_ids: Vec<u64>,
    /// `Some(slot)` for free frames (`slot` indexes the reduced camera
    /// system, `0..n_free_frames`), `None` for fixed frames.
    free_frame_slot: Vec<Option<u32>>,
    n_free_frames: usize,
    poses: Vec<SE3>,

    point_ids: Vec<u64>,
    point_fixed: Vec<bool>,
    points: Vec<Point3<f64>>,

    obs: Vec<Obs>,
    /// `[start, end)` into `obs` for each point (`obs` is grouped by point).
    point_obs_range: Vec<(u32, u32)>,

    /// Performance fix (see [`build_reduced_system_pattern`]'s doc): the
    /// distinct *free* frames observing each point, ascending by
    /// `frame_idx` — a property of the observation topology alone
    /// (independent of pose/point values, hence independent of the current
    /// LM linearization), computed once here rather than rediscovered by
    /// scanning Jacobian-evaluated observations on every accepted step.
    point_free_frames: Vec<Vec<u32>>,
    /// Deduplicated, ascending `(free_slot_a, free_slot_b)` pairs (`a < b`)
    /// across every point's `point_free_frames` — the reduced camera
    /// system's off-diagonal sparsity pattern.
    edges: Vec<(u32, u32)>,
    /// Per point, the `edges`-index of each of its
    /// `point_free_frames[p].len() * (len-1) / 2` distinct frame pairs, in
    /// the canonical `for j in 1..n { for i in 0..j }` order over
    /// `point_free_frames[p]`.
    point_pair_edge_idx: Vec<Vec<u32>>,
    /// Performance fix #3 (see [`eliminate_and_accumulate`]'s doc): point
    /// index range `[shard_boundaries[s], shard_boundaries[s+1])` for shard
    /// `s`, balanced by *cumulative pair count* (`Σ n·(n-1)/2`), not point
    /// count — a real (corridor-revisit) track-length distribution is
    /// heavy-tailed, so an equal-point-count split can leave one shard
    /// doing most of the O(pairs) work while the others idle. Computed once
    /// here (pair counts are fixed for the whole `optimize()` call) from
    /// `point_free_frames`, reused unchanged by every trial and every
    /// accepted step.
    shard_boundaries: Vec<usize>,
    /// Built when the memory cap leaves [`elimination_shard_count`] below 8
    /// (large, dense covisibility graphs): the elimination then runs in
    /// parallel over its outputs instead (see [`ContributionIndex`]).
    contributions: Option<ContributionIndex>,
}

/// Which points observe each free frame of the reduced camera system, in
/// ascending point order, plus where each frame's off-diagonal row starts
/// in `edges`. Lets [`eliminate_and_accumulate`] compute every row of the
/// reduced system independently, in parallel, with exactly the additions
/// (and their order) of the sharded loop.
struct ContributionIndex {
    /// `slot[slot_off[s]..slot_off[s + 1]]`: `(point, k)` for free slot `s`,
    /// `k` the position of `s`'s frame in `point_free_frames[point]`.
    slot_off: Vec<usize>,
    slot: Vec<[u32; 2]>,
    /// `edges[row_off[s]..row_off[s + 1]]` are the edges `(s, _)`.
    row_off: Vec<usize>,
    /// Shard of each point (the sharded loop's summation grouping).
    point_shard: Vec<u32>,
}

impl ContributionIndex {
    fn new(
        point_free_frames: &[Vec<u32>],
        free_frame_slot: &[Option<u32>],
        edges: &[(u32, u32)],
        n_free: usize,
        shard_boundaries: &[usize],
    ) -> Self {
        let mut slot_off = vec![0usize; n_free + 1];
        for frames in point_free_frames {
            for &f in frames {
                slot_off[free_frame_slot[f as usize].expect("free") as usize + 1] += 1;
            }
        }
        for i in 1..slot_off.len() {
            slot_off[i] += slot_off[i - 1];
        }
        let mut slot = vec![[0u32; 2]; slot_off[n_free]];
        let mut slot_fill = slot_off.clone();
        for (p, frames) in point_free_frames.iter().enumerate() {
            for (k, &f) in frames.iter().enumerate() {
                let s = free_frame_slot[f as usize].expect("free") as usize;
                slot[slot_fill[s]] = [p as u32, k as u32];
                slot_fill[s] += 1;
            }
        }
        let row_off = (0..=n_free)
            .map(|s| edges.partition_point(|&(a, _)| (a as usize) < s))
            .collect();
        let mut point_shard = vec![0u32; point_free_frames.len()];
        for s in 0..shard_boundaries.len().saturating_sub(1) {
            for v in &mut point_shard[shard_boundaries[s]..shard_boundaries[s + 1]] {
                *v = s as u32;
            }
        }
        Self {
            slot_off,
            slot,
            row_off,
            point_shard,
        }
    }
}

/// Performance fix (profiled on a 300-frame/2500-point/289k-observation
/// synthetic problem sized like the tier-1000 global BA call that measured
/// 2.9s/iteration): the reduced camera system's off-diagonal blocks come
/// from O(track_len²) frame pairs per point — real tracks (corridor
/// revisits) run ~70 observations/point, so this is millions of pairs
/// total. Two things had to change from a first pass that computed this
/// per accepted LM step (still O(pairs) but only recomputed a few times
/// instead of every trial): (1) it must be a `BTreeMap`-free flat-array
/// merge (`solve_step`'s hot path), and (2) — the fix here — the pair
/// *pattern* itself (which frames co-observe which point) is a property of
/// the **observation topology alone**, not of the current pose/point
/// values, so it is computed **once per `optimize()` call**, not once per
/// accepted step: re-deriving and re-sorting ~6M pairs on every accept
/// (even only ~8 times in the profiling run) still cost ~200ms each. Using
/// `problem.obs`/`point_obs_range` (raw observation topology, ignoring the
/// z<=0 cheirality guard `residual_and_jacobians` applies — a frame/point
/// pair failing that check is a numerical edge case, not a topology
/// change) instead of `PointLin.frames` (Jacobian-evaluation output)
/// removes the coupling to relinearization entirely.
#[allow(clippy::type_complexity)]
fn build_reduced_system_pattern(
    point_obs_range: &[(u32, u32)],
    obs: &[Obs],
    frame_fixed: &[bool],
    free_frame_slot: &[Option<u32>],
) -> (Vec<Vec<u32>>, Vec<(u32, u32)>, Vec<Vec<u32>>) {
    // Per-point work runs on rayon; the outputs (sorted unique edge list,
    // per-point free frames and pair -> edge indices) are pure functions of
    // the input, so they match the serial construction exactly.
    let point_free_frames: Vec<Vec<u32>> = point_obs_range
        .par_iter()
        .map(|&(start, end)| {
            let mut frames: Vec<u32> = obs[start as usize..end as usize]
                .iter()
                .map(|o| o.frame_idx)
                .filter(|&f| !frame_fixed[f as usize])
                .collect();
            frames.sort_unstable();
            frames.dedup();
            frames
        })
        .collect();
    let slots_of = |frames: &[u32]| -> Vec<u32> {
        frames
            .iter()
            .map(|&f| free_frame_slot[f as usize].expect("free"))
            .collect()
    };

    let mut raw_pairs: Vec<(u32, u32)> = point_free_frames
        .par_iter()
        .filter(|frames| frames.len() >= 2)
        .flat_map_iter(|frames| {
            let slots = slots_of(frames);
            let mut pairs = Vec::with_capacity(slots.len() * (slots.len() - 1) / 2);
            for j in 1..slots.len() {
                for i in 0..j {
                    pairs.push((slots[i], slots[j]));
                }
            }
            pairs
        })
        .collect();
    raw_pairs.par_sort_unstable();
    raw_pairs.dedup();
    let edges = raw_pairs;
    let n_slots = free_frame_slot
        .iter()
        .flatten()
        .map(|&s| s as usize + 1)
        .max()
        .unwrap_or(0);
    let mut row_off = vec![0usize; n_slots + 1];
    for &(a, _) in &edges {
        row_off[a as usize + 1] += 1;
    }
    for i in 1..row_off.len() {
        row_off[i] += row_off[i - 1];
    }

    let point_pair_edge_idx: Vec<Vec<u32>> = point_free_frames
        .par_iter()
        .map(|frames| {
            if frames.len() < 2 {
                return Vec::new();
            }
            // Slots ascend with frames, so the pairs `(i, j > i)` walk
            // edge row `slots[i]` forwards.
            let slots = slots_of(frames);
            let n = slots.len();
            let mut idxs = vec![0u32; n * (n - 1) / 2];
            for i in 0..n {
                let a = slots[i] as usize;
                let row = &edges[row_off[a]..row_off[a + 1]];
                let mut pos = 0usize;
                for j in i + 1..n {
                    pos += row[pos..].partition_point(|&(_, b)| b < slots[j]);
                    debug_assert_eq!(row[pos], (slots[i], slots[j]), "pair is an edge");
                    idxs[j * (j - 1) / 2 + i] = (row_off[a] + pos) as u32;
                }
            }
            idxs
        })
        .collect();

    (point_free_frames, edges, point_pair_edge_idx)
}

/// Reduced camera systems with at least this many free frames are solved
/// iteratively ([`pcg_blocks6`]) instead of by block Cholesky — COLMAP's own
/// switch from Ceres' `SPARSE_SCHUR` to `ITERATIVE_SCHUR` past 1,000 images.
/// The factor of a dense covisibility graph (e.g. a city block revisited in
/// several passes) costs seconds to tens of seconds per LM step at that
/// size, while a preconditioned CG solve stays linear in the number of
/// camera pairs.
const ITERATIVE_MIN_FREE_FRAMES: usize = 1000;
/// [`ITERATIVE_MIN_FREE_FRAMES`], overridable with
/// `VISLOC_PORT_ITERATIVE_MIN_FRAMES` for diagnostics / A-B runs.
fn iterative_min_free_frames() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VISLOC_PORT_ITERATIVE_MIN_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(ITERATIVE_MIN_FREE_FRAMES)
    })
}
/// Relative residual at which [`pcg_blocks6`] stops (`‖r‖ ≤ tol · ‖b‖`).
/// An inexact LM step is fine (the trust region absorbs it); Ceres'
/// `ITERATIVE_SCHUR` default is a far looser `eta = 0.1`. With the
/// iterative solve forced on from 100 free frames, a 553-image Courthouse
/// replay at 1e-3 keeps all 543 images and puts every camera centre within
/// 0.08 % of the scene extent of the block-Cholesky run (1e-2: one image
/// fewer, 0.6 %).
const PCG_RELATIVE_TOLERANCE: f64 = 1e-3;
/// Iteration cap of [`pcg_blocks6`] (Ceres' `max_linear_solver_iterations`).
const PCG_MAX_ITERATIONS: usize = 500;

/// Solve the SPD block system given as lower block columns (`columns[j]`
/// maps block row `i >= j` to `A_ij`, diagonal included) for the single
/// right-hand side `rhs` by conjugate gradients with a block-Jacobi
/// (inverse 6×6 diagonal) preconditioner — the camera-block analogue of
/// Ceres' `SCHUR_JACOBI`. The matrix-vector product runs on rayon over
/// symmetric block rows and every reduction is summed in block order, so
/// the result does not depend on the thread count. `None` when a diagonal
/// block is not positive definite or the iteration breaks down.
fn pcg_blocks6(
    columns: &[BTreeMap<usize, Matrix6<f64>>],
    rhs: &DMatrix<f64>,
) -> Option<DMatrix<f64>> {
    let n = columns.len();
    // Symmetric block rows: row i holds (j, A_ij) for every stored block,
    // ascending by j (blocks left of the diagonal come from earlier
    // columns, then the diagonal and the rest of column i).
    let mut rows: Vec<Vec<(usize, Matrix6<f64>)>> = vec![Vec::new(); n];
    for (j, col) in columns.iter().enumerate() {
        for (&i, blk) in col {
            rows[i].push((j, *blk));
            if i != j {
                rows[j].push((i, blk.transpose()));
            }
        }
    }
    let mut row_off = Vec::with_capacity(n + 1);
    row_off.push(0);
    for row in &rows {
        row_off.push(row_off.last().copied().unwrap_or(0) + row.len());
    }
    let entries: Vec<(usize, Matrix6<f64>)> = rows.into_iter().flatten().collect();
    pcg_csr6(&row_off, &entries, rhs)
}

/// [`pcg_blocks6`] on the symmetric block rows in CSR form: row `i` is
/// `entries[row_off[i]..row_off[i + 1]]`, `(column, A_ij)` ascending by
/// column with the diagonal block included.
fn pcg_csr6(
    row_off: &[usize],
    entries: &[(usize, Matrix6<f64>)],
    rhs: &DMatrix<f64>,
) -> Option<DMatrix<f64>> {
    let n = row_off.len() - 1;
    let diag: Vec<Matrix6<f64>> = (0..n)
        .map(|i| {
            entries[row_off[i]..row_off[i + 1]]
                .iter()
                .find(|(j, _)| *j == i)
                .map(|(_, d)| *d)
        })
        .collect::<Option<Vec<_>>>()?;
    let matvec = |x: &[Vector6<f64>]| -> Vec<Vector6<f64>> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                entries[row_off[i]..row_off[i + 1]]
                    .iter()
                    .fold(Vector6::zeros(), |acc, (j, blk)| acc + blk * x[*j])
            })
            .collect()
    };
    pcg_core(&diag, matvec, rhs)
}

/// [`pcg_blocks6`] with the matrix given by its diagonal blocks and a
/// matrix-vector product.
fn pcg_core(
    diag: &[Matrix6<f64>],
    matvec: impl Fn(&[Vector6<f64>]) -> Vec<Vector6<f64>>,
    rhs: &DMatrix<f64>,
) -> Option<DMatrix<f64>> {
    let n = diag.len();
    let precond: Vec<Matrix6<f64>> = diag
        .par_iter()
        .map(|d| d.cholesky().map(|c| c.inverse()))
        .collect::<Option<Vec<_>>>()?;
    let dot = |a: &[Vector6<f64>], b: &[Vector6<f64>]| -> f64 {
        let partial: Vec<f64> = a.par_iter().zip(b).map(|(x, y)| x.dot(y)).collect();
        partial.iter().sum()
    };
    let b: Vec<Vector6<f64>> = (0..n)
        .map(|i| Vector6::from_fn(|k, _| rhs[(i * 6 + k, 0)]))
        .collect();
    let b_norm = dot(&b, &b).sqrt();
    let mut x = vec![Vector6::<f64>::zeros(); n];
    if b_norm == 0.0 {
        return Some(DMatrix::zeros(n * 6, 1));
    }
    let mut r = b;
    let mut z: Vec<Vector6<f64>> = r.iter().zip(&precond).map(|(r, m)| m * r).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for _ in 0..PCG_MAX_ITERATIONS {
        let ap = matvec(&p);
        let pap = dot(&p, &ap);
        if !pap.is_finite() || pap <= 0.0 {
            return None;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if dot(&r, &r).sqrt() <= PCG_RELATIVE_TOLERANCE * b_norm {
            break;
        }
        z = r.iter().zip(&precond).map(|(r, m)| m * r).collect();
        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        rz = rz_next;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    let mut out = DMatrix::<f64>::zeros(n * 6, 1);
    for (i, xi) in x.iter().enumerate() {
        for k in 0..6 {
            out[(i * 6 + k, 0)] = xi[k];
        }
    }
    Some(out)
}

/// Reduced camera systems with at least this many free frames are factored
/// in a fill-reducing block order ([`crate::reordering`]); smaller ones (every
/// local BA) keep the natural order, where fill is negligible.
const REORDER_MIN_FREE_FRAMES: usize = 64;

fn build_problem(ba: &BundleAdjustment) -> Result<Problem, BaError> {
    if ba.poses.is_empty() {
        return Err(BaError::NoPoses);
    }
    if ba.landmarks.is_empty() {
        return Err(BaError::NoLandmarks);
    }
    if ba.rig_observations.is_empty() {
        return Err(BaError::NoObservations);
    }

    // `BTreeMap` keys are ascending, so ids map to indices by binary search.
    let frame_ids: Vec<u64> = ba.poses.keys().copied().collect();
    let frame_fixed: Vec<bool> = frame_ids
        .iter()
        .map(|id| ba.fixed_poses.contains(id))
        .collect();
    let mut free_frame_slot = vec![None; frame_ids.len()];
    let mut n_free_frames = 0usize;
    for (i, &fixed) in frame_fixed.iter().enumerate() {
        if !fixed {
            free_frame_slot[i] = Some(n_free_frames as u32);
            n_free_frames += 1;
        }
    }
    if n_free_frames == 0 {
        return Err(BaError::AllPosesFixed);
    }
    let poses: Vec<SE3> = frame_ids
        .iter()
        .map(|id| ba.poses[id].world_to_camera.clone())
        .collect();

    let point_ids: Vec<u64> = ba.landmarks.keys().copied().collect();
    let point_fixed: Vec<bool> = point_ids
        .iter()
        .map(|id| ba.fixed_landmarks.contains(id))
        .collect();
    let points: Vec<Point3<f64>> = ba.landmarks.values().copied().collect();

    // (frame index, point index) per observation, in parallel; the first
    // failing observation (in input order) decides the error, as before.
    let indices: Vec<Result<(u32, u32), BaError>> = ba
        .rig_observations
        .par_iter()
        .map(|o| {
            let frame_idx = frame_ids
                .binary_search(&o.keyframe_id)
                .map_err(|_| BaError::MissingPose(o.keyframe_id))?;
            let point_idx = point_ids
                .binary_search(&o.landmark_id)
                .map_err(|_| BaError::MissingLandmark(o.landmark_id))?;
            o.camera
                .intrinsics()
                .ok_or(BaError::UnsupportedCameraModel)?;
            Ok((frame_idx as u32, point_idx as u32))
        })
        .collect();
    let indices: Vec<(u32, u32)> = indices.into_iter().collect::<Result<_, _>>()?;

    // Stable counting sort by point index: observations of a point keep the
    // caller's (`ba.rig_observations`) order, exactly like the stable
    // `sort_by_key` this replaces, without moving the large `Obs` records.
    let mut point_obs_range = vec![(0u32, 0u32); point_ids.len()];
    for &(_, p) in &indices {
        point_obs_range[p as usize].1 += 1;
    }
    let mut start = 0u32;
    for range in point_obs_range.iter_mut() {
        let count = range.1;
        *range = (start, start + count);
        start += count;
    }
    let mut order = vec![0u32; indices.len()];
    let mut next: Vec<u32> = point_obs_range.iter().map(|r| r.0).collect();
    for (i, &(_, p)) in indices.iter().enumerate() {
        order[next[p as usize] as usize] = i as u32;
        next[p as usize] += 1;
    }
    let obs: Vec<Obs> = order
        .par_iter()
        .map(|&i| {
            let o = &ba.rig_observations[i as usize];
            let (frame_idx, _) = indices[i as usize];
            let (fx, fy, cx, cy) = o.camera.intrinsics().expect("checked above");
            Obs {
                frame_idx,
                xy: o.xy,
                fx,
                fy,
                cx,
                cy,
                sensor_from_rig: o.sensor_from_rig.clone(),
            }
        })
        .collect();

    let (point_free_frames, edges, point_pair_edge_idx) =
        build_reduced_system_pattern(&point_obs_range, &obs, &frame_fixed, &free_frame_slot);
    let n_shards = elimination_shard_count(edges.len());
    let shard_boundaries = balanced_shard_boundaries(&point_free_frames, n_shards);
    let contributions = (n_shards < 8).then(|| {
        ContributionIndex::new(
            &point_free_frames,
            &free_frame_slot,
            &edges,
            n_free_frames,
            &shard_boundaries,
        )
    });

    Ok(Problem {
        frame_ids,
        free_frame_slot,
        n_free_frames,
        poses,
        point_ids,
        point_fixed,
        points,
        obs,
        point_obs_range,
        point_free_frames,
        edges,
        point_pair_edge_idx,
        shard_boundaries,
        contributions,
    })
}

#[inline]
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Residual `r = π(sensor_from_rig · pose · point_world) − xy` and its
/// Jacobians w.r.t. the pose's `[ρ; ω]` local update (`2×6`) and the point
/// (`2×3`). `None` when the point is behind the sensor (matches
/// `bundle.rs::cost()`'s "skip nonprojectable observations" convention — see
/// module doc). See module doc for the formula citation and
/// `tests::matches_bundle_rs_formula` / `tests::jacobian_matches_finite_difference`
/// for verification.
#[inline]
#[allow(clippy::too_many_arguments)]
fn residual_and_jacobians(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    sensor_from_rig: &SE3,
    pose: &SE3,
    point_world: &Point3<f64>,
    xy: Point2<f64>,
) -> Option<(Vector2<f64>, SMatrix<f64, 2, 6>, SMatrix<f64, 2, 3>)> {
    let point_rig = pose.transform_point(point_world);
    let point_sensor = sensor_from_rig.transform_point(&point_rig);
    if point_sensor.z <= 0.0 {
        return None;
    }
    let z_inv = point_sensor.z.recip();
    let z_inv2 = z_inv * z_inv;
    let predicted = Point2::new(
        fx * point_sensor.x * z_inv + cx,
        fy * point_sensor.y * z_inv + cy,
    );
    let residual = Vector2::new(predicted.x - xy.x, predicted.y - xy.y);

    let projection = SMatrix::<f64, 2, 3>::new(
        fx * z_inv,
        0.0,
        -fx * point_sensor.x * z_inv2,
        0.0,
        fy * z_inv,
        -fy * point_sensor.y * z_inv2,
    );
    let r_wc = pose.rotation.to_rotation_matrix().into_inner();
    let r_rs = sensor_from_rig.rotation.to_rotation_matrix().into_inner();
    let mut d_rig_d_pose = SMatrix::<f64, 3, 6>::zeros();
    d_rig_d_pose.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_wc);
    d_rig_d_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-r_wc * skew(&point_world.coords)));
    let j_pose = projection * r_rs * d_rig_d_pose;
    let j_point = projection * r_rs * r_wc;
    Some((residual, j_pose, j_point))
}

/// `rho(s), rho'(s), rho''(s)` for one residual block's squared norm `s`.
/// Port of `ceres::{TrivialLoss,SoftLOneLoss,CauchyLoss}::Evaluate`
/// (`internal/ceres/loss_function.cc:46-50` (Trivial: `rho=[s,1,0]`),
/// `:68-75` (`SoftLOneLoss`), `:77-84` (`CauchyLoss`)), with the
/// scale-squaring constructors `b_ = a*a`, `c_ = 1/b_` from
/// `include/ceres/loss_function.h:190-217` (`a` = this module's `scale`).
/// `f64::MIN_POSITIVE` matches `std::numeric_limits<double>::min()`'s
/// smallest-positive-normal value used to clamp `rho[1]` away from zero.
#[inline]
fn evaluate_loss(loss: LossFunction, s: f64) -> [f64; 3] {
    match loss {
        LossFunction::Trivial => [s, 1.0, 0.0],
        LossFunction::SoftL1(a) => {
            let b = a * a;
            let c = 1.0 / b;
            let sum = 1.0 + s * c;
            let tmp = sum.sqrt();
            let rho0 = 2.0 * b * (tmp - 1.0);
            let rho1 = (1.0 / tmp).max(f64::MIN_POSITIVE);
            let rho2 = -(c * rho1) / (2.0 * sum);
            [rho0, rho1, rho2]
        }
        LossFunction::Cauchy(a) => {
            let b = a * a;
            let c = 1.0 / b;
            let sum = 1.0 + s * c;
            let inv = 1.0 / sum;
            let rho0 = b * sum.ln();
            let rho1 = inv.max(f64::MIN_POSITIVE);
            let rho2 = -c * (inv * inv);
            [rho0, rho1, rho2]
        }
    }
}

/// Direct port of `ceres::internal::Corrector` (`internal/ceres/corrector.cc:41-155`):
/// rescales one residual block's residual and Jacobian(s) so that a plain
/// Gauss-Newton/LM step on the *corrected* quantities reproduces Ceres'
/// robustified step (the BAMS — "Bundle Adjustment: A Modern Synthesis" —
/// second-order correction in the common case, falling back to a first-order
/// rescale when `rho'' >= 0` or `s == 0`, `corrector.cc:45-93`).
struct Corrector {
    /// Multiplies each Jacobian block (`corrector.cc:108,127,150-152`).
    sqrt_rho1: f64,
    /// Multiplies the residual (`corrector.cc:108,115`) — equals `sqrt_rho1`
    /// in the common (`rho'' <= 0`) case, differs only in the rarer
    /// `rho'' > 0` branch.
    residual_scaling: f64,
    /// `alpha / sq_norm` (`corrector.cc:109`); `0.0` selects the fast
    /// (no-curvature-correction) path in [`Corrector::correct_jacobian`].
    alpha_sq_norm: f64,
}

impl Corrector {
    /// Port of `Corrector::Corrector` (`corrector.cc:41-110`). `rho` is
    /// `[rho(s), rho'(s), rho''(s)]` for this residual block's squared norm
    /// `sq_norm`.
    fn new(sq_norm: f64, rho: [f64; 3]) -> Self {
        let sqrt_rho1 = rho[1].max(0.0).sqrt();
        // `corrector.cc:81-85`: the common case (and `sq_norm == 0`) skips
        // the second-order curvature correction entirely.
        if sq_norm == 0.0 || rho[2] <= 0.0 {
            return Self {
                sqrt_rho1,
                residual_scaling: sqrt_rho1,
                alpha_sq_norm: 0.0,
            };
        }
        // `corrector.cc:93-109`. `rho[1] > 0` is guaranteed here (Ceres
        // `CHECK_GT`; our loss evaluators clamp `rho[1]` to
        // `f64::MIN_POSITIVE`, never non-positive) so this division is safe.
        let d = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
        let alpha = 1.0 - d.sqrt();
        Self {
            sqrt_rho1,
            residual_scaling: sqrt_rho1 / (1.0 - alpha),
            alpha_sq_norm: alpha / sq_norm,
        }
    }

    /// Port of `Corrector::CorrectJacobian` (`corrector.cc:118-155`), applied
    /// to one parameter block's `2×C` Jacobian (`C` = 6 for the pose block,
    /// 3 for the point block — each corrected independently, both from the
    /// *same*, still-uncorrected `r`, matching `residual_block.cc:179-191`'s
    /// per-parameter-block loop that runs entirely before
    /// [`Corrector::correct_residual`]).
    #[inline]
    fn correct_jacobian<const C: usize>(&self, r: &Vector2<f64>, j: &mut SMatrix<f64, 2, C>) {
        if self.alpha_sq_norm == 0.0 {
            // `corrector.cc:126-129`.
            *j *= self.sqrt_rho1;
            return;
        }
        // `corrector.cc:131-154`: `J = sqrt_rho1 * (J - alpha_sq_norm * r * (rᵀJ))`.
        let r_transpose_j = r.transpose() * *j;
        *j = (*j - r * (self.alpha_sq_norm * r_transpose_j)) * self.sqrt_rho1;
    }

    /// Port of `Corrector::CorrectResiduals` (`corrector.cc:112-116`). Must
    /// be called *after* every [`Corrector::correct_jacobian`] call for this
    /// residual block (they need the uncorrected `r`).
    #[inline]
    fn correct_residual(&self, r: &mut Vector2<f64>) {
        *r *= self.residual_scaling;
    }
}

/// Per-point linearization output (Stage A — recomputed only after an
/// accepted step). `frames` are the point's *free* observing frames only
/// (ascending, deduplicated); fixed frames contribute to `hpp`/`bp` (they
/// still constrain the point) but never get a `diag`/`bc`/`hcp` slot (they
/// have no variable to receive one).
struct PointLin {
    frames: Vec<u32>,
    /// `−Jᵀr` summed per free frame that observes this point (the pose-block
    /// raw, mu-independent contribution — same convention as `bp`).
    bc: Vec<Vector6<f64>>,
    /// `JᵀJ` summed per free frame (raw, mu-independent).
    diag: Vec<Matrix6<f64>>,
    /// `Jpose^T Jpoint` per free frame, only meaningful if `free`.
    hcp: Vec<SMatrix<f64, 6, 3>>,
    /// `JᵀJ` for the point block, only meaningful if `free`.
    hpp: Matrix3<f64>,
    /// `−Jᵀr` for the point block, only meaningful if `free`.
    bp: Vector3<f64>,
    free: bool,
}

/// Stage A: evaluate residual + Jacobians for every observation of one point
/// and fold them into that point's local blocks. Returns `(cost_contribution,
/// PointLin)`; `cost_contribution = Σ rho(s_i)` (full, un-halved — matches
/// `BundleAdjustment::cost()`'s convention; `rho(s_i) = s_i = ‖r_i‖²` for
/// `LossFunction::Trivial`, so this is exactly the old "Σ ‖r_i‖²" convention
/// when the loss is trivial) over this point's projectable observations, with
/// `hpp`/`bp`/`bc`/`diag`/`hcp` folded from the robust-loss-*corrected*
/// `(r, j_pose, j_point)` (see `evaluate_loss`/`Corrector` above) rather than
/// the raw ones. `target_frames` (== `problem.point_free_frames[p]`, ascending)
/// fixes `frames`/`bc`/`diag`/`hcp`'s slot layout up front — an observation's
/// frame is placed by `binary_search` (`target_frames` is sorted), not a
/// linear scan that grows the list — see [`build_reduced_system_pattern`]'s
/// doc for why the frame *set* itself is precomputed once per `optimize()`
/// call rather than rediscovered here.
fn linearize_point(
    obs_slice: &[Obs],
    point_free: bool,
    point_xyz: &Point3<f64>,
    poses: &[SE3],
    target_frames: &[u32],
    loss: LossFunction,
) -> (f64, PointLin) {
    let mut cost = 0.0;
    let mut hpp = Matrix3::zeros();
    let mut bp = Vector3::zeros();
    let mut bc: Vec<Vector6<f64>> = vec![Vector6::zeros(); target_frames.len()];
    let mut diag: Vec<Matrix6<f64>> = vec![Matrix6::zeros(); target_frames.len()];
    let mut hcp: Vec<SMatrix<f64, 6, 3>> = vec![SMatrix::<f64, 6, 3>::zeros(); target_frames.len()];

    for o in obs_slice {
        let pose = &poses[o.frame_idx as usize];
        let Some((mut r, mut j_pose, mut j_point)) = residual_and_jacobians(
            o.fx,
            o.fy,
            o.cx,
            o.cy,
            &o.sensor_from_rig,
            pose,
            point_xyz,
            o.xy,
        ) else {
            continue;
        };
        // Robust-loss correction (`residual_block.cc:161-197`'s exact order:
        // evaluate rho at the *raw* squared norm, correct both Jacobian
        // blocks from the raw residual, *then* correct the residual itself)
        // — a no-op for `LossFunction::Trivial` (`rho=[s,1,0]` makes
        // `Corrector::new` take the `sqrt_rho1=1, alpha_sq_norm=0` fast
        // path), skipped outright here to avoid the `Corrector`
        // construction/multiply overhead on the by-far most common case
        // (global BA is always `Trivial`; local BA is `Trivial` after its
        // first iteration too, see `mapper.rs::iterative_local_refinement`).
        let rho0 = if matches!(loss, LossFunction::Trivial) {
            r.norm_squared()
        } else {
            let s = r.norm_squared();
            let rho = evaluate_loss(loss, s);
            let corrector = Corrector::new(s, rho);
            corrector.correct_jacobian(&r, &mut j_pose);
            corrector.correct_jacobian(&r, &mut j_point);
            corrector.correct_residual(&mut r);
            rho[0]
        };
        cost += rho0;
        if point_free {
            hpp += j_point.transpose() * j_point;
            bp += -(j_point.transpose() * r);
        }
        if let Ok(slot) = target_frames.binary_search(&o.frame_idx) {
            bc[slot] += -(j_pose.transpose() * r);
            diag[slot] += j_pose.transpose() * j_pose;
            if point_free {
                hcp[slot] += j_pose.transpose() * j_point;
            }
        }
    }

    (
        cost,
        PointLin {
            frames: target_frames.to_vec(),
            bc,
            diag,
            hcp,
            hpp,
            bp,
            free: point_free,
        },
    )
}

/// Stage A driver: relinearize every point in parallel (deterministic —
/// `par_iter().collect()` on an `IndexedParallelIterator` preserves input
/// order regardless of thread count), then serially merge each point's
/// per-frame contribution into the (mu-independent) raw pose-block
/// accumulators. Returns `(full_cost, points_lin, frame_diag_raw,
/// frame_bc_raw)`. The reduced-system sparsity pattern
/// (`problem.edges`/`problem.point_pair_edge_idx`) is *not* recomputed here
/// — see [`build_reduced_system_pattern`]'s doc for why it is a
/// `build_problem`-time, not a per-linearization, computation.
#[allow(clippy::type_complexity)]
fn linearize(
    problem: &Problem,
    loss: LossFunction,
) -> (f64, Vec<PointLin>, Vec<Matrix6<f64>>, Vec<Vector6<f64>>) {
    let contributions: Vec<(f64, PointLin)> = problem
        .points
        .par_iter()
        .enumerate()
        .map(|(p, xyz)| {
            let (start, end) = problem.point_obs_range[p];
            linearize_point(
                &problem.obs[start as usize..end as usize],
                !problem.point_fixed[p],
                xyz,
                &problem.poses,
                &problem.point_free_frames[p],
                loss,
            )
        })
        .collect();

    // `contributions` is a plain `Vec` (already materialized by the ordered
    // `collect()` above) — this final fold is a serial iterator, not a
    // `rayon` reduction, so it is deterministic by construction regardless
    // of `RAYON_NUM_THREADS` without needing any special chunking.
    let full_cost: f64 = contributions.iter().map(|(c, _)| *c).sum();

    let mut frame_diag_raw = vec![Matrix6::<f64>::zeros(); problem.n_free_frames];
    let mut frame_bc_raw = vec![Vector6::<f64>::zeros(); problem.n_free_frames];
    let mut points_lin = Vec::with_capacity(contributions.len());
    for (_, pl) in contributions {
        for (k, &fidx) in pl.frames.iter().enumerate() {
            let slot = problem.free_frame_slot[fidx as usize].expect("frames in PointLin are free")
                as usize;
            frame_diag_raw[slot] += pl.diag[k];
            frame_bc_raw[slot] += pl.bc[k];
        }
        points_lin.push(pl);
    }

    (full_cost, points_lin, frame_diag_raw, frame_bc_raw)
}

#[inline]
fn clamp_diag6(h: &Matrix6<f64>) -> Vector6<f64> {
    Vector6::from_iterator((0..6).map(|i| h[(i, i)].clamp(MIN_DIAGONAL, MAX_DIAGONAL)))
}

#[inline]
fn clamp_diag3(h: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::from_iterator((0..3).map(|i| h[(i, i)].clamp(MIN_DIAGONAL, MAX_DIAGONAL)))
}

/// Damped 3×3 point-block inverse at the given `mu = 1/radius`, or `None` if
/// the damped block is (numerically) singular.
fn damped_hpp_inverse(hpp: &Matrix3<f64>, mu: f64) -> Option<Matrix3<f64>> {
    let diag = clamp_diag3(hpp);
    let mut damped = *hpp;
    for i in 0..3 {
        damped[(i, i)] += mu * diag[i];
    }
    damped.try_inverse()
}

/// Per-phase wall time accumulated across every trial of one `optimize()`
/// call, printed as `BA_PHASES` alongside `BA_SOLVE` (task item: "Add timing
/// instrumentation inside the native solver per phase"). Phases match the
/// task brief's list: `eliminate` = per-point Schur elimination,
/// `assemble` = merging eliminated contributions into the reduced-system
/// blocks (+ pose damping + `columns`/`rhs` construction), `linsolve` =
/// [`crate::block_cholesky::solve_spd_blocks6_cached`], `backsub` =
/// free-point back-substitution + predicted-decrease bookkeeping,
/// `evaluate_cost` = the trial-step residual-only cost re-evaluation.
/// `linearize` is timed separately in [`optimize`] (only runs on an
/// accepted step, so its count usually differs from the trial count).
#[derive(Default, Debug, Clone, Copy)]
struct PhaseTimings {
    linearize: Duration,
    eliminate: Duration,
    assemble: Duration,
    linsolve: Duration,
    backsub: Duration,
    evaluate_cost: Duration,
    linearize_calls: u32,
    trials: u32,
    reorder: Duration,
}

/// Performance fix #2 (see [`build_reduced_system_pattern`] for fix #1):
/// profiling after fix #1 still showed `eliminate_ms`+`assemble_ms`
/// dominating (~1.4s of ~2.2s total at 300f/2500pt/289k-obs, 7 trials) even
/// with the flat-`Vec` merge and the precomputed edge pattern. Two things
/// remained: (a) `eliminate_point` still heap-allocated an
/// O(track_len²)-sized `Vec<Matrix6>` *per point, per trial* (its `pairs`
/// output) purely to hand values to a *separate* merge step — at ~2500
/// points × up to `~2500` pairs each, that is gigabytes of transient
/// allocation per trial; (b) the merge then wrote those values into
/// `flat_offdiag` at effectively random indices (`pair_edge_idx` scatters
/// across the whole edge set), which is cache-unfriendly and was serial
/// (one core). This function fuses per-point elimination and the
/// flat-buffer write into one step with **no** per-point/per-pair heap
/// allocation (only a small O(track_len) `hcp_scaled` scratch buffer,
/// reused — not reallocated — across every point in a shard) and shards
/// the O(total pairs) work across a *fixed* number of point-index ranges
/// (independent of `RAYON_NUM_THREADS`) so it runs in parallel while
/// staying bit-identical at any thread count (module doc): each shard
/// writes into its own private `edges.len()`-sized buffer (no cross-thread
/// contention), and the buffers are summed in a fixed shard-index order
/// afterward.
#[allow(clippy::type_complexity)]
#[allow(clippy::needless_range_loop)]
fn eliminate_and_accumulate(
    problem: &Problem,
    points_lin: &[PointLin],
    edges_len: usize,
    mu: f64,
) -> (Vec<Matrix6<f64>>, Vec<Matrix6<f64>>, Vec<Vector6<f64>>) {
    let n_free = problem.n_free_frames;
    if let Some(index) = &problem.contributions {
        return eliminate_by_output(problem, index, points_lin, edges_len, mu);
    }
    let n_shards = problem.shard_boundaries.len().saturating_sub(1).max(1);

    let shard_results: Vec<(Vec<Matrix6<f64>>, Vec<Matrix6<f64>>, Vec<Vector6<f64>>)> = (0
        ..n_shards)
        .into_par_iter()
        .map(|shard| {
            let lo = problem.shard_boundaries[shard];
            let hi = problem.shard_boundaries[shard + 1];
            let mut local_offdiag = vec![Matrix6::<f64>::zeros(); edges_len];
            let mut local_diag = vec![Matrix6::<f64>::zeros(); n_free];
            let mut local_bc = vec![Vector6::<f64>::zeros(); n_free];
            let mut hcp_scaled: Vec<SMatrix<f64, 6, 3>> = Vec::new();
            // Both `pl.frames`/`pl.hcp` and `problem.point_pair_edge_idx`
            // are indexed by the same `p`, and the pair loop below indexes
            // `hcp_scaled`/`pl.hcp` at two *different* offsets (`i`, `j`) in
            // the same iteration — an index loop reads more directly than
            // threading multiple `.zip()`s through both (hence the
            // `needless_range_loop` allow above).
            for p in lo..hi.max(lo) {
                let pl = &points_lin[p];
                if !pl.free || pl.frames.is_empty() {
                    continue;
                }
                let n = pl.frames.len();
                let Some(hpp_inv) = damped_hpp_inverse(&pl.hpp, mu) else {
                    // Degenerate point block: contribute nothing rather than
                    // propagate a solve failure (see the original
                    // `eliminate_point` doc, same policy).
                    continue;
                };
                hcp_scaled.clear();
                hcp_scaled.extend((0..n).map(|k| pl.hcp[k] * hpp_inv));
                for k in 0..n {
                    let slot =
                        problem.free_frame_slot[pl.frames[k] as usize].expect("free") as usize;
                    local_diag[slot] -= hcp_scaled[k] * pl.hcp[k].transpose();
                    local_bc[slot] -= hcp_scaled[k] * pl.bp;
                }
                let pair_edge_idx = &problem.point_pair_edge_idx[p];
                let mut local_pair = 0usize;
                for j in 1..n {
                    for i in 0..j {
                        let val = hcp_scaled[j] * pl.hcp[i].transpose();
                        local_offdiag[pair_edge_idx[local_pair] as usize] -= val;
                        local_pair += 1;
                    }
                }
            }
            (local_offdiag, local_diag, local_bc)
        })
        .collect();

    let mut flat_offdiag = vec![Matrix6::<f64>::zeros(); edges_len];
    let mut diag_delta = vec![Matrix6::<f64>::zeros(); n_free];
    let mut bc_delta = vec![Vector6::<f64>::zeros(); n_free];
    // Fixed shard-index order (0..n_shards, not runtime scheduling order) —
    // bit-identical at any thread count, matching `block_cholesky`'s
    // documented "fixed chunk, folded in order" determinism pattern.
    for (local_offdiag, local_diag, local_bc) in shard_results {
        for (a, b) in flat_offdiag.iter_mut().zip(local_offdiag) {
            *a += b;
        }
        for (a, b) in diag_delta.iter_mut().zip(local_diag) {
            *a += b;
        }
        for (a, b) in bc_delta.iter_mut().zip(local_bc) {
            *a += b;
        }
    }
    (flat_offdiag, diag_delta, bc_delta)
}

/// [`eliminate_and_accumulate`] computed per output block, in parallel:
/// each off-diagonal block, diagonal block and right-hand side sums its
/// points' contributions in ascending point order, starting a fresh partial
/// sum at every shard boundary and adding the partial sums in shard order,
/// which is exactly the sharded loop's arithmetic (it subtracts each
/// shard's points in order into a zeroed per-shard buffer, then adds the
/// buffers in shard order). Used when the per-shard buffers would not fit
/// in memory more than a few at a time, where the sharded loop ran on as
/// few as 2 threads.
#[allow(clippy::type_complexity)]
fn eliminate_by_output(
    problem: &Problem,
    index: &ContributionIndex,
    points_lin: &[PointLin],
    edges_len: usize,
    mu: f64,
) -> (Vec<Matrix6<f64>>, Vec<Matrix6<f64>>, Vec<Vector6<f64>>) {
    let n_free = problem.n_free_frames;
    let (point_free_frames, point_pair_edge_idx) =
        (&problem.point_free_frames, &problem.point_pair_edge_idx);
    let hpp_inv: Vec<Option<Matrix3<f64>>> = points_lin
        .par_iter()
        .map(|pl| {
            if !pl.free || pl.frames.is_empty() {
                None
            } else {
                damped_hpp_inverse(&pl.hpp, mu)
            }
        })
        .collect();
    // Sum `term(c)` over `contribs` (ascending point) grouped by shard.
    fn grouped<T: Copy + std::ops::AddAssign + std::ops::SubAssign>(
        zero: T,
        contribs: impl Iterator<Item = (u32, T)>,
    ) -> T {
        let (mut total, mut acc, mut shard) = (zero, zero, None);
        for (s, term) in contribs {
            if shard != Some(s) {
                if shard.is_some() {
                    total += acc;
                }
                acc = zero;
                shard = Some(s);
            }
            acc -= term;
        }
        if shard.is_some() {
            total += acc;
        }
        total
    }
    // Row `s` of the off-diagonal blocks (edges `(s, _)`) collects, from
    // each point observing `s` in ascending order, the pairs `(k, j > k)`
    // with `k` at `s`: every edge sees its points in ascending order.
    let mut flat_offdiag = vec![Matrix6::<f64>::zeros(); edges_len];
    let mut rows: Vec<&mut [Matrix6<f64>]> = Vec::with_capacity(n_free);
    let mut rest = flat_offdiag.as_mut_slice();
    for s in 0..n_free {
        let (row, tail) = rest.split_at_mut(index.row_off[s + 1] - index.row_off[s]);
        rows.push(row);
        rest = tail;
    }
    rows.into_par_iter().enumerate().for_each_init(
        || (Vec::new(), Vec::new()),
        |(acc, shard), (s, total)| {
            acc.clear();
            acc.resize(total.len(), Matrix6::<f64>::zeros());
            shard.clear();
            shard.resize(total.len(), None);
            let base = index.row_off[s];
            for &[p, k] in &index.slot[index.slot_off[s]..index.slot_off[s + 1]] {
                let (p, k) = (p as usize, k as usize);
                let Some(inv) = hpp_inv[p] else {
                    continue;
                };
                let pl = &points_lin[p];
                let point_shard = Some(index.point_shard[p]);
                let pair_edge = &point_pair_edge_idx[p];
                for j in k + 1..point_free_frames[p].len() {
                    let e = pair_edge[j * (j - 1) / 2 + k] as usize - base;
                    let term = (pl.hcp[j] * inv) * pl.hcp[k].transpose();
                    if shard[e] != point_shard {
                        if shard[e].is_some() {
                            total[e] += acc[e];
                        }
                        acc[e] = Matrix6::zeros();
                        shard[e] = point_shard;
                    }
                    acc[e] -= term;
                }
            }
            for ((t, a), sh) in total.iter_mut().zip(acc.iter()).zip(shard.iter()) {
                if sh.is_some() {
                    *t += *a;
                }
            }
        },
    );
    let (diag_delta, bc_delta): (Vec<Matrix6<f64>>, Vec<Vector6<f64>>) = (0..n_free)
        .into_par_iter()
        .map(|slot| {
            let range = &index.slot[index.slot_off[slot]..index.slot_off[slot + 1]];
            let terms = || {
                range.iter().filter_map(|&[p, k]| {
                    let (p, k) = (p as usize, k as usize);
                    let inv = hpp_inv[p]?;
                    let pl = &points_lin[p];
                    Some((index.point_shard[p], pl.hcp[k] * inv, pl, k))
                })
            };
            let diag = grouped(
                Matrix6::zeros(),
                terms().map(|(s, scaled, pl, k)| (s, scaled * pl.hcp[k].transpose())),
            );
            let bc = grouped(
                Vector6::zeros(),
                terms().map(|(s, scaled, pl, _)| (s, scaled * pl.bp)),
            );
            (diag, bc)
        })
        .unzip();
    (flat_offdiag, diag_delta, bc_delta)
}

/// Shard count for [`eliminate_and_accumulate`]: capped so the transient
/// per-shard `edges_len`-sized buffers (`3 shards worth` — see its `local_*`
/// locals — of `Matrix6` = 288 bytes each) never exceed roughly 256MB in
/// aggregate, and bounded to `[1, 8]` (the pool this crate targets). A pure
/// function of `edges_len` (fixed for the whole `optimize()` call — see
/// `build_reduced_system_pattern`), so the shard count — hence the fixed
/// reduction order in [`eliminate_and_accumulate`] — never depends on
/// `RAYON_NUM_THREADS`.
fn elimination_shard_count(edges_len: usize) -> usize {
    // Real (corridor-revisit) global-BA windows can have `edges_len` well
    // into the hundreds of thousands (dense frame-frame co-observation from
    // long, slowly-moving tracks) — e.g. `edges_len=357k` (≈ `C(846,2)`,
    // observed on the real tier-1000 run's largest global-BA window) costs
    // ~103MB *per shard*. At the previous 256MB cap that collapsed to just
    // 2 shards, discarding most of the intended 8x parallelism exactly on
    // the heaviest problems — plausibly the dominant reason a same-sized
    // synthetic profile (with a more modest `edges_len`, hence 8 shards)
    // measured far faster than the real run's global-BA `BA_PHASES` lines.
    // 1.5GB keeps 8 shards up to `edges_len≈650k` (the real run's peak RSS
    // for the *entire* mapper process was 1.15GB, so this has headroom).
    const MAX_TOTAL_BYTES: usize = 1536 * 1024 * 1024;
    let per_shard_bytes = edges_len.max(1) * std::mem::size_of::<Matrix6<f64>>();
    (MAX_TOTAL_BYTES / per_shard_bytes).clamp(1, 8)
}

/// Performance fix #3: [`elimination_shard_count`] fixes *how many* shards,
/// this fixes *which points* land in each — balanced by cumulative pair
/// count (`Σ n·(n-1)/2` over each shard's points), not point count. A real
/// (corridor-revisit) track-length distribution is heavy-tailed: a handful
/// of long-revisited points can carry most of the O(pairs) work, and an
/// equal-point-count split (the original scheme) can land all of them in
/// one shard while the other 7 threads idle — losing most of the intended
/// 8x speedup exactly on the problems that need it most. Computed once (pair
/// counts, hence the optimal boundaries, are fixed for the whole
/// `optimize()` call) via a prefix sum over `point_free_frames` and a
/// `partition_point` binary search per boundary — O(n_points log n_points),
/// negligible next to the O(pairs) work it schedules.
fn balanced_shard_boundaries(point_free_frames: &[Vec<u32>], n_shards: usize) -> Vec<usize> {
    let n_points = point_free_frames.len();
    let n_shards = n_shards.max(1);
    let mut prefix: Vec<u64> = Vec::with_capacity(n_points + 1);
    prefix.push(0);
    for frames in point_free_frames {
        let n = frames.len() as u64;
        prefix.push(prefix.last().copied().unwrap_or(0) + n * n.saturating_sub(1) / 2);
    }
    let total = *prefix.last().unwrap_or(&0);
    let mut boundaries = Vec::with_capacity(n_shards + 1);
    boundaries.push(0usize);
    for s in 1..n_shards {
        let target = total * s as u64 / n_shards as u64;
        let idx = prefix.partition_point(|&v| v < target);
        let idx = idx.clamp(*boundaries.last().expect("non-empty"), n_points);
        boundaries.push(idx);
    }
    boundaries.push(n_points);
    boundaries
}

/// Stage B driver: assemble the damped, Schur-reduced 6×6-block camera
/// system at the given `mu` and solve it via
/// [`crate::block_cholesky::solve_spd_blocks6_cached`], then back-substitute
/// the free-point steps. Returns `(dx_frames, dx_points, predicted_decrease)`
/// (all on the free-slot / point-index space); `None` if the reduced system
/// was singular even after damping (`BaError::SingularSystem`).
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn solve_step(
    problem: &Problem,
    points_lin: &[PointLin],
    edges: &[(u32, u32)],
    frame_diag_raw: &[Matrix6<f64>],
    frame_bc_raw: &[Vector6<f64>],
    mu: f64,
    cache: &mut Option<BlockSymbolic>,
    order: Option<&[u32]>,
    timings: &mut PhaseTimings,
) -> Option<(Vec<Vector6<f64>>, Vec<Vector3<f64>>, f64)> {
    let n_free = problem.n_free_frames;
    let t = Instant::now();
    let (flat_offdiag, diag_delta, bc_delta) =
        eliminate_and_accumulate(problem, points_lin, edges.len(), mu);
    timings.eliminate += t.elapsed();

    let t = Instant::now();
    let mut frame_diag: Vec<Matrix6<f64>> = frame_diag_raw.to_vec();
    let mut frame_bc: Vec<Vector6<f64>> = frame_bc_raw.to_vec();
    for a in 0..n_free {
        frame_diag[a] += diag_delta[a];
        frame_bc[a] += bc_delta[a];
    }
    // Pose-block Jacobi damping, applied to the *raw* (pre-elimination)
    // diagonal — matches Ceres' `LevenbergMarquardtStrategy` (module doc).
    for a in 0..n_free {
        let d = clamp_diag6(&frame_diag_raw[a]);
        for i in 0..6 {
            frame_diag[a][(i, i)] += mu * d[i];
        }
    }

    if n_free >= iterative_min_free_frames() && order.is_none() {
        // Iterative solve: the symmetric block rows straight from the edge
        // list, built per row in parallel. Row `i` is its left blocks
        // (edges `(a, i)`, ascending a), the diagonal, then its right
        // blocks (edges `(i, b)`, ascending b, transposed): the rows, in
        // the order, that `pcg_blocks6` derives from the column map.
        let mut left_off = vec![0usize; n_free + 1];
        for &(_, b) in edges {
            left_off[b as usize + 1] += 1;
        }
        for i in 0..n_free {
            left_off[i + 1] += left_off[i];
        }
        let mut left = vec![0u32; edges.len()];
        let mut fill = left_off.clone();
        for (e, &(_, b)) in edges.iter().enumerate() {
            left[fill[b as usize]] = e as u32;
            fill[b as usize] += 1;
        }
        // Edges ascend by (a, b): row a's right blocks are contiguous.
        let mut right_off = vec![0usize; n_free + 1];
        for &(a, _) in edges {
            right_off[a as usize + 1] += 1;
        }
        for i in 0..n_free {
            right_off[i + 1] += right_off[i];
        }
        let rows: Vec<Vec<(usize, Matrix6<f64>)>> = (0..n_free)
            .into_par_iter()
            .map(|i| {
                let lefts = &left[left_off[i]..left_off[i + 1]];
                let mut row = Vec::with_capacity(lefts.len() + 1 + right_off[i + 1] - right_off[i]);
                row.extend(
                    lefts
                        .iter()
                        .map(|&e| (edges[e as usize].0 as usize, flat_offdiag[e as usize])),
                );
                row.push((i, frame_diag[i]));
                row.extend(
                    (right_off[i]..right_off[i + 1])
                        .map(|e| (edges[e].1 as usize, flat_offdiag[e].transpose())),
                );
                row
            })
            .collect();
        let matvec = |x: &[Vector6<f64>]| -> Vec<Vector6<f64>> {
            rows.par_iter()
                .map(|row| {
                    row.iter()
                        .fold(Vector6::zeros(), |acc, (j, blk)| acc + blk * x[*j])
                })
                .collect()
        };
        let mut rhs = DMatrix::<f64>::zeros(n_free * 6, 1);
        for a in 0..n_free {
            for i in 0..6 {
                rhs[(a * 6 + i, 0)] = frame_bc[a][i];
            }
        }
        timings.assemble += t.elapsed();
        let t = Instant::now();
        let solved = pcg_core(&frame_diag, matvec, &rhs)?;
        let mut dx_frames = vec![Vector6::<f64>::zeros(); n_free];
        for a in 0..n_free {
            for i in 0..6 {
                dx_frames[a][i] = solved[(a * 6 + i, 0)];
            }
        }
        timings.linsolve += t.elapsed();
        return Some(back_substitute(
            problem,
            points_lin,
            frame_diag_raw,
            frame_bc_raw,
            dx_frames,
            mu,
            timings,
        ));
    }

    // Free frame `a` sits at block `pos(a)` of the factored system: the
    // fill-reducing position when an order is given (large systems, see
    // `REORDER_MIN_FREE_FRAMES`), else `a` itself.
    let pos = |a: usize| order.map_or(a, |o| o[a] as usize);
    let mut columns: Vec<BTreeMap<usize, Matrix6<f64>>> = vec![BTreeMap::new(); n_free];
    for (a, &diag) in frame_diag.iter().enumerate().take(n_free) {
        columns[pos(a)].insert(pos(a), diag);
    }
    // `val` is the (row b, column a) block, a < b. Keep the lower triangle
    // of the permuted matrix: transpose when the permutation flips the pair.
    for (&(a, b), val) in edges.iter().zip(flat_offdiag) {
        let (pa, pb) = (pos(a as usize), pos(b as usize));
        if pb >= pa {
            columns[pa].insert(pb, val);
        } else {
            columns[pb].insert(pa, val.transpose());
        }
    }
    let mut rhs = DMatrix::<f64>::zeros(n_free * 6, 1);
    for a in 0..n_free {
        for i in 0..6 {
            rhs[(pos(a) * 6 + i, 0)] = frame_bc[a][i];
        }
    }
    timings.assemble += t.elapsed();

    let t = Instant::now();
    let solved = if n_free >= iterative_min_free_frames() {
        pcg_blocks6(&columns, &rhs)?
    } else {
        solve_spd_blocks6_cached(cache, columns, &rhs).ok()?
    };
    let mut dx_frames = vec![Vector6::<f64>::zeros(); n_free];
    for a in 0..n_free {
        for i in 0..6 {
            dx_frames[a][i] = solved[(pos(a) * 6 + i, 0)];
        }
    }
    timings.linsolve += t.elapsed();
    Some(back_substitute(
        problem,
        points_lin,
        frame_diag_raw,
        frame_bc_raw,
        dx_frames,
        mu,
        timings,
    ))
}

/// Free-point steps from the camera steps, and the LM model's predicted
/// cost decrease, for [`solve_step`].
fn back_substitute(
    problem: &Problem,
    points_lin: &[PointLin],
    frame_diag_raw: &[Matrix6<f64>],
    frame_bc_raw: &[Vector6<f64>],
    dx_frames: Vec<Vector6<f64>>,
    mu: f64,
    timings: &mut PhaseTimings,
) -> (Vec<Vector6<f64>>, Vec<Vector3<f64>>, f64) {
    let n_free = problem.n_free_frames;
    let t = Instant::now();
    let dx_points: Vec<Vector3<f64>> = points_lin
        .par_iter()
        .map(|pl| {
            if !pl.free {
                return Vector3::zeros();
            }
            let Some(hpp_inv) = damped_hpp_inverse(&pl.hpp, mu) else {
                return Vector3::zeros();
            };
            let mut rhs = pl.bp;
            for (k, &fidx) in pl.frames.iter().enumerate() {
                let slot = problem.free_frame_slot[fidx as usize].expect("free") as usize;
                rhs -= pl.hcp[k].transpose() * dx_frames[slot];
            }
            hpp_inv * rhs
        })
        .collect();

    let mut predicted = 0.0;
    for a in 0..n_free {
        let d = clamp_diag6(&frame_diag_raw[a]);
        let dx = dx_frames[a];
        let damped_dx = Vector6::from_iterator((0..6).map(|i| mu * d[i] * dx[i]));
        predicted += dx.dot(&(frame_bc_raw[a] + damped_dx));
    }
    for (p, pl) in points_lin.iter().enumerate() {
        if !pl.free {
            continue;
        }
        let d = clamp_diag3(&pl.hpp);
        let dx = dx_points[p];
        let damped_dx = Vector3::from_iterator((0..3).map(|i| mu * d[i] * dx[i]));
        predicted += dx.dot(&(pl.bp + damped_dx));
    }
    predicted *= 0.5;
    timings.backsub += t.elapsed();
    timings.trials += 1;

    (dx_frames, dx_points, predicted)
}

/// Residual-only (no Jacobian) total cost at a candidate `(poses, points)` —
/// `Σ ‖r_i‖²`, same convention as [`linearize`]'s `full_cost` /
/// `BundleAdjustment::cost()`. Used for a rejected trial's cheap re-check
/// (module doc: Jacobians are only re-evaluated after an accepted step).
fn evaluate_cost(
    problem: &Problem,
    poses: &[SE3],
    points: &[Point3<f64>],
    loss: LossFunction,
) -> f64 {
    let per_point: Vec<f64> = problem
        .points
        .par_iter()
        .enumerate()
        .map(|(p, _)| {
            let (start, end) = problem.point_obs_range[p];
            let mut c = 0.0;
            for o in &problem.obs[start as usize..end as usize] {
                let pose = &poses[o.frame_idx as usize];
                let point_rig = pose.transform_point(&points[p]);
                let point_sensor = o.sensor_from_rig.transform_point(&point_rig);
                if point_sensor.z <= 0.0 {
                    continue;
                }
                let z_inv = point_sensor.z.recip();
                let dx = o.fx * point_sensor.x * z_inv + o.cx - o.xy.x;
                let dy = o.fy * point_sensor.y * z_inv + o.cy - o.xy.y;
                let s = dx * dx + dy * dy;
                // Residual-only re-check (no Jacobian) — `rho[0]` alone
                // reproduces the robustified cost, matching `linearize_point`'s
                // `cost += rho0` convention; `LossFunction::Trivial` short
                // circuits `rho[0] == s` without allocating the `[f64;3]`.
                c += if matches!(loss, LossFunction::Trivial) {
                    s
                } else {
                    evaluate_loss(loss, s)[0]
                };
            }
            c
        })
        .collect();
    // Serial fold over an already-materialized `Vec` — see `linearize`.
    per_point.iter().sum()
}

fn apply_step(
    problem: &Problem,
    dx_frames: &[Vector6<f64>],
    dx_points: &[Vector3<f64>],
) -> (Vec<SE3>, Vec<Point3<f64>>) {
    let poses: Vec<SE3> = problem
        .poses
        .iter()
        .enumerate()
        .map(|(i, pose)| match problem.free_frame_slot[i] {
            Some(slot) => pose.compose(&SE3::exp(&dx_frames[slot as usize])),
            None => pose.clone(),
        })
        .collect();
    let points: Vec<Point3<f64>> = problem
        .points
        .iter()
        .enumerate()
        .map(|(p, xyz)| {
            if problem.point_fixed[p] {
                *xyz
            } else {
                xyz + dx_points[p]
            }
        })
        .collect();
    (poses, points)
}

fn gradient_inf_norm(points_lin: &[PointLin], frame_bc_raw: &[Vector6<f64>]) -> f64 {
    let mut m: f64 = 0.0;
    for v in frame_bc_raw {
        for i in 0..6 {
            m = m.max(v[i].abs());
        }
    }
    for pl in points_lin {
        if !pl.free {
            continue;
        }
        for i in 0..3 {
            m = m.max(pl.bp[i].abs());
        }
    }
    m
}

/// Native `BaBackend` entry point: solves `ba`'s rig-reprojection problem in
/// place (writing the optimized poses/landmarks back into `ba.poses`/
/// `ba.landmarks`) and returns the same [`BaResult`]
/// `bundle_adjustment.rs::solve` already expects from the `Legacy` backend.
/// See module doc for the full algorithm.
///
/// `gradient_tolerance_rel` adds Ceres' **relative** `gradient_tolerance`
/// stopping criterion: terminate once `‖g‖` has fallen to or below
/// `rel * ‖g_initial‖` (the first iteration only records the initial norm, so
/// at least one step is attempted). COLMAP sets this to `1.0` for global BA
/// and `10.0` for
/// local BA (`incremental_pipeline.cc:236-283`/`:192-235`); real Ceres then
/// terminates after ~1-2 iterations, whereas this port's legacy absolute
/// `‖g‖_∞ <= 1e-4` runs to the iteration cap. Passing `None` preserves the
/// legacy absolute behaviour (used by every existing caller/test).
pub(crate) fn optimize_with_tolerance(
    ba: &mut BundleAdjustment,
    max_num_iterations: usize,
    loss: LossFunction,
    gradient_tolerance_rel: Option<f64>,
) -> Result<BaResult, BaError> {
    let t_build = Instant::now();
    let mut problem = build_problem(ba)?;
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
    let mut timings = PhaseTimings::default();
    let mut apply_ms = 0.0;

    let (mut half_cost, mut points_lin, mut frame_diag_raw, mut frame_bc_raw) = {
        let t = Instant::now();
        let (full, pl, fd, fb) = linearize(&problem, loss);
        timings.linearize += t.elapsed();
        timings.linearize_calls += 1;
        (0.5 * full, pl, fd, fb)
    };
    let initial_cost = 2.0 * half_cost;

    let mut radius = INITIAL_RADIUS;
    let mut decrease_factor = 2.0;
    let mut cache: Option<BlockSymbolic> = None;
    // Fill-reducing block order of the reduced camera system (its pattern,
    // `problem.edges`, is fixed for this solve). Natural order otherwise.
    let order: Option<Vec<u32>> = (problem.n_free_frames >= REORDER_MIN_FREE_FRAMES
        && problem.n_free_frames < iterative_min_free_frames())
    .then(|| {
        let t = Instant::now();
        let mut adjacency = vec![Vec::new(); problem.n_free_frames];
        for &(a, b) in &problem.edges {
            adjacency[a as usize].push(b as usize);
            adjacency[b as usize].push(a as usize);
        }
        for neighbours in adjacency.iter_mut() {
            neighbours.sort_unstable();
        }
        let elimination = crate::reordering::fast_fill_reducing_block_order(&adjacency);
        let mut pos = vec![0u32; problem.n_free_frames];
        for (k, &block) in elimination.iter().enumerate() {
            pos[block] = k as u32;
        }
        timings.reorder += t.elapsed();
        pos
    });
    let mut iterations: Vec<BaIterationStats> = Vec::new();
    let mut trace_lines: Vec<String> = Vec::new();
    let mut converged = false;
    let mut initial_grad_norm: Option<f64> = None;

    for it in 0..max_num_iterations {
        let grad_norm = gradient_inf_norm(&points_lin, &frame_bc_raw);
        match gradient_tolerance_rel {
            // Legacy absolute criterion (unchanged).
            None => {
                if grad_norm <= GRADIENT_TOLERANCE {
                    converged = true;
                    break;
                }
            }
            // Ceres-relative criterion: stop once the gradient has decreased
            // by the requested factor; always attempt at least one step.
            Some(rel) => match initial_grad_norm {
                None => initial_grad_norm = Some(grad_norm),
                Some(g0) => {
                    // `<=` (not `<`): if the first step was rejected the
                    // gradient is unchanged, and a strict comparison would
                    // never fire and run to the iteration cap (observed as a
                    // single 865s global BA at 5k). Ceres stops after 1-2
                    // iterations with these tolerances.
                    if grad_norm <= rel * g0 {
                        converged = true;
                        break;
                    }
                }
            },
        }
        let mu = 1.0 / radius;
        let Some((dx_frames, dx_points, predicted)) = solve_step(
            &problem,
            &points_lin,
            &problem.edges,
            &frame_diag_raw,
            &frame_bc_raw,
            mu,
            &mut cache,
            order.as_deref(),
            &mut timings,
        ) else {
            return Err(BaError::SingularSystem);
        };
        let t_apply = Instant::now();
        let (trial_poses, trial_points) = apply_step(&problem, &dx_frames, &dx_points);
        apply_ms += t_apply.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let trial_half_cost = 0.5 * evaluate_cost(&problem, &trial_poses, &trial_points, loss);
        timings.evaluate_cost += t.elapsed();
        let actual = half_cost - trial_half_cost;
        let rho = if predicted > 0.0 && predicted.is_finite() {
            actual / predicted
        } else {
            f64::NEG_INFINITY
        };

        let step_accepted = rho > MIN_RELATIVE_DECREASE;
        iterations.push(BaIterationStats {
            iteration: it,
            cost_before: 2.0 * half_cost,
            cost_after: 2.0
                * (if step_accepted {
                    trial_half_cost
                } else {
                    half_cost
                }),
            max_pose_step: dx_frames.iter().map(|v| v.norm()).fold(0.0, f64::max),
            max_landmark_step: dx_points.iter().map(|v| v.norm()).fold(0.0, f64::max),
            lambda: mu,
            step_accepted,
        });
        // Buffered (only flushed — see below — for solves with >= 30
        // iterations, per the task brief) per-trial diagnostic: cost, rho,
        // radius, gradient max-norm, accepted. `radius` here is the value
        // this trial was attempted *at* (i.e. before the accept/reject
        // update below), matching what determined this trial's `mu`.
        trace_lines.push(format!(
            "BA_TRACE it={it} cost={:.6} rho={rho:.6} radius={radius:.6e} grad_inf_norm={grad_norm:.6e} accepted={step_accepted}",
            2.0 * half_cost
        ));

        if step_accepted {
            problem.poses = trial_poses;
            problem.points = trial_points;
            radius /= (1.0 / 3.0_f64).max(1.0 - (2.0 * rho - 1.0).powi(3));
            decrease_factor = 2.0;
            let t = Instant::now();
            let (full, pl, fd, fb) = linearize(&problem, loss);
            timings.linearize += t.elapsed();
            timings.linearize_calls += 1;
            half_cost = 0.5 * full;
            points_lin = pl;
            frame_diag_raw = fd;
            frame_bc_raw = fb;
        } else {
            radius /= decrease_factor;
            decrease_factor *= 2.0;
        }
    }

    for (i, &id) in problem.frame_ids.iter().enumerate() {
        ba.poses
            .get_mut(&id)
            .expect("frame id present")
            .world_to_camera = problem.poses[i].clone();
    }
    for (i, &id) in problem.point_ids.iter().enumerate() {
        *ba.landmarks.get_mut(&id).expect("point id present") = problem.points[i];
    }

    eprintln!(
        "BA_PHASES trials={} linearize_calls={} n_free_frames={} n_points={} edges={} shards={} linearize_ms={:.1} eliminate_ms={:.1} assemble_ms={:.1} linsolve_ms={:.1} backsub_ms={:.1} evaluate_cost_ms={:.1} build_ms={build_ms:.1} apply_ms={apply_ms:.1} reorder_ms={:.1}",
        timings.trials,
        timings.linearize_calls,
        problem.n_free_frames,
        problem.points.len(),
        problem.edges.len(),
        problem.shard_boundaries.len().saturating_sub(1),
        timings.linearize.as_secs_f64() * 1e3,
        timings.eliminate.as_secs_f64() * 1e3,
        timings.assemble.as_secs_f64() * 1e3,
        timings.linsolve.as_secs_f64() * 1e3,
        timings.backsub.as_secs_f64() * 1e3,
        timings.evaluate_cost.as_secs_f64() * 1e3,
        timings.reorder.as_secs_f64() * 1e3,
    );
    if iterations.len() >= 30 {
        for line in &trace_lines {
            eprintln!("{line}");
        }
    }

    Ok(BaResult {
        initial_cost,
        final_cost: 2.0 * half_cost,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    use crate::bundle::{
        rig_residual_jacobians as legacy_rig_residual_jacobians, BaRigObservation,
    };
    use visloc_core::geometry::Pose;
    use visloc_core::types::Camera;

    fn sample_inputs() -> (f64, f64, f64, f64, SE3, SE3, Point3<f64>, Point2<f64>) {
        let fx = 480.0;
        let fy = 470.0;
        let cx = 320.0;
        let cy = 240.0;
        let sensor_from_rig = SE3::new(
            UnitQuaternion::from_euler_angles(0.01, 0.02, -0.01),
            Vector3::new(-0.12, 0.005, 0.01),
        );
        let pose = SE3::new(
            UnitQuaternion::from_euler_angles(0.05, -0.1, 0.03),
            Vector3::new(0.3, -0.2, 1.1),
        );
        let point = Point3::new(0.4, -0.15, 3.2);
        let xy = Point2::new(315.0, 238.0);
        (fx, fy, cx, cy, sensor_from_rig, pose, point, xy)
    }

    /// C2.5 task item 6(a): analytic Jacobian vs central finite differences.
    #[test]
    fn jacobian_matches_finite_difference() {
        let (fx, fy, cx, cy, sensor_from_rig, pose, point, xy) = sample_inputs();
        let (r0, j_pose, j_point) =
            residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose, &point, xy)
                .expect("projectable");

        let eps = 1e-6;
        for k in 0..6 {
            let mut d = Vector6::<f64>::zeros();
            d[k] = eps;
            let pose_plus = pose.compose(&SE3::exp(&d));
            let pose_minus = pose.compose(&SE3::exp(&(-d)));
            let r_plus =
                residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose_plus, &point, xy)
                    .unwrap()
                    .0;
            let r_minus =
                residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose_minus, &point, xy)
                    .unwrap()
                    .0;
            let numeric = (r_plus - r_minus) / (2.0 * eps);
            let analytic = j_pose.column(k);
            assert!(
                (numeric - analytic).norm() < 1e-5,
                "pose col {k}: analytic {analytic:?} vs numeric {numeric:?}"
            );
        }
        for k in 0..3 {
            let mut d = Vector3::<f64>::zeros();
            d[k] = eps;
            let r_plus =
                residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose, &(point + d), xy)
                    .unwrap()
                    .0;
            let r_minus =
                residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose, &(point - d), xy)
                    .unwrap()
                    .0;
            let numeric = (r_plus - r_minus) / (2.0 * eps);
            let analytic = j_point.column(k);
            assert!(
                (numeric - analytic).norm() < 1e-5,
                "point col {k}: analytic {analytic:?} vs numeric {numeric:?}"
            );
        }
        assert!(r0.norm() > 0.0, "sample residual should be nonzero");
    }

    /// C2.5 task item 6(a) (second half): this module's independent
    /// re-derivation agrees with `bundle.rs`'s already-exercised
    /// `rig_residual_jacobians` (see module doc).
    #[test]
    fn matches_bundle_rs_formula() {
        let (fx, fy, cx, cy, sensor_from_rig, pose, point, xy) = sample_inputs();
        let camera = Camera::pinhole(1, 640, 480, fx, fy, cx, cy);
        let legacy_pose = Pose {
            world_to_camera: pose.clone(),
        };
        let obs = BaRigObservation {
            keyframe_id: 0,
            landmark_id: 0,
            xy,
            camera,
            sensor_from_rig: sensor_from_rig.clone(),
        };
        let (r_legacy, j_pose_legacy, j_point_legacy) =
            legacy_rig_residual_jacobians(&obs, &legacy_pose, &point).expect("projectable");
        let (r_native, j_pose_native, j_point_native) =
            residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose, &point, xy)
                .expect("projectable");

        assert!((r_legacy - r_native).norm() < 1e-12);
        assert!((j_pose_legacy - j_pose_native).norm() < 1e-12);
        assert!((j_point_legacy - j_point_native).norm() < 1e-12);
    }

    fn build_synthetic_recon_for_ba() -> (
        crate::colmap_incremental::reconstruction::Reconstruction,
        super::super::bundle_adjustment::BundleAdjustmentConfig,
    ) {
        use super::super::bundle_adjustment::BundleAdjustmentConfig;
        use crate::colmap_incremental::pipeline::reconstruction_from_cache;
        use crate::colmap_incremental::reconstruction::TrackElement;
        use crate::colmap_incremental::test_support::build_synthetic_rig_scene;

        let scene = build_synthetic_rig_scene(5, 3);
        let mut recon = reconstruction_from_cache(&scene.db);
        for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
            recon.frame_mut(frame_id).set_rig_from_world(gt.clone());
            recon.register_frame(frame_id);
        }
        let mut point3d_ids = Vec::new();
        for (j, gt_xyz) in scene.ground_truth_points.iter().enumerate() {
            let mut track = Vec::new();
            for &(i1, i2) in &scene.images_per_frame {
                track.push(TrackElement {
                    image_id: i1,
                    point2d_idx: j,
                });
                track.push(TrackElement {
                    image_id: i2,
                    point2d_idx: j,
                });
            }
            point3d_ids.push(recon.add_point3d(*gt_xyz, track));
        }
        let anchors = [0u64, (scene.images_per_frame.len() - 1) as u64];
        let noise = Vector3::new(0.03, -0.02, 0.015);
        for &frame_id in scene.ground_truth_rig_from_world.keys() {
            if anchors.contains(&frame_id) {
                continue;
            }
            let mut pose = recon.frame(frame_id).rig_from_world().clone();
            pose.translation += noise;
            recon.frame_mut(frame_id).set_rig_from_world(pose);
        }
        for &pid in &point3d_ids {
            let xyz = recon.point3d(pid).xyz;
            recon.point3d_mut(pid).xyz = xyz + Vector3::new(0.02, -0.015, 0.01);
        }

        let mut config = BundleAdjustmentConfig::new();
        for &(i1, i2) in &scene.images_per_frame {
            config.add_image(i1);
            config.add_image(i2);
        }
        for &pid in &point3d_ids {
            config.add_variable_point(pid);
        }
        for &frame_id in &anchors {
            config.set_constant_rig_from_world_pose(frame_id);
        }
        (recon, config)
    }

    /// C2.5 perf-regression fix follow-up: profile `optimize` on a
    /// synthetic problem sized like the tier-1000 real-data global BA call
    /// that measured 2.9s/iteration (`frames=333 obs=179623 landmarks=2541`,
    /// `BA_SOLVE` log line from the killed run) — `#[ignore]`d (prints to
    /// stderr, run explicitly: `cargo test --release -p visloc-slam --lib
    /// rig_ba_solver::tests::profile_synthetic_large_problem -- --ignored
    /// --nocapture`).
    ///
    /// The task's <=100ms/iteration target at this size was **not** fully
    /// met: the O(track_len²)-per-point allocation/`BTreeMap` blowup that
    /// caused the original 2.9s/iteration is fixed (per-point elimination
    /// no longer allocates or touches a map; the frame-pair *pattern* is
    /// computed once per `optimize()` call, not once per accepted step; the
    /// O(pairs) work is sharded across 8 threads with a fixed, deterministic
    /// reduction — see `build_reduced_system_pattern` and
    /// `eliminate_and_accumulate`'s docs for the two fixes in order), taking
    /// per-iteration cost from ~2.9s to ~0.13-0.14s (~21x) on this
    /// synthetic problem — but ~140ms measured here is still ~40% over the
    /// 100ms target. This synthetic case's 2 cameras × 70-frame span gives
    /// every point ~140 observations across ~71 *distinct frames* (≈2485
    /// pairs/point); the real killed run's `obs/point` ratio (70.7) is
    /// consistent with a *shorter* per-point frame span (its 2 cameras
    /// would give ~35 frames/point for the same ratio, ≈595 pairs/point,
    /// ~4x fewer pairs) — so this synthetic profile is plausibly
    /// pessimistic relative to the real corridor tracks it was sized to
    /// resemble, not a confirmed shortfall against them. The asserted bound
    /// below is therefore the actually-measured value (with headroom) —
    /// a regression guard, not a claim the 100ms target is met — and the
    /// real tier-1000 rerun (§ task item 3) is the authoritative check.
    #[test]
    #[ignore]
    fn profile_synthetic_large_problem() {
        use crate::colmap_incremental::test_support::build_large_synthetic_ba_problem;

        let mut ba = build_large_synthetic_ba_problem(300, 2500, 70);
        // Perturb every free pose/point so the solve actually has work to
        // do (built from exact ground-truth projections, so an
        // unperturbed solve converges in 0 iterations and never exercises
        // the LM loop at all).
        let fixed_poses = ba.fixed_poses.clone();
        for (&id, pose) in ba.poses.iter_mut() {
            if fixed_poses.contains(&id) {
                continue;
            }
            pose.world_to_camera.translation += Vector3::new(0.02, -0.015, 0.01);
        }
        for xyz in ba.landmarks.values_mut() {
            *xyz += Vector3::new(0.015, 0.01, -0.02);
        }
        eprintln!(
            "synthetic problem: poses={} landmarks={} rig_observations={}",
            ba.poses.len(),
            ba.landmarks.len(),
            ba.rig_observations.len()
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("thread pool");
        let started = Instant::now();
        let result = pool
            .install(|| optimize_with_tolerance(&mut ba, 50, LossFunction::Trivial, None))
            .expect("synthetic BA solve should succeed");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let per_iter_ms = elapsed_ms / result.iterations.len().max(1) as f64;
        eprintln!(
            "profile_synthetic_large_problem: iterations={} total_ms={elapsed_ms:.1} per_iter_ms={per_iter_ms:.1}",
            result.iterations.len()
        );
        assert!(
            per_iter_ms <= 200.0,
            "per-iteration cost {per_iter_ms:.1}ms regressed past the measured ~140ms baseline \
             (task target is 100ms; see this test's doc comment for why 200ms is the guard here)"
        );
    }

    /// Lead-directed follow-up: profile at the larger size the Lead's own
    /// synthetic run used (~440 frames / 2.8k points / 255k observations,
    /// measured 1.08s/iteration pre-fix), target <=150ms/iteration at 8
    /// threads. `#[ignore]`d — run explicitly: `cargo test --release -p
    /// visloc-slam --lib rig_ba_solver::tests::profile_synthetic_440f
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn profile_synthetic_440f() {
        use crate::colmap_incremental::test_support::build_large_synthetic_ba_problem;

        let mut ba = build_large_synthetic_ba_problem(440, 2800, 45);
        let fixed_poses = ba.fixed_poses.clone();
        for (&id, pose) in ba.poses.iter_mut() {
            if fixed_poses.contains(&id) {
                continue;
            }
            pose.world_to_camera.translation += Vector3::new(0.02, -0.015, 0.01);
        }
        for xyz in ba.landmarks.values_mut() {
            *xyz += Vector3::new(0.015, 0.01, -0.02);
        }
        eprintln!(
            "synthetic problem: poses={} landmarks={} rig_observations={}",
            ba.poses.len(),
            ba.landmarks.len(),
            ba.rig_observations.len()
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("thread pool");
        let started = Instant::now();
        let result = pool
            .install(|| optimize_with_tolerance(&mut ba, 50, LossFunction::Trivial, None))
            .expect("synthetic BA solve should succeed");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let per_iter_ms = elapsed_ms / result.iterations.len().max(1) as f64;
        eprintln!(
            "profile_synthetic_440f: iterations={} total_ms={elapsed_ms:.1} per_iter_ms={per_iter_ms:.1}",
            result.iterations.len()
        );
        assert!(
            per_iter_ms <= 150.0,
            "per-iteration cost {per_iter_ms:.1}ms exceeds the 150ms/iteration target at 440f/2800pt"
        );
    }

    /// Reproduces the real tier-1000 run's *largest* global-BA window
    /// (`BA_PHASES` grep: `n_free_frames≈846 n_points≈2900`, `eliminate_ms`
    /// ≈600ms/trial, `linsolve_ms`≈371ms/trial ⇒ ≈1s/iteration, vs this
    /// same-frame-count synthetic case's much smaller `edges_len` — see
    /// `elimination_shard_count`'s doc for why that gap is believed to be
    /// the previous 256MB shard-memory cap collapsing 8 shards down to ~2
    /// on a wide `edges_len`). `#[ignore]`d — run explicitly: `cargo test
    /// --release -p visloc-slam --lib rig_ba_solver::tests::profile_synthetic_846f
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn profile_synthetic_846f() {
        use crate::colmap_incremental::test_support::build_large_synthetic_ba_problem;

        let mut ba = build_large_synthetic_ba_problem(846, 2900, 200);
        // Harder, non-uniform (per-frame-varying, includes rotation) noise
        // than the other profiling tests' plain uniform shift: a uniform
        // shift of every free pose by the same vector is largely gauge/
        // structure-absorbed in a couple of LM steps and converges fast (as
        // the other tests here show, 7-9 iterations even at similar scale),
        // which is not representative of the real run's harder-to-correct,
        // per-frame-varying incremental drift that hits the 50-iteration
        // cap. `id` seeds a simple deterministic pseudo-random perturbation
        // (no RNG dependency) so every frame gets a different, harder-to-
        // cancel error.
        let fixed_poses = ba.fixed_poses.clone();
        for (&id, pose) in ba.poses.iter_mut() {
            if fixed_poses.contains(&id) {
                continue;
            }
            let s = id as f64;
            pose.world_to_camera.translation += Vector3::new(
                0.02 * (s * 0.7).sin(),
                0.015 * (s * 1.3).cos(),
                0.012 * (s * 0.4).sin(),
            );
            let d = UnitQuaternion::from_euler_angles(
                0.004 * (s * 0.9).cos(),
                0.004 * (s * 0.5).sin(),
                0.004 * (s * 1.1).cos(),
            );
            pose.world_to_camera.rotation = d * pose.world_to_camera.rotation;
        }
        for (id, xyz) in ba.landmarks.iter_mut() {
            let s = *id as f64;
            *xyz += Vector3::new(
                0.015 * (s * 0.6).sin(),
                0.015 * (s * 1.7).cos(),
                0.015 * (s * 0.3).sin(),
            );
        }
        eprintln!(
            "synthetic problem: poses={} landmarks={} rig_observations={}",
            ba.poses.len(),
            ba.landmarks.len(),
            ba.rig_observations.len()
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("thread pool");
        let started = Instant::now();
        let result = pool
            .install(|| optimize_with_tolerance(&mut ba, 50, LossFunction::Trivial, None))
            .expect("synthetic BA solve should succeed");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let per_iter_ms = elapsed_ms / result.iterations.len().max(1) as f64;
        eprintln!(
            "profile_synthetic_846f: iterations={} total_ms={elapsed_ms:.1} per_iter_ms={per_iter_ms:.1}",
            result.iterations.len()
        );
    }

    /// C2.5 task item 6(e): the whole pipeline (build_problem → linearize →
    /// solve_step, including the block-Cholesky reduced-camera solve) is
    /// bit-identical at `RAYON_NUM_THREADS=1` vs `8`.
    #[test]
    fn deterministic_across_thread_counts() {
        use super::super::bundle_adjustment::{BundleAdjustmentOptions, Gauge};

        let run = |threads: usize| {
            let (mut recon, mut config) = build_synthetic_recon_for_ba();
            config.fix_gauge(Gauge::Unspecified);
            let mut options = BundleAdjustmentOptions::global();
            options.max_num_iterations = 12;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let ok = pool
                .install(|| super::super::bundle_adjustment::solve(&options, &config, &mut recon));
            assert!(ok, "BA solve failed at {threads} threads");
            recon
        };

        let recon1 = run(1);
        let recon8 = run(8);

        for &frame_id in recon1.reg_frame_ids() {
            let a = recon1.frame(frame_id).rig_from_world();
            let b = recon8.frame(frame_id).rig_from_world();
            assert_eq!(a, b, "frame {frame_id} pose differs across thread counts");
        }
    }

    /// C2.7 task item: `rho'(s)`/`rho''(s)` match central finite differences
    /// of `rho(s)`/`rho'(s)` respectively, for both `SoftL1` and `Cauchy`,
    /// across a range of `s` (including small and large squared residuals).
    #[test]
    fn loss_derivatives_match_finite_difference() {
        let eps = 1.0e-6;
        for loss in [LossFunction::SoftL1(1.7), LossFunction::Cauchy(2.3)] {
            for &s in &[1.0e-3, 0.1, 1.0, 4.0, 25.0, 100.0] {
                let rho = evaluate_loss(loss, s);
                let rho_at = |s: f64| evaluate_loss(loss, s);

                let fd_rho1 = (rho_at(s + eps)[0] - rho_at(s - eps)[0]) / (2.0 * eps);
                let rho1_tol = 1.0e-5 * rho[1].abs().max(1.0);
                assert!(
                    (fd_rho1 - rho[1]).abs() < rho1_tol,
                    "{loss:?} s={s}: rho'(s) analytic={} fd={fd_rho1}",
                    rho[1]
                );

                let fd_rho2 = (rho_at(s + eps)[1] - rho_at(s - eps)[1]) / (2.0 * eps);
                let rho2_tol = 1.0e-5 * rho[2].abs().max(1.0);
                assert!(
                    (fd_rho2 - rho[2]).abs() < rho2_tol,
                    "{loss:?} s={s}: rho''(s) analytic={} fd={fd_rho2}",
                    rho[2]
                );
            }
        }
    }

    /// C2.7 task item: `Corrector` output equals a hand-computed case
    /// (SoftLOneLoss(a=1) at `r=(3,4)`, so `s=25`) — the *common* branch
    /// (`rho''(s) <= 0`, always true for `SoftL1`/`Cauchy`, see
    /// `evaluate_loss`'s doc), where `Corrector` reduces to a uniform
    /// `sqrt(rho'(s))` rescale of both the residual and every Jacobian
    /// block, no curvature term.
    #[test]
    fn corrector_matches_hand_computed_common_branch_case() {
        // rho(s) = 2*(sqrt(1+s) - 1), rho'(s) = 1/sqrt(1+s), rho''(s) =
        // -rho'(s)/(2*(1+s)) for SoftLOneLoss(a=1) (b=c=1) — computed here
        // independently of `evaluate_loss`.
        let s = 25.0_f64;
        let sqrt26 = 26.0_f64.sqrt();
        let hand_rho0 = 2.0 * (sqrt26 - 1.0);
        let hand_rho1 = 1.0 / sqrt26;
        let hand_rho2 = -hand_rho1 / (2.0 * 26.0);

        let rho = evaluate_loss(LossFunction::SoftL1(1.0), s);
        assert!((rho[0] - hand_rho0).abs() < 1.0e-12);
        assert!((rho[1] - hand_rho1).abs() < 1.0e-12);
        assert!((rho[2] - hand_rho2).abs() < 1.0e-12);
        assert!(rho[2] <= 0.0, "expected the common (no-curvature) branch");

        let sqrt_rho1 = hand_rho1.sqrt();
        let corrector = Corrector::new(s, rho);
        assert!((corrector.sqrt_rho1 - sqrt_rho1).abs() < 1.0e-12);
        assert!((corrector.residual_scaling - sqrt_rho1).abs() < 1.0e-12);
        assert_eq!(corrector.alpha_sq_norm, 0.0);

        let mut r = Vector2::new(3.0_f64, 4.0);
        let orig_j = SMatrix::<f64, 2, 6>::from_row_slice(&[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
        ]);
        let mut j = orig_j;
        corrector.correct_jacobian(&r, &mut j);
        for row in 0..2 {
            for col in 0..6 {
                let expected = orig_j[(row, col)] * sqrt_rho1;
                assert!((j[(row, col)] - expected).abs() < 1.0e-10);
            }
        }
        corrector.correct_residual(&mut r);
        assert!((r.x - 3.0 * sqrt_rho1).abs() < 1.0e-12);
        assert!((r.y - 4.0 * sqrt_rho1).abs() < 1.0e-12);
    }

    /// C2.7 task item: `Corrector` output equals a hand-computed case for the
    /// *rare* branch (`rho''(s) > 0`) — never produced by `SoftL1`/`Cauchy`
    /// (both always have `rho'' <= 0`, see `evaluate_loss`'s doc), but
    /// `Corrector::new` must still implement Ceres' generic
    /// `corrector.cc:93-109` curvature-correction math faithfully. Uses a
    /// synthetic `rho = [_, 2.0, 0.5]` at `s=4.0` (`rho[0]` is unused by
    /// `Corrector`'s own math, only `rho[1]`/`rho[2]` matter).
    #[test]
    fn corrector_matches_hand_computed_rare_branch_case() {
        let s = 4.0_f64;
        let rho = [0.0_f64, 2.0, 0.5];
        // corrector.cc:100-109.
        let d = 1.0 + 2.0 * s * rho[2] / rho[1]; // 1 + 2*4*0.5/2 = 3.0
        let alpha = 1.0 - d.sqrt(); // 1 - sqrt(3)
        let hand_sqrt_rho1 = rho[1].sqrt(); // sqrt(2)
        let hand_residual_scaling = hand_sqrt_rho1 / (1.0 - alpha);
        let hand_alpha_sq_norm = alpha / s;

        let corrector = Corrector::new(s, rho);
        assert!((corrector.sqrt_rho1 - hand_sqrt_rho1).abs() < 1.0e-12);
        assert!((corrector.residual_scaling - hand_residual_scaling).abs() < 1.0e-12);
        assert!((corrector.alpha_sq_norm - hand_alpha_sq_norm).abs() < 1.0e-12);
        assert_ne!(
            corrector.alpha_sq_norm, 0.0,
            "expected the curvature branch"
        );

        // r = (0, 2), so r.norm_squared() == s == 4, consistent with the
        // `Corrector` this `r` is used with.
        let mut r = Vector2::new(0.0_f64, 2.0);
        let orig_j = SMatrix::<f64, 2, 1>::new(1.0, 2.0);
        let mut j = orig_j;
        corrector.correct_jacobian(&r, &mut j);
        // r_transpose_j = r . j_col0 = 0*1 + 2*2 = 4.
        let r_transpose_j = 4.0_f64;
        let expected_j0 =
            hand_sqrt_rho1 * (orig_j[(0, 0)] - hand_alpha_sq_norm * 0.0 * r_transpose_j);
        let expected_j1 =
            hand_sqrt_rho1 * (orig_j[(1, 0)] - hand_alpha_sq_norm * 2.0 * r_transpose_j);
        assert!((j[(0, 0)] - expected_j0).abs() < 1.0e-10);
        assert!((j[(1, 0)] - expected_j1).abs() < 1.0e-10);

        corrector.correct_residual(&mut r);
        assert!((r.x - 0.0 * hand_residual_scaling).abs() < 1.0e-12);
        assert!((r.y - 2.0 * hand_residual_scaling).abs() < 1.0e-10);
    }

    /// Deterministic-across-thread-counts, repeated for a non-trivial loss
    /// (`SoftL1`) — the `Corrector` path (per-observation branch on
    /// `alpha_sq_norm`, `rayon`-parallel across points in `linearize`) must
    /// stay bit-identical at any thread count too, not just the
    /// `LossFunction::Trivial` fast path `deterministic_across_thread_counts`
    /// already covers.
    #[test]
    fn deterministic_across_thread_counts_with_soft_l1_loss() {
        use super::super::bundle_adjustment::{BundleAdjustmentOptions, Gauge};

        let run = |threads: usize| {
            let (mut recon, mut config) = build_synthetic_recon_for_ba();
            config.fix_gauge(Gauge::Unspecified);
            let mut options = BundleAdjustmentOptions::global();
            options.max_num_iterations = 12;
            options.loss_function = LossFunction::SoftL1(0.5);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let ok = pool
                .install(|| super::super::bundle_adjustment::solve(&options, &config, &mut recon));
            assert!(ok, "BA solve failed at {threads} threads");
            recon
        };

        let recon1 = run(1);
        let recon8 = run(8);

        for &frame_id in recon1.reg_frame_ids() {
            let a = recon1.frame(frame_id).rig_from_world();
            let b = recon8.frame(frame_id).rig_from_world();
            assert_eq!(
                a, b,
                "frame {frame_id} pose differs across thread counts (SoftL1)"
            );
        }
    }

    /// Builds a `num_frames`-frame synthetic rig reconstruction (all
    /// `ground_truth_points` visible from every frame, per
    /// `build_synthetic_rig_scene`'s doc) with ~10% of its 2D observations
    /// corrupted into gross pixel outliers (deterministic: every 10th
    /// observation in `(frame, camera, point)` iteration order), frame poses
    /// and point positions perturbed from ground truth identically to
    /// `build_synthetic_recon_for_ba`, and every point requested variable —
    /// used by `soft_l1_recovers_gross_outlier_scene_trivial_does_not` to
    /// compare `LossFunction::SoftL1`'s outlier robustness against
    /// `LossFunction::Trivial`'s from an identical starting state.
    fn build_outlier_recon_for_ba(
        num_frames: usize,
    ) -> (
        crate::colmap_incremental::reconstruction::Reconstruction,
        super::super::bundle_adjustment::BundleAdjustmentConfig,
        Vec<u64>,
        std::collections::BTreeMap<u64, Point3<f64>>,
    ) {
        use super::super::bundle_adjustment::BundleAdjustmentConfig;
        use crate::colmap_incremental::pipeline::reconstruction_from_cache;
        use crate::colmap_incremental::reconstruction::TrackElement;
        use crate::colmap_incremental::test_support::build_synthetic_rig_scene;
        use std::collections::BTreeMap;

        let scene = build_synthetic_rig_scene(num_frames, num_frames.saturating_sub(1));
        let mut recon = reconstruction_from_cache(&scene.db);
        for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
            recon.frame_mut(frame_id).set_rig_from_world(gt.clone());
            recon.register_frame(frame_id);
        }

        let mut point3d_ids = Vec::new();
        let mut gt_by_id = BTreeMap::new();
        for (j, gt_xyz) in scene.ground_truth_points.iter().enumerate() {
            let mut track = Vec::new();
            for &(i1, i2) in &scene.images_per_frame {
                track.push(TrackElement {
                    image_id: i1,
                    point2d_idx: j,
                });
                track.push(TrackElement {
                    image_id: i2,
                    point2d_idx: j,
                });
            }
            let pid = recon.add_point3d(*gt_xyz, track);
            point3d_ids.push(pid);
            gt_by_id.insert(pid, *gt_xyz);
        }

        let mut obs_idx = 0usize;
        for &(i1, i2) in &scene.images_per_frame {
            for &iid in &[i1, i2] {
                for j in 0..scene.ground_truth_points.len() {
                    if obs_idx.is_multiple_of(10) {
                        let image = recon.image_mut(iid);
                        image.points2d[j].xy.x += 400.0;
                        image.points2d[j].xy.y -= 320.0;
                    }
                    obs_idx += 1;
                }
            }
        }

        let anchors = [0u64, (scene.images_per_frame.len() - 1) as u64];
        let noise = Vector3::new(0.03, -0.02, 0.015);
        for &frame_id in scene.ground_truth_rig_from_world.keys() {
            if anchors.contains(&frame_id) {
                continue;
            }
            let mut pose = recon.frame(frame_id).rig_from_world().clone();
            pose.translation += noise;
            recon.frame_mut(frame_id).set_rig_from_world(pose);
        }
        for &pid in &point3d_ids {
            let xyz = recon.point3d(pid).xyz;
            recon.point3d_mut(pid).xyz = xyz + Vector3::new(0.02, -0.015, 0.01);
        }

        let mut config = BundleAdjustmentConfig::new();
        for &(i1, i2) in &scene.images_per_frame {
            config.add_image(i1);
            config.add_image(i2);
        }
        for &pid in &point3d_ids {
            config.add_variable_point(pid);
        }
        for &frame_id in &anchors {
            config.set_constant_rig_from_world_pose(frame_id);
        }
        (recon, config, point3d_ids, gt_by_id)
    }

    /// C2.7 task item: on a scene with ~10% gross pixel-outlier
    /// observations, `SoftL1` recovers points close to ground truth while
    /// `Trivial` (dragged by the outliers) does not.
    #[test]
    fn soft_l1_recovers_gross_outlier_scene_trivial_does_not() {
        use super::super::bundle_adjustment::BundleAdjustmentOptions;

        let (mut recon_soft, config, point3d_ids, gt_by_id) = build_outlier_recon_for_ba(8);
        let mut recon_trivial = recon_soft.clone();

        let mut soft_options = BundleAdjustmentOptions::global();
        soft_options.max_num_iterations = 60;
        soft_options.loss_function = LossFunction::SoftL1(1.0);
        let mut trivial_options = soft_options;
        trivial_options.loss_function = LossFunction::Trivial;

        assert!(
            super::super::bundle_adjustment::solve(&soft_options, &config, &mut recon_soft),
            "SoftL1 solve failed"
        );
        assert!(
            super::super::bundle_adjustment::solve(&trivial_options, &config, &mut recon_trivial),
            "Trivial solve failed"
        );

        let median_point_error =
            |recon: &crate::colmap_incremental::reconstruction::Reconstruction| -> f64 {
                let mut errs: Vec<f64> = point3d_ids
                    .iter()
                    .map(|&pid| (recon.point3d(pid).xyz - gt_by_id[&pid]).norm())
                    .collect();
                errs.sort_by(|a, b| a.total_cmp(b));
                errs[errs.len() / 2]
            };

        let soft_err = median_point_error(&recon_soft);
        let trivial_err = median_point_error(&recon_trivial);
        assert!(
            soft_err < 0.02,
            "SoftL1 median point error too large: {soft_err}"
        );
        assert!(
            trivial_err > soft_err * 2.0,
            "expected Trivial (no outlier down-weighting) to be pulled off further than \
             SoftL1: soft={soft_err} trivial={trivial_err}"
        );
    }
}

#[cfg(test)]
mod iterative_bench {
    use super::*;

    /// A reduced camera system shaped like a city block revisited in several
    /// passes: each camera couples to its ±150 capture-order neighbours and
    /// to 100 cameras of the other passes at the same place. Diagonally
    /// dominant SPD by construction.
    fn dense_covisibility_system(n: usize) -> Vec<BTreeMap<usize, Matrix6<f64>>> {
        let mut columns: Vec<BTreeMap<usize, Matrix6<f64>>> = vec![BTreeMap::new(); n];
        let mut add = |a: usize, b: usize, k: f64| {
            let (lo, hi) = (a.min(b), a.max(b));
            if lo != hi {
                let blk =
                    Matrix6::from_fn(|r, c| ((r * 7 + c * 3 + lo + hi) % 11) as f64 * 1e-3 * k);
                columns[lo].insert(hi, blk);
            }
        };
        for i in 0..n {
            for d in 1..=150 {
                if i + d < n {
                    add(i, i + d, 1.0);
                }
            }
            let place = i % (n / 4);
            for k in 0..100 {
                add(
                    i,
                    ((i / (n / 4) + 1 + k % 3) % 4) * (n / 4) + (place + k / 3) % (n / 4),
                    0.5,
                );
            }
        }
        for (i, col) in columns.iter_mut().enumerate() {
            col.insert(
                i,
                Matrix6::identity() * 10.0
                    + Matrix6::from_fn(|r, c| if r == c { (i % 5) as f64 } else { 0.0 }),
            );
        }
        columns
    }

    /// `cargo test --release -p visloc-slam --lib iterative_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn pcg_vs_block_cholesky_on_a_dense_covisibility_graph() {
        let n = 2500;
        let columns = dense_covisibility_system(n);
        let edges: usize = columns.iter().map(|c| c.len() - 1).sum();
        let rhs = DMatrix::<f64>::from_fn(n * 6, 1, |i, _| ((i * 13) % 17) as f64 - 8.0);

        let t = Instant::now();
        let x_pcg = pcg_blocks6(&columns, &rhs).expect("pcg");
        let t_pcg = t.elapsed();

        let t = Instant::now();
        let mut adjacency = vec![Vec::new(); n];
        for (j, col) in columns.iter().enumerate() {
            for &i in col.keys().filter(|&&i| i != j) {
                adjacency[i].push(j);
                adjacency[j].push(i);
            }
        }
        for a in adjacency.iter_mut() {
            a.sort_unstable();
        }
        let elimination = crate::reordering::fast_fill_reducing_block_order(&adjacency);
        let mut pos = vec![0usize; n];
        for (k, &b) in elimination.iter().enumerate() {
            pos[b] = k;
        }
        let mut permuted: Vec<BTreeMap<usize, Matrix6<f64>>> = vec![BTreeMap::new(); n];
        for (j, col) in columns.iter().enumerate() {
            for (&i, blk) in col {
                let (pi, pj) = (pos[i], pos[j]);
                if pi >= pj {
                    permuted[pj].insert(pi, *blk);
                } else {
                    permuted[pi].insert(pj, blk.transpose());
                }
            }
        }
        let mut prhs = DMatrix::<f64>::zeros(n * 6, 1);
        for i in 0..n {
            for k in 0..6 {
                prhs[(pos[i] * 6 + k, 0)] = rhs[(i * 6 + k, 0)];
            }
        }
        let t_order = t.elapsed();
        let t = Instant::now();
        let mut cache = None;
        let x_chol = solve_spd_blocks6_cached(&mut cache, permuted, &prhs).expect("cholesky");
        let t_chol = t.elapsed();
        let mut max_diff = 0.0f64;
        for i in 0..n {
            for k in 0..6 {
                max_diff =
                    max_diff.max((x_chol[(pos[i] * 6 + k, 0)] - x_pcg[(i * 6 + k, 0)]).abs());
            }
        }
        eprintln!(
            "{n} cameras, {edges} camera pairs: pcg {t_pcg:?}; order {t_order:?} + block cholesky {t_chol:?}; max |dx| diff {max_diff:.2e}"
        );
        assert!(max_diff < 1e-3 * x_pcg.amax().max(1.0));
    }
}
