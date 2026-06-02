//! KITTI loop-closure pose-graph demo on real public-data ground-truth poses,
//! with optional outlier-robust back-end evaluation.
//!
//! Loads a KITTI odometry ground-truth pose file (e.g.,
//! `<dataset>/poses/00.txt`), subsamples it to a manageable keyframe set,
//! fabricates a realistic odometry drift by perturbing each sequential edge's
//! yaw, and adds truth-relative loop-closure constraints for every revisit
//! (temporally-distant keyframe pairs that are spatially close in the ground
//! truth — KITTI 00 is the canonical loopy sequence). The full SE(3)
//! Levenberg-Marquardt + Cholesky solver (`PoseGraph::optimize_se3_iterative`)
//! is then run on the resulting graph and the truth / drifted / corrected
//! trajectories are written as CSV files for downstream visualization
//! (`scripts/build_kitti_loop_asset.py`).
//!
//! With `--inject-wrong-loops N` the demo additionally injects `N` *wrong*
//! loop closures (the perceptual-aliasing failure: a place-recognition false
//! positive claims two far-apart keyframes are co-located) and runs BOTH the
//! plain solve and a Graduated-Non-Convexity robust solve on the corrupted
//! graph, reporting the absolute trajectory error (ATE, in metres against the
//! KITTI ground truth) of each plus GNC's outlier recall / false-positive
//! count. This makes GNC's value quantitative on a *real* trajectory: the
//! plain solve is dragged off by the wrong closures; GNC rejects them.
//!
//! With `--pcm` the demo also runs Pairwise Consistency Maximization
//! ([`visloc_rs::slam::pcm`]) as a *front-end* screen: it keeps only the
//! largest mutually-consistent subset of loop closures and runs a plain solve
//! on that cleaned set. On KITTI 00 (truth init) PCM drops every injected wrong
//! loop with zero genuine loops lost and recovers ATE to ~0.01 m at every seed
//! and contamination level — strictly more reliable than the back-end GNC here
//! (GNC's recall/FP are seed-variable). Front-end screening and back-end
//! robustness are complementary.
//!
//! Usage:
//!
//! ```sh
//! cargo run --example online_slam_kitti_loop_demo -- \
//!     --kitti-poses /path/to/KITTI_odometry/poses/00.txt \
//!     --keyframe-stride 30 \
//!     --yaw-drift-deg-per-edge 0.45 \
//!     --inject-wrong-loops 5 \
//!     --out-dir target/kitti_loop_demo
//! ```

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::slam::gnc::{GncConfig, GncKernel, AUTO_SCALE_K};
use visloc_rs::slam::pcm::{self, LoopMeasurement, PcmConfig};
use visloc_rs::tracking::PoseTrajectory;
use visloc_rs::{
    relative_world_to_camera, LoopClosureConstraint, PoseGraph, PoseGraphEdgeKind,
    PoseGraphSe3Config,
};

#[derive(Debug)]
struct CliArgs {
    kitti_poses: PathBuf,
    keyframe_stride: usize,
    yaw_drift_per_edge_rad: f64,
    out_dir: PathBuf,
    max_keyframes: usize,
    inject_wrong_loops: usize,
    inject_seed: u64,
    pcm: bool,
    marginalize_window: Option<usize>,
    covisibility_radius: usize,
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut kitti_poses: Option<PathBuf> = None;
    let mut keyframe_stride: usize = 30;
    let mut yaw_drift_deg: f64 = 0.45;
    let mut out_dir: PathBuf = PathBuf::from("target/kitti_loop_demo");
    let mut max_keyframes: usize = 200;
    let mut inject_wrong_loops: usize = 0;
    let mut inject_seed: u64 = 1;
    let mut pcm = false;
    let mut marginalize_window: Option<usize> = None;
    let mut covisibility_radius: usize = 0;

    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--kitti-poses" => {
                kitti_poses = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--keyframe-stride" => {
                keyframe_stride = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--yaw-drift-deg-per-edge" => {
                yaw_drift_deg = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.remove(i + 1));
                args.remove(i);
            }
            "--max-keyframes" => {
                max_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--inject-wrong-loops" => {
                inject_wrong_loops = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--inject-seed" => {
                inject_seed = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--pcm" => {
                pcm = true;
                args.remove(i);
            }
            "--marginalize-window" => {
                marginalize_window = Some(args.remove(i + 1).parse()?);
                args.remove(i);
            }
            "--covisibility-radius" => {
                covisibility_radius = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let kitti_poses = kitti_poses
        .ok_or("--kitti-poses <path/to/poses/SS.txt> is required (KITTI odometry GT pose file)")?;
    Ok(CliArgs {
        kitti_poses,
        keyframe_stride,
        yaw_drift_per_edge_rad: yaw_drift_deg.to_radians(),
        out_dir,
        max_keyframes,
        inject_wrong_loops,
        inject_seed,
        pcm,
        marginalize_window,
        covisibility_radius,
    })
}

/// Small deterministic linear-congruential RNG (no `rand` dependency, so the
/// injected wrong loops are reproducible across runs and platforms).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Absolute trajectory error (mean / RMSE / max of per-keyframe camera-centre
/// error, in metres). No Umeyama alignment: the anchor (keyframe 0) is
/// gauge-fixed and the truth poses live in the same world frame, so the
/// estimate and truth are directly comparable.
fn ate(estimate: &[Pose], truth: &[Pose]) -> (f64, f64, f64) {
    let errs: Vec<f64> = estimate
        .iter()
        .zip(truth)
        .map(|(e, t)| (e.camera_center_world() - t.camera_center_world()).norm())
        .collect();
    let n = errs.len().max(1) as f64;
    let mean = errs.iter().sum::<f64>() / n;
    let rmse = (errs.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
    let max = errs.iter().copied().fold(0.0_f64, f64::max);
    (mean, rmse, max)
}

/// Simulate an incremental fixed-lag / sliding-window smoother over the
/// keyframes: add each keyframe (drifted pose + sequential edge) plus any loop
/// closures whose *earlier* endpoint is still in the active window, solve the
/// bounded window, then [`PoseGraph::marginalize_oldest`] back to `window`
/// poses. The gauge anchor (keyframe 0) is always retained, so revisits to the
/// origin survive even at a small window; a loop whose earlier endpoint has
/// already been marginalized is *dropped* (its node has left the graph) — the
/// fundamental fixed-lag trade-off: bounded per-solve cost at the price of
/// losing loop closures that span more than the window.
///
/// With `sparsify`, each marginalized blanket prior is replaced by its KL-optimal
/// Chow-Liu tree ([`PoseGraph::marginalize_oldest_sparsified`]) so a window with
/// in-window loop closures (which give a marginalized pose ≥3 neighbours → a dense
/// clique prior) stays sparse instead of accumulating fill-in.
///
/// Returns `(trajectory, loops_applied, loops_dropped, max_active_poses,
/// peak_prior_offdiag_blocks, elapsed)`. Each keyframe's reported pose is its
/// estimate at the last solve before it was marginalized (frozen), or the final
/// estimate if still active.
#[allow(clippy::type_complexity)]
fn windowed_fixed_lag(
    drifted: &[Pose],
    noisy_edges: &[SE3],
    loops: &[LoopClosureConstraint],
    window: usize,
    n: usize,
    sparsify: bool,
) -> Result<(Vec<Pose>, usize, usize, usize, usize, std::time::Duration), Box<dyn std::error::Error>>
{
    use std::collections::BTreeMap;
    let start = std::time::Instant::now();
    // Loops grouped by their later (`to`) endpoint — the step at which both
    // endpoints first exist.
    let mut loops_by_to: BTreeMap<u64, Vec<&LoopClosureConstraint>> = BTreeMap::new();
    for c in loops {
        loops_by_to.entry(c.to_keyframe_id).or_default().push(c);
    }

    let mut graph = PoseGraph::new();
    let mut trajectory: Vec<Pose> = drifted.to_vec();
    let (mut applied, mut dropped, mut max_active) = (0usize, 0usize, 0usize);
    let mut peak_prior_blocks = 0usize;

    for k in 0..n {
        let id = k as u64;
        graph.add_pose(id, drifted[k].clone());
        if k == 0 {
            graph.anchor(0);
        } else {
            graph.add_sequential_edge((k - 1) as u64, id, noisy_edges[k - 1].clone());
        }
        if let Some(cs) = loops_by_to.get(&id) {
            for c in cs {
                if graph.poses.contains_key(&c.from_keyframe_id) {
                    graph.add_loop_closure_constraint(c);
                    applied += 1;
                } else {
                    dropped += 1; // earlier endpoint already marginalized away
                }
            }
        }
        // Solve once there is at least one edge (the first keyframe is the lone
        // anchor with nothing to optimize).
        if !graph.edges.is_empty() {
            graph.optimize_se3_iterative(&pgo_config())?;
        }
        // Freeze the current estimate of every active pose (marginalized ones
        // keep their last write).
        for (&pid, pose) in graph.poses.iter() {
            trajectory[pid as usize] = pose.clone();
        }
        max_active = max_active.max(graph.poses.len());
        if sparsify {
            graph.marginalize_oldest_sparsified(window)?;
        } else {
            graph.marginalize_oldest(window)?;
        }
        // Fill-in carried in the window: total nonzero off-diagonal 6×6 prior
        // blocks across all current priors (a dense clique prior over m poses has
        // m(m−1)/2; its Chow-Liu tree has m−1).
        peak_prior_blocks = peak_prior_blocks.max(count_offdiag_prior_blocks(&graph));
    }
    Ok((
        trajectory,
        applied,
        dropped,
        max_active,
        peak_prior_blocks,
        start.elapsed(),
    ))
}

/// Total number of nonzero off-diagonal 6×6 blocks across all of `graph`'s
/// Gaussian priors — the marginalization fill-in carried in the window.
fn count_offdiag_prior_blocks(graph: &PoseGraph) -> usize {
    let mut total = 0usize;
    for p in &graph.priors {
        let m = p.ids.len();
        for i in 0..m {
            for j in (i + 1)..m {
                let blk = p.information.view((i * 6, j * 6), (6, 6));
                if blk.iter().any(|&v| v.abs() > 1e-9) {
                    total += 1;
                }
            }
        }
    }
    total
}

/// Build `n` *wrong* loop closures (the perceptual-aliasing false positive).
/// Each picks two keyframes that are far apart in the sequence (so they are
/// genuinely at different places) and asserts a near-identity relative pose —
/// "place recognition says these two are the same location". Such a closure has
/// a huge residual and folds the trajectory onto itself unless rejected.
fn make_wrong_loops(n: usize, n_kf: usize, seed: u64) -> Vec<LoopClosureConstraint> {
    let mut rng = Lcg(seed.wrapping_mul(2) | 1);
    let mut wrong = Vec::with_capacity(n);
    let min_gap = (n_kf / 4).max(2);
    let mut guard = 0usize;
    while wrong.len() < n && guard < n * 1000 {
        guard += 1;
        let i = (rng.next_u64() as usize) % n_kf;
        let j = (rng.next_u64() as usize) % n_kf;
        if (i as isize - j as isize).unsigned_abs() < min_gap {
            continue;
        }
        // Near-identity relative pose with a small jitter (not a degenerate
        // exact identity).
        let translation = Vector3::new(
            rng.range(-2.0, 2.0),
            rng.range(-1.0, 1.0),
            rng.range(-2.0, 2.0),
        );
        let axis = Vector3::new(
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        );
        let rotation = nalgebra::Unit::try_new(axis, 1e-9)
            .map(|a| UnitQuaternion::from_axis_angle(&a, rng.range(0.0, 0.2)))
            .unwrap_or_else(UnitQuaternion::identity);
        wrong.push(LoopClosureConstraint {
            from_keyframe_id: i as u64,
            to_keyframe_id: j as u64,
            relative_pose: SE3::new(rotation, translation),
            // High inlier count → trusted like a verified closure (the whole
            // point: it passed verification but is wrong).
            inlier_count: 80,
            inlier_ratio: 0.9,
            mean_sampson_error: 0.0,
            score: 80.0,
        });
    }
    wrong
}

fn pgo_config() -> PoseGraphSe3Config {
    PoseGraphSe3Config {
        initial_lambda: Some(1.0e-3),
        max_iterations: 50,
        // The drifted odometry is a usable seed and the genuine loops are
        // truth-relative, so skip chordal init: chordal rotation least-squares
        // is NOT robust, and even a few wrong loops (claiming a near-identity
        // rotation between truly far-apart frames) dominate the linear average
        // and corrupt the seed, inflating the MAD auto-scale until GNC can no
        // longer tell the wrong loops apart. GNC's own convex-first annealing
        // is the robust initializer here.
        chordal_init: false,
        ..PoseGraphSe3Config::default()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let trajectory = PoseTrajectory::read_kitti_poses(&args.kitti_poses)?;
    let samples = trajectory.samples();
    if samples.is_empty() {
        return Err("KITTI pose file is empty".into());
    }
    println!(
        "kitti_poses={} total_samples={} stride={} max_keyframes={} yaw_drift_per_edge={:.4} rad inject_wrong_loops={}",
        args.kitti_poses.display(),
        samples.len(),
        args.keyframe_stride,
        args.max_keyframes,
        args.yaw_drift_per_edge_rad,
        args.inject_wrong_loops,
    );

    // Subsample with stride; cap at max_keyframes.
    let mut keyframes: Vec<Pose> = samples
        .iter()
        .step_by(args.keyframe_stride.max(1))
        .map(|s| s.pose.clone())
        .collect();
    if keyframes.len() > args.max_keyframes {
        keyframes.truncate(args.max_keyframes);
    }
    let n = keyframes.len();
    if n < 4 {
        return Err(format!("need at least 4 keyframes after subsampling, got {n}").into());
    }
    println!("keyframe_count={n}");

    // Truth sequential edges.
    let mut truth_edges: Vec<SE3> = Vec::with_capacity(n - 1);
    for w in keyframes.windows(2) {
        truth_edges.push(relative_world_to_camera(&w[0], &w[1]));
    }

    // Inject yaw drift on each sequential edge to simulate odometry error.
    let yaw_drift_rot =
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), args.yaw_drift_per_edge_rad);
    let noisy_edges: Vec<SE3> = truth_edges
        .iter()
        .map(|edge| SE3::new(yaw_drift_rot * edge.rotation, edge.translation))
        .collect();

    // Initial drifted node estimates: integrate noisy edges from the anchor.
    let mut drifted: Vec<Pose> = vec![keyframes[0].clone()];
    for edge in &noisy_edges {
        let last = drifted.last().unwrap();
        let new_world_to_camera = edge.compose(&last.world_to_camera);
        drifted.push(Pose {
            world_to_camera: new_world_to_camera,
        });
    }

    // Build the base pose graph from the drifted state.
    let mut graph = PoseGraph::new();
    for (id, pose) in drifted.iter().enumerate() {
        graph.add_pose(id as u64, pose.clone());
    }
    graph.anchor(0);
    for (i, edge) in noisy_edges.iter().enumerate() {
        graph.add_sequential_edge(i as u64, (i + 1) as u64, edge.clone());
    }

    // Genuine loop closures: every temporally-distant pair of keyframes that
    // is *spatially* close in the ground truth is a real revisit (KITTI 00 is
    // the canonical loopy sequence). Add a truth-relative constraint for each.
    // A dense set of genuine loops forms a strong consensus, so chordal init
    // and the GNC convex phase are dominated by the inliers — that consensus
    // is exactly what lets GNC separate the wrong loops from the real ones
    // (with a single genuine loop, its drift-correcting residual is as large
    // as a wrong loop's and the two are indistinguishable by residual).
    let proximity_m = 10.0;
    let min_gap = 10usize;
    let mut loop_constraints: Vec<LoopClosureConstraint> = Vec::new();
    let mut loop_is_wrong: Vec<bool> = Vec::new();
    for i in 0..n {
        let ci = keyframes[i].camera_center_world();
        for j in (i + min_gap)..n {
            let cj = keyframes[j].camera_center_world();
            if (ci - cj).norm() <= proximity_m {
                loop_constraints.push(LoopClosureConstraint {
                    from_keyframe_id: i as u64,
                    to_keyframe_id: j as u64,
                    relative_pose: relative_world_to_camera(&keyframes[i], &keyframes[j]),
                    inlier_count: 100,
                    inlier_ratio: 1.0,
                    mean_sampson_error: 0.0,
                    score: 100.0,
                });
                loop_is_wrong.push(false);
            }
        }
    }
    // Optional genuine COVISIBILITY edges: every keyframe to its near neighbours
    // `i ↔ i+d` for `d` in `2..=radius` (the GT relative pose). These are
    // short-span, so both endpoints coexist inside a fixed-lag window — giving a
    // marginalized pose ≥3 neighbours, hence a DENSE clique prior. This is the
    // realistic covisibility-graph connectivity where marginalization fill-in
    // (and so Chow-Liu sparsification) actually matters; the long-range genuine
    // loops above are dropped by a bounded window before both endpoints coexist.
    for i in 0..n {
        for d in 2..=args.covisibility_radius {
            let j = i + d;
            if j >= n {
                break;
            }
            loop_constraints.push(LoopClosureConstraint {
                from_keyframe_id: i as u64,
                to_keyframe_id: j as u64,
                relative_pose: relative_world_to_camera(&keyframes[i], &keyframes[j]),
                inlier_count: 100,
                inlier_ratio: 1.0,
                mean_sampson_error: 0.0,
                score: 100.0,
            });
            loop_is_wrong.push(false);
        }
    }
    let genuine_loops = loop_constraints.len();
    println!(
        "genuine_loop_closures={genuine_loops} (covisibility_radius={})",
        args.covisibility_radius
    );

    // Optionally corrupt the loop set with wrong loop closures.
    if args.inject_wrong_loops > 0 {
        if args.yaw_drift_per_edge_rad > 1e-9 {
            // The robust-rejection comparison is cleanest from a GOOD init
            // (the standard robust-PGO protocol injects into a near-optimal
            // graph). With large odometry drift on top, the genuine loop
            // closures that must correct the drift carry residuals as large as
            // the wrong loops' — they become indistinguishable by residual, so
            // GNC can only fall back to the (uncorrected) odometry estimate.
            // Run with `--yaw-drift-deg-per-edge 0` for the clean rejection
            // demonstration; this is a documented hard regime, not a bug.
            println!(
                "NOTE: --inject-wrong-loops with yaw-drift > 0 is the HARD regime \
                 (drift-correcting genuine loops look like outliers); \
                 use --yaw-drift-deg-per-edge 0 for the clean robust-rejection result"
            );
        }
        for w in make_wrong_loops(args.inject_wrong_loops, n, args.inject_seed) {
            loop_constraints.push(w);
            loop_is_wrong.push(true);
        }
    }

    // Add every loop closure (genuine + wrong) to the base graph. Edge order is
    // sequential (n-1) then the loops in `loop_constraints` order, so the wrong
    // loops sit at edge index `(n-1) + i` for each wrong `i`.
    for c in &loop_constraints {
        graph.add_loop_closure_constraint(c);
    }
    let injected: Vec<usize> = loop_is_wrong
        .iter()
        .enumerate()
        .filter(|(_, &w)| w)
        .map(|(i, _)| (n - 1) + i)
        .collect();
    println!("se3_cost_before_optimization={:.6}", graph.se3_cost());

    // --- Plain (non-robust) solve ---
    let mut plain_graph = graph.clone();
    let plain_result = plain_graph.optimize_se3_iterative(&pgo_config())?;
    let plain_corrected: Vec<Pose> = (0..n as u64)
        .map(|id| plain_graph.poses[&id].clone())
        .collect();

    println!(
        "[plain] anchor={} edges={} variables={} initial_cost={:.4} final_cost={:.4} iterations={} converged={}",
        plain_result.anchor_id,
        plain_result.edge_count,
        plain_result.variable_count,
        plain_result.initial_cost,
        plain_result.final_cost,
        plain_result.iterations.len(),
        plain_result.converged,
    );

    // --- Trajectory accuracy (ATE in metres, anchor-fixed gauge) ---
    let (d_mean, d_rmse, d_max) = ate(&drifted, &keyframes);
    let (p_mean, p_rmse, p_max) = ate(&plain_corrected, &keyframes);
    println!("ATE drifted      mean={d_mean:.3} rmse={d_rmse:.3} max={d_max:.3} (m)");
    println!("ATE plain-PGO    mean={p_mean:.3} rmse={p_rmse:.3} max={p_max:.3} (m)");

    // --- Fixed-lag / sliding-window smoother (incremental) ---
    // Demonstrates the trade-off on a REAL loopy sequence: a bounded window caps
    // the per-solve cost but drops loop closures that span more than the window
    // (their earlier endpoint has been marginalized). The anchor is always kept,
    // so revisits to the origin still close. Compared against the full batch.
    if let Some(window) = args.marginalize_window {
        // Run the same window DENSE and SPARSE (Chow-Liu) and compare the
        // marginalization fill-in, wall time, and ATE. Sparsification preserves
        // each kept pose's marginal, so the ATE should match while the prior
        // density (and re-solve cost) drops wherever an in-window loop closure
        // made a marginalized pose's blanket a clique.
        let (dense, applied, dropped, max_active, dense_blocks, dense_t) =
            windowed_fixed_lag(&drifted, &noisy_edges, &loop_constraints, window, n, false)?;
        let (sparse, _, _, _, sparse_blocks, sparse_t) =
            windowed_fixed_lag(&drifted, &noisy_edges, &loop_constraints, window, n, true)?;
        let (d_mean, d_rmse, d_max) = ate(&dense, &keyframes);
        let (s_mean, s_rmse, s_max) = ate(&sparse, &keyframes);
        println!(
            "[fixed-lag] window={window} max_active_poses={max_active} (of {n}) loops_applied={applied} loops_dropped={dropped}"
        );
        println!(
            "  dense  : peak_prior_offdiag_blocks={dense_blocks} time={:.0}ms ATE mean={d_mean:.3} rmse={d_rmse:.3} max={d_max:.3} (m)",
            dense_t.as_secs_f64() * 1e3,
        );
        println!(
            "  sparse : peak_prior_offdiag_blocks={sparse_blocks} time={:.0}ms ATE mean={s_mean:.3} rmse={s_rmse:.3} max={s_max:.3} (m)",
            sparse_t.as_secs_f64() * 1e3,
        );
        println!(
            "NOTE: fixed-lag caps the active set to {max_active}/{n} poses; the {dropped} dropped \
             loop(s) span more than the window. Chow-Liu sparsification cuts the prior fill-in \
             {dense_blocks}→{sparse_blocks} off-diagonal blocks (a marginalized covisibility \
             clique → its spanning tree; the gap grows ~quadratically with the covisibility \
             radius) at near-equal ATE ({d_rmse:.3}→{s_rmse:.3} m). The benefit here is STRUCTURAL \
             (density/memory): on this small window the per-marginalization Chow-Liu overhead \
             roughly cancels the solve saving, so wall-time is ~flat — the time win needs a larger \
             window where the dense O(m²) fill dominates the solve. With covisibility_radius=0 the \
             windowed graph is a chain (≤2-pose blankets), so dense==sparse and there is nothing to \
             sparsify — the long-range KITTI loops are dropped before both endpoints share the \
             window."
        );
    }

    // --- Robust (GNC) solve, only meaningful when outliers were injected ---
    let mut gnc_corrected = plain_corrected.clone();
    if !injected.is_empty() {
        let gnc = GncConfig {
            kernel: GncKernel::TruncatedLeastSquares,
            // Tiny floor; the MAD auto-scale lifts the inlier band to the
            // graph's own (drift-induced) residual spread. NO readapt — it
            // over-rejects real edges on pose graphs (a BA-only win).
            c: 0.1,
            auto_scale: Some(AUTO_SCALE_K),
            auto_scale_readapt: false,
            ..GncConfig::default()
        };
        let mut gnc_graph = graph.clone();
        let gnc_result = gnc_graph.optimize_se3_gnc(&pgo_config(), &gnc)?;
        gnc_corrected = (0..n as u64)
            .map(|id| gnc_graph.poses[&id].clone())
            .collect();

        // Score the verdict: a wrong loop is "rejected" at weight < 0.5.
        let injected_rejected = injected
            .iter()
            .filter(|&&e| gnc_result.edge_weights[e] < 0.5)
            .count();
        // Genuine (sequential + the one true loop) edges wrongly rejected.
        let genuine_rejected = gnc_graph
            .edges
            .iter()
            .enumerate()
            .filter(|(e, edge)| {
                !injected.contains(e)
                    && (edge.kind == PoseGraphEdgeKind::Sequential
                        || edge.kind == PoseGraphEdgeKind::LoopClosure)
                    && gnc_result.edge_weights[*e] < 0.5
            })
            .count();

        let (g_mean, g_rmse, g_max) = ate(&gnc_corrected, &keyframes);
        println!(
            "[gnc] kernel=TLS inlier_scale={:.4} outer_iters={} converged={}",
            gnc_result.inlier_scale, gnc_result.outer_iterations, gnc_result.converged,
        );
        println!("ATE gnc-PGO      mean={g_mean:.3} rmse={g_rmse:.3} max={g_max:.3} (m)");
        println!(
            "gnc outlier recall = {injected_rejected}/{} wrong loops rejected; false positives = {genuine_rejected} genuine edges rejected",
            injected.len(),
        );
        println!(
            "summary: drifted rmse {d_rmse:.2}m -> plain {p_rmse:.2}m -> gnc {g_rmse:.2}m \
             (with {} wrong loop closures)",
            injected.len(),
        );
    }

    // --- PCM front-end screen, then a plain solve on the cleaned loop set ---
    let mut pcm_corrected = plain_corrected.clone();
    if args.pcm {
        // Odometry oracle for PCM: the drifted node estimates' world_to_camera
        // poses (truth when drift = 0). PCM uses only the relative odometry
        // between loop endpoints, so it needs no global optimization.
        let odometry: BTreeMap<u64, SE3> = drifted
            .iter()
            .enumerate()
            .map(|(id, pose)| (id as u64, pose.world_to_camera.clone()))
            .collect();
        let pcm_loops: Vec<LoopMeasurement> = loop_constraints
            .iter()
            .map(|c| LoopMeasurement {
                from: c.from_keyframe_id,
                to: c.to_keyframe_id,
                relative: c.relative_pose.clone(),
            })
            .collect();
        let kept = pcm::maximum_consistent_set(&pcm_loops, &odometry, &PcmConfig::default());
        let kept_set: std::collections::HashSet<usize> = kept.iter().copied().collect();

        let wrong_total = loop_is_wrong.iter().filter(|&&w| w).count();
        let wrong_dropped = (0..loop_constraints.len())
            .filter(|i| loop_is_wrong[*i] && !kept_set.contains(i))
            .count();
        let genuine_dropped = (0..loop_constraints.len())
            .filter(|i| !loop_is_wrong[*i] && !kept_set.contains(i))
            .count();

        // Rebuild the graph with sequential edges + ONLY the PCM-kept loops,
        // then a plain (non-robust) solve — the back-end never sees a wrong one.
        let mut pcm_graph = PoseGraph::new();
        for (id, pose) in drifted.iter().enumerate() {
            pcm_graph.add_pose(id as u64, pose.clone());
        }
        pcm_graph.anchor(0);
        for (i, edge) in noisy_edges.iter().enumerate() {
            pcm_graph.add_sequential_edge(i as u64, (i + 1) as u64, edge.clone());
        }
        for &i in &kept {
            pcm_graph.add_loop_closure_constraint(&loop_constraints[i]);
        }
        pcm_graph.optimize_se3_iterative(&pgo_config())?;
        pcm_corrected = (0..n as u64)
            .map(|id| pcm_graph.poses[&id].clone())
            .collect();

        let (m_mean, m_rmse, m_max) = ate(&pcm_corrected, &keyframes);
        println!(
            "[pcm] kept {}/{} loops ({} genuine dropped, {}/{} wrong dropped)",
            kept.len(),
            loop_constraints.len(),
            genuine_dropped,
            wrong_dropped,
            wrong_total,
        );
        println!("ATE pcm+plain    mean={m_mean:.3} rmse={m_rmse:.3} max={m_max:.3} (m)");
        if !injected.is_empty() {
            println!(
                "summary(pcm): drifted rmse {d_rmse:.2}m -> plain {p_rmse:.2}m -> pcm+plain {m_rmse:.2}m \
                 (PCM screened {wrong_dropped}/{wrong_total} wrong loops before the back-end)",
            );
        }
    }

    let corrected = if args.pcm {
        pcm_corrected
    } else if injected.is_empty() {
        plain_corrected
    } else {
        gnc_corrected
    };

    fs::create_dir_all(&args.out_dir)?;
    write_xz_csv(&args.out_dir.join("truth.csv"), &keyframes)?;
    write_xz_csv(&args.out_dir.join("drifted.csv"), &drifted)?;
    write_xz_csv(&args.out_dir.join("corrected.csv"), &corrected)?;

    println!("trajectories written to {}", args.out_dir.display());
    Ok(())
}

fn write_xz_csv(path: &Path, poses: &[Pose]) -> std::io::Result<()> {
    let mut s = String::from("id,x,y,z\n");
    for (i, pose) in poses.iter().enumerate() {
        let c = pose.camera_center_world();
        s.push_str(&format!("{i},{:.6},{:.6},{:.6}\n", c.x, c.y, c.z));
    }
    fs::write(path, s)
}
