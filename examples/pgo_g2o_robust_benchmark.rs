//! Outlier-robust SE(3) pose-graph benchmark on the `.g2o` exchange format.
//!
//! Real SLAM loop closures are sometimes **wrong** (perceptual aliasing,
//! place-recognition false positives). A single bad loop closure pulls a plain
//! least-squares pose-graph solve — and even a Huber/Cauchy IRLS solve — into a
//! corrupted basin. This benchmark makes that failure, and Graduated
//! Non-Convexity's recovery, quantitative on canonical data: it loads a real
//! `.g2o` graph, **injects** a controlled number of random wrong loop closures
//! (the standard robust-PGO evaluation protocol used by Switchable
//! Constraints / DCS / GNC papers), and compares four back-ends on the
//! corrupted graph:
//!
//! 1. `L2`        — no robustness ([`RobustKernel::None`]).
//! 2. `Huber`     — the existing IRLS M-estimator.
//! 3. `GNC-GM`    — Graduated Non-Convexity, Geman-McClure surrogate.
//! 4. `GNC-TLS`   — Graduated Non-Convexity, truncated-least-squares surrogate.
//!
//! The discriminating metric is the χ² over the **original** (inlier) edges
//! only: if a solver rejects the outliers it matches the outlier-free baseline;
//! if it is corrupted, this χ² is inflated. The GNC runs additionally report
//! how many injected outliers they rejected (recall) and how many genuine
//! edges they wrongly rejected (false positives).
//!
//! Usage:
//!
//! ```text
//! # Real benchmark dataset (download a .g2o first, e.g. sphere2500):
//! cargo run --release --example pgo_g2o_robust_benchmark -- --inject 30 path/to/sphere2500.g2o
//!
//! # Tune the inlier scale c (Mahalanobis residual-norm threshold) and the seed:
//! cargo run --release --example pgo_g2o_robust_benchmark -- --inject 30 --c 3 --seed 7 path/to/rim.g2o
//!
//! # No path: a built-in deterministic synthetic loop graph, so the demo runs
//! # end-to-end in CI without any external data.
//! cargo run --example pgo_g2o_robust_benchmark
//! ```

use nalgebra::{Matrix6, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};
use visloc_slam::gnc::{GncConfig, GncKernel};
use visloc_slam::{
    read_g2o, relative_world_to_camera, LinearSolver, PoseGraph, PoseGraphEdgeKind,
    PoseGraphSe3Config, RobustKernel,
};

/// Small deterministic linear-congruential RNG (no `rand` dependency, so the
/// injected outliers are reproducible across runs and platforms).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        // PCG/MMIX-style constants.
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

/// Format a χ²-to-baseline ratio, guarding against a near-zero baseline (the
/// synthetic fixture converges to essentially zero, which would print a
/// meaningless astronomically-large factor).
fn ratio_str(chi2: f64, baseline: f64) -> String {
    if baseline < 1e-9 {
        "n/a".to_string()
    } else {
        format!("{:.1}×", chi2 / baseline)
    }
}

/// χ² (sum of Mahalanobis edge residuals `rᵀΩr`, or `weight·‖r‖²` for isotropic
/// edges) over the first `count` edges at the current estimate.
fn chi2_over(graph: &PoseGraph, count: usize) -> f64 {
    let mut total = 0.0;
    for edge in graph.edges.iter().take(count) {
        let (Some(from), Some(to)) = (graph.poses.get(&edge.from), graph.poses.get(&edge.to))
        else {
            continue;
        };
        let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
        let r = edge.measurement.inverse().compose(&predicted).log();
        total += match &edge.information {
            Some(omega) => (r.transpose() * omega * r)[(0, 0)],
            None => edge.weight * r.norm_squared(),
        };
    }
    total
}

/// Append `n` random wrong loop closures between existing vertices and return
/// their edge indices (always the last `n` edges). Each gets a large random
/// relative translation and rotation, so its residual dwarfs a genuine edge's.
fn inject_outliers(graph: &mut PoseGraph, n: usize, seed: u64) -> Vec<usize> {
    let ids: Vec<u64> = graph.poses.keys().copied().collect();
    // Match the injected information scale to a typical real edge so the
    // outlier residual is comparable in magnitude to the inliers' units.
    let info = typical_information(graph);
    let mut rng = Lcg(seed.wrapping_mul(2) | 1);
    let mut injected = Vec::with_capacity(n);
    while injected.len() < n && ids.len() > 1 {
        let i = ids[(rng.next_u64() as usize) % ids.len()];
        let mut j = ids[(rng.next_u64() as usize) % ids.len()];
        if i == j {
            j = ids[(j as usize + 1) % ids.len()];
        }
        let translation = Vector3::new(
            rng.range(-10.0, 10.0),
            rng.range(-10.0, 10.0),
            rng.range(-10.0, 10.0),
        );
        let axis = Vector3::new(
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        );
        let rotation = nalgebra::Unit::try_new(axis, 1e-9)
            .map(|a| UnitQuaternion::from_axis_angle(&a, rng.range(0.5, std::f64::consts::PI)))
            .unwrap_or_else(UnitQuaternion::identity);
        graph.add_edge_with_information(
            i,
            j,
            SE3::new(rotation, translation),
            PoseGraphEdgeKind::LoopClosure,
            info,
        );
        injected.push(graph.edges.len() - 1);
    }
    injected
}

/// Mean information matrix of the edges that carry one, else unit information —
/// a representative scale for the injected outliers.
fn typical_information(graph: &PoseGraph) -> Matrix6<f64> {
    let mut sum = Matrix6::zeros();
    let mut count = 0;
    for edge in &graph.edges {
        if let Some(omega) = &edge.information {
            sum += omega;
            count += 1;
        }
    }
    if count == 0 {
        Matrix6::identity()
    } else {
        sum / count as f64
    }
}

fn base_config() -> PoseGraphSe3Config {
    PoseGraphSe3Config {
        max_iterations: 50,
        initial_lambda: Some(1e-3),
        linear_solver: LinearSolver::Sparse,
        // Chordal seeding is robust to a handful of outliers among many edges
        // (rotation least squares averages them out), and the hard 3D graphs
        // need it to converge — keep it on for a realistic comparison.
        chordal_init: true,
        ..PoseGraphSe3Config::default()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut path: Option<String> = None;
    let mut inject = 10usize;
    let mut c = 3.0;
    let mut seed = 1u64;
    // `None` = use the fixed `c`; `Some(k)` = MAD auto-estimate the inlier scale
    // with multiplier `k` (floored at `c`). `--auto-c` uses the recommended k.
    let mut auto_scale: Option<f64> = None;
    // Re-estimate the auto scale every μ level instead of once. Only meaningful
    // with an auto scale, so `--readapt` implies `--auto-c` if none was set.
    //
    // WARNING — this is a BA feature, not a PGO one. Re-adapting helps bundle
    // adjustment (where the one-shot MAD scale *inflates* on the contaminated
    // VO-chain init, so tightening it per μ level recovers recall — see the
    // `--ba-gnc-readapt` results in online_slam_stereo_vo_kitti_demo). On PGO it
    // is HARMFUL: TLS's hard 0/1 rejection drives the inlier residuals down each
    // μ level, so the MAD re-estimate keeps shrinking `c`, the inlier band
    // over-tightens, and real edges get over-rejected (FP explodes, the graph
    // shatters). Measured on sphere2500 +300 (6 % contamination, seed 1) the
    // GNC-TLS inlier-χ² goes 121.6× (one-shot, c≈30) → 4338.8× (readapt, floor
    // 3, c collapses to the floor) → 925.2× (readapt, floor 12) — every readapt
    // variant loses badly to one-shot. The loose one-shot `c` is *correct* for
    // PGO; being loose protects real-edge recall. Kept only for A/B diagnosis.
    let mut readapt = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--inject" => inject = args.next().and_then(|s| s.parse().ok()).unwrap_or(inject),
            "--c" => c = args.next().and_then(|s| s.parse().ok()).unwrap_or(c),
            "--seed" => seed = args.next().and_then(|s| s.parse().ok()).unwrap_or(seed),
            "--auto-c" => auto_scale = Some(visloc_slam::gnc::AUTO_SCALE_K),
            "--auto-c-k" => {
                let k = args.next().and_then(|s| s.parse().ok());
                auto_scale = Some(k.unwrap_or(visloc_slam::gnc::AUTO_SCALE_K));
            }
            "--readapt" => readapt = true,
            other => path = Some(other.to_string()),
        }
    }
    if readapt && auto_scale.is_none() {
        auto_scale = Some(visloc_slam::gnc::AUTO_SCALE_K);
    }

    let (clean, source) = match &path {
        Some(p) => (read_g2o(p)?, p.clone()),
        None => (
            build_synthetic_loop_graph(),
            "<synthetic loop graph>".to_string(),
        ),
    };
    let original_edges = clean.edges.len();

    println!("pgo_g2o_robust_benchmark");
    println!("  source         : {source}");
    println!("  vertices       : {}", clean.poses.len());
    println!("  original edges : {original_edges}");
    println!("  injected       : {inject} wrong loop closures (seed {seed})");
    match auto_scale {
        Some(k) => println!(
            "  inlier scale c : auto (MAD, k={k}, floor {c}){}",
            if readapt {
                ", re-adapted per μ level"
            } else {
                ""
            }
        ),
        None => println!("  inlier scale c : {c} (fixed)"),
    }
    println!();

    // Outlier-free baseline: the target inlier-edge χ² a robust method should
    // match on the corrupted graph.
    let mut baseline = clean.clone();
    let baseline_result = baseline.optimize_se3_iterative(&base_config())?;
    let baseline_chi2 = chi2_over(&baseline, original_edges);
    println!(
        "  baseline (no outliers, L2)   inlier χ² = {:.6e}  (final {:.6e})",
        baseline_chi2, baseline_result.final_cost
    );
    println!();

    // Corrupt the graph once, then run every back-end on a clone of it.
    let mut corrupted = clean.clone();
    let injected = inject_outliers(&mut corrupted, inject, seed);
    let injected_set: std::collections::HashSet<usize> = injected.iter().copied().collect();

    println!(
        "  {:<10} {:>14} {:>14} {:>22} {:>8}",
        "method", "inlier χ²", "ratio×base", "outliers (recall / FP)", "c used"
    );

    // 1. Plain L2.
    run_plain(
        "L2",
        &corrupted,
        original_edges,
        baseline_chi2,
        RobustKernel::None,
    )?;
    // 2. Huber IRLS.
    run_plain(
        "Huber",
        &corrupted,
        original_edges,
        baseline_chi2,
        RobustKernel::Huber { delta: 1.0 },
    )?;
    // 3 & 4. GNC.
    run_gnc(
        "GNC-GM",
        &corrupted,
        original_edges,
        baseline_chi2,
        &injected_set,
        GncKernel::GemanMcClure,
        c,
        auto_scale,
        readapt,
    )?;
    run_gnc(
        "GNC-TLS",
        &corrupted,
        original_edges,
        baseline_chi2,
        &injected_set,
        GncKernel::TruncatedLeastSquares,
        c,
        auto_scale,
        readapt,
    )?;

    Ok(())
}

fn run_plain(
    name: &str,
    corrupted: &PoseGraph,
    original_edges: usize,
    baseline_chi2: f64,
    kernel: RobustKernel,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = corrupted.clone();
    let config = PoseGraphSe3Config {
        robust_kernel: kernel,
        ..base_config()
    };
    graph.optimize_se3_iterative(&config)?;
    let chi2 = chi2_over(&graph, original_edges);
    println!(
        "  {:<10} {:>14.6e} {:>14} {:>22} {:>8}",
        name,
        chi2,
        ratio_str(chi2, baseline_chi2),
        "-",
        "-"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_gnc(
    name: &str,
    corrupted: &PoseGraph,
    original_edges: usize,
    baseline_chi2: f64,
    injected: &std::collections::HashSet<usize>,
    kernel: GncKernel,
    c: f64,
    auto_scale: Option<f64>,
    auto_scale_readapt: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = corrupted.clone();
    let gnc = GncConfig {
        kernel,
        c,
        auto_scale,
        auto_scale_readapt,
        ..GncConfig::default()
    };
    let result = graph.optimize_se3_gnc(&base_config(), &gnc)?;
    let chi2 = chi2_over(&graph, original_edges);

    // Classify against ground truth: injected edges should be rejected
    // (weight < 0.5), original edges kept.
    let threshold = 0.5;
    let rejected_outliers = injected
        .iter()
        .filter(|&&i| result.edge_weights[i] < threshold)
        .count();
    let false_positives = (0..original_edges)
        .filter(|&i| result.edge_weights[i] < threshold)
        .count();
    let recall = format!(
        "{}/{} ({} FP)",
        rejected_outliers,
        injected.len(),
        false_positives
    );
    println!(
        "  {:<10} {:>14.6e} {:>14} {:>22} {:>8.3}",
        name,
        chi2,
        ratio_str(chi2, baseline_chi2),
        recall,
        result.inlier_scale,
    );
    Ok(())
}

/// Build a deterministic 3D loop pose graph (same fixture as
/// `pgo_g2o_benchmark`): a circular trajectory climbing in z, true relative
/// sequential edges plus a closing loop edge, drifted initial estimates. No
/// RNG, so the demo output is reproducible.
fn build_synthetic_loop_graph() -> PoseGraph {
    const N: u64 = 120;

    let pose_from_camera_to_world = |c2w: SE3| Pose {
        world_to_camera: c2w.inverse(),
    };
    let truth_camera_to_world = |i: u64| -> SE3 {
        let t = i as f64 / N as f64;
        let angle = t * std::f64::consts::TAU;
        let translation = Vector3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 3.0 * t);
        SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, angle),
            translation,
        )
    };
    let truth = |i: u64| pose_from_camera_to_world(truth_camera_to_world(i));

    let mut graph = PoseGraph::new();
    let drift = SE3::new(
        UnitQuaternion::from_euler_angles(0.0, 0.0, 0.01),
        Vector3::zeros(),
    );
    let mut estimate = truth_camera_to_world(0);
    graph.add_pose(0, pose_from_camera_to_world(estimate.clone()));
    for i in 1..N {
        let true_rel = truth_camera_to_world(i - 1)
            .inverse()
            .compose(&truth_camera_to_world(i));
        estimate = estimate.compose(&true_rel).compose(&drift);
        graph.add_pose(i, pose_from_camera_to_world(estimate.clone()));
    }
    graph.anchor(0);
    for i in 1..N {
        graph.add_sequential_edge(i - 1, i, relative_world_to_camera(&truth(i - 1), &truth(i)));
    }
    graph.add_edge_with_information(
        N - 1,
        0,
        relative_world_to_camera(&truth(N - 1), &truth(0)),
        PoseGraphEdgeKind::LoopClosure,
        Matrix6::identity() * 10.0,
    );
    graph
}
