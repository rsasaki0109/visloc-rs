//! SE(3) pose-graph optimization benchmark on the `.g2o` exchange format.
//!
//! Runs visloc-rs's pure-Rust, deterministic [`PoseGraph::optimize_se3_iterative`]
//! on a `.g2o` graph and reports the χ² (sum of Mahalanobis edge residuals)
//! before and after, the iteration count, and wall-clock time — the standard
//! way pose-graph back-ends (g2o, GTSAM, Ceres) are compared on canonical
//! datasets such as `sphere2500` and `parking-garage-3500`.
//!
//! Usage:
//!
//! ```text
//! # Real benchmark dataset (download a .g2o first, e.g. sphere2500):
//! cargo run --release --example pgo_g2o_benchmark -- path/to/sphere2500.g2o
//!
//! # Seed the SE(3) solve with a chordal rotation initialization (Carlone et
//! # al.) — essential for hard 3D graphs (e.g. `rim`) where odometry-initialized
//! # Levenberg-Marquardt stalls in a poor basin:
//! cargo run --release --example pgo_g2o_benchmark -- --chordal-init path/to/rim.g2o
//!
//! # No argument: a built-in deterministic synthetic loop graph, so the demo
//! # runs end-to-end in CI without any external data.
//! cargo run --example pgo_g2o_benchmark
//! ```

use std::time::Instant;

use nalgebra::{Matrix6, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};
use visloc_slam::{
    read_g2o, relative_world_to_camera, DampingMode, LinearSolver, PoseGraph, PoseGraphEdgeKind,
    PoseGraphSe3Config, RobustKernel,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut chordal_init = false;
    let mut diag_damping = false;
    let mut path: Option<String> = None;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--chordal-init" => chordal_init = true,
            "--diag-damping" => diag_damping = true,
            _ => path = Some(arg),
        }
    }
    let (mut graph, source) = match path {
        Some(path) => (read_g2o(&path)?, path),
        None => (
            build_synthetic_loop_graph(),
            "<synthetic loop graph>".to_string(),
        ),
    };

    println!("pgo_g2o_benchmark");
    println!("  source        : {source}");
    println!("  vertices      : {}", graph.poses.len());
    println!("  edges         : {}", graph.edges.len());
    println!("  chordal init  : {chordal_init}");
    println!(
        "  damping       : {}",
        if diag_damping {
            "diagonal (H + λ·diag(H))"
        } else {
            "identity (H + λI)"
        }
    );

    // Optional chordal rotation initialization: solve the relaxed rotation
    // sub-problem and re-derive translations before the full SE(3) solve. On
    // strongly non-convex 3D graphs this reseeds LM near the global optimum.
    if chordal_init {
        let started = Instant::now();
        let rot = graph.initialize_rotations_chordal(LinearSolver::Sparse)?;
        graph.optimize_translations_once_with(LinearSolver::Sparse)?;
        let elapsed = started.elapsed();
        println!(
            "  chordal rot   : {:.6e} -> {:.6e} (max Δrot {:.1}°, {:.1} ms)",
            rot.cost_before,
            rot.cost_after,
            rot.max_rotation_update_deg,
            elapsed.as_secs_f64() * 1e3,
        );
    }

    // Levenberg-Marquardt with the sparse Cholesky backend: robust on noisy
    // real graphs and linear in the (sparse) edge count rather than cubic in
    // the vertex count. The library now seeds with a chordal init by default,
    // but this benchmark drives that step manually above (gated on the flag)
    // so its before/after χ² stays a clean, independently-measured comparison;
    // disabling the in-solver seeding here keeps it from running twice.
    let config = PoseGraphSe3Config {
        max_iterations: 50,
        initial_lambda: Some(1e-3),
        linear_solver: LinearSolver::Sparse,
        robust_kernel: RobustKernel::None,
        chordal_init: false,
        damping: if diag_damping {
            DampingMode::Diagonal
        } else {
            DampingMode::Identity
        },
        ..PoseGraphSe3Config::default()
    };

    let started = Instant::now();
    let result = graph.optimize_se3_iterative(&config)?;
    let elapsed = started.elapsed();

    let accepted = result
        .iterations
        .iter()
        .filter(|it| it.step_accepted)
        .count();
    let reduction = if result.initial_cost > 0.0 {
        100.0 * (1.0 - result.final_cost / result.initial_cost)
    } else {
        0.0
    };

    println!("  initial chi^2 : {:.6e}", result.initial_cost);
    println!("  final chi^2   : {:.6e}", result.final_cost);
    println!("  reduction     : {reduction:.3} %");
    println!(
        "  iterations    : {} accepted / {} total",
        accepted,
        result.iterations.len()
    );
    println!("  converged     : {}", result.converged);
    println!("  elapsed       : {:.1} ms", elapsed.as_secs_f64() * 1e3);

    Ok(())
}

/// Build a deterministic 3D loop pose graph: a circular trajectory climbing in
/// z, with sequential edges carrying the *true* relative poses and a closing
/// loop edge, while the initial vertex estimates are a dead-reckoned chain with
/// a constant per-step yaw bias (classic accumulated odometry drift). No RNG,
/// so the demo output is reproducible.
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

    // Drifted initial estimate: integrate the true relative motion plus a small
    // constant yaw bias each step.
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

    // Sequential edges carry the true relative pose (the measurement); drift
    // lives only in the initial vertex estimates above.
    for i in 1..N {
        graph.add_sequential_edge(i - 1, i, relative_world_to_camera(&truth(i - 1), &truth(i)));
    }
    // Loop-closure edge from the last pose back to the start, weighted up.
    graph.add_edge_with_information(
        N - 1,
        0,
        relative_world_to_camera(&truth(N - 1), &truth(0)),
        PoseGraphEdgeKind::LoopClosure,
        Matrix6::identity() * 10.0,
    );

    graph
}
