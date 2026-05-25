//! Sim(3) pose-graph optimization for monocular scale-drift correction.
//!
//! Monocular SLAM recovers a trajectory only up to scale, and that scale drifts
//! slowly along the path, so a loop that should close is left offset by a scale
//! factor. A rigid SE(3) pose graph cannot fix this — it has no scale degree of
//! freedom. This demo builds a deterministic circular loop at unit (metric)
//! scale, then a *drifted* initial estimate whose per-keyframe scale shrinks
//! cumulatively (the classic monocular failure), and runs
//! [`Sim3PoseGraph::optimize`] to redistribute the scale error and pull the
//! trajectory back to ground truth.
//!
//! No external data and no RNG, so it runs end-to-end in CI and the output is
//! reproducible. Run with:
//!
//! ```text
//! cargo run --release --example sim3_scale_drift_pgo_demo
//! ```

use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::{Sim3, Sim3PoseGraph, Sim3PoseGraphConfig};

const NODE_COUNT: u64 = 24;

fn main() {
    let truth = ground_truth_loop(NODE_COUNT);

    // Build the graph: sequential odometry edges plus one loop closure, every
    // measurement the *true* relative similarity (so ground truth is the unique
    // zero-cost solution given the fixed anchor).
    let mut graph = Sim3PoseGraph::new();
    for (id, pose) in &truth {
        graph.add_pose(*id, pose.clone());
    }
    for window in truth.windows(2) {
        let measurement = window[1].1.compose(&window[0].1.inverse());
        graph.add_edge(window[0].0, window[1].0, measurement, 1.0);
    }
    let last = truth.len() - 1;
    let loop_measurement = truth[0].1.compose(&truth[last].1.inverse());
    graph.add_edge(truth[last].0, truth[0].0, loop_measurement, 10.0);
    graph.anchor(0);

    // Drifted initial estimate: dead-reckon from the fixed anchor, but shrink
    // every odometry step by 3 % (a per-step scale bias). The error compounds,
    // so by the far side of the loop the estimate is badly under-scaled and
    // geometrically compressed — exactly how monocular scale drift looks. The
    // measurements stay at their true values, so ground truth remains the unique
    // zero-cost solution; only the starting estimates are corrupted.
    let per_step_scale_bias = Sim3::new(UnitQuaternion::identity(), Vector3::zeros(), 0.97);
    let mut estimate = truth[0].1.clone();
    for window in truth.windows(2) {
        let true_relative = window[1].1.compose(&window[0].1.inverse());
        estimate = per_step_scale_bias
            .compose(&true_relative)
            .compose(&estimate);
        graph.add_pose(window[1].0, estimate.clone());
    }

    println!("sim3_scale_drift_pgo_demo");
    println!("  nodes         : {}", graph.poses.len());
    println!(
        "  edges         : {} (sequential + 1 loop closure)",
        graph.edges.len()
    );
    report("before", &graph, &truth);

    let config = Sim3PoseGraphConfig::default();
    let result = graph
        .optimize(&config)
        .expect("optimization should succeed");

    report("after ", &graph, &truth);
    let accepted = result
        .iterations
        .iter()
        .filter(|it| it.step_accepted)
        .count();
    println!(
        "  cost          : {:.6e} -> {:.6e}  ({:.4} % reduction)",
        result.initial_cost,
        result.final_cost,
        100.0 * (1.0 - result.final_cost / result.initial_cost.max(f64::MIN_POSITIVE))
    );
    println!(
        "  iterations    : {} accepted / {} total, converged = {}",
        accepted,
        result.iterations.len(),
        result.converged
    );
}

/// A ground-truth circular trajectory climbing in z, at unit (metric) scale.
fn ground_truth_loop(node_count: u64) -> Vec<(u64, Sim3)> {
    (0..node_count)
        .map(|i| {
            let angle = i as f64 / node_count as f64 * std::f64::consts::TAU;
            let pose = Sim3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, angle),
                Vector3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 0.2 * i as f64),
                1.0,
            );
            (i, pose)
        })
        .collect()
}

/// Print the worst scale error and the mean / max position error against the
/// ground-truth nodes.
fn report(tag: &str, graph: &Sim3PoseGraph, truth: &[(u64, Sim3)]) {
    let mut worst_scale_error = 0.0_f64;
    let mut sum_position_error = 0.0;
    let mut max_position_error = 0.0_f64;
    for (id, pose) in truth {
        let estimate = &graph.poses[id];
        worst_scale_error = worst_scale_error.max((estimate.scale - pose.scale).abs());
        let position_error = (estimate.translation - pose.translation).norm();
        sum_position_error += position_error;
        max_position_error = max_position_error.max(position_error);
    }
    println!(
        "  {tag}        : worst |scale-1| = {:.4}, position error mean = {:.4} m, max = {:.4} m",
        worst_scale_error,
        sum_position_error / truth.len() as f64,
        max_position_error
    );
}
