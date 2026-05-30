//! Integration tests for [`PoseGraph::pose_marginal_covariances`] — covariance
//! recovery from the information matrix at the solution.
//!
//! These assert the *physical* correctness of the recovered marginals on real
//! pose-graph structures: uncertainty grows with distance from the gauge-fixed
//! anchor along an open chain, and a loop closure tightens it. The numerical
//! agreement of the underlying Takahashi recursion with the dense inverse is
//! covered by `visloc_slam::covariance`'s unit tests.

use nalgebra::{UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_slam::{relative_world_to_camera, LoopClosureConstraint, PoseGraph, PoseGraphSe3Config};

fn pose_at(x: f64) -> Pose {
    // Pure translation along +x, no rotation.
    Pose::from_world_to_camera(UnitQuaternion::identity(), -Vector3::new(x, 0.0, 0.0))
}

/// An `n`-pose chain anchored at id 0, each pose 1 m further along +x, with
/// truth-relative (zero-residual) sequential edges of unit weight.
fn chain(n: usize) -> (PoseGraph, Vec<Pose>) {
    let poses: Vec<Pose> = (0..n).map(|i| pose_at(i as f64)).collect();
    let mut graph = PoseGraph::new();
    for (i, p) in poses.iter().enumerate() {
        graph.add_pose(i as u64, p.clone());
    }
    graph.anchor(0);
    for i in 0..n - 1 {
        graph.add_sequential_edge(
            i as u64,
            (i + 1) as u64,
            relative_world_to_camera(&poses[i], &poses[i + 1]),
        );
    }
    (graph, poses)
}

fn config() -> PoseGraphSe3Config {
    PoseGraphSe3Config {
        initial_lambda: Some(1.0e-6),
        max_iterations: 20,
        ..PoseGraphSe3Config::default()
    }
}

#[test]
fn marginal_covariances_are_symmetric_positive_definite() {
    let (mut graph, _) = chain(6);
    graph.optimize_se3_iterative(&config()).unwrap();
    let cov = graph.pose_marginal_covariances().unwrap();

    // One covariance per non-anchor pose (anchor 0 omitted).
    assert_eq!(cov.len(), 5);
    assert!(!cov.contains_key(&0));
    for (id, sigma) in &cov {
        let asym = (sigma - sigma.transpose()).abs().max();
        assert!(asym < 1e-9, "pose {id} covariance not symmetric: {asym}");
        assert!(
            sigma.cholesky().is_some(),
            "pose {id} covariance not positive-definite"
        );
    }
}

#[test]
fn uncertainty_grows_with_distance_from_the_anchor() {
    let (mut graph, _) = chain(7);
    graph.optimize_se3_iterative(&config()).unwrap();
    let cov = graph.pose_marginal_covariances().unwrap();

    // Along an open chain the marginal covariance accumulates monotonically:
    // a pose further from the anchor is strictly more uncertain.
    let traces: Vec<f64> = (1..7).map(|id| cov[&(id as u64)].trace()).collect();
    for w in traces.windows(2) {
        assert!(
            w[1] > w[0],
            "covariance trace should grow away from the anchor: {traces:?}"
        );
    }
}

#[test]
fn loop_closure_tightens_the_far_pose_covariance() {
    // Open chain: the last pose is the most uncertain.
    let (mut open, poses) = chain(8);
    open.optimize_se3_iterative(&config()).unwrap();
    let open_cov = open.pose_marginal_covariances().unwrap();
    let last = (poses.len() - 1) as u64;
    let open_trace = open_cov[&last].trace();

    // Same chain plus a truth-relative loop closure from the anchor to the last
    // pose: it injects information, so the far pose must become more certain.
    let (mut looped, _) = chain(8);
    looped.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 0,
        to_keyframe_id: last,
        relative_pose: relative_world_to_camera(&poses[0], &poses[last as usize]),
        inlier_count: 100,
        inlier_ratio: 1.0,
        mean_sampson_error: 0.0,
        score: 100.0,
    });
    looped.optimize_se3_iterative(&config()).unwrap();
    let looped_cov = looped.pose_marginal_covariances().unwrap();
    let looped_trace = looped_cov[&last].trace();

    assert!(
        looped_trace < open_trace,
        "a loop closure must tighten the far pose covariance: open={open_trace} looped={looped_trace}"
    );
}

#[test]
fn errors_when_no_anchor_or_no_edges() {
    use visloc_slam::PoseGraphError;
    let empty = PoseGraph::new();
    assert_eq!(
        empty.pose_marginal_covariances(),
        Err(PoseGraphError::NoAnchor)
    );

    let mut no_edges = PoseGraph::new();
    no_edges.add_pose(0, pose_at(0.0));
    no_edges.anchor(0);
    assert_eq!(
        no_edges.pose_marginal_covariances(),
        Err(PoseGraphError::NoEdges)
    );
}
