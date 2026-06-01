//! Integration tests for multi-session pose-graph merging
//! ([`PoseGraph::merge_session`]) — welding two trajectories at a cross-session
//! bridge constraint (ORB-SLAM3-Atlas-style). Validates the alignment geometry
//! exactly (no solve) and the joint equivalence after optimization.

use nalgebra::{UnitQuaternion, Vector3, Vector6};
use visloc_core::geometry::{Pose, SE3};
use visloc_slam::pcm::PcmConfig;
use visloc_slam::{
    relative_world_to_camera, LoopClosureConstraint, PoseGraph, PoseGraphEdgeKind,
    PoseGraphSe3Config,
};

fn pose_at(x: f64) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -Vector3::new(x, 0.0, 0.0))
}

fn config() -> PoseGraphSe3Config {
    PoseGraphSe3Config {
        initial_lambda: Some(1.0e-6),
        max_iterations: 50,
        chordal_init: false,
        ..PoseGraphSe3Config::default()
    }
}

/// Build a chain graph over `poses` with truth-relative sequential edges,
/// anchored at the first node, ids `0..poses.len()`.
fn chain_graph(poses: &[Pose]) -> PoseGraph {
    let mut g = PoseGraph::new();
    for (i, p) in poses.iter().enumerate() {
        g.add_pose(i as u64, p.clone());
    }
    g.anchor(0);
    for i in 0..poses.len() - 1 {
        g.add_sequential_edge(
            i as u64,
            (i + 1) as u64,
            relative_world_to_camera(&poses[i], &poses[i + 1]),
        );
    }
    g
}

fn bridge(from: u64, to: u64, relative: SE3) -> LoopClosureConstraint {
    LoopClosureConstraint {
        from_keyframe_id: from,
        to_keyframe_id: to,
        relative_pose: relative,
        inlier_count: 100,
        inlier_ratio: 1.0,
        mean_sampson_error: 0.0,
        score: 100.0,
    }
}

/// The alignment geometry, checked exactly (no solve): session B lives in a
/// different world frame `G`, and merging it into session A at a truth bridge
/// must place every B node back onto the shared truth trajectory.
#[test]
fn merge_session_aligns_a_second_session_into_the_shared_frame() {
    let truth: Vec<Pose> = (0..8).map(|i| pose_at(i as f64)).collect();

    // Session A: global keyframes 0..3, at truth.
    let mut a = chain_graph(&truth[0..4]);

    // Session B: global keyframes 4..7, but expressed in an ARBITRARY local
    // world frame — each pose right-multiplied by a fixed world offset `g`.
    let g = SE3::exp(&Vector6::new(0.3, -0.2, 0.5, 0.4, -0.1, 0.25));
    let b_poses: Vec<Pose> = truth[4..8]
        .iter()
        .map(|p| Pose {
            world_to_camera: p.world_to_camera.compose(&g),
        })
        .collect();
    let b = chain_graph(&b_poses);

    // Bridge: the (frame-invariant) relative camera pose between global kf3 and
    // kf4 — what a cross-session place-recognition match would report.
    let z = relative_world_to_camera(&truth[3], &truth[4]);
    a.merge_session(&b, 4, &bridge(3, 0, z)).unwrap();

    // Merged graph spans ids 0..7; B's nodes (4..7) must land on the truth.
    assert_eq!(a.poses.len(), 8);
    for i in 0..8u64 {
        let merged_center = a.poses[&i].camera_center_world();
        let truth_center = truth[i as usize].camera_center_world();
        let d = (merged_center - truth_center).norm();
        assert!(
            d < 1e-9,
            "merged node {i} must align onto the shared truth trajectory: {d}"
        );
    }
    // The weld edge and B's spliced sequential edges are present.
    assert_eq!(
        a.edges
            .iter()
            .filter(|e| e.kind == PoseGraphEdgeKind::LoopClosure)
            .count(),
        1,
        "exactly one bridge (loop-closure) edge"
    );
}

/// Merging two drifted half-sessions then jointly optimizing must reach the same
/// optimum as a single full-graph solve over the same constraints — merge builds
/// the equivalent joint problem.
#[test]
fn merged_then_optimized_matches_a_single_full_graph_solve() {
    let truth: Vec<Pose> = (0..8).map(|i| pose_at(i as f64)).collect();
    let edges: Vec<SE3> = truth
        .windows(2)
        .map(|w| relative_world_to_camera(&w[0], &w[1]))
        .collect();

    // Drifted initial node estimates: integrate slightly yaw-perturbed edges so
    // the optimizer has real work to do; the edges themselves stay truth-relative
    // (so the optimum is the truth, gauge-fixed at node 0).
    let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.02);
    let mut drifted = vec![truth[0].clone()];
    for e in &edges {
        let last = drifted.last().unwrap();
        let noisy = SE3::new(yaw * e.rotation, e.translation);
        drifted.push(Pose {
            world_to_camera: noisy.compose(&last.world_to_camera),
        });
    }

    // Full single-session graph 0..7.
    let mut full = chain_graph(&drifted);
    full.optimize_se3_iterative(&config()).unwrap();

    // Two half-sessions from the SAME drifted inits, merged at the bridge that
    // reproduces the full graph's seam edge (the relative pose between the two
    // boundary nodes' current estimates) — so the merged problem is identical to
    // the full one.
    let mut a = chain_graph(&drifted[0..4]);
    let b = chain_graph(&drifted[4..8]);
    let z = relative_world_to_camera(&drifted[3], &drifted[4]);
    a.merge_session(&b, 4, &bridge(3, 0, z)).unwrap();
    a.optimize_se3_iterative(&config()).unwrap();

    for i in 0..8u64 {
        let d = (a.poses[&i].camera_center_world() - full.poses[&i].camera_center_world()).norm();
        assert!(
            d < 1e-4,
            "merged-then-optimized node {i} must match the full-graph optimum: {d}"
        );
    }
}

/// Cross-session PCM bridge screening: two sessions revisit the same path (the
/// second in a different world frame). Several genuine same-place bridges are
/// mutually consistent; a wrong bridge (claiming two different places coincide)
/// is not — `consistent_session_bridges` must keep the genuine clique and drop
/// the wrong one, even though the sessions live in different frames.
#[test]
fn consistent_session_bridges_drops_a_wrong_cross_session_match() {
    let truth: Vec<Pose> = (0..4).map(|i| pose_at(i as f64)).collect();

    // Session A traverses the path at truth; session B re-traverses the SAME
    // path in an arbitrary world frame `g`.
    let a = chain_graph(&truth);
    let g = SE3::exp(&Vector6::new(-0.2, 0.4, 0.1, 0.3, 0.15, -0.2));
    let b_poses: Vec<Pose> = truth
        .iter()
        .map(|p| Pose {
            world_to_camera: p.world_to_camera.compose(&g),
        })
        .collect();
    let b = chain_graph(&b_poses);

    // Genuine bridges: A node i ↔ B node i are the SAME place → the relative
    // camera pose is identity (frame-invariant). Three of them form a consensus.
    let identity = relative_world_to_camera(&truth[0], &truth[0]);
    let mut candidates: Vec<LoopClosureConstraint> =
        (0..3).map(|i| bridge(i, i, identity.clone())).collect();
    // A wrong bridge: A node 0 ↔ B node 3 claimed co-located (identity) — but
    // they are different places, so it is inconsistent with the genuine clique.
    candidates.push(bridge(0, 3, identity.clone()));

    let cfg = PcmConfig {
        threshold: 0.5,
        require_individual: false, // no single-session relative across sessions
        noise: None,
    };
    let kept = a.consistent_session_bridges(&b, 4, &candidates, &cfg);

    assert_eq!(
        kept,
        vec![0, 1, 2],
        "the genuine consensus is kept, wrong dropped"
    );

    // Merging with a screened genuine bridge welds B onto the shared frame.
    let mut merged = a.clone();
    merged.merge_session(&b, 4, &candidates[kept[0]]).unwrap();
    assert_eq!(merged.poses.len(), 8);
}

#[test]
fn merge_session_rejects_missing_bridge_endpoints() {
    use visloc_slam::PoseGraphError;
    let truth: Vec<Pose> = (0..4).map(|i| pose_at(i as f64)).collect();
    let mut a = chain_graph(&truth[0..2]);
    let b = chain_graph(&truth[2..4]);
    let z = relative_world_to_camera(&truth[1], &truth[2]);
    // Bridge `from` 9 is absent from A.
    assert_eq!(
        a.merge_session(&b, 2, &bridge(9, 0, z.clone())),
        Err(PoseGraphError::MissingNode(9))
    );
    // Bridge `to` 9 is absent from B.
    assert_eq!(
        a.merge_session(&b, 2, &bridge(1, 9, z)),
        Err(PoseGraphError::MissingNode(9))
    );
}
