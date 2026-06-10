//! Admission gates applied to verified loop closures before they enter the
//! pose graph: pairwise-consistency (PCM) and covariance gating.

use super::*;

/// The PCM [`pcm::LoopMeasurement`] view of a verified loop-closure constraint.
pub(crate) fn loop_measurement_of(c: &LoopClosureConstraint) -> pcm::LoopMeasurement {
    pcm::LoopMeasurement {
        from: c.from_keyframe_id,
        to: c.to_keyframe_id,
        relative: c.relative_pose.clone(),
    }
}

/// PCM admission test for a single newly-verified loop closure (the online,
/// incremental variant of [`pcm::maximum_consistent_set`]). Admits `new` when it
/// is individually consistent with the odometry (if
/// [`pcm::PcmConfig::require_individual`]) and pairwise-consistent with a strict
/// majority of the already-`admitted` closures — so a perceptual-aliasing false
/// positive, inconsistent with the established consensus, is rejected before it
/// enters the graph. The first closure (empty `admitted`) is admitted on the
/// individual check alone.
pub(crate) fn pcm_admits_loop(
    new: &pcm::LoopMeasurement,
    admitted: &[pcm::LoopMeasurement],
    odometry: &BTreeMap<u64, SE3>,
    cfg: &pcm::PcmConfig,
) -> bool {
    if cfg.require_individual {
        match pcm::individual_residual(new, odometry) {
            Some(r) if r <= cfg.threshold => {}
            _ => return false,
        }
    }
    if admitted.is_empty() {
        return true;
    }
    let consistent = admitted
        .iter()
        .filter(|a| {
            pcm::pairwise_residual(new, a, odometry)
                .map(|r| r <= cfg.threshold)
                .unwrap_or(false)
        })
        .count();
    // Strict majority of the established set agrees with the new closure.
    consistent * 2 > admitted.len()
}

/// Batch PCM reconcile: recompute the order-independent maximum mutually-
/// consistent set ([`pcm::maximum_consistent_set`]) over the union of the
/// currently-admitted (`verified`) and `deferred` loop closures, then reconcile
/// `graph`'s loop edges to it — *promote* a deferred closure that joins the
/// consensus (add its edge) and *evict* an admitted closure that leaves it
/// (drop its edge). This heals the cold-start failure of the incremental,
/// order-dependent screen, where a perceptual-aliasing closure admitted first
/// (against the empty set) then poisons every genuine closure checked against
/// it. `verified` / `deferred` are rewritten to the new partition; returns
/// `(promoted, evicted)`.
pub(crate) fn pcm_batch_reconcile(
    graph: &mut PoseGraph,
    verified: &mut Vec<LoopClosureConstraint>,
    deferred: &mut Vec<LoopClosureConstraint>,
    odometry: &BTreeMap<u64, SE3>,
    cfg: &pcm::PcmConfig,
) -> (usize, usize) {
    let admitted_n = verified.len();
    // Union, admitted first so index `< admitted_n` ⇔ currently in the graph.
    let mut union: Vec<LoopClosureConstraint> = Vec::with_capacity(admitted_n + deferred.len());
    union.append(verified);
    union.append(deferred);
    let measurements: Vec<pcm::LoopMeasurement> = union.iter().map(loop_measurement_of).collect();
    let keep: std::collections::HashSet<usize> =
        pcm::maximum_consistent_set(&measurements, odometry, cfg)
            .into_iter()
            .collect();

    let (mut promoted, mut evicted) = (0usize, 0usize);
    for (i, constraint) in union.into_iter().enumerate() {
        let was_admitted = i < admitted_n;
        if keep.contains(&i) {
            if !was_admitted {
                graph.add_loop_closure_constraint(&constraint);
                promoted += 1;
            }
            verified.push(constraint);
        } else {
            if was_admitted {
                let (f, t) = (constraint.from_keyframe_id, constraint.to_keyframe_id);
                graph.edges.retain(|e| {
                    !(e.kind == PoseGraphEdgeKind::LoopClosure && e.from == f && e.to == t)
                });
                evicted += 1;
            }
            deferred.push(constraint);
        }
    }
    (promoted, evicted)
}

/// Covariance gate for a single verified loop closure: admit it only when the
/// squared Mahalanobis distance of its innovation (measured relative pose vs the
/// estimate's prediction) under the relative-pose covariance is `<= threshold`.
/// Conservatively *admits* when the covariance cannot be recovered (singular
/// system) or an endpoint pose is missing, rather than dropping a closure on a
/// numerical failure.
pub(crate) fn covariance_gate_admits(
    graph: &PoseGraph,
    constraint: &LoopClosureConstraint,
    threshold: f64,
) -> bool {
    let (a, b) = (constraint.from_keyframe_id, constraint.to_keyframe_id);
    let cov = match graph.relative_pose_covariance(a, b) {
        Ok(cov) => cov,
        Err(_) => return true,
    };
    let (Some(pa), Some(pb)) = (graph.poses.get(&a), graph.poses.get(&b)) else {
        return true;
    };
    let z_hat = relative_world_to_camera(pa, pb);
    let innovation = constraint.relative_pose.compose(&z_hat.inverse()).log();
    let cov_d = DMatrix::from_fn(6, 6, |r, c| cov[(r, c)]);
    match covariance::mahalanobis_distance_sq(
        &DVector::from_column_slice(innovation.as_slice()),
        &cov_d,
    ) {
        Some(m) => m <= threshold,
        None => true,
    }
}

#[cfg(test)]
mod pcm_batch_reconcile_tests {
    use super::*;
    use std::collections::BTreeMap;

    fn pose_at(x: f64) -> Pose {
        Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            -nalgebra::Vector3::new(x, 0.0, 0.0),
        )
    }

    fn loop_constraint(from: u64, to: u64, relative: SE3) -> LoopClosureConstraint {
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

    fn is_loop_edge(graph: &PoseGraph, from: u64, to: u64) -> bool {
        graph
            .edges
            .iter()
            .any(|e| e.kind == PoseGraphEdgeKind::LoopClosure && e.from == from && e.to == to)
    }

    /// The poisoning scenario: a wrong closure was admitted first (against the
    /// empty set), then two genuine closures were deferred because they
    /// disagreed with it. The batch re-screen must recover the genuine
    /// consensus — evict the wrong one, promote the two genuine ones.
    #[test]
    fn batch_rescreen_evicts_the_poisoning_closure_and_promotes_the_consensus() {
        let poses: Vec<Pose> = (0..4).map(|i| pose_at(i as f64)).collect();
        let odometry: BTreeMap<u64, SE3> = poses
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u64, p.world_to_camera.clone()))
            .collect();

        // Two genuine, truth-relative closures (mutually + odometry consistent).
        let g1 = loop_constraint(0, 3, relative_world_to_camera(&poses[0], &poses[3]));
        let g2 = loop_constraint(1, 3, relative_world_to_camera(&poses[1], &poses[3]));
        // A perceptual-aliasing closure claiming frames 0 and 2 are co-located.
        let wrong = loop_constraint(0, 2, SE3::identity());

        // State after the incremental screen poisoned itself: `wrong` admitted
        // (and in the graph), the genuine pair deferred.
        let mut graph = PoseGraph::new();
        for (i, p) in poses.iter().enumerate() {
            graph.add_pose(i as u64, p.clone());
        }
        graph.add_loop_closure_constraint(&wrong);
        let mut verified = vec![wrong];
        let mut deferred = vec![g1, g2];

        let cfg = pcm::PcmConfig {
            threshold: 0.5,
            require_individual: true,
            noise: None,
        };
        let (promoted, evicted) =
            pcm_batch_reconcile(&mut graph, &mut verified, &mut deferred, &odometry, &cfg);

        assert_eq!(
            (promoted, evicted),
            (2, 1),
            "promote both genuine, evict wrong"
        );
        // The admitted set is now the genuine consensus; the wrong one deferred.
        assert_eq!(verified.len(), 2);
        assert!(verified
            .iter()
            .any(|c| (c.from_keyframe_id, c.to_keyframe_id) == (0, 3)));
        assert!(verified
            .iter()
            .any(|c| (c.from_keyframe_id, c.to_keyframe_id) == (1, 3)));
        assert_eq!(deferred.len(), 1);
        assert_eq!(
            (deferred[0].from_keyframe_id, deferred[0].to_keyframe_id),
            (0, 2)
        );
        // Graph loop edges match: genuine present, wrong gone.
        assert!(is_loop_edge(&graph, 0, 3));
        assert!(is_loop_edge(&graph, 1, 3));
        assert!(
            !is_loop_edge(&graph, 0, 2),
            "wrong loop edge must be evicted"
        );
    }

    /// When the admitted set is already the consensus, the re-screen is a no-op:
    /// nothing promoted, nothing evicted, the graph unchanged.
    #[test]
    fn batch_rescreen_is_a_noop_when_already_consistent() {
        let poses: Vec<Pose> = (0..4).map(|i| pose_at(i as f64)).collect();
        let odometry: BTreeMap<u64, SE3> = poses
            .iter()
            .enumerate()
            .map(|(i, p)| (i as u64, p.world_to_camera.clone()))
            .collect();
        let g1 = loop_constraint(0, 3, relative_world_to_camera(&poses[0], &poses[3]));
        let g2 = loop_constraint(1, 3, relative_world_to_camera(&poses[1], &poses[3]));

        let mut graph = PoseGraph::new();
        for (i, p) in poses.iter().enumerate() {
            graph.add_pose(i as u64, p.clone());
        }
        graph.add_loop_closure_constraint(&g1);
        graph.add_loop_closure_constraint(&g2);
        let mut verified = vec![g1, g2];
        let mut deferred: Vec<LoopClosureConstraint> = Vec::new();

        let cfg = pcm::PcmConfig {
            threshold: 0.5,
            require_individual: true,
            noise: None,
        };
        let (promoted, evicted) =
            pcm_batch_reconcile(&mut graph, &mut verified, &mut deferred, &odometry, &cfg);

        assert_eq!((promoted, evicted), (0, 0));
        assert_eq!(verified.len(), 2);
        assert!(deferred.is_empty());
        assert!(is_loop_edge(&graph, 0, 3) && is_loop_edge(&graph, 1, 3));
    }
}
