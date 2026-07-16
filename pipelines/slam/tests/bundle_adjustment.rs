use nalgebra::{Point3, UnitQuaternion, Vector3, Vector6};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Camera, Frame, Keyframe, Landmark, Observation, VisualMap};
use visloc_mapping::{LocalMapWindow, LocalRefinementReason, LocalRefiner, StagedMapUpdate};
use visloc_slam::{
    pairwise_pose_factors_from_loop_closures, BaConfig, BaError, BaGeneralStereoObservation,
    BaObservation, BaStereoObservation, BiasRandomWalkFactor, BundleAdjustment,
    BundleAdjustmentRefiner, GravityPrior, ImuPreintegrationFactor, ImuPreintegrator, LinearSolver,
    LoopClosureConstraint, PairwisePoseFactor, PerPoseGravityObservation, PerPoseGravityPrior,
    PositionPrior, PositionPriorObservation, RobustKernel,
};

fn pinhole() -> Camera {
    Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
}

fn pose_at(camera_center_world: Vector3<f64>) -> Pose {
    let r = UnitQuaternion::identity();
    let t = -(r.transform_vector(&camera_center_world));
    Pose::from_world_to_camera(r, t)
}

fn pose_with_yaw(camera_center_world: Vector3<f64>, yaw_rad: f64) -> Pose {
    let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw_rad);
    let t = -(r.transform_vector(&camera_center_world));
    Pose::from_world_to_camera(r, t)
}

fn world_grid() -> Vec<(u64, Point3<f64>)> {
    // Six points in a small box ahead of the cameras.
    vec![
        (1, Point3::new(-1.0, -0.5, 5.0)),
        (2, Point3::new(1.0, -0.5, 5.0)),
        (3, Point3::new(-1.0, 0.5, 5.0)),
        (4, Point3::new(1.0, 0.5, 5.0)),
        (5, Point3::new(0.0, 0.0, 6.0)),
        (6, Point3::new(0.5, 0.25, 7.0)),
    ]
}

/// Build an exact (residual=0) bundle by projecting world points through
/// truth poses to obtain measurements.
fn truth_bundle() -> BundleAdjustment {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.0, 0.0))),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    let points = world_grid();
    for (id, point) in &points {
        ba.add_landmark(*id, *point);
    }
    for (kf_id, pose) in &truth_poses {
        for (lm_id, point) in &points {
            let xc = pose.transform_world_point(point);
            let uv = camera.project(&xc).expect("point in front of camera");
            ba.add_observation(BaObservation {
                keyframe_id: *kf_id,
                landmark_id: *lm_id,
                xy: uv,
            });
        }
    }
    ba
}

/// Like [`truth_bundle`] but the measurements are rendered through a camera that
/// carries radial distortion `(k1, k2)`, so a distortion-free BA cannot fit them.
fn truth_bundle_distorted(k1: f64, k2: f64) -> BundleAdjustment {
    let camera = Camera::pinhole_radial(1, 640, 480, 500.0, 500.0, 320.0, 240.0, k1, k2);
    let mut ba = BundleAdjustment::new(camera.clone());
    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.0, 0.0))),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    let points = world_grid();
    for (id, point) in &points {
        ba.add_landmark(*id, *point);
    }
    for (kf_id, pose) in &truth_poses {
        for (lm_id, point) in &points {
            let xc = pose.transform_world_point(point);
            // `Camera::project` applies the radial distortion baked into `camera`.
            let uv = camera.project(&xc).expect("point in front of camera");
            ba.add_observation(BaObservation {
                keyframe_id: *kf_id,
                landmark_id: *lm_id,
                xy: uv,
            });
        }
    }
    ba
}

#[test]
fn joint_ba_self_calibrates_radial_distortion() {
    // Measurements are rendered through a lens with radial distortion
    // (k1 = -0.05, k2 = 0.01). Start the BA from a distortion-free pinhole and
    // let the joint solve self-calibrate (k1, k2) alongside the intrinsics. As
    // with the focal-length test, fixing the structure makes the distortion
    // observable on this tiny scene (a large many-view reconstruction does so for
    // free).
    let (true_k1, true_k2) = (-0.05, 0.01);
    let mut ba = truth_bundle_distorted(true_k1, true_k2);
    ba.camera = pinhole(); // distortion-free start (k1 = k2 = 0)
    ba.fix_pose(10);
    ba.fix_pose(20);
    for (id, _) in world_grid() {
        ba.fix_landmark(id);
    }

    let cost_before = ba.cost();
    let config = BaConfig {
        refine_intrinsics: true,
        refine_distortion: true,
        ..BaConfig::default()
    };
    ba.optimize(&config)
        .expect("joint BA with distortion refinement");

    let (k1, k2) = ba
        .camera
        .radial_distortion()
        .expect("distortion slots populated");
    assert!(
        (k1 - true_k1).abs() < 5.0e-3 && (k2 - true_k2).abs() < 5.0e-3,
        "distortion not recovered: k1={k1}, k2={k2} (truth {true_k1}/{true_k2})"
    );
    assert!(
        ba.cost() < 0.1 && ba.cost() < cost_before,
        "cost must collapse once the lens is recovered: {} (was {cost_before})",
        ba.cost()
    );
}

#[test]
fn refine_intrinsics_recovers_perturbed_focal_length() {
    // Exact observations are generated with the true camera (fx = fy = 500).
    // Start the BA from a wrong focal length and let alternating intrinsics
    // refinement pull it back, the COLMAP small-scene-accuracy lever.
    let mut ba = truth_bundle();
    // Perturb the intrinsics the solver starts from (truth is 500/500/320/240).
    ba.camera = Camera::pinhole(1, 640, 480, 520.0, 515.0, 326.0, 233.0);
    // Fix two poses and all landmarks so the wrong focal length cannot be absorbed
    // into structure — on this tiny 3-camera scene a free point cloud would soak
    // up the fx error (the focal/depth ambiguity), leaving nothing for the
    // intrinsics step to correct. Pinning the structure makes fx observable, which
    // is exactly what a large many-view reconstruction does for free.
    ba.fix_pose(10);
    ba.fix_pose(20);
    for (id, _) in world_grid() {
        ba.fix_landmark(id);
    }

    let cost_before = ba.cost();
    let config = BaConfig {
        refine_intrinsics: true,
        ..BaConfig::default()
    };
    ba.optimize(&config).expect("BA with intrinsics refinement");

    let (fx, fy, cx, cy) = ba.camera.intrinsics().unwrap();
    assert!(
        (fx - 500.0).abs() < 0.5 && (fy - 500.0).abs() < 0.5,
        "focal length not recovered: fx={fx}, fy={fy} (truth 500)"
    );
    assert!(
        (cx - 320.0).abs() < 0.5 && (cy - 240.0).abs() < 0.5,
        "principal point not recovered: cx={cx}, cy={cy} (truth 320/240)"
    );
    assert!(
        ba.cost() < 0.1 && ba.cost() < cost_before,
        "cost must collapse once the camera is recovered: {} (was {cost_before})",
        ba.cost()
    );
}

#[test]
fn refine_intrinsics_is_noop_when_disabled() {
    // With the flag off, optimize() must leave the (wrong) intrinsics untouched —
    // the default path is unchanged.
    let mut ba = truth_bundle();
    ba.camera = Camera::pinhole(1, 640, 480, 520.0, 520.0, 320.0, 240.0);
    ba.fix_pose(10);
    ba.fix_pose(20);
    ba.optimize(&BaConfig::default()).expect("plain BA");
    let (fx, _, _, _) = ba.camera.intrinsics().unwrap();
    assert!(
        (fx - 520.0).abs() < 1e-9,
        "intrinsics must be untouched when refine_intrinsics is off, got fx={fx}"
    );
}

#[test]
fn bundle_cost_is_zero_at_truth() {
    let ba = truth_bundle();
    assert!(
        ba.cost() < 1.0e-20,
        "cost at truth must vanish: {}",
        ba.cost()
    );
}

#[test]
fn bundle_optimize_with_no_data_returns_appropriate_errors() {
    let camera = pinhole();
    let mut empty = BundleAdjustment::new(camera.clone());
    assert_eq!(empty.optimize(&BaConfig::default()), Err(BaError::NoPoses));

    // With a pose but no observations / IMU factors, the solver bails with
    // `NoObservations`. The historical check order (poses → landmarks →
    // observations) was relaxed in vi_motion_initializer's first cut so
    // the inertial-only VIBA1 stage can run with `landmarks.is_empty()`;
    // `NoLandmarks` now only fires when visual observations are present
    // but no landmarks are registered.
    empty.add_pose(1, pose_at(Vector3::zeros()));
    assert_eq!(
        empty.optimize(&BaConfig::default()),
        Err(BaError::NoObservations)
    );

    // Add a visual observation referencing a landmark that does not
    // exist yet. The solver now reaches the visual-residual path and
    // demands a landmark.
    empty.add_observation(BaObservation {
        keyframe_id: 1,
        landmark_id: 99,
        xy: nalgebra::Point2::new(320.0, 240.0),
    });
    assert_eq!(
        empty.optimize(&BaConfig::default()),
        Err(BaError::NoLandmarks)
    );
}

#[test]
fn bundle_optimize_pulls_drifted_pose_back_to_truth() {
    let mut ba = truth_bundle();
    // Fix the first two poses to remove SE(3) + scale gauge for monocular BA.
    ba.fix_pose(10);
    ba.fix_pose(20);

    // Drift the third pose so the BA has work to do.
    let drifted_30 = pose_at(Vector3::new(1.05, 0.02, 0.03));
    ba.add_pose(30, drifted_30);

    let cost_before = ba.cost();
    assert!(
        cost_before > 1.0e-3,
        "expected nontrivial drift, got {cost_before}"
    );

    let result = ba
        .optimize(&BaConfig::default())
        .expect("BA must succeed on a well-fixed bundle");
    assert!(
        result.final_cost < 1.0e-12,
        "final cost too high: {}",
        result.final_cost
    );
    assert!(result.final_cost < result.initial_cost);
    assert!(result.converged, "BA should converge: {:?}", result);

    let recovered = ba.poses[&30].camera_center_world();
    assert!(
        (recovered - Point3::new(1.0, 0.0, 0.0)).norm() < 1.0e-6,
        "recovered KF30 center {recovered:?}"
    );
}

#[test]
fn bundle_optimize_recovers_drifted_landmark_from_two_views() {
    let mut ba = truth_bundle();
    ba.fix_pose(10);
    ba.fix_pose(20);
    ba.fix_pose(30);
    // Drift one landmark in 3D; observations should pull it back since
    // three fixed views over-constrain its position.
    let drifted_5 = Point3::new(0.10, 0.05, 6.20);
    ba.add_landmark(5, drifted_5);

    let cost_before = ba.cost();
    assert!(cost_before > 1.0e-2);

    let result = ba.optimize(&BaConfig::default()).expect("BA must succeed");
    assert!(result.final_cost < 1.0e-14, "{}", result.final_cost);

    let recovered = ba.landmarks[&5];
    assert!(
        (recovered - Point3::new(0.0, 0.0, 6.0)).norm() < 1.0e-7,
        "recovered LM5 {recovered:?}"
    );
}

#[test]
fn bundle_lm_does_not_reduce_cost_by_moving_observation_behind_camera() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera);
    ba.add_pose(10, pose_at(Vector3::zeros()));
    ba.fix_pose(10);
    ba.add_landmark(1, Point3::new(1.0, 0.0, 0.1));
    // This deliberately inconsistent measurement asks the local linear model
    // to make depth negative.  A cost-only accept test would then see the
    // observation disappear from the objective and accept a zero-cost state.
    ba.add_observation(BaObservation {
        keyframe_id: 10,
        landmark_id: 1,
        xy: nalgebra::Point2::new(105_320.0, 240.0),
    });

    let initial = ba.landmarks[&1];
    let result = ba.optimize(&BaConfig {
        max_iterations: 1,
        ..BaConfig::default()
    });
    let result = result.expect("damped BA must return a rejected iteration");

    assert_eq!(ba.landmarks[&1], initial, "infeasible step must roll back");
    assert!(ba.landmarks[&1].z > 0.0);
    assert_eq!(result.iterations.len(), 1);
    assert!(!result.iterations[0].step_accepted);
    assert_eq!(result.final_cost, result.initial_cost);
}

#[test]
fn bundle_optimize_jointly_with_yaw_drift() {
    // Three poses, two fixed at truth, third drifted in both translation
    // and yaw. Poses + landmarks should coexist in the same LM iteration.
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    let truth_poses = [
        (10u64, pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0)),
        (20u64, pose_with_yaw(Vector3::new(0.5, 0.0, 0.2), 0.05)),
        (30u64, pose_with_yaw(Vector3::new(1.0, 0.0, 0.4), 0.1)),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
    }
    for (kf_id, pose) in &truth_poses {
        for (lm_id, point) in world_grid() {
            let xc = pose.transform_world_point(&point);
            let uv = camera.project(&xc).expect("project succeeds");
            ba.add_observation(BaObservation {
                keyframe_id: *kf_id,
                landmark_id: lm_id,
                xy: uv,
            });
        }
    }
    ba.fix_pose(10);
    ba.fix_pose(20);

    // Drift the third pose by 5 cm in translation and 1° in yaw.
    let drifted_30 = pose_with_yaw(Vector3::new(1.05, -0.02, 0.42), 0.115);
    ba.add_pose(30, drifted_30);

    let result = ba.optimize(&BaConfig::default()).expect("BA must succeed");
    assert!(
        result.final_cost < 1.0e-12,
        "BA failed to converge: {}",
        result.final_cost
    );

    let recovered = ba.poses[&30].camera_center_world();
    let expected = Vector3::new(1.0, 0.0, 0.4);
    assert!(
        (recovered.coords - expected).norm() < 1.0e-6,
        "recovered KF30 center {recovered:?}"
    );
}

#[test]
fn bundle_sparse_solver_matches_dense_on_small_scene() {
    let truth = truth_bundle();
    let mut dense = truth.clone();
    let mut sparse = truth;
    dense.fix_pose(10);
    dense.fix_pose(20);
    sparse.fix_pose(10);
    sparse.fix_pose(20);

    // Same drift on KF30 in both copies.
    let drifted_30 = pose_at(Vector3::new(1.04, 0.01, -0.02));
    dense.add_pose(30, drifted_30.clone());
    sparse.add_pose(30, drifted_30);

    let dense_result = dense
        .optimize(&BaConfig {
            linear_solver: LinearSolver::Dense,
            ..BaConfig::default()
        })
        .expect("dense BA must succeed");
    let sparse_result = sparse
        .optimize(&BaConfig {
            linear_solver: LinearSolver::Sparse,
            ..BaConfig::default()
        })
        .expect("sparse BA must succeed");

    assert_eq!(dense_result.converged, sparse_result.converged);
    assert!(
        (dense_result.final_cost - sparse_result.final_cost).abs() < 1.0e-12,
        "cost mismatch dense={} sparse={}",
        dense_result.final_cost,
        sparse_result.final_cost
    );
    let dense_center = dense.poses[&30].camera_center_world();
    let sparse_center = sparse.poses[&30].camera_center_world();
    assert!(
        (dense_center - sparse_center).norm() < 1.0e-9,
        "pose center mismatch dense={dense_center:?} sparse={sparse_center:?}"
    );
}

#[test]
fn ba_refiner_pulls_drifted_staged_keyframe_back_to_truth() {
    // Build a VisualMap with two truth keyframes plus six landmarks. Each
    // existing keyframe carries observations that fix the landmark gauge.
    let camera = pinhole();
    let truth_10 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let truth_20 = pose_at(Vector3::new(0.5, 0.0, 0.0));
    let truth_30 = pose_at(Vector3::new(1.0, 0.0, 0.0));

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());

    let landmarks_truth = world_grid();
    for (id, point) in &landmarks_truth {
        let mut lm = Landmark::new(*id, *point);
        lm.descriptor = Some(vec![0.0]);
        map.landmarks.insert(*id, lm);
    }

    fn build_keyframe(
        camera: &Camera,
        frame_id: u64,
        pose: &Pose,
        landmarks: &[(u64, Point3<f64>)],
    ) -> Keyframe {
        let mut frame = Frame::new(frame_id, camera.id);
        frame.pose = Some(pose.clone());
        let mut observations: Vec<Observation> = Vec::new();
        for (idx, (lm_id, point)) in landmarks.iter().enumerate() {
            let xc = pose.transform_world_point(point);
            let uv = camera.project(&xc).expect("project succeeds in fixture");
            frame.keypoints.push(uv);
            frame.descriptors.push(vec![0.0]);
            observations.push(Observation {
                frame_id,
                landmark_id: *lm_id,
                keypoint_index: idx,
                xy: uv,
            });
        }
        Keyframe {
            frame,
            observations,
        }
    }

    let kf_10 = build_keyframe(&camera, 10, &truth_10, &landmarks_truth);
    let kf_20 = build_keyframe(&camera, 20, &truth_20, &landmarks_truth);
    // Replicate observations into landmark.observations as well so window
    // reconstruction stays consistent with how the local-mapping pipeline
    // would have populated the map.
    for kf in [&kf_10, &kf_20] {
        for obs in &kf.observations {
            if let Some(lm) = map.landmarks.get_mut(&obs.landmark_id) {
                lm.observations.push(obs.clone());
            }
        }
        map.keyframes.insert(kf.frame.id, kf.clone());
    }

    // Build the new keyframe at truth, then drift its pose. Its
    // observations are noise-free (computed from truth) but the pose is
    // off, so BA can pull pose to truth using the fixed gauge.
    let drifted_30 = pose_at(Vector3::new(1.04, -0.01, 0.02));
    let mut kf_30_drifted = build_keyframe(&camera, 30, &truth_30, &landmarks_truth);
    kf_30_drifted.frame.pose = Some(drifted_30.clone());

    // working_map = map ∪ {new keyframe} (mirrors what LocalMappingPipeline
    // does internally before invoking the refiner).
    let mut working_map = map.clone();
    working_map.keyframes.insert(30, kf_30_drifted.clone());

    // Local window anchored on the new keyframe.
    let local_window = LocalMapWindow::from_anchor(
        &working_map,
        30,
        &visloc_mapping::LocalMapWindowConfig { max_keyframes: 5 },
    );
    assert!(local_window.keyframe_ids.contains(&30));
    assert_eq!(local_window.landmark_ids.len(), landmarks_truth.len());

    let mut staged_update = StagedMapUpdate::new();
    staged_update.stage_keyframe(kf_30_drifted);

    let refiner = BundleAdjustmentRefiner::default();
    let refinement = refiner.refine(&working_map, &local_window, &mut staged_update);
    assert!(refinement.refined, "refinement should run: {refinement:?}");
    assert_eq!(refinement.reason, LocalRefinementReason::Refined);
    assert_eq!(refinement.keyframe_count, 1);
    assert_eq!(refinement.landmark_count, 0);

    let refined_pose = staged_update.keyframes[0]
        .frame
        .pose
        .as_ref()
        .expect("refined pose populated");
    let recovered_center = refined_pose.camera_center_world();
    assert!(
        (recovered_center - Point3::new(1.0, 0.0, 0.0)).norm() < 1.0e-6,
        "refined KF30 center {recovered_center:?}"
    );
}

/// Helper: build an over-determined bundle with 4 fixed-anchor keyframes
/// plus one variable keyframe that starts at truth, projecting 12
/// landmarks observed from every keyframe (60 observations). Optionally
/// corrupt one observation on the variable keyframe by a 50-pixel shift
/// to test robust outlier rejection. Starting at truth means the inlier
/// residuals are zero — the only signal driving the optimizer is the
/// outlier, so the robust kernel's job is unambiguous: down-weight it.
fn over_determined_bundle_with_optional_outlier(inject_outlier: bool) -> BundleAdjustment {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let truth_poses: Vec<(u64, Pose)> = vec![
        (10, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30, pose_at(Vector3::new(1.0, 0.0, 0.0))),
        (40, pose_at(Vector3::new(1.5, 0.0, 0.0))),
        (50, pose_at(Vector3::new(2.0, 0.0, 0.0))),
    ];
    let mut truth_landmarks: Vec<(u64, Point3<f64>)> = Vec::new();
    for r in 0..3 {
        for c in 0..4 {
            let id = (r * 4 + c + 1) as u64;
            let x = -1.5 + 1.0 * c as f64;
            let y = -1.0 + 1.0 * r as f64;
            let z = 5.0 + 0.2 * ((r * 4 + c) as f64).sin();
            truth_landmarks.push((id, Point3::new(x, y, z)));
        }
    }

    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    for (id, p) in &truth_landmarks {
        ba.add_landmark(*id, *p);
    }
    for (kf_id, pose) in &truth_poses {
        for (lm_id, p) in &truth_landmarks {
            let xc = pose.transform_world_point(p);
            let uv = camera.project(&xc).expect("project");
            ba.add_observation(BaObservation {
                keyframe_id: *kf_id,
                landmark_id: *lm_id,
                xy: uv,
            });
        }
    }
    // Fix 4 of 5 keyframes; KF50 stays at truth as the only variable pose.
    ba.fix_pose(10);
    ba.fix_pose(20);
    ba.fix_pose(30);
    ba.fix_pose(40);

    if inject_outlier {
        // Corrupt one observation: KF50's view of LM1.
        for obs in ba.observations.iter_mut() {
            if obs.keyframe_id == 50 && obs.landmark_id == 1 {
                obs.xy.x += 50.0;
                obs.xy.y += 50.0;
                break;
            }
        }
    }
    ba
}

#[test]
fn bundle_robust_kernel_none_matches_pure_gauss_newton() {
    // Sanity check: kernel=None should behave identically to pre-robust BA.
    let mut without_kernel = over_determined_bundle_with_optional_outlier(false);
    let mut with_kernel = over_determined_bundle_with_optional_outlier(false);
    let r_a = without_kernel
        .optimize(&BaConfig::default())
        .expect("default BA must succeed");
    let r_b = with_kernel
        .optimize(&BaConfig {
            robust_kernel: RobustKernel::None,
            ..BaConfig::default()
        })
        .expect("explicit kernel=None BA must succeed");
    assert!(
        (r_a.final_cost - r_b.final_cost).abs() < 1.0e-15,
        "default and explicit None must match: {} vs {}",
        r_a.final_cost,
        r_b.final_cost,
    );
}

#[test]
fn confidence_weight_zero_mutes_a_bad_correspondence() {
    let mut ba = over_determined_bundle_with_optional_outlier(true);
    let mut weights = vec![1.0; ba.observations.len()];
    let bad_index = ba
        .observations
        .iter()
        .position(|obs| obs.keyframe_id == 50 && obs.landmark_id == 1)
        .expect("injected outlier");
    weights[bad_index] = 0.0;

    let weighted_breakdown = ba
        .cost_breakdown_with_observation_weights(&RobustKernel::None, &weights)
        .expect("weighted diagnostic cost");
    assert!(
        weighted_breakdown.visual < 1.0e-18,
        "muted outlier remained in diagnostic cost: {}",
        weighted_breakdown.visual
    );

    let result = ba
        .optimize_with_observation_weights(&BaConfig::default(), &weights)
        .expect("confidence-weighted BA");
    let recovered = ba.poses[&50].camera_center_world();

    assert!(
        recovered
            .coords
            .metric_distance(&Vector3::new(2.0, 0.0, 0.0))
            < 1.0e-10,
        "zero-confidence outlier moved the pose to {recovered:?}"
    );
    assert!(
        result.final_cost < 1.0e-18,
        "muted outlier remained in weighted cost: {}",
        result.final_cost
    );
}

#[test]
fn confidence_weights_reject_invalid_inputs() {
    let config = BaConfig::default();
    let mut ba = over_determined_bundle_with_optional_outlier(false);
    let expected = ba.observations.len();
    assert_eq!(
        ba.optimize_with_observation_weights(&config, &[1.0]),
        Err(BaError::ObservationWeightCount {
            expected,
            actual: 1,
        })
    );

    let mut weights = vec![1.0; expected];
    weights[3] = f64::NAN;
    assert_eq!(
        ba.optimize_with_observation_weights(&config, &weights),
        Err(BaError::InvalidObservationWeight(3))
    );
}

#[test]
fn bundle_robust_cost_clips_large_residuals_below_quadratic() {
    // Cost-function semantics: at a state with a single 50-pixel outlier,
    // the squared cost is `50² + 50² = 5000` while Huber(δ=10) returns
    // `2·δ·√s − δ² = 20·√5000 − 100 ≈ 1314` and Cauchy(c=10) returns
    // `c²·log(1 + s/c²) = 100·log(51) ≈ 393`. The robust cost must always
    // sit strictly below the squared cost when residuals exceed the
    // kernel scale.
    let ba = over_determined_bundle_with_optional_outlier(true);
    let squared = ba.cost();
    let huber = ba.robust_cost(&RobustKernel::Huber { delta: 10.0 });
    let cauchy = ba.robust_cost(&RobustKernel::Cauchy { c: 10.0 });
    assert!(
        huber < squared,
        "Huber must clip outliers: {huber} vs {squared}"
    );
    assert!(
        cauchy < huber,
        "Cauchy clips harder than Huber: {cauchy} vs {huber}"
    );
    assert!(
        (huber - 1314.0).abs() < 1.0,
        "Huber cost matches analytic value 1314: {huber}"
    );
}

#[test]
fn bundle_huber_kernel_lowers_robust_cost_vs_non_robust() {
    // When the BA bundle contains an outlier, the optimizer that uses a
    // Huber kernel finds a state with strictly lower Huber-cost than the
    // optimizer that runs unweighted (since the latter is biased by the
    // outlier into a state that the Huber objective considers worse).
    let mut non_robust = over_determined_bundle_with_optional_outlier(true);
    non_robust
        .optimize(&BaConfig::default())
        .expect("non-robust BA must terminate");
    let non_robust_huber_cost = non_robust.robust_cost(&RobustKernel::Huber { delta: 10.0 });

    let mut robust = over_determined_bundle_with_optional_outlier(true);
    let result = robust
        .optimize(&BaConfig {
            robust_kernel: RobustKernel::Huber { delta: 10.0 },
            max_iterations: 50,
            ..BaConfig::default()
        })
        .expect("robust BA must terminate");
    assert!(
        result.final_cost < non_robust_huber_cost,
        "robust optimizer should beat non-robust on Huber objective: \
         {} vs {}",
        result.final_cost,
        non_robust_huber_cost,
    );
    assert!(result.final_cost.is_finite());
}

/// Build a rectified-stereo bundle with three poses and the world grid,
/// projecting each landmark through every left camera and through the right
/// camera (left-camera frame translated by `+baseline · x̂`).
fn truth_stereo_bundle(baseline: f64) -> BundleAdjustment {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    ba.set_stereo_baseline(baseline);

    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.0, 0.0))),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    let points = world_grid();
    for (id, point) in &points {
        ba.add_landmark(*id, *point);
    }
    let (fx, _fy, _cx, _cy) = camera.intrinsics().unwrap();
    for (kf_id, pose) in &truth_poses {
        for (lm_id, point) in &points {
            let xc = pose.transform_world_point(point);
            let uv = camera.project(&xc).expect("point in front of camera");
            let u_right = uv.x - fx * baseline / xc.z;
            ba.add_stereo_observation(BaStereoObservation {
                keyframe_id: *kf_id,
                landmark_id: *lm_id,
                xy: uv,
                u_right,
            });
        }
    }
    ba
}

#[test]
fn bundle_stereo_cost_is_zero_at_truth() {
    let ba = truth_stereo_bundle(0.5);
    assert!(
        ba.cost() < 1.0e-20,
        "stereo cost at truth must vanish: {}",
        ba.cost()
    );
}

#[test]
fn bundle_stereo_optimize_rejects_missing_baseline() {
    let mut ba = truth_stereo_bundle(0.5);
    ba.stereo_baseline = None;
    assert_eq!(
        ba.optimize(&BaConfig::default()),
        Err(BaError::MissingStereoBaseline)
    );
    ba.stereo_baseline = Some(0.0);
    assert_eq!(
        ba.optimize(&BaConfig::default()),
        Err(BaError::MissingStereoBaseline)
    );
    ba.stereo_baseline = Some(f64::NAN);
    assert_eq!(
        ba.optimize(&BaConfig::default()),
        Err(BaError::MissingStereoBaseline)
    );
}

#[test]
fn bundle_stereo_recovers_metric_pose_with_single_anchor() {
    // Stereo BA has no scale gauge, so a single fixed pose is enough to
    // remove all 6 DoF gauge freedom. This is the headline win over
    // monocular BA, which would also need a second fixed pose / landmark.
    let mut ba = truth_stereo_bundle(0.5);
    ba.fix_pose(10);

    let drifted_30 = pose_at(Vector3::new(1.07, 0.04, 0.05));
    ba.add_pose(30, drifted_30);

    let cost_before = ba.cost();
    assert!(cost_before > 1.0e-2);
    let result = ba
        .optimize(&BaConfig::default())
        .expect("stereo BA must succeed with one anchor");
    assert!(
        result.final_cost < 1.0e-12,
        "stereo BA didn't converge: {}",
        result.final_cost
    );
    let recovered = ba.poses[&30].camera_center_world();
    assert!(
        (recovered - Point3::new(1.0, 0.0, 0.0)).norm() < 1.0e-7,
        "recovered KF30 center {recovered:?}"
    );
    // KF20 (variable) must also be at metric truth, not at a scaled
    // version of it (the test that monocular BA would fail).
    let recovered_20 = ba.poses[&20].camera_center_world();
    assert!(
        (recovered_20 - Point3::new(0.5, 0.0, 0.0)).norm() < 1.0e-7,
        "recovered KF20 center {recovered_20:?}"
    );
}

#[test]
fn bundle_stereo_recovers_drifted_landmark() {
    let mut ba = truth_stereo_bundle(0.5);
    ba.fix_pose(10);
    ba.fix_pose(20);
    ba.fix_pose(30);
    let drifted = Point3::new(0.20, 0.10, 6.30);
    ba.add_landmark(5, drifted);

    let result = ba.optimize(&BaConfig::default()).expect("stereo BA");
    assert!(result.final_cost < 1.0e-14, "{}", result.final_cost);
    let recovered = ba.landmarks[&5];
    assert!(
        (recovered - Point3::new(0.0, 0.0, 6.0)).norm() < 1.0e-8,
        "recovered LM5 {recovered:?}"
    );
}

fn truth_general_stereo_bundle() -> (BundleAdjustment, Point3<f64>) {
    let left_camera = pinhole();
    let right_camera = Camera::pinhole(2, 640, 480, 510.0, 495.0, 318.0, 242.0);
    // Deliberately include rotation, vertical translation, and unequal
    // intrinsics: none of these can be represented by BaStereoObservation.
    let left_to_right = SE3::new(
        UnitQuaternion::from_euler_angles(0.01, -0.02, 0.015),
        Vector3::new(-0.11, 0.002, 0.001),
    );
    let pose = Pose::identity();
    let truth = Point3::new(0.2, -0.1, 4.0);
    let point_left = pose.transform_world_point(&truth);
    let point_right = left_to_right.transform_point(&point_left);
    let xy_left = left_camera.project(&point_left).expect("left projection");
    let xy_right = right_camera
        .project(&point_right)
        .expect("right projection");

    let mut ba = BundleAdjustment::new(left_camera);
    ba.add_pose(10, pose);
    ba.fix_pose(10);
    ba.add_landmark(5, truth);
    ba.add_general_stereo_observation(BaGeneralStereoObservation {
        keyframe_id: 10,
        landmark_id: 5,
        xy_left,
        xy_right,
        right_camera,
        left_to_right,
    });
    (ba, truth)
}

#[test]
fn bundle_general_stereo_cost_is_zero_at_truth() {
    let (ba, _) = truth_general_stereo_bundle();
    assert!(
        ba.cost() < 1.0e-20,
        "general stereo cost at truth must vanish: {}",
        ba.cost()
    );
}

#[test]
fn bundle_general_stereo_recovers_landmark_with_rotated_right_camera() {
    let (mut ba, truth) = truth_general_stereo_bundle();
    ba.add_landmark(5, Point3::new(0.45, 0.08, 4.8));
    let cost_before = ba.cost();
    assert!(cost_before > 1.0);

    let result = ba
        .optimize(&BaConfig::default())
        .expect("general stereo BA");
    assert!(
        result.final_cost < 1.0e-14,
        "general stereo BA did not converge: {}",
        result.final_cost
    );
    let recovered = ba.landmarks[&5];
    assert!(
        (recovered - truth).norm() < 1.0e-8,
        "recovered {recovered:?}, truth {truth:?}"
    );
}

#[test]
fn bundle_general_stereo_recovers_pose_with_rotated_right_camera() {
    let left_camera = pinhole();
    let right_camera = Camera::pinhole(2, 640, 480, 510.0, 495.0, 318.0, 242.0);
    let left_to_right = SE3::new(
        UnitQuaternion::from_euler_angles(0.01, -0.02, 0.015),
        Vector3::new(-0.11, 0.002, 0.001),
    );
    let truth_pose = pose_with_yaw(Vector3::new(0.25, -0.03, 0.04), 0.035);
    let mut ba = BundleAdjustment::new(left_camera.clone());
    ba.add_pose(
        10,
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.025, 0.07, -0.018),
            Vector3::new(-0.39, 0.12, -0.08),
        ),
    );
    for (landmark_id, point) in world_grid() {
        let point_left = truth_pose.transform_world_point(&point);
        let point_right = left_to_right.transform_point(&point_left);
        let xy_left = left_camera.project(&point_left).expect("left projection");
        let xy_right = right_camera
            .project(&point_right)
            .expect("right projection");
        ba.add_landmark(landmark_id, point);
        ba.fix_landmark(landmark_id);
        ba.add_general_stereo_observation(BaGeneralStereoObservation {
            keyframe_id: 10,
            landmark_id,
            xy_left,
            xy_right,
            right_camera: right_camera.clone(),
            left_to_right: left_to_right.clone(),
        });
    }

    let result = ba
        .optimize(&BaConfig::default())
        .expect("general stereo pose BA");
    assert!(
        result.final_cost < 1.0e-12,
        "general stereo pose BA did not converge: {}",
        result.final_cost
    );
    let recovered = &ba.poses[&10];
    assert!((recovered.camera_center_world() - truth_pose.camera_center_world()).norm() < 1.0e-7);
    assert!(
        recovered
            .world_to_camera
            .rotation
            .angle_to(&truth_pose.world_to_camera.rotation)
            < 1.0e-7
    );
}

#[test]
fn bundle_stereo_sparse_solver_matches_dense() {
    let truth = truth_stereo_bundle(0.5);
    let mut dense = truth.clone();
    let mut sparse = truth;
    dense.fix_pose(10);
    sparse.fix_pose(10);
    let drifted = pose_at(Vector3::new(1.03, -0.02, 0.04));
    dense.add_pose(30, drifted.clone());
    sparse.add_pose(30, drifted);

    let dense_result = dense
        .optimize(&BaConfig {
            linear_solver: LinearSolver::Dense,
            ..BaConfig::default()
        })
        .expect("dense stereo BA");
    let sparse_result = sparse
        .optimize(&BaConfig {
            linear_solver: LinearSolver::Sparse,
            ..BaConfig::default()
        })
        .expect("sparse stereo BA");
    assert_eq!(dense_result.converged, sparse_result.converged);
    assert!(
        (dense_result.final_cost - sparse_result.final_cost).abs() < 1.0e-12,
        "stereo BA dense vs sparse cost mismatch: {} vs {}",
        dense_result.final_cost,
        sparse_result.final_cost
    );
    let dense_center = dense.poses[&30].camera_center_world();
    let sparse_center = sparse.poses[&30].camera_center_world();
    assert!(
        (dense_center - sparse_center).norm() < 1.0e-9,
        "dense {dense_center:?} sparse {sparse_center:?}"
    );
}

#[test]
fn bundle_mixed_mono_and_stereo_observations_optimize_together() {
    // One pose's view is monocular only; the other two are full stereo.
    // The mixed BA must converge when the stereo evidence anchors metric
    // scale and the mono evidence pins down KF40's pose.
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    let baseline = 0.5;
    ba.set_stereo_baseline(baseline);

    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (40u64, pose_at(Vector3::new(1.5, 0.0, 0.0))),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    for (id, p) in world_grid() {
        ba.add_landmark(id, p);
    }
    let (fx, _fy, _cx, _cy) = camera.intrinsics().unwrap();
    for (kf_id, pose) in &truth_poses {
        for (lm_id, p) in world_grid() {
            let xc = pose.transform_world_point(&p);
            let uv = camera.project(&xc).expect("project");
            if *kf_id == 40 {
                ba.add_observation(BaObservation {
                    keyframe_id: *kf_id,
                    landmark_id: lm_id,
                    xy: uv,
                });
            } else {
                let u_right = uv.x - fx * baseline / xc.z;
                ba.add_stereo_observation(BaStereoObservation {
                    keyframe_id: *kf_id,
                    landmark_id: lm_id,
                    xy: uv,
                    u_right,
                });
            }
        }
    }
    ba.fix_pose(10);
    let drifted_40 = pose_at(Vector3::new(1.55, 0.04, 0.06));
    ba.add_pose(40, drifted_40);

    let result = ba
        .optimize(&BaConfig::default())
        .expect("mixed mono+stereo BA must succeed");
    assert!(
        result.final_cost < 1.0e-12,
        "mixed BA cost {}",
        result.final_cost
    );
    let recovered_40 = ba.poses[&40].camera_center_world();
    assert!(
        (recovered_40 - Point3::new(1.5, 0.0, 0.0)).norm() < 1.0e-7,
        "mixed BA failed to recover KF40: {recovered_40:?}"
    );
}

/// A gravity prior on an already-correct trajectory should add zero
/// cost when both `g_world` and `g_camera_observed` agree with every
/// pose's rotation. This validates the residual computation and
/// ensures the prior does not "drag" a clean solution.
#[test]
fn gravity_prior_zero_cost_on_consistent_trajectory() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    // Three forward-translating poses, all with identity rotation.
    let poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.0, 0.0))),
    ];
    for (id, pose) in &poses {
        ba.add_pose(*id, pose.clone());
    }
    ba.fix_pose(10);
    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
    }
    for (pose_id, pose) in &poses {
        for (lm_id, point) in world_grid() {
            let xc = pose.transform_world_point(&point);
            let predicted = camera.project(&xc).unwrap();
            ba.add_observation(BaObservation {
                keyframe_id: *pose_id,
                landmark_id: lm_id,
                xy: predicted,
            });
        }
    }
    // Gravity in world frame = (0, 9.81, 0). Camera frame is the same
    // because rotations are identity for every pose.
    ba.set_gravity_prior(GravityPrior {
        g_world: Vector3::new(0.0, 9.81, 0.0),
        g_camera_observed: Vector3::new(0.0, 9.81, 0.0),
        weight: 1.0,
    });
    let cost = ba.cost();
    assert!(
        cost < 1.0e-20,
        "level-prior + level trajectory must give zero cost, got {cost}"
    );
}

/// Inject a pitch perturbation on one non-anchor pose; BA augmented
/// with a level gravity prior should recover the level rotation. The
/// reprojection residuals alone leave a rotation ambiguity (depth +
/// pitch trade-off) on a small-baseline 3-frame stereo bundle, so this
/// test specifically exercises the prior's effect on the rotation
/// gauge.
#[test]
fn gravity_prior_recovers_pitched_pose() {
    use nalgebra::UnitQuaternion;
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    // Three forward-translating poses, all with identity rotation as the
    // ground-truth. The middle pose will be perturbed.
    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.0, 0.0))),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    // Pitch the middle pose by +0.15 rad (~8.6°) around the camera x-axis.
    // KITTI-style cameras use x=right, y=down, z=forward, so this is a
    // forward / nose-up pitch.
    let pitch_rad = 0.15_f64;
    let r_pitch = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch_rad);
    let truth_center_20 = Vector3::new(0.5, 0.0, 0.0);
    let pitched_translation = -(r_pitch.transform_vector(&truth_center_20));
    *ba.poses.get_mut(&20).unwrap() = Pose::from_world_to_camera(r_pitch, pitched_translation);
    ba.fix_pose(10);

    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
    }
    // Generate measurements from the truth poses so the pitched pose 20
    // has non-zero reprojection residuals when seen against truth obs.
    for (pose_id, pose) in &truth_poses {
        for (lm_id, point) in world_grid() {
            let xc = pose.transform_world_point(&point);
            let predicted = camera.project(&xc).unwrap();
            ba.add_observation(BaObservation {
                keyframe_id: *pose_id,
                landmark_id: lm_id,
                xy: predicted,
            });
        }
    }

    ba.set_gravity_prior(GravityPrior {
        g_world: Vector3::new(0.0, 9.81, 0.0),
        g_camera_observed: Vector3::new(0.0, 9.81, 0.0),
        weight: 5.0,
    });
    let config = BaConfig {
        max_iterations: 30,
        initial_lambda: Some(1.0e-2),
        ..BaConfig::default()
    };
    let _ = ba.optimize(&config).expect("BA + gravity prior must run");

    // Recovered rotation at pose 20 should be near identity.
    let recovered_r = ba.poses[&20].world_to_camera.rotation;
    let identity = UnitQuaternion::identity();
    let angle_err = recovered_r.angle_to(&identity);
    assert!(
        angle_err < 0.02,
        "level prior must pull pitched pose back to level; angle_err={angle_err} rad"
    );
}

/// Pure horizontal-translation drift is invisible to a rotation-only
/// gravity prior — the residual `R · g_w − g_obs` depends on rotation
/// alone. This test documents that limitation so callers expecting the
/// prior to fix translation drift get a fast empirical reminder.
#[test]
fn gravity_prior_does_not_correct_translation_drift() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    ba.fix_pose(10);
    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
    }
    // Generate measurements from truth poses.
    for (pose_id, pose) in &truth_poses {
        for (lm_id, point) in world_grid() {
            let xc = pose.transform_world_point(&point);
            let predicted = camera.project(&xc).unwrap();
            ba.add_observation(BaObservation {
                keyframe_id: *pose_id,
                landmark_id: lm_id,
                xy: predicted,
            });
        }
    }
    // Perturb pose 20 in PURE translation (no rotation change).
    let drifted_20 = pose_at(Vector3::new(0.5, 0.10, 0.0));
    ba.poses.insert(20, drifted_20);
    let cost_before_prior = ba.cost();
    ba.set_gravity_prior(GravityPrior {
        g_world: Vector3::new(0.0, 9.81, 0.0),
        g_camera_observed: Vector3::new(0.0, 9.81, 0.0),
        weight: 100.0,
    });
    let cost_with_prior = ba.cost();
    // Cost equals reprojection-only cost: the gravity contribution is
    // zero for both poses (identity rotation), so the prior adds
    // nothing for a pure-translation drift.
    let prior_contribution = cost_with_prior - cost_before_prior;
    assert!(
        prior_contribution.abs() < 1.0e-9,
        "rotation-only gravity prior must add zero cost on pure translation drift; got {prior_contribution}"
    );
}

/// A per-keyframe gravity prior with `g_camera_observed` matching
/// every pose's true rotation should add zero cost. This validates
/// the per-observation residual path and ensures the prior does not
/// drag a clean solution.
#[test]
fn per_pose_gravity_prior_zero_cost_on_consistent_trajectory() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.0, 0.0))),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    ba.fix_pose(10);
    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
    }
    for (pose_id, pose) in &truth_poses {
        for (lm_id, point) in world_grid() {
            let xc = pose.transform_world_point(&point);
            let predicted = camera.project(&xc).unwrap();
            ba.add_observation(BaObservation {
                keyframe_id: *pose_id,
                landmark_id: lm_id,
                xy: predicted,
            });
        }
    }
    // Per-keyframe observations: every pose has identity rotation, so
    // each pose's gravity-in-camera-frame equals g_world exactly.
    let mut prior = PerPoseGravityPrior::new(Vector3::new(0.0, 9.81, 0.0), 1.0);
    for (id, _pose) in &truth_poses {
        prior.push(PerPoseGravityObservation {
            keyframe_id: *id,
            g_camera_observed: Vector3::new(0.0, 9.81, 0.0),
            weight: 1.0,
        });
    }
    ba.set_per_pose_gravity_prior(prior);
    let cost = ba.cost();
    assert!(
        cost < 1.0e-20,
        "per-pose level prior + level trajectory must give zero cost, got {cost}"
    );
}

/// Two non-anchor poses pitched in opposite directions; a per-pose
/// gravity prior that observes the correct (level) direction at each
/// keyframe should pull both back to level. This exercises the
/// per-keyframe Jacobian path with non-uniform observations.
#[test]
fn per_pose_gravity_prior_recovers_per_keyframe_pitch() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.0, 0.0))),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    ba.fix_pose(10);
    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
    }
    for (pose_id, pose) in &truth_poses {
        for (lm_id, point) in world_grid() {
            let xc = pose.transform_world_point(&point);
            let predicted = camera.project(&xc).unwrap();
            ba.add_observation(BaObservation {
                keyframe_id: *pose_id,
                landmark_id: lm_id,
                xy: predicted,
            });
        }
    }
    // Pitch pose 20 by +0.12 rad (~6.9°) around camera x-axis (nose up)
    // and pose 30 by -0.10 rad (nose down). The reprojection cost alone
    // does not uniquely determine the rotation; the prior should drive
    // both back to level.
    let pitch_20 = 0.12_f64;
    let pitch_30 = -0.10_f64;
    let r_pitch_20 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch_20);
    let r_pitch_30 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch_30);
    let center_20 = Vector3::new(0.5, 0.0, 0.0);
    let center_30 = Vector3::new(1.0, 0.0, 0.0);
    *ba.poses.get_mut(&20).unwrap() =
        Pose::from_world_to_camera(r_pitch_20, -(r_pitch_20.transform_vector(&center_20)));
    *ba.poses.get_mut(&30).unwrap() =
        Pose::from_world_to_camera(r_pitch_30, -(r_pitch_30.transform_vector(&center_30)));

    // Per-keyframe observations all report the level direction
    // (0, 9.81, 0), so each pose's residual depends on its own
    // perturbation — this confirms the Jacobian indexes per-observation
    // not per-pose.
    let mut prior = PerPoseGravityPrior::new(Vector3::new(0.0, 9.81, 0.0), 5.0);
    for (id, _pose) in &truth_poses {
        prior.push(PerPoseGravityObservation {
            keyframe_id: *id,
            g_camera_observed: Vector3::new(0.0, 9.81, 0.0),
            weight: 1.0,
        });
    }
    ba.set_per_pose_gravity_prior(prior);

    let config = BaConfig {
        max_iterations: 30,
        initial_lambda: Some(1.0e-2),
        ..BaConfig::default()
    };
    let _ = ba
        .optimize(&config)
        .expect("BA + per-pose gravity prior must run");

    let identity = UnitQuaternion::identity();
    let angle_err_20 = ba.poses[&20].world_to_camera.rotation.angle_to(&identity);
    let angle_err_30 = ba.poses[&30].world_to_camera.rotation.angle_to(&identity);
    assert!(
        angle_err_20 < 0.02,
        "per-pose prior must recover pose 20; angle_err={angle_err_20} rad"
    );
    assert!(
        angle_err_30 < 0.02,
        "per-pose prior must recover pose 30; angle_err={angle_err_30} rad"
    );
}

/// When the per-pose observation already matches the pose's pitched
/// rotation (e.g. accelerometer observation correctly reports the
/// nose-up gravity direction on a climbing vehicle), the prior must
/// add zero cost at that pose — so a non-level pose with a
/// consistent observation is left alone, while a level pose with a
/// pitched observation is pulled toward the observation. This is the
/// per-keyframe-observation independence property that motivates the
/// new type over the global [`GravityPrior`].
#[test]
fn per_pose_gravity_prior_respects_per_keyframe_observation_independence() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    // Pose 10: level. Pose 20: pitched by +0.10 rad. Each pose's
    // accelerometer-derived observation matches its actual rotation,
    // so the per-pose prior cost must be zero.
    let pitch = 0.10_f64;
    let r_pitch = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch);
    let center_20 = Vector3::new(0.5, 0.0, 0.0);
    ba.add_pose(10, pose_at(Vector3::new(0.0, 0.0, 0.0)));
    ba.add_pose(
        20,
        Pose::from_world_to_camera(r_pitch, -(r_pitch.transform_vector(&center_20))),
    );
    let g_world = Vector3::new(0.0, 9.81, 0.0);
    // Pose 20's observed gravity-in-camera-frame is R_wc · g_world.
    let g_cam_20 = r_pitch * g_world;
    let mut prior = PerPoseGravityPrior::new(g_world, 1.0);
    prior.push(PerPoseGravityObservation {
        keyframe_id: 10,
        g_camera_observed: g_world,
        weight: 1.0,
    });
    prior.push(PerPoseGravityObservation {
        keyframe_id: 20,
        g_camera_observed: g_cam_20,
        weight: 1.0,
    });
    ba.set_per_pose_gravity_prior(prior);
    let cost = ba.cost();
    assert!(
        cost < 1.0e-20,
        "per-pose consistent observations must give zero cost across mixed poses; got {cost}"
    );
}

/// Per-observation `weight` scales each observation's contribution to
/// the prior's cost. Two identical pitched-pose setups: one uses the
/// per-obs weight, the other folds the equivalent scale into the
/// global prior weight. The cost values must match exactly, and a
/// per-obs `weight = 0.0` must mute that observation entirely.
#[test]
fn per_pose_gravity_prior_per_obs_weight_scales_cost() {
    let camera = pinhole();
    // Setup A: per-obs weight = 4.0, global = 1.0  → effective 4.0
    let mut ba_a = BundleAdjustment::new(camera.clone());
    let pose_id = 10u64;
    let pitch = 0.10_f64;
    let r_pitch = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch);
    let center = Vector3::new(0.0, 0.0, 0.0);
    let pose = Pose::from_world_to_camera(r_pitch, -(r_pitch.transform_vector(&center)));
    ba_a.add_pose(pose_id, pose.clone());
    let g_world = Vector3::new(0.0, 9.81, 0.0);
    let mut prior_a = PerPoseGravityPrior::new(g_world, 1.0);
    prior_a.push(PerPoseGravityObservation {
        keyframe_id: pose_id,
        g_camera_observed: g_world, // observation = level, pose tilted → nonzero residual
        weight: 4.0,
    });
    ba_a.set_per_pose_gravity_prior(prior_a);
    let cost_a = ba_a.cost();

    // Setup B: per-obs weight = 1.0, global = 4.0  → effective 4.0
    let mut ba_b = BundleAdjustment::new(camera.clone());
    ba_b.add_pose(pose_id, pose.clone());
    let mut prior_b = PerPoseGravityPrior::new(g_world, 4.0);
    prior_b.push(PerPoseGravityObservation {
        keyframe_id: pose_id,
        g_camera_observed: g_world,
        weight: 1.0,
    });
    ba_b.set_per_pose_gravity_prior(prior_b);
    let cost_b = ba_b.cost();

    assert!(
        (cost_a - cost_b).abs() < 1.0e-18,
        "per-obs weight 4.0 must match global weight 4.0; got A={cost_a}, B={cost_b}"
    );

    // Setup C: per-obs weight = 0.0 must mute the observation regardless
    // of pose tilt.
    let mut ba_c = BundleAdjustment::new(camera);
    ba_c.add_pose(pose_id, pose);
    let mut prior_c = PerPoseGravityPrior::new(g_world, 100.0);
    prior_c.push(PerPoseGravityObservation {
        keyframe_id: pose_id,
        g_camera_observed: g_world,
        weight: 0.0,
    });
    ba_c.set_per_pose_gravity_prior(prior_c);
    let cost_c = ba_c.cost();
    assert!(
        cost_c < 1.0e-20,
        "per-obs weight 0.0 must mute the observation; got {cost_c}"
    );
}

/// A position prior on every pose with the truth target should add
/// zero cost when the trajectory is already correct. Validates the
/// residual computation and the axis-weight collapse-to-zero path.
#[test]
fn position_prior_zero_cost_on_truth_trajectory() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.0, 0.0))),
    ];
    for (id, pose) in &truth_poses {
        ba.add_pose(*id, pose.clone());
    }
    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
    }
    let mut prior = PositionPrior::new();
    for (id, pose) in &truth_poses {
        let c = pose.camera_center_world();
        prior.push(PositionPriorObservation {
            keyframe_id: *id,
            camera_center_world: c,
            axis_weights: Vector3::new(0.0, 1.0, 0.0),
        });
    }
    ba.set_position_prior(prior);
    let cost = ba.cost();
    assert!(
        cost < 1.0e-20,
        "position prior + truth must give zero cost, got {cost}"
    );
}

/// Inject a pure vertical translation drift (the seq08 failure mode:
/// rotation correct, y-translation biased) on the non-anchor poses,
/// and check that an altitude-only position prior + BA pulls the
/// y-coordinate back to truth. This is the synthetic equivalent of
/// "would a GNSS altitude prior fix seq08?" — yes, when the prior is
/// trustworthy.
#[test]
fn position_prior_corrects_pure_vertical_translation_drift() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    // Truth: forward translation, no vertical motion.
    let truth_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.0, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.0, 0.0))),
    ];
    // Drift: each non-anchor pose has its world y-coordinate shifted
    // upward by a constant. KITTI cameras use y-down, so +y in world
    // means the camera CENTRE moved down — equivalent in shape to
    // seq08's drifting y-bias.
    let drifted_poses = [
        (10u64, pose_at(Vector3::new(0.0, 0.0, 0.0))),
        (20u64, pose_at(Vector3::new(0.5, 0.18, 0.0))),
        (30u64, pose_at(Vector3::new(1.0, 0.36, 0.0))),
    ];
    for (id, pose) in &drifted_poses {
        ba.add_pose(*id, pose.clone());
    }
    ba.fix_pose(10);

    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
    }
    // Measurements come from TRUTH poses (so the reprojection part
    // pulls towards truth too, but on a small bundle reprojection
    // residuals alone may not converge to truth fast under LM with
    // λ damping — the prior makes it deterministic).
    for (pose_id, pose) in &truth_poses {
        for (lm_id, point) in world_grid() {
            let xc = pose.transform_world_point(&point);
            let predicted = camera.project(&xc).unwrap();
            ba.add_observation(BaObservation {
                keyframe_id: *pose_id,
                landmark_id: lm_id,
                xy: predicted,
            });
        }
    }

    let mut prior = PositionPrior::new();
    for (id, pose) in &truth_poses {
        let c = pose.camera_center_world();
        prior.push(PositionPriorObservation {
            keyframe_id: *id,
            camera_center_world: c,
            // Altitude-only: don't constrain x/z, only y. Mimics a
            // GNSS altimeter that gives vertical position but no
            // horizontal accuracy.
            axis_weights: Vector3::new(0.0, 50.0, 0.0),
        });
    }
    ba.set_position_prior(prior);

    let config = BaConfig {
        max_iterations: 30,
        initial_lambda: Some(1.0e-3),
        ..BaConfig::default()
    };
    let result = ba
        .optimize(&config)
        .expect("BA + position prior must succeed");
    assert!(result.final_cost < result.initial_cost);

    let recovered_20 = ba.poses[&20].camera_center_world();
    let recovered_30 = ba.poses[&30].camera_center_world();
    assert!(
        recovered_20.y.abs() < 5.0e-3,
        "altitude prior must pull KF20 y back to truth; got y={}",
        recovered_20.y
    );
    assert!(
        recovered_30.y.abs() < 5.0e-3,
        "altitude prior must pull KF30 y back to truth; got y={}",
        recovered_30.y
    );
}

/// A zero-weight axis must contribute nothing to either cost or
/// Jacobian. This pins the axis-mask semantics down so users can
/// confidently mix "anchor only y" with "anchor only x/z" priors.
#[test]
fn position_prior_zero_weight_axis_has_no_effect() {
    let mut ba = truth_bundle();
    ba.fix_pose(10);
    // Drift pose 30 in pure x. Add an altitude-only prior; that prior
    // should NOT pull pose 30 back in x at all.
    let drifted_30 = pose_at(Vector3::new(1.20, 0.0, 0.0));
    ba.poses.insert(30, drifted_30);
    let cost_no_prior = ba.cost();

    let mut prior = PositionPrior::new();
    prior.push(PositionPriorObservation {
        keyframe_id: 30,
        camera_center_world: Point3::new(1.0, 0.0, 0.0),
        axis_weights: Vector3::new(0.0, 1.0, 0.0), // y only
    });
    ba.set_position_prior(prior);
    let cost_with_prior = ba.cost();
    let delta = (cost_with_prior - cost_no_prior).abs();
    assert!(
        delta < 1.0e-9,
        "zero-weight x/z axes must add no cost for pure-x drift; delta={delta}"
    );
}

/// At convergence (poses match measurement), the pairwise pose
/// factor must contribute zero cost.
#[test]
fn pairwise_pose_factor_zero_cost_at_truth() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera);
    let pose_i = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let pose_j = pose_at(Vector3::new(0.5, 0.0, 0.2));
    ba.add_pose(10, pose_i.clone());
    ba.add_pose(20, pose_j.clone());
    // Measurement = actual relative pose (T_j · T_iⁱ).
    let measurement = Pose {
        world_to_camera: pose_j
            .world_to_camera
            .compose(&pose_i.world_to_camera.inverse()),
    };
    ba.add_pairwise_pose_factor(PairwisePoseFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        measurement,
        weight: 100.0,
    });
    let cost = ba.cost();
    assert!(
        cost < 1.0e-18,
        "pairwise factor at truth must give zero cost, got {cost}"
    );
}

/// Drag pose 30 in pure translation; an IMU-style pairwise factor
/// between pose 20 (fixed at truth) and pose 30 (drifted) should pull
/// pose 30 back to truth on a small bundle even without reprojection
/// observations.
#[test]
fn pairwise_pose_factor_corrects_translation_drift_solo() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera);
    let pose_20 = pose_at(Vector3::new(0.5, 0.0, 0.0));
    let truth_30 = pose_at(Vector3::new(1.0, 0.0, 0.4));
    let drifted_30 = pose_at(Vector3::new(1.07, 0.02, 0.35));
    ba.add_pose(20, pose_20.clone());
    ba.add_pose(30, drifted_30);
    ba.fix_pose(20);
    // Measurement = ground-truth relative pose (T_truth_30 · T_20ⁱ).
    let measurement = Pose {
        world_to_camera: truth_30
            .world_to_camera
            .compose(&pose_20.world_to_camera.inverse()),
    };
    ba.add_pairwise_pose_factor(PairwisePoseFactor {
        keyframe_id_from: 20,
        keyframe_id_to: 30,
        measurement,
        weight: 1.0e3,
    });
    // Add a single landmark + observation so BA has a landmark column;
    // the factor's translation pull must dominate.
    ba.add_landmark(1, Point3::new(0.0, 0.0, 5.0));
    let xc = pose_20.transform_world_point(&Point3::new(0.0, 0.0, 5.0));
    let uv = ba.camera.project(&xc).unwrap();
    ba.add_observation(BaObservation {
        keyframe_id: 20,
        landmark_id: 1,
        xy: uv,
    });
    let config = BaConfig {
        max_iterations: 30,
        initial_lambda: Some(1.0e-3),
        ..BaConfig::default()
    };
    let _ = ba
        .optimize(&config)
        .expect("BA + pairwise pose factor must run");
    let recovered = ba.poses[&30].camera_center_world();
    let expected = Point3::new(1.0, 0.0, 0.4);
    assert!(
        (recovered - expected).norm() < 5.0e-3,
        "pairwise factor must pull KF30 back to truth: got {recovered:?}"
    );
}

/// Drag pose 30 in yaw; the pairwise pose factor includes a rotation
/// component in its 6-vector residual, so it should also pull the
/// rotation back to truth (not just translation). This exercises the
/// `[ω]` half of the Jacobian.
#[test]
fn pairwise_pose_factor_corrects_rotation_drift() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera);
    let pose_20 = pose_with_yaw(Vector3::new(0.5, 0.0, 0.0), 0.0);
    let truth_30 = pose_with_yaw(Vector3::new(1.0, 0.0, 0.4), 0.10);
    let drifted_30 = pose_with_yaw(Vector3::new(1.0, 0.0, 0.4), 0.18);
    ba.add_pose(20, pose_20.clone());
    ba.add_pose(30, drifted_30);
    ba.fix_pose(20);
    let measurement = Pose {
        world_to_camera: truth_30
            .world_to_camera
            .compose(&pose_20.world_to_camera.inverse()),
    };
    ba.add_pairwise_pose_factor(PairwisePoseFactor {
        keyframe_id_from: 20,
        keyframe_id_to: 30,
        measurement,
        weight: 1.0e3,
    });
    ba.add_landmark(1, Point3::new(0.0, 0.0, 5.0));
    let xc = pose_20.transform_world_point(&Point3::new(0.0, 0.0, 5.0));
    let uv = ba.camera.project(&xc).unwrap();
    ba.add_observation(BaObservation {
        keyframe_id: 20,
        landmark_id: 1,
        xy: uv,
    });
    let config = BaConfig {
        max_iterations: 30,
        initial_lambda: Some(1.0e-3),
        ..BaConfig::default()
    };
    let _ = ba
        .optimize(&config)
        .expect("BA + pairwise factor must run on yaw drift");
    // Truth yaw is 0.10 rad; we drifted to 0.18 → residual ~0.08 rad.
    // After convergence, recovered angle should be near 0.10 rad.
    let recovered_axis = ba.poses[&30].world_to_camera.rotation.scaled_axis();
    // Yaw stored on the y-axis (Vector3::y_axis).
    let yaw_err = (recovered_axis.y - 0.10).abs();
    assert!(
        yaw_err < 2.0e-2,
        "pairwise factor must pull KF30 yaw back to truth; got {recovered_axis:?}"
    );
}

/// `LoopClosureConstraint::to_pairwise_pose_factor` should preserve all
/// identifying fields verbatim — ids and the SE(3) measurement — and only
/// stamp the supplied weight. Verifier metadata is intentionally dropped
/// because BA only consumes the geometric quantities.
#[test]
fn loop_closure_constraint_lifts_to_pairwise_pose_factor() {
    let kf_from = 10u64;
    let kf_to = 27u64;
    let pose_from = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let pose_to = pose_with_yaw(Vector3::new(2.5, 0.0, 0.7), 0.12);
    let relative = pose_to
        .world_to_camera
        .compose(&pose_from.world_to_camera.inverse());
    let constraint = LoopClosureConstraint {
        from_keyframe_id: kf_from,
        to_keyframe_id: kf_to,
        relative_pose: relative.clone(),
        inlier_count: 42,
        inlier_ratio: 0.9,
        mean_sampson_error: 1.2e-4,
        score: 0.75,
    };

    let factor = constraint.to_pairwise_pose_factor(123.0);
    assert_eq!(factor.keyframe_id_from, kf_from);
    assert_eq!(factor.keyframe_id_to, kf_to);
    assert!((factor.weight - 123.0).abs() < 1.0e-12);
    assert!(
        (factor.measurement.world_to_camera.translation - relative.translation).norm() < 1.0e-12
    );
    let dq = factor
        .measurement
        .world_to_camera
        .rotation
        .rotation_to(&relative.rotation);
    assert!(dq.angle() < 1.0e-9);

    let factors = pairwise_pose_factors_from_loop_closures(&[constraint], 50.0);
    assert_eq!(factors.len(), 1);
    assert!((factors[0].weight - 50.0).abs() < 1.0e-12);
}

/// End-to-end smoke for the loop-closure → BA path: build a small bundle with
/// reprojection observations between KF10 and KF20 (anchored), then drag KF40
/// by a translation error analogous to accumulated VO drift. Feed the truth
/// relative pose (KF10 → KF40) as a verified `LoopClosureConstraint`, lift it
/// to a `PairwisePoseFactor`, optimise, and assert KF40 is pulled back.
#[test]
fn loop_closure_pairwise_factor_corrects_drift_in_ba() {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let pose_10 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let pose_20 = pose_at(Vector3::new(0.5, 0.0, 0.0));
    let truth_40 = pose_at(Vector3::new(1.5, 0.0, 0.0));
    let drift_40 = pose_at(Vector3::new(1.65, 0.05, 0.08));

    ba.add_pose(10, pose_10.clone());
    ba.add_pose(20, pose_20.clone());
    ba.add_pose(40, drift_40);
    ba.fix_pose(10);
    ba.fix_pose(20);

    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
        for kf_id in [10u64, 20u64] {
            let kf_pose = if kf_id == 10 { &pose_10 } else { &pose_20 };
            let xc = kf_pose.transform_world_point(&point);
            let uv = camera.project(&xc).expect("in front");
            ba.add_observation(BaObservation {
                keyframe_id: kf_id,
                landmark_id: id,
                xy: uv,
            });
        }
    }

    let relative = truth_40
        .world_to_camera
        .compose(&pose_10.world_to_camera.inverse());
    let constraint = LoopClosureConstraint {
        from_keyframe_id: 10,
        to_keyframe_id: 40,
        relative_pose: relative,
        inlier_count: 80,
        inlier_ratio: 0.95,
        mean_sampson_error: 1.0e-4,
        score: 1.0,
    };
    ba.add_pairwise_pose_factor(constraint.to_pairwise_pose_factor(1.0e3));

    let config = BaConfig {
        max_iterations: 30,
        initial_lambda: Some(1.0e-3),
        ..BaConfig::default()
    };
    ba.optimize(&config)
        .expect("BA + loop-closure factor must converge");

    let recovered = ba.poses[&40].camera_center_world();
    let expected = Point3::new(1.5, 0.0, 0.0);
    let before = Point3::new(1.65, 0.05, 0.08);
    let drift_before = (before - expected).norm();
    let drift_after = (recovered - expected).norm();
    assert!(
        drift_after < 0.1 * drift_before,
        "loop-closure pairwise factor must shrink KF40 drift to <10% of input; \
         before={drift_before:.4}, after={drift_after:.4}, recovered={recovered:?}"
    );
}

/// Stack every BA-side factor — reprojection, `GravityPrior`,
/// `PositionPrior` (altitude only), and `PairwisePoseFactor` (loop-closure
/// shape) — on a single bundle. Each non-anchor keyframe is injected with a
/// DIFFERENT failure mode that only one of the three priors can fix:
///
/// - KF20 has a +0.12 rad pitch drift → only `GravityPrior` corrects it.
/// - KF30 has a +0.20 m vertical translation drift → only `PositionPrior` corrects it.
/// - KF40 has a yaw + horizontal-translation drift → only the verified
///   loop-closure `PairwisePoseFactor` (KF10 → KF40 truth) corrects it.
///
/// Reprojection observations come from TRUTH poses but on a small bundle
/// they can converge to spurious local minima — the three priors disambiguate
/// each axis of drift independently. This is the integration smoke for the
/// task #24 factor stack: gravity (rotation) + position (translation) +
/// pairwise (relative SE(3)) all converging in a single LM solve without
/// interference.
#[test]
fn stacked_factors_correct_independent_drift_modes() {
    use nalgebra::UnitQuaternion;
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let pose_10 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let truth_20 = pose_at(Vector3::new(0.5, 0.0, 0.0));
    let truth_30 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    let truth_40 = pose_at(Vector3::new(1.5, 0.0, 0.0));

    // KF20: pitch drift (rotation around camera x-axis).
    let pitch = 0.12_f64;
    let r20 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch);
    let drift_20 =
        Pose::from_world_to_camera(r20, -r20.transform_vector(&Vector3::new(0.5, 0.0, 0.0)));
    // KF30: pure vertical drift in world (camera centre +y_world).
    let drift_30 = pose_at(Vector3::new(1.0, 0.20, 0.0));
    // KF40: yaw + lateral translation drift.
    let drift_40 = pose_with_yaw(Vector3::new(1.55, 0.0, 0.08), 0.10);
    let drift_40_center = drift_40.camera_center_world();

    ba.add_pose(10, pose_10.clone());
    ba.add_pose(20, drift_20);
    ba.add_pose(30, drift_30);
    ba.add_pose(40, drift_40);
    ba.fix_pose(10);

    let truth = [
        (10u64, &pose_10),
        (20u64, &truth_20),
        (30u64, &truth_30),
        (40u64, &truth_40),
    ];
    for (id, point) in world_grid() {
        ba.add_landmark(id, point);
        for (kf_id, pose) in &truth {
            let xc = pose.transform_world_point(&point);
            let uv = camera.project(&xc).expect("in front");
            ba.add_observation(BaObservation {
                keyframe_id: *kf_id,
                landmark_id: id,
                xy: uv,
            });
        }
    }

    // Gravity prior — level world, identity-at-anchor camera-frame gravity.
    ba.set_gravity_prior(GravityPrior {
        g_world: Vector3::new(0.0, 9.81, 0.0),
        g_camera_observed: Vector3::new(0.0, 9.81, 0.0),
        weight: 5.0,
    });

    // Position prior — altitude-only on KF30 (the y-drifted keyframe).
    let mut pos_prior = PositionPrior::new();
    pos_prior.push(PositionPriorObservation {
        keyframe_id: 30,
        camera_center_world: truth_30.camera_center_world(),
        axis_weights: Vector3::new(0.0, 50.0, 0.0),
    });
    ba.set_position_prior(pos_prior);

    // Loop-closure pairwise factor — KF10 → KF40 truth relative pose.
    let relative_10_40 = truth_40
        .world_to_camera
        .compose(&pose_10.world_to_camera.inverse());
    let loop_constraint = LoopClosureConstraint {
        from_keyframe_id: 10,
        to_keyframe_id: 40,
        relative_pose: relative_10_40,
        inlier_count: 60,
        inlier_ratio: 0.9,
        mean_sampson_error: 1.0e-4,
        score: 1.0,
    };
    ba.add_pairwise_pose_factor(loop_constraint.to_pairwise_pose_factor(1.0e3));

    let config = BaConfig {
        max_iterations: 40,
        initial_lambda: Some(1.0e-3),
        ..BaConfig::default()
    };
    let result = ba
        .optimize(&config)
        .expect("stacked-factor BA must converge");
    assert!(result.final_cost < result.initial_cost);

    // KF20 rotation should be back to identity (gravity prior).
    let angle_err_20 = ba.poses[&20]
        .world_to_camera
        .rotation
        .angle_to(&UnitQuaternion::identity());
    assert!(
        angle_err_20 < 0.03,
        "gravity prior must pull KF20 rotation back; angle_err={angle_err_20:.4} rad"
    );

    // KF30 y-coordinate should be back to 0 (position prior).
    let c30 = ba.poses[&30].camera_center_world();
    assert!(
        c30.y.abs() < 1.0e-2,
        "position prior must pull KF30 y back to 0; got y={:.4}",
        c30.y
    );

    // KF40 should be back near truth in translation (loop-closure pairwise).
    let c40 = ba.poses[&40].camera_center_world();
    let drift_before = (drift_40_center - truth_40.camera_center_world()).norm();
    let drift_after = (c40 - truth_40.camera_center_world()).norm();
    assert!(
        drift_after < 0.2 * drift_before,
        "loop-closure factor must shrink KF40 drift; before={drift_before:.4}, after={drift_after:.4}"
    );
}

/// Build a pre-integrated delta for a 1-second window with constant
/// body-frame acceleration along world +x and zero rotation; matches
/// the `constant_linear_acceleration_matches_closed_form` scenario.
fn imu_constant_accel_delta(
    accel: Vector3<f64>,
    dt_total: f64,
    dt_step: f64,
) -> visloc_slam::ImuPreintegratedDelta {
    let mut pre = ImuPreintegrator::new();
    let steps = (dt_total / dt_step).round() as usize;
    for _ in 0..steps {
        pre.integrate_sample(Vector3::zeros(), accel, dt_step);
    }
    pre.delta()
}

#[test]
fn imu_factor_zero_cost_at_consistent_state() {
    // Two keyframes 1 s apart with constant body-frame +x acceleration
    // a = 2 m/s². Truth state:
    //   pose 0 at origin, v_0 = 0
    //   pose 1 at (1, 0, 0), v_1 = (2, 0, 0)
    // The IMU pre-integrated (Δv, Δp) match this kinematics exactly,
    // so adding the factor to the BA must not increase the cost over
    // the visual-only baseline.
    let mut ba = truth_bundle();
    let v0 = Vector3::zeros();
    let v1 = Vector3::new(2.0, 0.0, 0.0);
    let delta = imu_constant_accel_delta(Vector3::new(2.0, 0.0, 0.0), 1.0, 0.001);
    // truth_bundle has poses 10/20/30 at (0,0,0)/(0.5,0,0)/(1,0,0).
    // Re-anchor pose 30 to match the 1-s integration (cam centre at +1m).
    // The factor binds 10 → 30.
    ba.add_velocity(10, v0);
    ba.add_velocity(30, v1);
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 30,
        delta,
        gravity_world: Vector3::zeros(),
        weight_position: 1.0,
        weight_velocity: 1.0,
        weight_rotation: 1.0,
    });

    let cost_with_imu = ba.cost();
    // Baseline cost without the IMU factor.
    let ba_visual = truth_bundle();
    let cost_visual = ba_visual.cost();
    // Truth scene is zero-residual for vision, and IMU factor at truth
    // is also zero-residual, so the joint cost must remain ≈ 0.
    assert!(
        cost_with_imu < 1e-6,
        "IMU+visual cost at truth = {cost_with_imu}, visual-only = {cost_visual}"
    );
}

#[test]
fn imu_factor_uses_camera_to_body_extrinsic() {
    let camera = pinhole();
    let body_to_camera = SE3::new(
        UnitQuaternion::from_euler_angles(0.2, -0.3, 0.4),
        Vector3::new(0.11, -0.07, 0.05),
    );
    let body_pose_0 = SE3::new(UnitQuaternion::identity(), Vector3::zeros());
    let body_pose_1 = SE3::new(UnitQuaternion::identity(), Vector3::new(-1.0, 0.0, 0.0));
    let camera_pose_0 = body_to_camera.inverse().compose(&body_pose_0);
    let camera_pose_1 = body_to_camera.inverse().compose(&body_pose_1);
    let pose_0 = Pose::from_world_to_camera(camera_pose_0.rotation, camera_pose_0.translation);
    let pose_1 = Pose::from_world_to_camera(camera_pose_1.rotation, camera_pose_1.translation);
    let factor = ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        delta: imu_constant_accel_delta(Vector3::new(2.0, 0.0, 0.0), 1.0, 0.001),
        gravity_world: Vector3::zeros(),
        weight_position: 1.0,
        weight_velocity: 1.0,
        weight_rotation: 1.0,
    };

    let build = |with_extrinsic: bool| {
        let mut ba = BundleAdjustment::new(camera.clone());
        ba.add_pose(10, pose_0.clone());
        ba.add_pose(20, pose_1.clone());
        ba.add_velocity(10, Vector3::zeros());
        ba.add_velocity(20, Vector3::new(2.0, 0.0, 0.0));
        ba.add_imu_factor(factor.clone());
        if with_extrinsic {
            ba.set_imu_body_to_camera(body_to_camera.clone());
        }
        ba
    };

    let calibrated_cost = build(true).cost();
    let identity_rig_cost = build(false).cost();
    assert!(
        calibrated_cost < 1.0e-6,
        "calibrated IMU residual must vanish, cost={calibrated_cost}"
    );
    assert!(
        identity_rig_cost > 1.0e-2,
        "non-identity rig must expose the old frame error, cost={identity_rig_cost}"
    );
}

#[test]
fn imu_pose_jacobian_respects_camera_to_body_extrinsic() {
    let body_to_camera = SE3::new(
        UnitQuaternion::from_euler_angles(0.2, -0.3, 0.4),
        Vector3::new(0.11, -0.07, 0.05),
    );
    let body_pose_0 = SE3::identity();
    let body_pose_1 = SE3::new(UnitQuaternion::identity(), Vector3::new(-1.0, 0.0, 0.0));
    let camera_pose_0 = body_to_camera.inverse().compose(&body_pose_0);
    let camera_pose_1 = body_to_camera.inverse().compose(&body_pose_1);
    let mut drifted_camera_pose_1 = camera_pose_1.clone();
    drifted_camera_pose_1.translation += Vector3::new(0.25, -0.12, 0.08);

    let mut ba = BundleAdjustment::new(pinhole());
    ba.set_imu_body_to_camera(body_to_camera.clone());
    ba.add_pose(
        10,
        Pose::from_world_to_camera(camera_pose_0.rotation, camera_pose_0.translation),
    );
    ba.add_pose(
        20,
        Pose::from_world_to_camera(
            drifted_camera_pose_1.rotation,
            drifted_camera_pose_1.translation,
        ),
    );
    ba.fix_pose(10);
    ba.add_velocity(10, Vector3::zeros());
    ba.add_velocity(20, Vector3::new(2.0, 0.0, 0.0));
    ba.fix_velocity(10);
    ba.fix_velocity(20);
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        delta: imu_constant_accel_delta(Vector3::new(2.0, 0.0, 0.0), 1.0, 0.001),
        gravity_world: Vector3::zeros(),
        weight_position: 100.0,
        weight_velocity: 1.0,
        weight_rotation: 100.0,
    });

    let initial_cost = ba.cost();
    let result = ba.optimize(&BaConfig::default()).expect("IMU BA");
    let refined_camera = &ba.poses.get(&20).expect("refined pose").world_to_camera;
    let refined_body = body_to_camera.compose(refined_camera);
    let refined_body_center = refined_body.inverse().translation;
    assert!(result.final_cost < initial_cost * 1.0e-4);
    assert!(
        (refined_body_center - Vector3::new(1.0, 0.0, 0.0)).norm() < 1.0e-4,
        "refined body center={refined_body_center:?}"
    );
}

#[test]
fn imu_factor_pulls_drifted_velocity_back_to_truth() {
    // Truth scenario: constant +x acceleration of 2 m/s² for 1 s, two
    // keyframes 1 m apart. Fix both poses and v_1; only v_0 is free,
    // and it starts off at a wrong value (1 m/s instead of 0). The
    // single IMU factor must pull it back close to zero.
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let pose_0 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let pose_1 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    ba.add_pose(10, pose_0.clone());
    ba.add_pose(20, pose_1.clone());
    ba.fix_pose(10);
    ba.fix_pose(20);

    // Add at least one observation so the BA passes the no-data check.
    // It's against a single fixed landmark so it contributes zero cost
    // / Jacobian and doesn't interfere with the IMU-only velocity solve.
    let landmark_pos = Point3::new(0.0, 0.0, 10.0);
    ba.add_landmark(1, landmark_pos);
    ba.fix_landmark(1);
    let xc0 = pose_0.transform_world_point(&landmark_pos);
    let uv0 = camera.project(&xc0).unwrap();
    ba.add_observation(BaObservation {
        keyframe_id: 10,
        landmark_id: 1,
        xy: uv0,
    });

    // Truth velocities for the constant +x accel scenario.
    let v0_truth = Vector3::<f64>::zeros();
    let v1_truth = Vector3::new(2.0, 0.0, 0.0);

    // Inject a 1 m/s initial-velocity drift; v_1 is fixed at truth so
    // there is exactly one anchor pulling v_0 back.
    let v0_drifted = v0_truth + Vector3::new(1.0, 0.0, 0.0);
    ba.add_velocity(10, v0_drifted);
    ba.add_velocity(20, v1_truth);
    ba.fix_velocity(20);

    let delta = imu_constant_accel_delta(Vector3::new(2.0, 0.0, 0.0), 1.0, 0.001);
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        delta,
        gravity_world: Vector3::zeros(),
        weight_position: 1.0,
        weight_velocity: 1.0,
        weight_rotation: 1.0,
    });

    let config = BaConfig {
        max_iterations: 25,
        ..BaConfig::default()
    };
    let result = ba.optimize(&config).expect("BA converges");
    let recovered = ba.velocities[&10];
    let err = (recovered - v0_truth).norm();
    assert!(
        err < 1e-4,
        "v_0 = {:?}, truth = {:?}, err = {}, converged = {}",
        recovered,
        v0_truth,
        err,
        result.converged
    );
}

#[test]
fn imu_factor_pulls_drifted_pose_with_visual_anchor() {
    // 3-keyframe scene with constant velocity (zero accel, zero
    // gravity). The middle keyframe gets a small lateral drift
    // injected into its world camera centre; visual observations
    // alone struggle when the scene depth is uniform, so the IMU
    // factor (which knows the relative position must be (1, 0, 0)
    // per second) helps pull the lateral drift back to zero.
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let truth_centres = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(2.0, 0.0, 0.0),
    ];
    let truth_poses: Vec<Pose> = truth_centres.iter().map(|c| pose_at(*c)).collect();
    let ids = [10u64, 20, 30];
    for (id, pose) in ids.iter().zip(truth_poses.iter()) {
        ba.add_pose(*id, pose.clone());
    }
    // Anchor first and last keyframes; middle is the variable.
    ba.fix_pose(10);
    ba.fix_pose(30);

    // Landmarks + observations for visual support.
    let points = world_grid();
    for (id, point) in &points {
        ba.add_landmark(*id, *point);
        ba.fix_landmark(*id);
    }
    for (kf_id, pose) in ids.iter().zip(truth_poses.iter()) {
        for (lm_id, point) in &points {
            let xc = pose.transform_world_point(point);
            let uv = camera.project(&xc).expect("point in front of camera");
            ba.add_observation(BaObservation {
                keyframe_id: *kf_id,
                landmark_id: *lm_id,
                xy: uv,
            });
        }
    }

    // Inject lateral drift on the middle keyframe's pose. Visual
    // alone may pull this back, so we want a tight drift comparison
    // between with-IMU and without-IMU.
    let drift = Vector3::new(0.0, 0.0, 0.0); // start at truth — see below
    let drifted_pose = pose_at(truth_centres[1] + drift + Vector3::new(0.2, 0.0, 0.0));
    *ba.poses.get_mut(&20).unwrap() = drifted_pose;

    // Velocities: constant 1 m/s along +x; IMU integrates zero accel.
    let v = Vector3::new(1.0, 0.0, 0.0);
    for &id in &ids {
        ba.add_velocity(id, v);
        ba.fix_velocity(id); // We only want the IMU factor to constrain pose 20.
    }
    let delta_01 = imu_constant_accel_delta(Vector3::zeros(), 1.0, 0.01);
    let delta_12 = imu_constant_accel_delta(Vector3::zeros(), 1.0, 0.01);
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        delta: delta_01,
        gravity_world: Vector3::zeros(),
        weight_position: 1e3,
        weight_velocity: 1e3,
        weight_rotation: 1.0,
    });
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 20,
        keyframe_id_to: 30,
        delta: delta_12,
        gravity_world: Vector3::zeros(),
        weight_position: 1e3,
        weight_velocity: 1e3,
        weight_rotation: 1.0,
    });

    let config = BaConfig {
        max_iterations: 30,
        ..BaConfig::default()
    };
    ba.optimize(&config).expect("BA converges");

    let recovered = ba.poses[&20].camera_center_world();
    let truth_center = Point3::from(truth_centres[1]);
    let err = (recovered.coords - truth_center.coords).norm();
    assert!(
        err < 1e-2,
        "middle-keyframe centre = {:?}, truth = {:?}, err = {}",
        recovered,
        truth_center,
        err
    );
}

#[test]
fn imu_bias_zero_cost_at_truth() {
    // Same scenario as `imu_factor_zero_cost_at_consistent_state` but
    // with a registered (zero) bias slot on keyframe 10. The bias-
    // correction code path must be a no-op at `bias = linearisation`,
    // so the joint cost stays at zero.
    let mut ba = truth_bundle();
    let v0 = Vector3::zeros();
    let v1 = Vector3::new(2.0, 0.0, 0.0);
    let delta = imu_constant_accel_delta(Vector3::new(2.0, 0.0, 0.0), 1.0, 0.001);
    ba.add_velocity(10, v0);
    ba.add_velocity(30, v1);
    ba.add_bias(10, Vector6::zeros());
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 30,
        delta,
        gravity_world: Vector3::zeros(),
        weight_position: 1.0,
        weight_velocity: 1.0,
        weight_rotation: 1.0,
    });
    let cost = ba.cost();
    assert!(cost < 1e-6, "IMU+visual cost with zero bias = {cost}");
}

#[test]
fn imu_bias_recovers_hidden_accel_bias() {
    // Truth motion: constant +x accel of 2 m/s² for 1 s, so pose_1 is
    // 1 m forward of pose_0 and v_1 = (2, 0, 0). The IMU samples were
    // recorded by a sensor with a hidden +0.5 m/s² accel bias, and the
    // integrator was told to assume zero bias — so the pre-integrated
    // (Δv, Δp) come out at (2.5, 0, 0) and (1.25, 0, 0), inconsistent
    // with the truth poses/velocities.
    //
    // With every pose and velocity fixed at truth, the only remaining
    // DoF is the bias state on keyframe 10. The BA must recover
    // bias_a ≈ (0.5, 0, 0) so the first-order corrected delta lines up
    // with the truth motion.
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let pose_0 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let pose_1 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    ba.add_pose(10, pose_0.clone());
    ba.add_pose(20, pose_1.clone());
    ba.fix_pose(10);
    ba.fix_pose(20);

    // Single fixed landmark so the BA passes its no-observations guard.
    let landmark_pos = Point3::new(0.0, 0.0, 10.0);
    ba.add_landmark(1, landmark_pos);
    ba.fix_landmark(1);
    let xc0 = pose_0.transform_world_point(&landmark_pos);
    let uv0 = camera.project(&xc0).unwrap();
    ba.add_observation(BaObservation {
        keyframe_id: 10,
        landmark_id: 1,
        xy: uv0,
    });

    let v0_truth = Vector3::<f64>::zeros();
    let v1_truth = Vector3::new(2.0, 0.0, 0.0);
    ba.add_velocity(10, v0_truth);
    ba.add_velocity(20, v1_truth);
    ba.fix_velocity(10);
    ba.fix_velocity(20);

    // Integration uses bias linearisation = 0 but the raw samples carry
    // a +0.5 m/s² hidden offset.
    let raw_accel = Vector3::new(2.5, 0.0, 0.0);
    let b_truth_a = Vector3::new(0.5, 0.0, 0.0);
    let delta = imu_constant_accel_delta(raw_accel, 1.0, 0.001);

    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        delta,
        gravity_world: Vector3::zeros(),
        weight_position: 1.0,
        weight_velocity: 1.0,
        weight_rotation: 1.0,
    });

    // Initial bias guess: zero. The optimisation must move it toward
    // (0, 0, 0, 0.5, 0, 0) — i.e. recover the hidden accel bias.
    ba.add_bias(10, Vector6::zeros());

    let config = BaConfig {
        max_iterations: 25,
        ..BaConfig::default()
    };
    ba.optimize(&config).expect("BA converges");

    let recovered = ba.biases[&10];
    let recovered_bg: Vector3<f64> = recovered.fixed_rows::<3>(0).into_owned();
    let recovered_ba: Vector3<f64> = recovered.fixed_rows::<3>(3).into_owned();
    assert!(
        recovered_bg.norm() < 1e-3,
        "gyro bias should stay near zero (no rotation in scene): got {:?}",
        recovered_bg
    );
    assert!(
        (recovered_ba - b_truth_a).norm() < 1e-3,
        "accel bias should approach truth {:?}, got {:?}",
        b_truth_a,
        recovered_ba
    );
}

#[test]
fn imu_bias_fixed_acts_as_correction_only() {
    // When the bias slot is registered but FIXED at its initial value,
    // the residual must include the bias-correction term (so the
    // factor sees the right linearisation) but no bias DoF should
    // appear in the linear system. Verify by setting bias[10] = the
    // hidden truth bias up-front; with that, the residual should
    // vanish even though the integrator's delta was computed against
    // the wrong linearisation point.
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let pose_0 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let pose_1 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    ba.add_pose(10, pose_0.clone());
    ba.add_pose(20, pose_1.clone());
    ba.fix_pose(10);
    ba.fix_pose(20);

    let landmark_pos = Point3::new(0.0, 0.0, 10.0);
    ba.add_landmark(1, landmark_pos);
    ba.fix_landmark(1);
    let xc0 = pose_0.transform_world_point(&landmark_pos);
    let uv0 = camera.project(&xc0).unwrap();
    ba.add_observation(BaObservation {
        keyframe_id: 10,
        landmark_id: 1,
        xy: uv0,
    });

    let v0_truth = Vector3::<f64>::zeros();
    let v1_truth = Vector3::new(2.0, 0.0, 0.0);
    ba.add_velocity(10, v0_truth);
    ba.add_velocity(20, v1_truth);
    ba.fix_velocity(10);
    ba.fix_velocity(20);

    let raw_accel = Vector3::new(2.5, 0.0, 0.0);
    let b_truth_a = Vector3::new(0.5, 0.0, 0.0);
    let delta = imu_constant_accel_delta(raw_accel, 1.0, 0.001);
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        delta,
        gravity_world: Vector3::zeros(),
        weight_position: 1.0,
        weight_velocity: 1.0,
        weight_rotation: 1.0,
    });

    // Bias at truth, FIXED. Cost should be ≈ 0 immediately (no
    // optimisation needed).
    let mut bias_truth = Vector6::zeros();
    bias_truth.fixed_rows_mut::<3>(3).copy_from(&b_truth_a);
    ba.add_bias(10, bias_truth);
    ba.fix_bias(10);

    let cost = ba.cost();
    assert!(
        cost < 1e-6,
        "with bias fixed at truth, residual should vanish, got cost = {cost}"
    );
}

#[test]
fn bias_random_walk_zero_cost_at_truth() {
    // Same constant-+x-accel scenario as the bias zero-cost test, with
    // a random-walk factor 10↔20 between two equal biases. Both biases
    // are zero so the factor's residual `b_j − b_i = 0` and the cost
    // stays at zero on top of the visual + IMU contributions.
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let pose_0 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let pose_1 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    ba.add_pose(10, pose_0.clone());
    ba.add_pose(20, pose_1.clone());
    ba.fix_pose(10);
    ba.fix_pose(20);

    let landmark_pos = Point3::new(0.0, 0.0, 10.0);
    ba.add_landmark(1, landmark_pos);
    ba.fix_landmark(1);
    let xc0 = pose_0.transform_world_point(&landmark_pos);
    let uv0 = camera.project(&xc0).unwrap();
    ba.add_observation(BaObservation {
        keyframe_id: 10,
        landmark_id: 1,
        xy: uv0,
    });

    ba.add_velocity(10, Vector3::zeros());
    ba.add_velocity(20, Vector3::new(2.0, 0.0, 0.0));
    ba.fix_velocity(10);
    ba.fix_velocity(20);

    let delta = imu_constant_accel_delta(Vector3::new(2.0, 0.0, 0.0), 1.0, 0.001);
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        delta,
        gravity_world: Vector3::zeros(),
        weight_position: 1.0,
        weight_velocity: 1.0,
        weight_rotation: 1.0,
    });

    ba.add_bias(10, Vector6::zeros());
    ba.add_bias(20, Vector6::zeros());
    ba.add_bias_random_walk_factor(BiasRandomWalkFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        weight_gyro: 100.0,
        weight_accel: 100.0,
    });

    let cost = ba.cost();
    assert!(cost < 1e-6, "cost at truth should be ≈0, got {cost}");
}

#[test]
fn bias_random_walk_weights_gyro_and_accel_independently() {
    let mut ba = BundleAdjustment::new(pinhole());
    let mut bias_j = Vector6::zeros();
    bias_j[0] = 1.0;
    bias_j[3] = 2.0;
    ba.add_bias(10, Vector6::zeros());
    ba.add_bias(20, bias_j);
    ba.add_bias_random_walk_factor(BiasRandomWalkFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        weight_gyro: 3.0,
        weight_accel: 5.0,
    });
    assert!((ba.cost() - 23.0).abs() < 1.0e-12);
}

#[test]
fn bias_random_walk_pulls_drifted_bias_toward_neighbor() {
    // Two keyframes, both bias slots non-fixed, no IMU factor — only a
    // random-walk factor tying them together. KF10 bias starts at a
    // small drift, KF20 starts at zero. With a heavy weight on the
    // random-walk factor and the LM damping providing a weak pull to
    // zero, both biases should converge to the same value somewhere
    // between the two initial points.
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let pose_0 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let pose_1 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    ba.add_pose(10, pose_0.clone());
    ba.add_pose(20, pose_1.clone());
    ba.fix_pose(10);
    ba.fix_pose(20);

    let landmark_pos = Point3::new(0.0, 0.0, 10.0);
    ba.add_landmark(1, landmark_pos);
    ba.fix_landmark(1);
    let xc0 = pose_0.transform_world_point(&landmark_pos);
    let uv0 = camera.project(&xc0).unwrap();
    ba.add_observation(BaObservation {
        keyframe_id: 10,
        landmark_id: 1,
        xy: uv0,
    });

    let mut b_drifted = Vector6::zeros();
    b_drifted[3] = 0.4;
    ba.add_bias(10, b_drifted);
    ba.add_bias(20, Vector6::zeros());
    ba.add_bias_random_walk_factor(BiasRandomWalkFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        weight_gyro: 1e4,
        weight_accel: 1e4,
    });

    let config = BaConfig {
        max_iterations: 20,
        ..BaConfig::default()
    };
    ba.optimize(&config).expect("BA converges");

    let b_after_10 = ba.biases[&10];
    let b_after_20 = ba.biases[&20];
    let diff = (b_after_20 - b_after_10).norm();
    assert!(
        diff < 1e-3,
        "random-walk should pull both biases together, |b_20 − b_10| = {diff}"
    );
    // And they shouldn't both have stayed at their initial values.
    let moved = (b_after_10 - b_drifted).norm() + b_after_20.norm();
    assert!(
        moved > 1e-2,
        "both biases should have moved, total = {moved}"
    );
}

#[test]
fn bias_random_walk_propagates_observable_bias_to_neighbor() {
    // Three keyframes 0→1→2 along constant +x accel. KF10's bias is
    // observable through its IMU factor with the +0.5 m/s² hidden
    // accel offset; KF20 has no IMU factor of its own (we don't add
    // factor 20→30) but we tie bias[10] and bias[20] with a random-
    // walk factor. After BA: bias[20] should track bias[10] ≈ truth
    // through the random-walk connection.
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());

    let pose_0 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let pose_1 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    let pose_2 = pose_at(Vector3::new(3.0, 0.0, 0.0));
    ba.add_pose(10, pose_0.clone());
    ba.add_pose(20, pose_1.clone());
    ba.add_pose(30, pose_2.clone());
    ba.fix_pose(10);
    ba.fix_pose(20);
    ba.fix_pose(30);

    let landmark_pos = Point3::new(0.0, 0.0, 10.0);
    ba.add_landmark(1, landmark_pos);
    ba.fix_landmark(1);
    let xc0 = pose_0.transform_world_point(&landmark_pos);
    let uv0 = camera.project(&xc0).unwrap();
    ba.add_observation(BaObservation {
        keyframe_id: 10,
        landmark_id: 1,
        xy: uv0,
    });

    // Truth velocities for constant +x accel.
    ba.add_velocity(10, Vector3::zeros());
    ba.add_velocity(20, Vector3::new(2.0, 0.0, 0.0));
    ba.add_velocity(30, Vector3::new(4.0, 0.0, 0.0));
    ba.fix_velocity(10);
    ba.fix_velocity(20);
    ba.fix_velocity(30);

    // IMU factor on the 10→20 window with hidden accel bias.
    let raw_accel = Vector3::new(2.5, 0.0, 0.0);
    let b_truth_a = Vector3::new(0.5, 0.0, 0.0);
    let delta_01 = imu_constant_accel_delta(raw_accel, 1.0, 0.001);
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        delta: delta_01,
        gravity_world: Vector3::zeros(),
        weight_position: 1.0,
        weight_velocity: 1.0,
        weight_rotation: 1.0,
    });

    // Two bias slots; KF20 has no IMU factor anchoring it, so without
    // the random-walk tie it would just sit at its initial zero.
    ba.add_bias(10, Vector6::zeros());
    ba.add_bias(20, Vector6::zeros());
    ba.add_bias_random_walk_factor(BiasRandomWalkFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        weight_gyro: 1e4,
        weight_accel: 1e4,
    });

    let config = BaConfig {
        max_iterations: 30,
        ..BaConfig::default()
    };
    ba.optimize(&config).expect("BA converges");

    let b10_a: Vector3<f64> = ba.biases[&10].fixed_rows::<3>(3).into_owned();
    let b20_a: Vector3<f64> = ba.biases[&20].fixed_rows::<3>(3).into_owned();
    // The random-walk tie + the strong IMU pull should drag both
    // accel-bias estimates close to truth.
    assert!(
        (b10_a - b_truth_a).norm() < 1e-2,
        "b10 accel = {:?}, truth = {:?}",
        b10_a,
        b_truth_a
    );
    assert!(
        (b20_a - b_truth_a).norm() < 1e-2,
        "b20 accel = {:?}, truth = {:?}",
        b20_a,
        b_truth_a
    );
}

/// End-to-end BA timing on a large, covisibility-rich synthetic scene whose
/// Schur-reduced pose system is big and fairly dense — the regime where the
/// reduced-system factorization dominates each LM iteration. Synthetic is
/// deliberate: this measures *solver* time (the block vs scalar Cholesky
/// back-end behind `LinearSolver::Sparse`), not reconstruction quality, so it
/// needs a controllable problem size, not a real dataset.
///
/// `#[ignore]` so it never runs in CI; invoke explicitly with
/// `cargo test -p visloc-slam --release --test bundle_adjustment \
///   bench_ba_sparse_solver -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_ba_sparse_solver() {
    use std::time::Instant;

    let camera = pinhole();
    const KEYFRAMES: usize = 120;
    const WINDOW: usize = 40; // each landmark seen across this many keyframes
    const LM_PER_STEP: usize = 6;

    // Truth trajectory: a gentle forward arc with a slow yaw.
    let truth_poses: Vec<(u64, Pose)> = (0..KEYFRAMES)
        .map(|i| {
            let center = Vector3::new(0.3 * i as f64, 0.0, 0.04 * i as f64);
            (10 + i as u64 * 10, pose_with_yaw(center, 0.01 * i as f64))
        })
        .collect();

    // Landmarks tied to the trajectory so a window of consecutive keyframes
    // co-observes each one (banded → dense-ish Schur as WINDOW grows).
    let mut truth_landmarks: Vec<(u64, Point3<f64>)> = Vec::new();
    for i in 0..KEYFRAMES {
        for k in 0..LM_PER_STEP {
            let id = (i * LM_PER_STEP + k) as u64 + 1;
            let phase = (i * LM_PER_STEP + k) as f64;
            let x = 0.3 * i as f64 - 1.5 + 0.7 * (k as f64);
            let y = -1.0 + 0.5 * (k as f64 % 3.0) + 0.2 * phase.sin();
            let z = 5.0 + 0.4 * phase.cos();
            truth_landmarks.push((id, Point3::new(x, y, z)));
        }
    }

    let mut truth_ba = BundleAdjustment::new(camera.clone());
    for (id, pose) in &truth_poses {
        truth_ba.add_pose(*id, pose.clone());
    }
    for (id, point) in &truth_landmarks {
        truth_ba.add_landmark(*id, *point);
    }
    let mut observation_count = 0usize;
    for (li, (lm_id, point)) in truth_landmarks.iter().enumerate() {
        let first_kf = li / LM_PER_STEP;
        for (kf_id, pose) in truth_poses.iter().skip(first_kf).take(WINDOW) {
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            if let Some(uv) = camera.project(&xc) {
                truth_ba.add_observation(BaObservation {
                    keyframe_id: *kf_id,
                    landmark_id: *lm_id,
                    xy: uv,
                });
                observation_count += 1;
            }
        }
    }

    // Drift the poses (gauge-fix the first two) and a slice of landmarks.
    let mut ba = truth_ba.clone();
    ba.fix_pose(truth_poses[0].0);
    ba.fix_pose(truth_poses[1].0);
    for (i, (id, _)) in truth_poses.iter().enumerate().skip(2) {
        let truth = ba.poses[id].clone();
        let drift = Vector3::new(
            0.02 * (i as f64 * 0.7).sin(),
            0.015 * (i as f64 * 1.1).cos(),
            0.02 * (i as f64 * 0.3).sin(),
        );
        let center = truth.camera_center_world().coords + drift;
        let yaw = truth.world_to_camera.rotation.scaled_axis().y + 0.01 * (i as f64 * 0.5).sin();
        ba.add_pose(*id, pose_with_yaw(center, yaw));
    }
    for (i, (id, truth_point)) in truth_landmarks.iter().enumerate() {
        if i % 4 != 0 {
            continue;
        }
        let delta = Vector3::new(
            0.03 * (i as f64 * 0.7).sin(),
            0.04 * (i as f64 * 1.1).cos(),
            0.05 * (i as f64 * 0.3).sin(),
        );
        ba.add_landmark(*id, *truth_point + delta);
    }

    let free_poses = KEYFRAMES - 2;
    println!(
        "scene keyframes={KEYFRAMES} (free={free_poses}, Schur dim={}) landmarks={} observations={observation_count}",
        free_poses * 6,
        truth_landmarks.len(),
    );

    let start = Instant::now();
    let result = ba
        .optimize(&BaConfig {
            linear_solver: LinearSolver::Sparse,
            max_iterations: 20,
            ..BaConfig::default()
        })
        .expect("BA converges");
    let elapsed = start.elapsed();
    println!(
        "sparse BA: {:.3} s, {} iters, initial_cost={:.3e} final_cost={:.3e} converged={}",
        elapsed.as_secs_f64(),
        result.iterations.len(),
        result.initial_cost,
        result.final_cost,
        result.converged,
    );
}

#[test]
fn ba_refiner_skips_when_no_observations_align() {
    let camera = pinhole();
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    // Empty window + empty staging — refiner should produce a `Noop` skip
    // without panicking or attempting to run BA.
    let local_window = LocalMapWindow::default();
    let mut staged_update = StagedMapUpdate::new();
    let refiner = BundleAdjustmentRefiner::default();
    let refinement = refiner.refine(&map, &local_window, &mut staged_update);
    assert!(!refinement.refined);
    assert_eq!(refinement.reason, LocalRefinementReason::Noop);
}
