use super::{
    append_only_nn_matches, candidate_image_manifest_sha256, candidate_incident_images,
    f_to_e_stability_gate, hydrate_match_images, load_images_keypoints_only, parse_args_from,
    parse_persistent_match_worker_plan, project_fundamental_to_essential,
    refine_uncalibrated_f_winner, select_calibrated_essential_primary,
    sequence_f_to_e_high_support_override_gate, sequence_f_to_e_stability_gate,
    should_exclude_strict_uncalibrated_f_winner, snapshot_feature_manifest_hash,
    write_verified_pair_snapshot, write_verified_pair_snapshot_atomic, FToECandidateDiagnostics,
    PairMatcher, PairwiseMatches, SnapshotFeatureValidation, PERSISTENT_MATCH_WORKER_PLAN_MAGIC,
    PERSISTENT_MATCH_WORKER_PLAN_MAGIC_V2,
};
use nalgebra::{Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::path::PathBuf;
use visloc_rs::vision::two_view::{
    ConfigurationType, TwoViewCorrespondence, TwoViewGeometryReport,
};
use visloc_rs::FeatureSet;
use visloc_rs::{Camera, CameraModel};

fn minimal_args(extra: &[&str]) -> Vec<String> {
    let mut args = [
        "--width",
        "1600",
        "--height",
        "1066",
        "--fx",
        "879.4",
        "--fy",
        "879.4",
        "--cx",
        "803.4",
        "--cy",
        "532.6",
        "--out-colmap",
        "/tmp/persistent-cli-test",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    args.extend(extra.iter().map(|arg| (*arg).to_owned()));
    args
}

fn features(descriptors: Vec<Vec<f32>>) -> FeatureSet {
    let keypoints = (0..descriptors.len())
        .map(|index| Point2::new(index as f64, 0.0))
        .collect();
    FeatureSet::new(keypoints, descriptors).unwrap()
}

#[test]
fn append_only_preserves_primary_match_against_extra_distractor() {
    let query = features(vec![vec![0.0]]);
    // The primary prefix has a decisive 0.1-vs-1.0 Lowe match. The extra
    // 0.05 descriptor wins when matching the full set, replacing it in
    // the ordinary matcher.
    let train = features(vec![vec![0.1], vec![1.0], vec![0.05]]);
    let normal = PairMatcher::Nn.match_pair(0.8, true, 0, 1, &query, &train);
    assert_eq!(normal.len(), 1);
    assert_eq!(normal[0].train_index, 2);

    let append_only = append_only_nn_matches(0.8, true, &query, &train, 1, 2);
    assert_eq!(append_only.len(), 1);
    assert_eq!(append_only[0].query_index, 0);
    assert_eq!(append_only[0].train_index, 0);

    let configured = PairMatcher::NnAppendOnly {
        primary_keypoint_counts: vec![1, 2],
    };
    assert_eq!(
        configured.match_pair(0.8, true, 0, 1, &query, &train),
        append_only
    );
}

#[test]
fn descriptor_ensemble_appends_alternate_only_match_without_replacing_primary() {
    let query = features(vec![vec![0.0], vec![10.0]]);
    let train = features(vec![vec![0.1], vec![100.0]]);
    let primary = PairMatcher::Nn.match_pair(0.8, true, 0, 1, &query, &train);
    assert_eq!(primary.len(), 1);
    assert_eq!((primary[0].query_index, primary[0].train_index), (0, 0));

    let ensemble = PairMatcher::NnDescriptorEnsemble {
        primary_keypoint_counts: None,
        alternate_descriptors: vec![
            Some(vec![vec![0.0], vec![1.0]]),
            Some(vec![vec![0.1], vec![1.1]]),
        ],
    };
    let matches = ensemble.match_pair(0.8, true, 0, 1, &query, &train);
    assert_eq!(matches.len(), 2);
    assert_eq!((matches[0].query_index, matches[0].train_index), (0, 0));
    assert_eq!((matches[1].query_index, matches[1].train_index), (1, 1));
}

#[test]
fn fundamental_projection_uses_k_transpose_f_k_convention() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 510.0, 320.0, 240.0);
    let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 510.0, 240.0, 0.0, 0.0, 1.0);
    let rotation = UnitQuaternion::from_euler_angles(0.08, -0.12, 0.17)
        .to_rotation_matrix()
        .into_inner();
    let translation = Vector3::new(0.3, -0.2, 0.7);
    let skew = Matrix3::new(
        0.0,
        -translation.z,
        translation.y,
        translation.z,
        0.0,
        -translation.x,
        -translation.y,
        translation.x,
        0.0,
    );
    let essential = skew * rotation;
    let k_inverse = k.try_inverse().expect("synthetic K is invertible");
    let fundamental = k_inverse.transpose() * essential * k_inverse;
    let projected = project_fundamental_to_essential(&fundamental, &camera)
        .expect("synthetic F must project to E");
    let scale = projected.dot(&essential) / essential.dot(&essential);
    let relative_error = (projected - scale * essential).norm() / essential.norm();
    assert!(relative_error < 1.0e-10, "relative error={relative_error}");
}

fn synthetic_uncalibrated_f_winner() -> (Camera, Vec<TwoViewCorrespondence>, Matrix3<f64>) {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 510.0, 320.0, 240.0);
    let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 510.0, 240.0, 0.0, 0.0, 1.0);
    let rotation = UnitQuaternion::from_euler_angles(0.04, -0.06, 0.03)
        .to_rotation_matrix()
        .into_inner();
    let translation = Vector3::new(0.35, -0.08, 0.12);
    let skew = Matrix3::new(
        0.0,
        -translation.z,
        translation.y,
        translation.z,
        0.0,
        -translation.x,
        -translation.y,
        translation.x,
        0.0,
    );
    let essential = skew * rotation;
    let k_inverse = k.try_inverse().expect("synthetic K is invertible");
    let fundamental = k_inverse.transpose() * essential * k_inverse;
    let points = (0..32)
        .map(|index| {
            let phase = index as f64;
            let x = 0.95 * (phase * 0.71).sin();
            let y = 0.70 * (phase * 1.13).cos();
            let z = 4.0 + 0.35 * (phase * 0.37).sin() + 0.01 * phase;
            Point3::new(x, y, z)
        })
        .collect::<Vec<_>>();
    let correspondences = points
        .iter()
        .map(|point| {
            let moved = rotation * point.coords + translation;
            TwoViewCorrespondence::new(
                camera.project(point).expect("left point in front"),
                camera
                    .project(&Point3::new(moved.x, moved.y, moved.z))
                    .expect("right point in front"),
            )
        })
        .collect();
    (camera, correspondences, fundamental)
}

#[test]
fn guarded_f_winner_refinement_recovers_calibrated_inliers() {
    let (camera, correspondences, fundamental) = synthetic_uncalibrated_f_winner();
    let report = TwoViewGeometryReport {
        config: ConfigurationType::Uncalibrated,
        inliers: (0..correspondences.len()).collect(),
        essential: None,
        fundamental: Some(fundamental),
        homography: None,
        relative_pose: None,
        essential_inliers: Vec::new(),
        e_inlier_count: 0,
        f_inlier_count: correspondences.len(),
        h_inlier_count: 0,
    };
    let refinement = refine_uncalibrated_f_winner(&report, &correspondences, &camera, 12)
        .expect("exact calibrated F should produce a robust E_F");
    assert_eq!(
        refinement.inlier_indices,
        (0..correspondences.len()).collect::<Vec<_>>()
    );
    assert_eq!(refinement.f_inlier_count, correspondences.len());
    assert!(refinement.quality.cheirality_ratio >= 0.75);
    assert!(refinement.quality.angle_samples > 0);
}

#[test]
fn calibrated_essential_primary_can_beat_f_support_with_healthy_e() {
    let (camera, correspondences, fundamental) = synthetic_uncalibrated_f_winner();
    let essential = project_fundamental_to_essential(&fundamental, &camera)
        .expect("synthetic F must provide a calibrated E");
    let count = correspondences.len();
    let report = TwoViewGeometryReport {
        config: ConfigurationType::Uncalibrated,
        inliers: (0..count).collect(),
        essential: Some(essential),
        fundamental: Some(fundamental),
        homography: None,
        relative_pose: None,
        essential_inliers: (0..count).collect(),
        e_inlier_count: count,
        f_inlier_count: count + 8,
        h_inlier_count: 0,
    };
    let selected = select_calibrated_essential_primary(&report, &correspondences, &camera, 12)
        .expect("a healthy calibrated E should be selected even when F has more support");
    assert!(report.f_inlier_count > report.e_inlier_count);
    assert_eq!(selected.initial_inlier_count, count);
    assert!(selected.inlier_indices.len() >= count * 4 / 5);
    assert!(selected.quality.mean_sampson.is_finite());
}

#[test]
fn calibrated_essential_primary_rejects_degenerate_evidence() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 510.0, 320.0, 240.0);
    let correspondences = (0..32)
        .map(|_| TwoViewCorrespondence::new(Point2::new(320.0, 240.0), Point2::new(320.0, 240.0)))
        .collect::<Vec<_>>();
    let report = TwoViewGeometryReport {
        config: ConfigurationType::Uncalibrated,
        inliers: (0..correspondences.len()).collect(),
        essential: Some(Matrix3::identity()),
        fundamental: Some(Matrix3::identity()),
        homography: None,
        relative_pose: None,
        essential_inliers: (0..correspondences.len()).collect(),
        e_inlier_count: correspondences.len(),
        f_inlier_count: correspondences.len() + 8,
        h_inlier_count: 0,
    };
    assert!(
        select_calibrated_essential_primary(&report, &correspondences, &camera, 12).is_none(),
        "coincident observations must not create a calibrated primary edge"
    );
}

#[test]
fn guarded_f_winner_refinement_falls_back_for_invalid_weak_or_calibrated_input() {
    let (camera, correspondences, fundamental) = synthetic_uncalibrated_f_winner();
    let base = TwoViewGeometryReport {
        config: ConfigurationType::Uncalibrated,
        inliers: (0..correspondences.len()).collect(),
        essential: None,
        fundamental: Some(fundamental),
        homography: None,
        relative_pose: None,
        essential_inliers: Vec::new(),
        e_inlier_count: 0,
        f_inlier_count: correspondences.len(),
        h_inlier_count: 0,
    };
    let mut invalid = base.clone();
    invalid.fundamental = Some(Matrix3::zeros());
    assert!(refine_uncalibrated_f_winner(&invalid, &correspondences, &camera, 12).is_none());
    assert!(refine_uncalibrated_f_winner(&base, &correspondences[..8], &camera, 12).is_none());
    let mut calibrated = base;
    calibrated.config = ConfigurationType::Calibrated;
    assert!(refine_uncalibrated_f_winner(&calibrated, &correspondences, &camera, 12).is_none());
}

#[test]
fn strict_f_to_e_exclusion_is_opt_in_and_keeps_no_calibration_unchanged() {
    let (camera, correspondences, fundamental) = synthetic_uncalibrated_f_winner();
    let report = TwoViewGeometryReport {
        config: ConfigurationType::Uncalibrated,
        inliers: (0..correspondences.len()).collect(),
        essential: None,
        fundamental: Some(fundamental),
        homography: None,
        relative_pose: None,
        essential_inliers: Vec::new(),
        e_inlier_count: 0,
        f_inlier_count: correspondences.len(),
        h_inlier_count: 0,
    };
    let refinement = refine_uncalibrated_f_winner(&report, &correspondences, &camera, 12)
        .expect("synthetic calibrated F should pass the strict gate");
    assert!(!should_exclude_strict_uncalibrated_f_winner(
        false, &camera, &report, None,
    ));
    assert!(!should_exclude_strict_uncalibrated_f_winner(
        true,
        &camera,
        &report,
        Some(&refinement),
    ));
    assert!(should_exclude_strict_uncalibrated_f_winner(
        true, &camera, &report, None,
    ));

    let mut calibrated = report.clone();
    calibrated.config = ConfigurationType::Calibrated;
    assert!(!should_exclude_strict_uncalibrated_f_winner(
        true,
        &camera,
        &calibrated,
        None,
    ));

    let no_calibration = Camera {
        id: camera.id,
        model: CameraModel::Unknown("NONE".to_owned()),
        width: camera.width,
        height: camera.height,
        params: Vec::new(),
    };
    assert!(!should_exclude_strict_uncalibrated_f_winner(
        true,
        &no_calibration,
        &report,
        None,
    ));
}

#[test]
fn f_to_e_stability_gate_is_strict_and_deterministic() {
    let stable = FToECandidateDiagnostics {
        calibrated_s1: 1.0,
        calibrated_s2: 0.999,
        calibrated_s3: 1.0e-12,
        projection_distortion: 0.001,
        s1_s2_mismatch: 0.002,
        s3_s2_ratio: 1.0e-12,
        f_inliers: 100,
        ef_inliers: 98,
        ef_overlap_on_f: 0.98,
        f_normalized_residual: 0.001,
        ef_normalized_residual_on_f: 0.002,
        ef_to_f_residual_ratio: 2.0,
        cheirality_ratio: 0.95,
        cheirality_margin: 0.9,
        ef_angle_p25_deg: 1.5,
        stable_refits: 3,
        pose_rotation_spread_deg: 1.0,
        pose_translation_spread_deg: 2.0,
    };
    assert!(f_to_e_stability_gate(&stable));
    assert!(f_to_e_stability_gate(&stable));

    let mut unstable_pose = stable;
    unstable_pose.pose_translation_spread_deg = 5.01;
    assert!(!f_to_e_stability_gate(&unstable_pose));
    assert!(sequence_f_to_e_stability_gate(&unstable_pose));
    unstable_pose.pose_translation_spread_deg = 10.01;
    assert!(!sequence_f_to_e_stability_gate(&unstable_pose));
    unstable_pose.pose_translation_spread_deg = 2.0;
    unstable_pose.pose_rotation_spread_deg = 5.01;
    assert!(!sequence_f_to_e_stability_gate(&unstable_pose));
    let mut non_essential = stable;
    non_essential.projection_distortion = 0.0101;
    assert!(!f_to_e_stability_gate(&non_essential));
    let mut weak_overlap = stable;
    weak_overlap.ef_overlap_on_f = 0.899;
    assert!(!f_to_e_stability_gate(&weak_overlap));

    let mut high_support_override = stable;
    high_support_override.f_inliers = 556;
    high_support_override.ef_inliers = 555;
    high_support_override.ef_overlap_on_f = 555.0 / 556.0;
    high_support_override.cheirality_ratio = 555.0 / 555.0;
    high_support_override.cheirality_margin = 1.0;
    high_support_override.ef_angle_p25_deg = 1.528;
    high_support_override.pose_translation_spread_deg = 49.094;
    assert!(!sequence_f_to_e_stability_gate(&high_support_override));
    assert!(sequence_f_to_e_high_support_override_gate(
        &high_support_override
    ));

    let mut poor_override_overlap = high_support_override;
    poor_override_overlap.ef_overlap_on_f = 0.949;
    assert!(!sequence_f_to_e_high_support_override_gate(
        &poor_override_overlap
    ));
    let mut poor_override_cheirality = high_support_override;
    poor_override_cheirality.cheirality_ratio = 0.949;
    assert!(!sequence_f_to_e_high_support_override_gate(
        &poor_override_cheirality
    ));
    let mut poor_override_margin = high_support_override;
    poor_override_margin.cheirality_margin = 0.749;
    assert!(!sequence_f_to_e_high_support_override_gate(
        &poor_override_margin
    ));
    let mut poor_override_manifold = high_support_override;
    poor_override_manifold.projection_distortion = 0.0101;
    assert!(!sequence_f_to_e_high_support_override_gate(
        &poor_override_manifold
    ));
    let mut poor_override_parallax = high_support_override;
    poor_override_parallax.ef_angle_p25_deg = 0.999;
    assert!(!sequence_f_to_e_high_support_override_gate(
        &poor_override_parallax
    ));
}

#[test]
fn persistent_worker_plan_parser_rejects_path_traversal_and_reuse() {
    let root = std::env::temp_dir().join(format!(
        "visloc_persistent_plan_parser_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let hash = "a".repeat(64);
    let valid = format!(
        "{PERSISTENT_MATCH_WORKER_PLAN_MAGIC}\n\
             images 2\n\
             image 0 left.png\n\
             image 1 right.png\n\
             candidate_index_sha256 {hash}\n\
             feature_manifest_sha256 {hash}\n\
             pairs 1\n\
             shards 1\n\
             shard 4 candidates/candidate-000004.txt matches/verified-000004.vps {hash}\n"
    );
    let path = root.join("valid.plan");
    std::fs::write(&path, valid).unwrap();
    let parsed = parse_persistent_match_worker_plan(&path).unwrap();
    assert_eq!(parsed.image_names, ["left.png", "right.png"]);
    assert_eq!(parsed.shards[0].id, 4);
    assert_eq!(parsed.root, root);

    let traversal = root.join("traversal.plan");
    std::fs::write(
        &traversal,
        format!(
            "{PERSISTENT_MATCH_WORKER_PLAN_MAGIC}\n\
                 images 2\nimage 0 left.png\nimage 1 right.png\n\
                 candidate_index_sha256 {hash}\nfeature_manifest_sha256 {hash}\n\
                 pairs 1\nshards 1\n\
                 shard 0 ../candidate.txt matches/out.vps {hash}\n"
        ),
    )
    .unwrap();
    assert!(parse_persistent_match_worker_plan(&traversal)
        .unwrap_err()
        .contains("relative path"));

    let duplicate = root.join("duplicate.plan");
    std::fs::write(
        &duplicate,
        format!(
            "{PERSISTENT_MATCH_WORKER_PLAN_MAGIC}\n\
                 images 2\nimage 0 left.png\nimage 1 right.png\n\
                 candidate_index_sha256 {hash}\nfeature_manifest_sha256 {hash}\n\
                 pairs 2\nshards 2\n\
                 shard 0 candidates/a.txt matches/a.vps {hash}\n\
                 shard 1 candidates/a.txt matches/b.vps {hash}\n"
        ),
    )
    .unwrap();
    assert!(parse_persistent_match_worker_plan(&duplicate)
        .unwrap_err()
        .contains("repeats"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn persistent_worker_plan_v2_carries_compact_shard_bindings() {
    let root = std::env::temp_dir().join(format!(
        "visloc_persistent_plan_v2_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let names = vec!["left.png".to_owned(), "right.png".to_owned()];
    let source_hash = "a".repeat(64);
    let image_hash = candidate_image_manifest_sha256(&names);
    let plan = format!(
        "{PERSISTENT_MATCH_WORKER_PLAN_MAGIC_V2}\n\
             images 2\nimage 0 left.png\nimage 1 right.png\n\
             candidate_source_sha256 {source_hash}\n\
             image_manifest_sha256 {image_hash}\n\
             candidate_index_sha256 {source_hash}\n\
             feature_manifest_sha256 {source_hash}\n\
             pairs 1\nshards 1\n\
             shard 0 candidates/candidate-000000.txt matches/verified-000000.vps {source_hash}\n"
    );
    let path = root.join("v2.plan");
    std::fs::write(&path, plan).unwrap();
    let parsed = parse_persistent_match_worker_plan(&path).unwrap();
    assert_eq!(
        parsed.candidate_source_sha256.as_deref(),
        Some(source_hash.as_str())
    );
    assert_eq!(
        parsed.image_manifest_sha256.as_deref(),
        Some(image_hash.as_str())
    );
    assert_eq!(parsed.image_names, names);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn persistent_worker_cli_is_opt_in_and_fail_closed() {
    let defaults = parse_args_from(minimal_args(&[])).unwrap();
    assert!(defaults.persistent_match_worker_plan.is_none());
    assert!(!defaults.stream_match_features);
    let valid = parse_args_from(vec![
        "--input-colmap-calibration".to_owned(),
        "/tmp/calibration".to_owned(),
        "--verification-mode".to_owned(),
        "full".to_owned(),
        "--persistent-match-worker-plan".to_owned(),
        "/tmp/match-worker.plan".to_owned(),
        "--out-colmap".to_owned(),
        "/tmp/persistent-model".to_owned(),
    ])
    .unwrap();
    assert_eq!(
        valid.persistent_match_worker_plan,
        Some(PathBuf::from("/tmp/match-worker.plan"))
    );
    let streamed = parse_args_from(vec![
        "--input-colmap-calibration".to_owned(),
        "/tmp/calibration".to_owned(),
        "--verification-mode".to_owned(),
        "full".to_owned(),
        "--persistent-match-worker-plan".to_owned(),
        "/tmp/match-worker.plan".to_owned(),
        "--stream-match-features".to_owned(),
        "--out-colmap".to_owned(),
        "/tmp/persistent-model".to_owned(),
    ])
    .unwrap();
    assert!(streamed.stream_match_features);
    assert!(parse_args_from(minimal_args(&["--stream-match-features"])).is_err());

    let guided = parse_args_from(vec![
        "--input-colmap-calibration".to_owned(),
        "/tmp/calibration".to_owned(),
        "--verification-mode".to_owned(),
        "full".to_owned(),
        "--persistent-match-worker-plan".to_owned(),
        "/tmp/match-worker.plan".to_owned(),
        "--guided-matching".to_owned(),
        "--colmap-guided-matching".to_owned(),
        "--out-colmap".to_owned(),
        "/tmp/persistent-model".to_owned(),
    ])
    .unwrap();
    assert!(guided.guided_matching);
    assert!(guided.colmap_guided_matching);

    for extra in [
        vec!["--mapper", "global"],
        vec!["--matcher", "lightglue"],
        vec!["--verification-mode", "legacy"],
        vec!["--candidate-manifest", "/tmp/candidate.txt"],
        vec!["--import-verified-pairs-snapshot", "/tmp/pairs.vps"],
        vec!["--canonical-feature-order"],
        vec!["--union-traversal-order", "reverse-both"],
        vec!["--rematch-stems", "foo"],
        vec!["--diagnose-bearing-gt", "/tmp/gt/images.txt"],
        vec!["--global-ba-max-refinements", "1"],
    ] {
        let mut args = vec![
            "--input-colmap-calibration".to_owned(),
            "/tmp/calibration".to_owned(),
            "--verification-mode".to_owned(),
            "full".to_owned(),
            "--persistent-match-worker-plan".to_owned(),
            "/tmp/match-worker.plan".to_owned(),
            "--out-colmap".to_owned(),
            "/tmp/persistent-model".to_owned(),
        ];
        args.extend(extra.iter().map(|arg| (*arg).to_owned()));
        assert!(
            parse_args_from(args).is_err(),
            "persistent worker accepted unsupported options: {extra:?}"
        );
    }
}

#[test]
fn streamed_match_hydrates_only_candidate_incident_descriptors() {
    let root = std::env::temp_dir().join(format!(
        "visloc_streamed_match_features_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    for (name, row) in [
        ("a_features.txt", "1 2 0.1 0.2\n"),
        ("b_features.txt", "3 4 0.3 0.4\n"),
        ("c_features.txt", "5 6 0.5 0.6\n"),
    ] {
        std::fs::write(root.join(name), row).unwrap();
    }
    let loaded = load_images_keypoints_only(&root, "_features.txt", ".png").unwrap();
    let mut features = loaded.features;
    assert!(features
        .iter()
        .all(|set| set.descriptors.iter().all(Vec::is_empty)));

    let hydrated = candidate_incident_images(&[(0, 2)]);
    hydrate_match_images(
        &mut features,
        &loaded.paths,
        &loaded.fingerprints,
        &hydrated,
    )
    .unwrap();
    assert_eq!(hydrated, [0, 2]);
    assert_eq!(features[0].descriptors, [vec![0.2]]);
    assert!(features[1].descriptors[0].is_empty());
    assert_eq!(features[2].descriptors, [vec![0.6]]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn cached_snapshot_feature_validation_is_byte_identical_to_default_writer() {
    let root = std::env::temp_dir().join(format!(
        "visloc_persistent_snapshot_writer_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let features = vec![
        FeatureSet::new(vec![Point2::new(10.0, 20.0)], vec![vec![0.1, 0.2, 0.3]]).unwrap(),
        FeatureSet::new(vec![Point2::new(30.0, 40.0)], vec![vec![0.4, 0.5, 0.6]]).unwrap(),
    ];
    let image_names = vec!["left.png".to_owned(), "right.png".to_owned()];
    let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0)])];
    let metadata = HashMap::new();
    let args = parse_args_from(minimal_args(&[])).unwrap();
    let validation = SnapshotFeatureValidation {
        feature_counts: features.iter().map(FeatureSet::len).collect(),
        feature_manifest_hash: snapshot_feature_manifest_hash(&features),
    };
    write_verified_pair_snapshot(
        &root.join("default.vps"),
        &image_names,
        &features,
        &args.camera,
        &pairwise,
        &metadata,
        &args,
    )
    .unwrap();
    write_verified_pair_snapshot_atomic(
        &root.join("cached.vps"),
        &image_names,
        &features,
        &args.camera,
        &pairwise,
        &metadata,
        &args,
        &validation,
    )
    .unwrap();
    assert_eq!(
        std::fs::read(root.join("default.vps")).unwrap(),
        std::fs::read(root.join("cached.vps")).unwrap()
    );
    let _ = std::fs::remove_dir_all(root);
}
