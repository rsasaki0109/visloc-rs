use super::{
    apply_snapshot_coordinate_override, apply_union_traversal_order,
    apply_union_traversal_order_with_features, candidate_image_manifest_sha256,
    candidate_pairs_temporal_pyramid_from_retrieval, candidate_pairs_vlad_lsh_scored,
    candidate_pairs_vlad_scored, candidate_pairs_vlad_scored_from_globals,
    candidate_pairs_vlad_union, canonicalize_feature_order, canonicalize_pairwise_loci,
    cap_mapper_pair_matches, colmap_guided_geometry, colmap_guided_matches,
    descriptor_squared_distance, effective_ann_bits, effective_config_hash,
    effective_config_snapshot, exact_topk_similar_images,
    filter_imported_verified_pairs_by_stem_window, filter_pairs_by_stem_window,
    imported_reference_quality_is_strong, initial_poses_from_colmap_images_txt,
    initial_poses_from_colmap_images_txt_with_expected_cameras, load_images,
    load_images_keypoints_only, load_input_colmap_calibration, match_candidate_cmp,
    model_cross_validation_is_held_out, model_cross_validation_is_held_out_for_pixels,
    model_cross_validation_selection_score, parse_args_from, parse_candidate_manifest,
    parse_candidate_manifest_with_metadata, parse_candidate_shard_v2,
    parse_colmap_image_camera_assignments, parse_colmap_track_membership, parse_diagnose_stems,
    ranked_view_graph_components, read_feature_set, remap_feature_keypoints_by_old_to_new,
    replace_feature_keypoints_from_native, rig_local_pairs, rig_temporal_pyramid_pairs,
    rig_temporal_pyramid_pairs_with_manifest, robust_huber_mean, snapshot_export_config,
    snapshot_feature_manifest_hash, snapshot_feature_validation_from_files,
    stream_vlad_globals_from_feature_files, summarize_model_cross_validation_bucket,
    temporal_pyramid_offsets_string, translation_direction_delta_deg, unordered_pairwise_edge_hash,
    validate_diagnose_options, verified_pair_oracle_map, write_candidate_manifest,
    write_candidate_manifest_with_metadata, Camera, ColmapTrackMembership, ConfigurationType,
    EssentialPairQuality, FeatureLocusMetadata, ImportedVerifiedPair, IncrementalSfmConfig,
    LinearSolver, ModelCrossValidationBucket, ModelCrossValidationSummary, NextImagePolicy,
    PairwiseMatches, PerImageCameras, RobustKernel, TwoViewGeometryReport, UnionTraversalOrder,
};
use nalgebra::{Matrix3, Point2, Vector3};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use visloc_rs::vision::features::FeatureSet;
use visloc_rs::{BaConfig, DescriptorMatch};

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
        "/tmp/diagnose-cli-test",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    args.extend(extra.iter().map(|arg| (*arg).to_owned()));
    args
}

#[test]
fn next_image_policy_defaults_to_auto_and_snapshot_replay_stays_count() {
    let defaults = parse_args_from(minimal_args(&[])).unwrap();
    assert_eq!(defaults.next_image_policy, NextImagePolicy::Auto);

    let count = parse_args_from(minimal_args(&["--next-image-policy", "count"])).unwrap();
    assert_eq!(
        count.next_image_policy,
        NextImagePolicy::CorrespondenceCount
    );
    let visibility = parse_args_from(minimal_args(&["--next-image-policy", "visibility"])).unwrap();
    assert_eq!(
        visibility.next_image_policy,
        NextImagePolicy::VisibilityPyramid
    );
    let auto = parse_args_from(minimal_args(&["--next-image-policy", "auto"])).unwrap();
    assert_eq!(auto.next_image_policy, NextImagePolicy::Auto);
    let snapshot_default = parse_args_from(minimal_args(&[
        "--import-verified-pairs-snapshot",
        "/tmp/pairs.vps",
    ]))
    .unwrap();
    assert_eq!(
        snapshot_default.next_image_policy,
        NextImagePolicy::CorrespondenceCount
    );
    let snapshot_auto = parse_args_from(minimal_args(&[
        "--import-verified-pairs-snapshot",
        "/tmp/pairs.vps",
        "--next-image-policy",
        "auto",
    ]))
    .unwrap();
    assert_eq!(snapshot_auto.next_image_policy, NextImagePolicy::Auto);
    assert!(parse_args_from(minimal_args(&["--next-image-policy", "pyramid"])).is_err());
    assert!(parse_args_from(minimal_args(&["--next-image-policy"])).is_err());
}

#[test]
fn mapper_match_cap_is_opt_in_and_preserves_verified_prefixes() {
    let defaults = parse_args_from(minimal_args(&[])).unwrap();
    assert!(defaults.max_mapper_matches_per_pair.is_none());
    assert!(parse_args_from(minimal_args(&["--max-mapper-matches-per-pair", "0",])).is_err());

    let mut pairwise = vec![PairwiseMatches {
        image_i: 0,
        image_j: 1,
        matches: vec![(0, 10), (1, 11), (2, 12)],
        two_view_config: Some(ConfigurationType::Calibrated),
        essential_matches: Some(vec![(0, 10), (2, 12)]),
        essential_matrix: Some(Matrix3::identity()),
    }];
    let stats = cap_mapper_pair_matches(&mut pairwise, 2);
    assert_eq!(stats.pairs_capped, 1);
    assert_eq!(stats.matches_before, 3);
    assert_eq!(stats.matches_after, 2);
    assert_eq!(stats.essential_before, 2);
    assert_eq!(stats.essential_after, 2);
    assert_eq!(pairwise[0].matches, vec![(0, 10), (1, 11)]);
    assert_eq!(
        pairwise[0].essential_matches.as_deref(),
        Some(&[(0, 10), (2, 12)][..])
    );
}

#[test]
fn candidate_manifest_round_trip_binds_image_order_and_rejects_duplicates() {
    let root =
        std::env::temp_dir().join(format!("visloc_candidate_manifest_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("pairs.txt");
    let names = vec![
        "DSC_0001.JPG".to_owned(),
        "DSC_0002.JPG".to_owned(),
        "DSC_0003.JPG".to_owned(),
    ];
    let pairs = vec![(0, 2), (0, 1)];
    write_candidate_manifest(&path, &names, &pairs).unwrap();
    assert_eq!(parse_candidate_manifest(&path, &names).unwrap(), pairs);

    let mut wrong_names = names.clone();
    wrong_names.swap(0, 1);
    assert!(parse_candidate_manifest(&path, &wrong_names)
        .unwrap_err()
        .contains("image entry"));

    std::fs::write(
        &path,
        "visloc_candidate_manifest_v1\nimages 3\n\
             image 0 DSC_0001.JPG\nimage 1 DSC_0002.JPG\nimage 2 DSC_0003.JPG\n\
             pairs 2\npair 0 1\npair 1 0\n",
    )
    .unwrap();
    assert!(parse_candidate_manifest(&path, &names)
        .unwrap_err()
        .contains("must satisfy"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn compact_candidate_shard_v2_binds_source_and_plan_image_order() {
    let root = std::env::temp_dir().join(format!(
        "visloc_candidate_shard_v2_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("candidate-000000.txt");
    let names = vec!["left.png".to_owned(), "right.png".to_owned()];
    let source_hash = "a".repeat(64);
    let image_hash = candidate_image_manifest_sha256(&names);
    std::fs::write(
        &path,
        format!(
            "visloc_candidate_shard_v2\nsource_manifest_sha256 {source_hash}\n\
                 image_manifest_sha256 {image_hash}\nimages 2\npairs 1\npair 0 1\n"
        ),
    )
    .unwrap();
    assert_eq!(
        parse_candidate_shard_v2(&path, &names, Some(&source_hash), Some(&image_hash),).unwrap(),
        vec![(0, 1)]
    );

    let mut wrong_names = names.clone();
    wrong_names.swap(0, 1);
    assert!(
        parse_candidate_shard_v2(&path, &wrong_names, Some(&source_hash), Some(&image_hash),)
            .unwrap_err()
            .contains("image manifest hash")
    );
    assert!(
        parse_candidate_shard_v2(&path, &names, Some(&"b".repeat(64)), Some(&image_hash),)
            .unwrap_err()
            .contains("source manifest hash")
    );
    std::fs::write(
        &path,
        format!(
            "visloc_candidate_shard_v2\nsource_manifest_sha256 {source_hash}\n\
                 image_manifest_sha256 {image_hash}\nimages 3\npairs 1\npair 0 1\n"
        ),
    )
    .unwrap();
    assert!(
        parse_candidate_shard_v2(&path, &names, Some(&source_hash), Some(&image_hash),)
            .unwrap_err()
            .contains("image count")
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn candidate_schedule_flags_are_explicit_and_validated() {
    let union = parse_args_from(minimal_args(&[
        "--pair-source",
        "vlad-union",
        "--local-stem-window",
        "3",
        "--candidate-budget",
        "200",
    ]))
    .unwrap();
    assert!(matches!(union.pair_source, super::PairSource::VladUnion));
    assert_eq!(union.local_stem_window, Some(3));
    assert_eq!(union.candidate_budget, Some(200));
    let rig = parse_args_from(minimal_args(&[
        "--pair-source",
        "vlad-union",
        "--local-stem-window",
        "3",
        "--rig-local-grouping",
    ]))
    .unwrap();
    assert!(rig.rig_local_grouping);
    assert!(parse_args_from(minimal_args(&["--pair-source", "vlad-union",])).is_err());
    assert!(parse_args_from(minimal_args(&["--local-stem-window", "3",])).is_err());
    assert!(parse_args_from(minimal_args(&["--rig-local-grouping",])).is_err());
    assert!(parse_args_from(minimal_args(&["--candidate-budget", "0",])).is_err());

    let temporal = parse_args_from(minimal_args(&[
        "--pair-source",
        "temporal-pyramid",
        "--temporal-pyramid-max-offset",
        "64",
        "--candidate-budget",
        "12000",
        "--retrieval-min-frame-gap",
        "128",
    ]))
    .unwrap();
    assert!(matches!(
        temporal.pair_source,
        super::PairSource::TemporalPyramid
    ));
    assert_eq!(temporal.temporal_pyramid_max_offset, 64);
    assert_eq!(temporal.candidate_budget, Some(12000));
    assert_eq!(temporal.retrieval_min_frame_gap, Some(128));
    let temporal_manifest = parse_args_from(minimal_args(&[
        "--pair-source",
        "temporal-pyramid",
        "--rig-frame-manifest",
        "/tmp/rig.txt",
    ]))
    .unwrap();
    assert_eq!(
        temporal_manifest.rig_frame_manifest,
        Some(PathBuf::from("/tmp/rig.txt"))
    );
    assert!(parse_args_from(minimal_args(&["--rig-frame-manifest", "/tmp/rig.txt",])).is_err());
    assert!(parse_args_from(minimal_args(&["--retrieval-min-frame-gap", "128",])).is_err());
    assert!(parse_args_from(minimal_args(&[
        "--pair-source",
        "temporal-pyramid",
        "--temporal-pyramid-max-offset",
        "0",
    ]))
    .is_err());

    let ann_args: Vec<String> = [
        "--feature-extractor",
        "files",
        "--features-dir",
        "/tmp/features",
        "--input-colmap-calibration",
        "/tmp/calibration",
        "--pair-source",
        "temporal-pyramid",
        "--export-candidate-manifest",
        "/tmp/candidates.txt",
        "--stream-candidate-features",
        "--retrieval-backend",
        "lsh",
        "--ann-tables",
        "6",
        "--ann-bits",
        "14",
        "--ann-probes",
        "8",
        "--out-colmap",
        "/tmp/unused-model",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let ann = parse_args_from(ann_args).unwrap();
    assert_eq!(ann.retrieval_backend, super::RetrievalBackend::Lsh);
    assert_eq!((ann.ann_tables, ann.ann_bits, ann.ann_probes), (6, 14, 8));
    assert_eq!(effective_ann_bits(0, 999), 6);
    assert_eq!(effective_ann_bits(0, 1_000), 6);
    assert_eq!(effective_ann_bits(0, 2_500), 7);
    assert_eq!(effective_ann_bits(0, 5_000), 8);
    assert_eq!(effective_ann_bits(0, 10_000), 9);
    assert_eq!(effective_ann_bits(12, 10_000), 12);
    assert!(parse_args_from(minimal_args(&["--retrieval-backend", "lsh"])).is_err());
    assert!(parse_args_from(minimal_args(&[
        "--pair-source",
        "temporal-pyramid",
        "--rig-local-grouping",
    ]))
    .is_err());
}

#[test]
fn bounded_local_vlad_schedule_is_deterministic_and_respects_budget() {
    let features: Vec<FeatureSet> = (0..5)
        .map(|index| {
            FeatureSet::new(
                vec![Point2::new(index as f64, 0.0)],
                vec![vec![1.0, index as f32 + 1.0]],
            )
            .unwrap()
        })
        .collect();
    let names: Vec<String> = (0..5).map(|index| format!("DSC_{index:04}.JPG")).collect();
    let first = candidate_pairs_vlad_union(&features, &names, 4, 12, 1, Some(3)).unwrap();
    let second = candidate_pairs_vlad_union(&features, &names, 4, 12, 1, Some(3)).unwrap();
    assert_eq!(first, second);
    assert!(first.len() <= 3);
    assert!(first.iter().all(|&(i, j)| i < j && j < features.len()));
}

#[test]
fn bounded_exact_topk_matches_full_sort_with_stable_ties() {
    let globals = vec![
        vec![1.0, 0.0],
        vec![0.8, 0.2],
        vec![0.8, -0.2],
        vec![0.0, 1.0],
        vec![-1.0, 0.0],
    ];
    for query in 0..globals.len() {
        for topk in 0..=globals.len() + 1 {
            let mut full: Vec<_> = (0..globals.len())
                .filter(|&candidate| candidate != query)
                .map(|candidate| {
                    (
                        candidate,
                        super::cosine_similarity(&globals[query], &globals[candidate]),
                    )
                })
                .collect();
            full.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
            full.truncate(topk);
            assert_eq!(exact_topk_similar_images(query, &globals, topk), full);
        }
    }
}

#[test]
fn streamed_vlad_missing_vocabulary_refuses_exhaustive_fallback() {
    for globals in [None, Some(Vec::new())] {
        let streamed = super::StreamedVladGlobals {
            globals,
            total_descriptors: 0,
            sampled_descriptors: 0,
        };
        assert!(streamed
            .appearance_globals()
            .unwrap_err()
            .contains("refusing exhaustive fallback"));
    }
}

#[test]
fn streamed_vlad_empty_feature_files_refuse_exhaustive_fallback() {
    let root =
        std::env::temp_dir().join(format!("visloc_empty_streamed_vlad_{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let files = vec!["a_features.txt".to_owned(), "b_features.txt".to_owned()];
    for file in &files {
        std::fs::write(root.join(file), b"").unwrap();
    }
    let rig = PerImageCameras::new(
        (0..files.len())
            .map(|image| Camera::pinhole(image as u64, 100, 100, 50.0, 50.0, 50.0, 50.0))
            .collect(),
    )
    .unwrap();
    let streamed = stream_vlad_globals_from_feature_files(&root, &files, &rig, 3).unwrap();
    assert_eq!(streamed.total_descriptors, 0);
    assert!(streamed.appearance_globals().is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn streamed_vlad_globals_preserve_batch_candidate_pairs() {
    let root = std::env::temp_dir().join(format!(
        "visloc_streamed_vlad_candidates_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut files = Vec::new();
    let mut features = Vec::new();
    for image in 0..5 {
        let file = format!("image_{image:03}_features.txt");
        let rows = (0..4)
            .map(|row| {
                format!(
                    "{} {} 1.0 {} {}\n",
                    10 + row,
                    20 + image,
                    image as f32 + row as f32 * 0.25 + 1.0,
                    (image + row) as f32 * 0.5 + 0.5,
                )
            })
            .collect::<String>();
        std::fs::write(root.join(&file), rows).unwrap();
        files.push(file.clone());
        features.push(read_feature_set(&root.join(file)).unwrap());
    }
    let rig = PerImageCameras::new(
        (0..files.len())
            .map(|image| Camera::pinhole(image as u64, 100, 100, 50.0, 50.0, 50.0, 50.0))
            .collect(),
    )
    .unwrap();
    let streamed = stream_vlad_globals_from_feature_files(&root, &files, &rig, 3).unwrap();
    assert_eq!(streamed.total_descriptors, 20);
    assert_eq!(streamed.sampled_descriptors, 20);
    let streamed_pairs =
        candidate_pairs_vlad_scored_from_globals(streamed.appearance_globals().unwrap(), 2, false);
    let batch_pairs = candidate_pairs_vlad_scored(&features, 3, 2, false, false);
    assert_eq!(streamed_pairs, batch_pairs);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn vlad_lsh_is_deterministic_and_recovers_identical_neighbours() {
    let mut globals = Vec::new();
    for pair in 0..20 {
        let descriptor = (0..64)
            .map(|dimension| (((pair * 67 + dimension * 17) % 101) as f32 - 50.0) / 50.0)
            .collect::<Vec<_>>();
        globals.push(descriptor.clone());
        globals.push(descriptor);
    }
    let first = candidate_pairs_vlad_lsh_scored(&globals, 1, 4, 8, 8);
    let second = candidate_pairs_vlad_lsh_scored(&globals, 1, 4, 8, 8);
    assert_eq!(first, second);
    let pairs = first
        .into_iter()
        .map(|(pair, _)| pair)
        .collect::<HashSet<_>>();
    for pair in 0..20 {
        assert!(pairs.contains(&(pair * 2, pair * 2 + 1)));
    }
}

#[test]
fn rig_local_grouping_keeps_temporal_edges_per_camera_and_adds_same_timestamp_edges() {
    let names = vec![
        "cam4_100.png".to_owned(),
        "cam4_102.png".to_owned(),
        "cam4_110.png".to_owned(),
        "cam5_100.png".to_owned(),
        "cam5_103.png".to_owned(),
        "cam6_100.png".to_owned(),
    ];
    let pairs = rig_local_pairs(&names, 3).unwrap();
    assert_eq!(
        pairs,
        vec![
            (0, 1), // cam4 temporal, difference 2
            (0, 3), // cam4/cam5 same timestamp
            (0, 5), // cam4/cam6 same timestamp
            (3, 4), // cam5 temporal, difference 3
            (3, 5), // cam5/cam6 same timestamp
        ]
    );
    assert!(!pairs.contains(&(1, 3))); // different timestamps/cameras
    assert!(!pairs.contains(&(2, 3))); // outside cam4 temporal window
}

#[test]
fn rig_local_grouping_rejects_duplicate_timestamp_within_one_camera() {
    let names = vec![
        "cam4_100.png".to_owned(),
        "cam4_100.jpg".to_owned(),
        "cam5_100.png".to_owned(),
    ];
    let error = rig_local_pairs(&names, 3).unwrap_err();
    assert!(error.contains("repeats timestamp 100"));
}

#[test]
fn temporal_pyramid_uses_positional_offsets_and_same_timestamp_rig_edges() {
    let names = vec![
        "cam4_100.png".to_owned(),
        "cam4_300.png".to_owned(),
        "cam4_900.png".to_owned(),
        "cam4_1400.png".to_owned(),
        "cam5_100.png".to_owned(),
        "cam5_900.png".to_owned(),
    ];
    let (temporal, cross) = rig_temporal_pyramid_pairs(&names, 2).unwrap();
    // Offset 1 connects adjacent positions even though timestamp gaps
    // are irregular; offset 2 connects the first and third positions.
    assert_eq!(
        temporal,
        vec![(0, 1), (1, 2), (2, 3), (4, 5), (0, 2), (1, 3)]
    );
    assert_eq!(cross, vec![(0, 4), (2, 5)]);
    assert_eq!(temporal_pyramid_offsets_string(64), "1,2,4,8,16,32,64");
}

#[test]
fn temporal_pyramid_rejects_duplicate_timestamp_within_camera_but_allows_rig_duplicate() {
    let names = vec![
        "cam4_100.png".to_owned(),
        "cam4_100.jpg".to_owned(),
        "cam5_100.png".to_owned(),
    ];
    let error = rig_temporal_pyramid_pairs(&names, 32).unwrap_err();
    assert!(error.contains("repeats timestamp 100"));
    let names = vec!["cam4_100.png".to_owned(), "cam5_100.png".to_owned()];
    let (temporal, cross) = rig_temporal_pyramid_pairs(&names, 32).unwrap();
    assert!(temporal.is_empty());
    assert_eq!(cross, vec![(0, 1)]);
}

#[test]
fn temporal_pyramid_samples_long_levels_after_metric_rig_edges() {
    let names = (0..200)
        .map(|index| format!("cam1_{index:06}.png"))
        .chain((0..200).map(|index| format!("cam2_{index:06}.png")))
        .collect::<Vec<_>>();

    let (_, priority_tail) = rig_temporal_pyramid_pairs(&names, 128).unwrap();

    assert_eq!(priority_tail.len(), 200 + 68 + 18);
    assert_eq!(priority_tail[0], (0, 200));
    assert_eq!(priority_tail[199], (199, 399));
    assert!(priority_tail.contains(&(0, 64)));
    assert!(priority_tail.contains(&(0, 128)));
    assert!(priority_tail.contains(&(200, 264)));
    assert!(priority_tail.contains(&(200, 328)));
    assert!(!priority_tail.contains(&(1, 65)));
    assert!(!priority_tail.contains(&(1, 129)));
}

#[test]
fn temporal_pyramid_budget_fills_by_retrieval_score_not_pair_key() {
    let names = (0..4)
        .map(|index| format!("cam1_{index:06}.png"))
        .collect::<Vec<_>>();
    let retrieval = vec![((0, 2), 0.1), ((0, 3), 0.9)];

    let pairs = candidate_pairs_temporal_pyramid_from_retrieval(
        &names,
        retrieval,
        1,
        Some(4),
        None,
        None,
        None,
    )
    .unwrap();

    assert!(pairs.contains(&(0, 3)));
    assert!(!pairs.contains(&(0, 2)));
}

#[test]
fn temporal_pyramid_component_fill_round_robins_component_pairs() {
    let root = std::env::temp_dir().join(format!(
        "visloc_retrieval_component_manifest_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("components.txt");
    std::fs::write(
        &path,
        "# retrieval-component-manifest-v1\n\
             C 0 cam1_000000.png\n\
             C 0 cam1_000001.png\n\
             C 1 cam1_000003.png\n\
             C 1 cam1_000004.png\n\
             C 2 cam1_000005.png\n",
    )
    .unwrap();
    let names = (0..6)
        .map(|index| format!("cam1_{index:06}.png"))
        .collect::<Vec<_>>();
    let retrieval = vec![((0, 3), 0.9), ((1, 4), 0.8), ((0, 5), 0.7)];

    let pairs = candidate_pairs_temporal_pyramid_from_retrieval(
        &names,
        retrieval,
        1,
        Some(7),
        None,
        Some(&path),
        None,
    )
    .unwrap();

    assert!(pairs.contains(&(0, 3)));
    assert!(pairs.contains(&(0, 5)));
    assert!(!pairs.contains(&(1, 4)));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn temporal_pyramid_retrieval_gap_excludes_near_fill_only() {
    let names = (0..6)
        .map(|index| format!("cam1_{index:06}.png"))
        .collect::<Vec<_>>();
    let retrieval = vec![((0, 2), 0.9), ((0, 5), 0.8)];

    let pairs = candidate_pairs_temporal_pyramid_from_retrieval(
        &names,
        retrieval,
        1,
        Some(7),
        None,
        None,
        Some(4),
    )
    .unwrap();

    assert!(pairs.contains(&(0, 1)), "temporal base edge was removed");
    assert!(pairs.contains(&(0, 5)), "long retrieval edge was removed");
    assert!(!pairs.contains(&(0, 2)), "near retrieval edge was retained");
}

#[test]
fn temporal_pyramid_retrieval_gap_uses_explicit_rig_frames() {
    let root = std::env::temp_dir().join(format!(
        "visloc_retrieval_gap_rig_manifest_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("rig.txt");
    std::fs::write(
        &path,
        "# generalized-rig-manifest-v1\n\
             F 0 left_900.png 0\n\
             F 10 left_100.png 0\n\
             F 20 left_500.png 0\n\
             F 0 right_901.png 1\n\
             F 10 right_101.png 1\n\
             F 20 right_501.png 1\n",
    )
    .unwrap();
    let names = [
        "left_900.png",
        "left_100.png",
        "left_500.png",
        "right_901.png",
        "right_101.png",
        "right_501.png",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    let retrieval = vec![((0, 4), 0.9), ((0, 5), 0.8)];

    let pairs = candidate_pairs_temporal_pyramid_from_retrieval(
        &names,
        retrieval,
        1,
        Some(8),
        Some(&path),
        None,
        Some(15),
    )
    .unwrap();

    assert!(pairs.contains(&(0, 5)));
    assert!(!pairs.contains(&(0, 4)));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn temporal_pyramid_manifest_groups_different_camera_aliases_into_frames() {
    let root =
        std::env::temp_dir().join(format!("visloc_rig_frame_manifest_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("rig.txt");
    std::fs::write(
        &path,
        "# generalized-rig-manifest-v1\n\
             S 0 1 10 10 1 1 1 1 1 0 0 0 0 0 0\n\
             S 1 2 10 10 1 1 1 1 1 0 0 0 1 0 0\n\
             F 0 cam1_000000.png 0\n\
             F 0 cam2_000001.png 1\n\
             F 1 cam1_000002.png 0\n\
             F 1 cam2_000003.png 1\n",
    )
    .unwrap();
    let names = vec![
        "cam1_000000.png".to_owned(),
        "cam1_000002.png".to_owned(),
        "cam2_000001.png".to_owned(),
        "cam2_000003.png".to_owned(),
    ];

    let (temporal, cross) =
        rig_temporal_pyramid_pairs_with_manifest(&names, 1, Some(&path)).unwrap();

    assert_eq!(temporal, vec![(0, 1), (2, 3)]);
    assert_eq!(cross, vec![(0, 2), (1, 3)]);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn candidate_manifest_metadata_is_canonical_and_round_trips() {
    let root = std::env::temp_dir().join(format!(
        "visloc_candidate_manifest_metadata_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("pairs.txt");
    let names = vec!["cam4_100.png".to_owned(), "cam5_100.png".to_owned()];
    let mut metadata = BTreeMap::new();
    metadata.insert("pair_source".to_owned(), "vlad-union".to_owned());
    metadata.insert(
        "local_grouping".to_owned(),
        "rig-prefix-timestamp-v1".to_owned(),
    );
    write_candidate_manifest_with_metadata(&path, &names, &[(0, 1)], &metadata).unwrap();
    let (pairs, parsed) = parse_candidate_manifest_with_metadata(&path, &names).unwrap();
    assert_eq!(pairs, vec![(0, 1)]);
    assert_eq!(parsed, metadata);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("metadata local_grouping rig-prefix-timestamp-v1\n"));
    assert!(text.contains("metadata pair_source vlad-union\n"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn diagnose_stems_are_trimmed_and_require_a_value() {
    assert_eq!(
        parse_diagnose_stems(" DSC_0297, ,DSC_0309 ").unwrap(),
        vec!["DSC_0297", "DSC_0309"]
    );
    assert!(parse_diagnose_stems(" , \t").is_err());
}

#[test]
fn per_image_calibration_maps_by_stem_and_rejects_shared_camera_flags() {
    let root = std::env::temp_dir().join(format!(
        "visloc_per_image_calibration_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("cameras.txt"),
        concat!(
            "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n",
            "7 PINHOLE 100 80 50 50 50 40\n",
            "9 PINHOLE 200 160 100 80 100 80\n",
        ),
    )
    .unwrap();
    std::fs::write(
        root.join("images.txt"),
        concat!(
            "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
            "1 1 0 0 0 0 0 0 7 sub/a.jpg\n",
            "0 0 -1\n",
            "2 1 0 0 0 0 0 0 9 b.jpg\n",
            "0 0 -1\n",
        ),
    )
    .unwrap();
    let names = vec!["a.png".to_owned(), "b.png".to_owned()];
    let features = vec![
        FeatureSet::new(vec![Point2::new(50.0, 40.0)], vec![vec![1.0]]).unwrap(),
        FeatureSet::new(vec![Point2::new(100.0, 80.0)], vec![vec![2.0]]).unwrap(),
    ];
    let loaded = load_input_colmap_calibration(&root, &names, &features, None).unwrap();
    assert_eq!(
        loaded
            .native_cameras
            .iter()
            .map(|camera| camera.id)
            .collect::<Vec<_>>(),
        vec![7, 9]
    );
    assert_eq!(
        parse_colmap_image_camera_assignments(
            &std::fs::read_to_string(root.join("images.txt")).unwrap()
        )
        .unwrap()
        .len(),
        2
    );
    let poses = initial_poses_from_colmap_images_txt_with_expected_cameras(
        &root.join("images.txt"),
        &names,
        loaded.rig.reference_camera(),
        Some(&loaded.native_cameras),
    )
    .unwrap();
    assert_eq!(poses.iter().filter(|pose| pose.is_some()).count(), 2);
    let parsed = parse_args_from(vec![
        "--input-colmap-calibration".into(),
        root.display().to_string(),
        "--width".into(),
        "100".into(),
        "--out-colmap".into(),
        "/tmp/per-image-calibration-test".into(),
    ]);
    assert!(parsed.is_err(), "manual scalar intrinsics must be rejected");
    let parsed = parse_args_from(vec![
        "--input-colmap-calibration".into(),
        root.display().to_string(),
        "--out-colmap".into(),
        "/tmp/per-image-calibration-test".into(),
    ])
    .unwrap();
    assert_eq!(parsed.camera.width, 1, "placeholder is replaced in main");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn diagnose_modes_and_indices_are_validated() {
    assert!(validate_diagnose_options(None, &["DSC_0297".into()], &[], None).is_err());
    assert!(
        validate_diagnose_options(Some(Path::new("pairs.csv")), &[], &[(0, 1)], None,).is_err()
    );
    assert!(validate_diagnose_options(None, &[], &[(1, 1)], Some(2)).is_err());
    assert!(validate_diagnose_options(None, &[], &[(0, 2)], Some(2)).is_err());
    assert!(validate_diagnose_options(None, &[], &[(0, 1)], Some(2)).is_ok());
}

#[test]
fn initial_pose_model_is_mapped_by_stem_and_validated_against_camera() {
    let root =
        std::env::temp_dir().join(format!("visloc_initial_pose_model_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("temporary model directory");
    std::fs::write(
        root.join("cameras.txt"),
        "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n0 PINHOLE 1600 1066 879.4 879.4 803.4 532.6\n",
    )
    .expect("camera model");
    std::fs::write(
        root.join("images.txt"),
        concat!(
            "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
            "1 1 0 0 0 0 0 0 0 DSC_0001.png\n",
            "0 0 -1\n",
            "2 1 0 0 0 1 2 3 0 DSC_0002.png\n",
            "0 0 -1\n",
        ),
    )
    .expect("partial image model");
    let camera = Camera::pinhole(0, 1600, 1066, 879.4, 879.4, 803.4, 532.6);
    let names = vec![
        "DSC_0002.png".to_owned(),
        "DSC_0003.png".to_owned(),
        "DSC_0001.png".to_owned(),
    ];
    let poses = initial_poses_from_colmap_images_txt(&root.join("images.txt"), &names, &camera)
        .expect("valid partial model");
    assert!(
        poses[0].is_some(),
        "matching is by image stem, not row order"
    );
    assert!(poses[1].is_none(), "unseeded loaded images stay None");
    assert!(poses[2].is_some());
    assert_eq!(
        poses[0].as_ref().unwrap().world_to_camera.translation,
        Vector3::new(1.0, 2.0, 3.0)
    );
    let unknown = vec!["DSC_0002.png".to_owned(), "DSC_0009.png".to_owned()];
    assert!(
        initial_poses_from_colmap_images_txt(&root.join("images.txt"), &unknown, &camera,).is_err()
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn colmap_track_membership_maps_rows_and_skips_same_image_points() {
    let root = std::env::temp_dir().join(format!(
        "visloc_colmap_track_membership_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("temporary model directory");
    std::fs::write(
        root.join("images.txt"),
        concat!(
            "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
            "1 1 0 0 0 0 0 0 0 DSC_0001.png\n",
            "0 0 -1 1 1 -1 2 2 -1\n",
            "2 1 0 0 0 1 0 0 0 DSC_0002.png\n",
            "0 0 -1 1 1 -1 2 2 -1\n",
            "3 1 0 0 0 2 0 0 0 DSC_0003.png\n",
            "0 0 -1 1 1 -1 2 2 -1\n",
        ),
    )
    .expect("COLMAP image manifest");
    std::fs::write(
        root.join("points3D.txt"),
        concat!(
            "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n",
            "1 0 0 5 1 2 3 0.1 1 0 2 1 3 2\n",
            "2 0 0 5 1 2 3 0.1 1 0 1 1\n",
        ),
    )
    .expect("COLMAP point membership");
    let features = (0..3)
        .map(|_| {
            FeatureSet::new(
                vec![
                    Point2::new(0.0, 0.0),
                    Point2::new(1.0, 1.0),
                    Point2::new(2.0, 2.0),
                ],
                vec![vec![1.0], vec![2.0], vec![3.0]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let names = vec![
        "DSC_0001.png".to_owned(),
        "DSC_0002.png".to_owned(),
        "DSC_0003.png".to_owned(),
    ];
    let membership = parse_colmap_track_membership(&root.join("points3D.txt"), &names, &features)
        .expect("valid observation-only membership");
    assert_eq!(
        membership,
        ColmapTrackMembership {
            tracks: vec![vec![(0, 0), (1, 1), (2, 2)]],
            source_points: 2,
            source_observations: 5,
            retained_observations: 3,
            skipped_conflicting_points: 1,
            skipped_conflicting_observations: 2,
        }
    );

    std::fs::write(
        root.join("points3D_duplicate.txt"),
        concat!(
            "1 0 0 5 1 2 3 0.1 1 0 2 1 3 2\n",
            "2 0 0 5 1 2 3 0.1 1 0 2 1 3 2\n",
        ),
    )
    .expect("duplicate observation membership");
    let error =
        parse_colmap_track_membership(&root.join("points3D_duplicate.txt"), &names, &features)
            .expect_err("an observation cannot belong to two points");
    assert!(error.contains("more than one point"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn pair_stem_window_is_strict_deterministic_and_default_off() {
    let names = vec![
        "DSC_0001.png".to_owned(),
        "DSC_0003.png".to_owned(),
        "DSC_0004.png".to_owned(),
        "DSC_0010.png".to_owned(),
    ];
    let pairs = vec![(0, 1), (0, 2), (1, 2), (2, 3)];
    assert_eq!(
        filter_pairs_by_stem_window(pairs.clone(), &names, Some(2)).unwrap(),
        vec![(0, 1), (1, 2)]
    );
    assert_eq!(
        filter_pairs_by_stem_window(pairs.clone(), &names, None).unwrap(),
        pairs,
        "omitting the flag must preserve the candidate stream exactly"
    );

    let parsed = parse_args_from(minimal_args(&["--pair-stem-window", "3"])).unwrap();
    assert_eq!(parsed.pair_stem_window, Some(3));
    assert!(parse_args_from(minimal_args(&["--pair-stem-window", "0"])).is_err());
    assert!(parse_args_from(minimal_args(&["--pair-stem-window", "NaN"])).is_err());
    assert!(parse_args_from(minimal_args(&["--pair-stem-window"])).is_err());
    assert!(parse_args_from(minimal_args(&[]))
        .unwrap()
        .pair_stem_window
        .is_none());

    assert!(filter_pairs_by_stem_window(
        vec![(0, 1)],
        &["left.png".to_owned(), "right.png".to_owned()],
        Some(2),
    )
    .is_err());
    assert!(filter_pairs_by_stem_window(
        vec![(0, 1)],
        &["DSC_0001.png".to_owned(), "other_0001.png".to_owned()],
        Some(2),
    )
    .is_err());

    let imported = vec![
        ImportedVerifiedPair {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0)],
            config: ConfigurationType::Calibrated,
            essential_matrix: None,
        },
        ImportedVerifiedPair {
            image_i: 0,
            image_j: 3,
            matches: vec![(0, 0)],
            config: ConfigurationType::Calibrated,
            essential_matrix: None,
        },
    ];
    let imported_filtered =
        filter_imported_verified_pairs_by_stem_window(imported, &names, Some(2)).unwrap();
    assert_eq!(imported_filtered.len(), 1);
    assert_eq!(
        (imported_filtered[0].image_i, imported_filtered[0].image_j),
        (0, 1)
    );
}

#[test]
fn sequence_fallback_appends_only_missing_consecutive_candidates() {
    let names = vec![
        "DSC_0001.png".to_owned(),
        "DSC_0002.png".to_owned(),
        "DSC_0003.png".to_owned(),
        "DSC_0005.png".to_owned(),
    ];
    let mut pairs = vec![(0, 1), (0, 3)];
    assert_eq!(
        super::append_consecutive_stem_candidates(&mut pairs, &names).unwrap(),
        1
    );
    assert_eq!(pairs, vec![(0, 1), (0, 3), (1, 2)]);

    let before = pairs.clone();
    assert_eq!(
        super::append_consecutive_stem_candidates(&mut pairs, &names).unwrap(),
        0
    );
    assert_eq!(pairs, before, "the opt-in augmentation is deterministic");
    assert!(super::append_consecutive_stem_candidates(
        &mut Vec::new(),
        &["left.png".to_owned(), "right.png".to_owned()]
    )
    .is_err());
}

#[test]
fn canonical_feature_order_is_deterministic_and_reorders_alternate_bank() {
    let make_features = |reverse: bool| {
        let (keypoints, descriptors) = if reverse {
            (
                vec![Point2::new(10.0, 0.0), Point2::new(20.0, 0.0)],
                vec![vec![10.0], vec![20.0]],
            )
        } else {
            (
                vec![Point2::new(20.0, 0.0), Point2::new(10.0, 0.0)],
                vec![vec![20.0], vec![10.0]],
            )
        };
        let alternate = if reverse {
            vec![vec![100.0], vec![200.0]]
        } else {
            vec![vec![200.0], vec![100.0]]
        };
        (
            vec![FeatureSet::new(keypoints, descriptors).unwrap()],
            vec![Some(alternate)],
        )
    };
    let (mut first, mut first_alternate) = make_features(false);
    let (mut second, mut second_alternate) = make_features(true);
    let first_map = canonicalize_feature_order(&mut first, &mut first_alternate).unwrap();
    let second_map = canonicalize_feature_order(&mut second, &mut second_alternate).unwrap();
    assert_eq!(first, second);
    assert_eq!(first_alternate, second_alternate);
    assert_eq!(first_map[0], vec![1, 0]);
    assert_eq!(second_map[0], vec![0, 1]);
}

#[test]
fn native_keypoint_sidecar_reorders_without_copying_descriptors() {
    let mut native = vec![vec![Point2::new(100.0, 1.0), Point2::new(200.0, 2.0)]];
    remap_feature_keypoints_by_old_to_new(&mut native, &[vec![1, 0]]).unwrap();
    assert_eq!(
        native[0],
        vec![Point2::new(200.0, 2.0), Point2::new(100.0, 1.0)]
    );

    let mut output = vec![FeatureSet::new(
        vec![Point2::new(10.0, 1.0), Point2::new(20.0, 2.0)],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    )
    .unwrap()];
    let descriptor_storage = output[0].descriptors.as_ptr();
    let descriptors = output[0].descriptors.clone();
    replace_feature_keypoints_from_native(&mut output, &[0], &native).unwrap();
    assert_eq!(output[0].keypoints, native[0]);
    assert_eq!(output[0].descriptors, descriptors);
    assert_eq!(output[0].descriptors.as_ptr(), descriptor_storage);
}

#[test]
fn orientation_locus_canonicalization_collapses_variants_and_keeps_best_distance() {
    let features = vec![
        FeatureSet::new(
            vec![
                Point2::new(10.0, 10.0),
                Point2::new(10.0, 10.0),
                Point2::new(20.0, 20.0),
            ],
            vec![vec![0.0], vec![1.0], vec![5.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![
                Point2::new(11.0, 11.0),
                Point2::new(11.0, 11.0),
                Point2::new(21.0, 21.0),
            ],
            vec![vec![1.0], vec![1.0], vec![5.0]],
        )
        .unwrap(),
    ];
    let metadata = vec![
        Some(vec![
            FeatureLocusMetadata {
                x: 10.0,
                y: 10.0,
                scale: 2.0,
                orientation: 0.0,
            },
            FeatureLocusMetadata {
                x: 10.0,
                y: 10.0,
                scale: 2.0,
                orientation: 1.0,
            },
            FeatureLocusMetadata {
                x: 20.0,
                y: 20.0,
                scale: 3.0,
                orientation: 0.0,
            },
        ]),
        Some(vec![
            FeatureLocusMetadata {
                x: 11.0,
                y: 11.0,
                scale: 2.0,
                orientation: 0.0,
            },
            FeatureLocusMetadata {
                x: 11.0,
                y: 11.0,
                scale: 2.0,
                orientation: 1.0,
            },
            FeatureLocusMetadata {
                x: 21.0,
                y: 21.0,
                scale: 3.0,
                orientation: 0.0,
            },
        ]),
    ];
    let mut pair = PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1), (2, 2)]);
    let stats =
        canonicalize_pairwise_loci(&features, &metadata, std::slice::from_mut(&mut pair), None)
            .unwrap();
    assert_eq!(stats.metadata_images, 2);
    assert_eq!(stats.physical_loci, 4);
    assert_eq!(stats.collapsed_rows, 2);
    assert_eq!(stats.input_matches, 3);
    assert_eq!(stats.output_matches, 2);
    assert_eq!(stats.deduplicated_matches, 1);
    assert_eq!(
        pair.matches.len(),
        2,
        "two orientations must not create duplicate endpoint pairs"
    );
    assert_eq!(descriptor_squared_distance(&[1.0], &[1.0]), 0.0);
    assert!(
        match_candidate_cmp(&(1, 1, 0.0, None), &(0, 0, 1.0, None)) == CmpOrdering::Less,
        "the lower descriptor distance must win before stable tie-breaks"
    );
    assert!(
        pair.matches.iter().any(|&(i, j)| (i, j) == (2, 2)),
        "different scales at the same image location remain separate loci"
    );
}

#[test]
fn orientation_locus_canonicalization_is_permutation_invariant_and_default_noop() {
    let make = |reverse: bool| {
        let rows = [
            (Point2::new(10.0, 10.0), vec![0.0], 2.0, 0.0),
            (Point2::new(10.0, 10.0), vec![1.0], 2.0, 1.0),
            (Point2::new(20.0, 20.0), vec![5.0], 3.0, 0.0),
        ];
        let order = if reverse {
            vec![2, 1, 0]
        } else {
            vec![0, 1, 2]
        };
        let keypoints: Vec<Point2<f64>> = order.iter().map(|&i| rows[i].0).collect();
        let descriptors: Vec<Vec<f32>> = order.iter().map(|&i| rows[i].1.clone()).collect();
        let metadata: Vec<FeatureLocusMetadata> = order
            .iter()
            .map(|&i| FeatureLocusMetadata {
                x: rows[i].0.x,
                y: rows[i].0.y,
                scale: rows[i].2,
                orientation: rows[i].3,
            })
            .collect();
        (
            vec![
                FeatureSet::new(keypoints.clone(), descriptors.clone()).unwrap(),
                FeatureSet::new(keypoints, descriptors).unwrap(),
            ],
            vec![Some(metadata.clone()), Some(metadata)],
            PairwiseMatches::new(0, 1, order.iter().map(|&i| (i, i)).collect()),
        )
    };
    let (first_features, first_metadata, mut first_pair) = make(false);
    let (second_features, second_metadata, mut second_pair) = make(true);
    canonicalize_pairwise_loci(
        &first_features,
        &first_metadata,
        std::slice::from_mut(&mut first_pair),
        None,
    )
    .unwrap();
    canonicalize_pairwise_loci(
        &second_features,
        &second_metadata,
        std::slice::from_mut(&mut second_pair),
        None,
    )
    .unwrap();
    let physical = |features: &[FeatureSet], pair: &PairwiseMatches| {
        pair.matches
            .iter()
            .map(|&(i, j)| {
                (
                    features[pair.image_i].keypoints[i],
                    features[pair.image_j].keypoints[j],
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        physical(&first_features, &first_pair),
        physical(&second_features, &second_pair),
        "canonical output must depend on physical loci, not source row order"
    );

    let legacy_features = vec![
        FeatureSet::new(
            vec![Point2::new(1.0, 1.0), Point2::new(1.0, 1.0)],
            vec![vec![0.0], vec![1.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![Point2::new(2.0, 2.0), Point2::new(2.0, 2.0)],
            vec![vec![0.0], vec![1.0]],
        )
        .unwrap(),
    ];
    let legacy_pair = PairwiseMatches::new(0, 1, vec![(1, 1), (0, 0)]);
    let mut unchanged = vec![legacy_pair.clone()];
    let stats =
        canonicalize_pairwise_loci(&legacy_features, &[None, None], &mut unchanged, None).unwrap();
    assert_eq!(unchanged, vec![legacy_pair]);
    assert_eq!(stats.metadata_images, 0);
    assert_eq!(stats.physical_loci, 0);
    assert_eq!(stats.collapsed_rows, 0);
    assert_eq!(stats.input_matches, 2);
    assert_eq!(stats.output_matches, 2);
    assert_eq!(stats.deduplicated_matches, 0);
    assert_eq!(
        stats.changed_pairs, 0,
        "metadata-free files must retain legacy identity"
    );
}

#[test]
fn union_traversal_controls_preserve_edge_multiset_and_default_identity() {
    let original = vec![
        PairwiseMatches::new(2, 0, vec![(4, 5), (3, 2)]),
        PairwiseMatches::new(0, 1, vec![(7, 8)]),
        PairwiseMatches::new(0, 2, vec![(9, 6)]),
    ];
    let hash = unordered_pairwise_edge_hash(&original);
    let mut flat_edges = original
        .iter()
        .flat_map(|pair| {
            let (image_i, image_j, swapped) = if pair.image_i <= pair.image_j {
                (pair.image_i, pair.image_j, false)
            } else {
                (pair.image_j, pair.image_i, true)
            };
            pair.matches.iter().map(move |&(left, right)| {
                let (left, right) = if swapped {
                    (right, left)
                } else {
                    (left, right)
                };
                (image_i, image_j, left, right)
            })
        })
        .collect::<Vec<_>>();
    flat_edges.sort_unstable();
    let flat_hash = flat_edges.into_iter().fold(
        0xcbf29ce484222325u64,
        |mut hash, (image_i, image_j, left, right)| {
            for value in [image_i, image_j, left, right] {
                for byte in (value as u64).to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x100000001b3u64);
                }
            }
            hash
        },
    );
    assert_eq!(hash, flat_hash);

    let mut unchanged = original.clone();
    apply_union_traversal_order(&mut unchanged, UnionTraversalOrder::Original);
    assert_eq!(unchanged, original);

    for order in [
        UnionTraversalOrder::ReversePairs,
        UnionTraversalOrder::ReverseMatches,
        UnionTraversalOrder::ReverseBoth,
    ] {
        let mut reordered = original.clone();
        apply_union_traversal_order(&mut reordered, order);
        assert_eq!(unordered_pairwise_edge_hash(&reordered), hash);
        let original_edges: Vec<_> = original
            .iter()
            .flat_map(|pair| pair.matches.iter().copied())
            .collect();
        let reordered_edges: Vec<_> = reordered
            .iter()
            .flat_map(|pair| pair.matches.iter().copied())
            .collect();
        assert_eq!(
            original_edges.len(),
            reordered_edges.len(),
            "{order:?} must preserve correspondence count"
        );
    }

    let mut reversed_pairs = original.clone();
    apply_union_traversal_order(&mut reversed_pairs, UnionTraversalOrder::ReversePairs);
    assert_eq!(reversed_pairs[0], *original.last().unwrap());
    let mut reversed_matches = original.clone();
    apply_union_traversal_order(&mut reversed_matches, UnionTraversalOrder::ReverseMatches);
    assert_eq!(reversed_matches[0].matches, vec![(3, 2), (4, 5)]);
    assert!(parse_args_from(minimal_args(&["--union-traversal-order", "not-a-mode",])).is_err());
}

#[test]
fn physical_hash_traversal_is_seeded_and_preserves_the_edge_multiset() {
    let features = (0..3)
        .map(|image| {
            let keypoints = (0..10)
                .map(|keypoint| {
                    Point2::new(
                        (image * 100 + keypoint * 7) as f64,
                        (image * 10 + keypoint * 3) as f64,
                    )
                })
                .collect();
            let descriptors = (0..10).map(|value| vec![value as f32]).collect();
            FeatureSet::new(keypoints, descriptors).unwrap()
        })
        .collect::<Vec<_>>();
    let original = vec![
        PairwiseMatches::new(2, 0, vec![(4, 5), (3, 2)]),
        PairwiseMatches::new(0, 1, vec![(7, 8)]),
    ];
    let hash = unordered_pairwise_edge_hash(&original);
    let mut first = original.clone();
    apply_union_traversal_order_with_features(
        &mut first,
        UnionTraversalOrder::PhysicalHash(17),
        &features,
    );
    assert_eq!(unordered_pairwise_edge_hash(&first), hash);
    assert_eq!(
        UnionTraversalOrder::PhysicalHash(17).as_string(),
        "physical-hash:17"
    );
    assert_eq!(
        "physical-hash:0x11".parse::<UnionTraversalOrder>().unwrap(),
        UnionTraversalOrder::PhysicalHash(17)
    );
    assert_eq!(
        "physical-hash-reverse:17"
            .parse::<UnionTraversalOrder>()
            .unwrap(),
        UnionTraversalOrder::PhysicalHashReverse(17)
    );
    let parsed = parse_args_from(minimal_args(&[
        "--union-traversal-order",
        "physical-hash:0x11",
    ]))
    .unwrap();
    assert_eq!(
        parsed.union_traversal_order,
        UnionTraversalOrder::PhysicalHash(17)
    );
    assert!("physical-hash:".parse::<UnionTraversalOrder>().is_err());
    assert!("physical-hash:nope".parse::<UnionTraversalOrder>().is_err());

    let mut second = original.clone();
    apply_union_traversal_order_with_features(
        &mut second,
        UnionTraversalOrder::PhysicalHash(17),
        &features,
    );
    assert_eq!(
        first, second,
        "same physical hash seed must be deterministic"
    );
    let mut different_seed = original.clone();
    apply_union_traversal_order_with_features(
        &mut different_seed,
        UnionTraversalOrder::PhysicalHash(18),
        &features,
    );
    assert_ne!(
        first, different_seed,
        "the seed must parameterize the physical traversal order"
    );
    let mut descending = original.clone();
    apply_union_traversal_order_with_features(
        &mut descending,
        UnionTraversalOrder::PhysicalHashReverse(17),
        &features,
    );
    assert_eq!(unordered_pairwise_edge_hash(&descending), hash);
}

#[test]
fn verified_oracle_map_normalizes_pair_and_keeps_config() {
    let imported = vec![ImportedVerifiedPair {
        image_i: 3,
        image_j: 1,
        matches: vec![(0, 0), (1, 1)],
        config: ConfigurationType::Calibrated,
        essential_matrix: None,
    }];
    let oracle = verified_pair_oracle_map(&imported);
    assert_eq!(
        oracle.get(&(1, 3)),
        Some(&super::VerifiedPairOracle {
            inliers: 2,
            config: ConfigurationType::Calibrated,
        })
    );
}

#[test]
fn model_cross_validation_holdout_is_stable_and_bucket_metrics_are_bounded() {
    let first = model_cross_validation_is_held_out(3, 7, 11, 19);
    assert_eq!(first, model_cross_validation_is_held_out(3, 7, 11, 19));
    assert_ne!(
        first,
        model_cross_validation_is_held_out(3, 7, 11, 20),
        "the deterministic partition should not collapse nearby match keys"
    );
    let pixel_i = Point2::new(12.3456, 78.9012);
    let pixel_j = Point2::new(98.7654, 32.1098);
    assert_eq!(
        model_cross_validation_is_held_out_for_pixels(3, 7, &pixel_i, &pixel_j),
        model_cross_validation_is_held_out_for_pixels(3, 7, &pixel_i, &pixel_j)
    );
    let mut bucket = ModelCrossValidationBucket::default();
    bucket.record(Some(0.01), 0.02, true, true, true);
    bucket.record(Some(0.03), 0.02, true, false, false);
    bucket.record(None, 0.02, false, false, false);
    let summary = summarize_model_cross_validation_bucket(&mut bucket);
    assert_eq!(summary.observations, 3);
    assert_eq!(summary.residual_samples, 2);
    assert_eq!(summary.under_threshold, 1);
    assert_eq!(summary.triangulated, 2);
    assert_eq!(summary.positive_depth, 1);
    assert_eq!(summary.angle_ge_one_degree, 1);
    assert!((summary.under_fraction - 0.5).abs() < 1.0e-12);
    assert!((summary.positive_fraction - 0.5).abs() < 1.0e-12);
    assert!((summary.angle_fraction - 0.5).abs() < 1.0e-12);
}

#[test]
fn model_cross_validation_selection_score_requires_shared_calibrated_references() {
    let mut summary = ModelCrossValidationSummary::default();
    summary.pair_balanced_rotation_disagreement_deg = 2.5;
    assert!(model_cross_validation_selection_score(&summary).is_none());
    summary.rotation_reference_pairs = 3;
    assert_eq!(model_cross_validation_selection_score(&summary), Some(2.5));
    summary.pair_balanced_rotation_disagreement_deg = f64::NAN;
    assert!(model_cross_validation_selection_score(&summary).is_none());
}

#[test]
fn calibrated_reference_direction_uses_cheirality_and_stability_gates() {
    let forward = Vector3::new(1.0, 0.0, 0.0);
    let opposite = Vector3::new(-1.0, 0.0, 0.0);
    assert!(translation_direction_delta_deg(&forward, &forward) < 1.0e-9);
    assert!(translation_direction_delta_deg(&forward, &opposite) > 179.9);

    let quality = EssentialPairQuality {
        best_cheirality: 80,
        second_cheirality: 40,
        cheirality_ratio: 0.8,
        mean_sampson: 0.01,
        rotation_quaternion: [1.0, 0.0, 0.0, 0.0],
        center_direction: [1.0, 0.0, 0.0],
        angle_samples: 80,
        angle_ge_1deg: 70,
        angle_p10_deg: 1.2,
        angle_p25_deg: 2.0,
        angle_median_deg: 4.0,
        depth_ratio_p10: 0.5,
        depth_ratio_p25: 0.7,
        depth_ratio_median: 1.0,
    };
    assert!(imported_reference_quality_is_strong(&quality, 3, 12.0));
    assert!(!imported_reference_quality_is_strong(&quality, 1, 12.0));
    assert!(!imported_reference_quality_is_strong(&quality, 3, 20.0001));
    assert!(!imported_reference_quality_is_strong(&quality, 3, f64::NAN));

    // The Huber location must not be pulled to a single angular outlier;
    // this is the robust pair-balanced statistic used by the diagnostic.
    let robust = robust_huber_mean(&[1.0, 1.1, 0.9, 1.2, 100.0]);
    assert!(robust.is_finite());
    assert!(robust < 5.0, "robust location was {robust}");
}

#[test]
fn geometry_conflict_recovery_flag_is_default_off_and_parseable() {
    let defaults = parse_args_from(minimal_args(&[])).unwrap();
    assert!(!defaults.geometry_guided_conflict_recovery);
    assert!(!defaults.post_refinement_registration);
    assert!(!defaults.sequence_relative_pose_fallback);
    assert!(!defaults.sequence_fallback_after_post);
    assert!(!defaults.sequence_constant_velocity_scale);
    assert!(!defaults.sequence_relaxed_constant_velocity_scale);
    assert!(!defaults.sequence_fallback_carry_scale);
    assert!(!defaults.refine_uncalibrated_f_to_essential);
    assert!(!defaults.strict_uncalibrated_f_to_essential);
    assert!(!defaults.calibrated_essential_primary);
    assert_eq!(defaults.sift_descriptor_magnification, 8.0);
    assert!(!defaults.sift_scale_adaptive_gradients);
    assert!(!defaults.sift_vlfeat_compatible_descriptor);
    assert!(!defaults.sift_vlfeat_compatible_detector);
    assert!(!defaults.sift_vlfeat_bilinear_orientations);
    assert!(!defaults.sift_vlfeat_compatible_output_order);
    assert!(!defaults.sift_colmap_compatible_grayscale);
    assert!(!defaults.sift_split_colmap_detector_grayscale);
    assert_eq!(defaults.sift_append_descriptor_magnification, None);
    assert!(!defaults.sift_standard_orientations);
    assert_eq!(defaults.sift_extra_contrast_threshold, None);
    assert!(!defaults.sift_extra_matches_append_only);
    assert!(!defaults.orientation_locus_canonicalization);
    assert!(!defaults.incremental_correspondence_triangulation);
    assert!(!defaults.confidence_ordered_tracks);
    assert!(!defaults.geometric_confidence_tracks);
    assert!(!defaults.stable_track_order);
    assert!(!defaults.cycle_supported_tracks);
    assert!(!defaults.canonical_feature_order);
    assert_eq!(
        defaults.union_traversal_order,
        UnionTraversalOrder::Original
    );
    assert!(!defaults.geometry_weighted_ba);
    assert!(!defaults.freeze_ill_conditioned_landmarks);
    assert_eq!(defaults.landmark_ba_warm_start_iterations, 0);
    assert_eq!(defaults.landmark_ba_warm_start_min_registered_images, 0);
    assert_eq!(defaults.ba_max_iterations, None);
    assert_eq!(defaults.ba_huber_delta, None);
    assert_eq!(defaults.final_min_track_length, None);
    assert_eq!(
        IncrementalSfmConfig::default().ba_config.robust_kernel,
        RobustKernel::Huber { delta: 3.0 }
    );
    assert_eq!(defaults.periodic_ba_min_registered_images, 0);
    assert_eq!(defaults.final_ba_polish_iterations, 0);
    assert_eq!(defaults.diagnose_ba_oracle_poses_file, None);
    assert_eq!(defaults.diagnose_fixed_rotation_ba, None);
    assert_eq!(defaults.diagnose_model_score_file, None);
    assert_eq!(defaults.initial_poses_file, None);
    assert_eq!(defaults.diagnose_colmap_track_membership, None);
    assert!(!defaults.pose_guided_track_splitting);
    assert!(!defaults.pose_guided_track_splitting_graph_support);
    assert!(!defaults.pose_guided_track_splitting_bridge_cuts);
    assert!(!defaults.pose_guided_track_merging);
    assert_eq!(defaults.pose_guided_merge_max_reproj, None);
    assert_eq!(defaults.pose_guided_split_max_reproj, None);
    assert_eq!(defaults.pose_guided_track_splitting_iterations, None);
    assert_eq!(defaults.seed_pair, None);
    assert_eq!(defaults.component_model_min_images, None);
    assert_eq!(defaults.component_model_max_count, 16);

    let seed_pair = parse_args_from(minimal_args(&["--seed-pair", "9,8"])).unwrap();
    assert_eq!(seed_pair.seed_pair, Some((8, 9)));
    assert!(parse_args_from(minimal_args(&["--seed-pair", "8,8"])).is_err());
    assert!(parse_args_from(minimal_args(&["--seed-pair", "8-9"])).is_err());
    let components = parse_args_from(minimal_args(&[
        "--component-model-min-images",
        "100",
        "--component-model-max-count",
        "7",
    ]))
    .unwrap();
    assert_eq!(components.component_model_min_images, Some(100));
    assert_eq!(components.component_model_max_count, 7);
    assert!(parse_args_from(minimal_args(&["--component-model-min-images", "0"])).is_err());
    assert!(parse_args_from(minimal_args(&[
        "--component-model-min-images",
        "100",
        "--seed-pair",
        "8,9",
    ]))
    .is_err());

    let edge_scales = parse_args_from(minimal_args(&[
        "--mapper",
        "global",
        "--global-independent-edge-scales",
    ]))
    .unwrap();
    assert!(edge_scales.global_independent_edge_scales);
    assert!(parse_args_from(minimal_args(&["--global-independent-edge-scales"])).is_err());

    let enabled = parse_args_from(minimal_args(&[
        "--geometry-guided-conflict-recovery",
        "--post-refinement-registration",
        "--sequence-relative-pose-fallback",
        "--sequence-fallback-after-post",
        "--sequence-constant-velocity-scale",
        "--guided-matching",
        "--colmap-guided-matching",
        "--refine-uncalibrated-f-to-essential",
        "--strict-uncalibrated-f-to-essential",
        "--calibrated-essential-primary",
        "--sift-extra-keypoints-stems",
        "DSC_0299,DSC_0306",
        "--sift-extra-keypoints",
        "2048",
        "--sift-extra-contrast-threshold",
        "0.01",
        "--sift-extra-matches-append-only",
        "--orientation-locus-canonicalization",
        "--incremental-correspondence-triangulation",
        "--sift-descriptor-magnification",
        "3.0",
        "--sift-scale-adaptive-gradients",
        "--sift-standard-orientations",
        "--sift-append-descriptor-magnification",
        "3.0",
        "--confidence-ordered-tracks",
        "--geometric-confidence-tracks",
        "--stable-track-order",
        "--cycle-supported-tracks",
        "--canonical-feature-order",
        "--union-traversal-order",
        "reverse-both",
        "--geometry-weighted-ba",
        "--freeze-ill-conditioned-landmarks",
        "--landmark-ba-warm-start-iterations",
        "3",
        "--landmark-ba-warm-start-min-registered-images",
        "27",
    ]))
    .unwrap();
    assert!(enabled.geometry_guided_conflict_recovery);
    assert!(enabled.post_refinement_registration);
    assert!(enabled.sequence_relative_pose_fallback);
    assert!(enabled.sequence_fallback_after_post);
    assert!(enabled.sequence_constant_velocity_scale);
    assert!(!enabled.sequence_relaxed_constant_velocity_scale);
    assert!(enabled.guided_matching);
    assert!(enabled.colmap_guided_matching);
    assert!(parse_args_from(minimal_args(&["--sequence-constant-velocity-scale",])).is_err());
    let relaxed = parse_args_from(minimal_args(&[
        "--sequence-relative-pose-fallback",
        "--sequence-relaxed-constant-velocity-scale",
    ]))
    .unwrap();
    assert!(relaxed.sequence_relative_pose_fallback);
    assert!(relaxed.sequence_relaxed_constant_velocity_scale);
    assert!(!relaxed.sequence_constant_velocity_scale);
    let carried = parse_args_from(minimal_args(&[
        "--post-refinement-registration",
        "--sequence-relative-pose-fallback",
        "--sequence-fallback-after-post",
        "--sequence-relaxed-constant-velocity-scale",
        "--sequence-fallback-carry-scale",
    ]))
    .unwrap();
    assert!(carried.sequence_fallback_carry_scale);
    assert!(parse_args_from(minimal_args(&[
        "--sequence-relative-pose-fallback",
        "--sequence-fallback-carry-scale",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--post-refinement-registration",
        "--sequence-relative-pose-fallback",
        "--sequence-fallback-after-post",
        "--sequence-fallback-carry-scale",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--sequence-relaxed-constant-velocity-scale",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--sequence-relative-pose-fallback",
        "--sequence-constant-velocity-scale",
        "--sequence-relaxed-constant-velocity-scale",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--sequence-relative-pose-fallback",
        "--sequence-fallback-after-post",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--post-refinement-registration",
        "--sequence-fallback-after-post",
    ]))
    .is_err());
    let composed = parse_args_from(minimal_args(&[
        "--geometry-guided-conflict-recovery",
        "--pose-guided-track-splitting",
        "--pose-guided-split-max-reproj",
        "1.0",
    ]))
    .unwrap();
    assert!(composed.geometry_guided_conflict_recovery);
    assert!(composed.pose_guided_track_splitting);
    assert!(!composed.pose_guided_track_splitting_bridge_cuts);
    assert_eq!(composed.pose_guided_split_max_reproj, Some(1.0));
    let bridge_cuts = parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting",
        "--pose-guided-track-splitting-bridge-cuts",
    ]))
    .unwrap();
    assert!(bridge_cuts.pose_guided_track_splitting_bridge_cuts);
    assert!(parse_args_from(minimal_args(
        &["--pose-guided-track-splitting-bridge-cuts",]
    ))
    .is_err());
    let final_track_gate =
        parse_args_from(minimal_args(&["--final-min-track-length", "3"])).unwrap();
    assert_eq!(final_track_gate.final_min_track_length, Some(3));
    for invalid_length in ["0", "2", "4"] {
        assert!(
            parse_args_from(minimal_args(&["--final-min-track-length", invalid_length])).is_err(),
            "unsupported final track length {invalid_length:?} was accepted"
        );
    }
    assert!(parse_args_from(minimal_args(&[
        "--final-min-track-length",
        "3",
        "--no-final-ba",
    ]))
    .is_err());
    let merging = parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting",
        "--pose-guided-track-merging",
    ]))
    .unwrap();
    assert!(merging.pose_guided_track_merging);
    assert_eq!(merging.pose_guided_merge_max_reproj, None);
    assert!(parse_args_from(minimal_args(&["--pose-guided-track-merging"])).is_err());
    let merging_gate = parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting",
        "--pose-guided-track-merging",
        "--pose-guided-merge-max-reproj",
        "4.0",
    ]))
    .unwrap();
    assert_eq!(merging_gate.pose_guided_merge_max_reproj, Some(4.0));
    assert!(parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting",
        "--pose-guided-merge-max-reproj",
        "4.0",
    ]))
    .is_err());
    for invalid_merge_gate in ["0", "-1", "NaN", "inf"] {
        assert!(
            parse_args_from(minimal_args(&[
                "--pose-guided-track-splitting",
                "--pose-guided-track-merging",
                "--pose-guided-merge-max-reproj",
                invalid_merge_gate,
            ]))
            .is_err(),
            "invalid merge gate {invalid_merge_gate:?} was accepted"
        );
    }
    let pose_split = parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting",
        "--pose-guided-track-splitting-graph-support",
    ]))
    .unwrap();
    assert!(pose_split.pose_guided_track_splitting);
    assert!(pose_split.pose_guided_track_splitting_graph_support);
    let split_gate = parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting",
        "--pose-guided-split-max-reproj",
        "1.0",
    ]))
    .unwrap();
    assert_eq!(split_gate.pose_guided_split_max_reproj, Some(1.0));
    let split_iterations = parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting",
        "--pose-guided-track-splitting-iterations",
        "2",
    ]))
    .unwrap();
    assert_eq!(
        split_iterations.pose_guided_track_splitting_iterations,
        Some(2)
    );
    assert!(parse_args_from(minimal_args(&["--pose-guided-split-max-reproj", "1.0",])).is_err());
    assert!(parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting-iterations",
        "2",
    ]))
    .is_err());
    for invalid_iterations in ["0", "9"] {
        assert!(
            parse_args_from(minimal_args(&[
                "--pose-guided-track-splitting",
                "--pose-guided-track-splitting-iterations",
                invalid_iterations,
            ]))
            .is_err(),
            "invalid pose-guided split iterations {invalid_iterations:?} was accepted"
        );
    }
    for invalid_gate in ["0", "-1", "NaN", "inf"] {
        assert!(
            parse_args_from(minimal_args(&[
                "--pose-guided-track-splitting",
                "--pose-guided-split-max-reproj",
                invalid_gate,
            ]))
            .is_err(),
            "invalid pose-guided split gate {invalid_gate:?} was accepted"
        );
    }
    assert!(parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting-graph-support",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--pose-guided-track-splitting",
        "--colmap-style",
    ]))
    .is_err());
    let membership = parse_args_from(minimal_args(&[
        "--diagnose-colmap-track-membership",
        "/tmp/model/points3D.txt",
    ]))
    .unwrap();
    assert_eq!(
        membership.diagnose_colmap_track_membership,
        Some(PathBuf::from("/tmp/model/points3D.txt"))
    );
    assert!(parse_args_from(minimal_args(&[
        "--diagnose-colmap-track-membership",
        "/tmp/model/points3D.txt",
        "--mapper",
        "global",
    ]))
    .is_err());
    let ba_enabled = parse_args_from(minimal_args(&["--ba-max-iterations", "40"])).unwrap();
    assert_eq!(ba_enabled.ba_max_iterations, Some(40));
    assert!(parse_args_from(minimal_args(&["--ba-max-iterations", "0"])).is_err());
    let global_ba_rounds =
        parse_args_from(minimal_args(&["--global-ba-max-refinements", "0"])).unwrap();
    assert_eq!(global_ba_rounds.global_ba_max_refinements, Some(0));
    let global_ba_rounds_explicit =
        parse_args_from(minimal_args(&["--global-ba-max-refinements", "3"])).unwrap();
    assert_eq!(global_ba_rounds_explicit.global_ba_max_refinements, Some(3));
    for invalid_rounds in ["-1", "not-a-number"] {
        assert!(
            parse_args_from(minimal_args(&[
                "--global-ba-max-refinements",
                invalid_rounds
            ]))
            .is_err(),
            "invalid global BA refinement cap {invalid_rounds:?} was accepted"
        );
    }
    assert!(parse_args_from(minimal_args(&["--global-ba-max-refinements"])).is_err());
    let huber_enabled = parse_args_from(minimal_args(&["--ba-huber-delta", "1.0"])).unwrap();
    assert_eq!(huber_enabled.ba_huber_delta, Some(1.0));
    for invalid_delta in ["0", "-1", "NaN", "inf"] {
        assert!(
            parse_args_from(minimal_args(&["--ba-huber-delta", invalid_delta])).is_err(),
            "invalid Huber delta {invalid_delta:?} was accepted"
        );
    }
    assert!(parse_args_from(minimal_args(&["--ba-huber-delta"])).is_err());
    let sparse_solver = parse_args_from(minimal_args(&["--ba-linear-solver", "sparse"])).unwrap();
    assert_eq!(sparse_solver.ba_linear_solver, Some(LinearSolver::Sparse));
    let dense_solver = parse_args_from(minimal_args(&["--ba-linear-solver", "dense"])).unwrap();
    assert_eq!(dense_solver.ba_linear_solver, Some(LinearSolver::Dense));
    for invalid_solver in ["", "foo", "DENSE"] {
        assert!(
            parse_args_from(minimal_args(&["--ba-linear-solver", invalid_solver])).is_err(),
            "invalid BA linear solver {invalid_solver:?} was accepted"
        );
    }
    assert!(parse_args_from(minimal_args(&["--ba-linear-solver"])).is_err());
    let default_matrix_free = parse_args_from(minimal_args(&[])).unwrap();
    assert!(!default_matrix_free.matrix_free_ba);
    assert!(!BaConfig::default().matrix_free_ba);
    let matrix_free = parse_args_from(minimal_args(&["--matrix-free-ba"])).unwrap();
    assert!(matrix_free.matrix_free_ba);
    let periodic_deferred =
        parse_args_from(minimal_args(&["--periodic-ba-min-registered-images", "32"])).unwrap();
    assert_eq!(periodic_deferred.periodic_ba_min_registered_images, 32);
    let polish_enabled =
        parse_args_from(minimal_args(&["--final-ba-polish-iterations", "10"])).unwrap();
    assert_eq!(polish_enabled.final_ba_polish_iterations, 10);
    let polish_disabled =
        parse_args_from(minimal_args(&["--final-ba-polish-iterations", "0"])).unwrap();
    assert_eq!(polish_disabled.final_ba_polish_iterations, 0);
    let oracle_probe = parse_args_from(minimal_args(&[
        "--diagnose-ba-oracle-poses",
        "/tmp/oracle/images.txt",
    ]))
    .unwrap();
    assert_eq!(
        oracle_probe.diagnose_ba_oracle_poses_file,
        Some(Path::new("/tmp/oracle/images.txt").to_path_buf())
    );
    let fixed_rotation =
        parse_args_from(minimal_args(&["--diagnose-fixed-rotation-ba", "current"])).unwrap();
    assert_eq!(
        fixed_rotation.diagnose_fixed_rotation_ba,
        Some("current".to_owned())
    );
    assert!(parse_args_from(minimal_args(&["--diagnose-fixed-rotation-ba"])).is_err());
    let model_score = parse_args_from(minimal_args(&[
        "--diagnose-model-score",
        "/tmp/model/images.txt",
    ]))
    .unwrap();
    assert_eq!(
        model_score.diagnose_model_score_file,
        Some(Path::new("/tmp/model/images.txt").to_path_buf())
    );
    assert!(parse_args_from(minimal_args(&["--diagnose-model-score"])).is_err());
    let initial_poses = parse_args_from(minimal_args(&[
        "--initial-poses",
        "/tmp/partial-model/images.txt",
    ]))
    .unwrap();
    assert_eq!(
        initial_poses.initial_poses_file,
        Some(Path::new("/tmp/partial-model/images.txt").to_path_buf())
    );
    assert!(parse_args_from(minimal_args(&["--initial-poses"])).is_err());
    assert!(parse_args_from(minimal_args(&["--initial-poses", "",])).is_err());
    assert!(parse_args_from(minimal_args(&[
        "--initial-poses",
        "/tmp/partial-model/images.txt",
        "--seed-pair",
        "8,9",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--initial-poses",
        "/tmp/partial-model/images.txt",
        "--mapper",
        "global",
    ]))
    .is_err());
    assert!(enabled.refine_uncalibrated_f_to_essential);
    assert!(enabled.strict_uncalibrated_f_to_essential);
    assert!(enabled.calibrated_essential_primary);
    assert_eq!(enabled.sift_descriptor_magnification, 3.0);
    assert!(enabled.sift_scale_adaptive_gradients);
    assert_eq!(enabled.sift_append_descriptor_magnification, Some(3.0));
    assert!(enabled.sift_standard_orientations);
    assert!(enabled.confidence_ordered_tracks);
    assert!(enabled.geometric_confidence_tracks);
    assert!(enabled.stable_track_order);
    assert!(enabled.cycle_supported_tracks);
    assert!(enabled.canonical_feature_order);
    assert_eq!(
        enabled.union_traversal_order,
        UnionTraversalOrder::ReverseBoth
    );
    assert!(enabled.geometry_weighted_ba);
    assert!(enabled.freeze_ill_conditioned_landmarks);
    assert_eq!(enabled.landmark_ba_warm_start_iterations, 3);
    assert_eq!(enabled.landmark_ba_warm_start_min_registered_images, 27);
    assert_eq!(
        enabled.sift_extra_keypoints_stems,
        vec!["DSC_0299".to_owned(), "DSC_0306".to_owned()]
    );
    assert_eq!(enabled.sift_extra_keypoints, 2048);
    assert_eq!(enabled.sift_extra_contrast_threshold, Some(0.01));
    assert!(enabled.sift_extra_matches_append_only);
    assert!(enabled.orientation_locus_canonicalization);
    assert!(enabled.incremental_correspondence_triangulation);
    assert!(parse_args_from(minimal_args(&[
        "--incremental-correspondence-triangulation",
        "--colmap-style",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--incremental-correspondence-triangulation",
        "--mapper",
        "global",
    ]))
    .is_err());
    let dsp = parse_args_from(minimal_args(&[
        "--sift-dsp",
        "--sift-vlfeat-compatible-descriptor",
    ]))
    .unwrap();
    assert!(dsp.sift_dsp);
    assert_eq!(dsp.sift_dsp_num_scales, 15);
    assert!(parse_args_from(minimal_args(&["--sift-dsp"])).is_err());
    assert!(parse_args_from(minimal_args(&[
        "--sift-dsp",
        "--sift-vlfeat-compatible-descriptor",
        "--sift-dsp-num-scales",
        "0",
    ]))
    .is_err());
    let rounded = parse_args_from(minimal_args(&["--sift-colmap-compatible-grayscale"])).unwrap();
    assert!(rounded.sift_colmap_compatible_grayscale);
    let split = parse_args_from(minimal_args(&[
        "--sift-vlfeat-compatible-detector",
        "--sift-vlfeat-compatible-descriptor",
        "--sift-split-colmap-detector-grayscale",
    ]))
    .unwrap();
    assert!(split.sift_split_colmap_detector_grayscale);
    assert!(parse_args_from(minimal_args(&["--sift-split-colmap-detector-grayscale",])).is_err());
    assert!(parse_args_from(minimal_args(&[
        "--sift-vlfeat-compatible-detector",
        "--sift-vlfeat-compatible-descriptor",
        "--sift-colmap-compatible-grayscale",
        "--sift-split-colmap-detector-grayscale",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&["--sift-extra-contrast-threshold", "-0.01",])).is_err());
    assert!(parse_args_from(minimal_args(&["--sift-extra-contrast-threshold", "NaN",])).is_err());
    assert!(parse_args_from(minimal_args(&["--sift-descriptor-magnification", "0",])).is_err());
    assert!(parse_args_from(minimal_args(&["--sift-descriptor-magnification", "NaN",])).is_err());
    assert!(parse_args_from(minimal_args(&[
        "--sift-append-descriptor-magnification",
        "0",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--sift-append-descriptor-magnification",
        "NaN",
    ]))
    .is_err());
    assert!(
        parse_args_from(minimal_args(&[
            "--sift-vlfeat-compatible-descriptor",
            "--sift-descriptor-magnification",
            "3.0",
        ]))
        .unwrap()
        .sift_vlfeat_compatible_descriptor
    );
    assert!(parse_args_from(minimal_args(&[
        "--sift-vlfeat-compatible-descriptor",
        "--sift-scale-adaptive-gradients",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--sift-vlfeat-compatible-descriptor",
        "--sift-descriptor-magnification",
        "8.0",
    ]))
    .is_err());
    assert!(
        parse_args_from(minimal_args(&["--sift-vlfeat-compatible-detector"]))
            .unwrap()
            .sift_vlfeat_compatible_detector
    );
    let source_order = parse_args_from(minimal_args(&[
        "--sift-vlfeat-compatible-detector",
        "--sift-vlfeat-compatible-output-order",
    ]))
    .unwrap();
    assert!(source_order.sift_vlfeat_compatible_output_order);
    let bilinear = parse_args_from(minimal_args(&[
        "--sift-vlfeat-compatible-detector",
        "--sift-vlfeat-bilinear-orientations",
    ]))
    .unwrap();
    assert!(bilinear.sift_vlfeat_bilinear_orientations);
    assert!(parse_args_from(minimal_args(&[
        "--sift-vlfeat-compatible-detector",
        "--sift-affine",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&[
        "--sift-vlfeat-compatible-detector",
        "--sift-standard-orientations",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&["--sift-vlfeat-bilinear-orientations"])).is_err());
    assert!(parse_args_from(minimal_args(&["--sift-vlfeat-compatible-output-order"])).is_err());
    assert!(parse_args_from(minimal_args(&["--colmap-guided-matching"])).is_err());
}

#[test]
fn component_models_rank_large_components_deterministically() {
    let pairwise = vec![
        PairwiseMatches::new(4, 3, vec![(0, 0)]),
        PairwiseMatches::new(1, 2, vec![(0, 0)]),
        PairwiseMatches::new(0, 1, vec![(0, 0)]),
        PairwiseMatches::new(7, 8, vec![(0, 0)]),
    ];
    assert_eq!(
        ranked_view_graph_components(9, &pairwise, 2, 2),
        vec![vec![0, 1, 2], vec![3, 4]],
    );
    assert_eq!(
        ranked_view_graph_components(9, &pairwise, 2, 1),
        vec![vec![0, 1, 2]],
    );
}

#[test]
fn verified_pair_snapshot_flags_are_explicit_and_mutually_exclusive() {
    let exported = parse_args_from(minimal_args(&[
        "--export-verified-pairs-snapshot",
        "/tmp/pairs.vps",
    ]))
    .unwrap();
    assert_eq!(
        exported.export_verified_pairs_snapshot,
        Some(PathBuf::from("/tmp/pairs.vps"))
    );
    let imported = parse_args_from(minimal_args(&[
        "--import-verified-pairs-snapshot",
        "/tmp/pairs.vps",
    ]))
    .unwrap();
    assert_eq!(
        imported.import_verified_pairs_snapshot,
        Some(PathBuf::from("/tmp/pairs.vps"))
    );
    assert!(parse_args_from(minimal_args(&[
        "--import-verified-pairs-file",
        "/tmp/legacy.txt",
        "--import-verified-pairs-snapshot",
        "/tmp/pairs.vps",
    ]))
    .is_err());
    assert!(parse_args_from(minimal_args(&["--import-verified-pairs-snapshot", "",])).is_err());
    let export_only = parse_args_from(minimal_args(&[
        "--export-verified-pairs-snapshot",
        "/tmp/pairs.vps",
        "--export-verified-pairs-only",
    ]))
    .unwrap();
    assert!(export_only.export_verified_pairs_only);
    assert!(parse_args_from(minimal_args(&["--export-verified-pairs-only"])).is_err());
}

#[test]
fn snapshot_keypoints_only_is_opt_in_and_rejects_unsafe_combinations() {
    let defaults = parse_args_from(minimal_args(&[])).unwrap();
    assert!(!defaults.snapshot_keypoints_only);
    let enabled = parse_args_from(minimal_args(&[
        "--import-verified-pairs-snapshot",
        "/tmp/pairs.vps",
        "--snapshot-keypoints-only",
    ]))
    .unwrap();
    assert!(enabled.snapshot_keypoints_only);

    for extra in [
        vec!["--snapshot-keypoints-only", "--feature-extractor", "sift"],
        vec!["--snapshot-keypoints-only", "--mapper", "global"],
        vec!["--snapshot-keypoints-only", "--mapper", "hybrid"],
        vec!["--snapshot-keypoints-only", "--colmap-style"],
        vec![
            "--snapshot-keypoints-only",
            "--snapshot-coordinate-override-dir",
            "/tmp/override",
        ],
        vec![
            "--snapshot-keypoints-only",
            "--export-features-dir",
            "/tmp/features",
        ],
        vec!["--snapshot-keypoints-only", "--export-features-only"],
        vec![
            "--snapshot-keypoints-only",
            "--export-verified-pairs-snapshot",
            "/tmp/other.vps",
        ],
        vec!["--snapshot-keypoints-only", "--canonical-feature-order"],
        vec![
            "--snapshot-keypoints-only",
            "--orientation-locus-canonicalization",
        ],
        vec![
            "--snapshot-keypoints-only",
            "--diagnose-model-score",
            "/tmp/model/images.txt",
        ],
        vec!["--snapshot-keypoints-only", "--stable-track-order"],
    ] {
        let mut args = vec!["--import-verified-pairs-snapshot", "/tmp/pairs.vps"];
        args.extend(extra);
        assert!(
            parse_args_from(minimal_args(&args)).is_err(),
            "unsafe snapshot-keypoints-only combination was accepted: {args:?}"
        );
    }
}

#[test]
fn snapshot_keypoints_only_loader_matches_full_feature_hash_and_geometry_shape() {
    let root = std::env::temp_dir().join(format!(
        "visloc_snapshot_keypoints_only_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("a_features.txt"),
        "# feature file\n1.0 2.0 0.9 0.0 -0.0 0.25\n3.0 4.0 0.8 1.0 2.0 3.0\n",
    )
    .unwrap();
    std::fs::write(root.join("b_features.txt"), "5.0 6.0 0.7 -1.0 4.0 8.0\n").unwrap();

    let (full, names, _) = load_images(&root, "_features.txt", ".png").unwrap();
    let loaded = load_images_keypoints_only(&root, "_features.txt", ".png").unwrap();
    assert_eq!(loaded.image_names, names);
    assert_eq!(loaded.features.len(), full.len());
    assert!(loaded
        .features
        .iter()
        .all(|set| set.keypoints.len() == set.descriptors.len()
            && set.descriptors.iter().all(Vec::is_empty)));
    let validation = snapshot_feature_validation_from_files(
        &loaded.paths,
        &loaded.features,
        &loaded.fingerprints,
    )
    .unwrap();
    assert_eq!(
        validation.feature_counts,
        full.iter().map(FeatureSet::len).collect::<Vec<_>>()
    );
    assert_eq!(
        validation.feature_manifest_hash,
        snapshot_feature_manifest_hash(&full)
    );

    // Per-image calibration changes only the in-memory keypoints; the
    // descriptor re-read must still produce the same exact stream as a
    // full feature bank carrying those transformed keypoints.
    let mut calibrated = loaded.features;
    calibrated[0].keypoints[0].x += 0.125;
    let mut full_calibrated = full.clone();
    full_calibrated[0].keypoints[0].x += 0.125;
    assert_eq!(
        snapshot_feature_validation_from_files(&loaded.paths, &calibrated, &loaded.fingerprints,)
            .unwrap()
            .feature_manifest_hash,
        snapshot_feature_manifest_hash(&full_calibrated)
    );

    // Empty descriptor rows remain a valid feature shape for the camera
    // rig, preserving the mapper's row-index geometry contract.
    let rig = PerImageCameras::new(vec![
        Camera::pinhole(0, 100, 100, 50.0, 50.0, 50.0, 50.0),
        Camera::pinhole(1, 100, 100, 50.0, 50.0, 50.0, 50.0),
    ])
    .unwrap();
    rig.validate_features(&calibrated).unwrap();

    std::fs::write(
        root.join("a_features.txt"),
        "# feature file\n1.0 2.0 0.9 9.0 -0.0 0.25\n3.0 4.0 0.8 1.0 2.0 3.0\n",
    )
    .unwrap();
    let error =
        snapshot_feature_validation_from_files(&loaded.paths, &calibrated, &loaded.fingerprints)
            .unwrap_err();
    assert!(error.contains("changed between loads"));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn snapshot_coordinate_override_requires_import_and_preserves_default() {
    let default_args = parse_args_from(minimal_args(&[])).unwrap();
    assert!(default_args.snapshot_coordinate_override_dir.is_none());
    assert!(parse_args_from(minimal_args(&[
        "--snapshot-coordinate-override-dir",
        "/tmp/subpixel",
    ]))
    .is_err());
    let args = parse_args_from(minimal_args(&[
        "--import-verified-pairs-snapshot",
        "/tmp/pairs.vps",
        "--snapshot-coordinate-override-dir",
        "/tmp/subpixel",
    ]))
    .unwrap();
    assert_eq!(
        args.snapshot_coordinate_override_dir,
        Some(PathBuf::from("/tmp/subpixel"))
    );
    assert!(parse_args_from(minimal_args(&[
        "--import-verified-pairs-snapshot",
        "/tmp/pairs.vps",
        "--snapshot-coordinate-override-dir",
        "/tmp/subpixel",
        "--export-verified-pairs-snapshot",
        "/tmp/other.vps",
    ]))
    .is_err());
}

#[test]
fn snapshot_coordinate_override_is_descriptor_exact_and_coordinate_only() {
    let names = vec!["left.png".to_owned(), "right.png".to_owned()];
    let mut base = vec![
        FeatureSet::new(
            vec![Point2::new(1.0, 2.0), Point2::new(3.0, 4.0)],
            vec![vec![1.0, -0.0], vec![f32::from_bits(0x7fc0_0042), 5.0]],
        )
        .unwrap(),
        FeatureSet::new(vec![Point2::new(5.0, 6.0)], vec![vec![7.0, 8.0]]).unwrap(),
    ];
    let replacement = vec![
        FeatureSet::new(
            vec![Point2::new(1.25, 2.5), Point2::new(3.0, 4.0)],
            vec![vec![1.0, -0.0], vec![f32::from_bits(0x7fc0_0042), 5.0]],
        )
        .unwrap(),
        FeatureSet::new(vec![Point2::new(5.0, 6.5)], vec![vec![7.0, 8.0]]).unwrap(),
    ];
    let before_descriptor_bits = base
        .iter()
        .map(|features| {
            features
                .descriptors
                .iter()
                .map(|descriptor| descriptor.iter().map(|value| value.to_bits()).collect())
                .collect::<Vec<Vec<u32>>>()
        })
        .collect::<Vec<_>>();
    let descriptor_bits = |sets: &[FeatureSet]| {
        sets.iter()
            .map(|features| {
                features
                    .descriptors
                    .iter()
                    .map(|descriptor| descriptor.iter().map(|value| value.to_bits()).collect())
                    .collect::<Vec<Vec<u32>>>()
            })
            .collect::<Vec<_>>()
    };
    let stats =
        apply_snapshot_coordinate_override(&mut base, &names, &replacement, &names).unwrap();
    assert_eq!(stats.images, 2);
    assert_eq!(stats.rows, 3);
    assert_eq!(stats.changed_rows, 2);
    assert_eq!(base[0].keypoints, replacement[0].keypoints);
    assert_eq!(base[1].keypoints, replacement[1].keypoints);
    assert_eq!(descriptor_bits(&base), before_descriptor_bits);

    let mut changed_descriptor = replacement.clone();
    changed_descriptor[0].descriptors[0][0] = 1.5;
    let error =
        apply_snapshot_coordinate_override(&mut base.clone(), &names, &changed_descriptor, &names)
            .unwrap_err();
    assert!(error.contains("descriptor/index mismatch"));
    let error = apply_snapshot_coordinate_override(
        &mut base.clone(),
        &names,
        &replacement,
        &["right.png".to_owned(), "left.png".to_owned()],
    )
    .unwrap_err();
    assert!(error.contains("image names/order"));
}

fn guided_report(
    config: ConfigurationType,
    essential: Option<Matrix3<f64>>,
    fundamental: Option<Matrix3<f64>>,
    homography: Option<Matrix3<f64>>,
) -> TwoViewGeometryReport {
    TwoViewGeometryReport {
        config,
        inliers: vec![0],
        essential,
        fundamental,
        homography,
        relative_pose: None,
        essential_inliers: vec![0],
        e_inlier_count: 1,
        f_inlier_count: 1,
        h_inlier_count: 1,
    }
}

fn guided_camera() -> Camera {
    Camera::pinhole(0, 640, 480, 100.0, 100.0, 0.0, 0.0)
}

#[test]
fn colmap_guided_geometry_uses_reported_e_f_or_h_model() {
    let identity = Matrix3::identity();
    assert!(matches!(
        colmap_guided_geometry(&guided_report(
            ConfigurationType::Calibrated,
            Some(identity),
            None,
            None,
        )),
        Some(super::ColmapGuidedGeometry::Essential(_))
    ));
    assert!(matches!(
        colmap_guided_geometry(&guided_report(
            ConfigurationType::Uncalibrated,
            None,
            Some(identity),
            None,
        )),
        Some(super::ColmapGuidedGeometry::Fundamental(_))
    ));
    assert!(matches!(
        colmap_guided_geometry(&guided_report(
            ConfigurationType::Planar,
            None,
            None,
            Some(identity),
        )),
        Some(super::ColmapGuidedGeometry::Homography(_))
    ));
    assert!(colmap_guided_geometry(&guided_report(
        ConfigurationType::Multiple,
        Some(identity),
        Some(identity),
        Some(identity),
    ))
    .is_none());
}

#[test]
fn colmap_guided_epipolar_gate_admits_only_geometrically_valid_candidate() {
    // This F has horizontal epipolar lines: equal y coordinates have zero
    // algebraic/Sampson residual, while the distractor is outside the
    // pixel-unit guided gate.  Its descriptor is otherwise equally good.
    let fundamental = Matrix3::new(
        0.0, 0.0, 0.0, //
        0.0, 0.0, -1.0, //
        0.0, 1.0, 0.0,
    );
    let features_i = FeatureSet::new(vec![Point2::new(10.0, 20.0)], vec![vec![1.0, 0.0]]).unwrap();
    let features_j = FeatureSet::new(
        vec![Point2::new(30.0, 20.0), Point2::new(30.0, 24.0)],
        vec![vec![1.0, 0.0], vec![1.0, 0.0]],
    )
    .unwrap();
    let extras = colmap_guided_matches(
        &guided_camera(),
        &features_i,
        &features_j,
        &[],
        &guided_report(
            ConfigurationType::Uncalibrated,
            None,
            Some(fundamental),
            None,
        ),
        1.0,
        0.9,
        true,
    );
    assert_eq!(
        extras
            .iter()
            .map(|m| (m.query_index, m.train_index))
            .collect::<Vec<_>>(),
        vec![(0, 0)]
    );
}

#[test]
fn colmap_guided_cross_check_is_unique_and_initial_matches_are_preserved() {
    let features_i = FeatureSet::new(
        vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)],
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    )
    .unwrap();
    let features_j = FeatureSet::new(
        vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)],
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    )
    .unwrap();
    let initial = vec![DescriptorMatch {
        query_index: 0,
        train_index: 0,
        distance: 0.0,
        second_best_distance: None,
        ratio: None,
        confidence: None,
    }];
    let before = initial.clone();
    let extras = colmap_guided_matches(
        &guided_camera(),
        &features_i,
        &features_j,
        &initial,
        &guided_report(
            ConfigurationType::Planar,
            None,
            None,
            Some(Matrix3::identity()),
        ),
        1.0,
        0.9,
        true,
    );
    // Query 1 has a distinct descriptor/geometry-consistent endpoint, so
    // the append-only helper may add it while retaining the baseline.
    assert_eq!(
        extras
            .iter()
            .map(|m| (m.query_index, m.train_index))
            .collect::<Vec<_>>(),
        vec![(1, 1)]
    );
    assert_eq!(initial, before);
    let mut expanded = initial.clone();
    expanded.extend(extras);
    assert_eq!(expanded.len(), 2);
    assert_eq!((expanded[0].query_index, expanded[0].train_index), (0, 0));
}

#[test]
fn effective_config_snapshot_is_stable_and_experimental_defaults_are_off() {
    let first = parse_args_from(minimal_args(&[])).unwrap();
    let second = parse_args_from(minimal_args(&[])).unwrap();
    let first_snapshot = effective_config_snapshot(&first);
    let second_snapshot = effective_config_snapshot(&second);
    assert_eq!(first_snapshot, second_snapshot);
    assert_eq!(
        effective_config_hash(&first_snapshot),
        effective_config_hash(&second_snapshot)
    );
    assert_eq!(
        effective_config_hash(""),
        0xcbf29ce484222325,
        "the snapshot label uses the stable FNV-1a offset basis"
    );
    let mut path_a = parse_args_from(minimal_args(&[])).unwrap();
    let mut path_b = parse_args_from(minimal_args(&[])).unwrap();
    path_a.out_colmap = PathBuf::from("/tmp/electro-repeat-a/model");
    path_b.out_colmap = PathBuf::from("/tmp/electro-repeat-b/model");
    assert_ne!(
        effective_config_snapshot(&path_a),
        effective_config_snapshot(&path_b)
    );
    assert_eq!(
        snapshot_export_config(&path_a),
        snapshot_export_config(&path_b)
    );

    // Every experimental mapper/verification switch is opt-in. Values
    // such as the ordinary matcher, mapper, and final BA are checked here
    // too so an omitted flag cannot silently select a newer path.
    assert!(!first.refine_intrinsics);
    assert!(!first.refine_distortion);
    assert!(!first.refine_tangential_distortion);
    assert!(!first.shared_focal);
    assert!(!first.colmap_style);
    assert!(!first.final_iterative_global_refinement);
    assert_eq!(first.global_ba_max_refinements, None);
    assert!(!first.post_refinement_registration);
    assert!(!first.structureless_registration);
    assert!(!first.guided_matching);
    assert!(!first.colmap_guided_matching);
    assert!(!first.multiple_models);
    assert!(first.min_e_f_inlier_ratio.is_none());
    assert!(!first.calibrated_prefer_essential);
    assert!(!first.refine_uncalibrated_f_to_essential);
    assert!(!first.strict_uncalibrated_f_to_essential);
    assert!(!first.calibrated_essential_primary);
    assert!(!first.prefer_essential_inliers);
    assert!(!first.prefer_essential_free_endpoints);
    assert!(first.prefer_essential_stems.is_empty());
    assert!(first.prefer_essential_pairs.is_empty());
    assert!(!first.require_essential_selected_edges);
    assert!(first.require_essential_stems.is_empty());
    assert!(first.rematch_stems.is_empty());
    assert!(!first.rematch_guided);
    assert!(!first.rematch_free_vs_priors);
    assert!(!first.rematch_tracks_use_essential);
    assert_eq!(first.rematch_prefer_min_e_inliers, 0);
    assert!(first.rematch_prefer_strong_stems.is_empty());
    assert!((first.rematch_min_chirality_margin - 0.0).abs() < f64::EPSILON);
    assert!(!first.rematch_prior_anchor);
    assert!(first.rematch_min_e_f_inlier_ratio.is_none());
    assert!(!first.rematch_calibrated_prefer_essential);
    assert!(!first.rematch_prior_ray_guided);
    assert_eq!(first.rematch_prior_ray_min_rays, 2);
    assert_eq!(first.rematch_prior_ray_min_e_inliers, 25);
    assert!(first.rematch_verification_mode.is_none());
    assert!(!first.rematch_pose_guided_after_global);
    assert!((first.rematch_max_gt_bearing_deg - 0.0).abs() < f64::EPSILON);
    assert!(first.rematch_gt_bearing_path.is_none());
    assert!(first.rematch_guided_max_error_px.is_none());
    assert!(first.rematch_guided_lowe_ratio.is_none());
    assert!(!first.rematch_require_calibrated);
    assert!((first.rematch_max_mean_sampson - 0.0).abs() < f64::EPSILON);
    assert!((first.essential_edge_weight_boost - 1.0).abs() < f64::EPSILON);
    assert!(!first.force_essential_matches);
    assert!((first.force_essential_min_ef_ratio - 0.7).abs() < f64::EPSILON);
    assert_eq!(first.force_essential_min_e_inliers, 0);
    assert!(!first.force_essential_uncalibrated_only);
    assert!(!first.repnp_free_from_priors);
    assert_eq!(first.repnp_free_min_corrs, 0);
    assert!(!first.repnp_seed_free_as_priors);
    assert!(!first.repair_prior_edges);
    assert!(!first.repair_free_edges_from_solved);
    assert!(!first.drop_free_edges_antipodal);
    assert!(!first.prior_guided_free_chirality);
    assert!(!first.metric_prior_chirality_edges);
    assert!(!first.metric_prior_scale);
    assert!(!first.chirality_harden);
    assert!(!first.refine_global_translations);
    assert!(!first.global_independent_edge_scales);
    assert!(!first.multi_hypothesis_edges);
    assert!(!first.weight_by_chirality_margin);
    assert!(!first.hybrid_filter_priors);
    assert!(!first.hybrid_drop_inconsistent_priors);
    assert!(!first.verify_registration_two_view);
    assert!(!first.hybrid_rotation_priors_only);
    assert!(!first.joint_global_positioning);
    assert!(!first.calibrated_view_edges_only);
    assert!(!first.filter_images);
    assert!(!first.confidence_ordered_tracks);
    assert!(!first.geometric_confidence_tracks);
    assert!(!first.stable_track_order);
    assert!(!first.cycle_supported_tracks);
    assert!(!first.canonical_feature_order);
    assert_eq!(first.union_traversal_order, UnionTraversalOrder::Original);
    assert!(!first.geometry_guided_conflict_recovery);
    assert!(!first.rescue_bridging);
    assert!(!first.rescue_cross_check);
    assert!(first.diagnose_pairs.is_empty());
    assert!(first.diagnose_pairs_csv.is_none());
    assert!(first.diagnose_pair_stems.is_empty());
    assert!(matches!(first.matcher, super::MatcherKind::Nn));
    assert!(matches!(first.mapper, super::MapperKind::Incremental));
    assert!(matches!(first.pair_source, super::PairSource::Vlad));
    assert!(matches!(first.track_source, super::TrackSource::UnionFind));
    assert!(!first.export_features_only);
    assert!(!first.sift_stream_export);
    assert!(!first.sift_stream_resume);
    assert!(first.import_matches_file.is_none());
    assert!(first.import_matches_supplement_file.is_none());
    assert!(!first.sift_stream_export);
    assert!(first.import_verified_pairs_file.is_none());
    assert!(first.export_verified_pairs_snapshot.is_none());
    assert!(first.import_verified_pairs_snapshot.is_none());
    assert!(first.snapshot_coordinate_override_dir.is_none());
    assert!(first.diagnose_ba_oracle_poses_file.is_none());
    assert!(first.diagnose_fixed_rotation_ba.is_none());
    assert!(first.ba_max_iterations.is_none());
    assert_eq!(first.periodic_ba_min_registered_images, 0);
    assert!(first.pair_stem_window.is_none());
    assert_eq!(first.final_ba_polish_iterations, 0);
    assert!(!first.geometry_weighted_ba);
    assert!(!first.freeze_ill_conditioned_landmarks);
    assert_eq!(first.landmark_ba_warm_start_iterations, 0);
    assert_eq!(first.landmark_ba_warm_start_min_registered_images, 0);
    assert!(!first.sift_affine);
    assert!(!first.sift_multi_anisotropy);
    assert!(!first.sift_dsp);
    assert!(!first.sift_l1_root);
    assert!(!first.sift_standard_orientations);
    assert!(!first.sift_prefer_larger_scale);
    assert!(!first.sift_full_pyramid);
    assert!(!first.sift_scale_adaptive_gradients);
    assert!(!first.sift_vlfeat_compatible_descriptor);
    assert!(!first.sift_vlfeat_compatible_detector);
    assert!(!first.sift_vlfeat_bilinear_orientations);
    assert!(!first.sift_vlfeat_compatible_output_order);
    assert!(!first.sift_colmap_compatible_grayscale);
    assert!(!first.sift_split_colmap_detector_grayscale);
    assert!(first.sift_append_descriptor_magnification.is_none());
    assert!(first.sift_extra_keypoints_stems.is_empty());
    assert_eq!(first.sift_extra_keypoints, 0);
    assert!(first.sift_extra_contrast_threshold.is_none());
    assert!(!first.sift_extra_matches_append_only);

    let stream = parse_args_from(minimal_args(&[
        "--feature-extractor",
        "sift",
        "--images-dir",
        "/tmp/sift-images",
        "--export-features-dir",
        "/tmp/sift-features",
        "--export-features-only",
        "--sift-stream-export",
        "--sift-stream-resume",
    ]))
    .unwrap();
    assert!(stream.sift_stream_export);
    assert!(stream.sift_stream_resume);
    assert!(parse_args_from(minimal_args(&["--sift-stream-resume"])).is_err());
    assert!(parse_args_from(minimal_args(&["--sift-stream-export"])).is_err());
    assert!(parse_args_from(minimal_args(&[
        "--export-features-dir",
        "/tmp/sift-features",
        "--export-features-only",
        "--sift-stream-export",
    ]))
    .is_err());
}
