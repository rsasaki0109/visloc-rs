use super::*;
use nalgebra::{UnitQuaternion, Vector3};

#[test]
fn sfm_debug_image_filter_parses_trimmed_unique_indices() {
    let expected: HashSet<usize> = [20, 21].into_iter().collect();
    assert_eq!(parse_sfm_debug_images(" 20, 21,20 ").unwrap(), expected);
    assert!(parse_sfm_debug_images("20,nope").is_err());
    assert!(parse_sfm_debug_images(" , \t").is_err());
}

#[test]
fn periodic_ba_schedule_default_identity_and_deferred_boundary() {
    let config = IncrementalSfmConfig::default();
    assert_eq!(config.periodic_ba_min_registered_images, 0);
    for registered in [5, 27, 32, 38] {
        assert!(
            periodic_ba_due(config.ba_every, 0, config.ba_every, registered),
            "minimum=0 must retain the historical schedule"
        );
    }

    // The observed champion growth has five registrations at the first
    // basin-jump boundary (27 cameras), followed by the next periodic
    // boundary at 32. The evidence-derived minimum defers only the former.
    assert!(!periodic_ba_due(5, 32, 5, 27));
    assert!(periodic_ba_due(5, 32, 10, 32));
    assert!(!periodic_ba_due(0, 32, 10, 32));
}

#[test]
fn disabled_final_ba_does_not_run_post_ba_refinement_rounds() {
    let mut config = IncrementalSfmConfig::default();
    assert_eq!(simple_final_refinement_rounds(&config), 2);

    config.final_global_ba = false;
    assert_eq!(
        simple_final_refinement_rounds(&config),
        0,
        "a growth-only run must not launch post-filter BA"
    );

    config.retriangulate = true;
    config.track_filter_iterations = 0;
    assert_eq!(simple_final_refinement_rounds(&config), 0);

    config.final_global_ba = true;
    assert_eq!(simple_final_refinement_rounds(&config), 1);
}

#[test]
fn targeted_growth_fast_path_excludes_support_changing_modes() {
    let defaults = IncrementalSfmConfig::default();
    assert!(targeted_plain_growth_enabled(&defaults, false));
    assert!(!targeted_plain_growth_enabled(&defaults, true));

    let mut colmap = defaults.clone();
    colmap.colmap_style_mapper = true;
    assert!(!targeted_plain_growth_enabled(&colmap, false));

    let mut correspondence = defaults.clone();
    correspondence.incremental_correspondence_triangulation = true;
    assert!(!targeted_plain_growth_enabled(&correspondence, false));

    let mut sequence = defaults;
    sequence.sequence_relative_pose_fallback = true;
    assert!(!targeted_plain_growth_enabled(&sequence, false));
}

#[test]
fn correspondence_count_cache_matches_fresh_scan_after_point_additions() {
    let feature = |count| {
        FeatureSet::new(
            (0..count)
                .map(|index| Point2::new(index as f64, index as f64))
                .collect(),
            (0..count).map(|_| vec![0.0f32]).collect(),
        )
        .expect("synthetic feature set")
    };
    let features = vec![feature(2), feature(2), feature(2)];
    let tracks = vec![vec![(0, 0), (1, 0)], vec![(1, 1), (2, 1)]];
    let obs_by_image = vec![vec![(0, 0)], vec![(0, 0), (1, 1)], vec![(1, 1)]];
    let mut points = vec![None, None];
    let mut cache = build_correspondence_count_cache(&features, &obs_by_image, &points);
    assert_eq!(cache, vec![0, 0, 0]);

    points[0] = Some(Point3::new(0.0, 0.0, 1.0));
    update_correspondence_count_cache(&features, &tracks, &points, &[0], &mut cache);
    assert_eq!(cache, vec![1, 1, 0]);
    assert_eq!(
        cache,
        build_correspondence_count_cache(&features, &obs_by_image, &points)
    );

    points[1] = Some(Point3::new(0.0, 0.0, 1.0));
    update_correspondence_count_cache(&features, &tracks, &points, &[1], &mut cache);
    assert_eq!(cache, vec![1, 2, 1]);
    assert_eq!(
        cache,
        build_correspondence_count_cache(&features, &obs_by_image, &points)
    );
}

fn pose_with_world_center(x: f64) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-x, 0.0, 0.0))
}

#[test]
fn sequence_fallback_scale_uses_latest_consecutive_median_and_needs_two_steps() {
    let poses = vec![
        Some(pose_with_world_center(0.0)),
        Some(pose_with_world_center(1.0)),
        Some(pose_with_world_center(3.0)),
        Some(pose_with_world_center(6.0)),
        None,
    ];
    let stems = vec![286, 287, 288, 289, 290];
    let (scale, mad, samples) =
        robust_recent_consecutive_step_scale(&poses, &stems).expect("three steps");
    assert!((scale - 2.0).abs() < 1.0e-12);
    assert!((mad - 1.0).abs() < 1.0e-12);
    assert_eq!(samples, 3);

    let insufficient = vec![
        Some(pose_with_world_center(0.0)),
        Some(pose_with_world_center(1.0)),
        None,
    ];
    assert!(robust_recent_consecutive_step_scale(&insufficient, &[1, 2, 3]).is_none());
}

#[test]
fn sequence_projected_scale_follows_straight_velocity() {
    let poses = vec![
        Some(pose_with_world_center(0.0)),
        Some(pose_with_world_center(1.0)),
        Some(pose_with_world_center(2.0)),
        Some(pose_with_world_center(3.0)),
    ];
    let estimate = projected_recent_consecutive_step_scale(
        &poses,
        &[1, 2, 3, 4],
        3,
        Vector3::new(1.0, 0.0, 0.0),
    )
    .expect("two straight steps should project");
    assert!((estimate.0 - 1.0).abs() < 1.0e-12);
    assert!((estimate.1 - 1.0).abs() < 1.0e-12);
    assert!(estimate.2.abs() < 1.0e-12);
    assert_eq!(estimate.3, 2);
    assert!((estimate.4 - Vector3::new(1.0, 0.0, 0.0)).norm() < 1.0e-12);
}

#[test]
fn sequence_projected_scale_handles_turn_and_rejects_bad_direction() {
    let poses = vec![
        Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(0.0, 0.0, 0.0),
        )),
        Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(-1.0, 0.0, 0.0),
        )),
        Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(-2.0, -1.0, 0.0),
        )),
    ];
    let turn =
        projected_recent_consecutive_step_scale(&poses, &[1, 2, 3], 3, Vector3::new(1.0, 0.0, 0.0))
            .expect("a bounded turn should retain a positive projection");
    assert!(turn.0 > 0.0);
    assert!((turn.0 - 1.0).abs() < 1.0e-12);
    assert!(projected_recent_consecutive_step_scale(
        &poses,
        &[1, 2, 3],
        3,
        Vector3::new(-1.0, 0.0, 0.0),
    )
    .is_none());
    assert!(projected_recent_consecutive_step_scale(
        &poses,
        &[1, 2, 3],
        3,
        Vector3::new(0.0, 0.0, 1.0),
    )
    .is_none());
}

#[test]
fn sequence_projected_scale_uses_mad_robustly_and_requires_history() {
    let poses = vec![
        Some(pose_with_world_center(0.0)),
        Some(pose_with_world_center(1.0)),
        Some(pose_with_world_center(3.0)),
        Some(pose_with_world_center(13.0)),
    ];
    let estimate = projected_recent_consecutive_step_scale(
        &poses,
        &[1, 2, 3, 4],
        4,
        Vector3::new(1.0, 0.0, 0.0),
    )
    .expect("the median velocity should reject the isolated long step");
    assert!((estimate.0 - 1.5).abs() < 1.0e-12);
    assert_eq!(estimate.3, 2);
    let insufficient = vec![
        Some(pose_with_world_center(0.0)),
        Some(pose_with_world_center(1.0)),
        None,
    ];
    assert!(projected_recent_consecutive_step_scale(
        &insufficient,
        &[1, 2, 3],
        2,
        Vector3::new(1.0, 0.0, 0.0),
    )
    .is_none());

    // A zero-MAD magnitude sample is still bounded: a component-wise
    // velocity median that points between equal-length turns must not
    // invent a larger step than the recent median.
    let turning = vec![
        Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(0.0, 0.0, 0.0),
        )),
        Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(-1.0, 0.0, 0.0),
        )),
        Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(-1.0, -1.0, 0.0),
        )),
        Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(-6.0, -6.0, 0.0),
        )),
    ];
    assert!(projected_recent_consecutive_step_scale(
        &turning,
        &[1, 2, 3, 4],
        4,
        Vector3::new(1.0, 1.0, 0.0),
    )
    .is_none());
    let diagnostic = projected_recent_consecutive_step_scale_diagnostic(
        &turning,
        &[1, 2, 3, 4],
        4,
        Vector3::new(1.0, 1.0, 0.0),
    )
    .expect("relaxed mode still has a finite projection");
    assert!(relaxed_projected_scale_is_valid(
        diagnostic.projected_scale,
        diagnostic.recent_median
    ));
}

#[test]
fn relaxed_projected_scale_uses_only_broad_bounds() {
    assert!(relaxed_projected_scale_is_valid(0.25, 1.0));
    assert!(relaxed_projected_scale_is_valid(4.0, 1.0));
    assert!(!relaxed_projected_scale_is_valid(0.249, 1.0));
    assert!(!relaxed_projected_scale_is_valid(4.001, 1.0));
    assert!(!relaxed_projected_scale_is_valid(1.0, 0.0));
    assert!(!relaxed_projected_scale_is_valid(-1.0, 1.0));
    assert!(!relaxed_projected_scale_is_valid(f64::NAN, 1.0));
}

#[test]
fn carried_sequence_scale_uses_first_projection_then_previous_baseline() {
    // The first fallback has no carry state and therefore keeps the
    // freshly projected scale.  The following consecutive fallback uses
    // the accepted baseline, not a newly projected value.
    assert_eq!(
        carried_sequence_scale_or_projection(None, 1.4, 1.0),
        (1.4, false)
    );
    assert_eq!(
        carried_sequence_scale_or_projection(Some(1.4), 2.8, 1.0),
        (1.4, true)
    );
}

#[test]
fn carried_sequence_scale_invalid_value_falls_back_and_pose_rescale_preserves_rotation() {
    assert_eq!(
        carried_sequence_scale_or_projection(Some(4.001), 1.25, 1.0),
        (1.25, false)
    );
    assert_eq!(
        carried_sequence_scale_or_projection(Some(f64::NAN), 1.25, 1.0),
        (1.25, false)
    );
    assert_eq!(
        next_sequence_fallback_carry_state(23, 1.4, 0),
        Some((23, 1.4))
    );
    assert_eq!(next_sequence_fallback_carry_state(23, 1.4, 1), None);

    let previous = pose_with_world_center(2.0);
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.3);
    let proposed = Pose::from_world_to_camera(rotation, Vector3::new(-5.0, 0.0, 0.0));
    let carried = rescale_sequence_pose_translation(&previous, &proposed, 1.5)
        .expect("finite proposed displacement should rescale");
    assert!(
        ((carried.camera_center_world() - previous.camera_center_world()).norm() - 1.5).abs()
            < 1.0e-12
    );
    assert!(
        carried
            .world_to_camera
            .rotation
            .rotation_to(&rotation)
            .angle()
            < 1.0e-12
    );
    assert!(rescale_sequence_pose_translation(&previous, &proposed, 0.0).is_none());
}

#[test]
fn sequence_fallback_pose_composition_preserves_relative_pose_convention() {
    let previous = pose_with_world_center(4.0);
    let relative_rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.2);
    let relative_translation = Vector3::new(0.0, 1.0, 0.0);
    let relative = visloc_core::geometry::SE3::new(relative_rotation, relative_translation * 2.0);
    let expected = relative.compose(&previous.world_to_camera);
    let actual =
        compose_sequence_relative_pose(&previous, relative_rotation, relative_translation, 2.0)
            .expect("finite relative pose");
    assert!((actual.world_to_camera.translation - expected.translation).norm() < 1.0e-12);
    assert!(
        actual
            .world_to_camera
            .rotation
            .rotation_to(&expected.rotation)
            .angle()
            < 1.0e-12
    );
}

#[test]
fn sequence_triangulation_admission_high_support_boundaries() {
    // The relaxed path is inclusive at both evidence-backed boundaries:
    // 100 valid points and exactly 30% of the selected support.
    assert!(sequence_triangulation_admission_ok(100, 300, 30, true));
    assert!(sequence_triangulation_admission_ok(100, 333, 30, true));
    assert!(!sequence_triangulation_admission_ok(100, 334, 30, true));
    assert!(!sequence_triangulation_admission_ok(99, 300, 30, true));
    assert!(!sequence_triangulation_admission_ok(120, 401, 30, true));

    // The override never weakens the configured absolute seed floor.
    assert!(!sequence_triangulation_admission_ok(100, 300, 101, true));

    // A sequence pair without the explicit high-support mark retains the
    // historical half-support gate, including the same selected count.
    assert!(!sequence_triangulation_admission_ok(100, 300, 30, false));
    assert!(sequence_triangulation_admission_ok(150, 300, 30, false));
    assert!(!sequence_triangulation_admission_ok(149, 300, 30, false));
}

#[test]
fn sequence_fallback_is_default_off_and_rejects_bad_essential() {
    let defaults = IncrementalSfmConfig::default();
    assert!(!defaults.sequence_relative_pose_fallback);
    assert!(!defaults.sequence_fallback_after_post);
    assert!(defaults.sequence_stem_values.is_none());

    let features = (0..4)
        .map(|_| {
            FeatureSet::new(
                (0..30)
                    .map(|index| Point2::new(index as f64 + 1.0, 100.0))
                    .collect(),
                (0..30).map(|_| vec![0.0f32, 1.0]).collect(),
            )
            .expect("synthetic feature set")
        })
        .collect::<Vec<_>>();
    let matches = (0..30).map(|index| (index, index)).collect::<Vec<_>>();
    let pair = PairwiseMatches {
        image_i: 2,
        image_j: 3,
        matches: matches.clone(),
        two_view_config: Some(ConfigurationType::Uncalibrated),
        essential_matches: Some(matches),
        essential_matrix: Some(Matrix3::zeros()),
    };
    let mut config = defaults;
    config.sequence_relative_pose_fallback = true;
    config.sequence_stem_values = Some(vec![286, 287, 288, 289]);
    let poses = vec![
        Some(pose_with_world_center(0.0)),
        Some(pose_with_world_center(1.0)),
        Some(pose_with_world_center(2.0)),
        None,
    ];
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    assert!(sequence_relative_pose_fallback_with_overrides(
        &camera,
        &features,
        &[pair],
        &poses,
        &config,
        None,
    )
    .is_none());
}

#[test]
fn sequence_fallback_after_post_defers_eager_growth_only() {
    let defaults = IncrementalSfmConfig::default();
    assert!(!sequence_fallback_enabled_during_growth(&defaults));

    let mut eager = defaults;
    eager.sequence_relative_pose_fallback = true;
    assert!(sequence_fallback_enabled_during_growth(&eager));

    let mut deferred = eager;
    deferred.sequence_fallback_after_post = true;
    assert!(!sequence_fallback_enabled_during_growth(&deferred));
    // The scheduling bit is orthogonal to the scale policy; it only moves
    // the same fallback proposal out of the ordinary growth loop.
    deferred.sequence_constant_velocity_scale = true;
    assert!(!sequence_fallback_enabled_during_growth(&deferred));
}

#[test]
fn sfm_oracle_metrics_are_sim3_and_rotation_invariant() {
    let centres = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(1.0, 0.0, 0.2),
        Vector3::new(0.1, 1.4, 0.7),
        Vector3::new(-0.5, 0.4, 1.8),
    ];
    let make_pose = |centre: Vector3<f64>, camera_to_world: UnitQuaternion<f64>| {
        Pose::from_world_to_camera(
            camera_to_world.inverse(),
            -(camera_to_world.inverse() * centre),
        )
    };
    let oracle: Vec<Option<Pose>> = centres
        .iter()
        .copied()
        .map(|centre| make_pose(centre, UnitQuaternion::identity()))
        .map(Some)
        .collect();
    let transform = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.37);
    let scale = 2.4;
    let offset = Vector3::new(3.0, -1.0, 0.8);
    let mapped: Vec<Option<Pose>> = centres
        .iter()
        .copied()
        .map(|centre| make_pose(scale * (transform * centre) + offset, transform))
        .map(Some)
        .collect();
    let metrics = sfm_oracle_metrics(&mapped, &oracle).expect("four common poses align");
    assert!(
        metrics.center_rmse < 1.0e-10,
        "center rmse={}",
        metrics.center_rmse
    );
    assert!(
        metrics.rotation_mean < 1.0e-10,
        "rotation={}",
        metrics.rotation_mean
    );
    assert!(sfm_oracle_metrics(&mapped[..2], &oracle[..2]).is_none());
}

/// `LocalSubmapBuilder::build`'s scale-pathology retry
/// (`NOROBUSTFIT_CLUSTER_DIAGNOSIS.md` §6(b)) relies on
/// `seed_candidate_order` walking to the *next*-ranked seed candidate,
/// deterministically, once the previously tried pair is excluded. This
/// pins that mechanism directly: descending match-count order by
/// default, and excluding a pair (regardless of which of its two image
/// orderings is recorded) removes exactly that pair and nothing else,
/// repeatably.
#[test]
fn seed_candidate_order_skips_excluded_pairs_deterministically() {
    let pairwise = vec![
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0); 50],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
        PairwiseMatches {
            image_i: 1,
            image_j: 2,
            matches: vec![(0, 0); 40],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
        PairwiseMatches {
            image_i: 2,
            image_j: 3,
            matches: vec![(0, 0); 30],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
    ];
    let config = IncrementalSfmConfig::default();
    assert_eq!(seed_candidate_order(&pairwise, &config), vec![0, 1, 2]);

    let mut selected_pair = config.clone();
    selected_pair.seed_pair = Some((1, 2));
    assert_eq!(seed_candidate_order(&pairwise, &selected_pair), vec![1]);
    // The library field is documented as normalized; the CLI parser
    // canonicalizes reversed user input before constructing this config.
    selected_pair.seed_pair = Some((2, 1));
    assert_eq!(
        seed_candidate_order(&pairwise, &selected_pair),
        Vec::<usize>::new()
    );
    selected_pair.seed_pair = Some((1, 2));
    assert_eq!(seed_candidate_order(&pairwise, &selected_pair), vec![1]);

    let mut excluded_first = config.clone();
    excluded_first.excluded_seed_pairs.insert((0, 1));
    assert_eq!(seed_candidate_order(&pairwise, &excluded_first), vec![1, 2]);
    // Deterministic: repeated calls on the same (excluded) config agree.
    assert_eq!(seed_candidate_order(&pairwise, &excluded_first), vec![1, 2]);

    // The pairwise-side key is normalized regardless of which image is
    // recorded as `image_i`/`image_j`: a reversed-direction entry for
    // the same underlying pair still matches a normalized `(0, 1)`
    // exclusion key.
    let mut reversed_first_pair = pairwise.clone();
    reversed_first_pair[0] = PairwiseMatches {
        image_i: 1,
        image_j: 0,
        matches: vec![(0, 0); 50],
        two_view_config: None,
        essential_matches: None,
        essential_matrix: None,
    };
    let mut excluded_normalized = config.clone();
    excluded_normalized.excluded_seed_pairs.insert((0, 1));
    assert_eq!(
        seed_candidate_order(&reversed_first_pair, &excluded_normalized),
        vec![1, 2]
    );

    // Excluding the two strongest pairs walks to the third-ranked
    // candidate, still in descending order among what remains.
    let mut excluded_two = config;
    excluded_two.excluded_seed_pairs.insert((0, 1));
    excluded_two.excluded_seed_pairs.insert((1, 2));
    assert_eq!(seed_candidate_order(&pairwise, &excluded_two), vec![2]);
}

fn physical_track_signature(
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
) -> Vec<Vec<(usize, i64, i64)>> {
    let mut signature = tracks
        .iter()
        .map(|track| {
            track
                .iter()
                .map(|&(image, keypoint)| {
                    let point = features[image].keypoints[keypoint];
                    (
                        image,
                        (point.x * 1_000_000.0).round() as i64,
                        (point.y * 1_000_000.0).round() as i64,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    signature.sort();
    signature
}

#[test]
fn cycle_supported_tracks_prefer_supported_three_view_edges() {
    let features = vec![
        FeatureSet::new(
            vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)],
            vec![vec![0.0f32], vec![1.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![Point2::new(0.0, 1.0), Point2::new(10.0, 1.0)],
            vec![vec![0.0f32], vec![1.0]],
        )
        .unwrap(),
        FeatureSet::new(vec![Point2::new(0.0, 2.0)], vec![vec![0.0f32]]).unwrap(),
    ];
    let pairwise = vec![
        // An unsupported edge is intentionally first. Legacy union-find
        // would absorb it and later drop the whole same-image conflict.
        PairwiseMatches::new(0, 1, vec![(0, 1)]),
        PairwiseMatches::new(0, 1, vec![(0, 0)]),
        PairwiseMatches::new(0, 2, vec![(0, 0)]),
        PairwiseMatches::new(1, 2, vec![(0, 0)]),
    ];
    let adjacency = {
        let mut lookup = HashMap::new();
        for pair in &pairwise {
            let forward = lookup
                .entry((pair.image_i, pair.image_j))
                .or_insert_with(HashMap::new);
            for &(a, b) in &pair.matches {
                forward.entry(a).or_insert_with(HashSet::new).insert(b);
            }
            let reverse = lookup
                .entry((pair.image_j, pair.image_i))
                .or_insert_with(HashMap::new);
            for &(a, b) in &pair.matches {
                reverse.entry(b).or_insert_with(HashSet::new).insert(a);
            }
        }
        lookup
    };
    assert_eq!(cycle_support_for_edge(3, 0, 0, 1, 0, &adjacency), (1, 1));
    assert_eq!(cycle_support_for_edge(3, 0, 0, 1, 1, &adjacency), (0, 0));

    let cycle = build_tracks_cycle_supported(&features, None, &pairwise, 3);
    assert_eq!(cycle.tracks, vec![vec![(0, 0), (1, 0), (2, 0)]]);
    assert_eq!(cycle.stats.retained_tracks, 1);
    assert_eq!(cycle.stats.retained_observations, 3);
    assert!(!cycle.tracks[0].contains(&(1, 1)));
}

#[test]
fn cycle_supported_tracks_are_permutation_invariant() {
    let features = vec![
        FeatureSet::new(
            vec![Point2::new(20.0, 0.0), Point2::new(10.0, 0.0)],
            vec![vec![20.0f32], vec![10.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![Point2::new(20.0, 1.0), Point2::new(10.0, 1.0)],
            vec![vec![20.0f32], vec![10.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![Point2::new(20.0, 2.0), Point2::new(10.0, 2.0)],
            vec![vec![20.0f32], vec![10.0]],
        )
        .unwrap(),
    ];
    let pairwise = vec![
        PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(0, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
    ];
    let permutations = [[1usize, 0], [1, 0], [1, 0]];
    let permuted_features = features
        .iter()
        .zip(permutations)
        .map(|(set, permutation)| {
            FeatureSet::new(
                permutation
                    .iter()
                    .map(|&index| set.keypoints[index])
                    .collect(),
                permutation
                    .iter()
                    .map(|&index| set.descriptors[index].clone())
                    .collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let remapped = pairwise
        .iter()
        .map(|pair| {
            PairwiseMatches::new(
                pair.image_i,
                pair.image_j,
                pair.matches
                    .iter()
                    .map(|&(lhs, rhs)| {
                        (
                            permutations[pair.image_i][lhs],
                            permutations[pair.image_j][rhs],
                        )
                    })
                    .rev()
                    .collect(),
            )
        })
        .rev()
        .collect::<Vec<_>>();
    let original = build_tracks_cycle_supported(&features, None, &pairwise, 2);
    let permuted = build_tracks_cycle_supported(&permuted_features, None, &remapped, 2);
    assert_eq!(
        physical_track_signature(&features, &original.tracks),
        physical_track_signature(&permuted_features, &permuted.tracks)
    );
    assert_eq!(original.stats, permuted.stats);
}

#[test]
fn cycle_supported_tracks_have_deterministic_no_cycle_fallback() {
    let features = vec![
        FeatureSet::new(
            vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)],
            vec![vec![0.0f32], vec![1.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![Point2::new(0.0, 1.0), Point2::new(10.0, 1.0)],
            vec![vec![0.0f32], vec![1.0]],
        )
        .unwrap(),
        FeatureSet::new(vec![Point2::new(100.0, 2.0)], vec![vec![0.0f32]]).unwrap(),
    ];
    let first = PairwiseMatches::new(0, 1, vec![(0, 1), (0, 0)]);
    let second = PairwiseMatches::new(0, 1, vec![(0, 0), (0, 1)]);
    let output_a =
        build_tracks_cycle_supported(&features, None, &[first.clone(), second.clone()], 2);
    let output_b = build_tracks_cycle_supported(&features, None, &[second, first], 2);
    assert_eq!(output_a.tracks, vec![vec![(0, 0), (1, 0)]]);
    assert_eq!(output_a.tracks, output_b.tracks);
    assert_eq!(output_a.stats, output_b.stats);
}

/// A synthetic 3D point cloud and a ring of cameras looking at it, used to
/// exercise the full unordered pipeline end-to-end.
struct Scene {
    camera: Camera,
    points: Vec<Point3<f64>>,
    poses: Vec<Pose>,
}

fn build_scene() -> Scene {
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    // A 3D grid of points around the origin.
    let mut points = Vec::new();
    for xi in -2..=2 {
        for yi in -2..=2 {
            for zi in 0..=2 {
                points.push(Point3::new(
                    xi as f64 * 0.3,
                    yi as f64 * 0.3,
                    zi as f64 * 0.3,
                ));
            }
        }
    }
    // Cameras on an arc, all looking roughly toward the cloud centre from
    // ~3 m away (enough parallax between neighbours).
    let mut poses = Vec::new();
    for k in 0..6 {
        let angle = -0.5 + k as f64 * 0.2; // radians along the arc
        let radius = 3.0;
        let cam_center = Point3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
        // Look-at the origin: build world_to_camera.
        let forward = (Point3::origin() - cam_center).normalize();
        let world_up = Vector3::new(0.0, 1.0, 0.0);
        let right = forward.cross(&world_up).normalize();
        let up = right.cross(&forward);
        // Rotation columns map camera axes (x=right, y=down, z=forward) to world.
        let r_cam_to_world = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
        let rot_c2w = nalgebra::Rotation3::from_matrix_unchecked(r_cam_to_world);
        let q_c2w = UnitQuaternion::from_rotation_matrix(&rot_c2w);
        let q_w2c = q_c2w.inverse();
        let t_w2c = -(q_w2c * cam_center.coords);
        poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
    }
    Scene {
        camera,
        points,
        poses,
    }
}

/// Project a world point into a pose; `None` if behind camera or off-image.
fn project(camera: &Camera, pose: &Pose, p: &Point3<f64>) -> Option<Point2<f64>> {
    let cam = pose.transform_world_point(p);
    if cam.z <= 0.05 {
        return None;
    }
    let px = camera.project(&cam)?;
    if px.x < 0.0 || px.x >= camera.width as f64 || px.y < 0.0 || px.y >= camera.height as f64 {
        return None;
    }
    Some(px)
}

/// Render the scene to per-image features (keypoint per visible point, the
/// point index baked into a trivial descriptor) and ground-truth pairwise
/// matches between every image pair that co-observes ≥8 points.
fn render(scene: &Scene) -> (Vec<FeatureSet>, Vec<PairwiseMatches>) {
    let n = scene.poses.len();
    // visible[image] = map point_index -> keypoint_index
    let mut features = Vec::new();
    let mut visible: Vec<HashMap<usize, usize>> = Vec::new();
    for pose in &scene.poses {
        let mut kps = Vec::new();
        let mut descs = Vec::new();
        let mut vis = HashMap::new();
        for (pidx, p) in scene.points.iter().enumerate() {
            if let Some(px) = project(&scene.camera, pose, p) {
                vis.insert(pidx, kps.len());
                kps.push(px);
                // Descriptor is irrelevant here (matches are ground truth),
                // but FeatureSet wants one; use a tiny unique vector.
                descs.push(vec![pidx as f32, 1.0, 0.0, 0.0]);
            }
        }
        features.push(FeatureSet::new(kps, descs).unwrap());
        visible.push(vis);
    }

    let mut pairwise = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut matches = Vec::new();
            for (pidx, &ki) in &visible[i] {
                if let Some(&kj) = visible[j].get(pidx) {
                    matches.push((ki, kj));
                }
            }
            if matches.len() >= 8 {
                pairwise.push(PairwiseMatches {
                    image_i: i,
                    image_j: j,
                    matches,
                    two_view_config: None,
                    essential_matches: None,
                    essential_matrix: None,
                });
            }
        }
    }
    (features, pairwise)
}

/// M2 acceptance test: on the same realistic multi-image synthetic scene
/// every other integration test in this module uses
/// (`build_scene`/`render` — a 45-point cloud seen by a 6-camera ring),
/// [`build_tracks_via_graph`] must produce **byte-identical** tracks to
/// the legacy [`build_tracks`] union-find — the refactor gate
/// `docs/colmap_port_plan.md`'s M2 milestone specifies ("byte-identical
/// tracks... a refactor gate, not an accuracy claim"), exercised here on
/// real transitive (multi-hop, multi-image) structure rather than the
/// small hand-built fixtures above.
#[test]
fn graph_tracks_match_union_find_tracks_on_synthetic_scene() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    assert!(
        !pairwise.is_empty(),
        "fixture sanity: the scene must produce at least one verified pair"
    );

    let union_find_tracks = build_tracks(features.len(), &pairwise, 2);
    let graph_tracks = build_tracks_via_graph(&features, &pairwise, 2);
    assert_eq!(
        union_find_tracks, graph_tracks,
        "CorrespondenceGraph-derived tracks must byte-match the legacy union-find's"
    );
    assert!(
        !union_find_tracks.is_empty(),
        "fixture sanity: some tracks must form"
    );
}

#[test]
fn post_refinement_pass_registers_against_tightened_structure_once() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let tracks = build_tracks(features.len(), &pairwise, 2);
    let mut poses = vec![None; features.len()];
    poses[0] = Some(scene.poses[0].clone());
    poses[1] = Some(scene.poses[1].clone());
    let mut track_point = vec![None; tracks.len()];
    let config = IncrementalSfmConfig {
        min_pnp_inliers: 8,
        max_reprojection_error_px: 2.0,
        ..IncrementalSfmConfig::default()
    };
    triangulate_pending(
        &scene.camera,
        &features,
        &tracks,
        &poses,
        &config,
        &mut track_point,
    );
    assert!(track_point.iter().filter(|p| p.is_some()).count() >= 8);

    let added = post_refinement_registration_pass(
        &scene.camera,
        &features,
        &tracks,
        &config,
        &mut poses,
        &mut track_point,
    )
    .unwrap();
    assert!(added > 0);
    assert_eq!(poses.iter().filter(|p| p.is_some()).count(), 2 + added);
}

#[test]
fn pose_guided_split_recovers_two_points_prunes_outlier_and_is_deterministic() {
    let scene = build_scene();
    let mut visible_points = scene
        .points
        .iter()
        .filter(|point| {
            (0..4).all(|image| project(&scene.camera, &scene.poses[image], point).is_some())
        })
        .copied();
    let point_a = visible_points
        .next()
        .expect("a point visible in four views");
    let point_b = visible_points
        .next()
        .expect("a second point visible in four views");
    let mut features = Vec::new();
    for image in 0..4 {
        let pixel_a = project(&scene.camera, &scene.poses[image], &point_a).unwrap();
        let pixel_b = project(&scene.camera, &scene.poses[image], &point_b).unwrap();
        let outlier = if image == 3 {
            Point2::new(pixel_a.x + 80.0, pixel_a.y + 45.0)
        } else {
            pixel_a
        };
        features.push(
            FeatureSet::new(
                vec![pixel_a, pixel_b, outlier],
                vec![vec![0.0f32], vec![1.0], vec![2.0]],
            )
            .unwrap(),
        );
    }

    // The false cross-edge joins two physical points into one legacy
    // component; the final edge joins one of those points to an outlier
    // in an already represented image.  A posed split must recover the
    // two four-view tracks and discard the lone outlier.
    let pairwise = vec![
        PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(2, 3, vec![(0, 0), (1, 1), (0, 2)]),
    ];
    let component = vec![
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
    ];
    let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
    let config = IncrementalSfmConfig {
        min_track_length: 2,
        max_reprojection_error_px: 4.0,
        conflict_recovery_max_hypotheses: 16,
        pose_guided_track_splitting: true,
        ..IncrementalSfmConfig::default()
    };
    let mut incomplete_poses = poses.clone();
    incomplete_poses[3] = None;
    assert!(pose_guided_split_tracks(
        &scene.camera,
        &features,
        &pairwise,
        &[],
        std::slice::from_ref(&component),
        &[],
        &incomplete_poses,
        &config,
    )
    .is_none());
    let first = pose_guided_split_tracks(
        &scene.camera,
        &features,
        &pairwise,
        &[],
        std::slice::from_ref(&component),
        &[],
        &poses,
        &config,
    )
    .expect("complete poses should enable the split");
    assert_eq!(first.stats.split_components, 1);
    assert_eq!(first.stats.emitted_tracks, 2);
    assert!(first.stats.discarded_observations >= 1);
    assert!(first
        .tracks
        .iter()
        .any(|track| track == &vec![(0, 0), (1, 0), (2, 0), (3, 0)]));
    assert!(first
        .tracks
        .iter()
        .any(|track| track == &vec![(0, 1), (1, 1), (2, 1), (3, 1)]));

    let reordered_pairwise = pairwise
        .iter()
        .rev()
        .map(|pair| {
            let mut pair = pair.clone();
            pair.matches.reverse();
            pair
        })
        .collect::<Vec<_>>();
    let mut reversed_component = component;
    reversed_component.reverse();
    let second = pose_guided_split_tracks(
        &scene.camera,
        &features,
        &reordered_pairwise,
        &[],
        std::slice::from_ref(&reversed_component),
        &[],
        &poses,
        &config,
    )
    .expect("reordered input should remain deterministic");
    assert_eq!(first.tracks, second.tracks);
    assert_eq!(first.points, second.points);
    assert_eq!(first.stats, second.stats);
    assert!(!IncrementalSfmConfig::default().pose_guided_track_splitting);
    assert_eq!(
        IncrementalSfmConfig::default().pose_guided_split_max_reprojection_error_px,
        None
    );
    assert_eq!(
        IncrementalSfmConfig::default().pose_guided_track_splitting_iterations,
        1
    );
    assert!(!IncrementalSfmConfig::default().pose_guided_bridge_cuts);
    assert!(!IncrementalSfmConfig::default().pose_guided_track_merging);
    assert_eq!(
        IncrementalSfmConfig::default().pose_guided_merge_max_reprojection_error_px,
        None
    );
    assert_eq!(IncrementalSfmConfig::default().final_min_track_length, None);
}

#[test]
fn pose_guided_track_merge_requires_geometry_and_is_permutation_invariant() {
    let scene = build_scene();
    let defaults = IncrementalSfmConfig::default();
    assert_eq!(
        pose_guided_merge_reprojection_gate(&defaults, 2.0),
        Some(2.0)
    );
    let explicit = IncrementalSfmConfig {
        pose_guided_merge_max_reprojection_error_px: Some(4.0),
        ..defaults.clone()
    };
    assert_eq!(
        pose_guided_merge_reprojection_gate(&explicit, 2.0),
        Some(4.0)
    );
    let invalid = IncrementalSfmConfig {
        pose_guided_merge_max_reprojection_error_px: Some(0.0),
        ..defaults
    };
    assert_eq!(pose_guided_merge_reprojection_gate(&invalid, 2.0), None);
    let point = scene
        .points
        .iter()
        .find(|point| {
            (0..6).all(|image| project(&scene.camera, &scene.poses[image], point).is_some())
        })
        .copied()
        .expect("fixture point must be visible in every camera");
    let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
    let config = IncrementalSfmConfig {
        max_reprojection_error_px: 0.1,
        min_track_length: 2,
        ..IncrementalSfmConfig::default()
    };
    let features = (0..6)
        .map(|image| {
            FeatureSet::new(
                vec![project(&scene.camera, &scene.poses[image], &point).unwrap()],
                vec![vec![0.0f32]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    // A verified edge is enough for a complementary pair of tracks.  The
    // union is refit from all four observations, not accepted merely from
    // the edge's two endpoints.
    let fragments = vec![vec![(0, 0), (1, 0)], vec![(2, 0), (3, 0)]];
    let pairwise = vec![PairwiseMatches::new(1, 2, vec![(0, 0)])];
    let (merged, merged_points, merges, tested) = pose_guided_merge_tracks(
        &scene.camera,
        &features,
        &pairwise,
        &poses,
        &fragments,
        &[None, None],
        &config,
    );
    assert_eq!(merges, 1);
    assert_eq!(tested, 1);
    assert_eq!(merged, vec![vec![(0, 0), (1, 0), (2, 0), (3, 0)]]);
    assert!(merged_points[0].is_some());
    assert!(pose_guided_track_reprojection_valid(
        &scene.camera,
        &features,
        &merged[0],
        &poses,
        merged_points[0].as_ref(),
        0.1,
    ));
    assert!(!pose_guided_track_reprojection_valid(
        &scene.camera,
        &features,
        &merged[0],
        &poses,
        merged_points[0].as_ref(),
        0.0,
    ));

    // Two distinct physical points on disjoint image sets can still have
    // a false verified edge.  The all-observation reprojection gate must
    // reject their union even though the pair itself is geometrically
    // valid as an edge.
    let point_b = scene
        .points
        .iter()
        .copied()
        .find(|candidate| {
            *candidate != point
                && (0..6)
                    .all(|image| project(&scene.camera, &scene.poses[image], candidate).is_some())
        })
        .expect("fixture needs a second visible point");
    let mixed_features = (0..4)
        .map(|image| {
            let physical = if image < 2 { point } else { point_b };
            FeatureSet::new(
                vec![project(&scene.camera, &scene.poses[image], &physical).unwrap()],
                vec![vec![0.0f32]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let (not_merged, _, false_merges, _) = pose_guided_merge_tracks(
        &scene.camera,
        &mixed_features,
        &[PairwiseMatches::new(1, 2, vec![(0, 0)])],
        &poses[..4],
        &fragments,
        &[None, None],
        &config,
    );
    assert_eq!(false_merges, 0);
    assert_eq!(not_merged, fragments);

    // Same-image overlap is rejected before any triangulation, even when
    // a verified edge crosses the two fragments.
    let conflict_tracks = vec![vec![(0, 0), (1, 0)], vec![(1, 1), (2, 0)]];
    let conflict_features = vec![
        FeatureSet::new(
            vec![project(&scene.camera, &scene.poses[0], &point).unwrap()],
            vec![vec![0.0f32]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![
                project(&scene.camera, &scene.poses[1], &point).unwrap(),
                project(&scene.camera, &scene.poses[1], &point_b).unwrap(),
            ],
            vec![vec![0.0f32], vec![1.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![project(&scene.camera, &scene.poses[2], &point).unwrap()],
            vec![vec![0.0f32]],
        )
        .unwrap(),
    ];
    let (conflict_result, _, conflict_merges, _) = pose_guided_merge_tracks(
        &scene.camera,
        &conflict_features,
        &[PairwiseMatches::new(0, 2, vec![(0, 0)])],
        &poses[..3],
        &conflict_tracks,
        &[None, None],
        &config,
    );
    assert_eq!(conflict_merges, 0);
    assert_eq!(conflict_result, conflict_tracks);

    // A chain requires recomputing candidates after the first union: the
    // second edge touches the newly created four-view track.
    let chain_tracks = vec![
        vec![(0, 0), (1, 0)],
        vec![(2, 0), (3, 0)],
        vec![(4, 0), (5, 0)],
    ];
    let chain_edges = vec![
        PairwiseMatches::new(1, 2, vec![(0, 0)]),
        PairwiseMatches::new(3, 4, vec![(0, 0)]),
    ];
    let (chain_result, _, chain_merges, _) = pose_guided_merge_tracks(
        &scene.camera,
        &features,
        &chain_edges,
        &poses,
        &chain_tracks,
        &[None, None, None],
        &config,
    );
    assert_eq!(chain_merges, 2);
    assert_eq!(
        chain_result,
        vec![vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]]
    );

    let mut reversed_tracks = chain_tracks;
    reversed_tracks.reverse();
    let reversed_edges = chain_edges
        .iter()
        .rev()
        .map(|pair| {
            let mut pair = pair.clone();
            pair.matches.reverse();
            pair
        })
        .collect::<Vec<_>>();
    let (reordered_result, _, reordered_merges, _) = pose_guided_merge_tracks(
        &scene.camera,
        &features,
        &reversed_edges,
        &poses,
        &reversed_tracks,
        &[None, None, None],
        &config,
    );
    assert_eq!(chain_result, reordered_result);
    assert_eq!(chain_merges, reordered_merges);
}

#[test]
fn pose_guided_invalid_merge_restores_only_that_fragment_set() {
    let scene = build_scene();
    let point = scene.points[0];
    let point_b = scene
        .points
        .iter()
        .copied()
        .find(|candidate| *candidate != point)
        .expect("fixture needs two points");
    let poses = scene.poses[..4]
        .iter()
        .cloned()
        .map(Some)
        .collect::<Vec<_>>();
    let features = (0..4)
        .map(|image| {
            FeatureSet::new(
                vec![
                    project(&scene.camera, &scene.poses[image], &point).unwrap(),
                    project(&scene.camera, &scene.poses[image], &point_b).unwrap(),
                ],
                vec![vec![0.0f32], vec![1.0]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let good_left = vec![(0, 0), (1, 0)];
    let good_right = vec![(2, 0), (3, 0)];
    let bad_left = vec![(0, 1), (1, 1)];
    let bad_right = vec![(2, 1), (3, 1)];
    let good_merged = vec![(0, 0), (1, 0), (2, 0), (3, 0)];
    let bad_merged = vec![(0, 1), (1, 1), (2, 1), (3, 1)];
    let restorations = vec![
        PoseGuidedMergeRestoration {
            source_track_ids: vec![0, 1],
            source_tracks: vec![good_left.clone(), good_right.clone()],
            source_points: vec![Some(point), Some(point)],
            merged_track: good_merged.clone(),
        },
        PoseGuidedMergeRestoration {
            source_track_ids: vec![2, 3],
            source_tracks: vec![bad_left.clone(), bad_right.clone()],
            source_points: vec![Some(point_b), Some(point_b)],
            merged_track: bad_merged.clone(),
        },
    ];
    assert_eq!(
        pose_guided_merge_restorations(
            &[good_left, good_right, bad_left.clone(), bad_right.clone(),],
            &[Some(point), Some(point), Some(point_b), Some(point_b)],
            &[good_merged.clone(), bad_merged.clone()],
        ),
        restorations
    );
    let mut tracks = vec![good_merged.clone(), bad_merged];
    let mut points = vec![Some(point), Some(point)];
    let result = pose_guided_restore_invalid_merges(
        &scene.camera,
        &features,
        &poses,
        &mut tracks,
        &mut points,
        &restorations,
        0.1,
    );
    assert_eq!(result, (2, 1));
    assert_eq!(tracks, vec![good_merged, bad_left, bad_right]);
    assert_eq!(points, vec![Some(point), Some(point_b), Some(point_b)]);
    assert!(pose_guided_merge_restorations_reprojection_valid(
        &scene.camera,
        &features,
        &poses,
        &tracks,
        &points,
        &restorations,
        1,
        0.1,
    ));

    // Input traversal order does not affect which merged set is restored
    // or the exact source-fragment order in the final partition.
    let mut reversed_tracks = vec![
        vec![(0, 1), (1, 1), (2, 1), (3, 1)],
        vec![(0, 0), (1, 0), (2, 0), (3, 0)],
    ];
    let mut reversed_points = vec![Some(point), Some(point)];
    let mut reversed_restorations = restorations.clone();
    reversed_restorations.reverse();
    let reversed_result = pose_guided_restore_invalid_merges(
        &scene.camera,
        &features,
        &poses,
        &mut reversed_tracks,
        &mut reversed_points,
        &reversed_restorations,
        0.1,
    );
    assert_eq!(reversed_result, result);
    assert_eq!(reversed_tracks, tracks);
    assert_eq!(reversed_points, points);
}

#[test]
fn pose_guided_split_iteration_guard_covers_identity_improvement_rollback_and_stop() {
    let defaults = IncrementalSfmConfig::default();
    assert_eq!(defaults.pose_guided_track_splitting_iterations, 1);

    // The first pass retains the historical acceptance rule: a denser
    // candidate may have a slightly larger pre-BA mean, provided its own
    // guarded BA lowers that candidate objective.
    let first =
        pose_guided_split_candidate_accepts(0, 38, 38, 100, 120, 120, 0.30, 0.31, 0.30, 1.0);
    assert!(first);

    // A genuinely improving second pass is admitted from the same source
    // components, while a non-improving pass is the deterministic stop.
    let second_improves =
        pose_guided_split_candidate_accepts(1, 38, 38, 120, 125, 125, 0.30, 0.29, 0.28, 1.0);
    assert!(second_improves);
    let second_stops =
        pose_guided_split_candidate_accepts(1, 38, 38, 120, 125, 125, 0.30, 0.29, 0.30, 1.0);
    assert!(!second_stops);

    // A support or registration regression, non-finite objective, and an
    // invalid gate all force rollback. Re-evaluating the same values is
    // pure/deterministic.
    assert!(!pose_guided_split_candidate_accepts(
        1, 38, 37, 120, 125, 125, 0.30, 0.29, 0.28, 1.0,
    ));
    assert!(!pose_guided_split_candidate_accepts(
        1, 38, 38, 120, 125, 119, 0.30, 0.29, 0.28, 1.0,
    ));
    assert!(!pose_guided_split_candidate_accepts(
        1,
        38,
        38,
        120,
        125,
        125,
        0.30,
        f64::NAN,
        0.28,
        1.0,
    ));
    assert!(!pose_guided_split_candidate_accepts(
        1, 38, 38, 120, 125, 125, 0.30, 0.29, 0.28, 0.0,
    ));
    assert_eq!(
        second_improves,
        pose_guided_split_candidate_accepts(1, 38, 38, 120, 125, 125, 0.30, 0.29, 0.28, 1.0,)
    );
}

#[test]
fn pose_guided_composition_snapshots_original_components_and_is_default_off() {
    let tracks = vec![vec![(0, 0), (1, 1)]];
    let conflicts = vec![vec![(0, 2), (1, 3)]];
    let points = vec![Some(Point3::new(1.0, 2.0, 3.0))];
    let source = capture_pose_guided_split_source(true, None, &tracks, &conflicts, &points)
        .expect("enabled composition must capture its source");

    // Simulate geometry recovery appending/replacing state after the
    // snapshot.  The splitter must still receive the original components,
    // not recovered tracks recursively.
    let mut recovered_tracks = tracks.clone();
    recovered_tracks.push(vec![(0, 4), (1, 5), (2, 6)]);
    assert_eq!(source.0, tracks);
    assert_eq!(source.1, conflicts);
    assert_eq!(source.2, points);
    assert_ne!(source.0, recovered_tracks);

    // The ordinary/default path and imported membership diagnostics never
    // allocate a composition snapshot.
    assert!(capture_pose_guided_split_source(false, None, &tracks, &conflicts, &points).is_none());
    let membership = vec![vec![(0, 0), (1, 1)]];
    assert!(capture_pose_guided_split_source(
        true,
        Some(&membership),
        &tracks,
        &conflicts,
        &points,
    )
    .is_none());
    assert!(!IncrementalSfmConfig::default().pose_guided_track_splitting);
}

#[test]
fn pose_guided_bridge_cuts_require_two_valid_sides_and_reject_sparse_chain_cuts() {
    let scene = build_scene();
    let (features, _) = render(&scene);
    let visible_points = scene
        .points
        .iter()
        .enumerate()
        .filter(|(_, point)| {
            (0..4).all(|image| project(&scene.camera, &scene.poses[image], point).is_some())
        })
        .take(2)
        .map(|(point, _)| point)
        .collect::<Vec<_>>();
    assert_eq!(visible_points.len(), 2);
    let keypoint_for_point = |image: usize, point: usize| {
        features[image]
            .descriptors
            .iter()
            .position(|descriptor| descriptor[0] == point as f32)
            .unwrap()
    };
    let observations_for = |point: usize| {
        (0..4)
            .map(|image| (image, keypoint_for_point(image, point)))
            .collect::<Vec<_>>()
    };
    let edge = |left: TrackObservation, right: TrackObservation| {
        if left <= right {
            (left, right)
        } else {
            (right, left)
        }
    };
    let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
    let config = IncrementalSfmConfig {
        max_reprojection_error_px: 0.1,
        conflict_recovery_max_hypotheses: 32,
        ..IncrementalSfmConfig::default()
    };
    let first = observations_for(visible_points[0]);
    let second = observations_for(visible_points[1]);
    let mut false_bridge_edges = vec![
        edge(first[0], first[1]),
        edge(first[1], first[2]),
        edge(first[2], first[3]),
        edge(second[0], second[1]),
        edge(second[1], second[2]),
        edge(second[2], second[3]),
        edge(first[0], second[1]),
    ];
    false_bridge_edges.sort_unstable();
    let mut false_bridge_observations = first.clone();
    false_bridge_observations.extend(second.clone());
    let cut = pose_guided_bridge_cut_component(
        &scene.camera,
        &features,
        &poses,
        &false_bridge_observations,
        &false_bridge_edges,
        &config,
    );
    assert_eq!(cut.cut_edges, vec![edge(first[0], second[1])]);
    assert_eq!(cut.cut_sizes, vec![(4, 4)]);
    assert_eq!(cut.components, vec![first.clone(), second]);

    let mut reversed_edges = false_bridge_edges.clone();
    reversed_edges.reverse();
    let mut reversed_observations = false_bridge_observations.clone();
    reversed_observations.reverse();
    let reordered = pose_guided_bridge_cut_component(
        &scene.camera,
        &features,
        &poses,
        &reversed_observations,
        &reversed_edges,
        &config,
    );
    assert_eq!(cut.components, reordered.components);
    assert_eq!(cut.cut_edges, reordered.cut_edges);
    assert_eq!(cut.cut_sizes, reordered.cut_sizes);

    // A genuine one-point chain has bridge edges, but the complete side
    // still fits one posed point, so no bridge is cut.
    let chain_edges = (0..3)
        .map(|index| edge(first[index], first[index + 1]))
        .collect::<Vec<_>>();
    let chain = pose_guided_bridge_cut_component(
        &scene.camera,
        &features,
        &poses,
        &first,
        &chain_edges,
        &config,
    );
    assert!(chain.cut_edges.is_empty());
    assert_eq!(chain.components, vec![first.clone()]);

    // A singleton leaf is not an eligible side even when the other side
    // is a valid multi-view point.
    let singleton_observation = (4, keypoint_for_point(4, visible_points[1]));
    let mut singleton_observations = first.clone();
    singleton_observations.push(singleton_observation);
    let singleton_edges = chain_edges
        .into_iter()
        .chain(std::iter::once(edge(first[3], singleton_observation)))
        .collect::<Vec<_>>();
    let singleton = pose_guided_bridge_cut_component(
        &scene.camera,
        &features,
        &poses,
        &singleton_observations,
        &singleton_edges,
        &config,
    );
    assert!(singleton.cut_edges.is_empty());
    assert_eq!(singleton.components, vec![singleton_observations]);
}

#[test]
fn final_track_length_gate_removes_only_short_tracks_and_preserves_support() {
    let tracks = vec![
        vec![(0, 0), (1, 0)],
        vec![(0, 1), (1, 1), (2, 1)],
        vec![(0, 2), (1, 2), (2, 2), (3, 2)],
    ];
    let points = vec![
        Some(Point3::new(0.0, 0.0, 1.0)),
        Some(Point3::new(0.1, 0.0, 1.0)),
        Some(Point3::new(0.2, 0.0, 1.0)),
    ];
    let tracks_before = tracks.clone();
    let points_before = points.clone();
    let mut no_op_tracks = tracks.clone();
    let mut no_op_points = points.clone();
    assert_eq!(
        retain_final_track_length(&mut no_op_tracks, &mut no_op_points, 2),
        (0, 0)
    );
    assert_eq!(no_op_tracks, tracks_before);
    assert_eq!(no_op_points, points_before);

    let mut filtered_tracks = tracks;
    let mut filtered_points = points;
    assert_eq!(
        retain_final_track_length(&mut filtered_tracks, &mut filtered_points, 3),
        (1, 2)
    );
    assert_eq!(
        filtered_tracks,
        vec![
            vec![(0, 1), (1, 1), (2, 1)],
            vec![(0, 2), (1, 2), (2, 2), (3, 2)],
        ]
    );
    assert_eq!(filtered_points, vec![points_before[1], points_before[2]]);

    let pose = || Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let poses = vec![Some(pose()), Some(pose()), Some(pose()), Some(pose())];
    assert!(final_track_length_support_is_valid(
        &filtered_tracks,
        &poses,
        &filtered_points,
    ));
    let unsupported = vec![vec![(0, 1), (1, 1), (2, 1)], vec![(0, 2), (1, 2), (2, 2)]];
    let unsupported_points = vec![Some(Point3::new(0.0, 0.0, 1.0)); 2];
    assert!(!final_track_length_support_is_valid(
        &unsupported,
        &poses,
        &unsupported_points,
    ));

    // Repeating the same operation is deterministic and keeps the same
    // point/track pairing, which is the only state the final BA consumes.
    let mut repeat_tracks = tracks_before;
    let mut repeat_points = points_before;
    let first = retain_final_track_length(&mut repeat_tracks, &mut repeat_points, 3);
    let mut repeat_tracks_again = vec![
        vec![(0, 0), (1, 0)],
        vec![(0, 1), (1, 1), (2, 1)],
        vec![(0, 2), (1, 2), (2, 2), (3, 2)],
    ];
    let mut repeat_points_again = vec![
        Some(Point3::new(0.0, 0.0, 1.0)),
        Some(Point3::new(0.1, 0.0, 1.0)),
        Some(Point3::new(0.2, 0.0, 1.0)),
    ];
    let second = retain_final_track_length(&mut repeat_tracks_again, &mut repeat_points_again, 3);
    assert_eq!(first, second);
    assert_eq!(repeat_tracks, repeat_tracks_again);
    assert_eq!(repeat_points, repeat_points_again);
}

#[test]
fn pose_guided_graph_support_rejects_single_bridge_and_keeps_two_view_fallback() {
    let scene = build_scene();
    let mut visible_points = scene
        .points
        .iter()
        .filter(|point| {
            (0..4).all(|image| project(&scene.camera, &scene.poses[image], point).is_some())
        })
        .copied();
    let point_a = visible_points
        .next()
        .expect("a point visible in four views");
    let point_b = visible_points
        .next()
        .expect("a second point visible in four views");
    let mut features = Vec::new();
    for image in 0..4 {
        features.push(
            FeatureSet::new(
                vec![
                    project(&scene.camera, &scene.poses[image], &point_a).unwrap(),
                    project(&scene.camera, &scene.poses[image], &point_b).unwrap(),
                ],
                vec![vec![0.0f32], vec![1.0]],
            )
            .unwrap(),
        );
    }

    // Every physical point has two independent supports for every
    // additional view.  One cross-point edge is deliberately present, but
    // it cannot win over a geometrically valid multi-view hypothesis.
    let mut pairwise = Vec::new();
    for image_i in 0..4 {
        for image_j in (image_i + 1)..4 {
            let mut matches = vec![(0, 0), (1, 1)];
            if image_i == 0 && image_j == 1 {
                matches.push((0, 1));
            }
            pairwise.push(PairwiseMatches::new(image_i, image_j, matches));
        }
    }
    let component = (0..4)
        .flat_map(|image| [(image, 0), (image, 1)])
        .collect::<Vec<_>>();
    let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
    let config = IncrementalSfmConfig {
        min_track_length: 2,
        max_reprojection_error_px: 4.0,
        conflict_recovery_max_hypotheses: 32,
        pose_guided_track_splitting: true,
        pose_guided_graph_support: true,
        ..IncrementalSfmConfig::default()
    };
    let first = pose_guided_split_tracks(
        &scene.camera,
        &features,
        &pairwise,
        &[],
        std::slice::from_ref(&component),
        &[],
        &poses,
        &config,
    )
    .expect("complete poses should enable graph-supported splitting");
    assert_eq!(first.stats.emitted_tracks, 2);
    assert_eq!(first.stats.graph_supported_tracks, 2);
    assert!(first.stats.graph_support_histogram[2] > 0);
    assert!(first
        .tracks
        .iter()
        .all(|track| track.iter().filter(|(image, _)| *image == 0).count() <= 1));
    assert!(first
        .tracks
        .iter()
        .any(|track| track == &vec![(0, 0), (1, 0), (2, 0), (3, 0)]));
    assert!(first
        .tracks
        .iter()
        .any(|track| track == &vec![(0, 1), (1, 1), (2, 1), (3, 1)]));
    assert!(first
        .tracks
        .iter()
        .all(|track| !track.contains(&(0, 0)) || !track.contains(&(1, 1))));

    // A component with no third-view support retains its genuine two-view
    // fallback rather than being dropped by the admission rule.
    let two_view = pose_guided_split_tracks(
        &scene.camera,
        &features,
        &[PairwiseMatches::new(0, 1, vec![(0, 0)])],
        &[],
        &[vec![(0, 0), (1, 0)]],
        &[],
        &poses,
        &config,
    )
    .expect("two-view fallback should remain deterministic");
    assert_eq!(two_view.tracks, vec![vec![(0, 0), (1, 0)]]);
    assert_eq!(two_view.stats.graph_length_two_tracks, 1);

    let reordered_pairwise = pairwise
        .iter()
        .rev()
        .map(|pair| {
            let mut pair = pair.clone();
            pair.matches.reverse();
            pair
        })
        .collect::<Vec<_>>();
    let mut reversed_component = component;
    reversed_component.reverse();
    let second = pose_guided_split_tracks(
        &scene.camera,
        &features,
        &reordered_pairwise,
        &[],
        std::slice::from_ref(&reversed_component),
        &[],
        &poses,
        &config,
    )
    .expect("graph-supported split should be permutation invariant");
    assert_eq!(first.tracks, second.tracks);
    assert_eq!(first.points, second.points);
    assert_eq!(first.stats, second.stats);
}

/// M2 acceptance test, end-to-end: running the *full* `incremental_sfm`
/// pipeline with [`TrackSource::CorrespondenceGraph`] instead of the
/// default [`TrackSource::UnionFind`] on the same synthetic scene must
/// register the same images and produce the same track count and mean
/// reprojection error — i.e. the track-builder swap is invisible to
/// every downstream stage (seeding, growth, bundle adjustment).
#[test]
fn incremental_sfm_matches_between_track_sources() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);

    let base_config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        ..IncrementalSfmConfig::default()
    };
    let union_find_config = IncrementalSfmConfig {
        track_source: TrackSource::UnionFind,
        ..base_config.clone()
    };
    let graph_config = IncrementalSfmConfig {
        track_source: TrackSource::CorrespondenceGraph,
        ..base_config
    };

    let union_find_result =
        incremental_sfm(&scene.camera, &features, &pairwise, &union_find_config)
            .expect("union-find track source must reconstruct this scene");
    let graph_result = incremental_sfm(&scene.camera, &features, &pairwise, &graph_config)
        .expect("CorrespondenceGraph track source must reconstruct this scene");

    assert_eq!(
        union_find_result.registered_images, graph_result.registered_images,
        "both track sources must register the same number of images"
    );
    assert_eq!(
        union_find_result.tracks.len(),
        graph_result.tracks.len(),
        "both track sources must produce the same number of output tracks"
    );
    assert!(
        (union_find_result.mean_reprojection_px - graph_result.mean_reprojection_px).abs() < 1.0e-6,
        "both track sources must reach the same mean reprojection error: {} vs {}",
        union_find_result.mean_reprojection_px,
        graph_result.mean_reprojection_px,
    );
}

#[test]
fn seeded_incremental_growth_keeps_supplied_poses_fixed_and_registers_missing_images() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let initial_poses = (0..scene.poses.len())
        .map(|image| (image < 2).then(|| scene.poses[image].clone()))
        .collect::<Vec<_>>();
    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        // Keep this test focused on the staged growth phase. The normal
        // final-BA release is covered by the following test.
        final_global_ba: false,
        ba_every: 1,
        ..IncrementalSfmConfig::default()
    };

    let result = incremental_sfm_with_initial_poses(
        &scene.camera,
        &features,
        &pairwise,
        &config,
        Some(&initial_poses),
    )
    .expect("two exact supplied poses must seed the synthetic scene");

    assert!(
        result.registered_images >= 3,
        "the missing-camera PnP loop must grow beyond the supplied seed"
    );
    assert_eq!(result.poses[0], initial_poses[0]);
    assert_eq!(result.poses[1], initial_poses[1]);

    // The public legacy entry point remains the same path as an explicit
    // `None` staged seed. Compare the observable result rather than the
    // internal timing fields so this also guards the default no-op.
    let legacy = incremental_sfm(&scene.camera, &features, &pairwise, &config)
        .expect("legacy synthetic reconstruction must still succeed");
    let explicit_none =
        incremental_sfm_with_initial_poses(&scene.camera, &features, &pairwise, &config, None)
            .expect("explicit None must use the legacy growth path");
    assert_eq!(legacy.poses, explicit_none.poses);
    assert_eq!(legacy.tracks, explicit_none.tracks);
    assert_eq!(legacy.registered_images, explicit_none.registered_images);
    assert_eq!(
        legacy.mean_reprojection_px,
        explicit_none.mean_reprojection_px
    );
}

#[test]
fn seeded_growth_is_deterministic_and_final_ba_is_run_after_fixed_phase() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let initial_poses = (0..scene.poses.len())
        .map(|image| (image < 2).then(|| scene.poses[image].clone()))
        .collect::<Vec<_>>();
    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        track_filter_iterations: 0,
        ..IncrementalSfmConfig::default()
    };

    let first = incremental_sfm_with_initial_poses(
        &scene.camera,
        &features,
        &pairwise,
        &config,
        Some(&initial_poses),
    )
    .expect("seeded final refinement must succeed");
    let second = incremental_sfm_with_initial_poses(
        &scene.camera,
        &features,
        &pairwise,
        &config,
        Some(&initial_poses),
    )
    .expect("the same staged input must be repeatable");
    assert_eq!(first.poses, second.poses);
    assert_eq!(first.tracks, second.tracks);
    assert_eq!(first.registered_images, second.registered_images);
    assert_eq!(first.mean_reprojection_px, second.mean_reprojection_px);
    assert!(
        first.ba_result.is_some(),
        "normal final BA must still run after staged growth, releasing non-gauge poses"
    );
    assert!(first.registered_images >= 3);
}

#[test]
fn seeded_incremental_rejects_short_or_nonfinite_pose_vectors() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        final_global_ba: false,
        ..IncrementalSfmConfig::default()
    };
    let short = vec![Some(scene.poses[0].clone())];
    assert!(matches!(
        incremental_sfm_with_initial_poses(
            &scene.camera,
            &features,
            &pairwise,
            &config,
            Some(&short),
        ),
        Err(IncrementalSfmError::InvalidInitialPoses(_))
    ));

    let mut one_seed = vec![None; scene.poses.len()];
    one_seed[0] = Some(scene.poses[0].clone());
    assert!(matches!(
        incremental_sfm_with_initial_poses(
            &scene.camera,
            &features,
            &pairwise,
            &config,
            Some(&one_seed),
        ),
        Err(IncrementalSfmError::InvalidInitialPoses(_))
    ));

    let mut nonfinite = vec![None; scene.poses.len()];
    nonfinite[0] = Some(scene.poses[0].clone());
    nonfinite[1] = Some(Pose::from_world_to_camera(
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(f64::NAN, 0.0, 0.0, 1.0)),
        Vector3::zeros(),
    ));
    assert!(matches!(
        incremental_sfm_with_initial_poses(
            &scene.camera,
            &features,
            &pairwise,
            &config,
            Some(&nonfinite),
        ),
        Err(IncrementalSfmError::InvalidInitialPoses(_))
    ));
}

/// M2.1 acceptance: `docs/colmap_port_plan.md`'s M2.1 milestone widens
/// `examples/unordered_sfm_demo.rs`'s verified-pair keep-list so a
/// `PANORAMIC` (pure-rotation, zero-baseline) pair now reaches
/// `PairwiseMatches`/this mapper, matching COLMAP's own
/// `database_cache.cc` `UseInlierMatchesCheck` gate. This must not make
/// such a pair *seedable*: COLMAP's own
/// `IncrementalMapperImpl::EstimateInitialTwoViewGeometry` re-derives its
/// own relative pose and rejects init candidates whose triangulation
/// angle doesn't clear `init_min_tri_angle`, independent of any stored
/// `ConfigurationType` — this mapper's [`place_seed_pair`] already has
/// the same independent architecture (re-estimate the relative pose,
/// gate on how many inliers actually triangulate), so no new exclusion
/// mechanism is needed; this test pins that the existing gate covers the
/// newly-admitted pair type too.
#[test]
fn pure_rotation_pair_is_rejected_as_a_seed_even_though_it_now_reaches_pairwise() {
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    // A scattered point cloud with real depth variation (same shape as
    // `colmap_verification.rs`'s `general_scene_points` fixture).
    let mut points = Vec::new();
    for i in 0..6 {
        for j in 0..4 {
            points.push(Point3::new(
                -1.5 + 0.6 * i as f64,
                -1.0 + 0.7 * j as f64,
                3.0 + 0.8 * ((i + j) % 5) as f64,
            ));
        }
    }

    // Camera 0 at the world origin; camera 1 at the SAME origin, only
    // rotated — a pure-rotation pair, zero baseline, exactly the
    // `PANORAMIC` configuration `TwoViewGeometryVerifier` would classify
    // this as (see `colmap_verification.rs`'s
    // `pure_rotation_classifies_panoramic`).
    let pose0 = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.12);
    let pose1 = Pose::from_world_to_camera(yaw, Vector3::zeros());

    let mut kp0 = Vec::new();
    let mut kp1 = Vec::new();
    let mut matches = Vec::new();
    for p in &points {
        if let (Some(px0), Some(px1)) = (project(&camera, &pose0, p), project(&camera, &pose1, p)) {
            matches.push((kp0.len(), kp1.len()));
            kp0.push(px0);
            kp1.push(px1);
        }
    }
    assert!(
        matches.len() >= 15,
        "fixture sanity: pure rotation should still leave most points in both views"
    );

    let features = vec![
        FeatureSet::new(kp0, vec![vec![0.0f32; 4]; matches.len()]).unwrap(),
        FeatureSet::new(kp1, vec![vec![0.0f32; 4]; matches.len()]).unwrap(),
    ];
    let pair = PairwiseMatches {
        image_i: 0,
        image_j: 1,
        matches,
        two_view_config: None,
        essential_matches: None,
        essential_matrix: None,
    };
    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        ..IncrementalSfmConfig::default()
    };
    let mut poses = vec![None, None];
    assert!(
        !place_seed_pair(&camera, &features, &pair, &config, &mut poses),
        "a zero-baseline (panoramic) pair must never bootstrap a seed, \
             even though M2.1 now lets its correspondences reach PairwiseMatches"
    );
    assert!(
        poses[0].is_none() && poses[1].is_none(),
        "rejected seed must leave poses untouched"
    );
}

#[test]
fn build_tracks_merges_shared_observations() {
    // Two images both see point P (kp 0 in each) and image-2 sees it too.
    let pairwise = vec![
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
        PairwiseMatches {
            image_i: 1,
            image_j: 2,
            matches: vec![(0, 0)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
    ];
    let tracks = build_tracks(3, &pairwise, 2);
    assert_eq!(tracks.len(), 1, "the chained matches form one track");
    assert_eq!(tracks[0].len(), 3, "track spans all three images");
}

#[test]
fn track_build_preview_matches_union_find_topology_without_mapping() {
    let pairwise = vec![
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
        PairwiseMatches {
            image_i: 1,
            image_j: 2,
            matches: vec![(0, 0)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
    ];
    let features = vec![
        FeatureSet {
            keypoints: vec![Point2::new(0.0, 0.0)],
            descriptors: vec![vec![0.0]],
        };
        3
    ];
    let stats = preview_track_build_stats(&features, &pairwise, &IncrementalSfmConfig::default());
    assert_eq!(stats.input_correspondences, 2);
    assert_eq!(stats.connected_components, 1);
    assert_eq!(stats.retained_tracks, 1);
    assert_eq!(stats.retained_observations, 3);
}

#[test]
fn incremental_correspondence_tracks_create_continue_merge_and_are_permutation_invariant() {
    let features = vec![
        FeatureSet {
            keypoints: vec![Point2::new(0.0, 0.0)],
            descriptors: vec![vec![0.0]],
        };
        4
    ];
    let pairwise = vec![
        PairwiseMatches::new(2, 3, vec![(0, 0)]),
        PairwiseMatches::new(1, 2, vec![(0, 0)]),
        PairwiseMatches::new(0, 1, vec![(0, 0)]),
    ];
    let output = build_tracks_incremental_correspondence(&features, &pairwise, 2);
    assert_eq!(output.stats.connected_components, 1);
    assert_eq!(output.stats.retained_tracks, 1);
    assert_eq!(output.tracks, vec![vec![(0, 0), (1, 0), (2, 0), (3, 0)]]);

    let mut permuted = pairwise;
    permuted.reverse();
    permuted[0].matches.reverse();
    let permuted_output = build_tracks_incremental_correspondence(&features, &permuted, 2);
    assert_eq!(output.tracks, permuted_output.tracks);
    assert_eq!(output.stats, permuted_output.stats);
}

#[test]
fn incremental_correspondence_rejects_only_conflicting_edge() {
    let features = vec![
        FeatureSet {
            keypoints: vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
            descriptors: vec![vec![0.0], vec![1.0]],
        };
        2
    ];
    let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0), (0, 1)])];
    let output = build_tracks_incremental_correspondence(&features, &pairwise, 2);
    assert_eq!(output.stats.conflicting_components, 1);
    assert_eq!(output.stats.conflicting_observations, 2);
    assert_eq!(output.stats.retained_tracks, 1);
    assert_eq!(output.stats.retained_observations, 2);
}

#[test]
fn correspondence_point_state_enforces_conflicts_and_retriangulates() {
    let mut state = CorrespondencePointState::default();
    let first = state
        .create_point(&[(0, 0), (1, 0)], Point3::new(0.0, 0.0, 1.0))
        .unwrap();
    assert!(state.continue_point(first, (2, 0)));
    assert!(!state.continue_point(first, (1, 1)));
    let second = state
        .create_point(&[(3, 0), (4, 0)], Point3::new(1.0, 0.0, 1.0))
        .unwrap();
    assert!(state.merge_points(first, second, Point3::new(0.5, 0.0, 1.0)));
    let conflicting = state
        .create_point(&[(1, 99), (6, 0)], Point3::new(2.0, 0.0, 1.0))
        .unwrap();
    assert!(!state.merge_points(first, conflicting, Point3::new(0.0, 0.0, 1.0)));
    assert!(state.retriangulate_point(first, Point3::new(0.25, 0.0, 1.0)));
    assert_eq!(state.points[first], Some(Point3::new(0.25, 0.0, 1.0)));
    assert!(!state.retriangulate_point(first, Point3::new(f64::NAN, 0.0, 1.0)));
}

#[test]
fn incremental_correspondence_mode_is_default_noop() {
    let features = vec![
        FeatureSet {
            keypoints: vec![Point2::new(0.0, 0.0)],
            descriptors: vec![vec![0.0]],
        };
        2
    ];
    let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0)])];
    let config = IncrementalSfmConfig::default();
    let default_output = build_track_output(&features, &pairwise, &config, None);
    let legacy_output = build_tracks_detailed(features.len(), &pairwise, config.min_track_length);
    assert_eq!(default_output.tracks, legacy_output.tracks);
    assert_eq!(default_output.stats, legacy_output.stats);
    assert!(!config.incremental_correspondence_triangulation);
}

#[test]
fn build_tracks_drops_same_image_conflict() {
    // kp0 and kp1 of image 1 get merged into one component -> inconsistent.
    let pairwise = vec![
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 1)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
    ];
    let (tracks, stats) = build_tracks_with_stats(2, &pairwise, 2);
    assert!(tracks.is_empty(), "same-image conflict track is dropped");
    assert_eq!(stats.input_correspondences, 2);
    assert_eq!(stats.connected_components, 1);
    assert_eq!(stats.conflicting_components, 1);
    assert_eq!(stats.conflicting_observations, 3);
    assert_eq!(stats.retained_tracks, 0);
    assert_eq!(stats.retained_observations, 0);
}

#[test]
fn stable_track_order_is_permutation_invariant_at_coordinate_level() {
    let features = vec![
        FeatureSet::new(
            vec![Point2::new(20.0, 0.0), Point2::new(10.0, 0.0)],
            vec![vec![20.0], vec![10.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![Point2::new(20.0, 1.0), Point2::new(10.0, 1.0)],
            vec![vec![20.0], vec![10.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![Point2::new(20.0, 2.0), Point2::new(10.0, 2.0)],
            vec![vec![20.0], vec![10.0]],
        )
        .unwrap(),
    ];
    let pairwise = vec![
        PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
    ];

    let permutations = [[1usize, 0], [1, 0], [1, 0]];
    let permuted_features: Vec<FeatureSet> = features
        .iter()
        .zip(permutations)
        .map(|(set, permutation)| {
            FeatureSet::new(
                permutation
                    .iter()
                    .map(|&index| set.keypoints[index])
                    .collect(),
                permutation
                    .iter()
                    .map(|&index| set.descriptors[index].clone())
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let remapped = pairwise
        .iter()
        .map(|pair| {
            PairwiseMatches::new(
                pair.image_i,
                pair.image_j,
                pair.matches
                    .iter()
                    .map(|&(lhs, rhs)| {
                        (
                            permutations[pair.image_i][lhs],
                            permutations[pair.image_j][rhs],
                        )
                    })
                    .rev()
                    .collect(),
            )
        })
        .rev()
        .collect::<Vec<_>>();
    let config = IncrementalSfmConfig {
        stable_track_order: true,
        ..IncrementalSfmConfig::default()
    };
    let original = build_track_output(&features, &pairwise, &config, None).tracks;
    let permuted = build_track_output(&permuted_features, &remapped, &config, None).tracks;
    let physical = |sets: &[FeatureSet], tracks: &[Vec<(usize, usize)>]| {
        let mut output: Vec<Vec<(usize, i64, i64)>> = tracks
            .iter()
            .map(|track| {
                let mut observations = track
                    .iter()
                    .map(|&(image, index)| {
                        let point = sets[image].keypoints[index];
                        (
                            image,
                            (point.x * 1_000_000.0).round() as i64,
                            (point.y * 1_000_000.0).round() as i64,
                        )
                    })
                    .collect::<Vec<_>>();
                observations.sort_unstable();
                observations
            })
            .collect();
        output.sort_unstable();
        output
    };
    assert_eq!(
        physical(&features, &original),
        physical(&permuted_features, &permuted),
        "physical track components must not depend on feature/match order"
    );
}

#[test]
fn confidence_ordered_tracks_keep_strong_multiview_chain() {
    // The weak edge maps image 0's point to the wrong image-1 keypoint.
    // Legacy union-find merges it into the good chain and drops the whole
    // component; confidence ordering accepts the two stronger pair sets
    // first and rejects only that conflicting edge.
    let weak_conflict = PairwiseMatches::new(0, 1, vec![(0, 1)]);
    let strong_01 = PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]);
    let strong_12 = PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]);
    let pairwise = vec![weak_conflict, strong_12, strong_01];

    let legacy = build_tracks(3, &pairwise, 2);
    assert!(legacy.is_empty(), "the weak chain should poison legacy UF");
    let default_config = IncrementalSfmConfig::default();
    assert!(!default_config.confidence_ordered_tracks);
    assert_eq!(
        build_track_output(
            &dummy_features(&[2, 2, 2]),
            &pairwise,
            &default_config,
            None,
        )
        .tracks,
        legacy,
        "the confidence policy must remain opt-in"
    );

    let ordered = build_tracks_confidence_ordered(3, &pairwise, 2);
    assert_eq!(ordered.stats.retained_tracks, 2);
    assert_eq!(ordered.stats.retained_observations, 6);
    assert_eq!(
        ordered.tracks,
        vec![vec![(0, 0), (1, 0), (2, 0)], vec![(0, 1), (1, 1), (2, 1)],]
    );
}

#[test]
fn confidence_ordered_tracks_are_permutation_deterministic() {
    let pairwise = vec![
        PairwiseMatches::new(0, 1, vec![(0, 1)]),
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
    ];
    let expected = build_tracks_confidence_ordered(3, &pairwise, 2).tracks;
    for permutation in [vec![2, 0, 1], vec![1, 2, 0], vec![0, 2, 1]] {
        let reordered = permutation
            .into_iter()
            .map(|index| pairwise[index].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            build_tracks_confidence_ordered(3, &reordered, 2).tracks,
            expected
        );
    }

    let reversed_orientation = [
        PairwiseMatches::new(1, 0, vec![(1, 0)]),
        PairwiseMatches::new(2, 1, vec![(1, 1), (0, 0)]),
        PairwiseMatches::new(1, 0, vec![(1, 1), (0, 0)]),
    ];
    for permutation in [vec![1, 2, 0], vec![0, 2, 1]] {
        let reordered = permutation
            .into_iter()
            .map(|index| reversed_orientation[index].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            build_tracks_confidence_ordered(3, &reordered, 2).tracks,
            expected,
            "pair orientation and match order must not change the builder"
        );
    }
}

#[test]
fn pair_confidence_conflict_preview_reports_zero_conflicts() {
    let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)])];
    let stats = preview_pair_confidence_conflicts(2, &pairwise, 0);
    assert_eq!(
        stats,
        PairConfidenceConflictStats {
            correspondences: 2,
            nodes: 4,
            accepted_edges: 2,
            rejected_edges: 0,
            final_components: 2,
            conflict_regions: 0,
            involved_components: 0,
            involved_observations: 0,
            max_region_components: 0,
            max_region_observations: 0,
            max_overlapping_images_per_rejected_edge: 0,
            region_component_count_histogram: Vec::new(),
        }
    );
}

#[test]
fn pair_confidence_conflict_preview_reports_one_region() {
    let pairwise = vec![
        PairwiseMatches::new(0, 1, vec![(0, 1)]),
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
    ];
    let stats = preview_pair_confidence_conflicts(3, &pairwise, 0);
    assert_eq!(stats.correspondences, 5);
    assert_eq!(stats.nodes, 6);
    assert_eq!(stats.accepted_edges, 4);
    assert_eq!(stats.rejected_edges, 1);
    assert_eq!(stats.final_components, 2);
    assert_eq!(stats.conflict_regions, 1);
    assert_eq!(stats.involved_components, 2);
    assert_eq!(stats.involved_observations, 6);
    assert_eq!(stats.max_region_components, 2);
    assert_eq!(stats.max_region_observations, 6);
    assert_eq!(stats.max_overlapping_images_per_rejected_edge, 3);
    assert_eq!(stats.region_component_count_histogram, vec![(2, 1)]);
}

#[test]
fn pair_confidence_conflict_preview_chained_rejections_share_one_region() {
    let pairwise = vec![PairwiseMatches::new(
        0,
        1,
        vec![(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    )];
    let stats = preview_pair_confidence_conflicts(2, &pairwise, 0);
    assert_eq!(stats.correspondences, 5);
    assert_eq!(stats.nodes, 6);
    assert_eq!(stats.accepted_edges, 3);
    assert_eq!(stats.rejected_edges, 2);
    assert_eq!(stats.final_components, 3);
    assert_eq!(stats.conflict_regions, 1);
    assert_eq!(stats.involved_components, 3);
    assert_eq!(stats.involved_observations, 6);
    assert_eq!(stats.max_region_components, 3);
    assert_eq!(stats.max_region_observations, 6);
    assert_eq!(stats.max_overlapping_images_per_rejected_edge, 1);
    assert_eq!(stats.region_component_count_histogram, vec![(3, 1)]);
}

#[test]
fn pair_confidence_conflict_preview_is_permutation_deterministic() {
    let pairwise = vec![
        PairwiseMatches::new(0, 1, vec![(0, 1)]),
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
    ];
    let expected = preview_pair_confidence_conflicts(3, &pairwise, 0);
    for permutation in [vec![2, 0, 1], vec![1, 2, 0], vec![0, 2, 1]] {
        let reordered = permutation
            .into_iter()
            .map(|index| pairwise[index].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            preview_pair_confidence_conflicts(3, &reordered, 0),
            expected
        );
    }

    let reversed_orientation = [
        PairwiseMatches::new(1, 0, vec![(1, 0)]),
        PairwiseMatches::new(2, 1, vec![(1, 1), (0, 0)]),
        PairwiseMatches::new(1, 0, vec![(1, 1), (0, 0)]),
    ];
    for permutation in [vec![1, 2, 0], vec![0, 2, 1]] {
        let reordered = permutation
            .into_iter()
            .map(|index| reversed_orientation[index].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            preview_pair_confidence_conflicts(3, &reordered, 0),
            expected,
            "pair orientation and match order must not change the preview"
        );
    }
}

#[test]
fn geometry_observation_weights_are_parallax_ordered_and_deterministic() {
    let poses = vec![
        Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::zeros(),
        )),
        Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(-1.0, 0.0, 0.0),
        )),
    ];
    let tracks = vec![vec![(0, 0), (1, 0)], vec![(0, 1), (1, 1)]];
    let points = vec![
        Some(Point3::new(0.0, 0.0, 100.0)), // weak baseline information
        Some(Point3::new(0.0, 0.0, 2.0)),   // strong baseline information
    ];
    let observations = vec![
        BaObservation {
            keyframe_id: 0,
            landmark_id: 0,
            xy: Point2::origin(),
        },
        BaObservation {
            keyframe_id: 1,
            landmark_id: 0,
            xy: Point2::origin(),
        },
        BaObservation {
            keyframe_id: 0,
            landmark_id: 1,
            xy: Point2::origin(),
        },
        BaObservation {
            keyframe_id: 1,
            landmark_id: 1,
            xy: Point2::origin(),
        },
    ];
    let weights = track_geometry_observation_weights(&poses, &tracks, &points, &observations);
    assert_eq!(weights.len(), observations.len());
    assert!(weights[0] < weights[2]);
    assert_eq!(weights[0], weights[1]);
    assert_eq!(weights[2], weights[3]);
    assert!((0.25..=4.0).contains(&weights[0]));
    assert!((0.25..=4.0).contains(&weights[2]));
    assert_eq!(
        weights,
        track_geometry_observation_weights(&poses, &tracks, &points, &observations)
    );
    assert!(!IncrementalSfmConfig::default().geometry_weighted_ba);
}

#[test]
fn ill_conditioned_landmark_gate_is_deterministic_and_freeze_preserves_point() {
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let make_pose =
        |centre: Vector3<f64>| Pose::from_world_to_camera(UnitQuaternion::identity(), -centre);
    let poses = vec![
        Some(make_pose(Vector3::zeros())),
        Some(make_pose(Vector3::new(1.0, 0.0, 0.0))),
    ];
    let point = Point3::new(0.0, 0.0, 2.0);
    let pixels: Vec<Point2<f64>> = poses
        .iter()
        .map(|pose| {
            camera
                .project(&pose.as_ref().unwrap().transform_world_point(&point))
                .unwrap()
        })
        .collect();
    let features = vec![
        FeatureSet::new(vec![pixels[0]], vec![vec![0.0f32]]).unwrap(),
        FeatureSet::new(vec![pixels[1]], vec![vec![0.0f32]]).unwrap(),
    ];
    let track = vec![(0, 0), (1, 0)];
    let healthy = ba_landmark_geometry(
        &camera,
        &features,
        &poses,
        &track,
        &point,
        &RobustKernel::None,
    );
    assert_eq!(healthy.track_length, 2);
    assert!(healthy.point_condition.is_finite());
    assert!(!ba_landmark_is_ill_conditioned(&healthy, 2.0));
    assert_eq!(
        healthy,
        ba_landmark_geometry(
            &camera,
            &features,
            &poses,
            &track,
            &point,
            &RobustKernel::None,
        )
    );

    let weak_poses = vec![
        Some(make_pose(Vector3::zeros())),
        Some(make_pose(Vector3::new(1.0e-6, 0.0, 0.0))),
    ];
    let weak_pixels: Vec<Point2<f64>> = weak_poses
        .iter()
        .map(|pose| {
            camera
                .project(&pose.as_ref().unwrap().transform_world_point(&point))
                .unwrap()
        })
        .collect();
    let weak_features = vec![
        FeatureSet::new(vec![weak_pixels[0]], vec![vec![0.0f32]]).unwrap(),
        FeatureSet::new(vec![weak_pixels[1]], vec![vec![0.0f32]]).unwrap(),
    ];
    let weak = ba_landmark_geometry(
        &camera,
        &weak_features,
        &weak_poses,
        &track,
        &point,
        &RobustKernel::None,
    );
    assert!(weak.point_condition > BA_POINT_BLOCK_MAX_CONDITION);
    assert!(ba_landmark_is_ill_conditioned(&weak, 2.0));
    assert!(!ba_landmark_should_exclude(&weak, 2.0, 4.0));
    let weak_bad = LandmarkBaGeometry {
        median_reprojection_px: 5.0,
        ..weak
    };
    assert!(ba_landmark_should_exclude(&weak_bad, 2.0, 4.0));

    // The freeze is implemented through BundleAdjustment's existing fixed
    // landmark semantics: residual rows remain present, while the point
    // has no Schur variable. A healthy point is free to move instead.
    let initial = Point3::new(0.2, 0.1, 2.2);
    let mut healthy_ba = BundleAdjustment::new(camera);
    for (id, pose) in poses.iter().enumerate() {
        healthy_ba.add_pose(id as u64, pose.as_ref().unwrap().clone());
        healthy_ba.fix_pose(id as u64);
    }
    healthy_ba.add_landmark(0, initial);
    for (id, pixel) in pixels.iter().enumerate() {
        healthy_ba.add_observation(BaObservation {
            keyframe_id: id as u64,
            landmark_id: 0,
            xy: *pixel,
        });
    }
    healthy_ba
        .optimize(&BaConfig {
            robust_kernel: RobustKernel::None,
            ..BaConfig::default()
        })
        .unwrap();
    assert!((healthy_ba.landmarks[&0].coords - initial.coords).norm() > 1.0e-6);

    let mut frozen_ba = healthy_ba.clone();
    frozen_ba.landmarks.insert(0, initial);
    frozen_ba.fix_landmark(0);
    // Leave one pose variable so the solver has a legitimate camera block;
    // the test is about the landmark variable being absent, not an
    // all-fixed empty solve.
    frozen_ba.fixed_poses.remove(&1);
    frozen_ba
        .optimize(&BaConfig {
            robust_kernel: RobustKernel::None,
            ..BaConfig::default()
        })
        .unwrap();
    assert_eq!(frozen_ba.landmarks[&0], initial);
    assert!(!IncrementalSfmConfig::default().freeze_ill_conditioned_landmarks);
}

#[test]
fn landmark_ba_warm_start_is_camera_fixed_monotone_and_deterministic() {
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let make_pose =
        |centre: Vector3<f64>| Pose::from_world_to_camera(UnitQuaternion::identity(), -centre);
    let poses = vec![
        Some(make_pose(Vector3::zeros())),
        Some(make_pose(Vector3::new(1.0, 0.0, 0.0))),
    ];
    let truth = Point3::new(0.15, -0.08, 2.4);
    let pixels: Vec<Point2<f64>> = poses
        .iter()
        .map(|pose| {
            camera
                .project(&pose.as_ref().unwrap().transform_world_point(&truth))
                .unwrap()
        })
        .collect();
    let features = vec![
        FeatureSet::new(vec![pixels[0]], vec![vec![0.0f32]]).unwrap(),
        FeatureSet::new(vec![pixels[1]], vec![vec![0.0f32]]).unwrap(),
    ];
    let tracks = vec![vec![(0, 0), (1, 0)]];
    let initial = Point3::new(0.35, 0.12, 2.8);
    let mut config = IncrementalSfmConfig {
        landmark_ba_warm_start_iterations: 5,
        ..IncrementalSfmConfig::default()
    };
    config.ba_config.robust_kernel = RobustKernel::None;

    let mut points_a = vec![Some(initial)];
    let poses_before = poses.clone();
    let stats_a =
        run_landmark_ba_warm_start(&camera, &features, &tracks, &config, &poses, &mut points_a);
    assert!(stats_a.attempted);
    assert!(stats_a.accepted);
    assert!(stats_a.final_cost < stats_a.initial_cost);
    assert!(stats_a.max_displacement > 0.0);
    assert_eq!(poses, poses_before, "warm start must not mutate cameras");

    let mut points_b = vec![Some(initial)];
    let stats_b =
        run_landmark_ba_warm_start(&camera, &features, &tracks, &config, &poses, &mut points_b);
    assert_eq!(stats_a, stats_b);
    assert_eq!(points_a, points_b);

    let mut untouched = vec![Some(initial)];
    let no_op = run_landmark_ba_warm_start(
        &camera,
        &features,
        &tracks,
        &IncrementalSfmConfig::default(),
        &poses,
        &mut untouched,
    );
    assert!(!no_op.attempted);
    assert!(!no_op.accepted);
    assert_eq!(untouched, vec![Some(initial)]);
}

#[test]
fn normalized_sampson_residual_is_stable_and_rejects_invalid_inputs() {
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let essential = Matrix3::new(
        0.0, 0.0, 0.0, // translation along x, identity rotation
        0.0, 0.0, -1.0, 0.0, 1.0, 0.0,
    );
    let centre = Point2::new(320.0, 240.0);
    let off_epipolar_line = Point2::new(320.0, 340.0);
    let good = normalized_sampson_residual(&camera, &essential, &centre, &centre)
        .expect("finite E residual");
    let bad = normalized_sampson_residual(&camera, &essential, &centre, &off_epipolar_line)
        .expect("finite off-line residual");
    assert_eq!(good, 0.0);
    assert!(bad > good && bad.is_finite());

    let mut invalid_essential = essential;
    invalid_essential[(0, 0)] = f64::NAN;
    assert!(normalized_sampson_residual(&camera, &invalid_essential, &centre, &centre).is_none());
    let invalid_point = Point2::new(f64::NAN, 240.0);
    assert!(normalized_sampson_residual(&camera, &essential, &invalid_point, &centre).is_none());
    assert!(normalized_sampson_residual(&camera, &Matrix3::zeros(), &centre, &centre,).is_none());
}

#[test]
fn geometric_confidence_prefers_low_residual_and_is_permutation_deterministic() {
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let centre = Point2::new(320.0, 240.0);
    let off_epipolar_line = Point2::new(320.0, 340.0);
    let features = vec![
        FeatureSet::new(
            vec![centre, centre],
            vec![vec![0.0f32; 2], vec![1.0f32, 0.0]],
        )
        .unwrap(),
        FeatureSet::new(
            vec![centre, off_epipolar_line, off_epipolar_line],
            vec![vec![0.0f32; 2], vec![1.0f32, 0.0], vec![2.0f32, 0.0]],
        )
        .unwrap(),
        FeatureSet::new(vec![centre], vec![vec![0.0f32; 2]]).unwrap(),
    ];
    let essential = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
    // This pair has more matches and therefore wins the old pair-level
    // ordering, but both of its E-supported matches have a larger
    // normalized residual than the single correct edge below.
    let high_support_wrong = PairwiseMatches {
        image_i: 0,
        image_j: 1,
        matches: vec![(0, 1), (1, 2)],
        two_view_config: Some(ConfigurationType::Calibrated),
        essential_matches: Some(vec![(0, 1), (1, 2)]),
        essential_matrix: Some(essential),
    };
    let low_support_correct = PairwiseMatches {
        image_i: 0,
        image_j: 1,
        matches: vec![(0, 0)],
        two_view_config: Some(ConfigurationType::Calibrated),
        essential_matches: Some(vec![(0, 0)]),
        essential_matrix: Some(essential),
    };
    let continuation = PairwiseMatches::new(1, 2, vec![(0, 0)]);
    let pairwise = vec![high_support_wrong, continuation, low_support_correct];

    let legacy = build_tracks_confidence_ordered(3, &pairwise, 2).tracks;
    assert!(!legacy.contains(&vec![(0, 0), (1, 0), (2, 0)]));
    let default_config = IncrementalSfmConfig::default();
    assert!(!default_config.geometric_confidence_tracks);
    assert_eq!(
        build_track_output(&features, &pairwise, &default_config, Some(&camera)).tracks,
        build_tracks(3, &pairwise, 2),
        "the geometric strategy must remain opt-in"
    );
    let geometric = build_tracks_geometric_confidence(&features, &camera, &pairwise, 2);
    assert!(geometric.tracks.contains(&vec![(0, 0), (1, 0), (2, 0)]));

    // Every tie-break after the residual is explicit, so input pair order
    // cannot alter the selected topology.
    let expected = geometric.tracks;
    for permutation in [[2, 1, 0], [1, 0, 2], [0, 2, 1]] {
        let reordered = permutation
            .into_iter()
            .map(|index| pairwise[index].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            build_tracks_geometric_confidence(&features, &camera, &reordered, 2).tracks,
            expected
        );
    }
}

#[test]
fn geometry_recovery_splits_conflict_from_trusted_multiview_poses() {
    let scene = build_scene();
    let (features, mut pairwise) = render(&scene);
    let keypoint_for_point = |image: usize, point: usize| {
        features[image]
            .descriptors
            .iter()
            .position(|descriptor| descriptor[0] == point as f32)
            .unwrap()
    };
    let kp_0_image_0 = keypoint_for_point(0, 0);
    let kp_1_image_1 = keypoint_for_point(1, 1);
    let pair_0_1 = pairwise
        .iter_mut()
        .find(|pair| pair.image_i == 0 && pair.image_j == 1)
        .unwrap();
    // One erroneous bridge merges two otherwise-complete six-view tracks.
    pair_0_1.matches.push((kp_0_image_0, kp_1_image_1));

    let built = build_tracks_detailed(features.len(), &pairwise, 2);
    assert_eq!(built.stats.conflicting_components, 1);
    assert_eq!(built.conflicting_components.len(), 1);

    let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
    let config = IncrementalSfmConfig {
        geometry_guided_conflict_recovery: true,
        conflict_recovery_min_views: 3,
        conflict_recovery_max_hypotheses: 32,
        conflict_recovery_max_reprojection_error_px: 0.1,
        conflict_recovery_max_mean_reprojection_px: 0.05,
        ..IncrementalSfmConfig::default()
    };
    let recovered = recover_conflict_tracks_geometry(
        &scene.camera,
        &features,
        &pairwise,
        &built.conflicting_components,
        &poses,
        &config,
    );
    assert_eq!(
        recovered.len(),
        1,
        "first slice keeps one guarded hypothesis"
    );
    let track = &recovered[0];
    assert!(track.registered_observations >= 3);
    assert!(track.mean_reprojection_px < 1e-6);
    let unique_images: HashSet<_> = track.observations.iter().map(|&(image, _)| image).collect();
    assert_eq!(unique_images.len(), track.observations.len());
    let nearest_truth = scene
        .points
        .iter()
        .take(2)
        .map(|point| (track.point - point).norm())
        .fold(f64::INFINITY, f64::min);
    assert!(
        nearest_truth < 1e-6,
        "recovered point error {nearest_truth}"
    );
}

#[test]
fn geometry_recovery_rejects_three_view_chain_without_cycle() {
    let scene = build_scene();
    let (features, _) = render(&scene);
    let observation = |image: usize| {
        let kp = features[image]
            .descriptors
            .iter()
            .position(|descriptor| descriptor[0] == 0.0)
            .unwrap();
        (image, kp)
    };
    let component = vec![observation(0), observation(1), observation(2)];
    let pairwise = vec![
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(component[0].1, component[1].1)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
        PairwiseMatches {
            image_i: 1,
            image_j: 2,
            matches: vec![(component[1].1, component[2].1)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
    ];
    let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
    let recovered = recover_conflict_tracks_geometry(
        &scene.camera,
        &features,
        &pairwise,
        &[component],
        &poses,
        &IncrementalSfmConfig {
            conflict_recovery_max_reprojection_error_px: 0.1,
            conflict_recovery_max_mean_reprojection_px: 0.05,
            ..IncrementalSfmConfig::default()
        },
    );
    assert!(
        recovered.is_empty(),
        "a tree is not independent multi-view evidence"
    );
}

#[test]
fn incremental_sfm_admits_geometry_recovery_only_after_clean_model() {
    let scene = build_scene();
    let (features, mut pairwise) = render(&scene);
    // Leave image 5 outside the verified component so the clean model is
    // intentionally incomplete and recovery may exercise its guarded BA.
    pairwise.retain(|pair| pair.image_i != 5 && pair.image_j != 5);
    let keypoint_for_point = |image: usize, point: usize| {
        features[image]
            .descriptors
            .iter()
            .position(|descriptor| descriptor[0] == point as f32)
            .unwrap()
    };
    pairwise
        .iter_mut()
        .find(|pair| pair.image_i == 0 && pair.image_j == 1)
        .unwrap()
        .matches
        .push((keypoint_for_point(0, 0), keypoint_for_point(1, 1)));

    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        geometry_guided_conflict_recovery: true,
        conflict_recovery_max_reprojection_error_px: 0.1,
        conflict_recovery_max_mean_reprojection_px: 0.05,
        // Noise-free BA can move at floating-point epsilon around zero.
        conflict_recovery_max_clean_error_increase_ratio: 0.01,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
    assert_eq!(result.track_build_stats.conflicting_components, 1);
    assert_eq!(result.geometry_recovered_tracks, 1);
    assert!(result.geometry_recovered_observations >= 3);
    assert!(result.geometry_recovery_pose_ba_applied);
    assert_eq!(result.registered_images, 5);
    assert!(result.mean_reprojection_px < 0.1);
}

#[test]
fn complete_model_geometry_recovery_keeps_poses_byte_identical() {
    let scene = build_scene();
    let (features, mut pairwise) = render(&scene);
    let keypoint_for_point = |image: usize, point: usize| {
        features[image]
            .descriptors
            .iter()
            .position(|descriptor| descriptor[0] == point as f32)
            .unwrap()
    };
    pairwise
        .iter_mut()
        .find(|pair| pair.image_i == 0 && pair.image_j == 1)
        .unwrap()
        .matches
        .push((keypoint_for_point(0, 0), keypoint_for_point(1, 1)));
    let base = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        ..IncrementalSfmConfig::default()
    };
    let control = incremental_sfm(&scene.camera, &features, &pairwise, &base).unwrap();
    let recovered = incremental_sfm(
        &scene.camera,
        &features,
        &pairwise,
        &IncrementalSfmConfig {
            geometry_guided_conflict_recovery: true,
            conflict_recovery_max_reprojection_error_px: 0.1,
            conflict_recovery_max_mean_reprojection_px: 0.05,
            ..base
        },
    )
    .unwrap();
    assert_eq!(control.registered_images, features.len());
    assert_eq!(recovered.registered_images, features.len());
    assert_eq!(recovered.geometry_recovered_tracks, 1);
    assert!(!recovered.geometry_recovery_pose_ba_applied);
    assert_eq!(recovered.poses, control.poses);
    assert_eq!(recovered.tracks.len(), control.tracks.len() + 1);
}

#[test]
fn rejected_geometry_recovery_rolls_back_byte_identical_clean_model() {
    let scene = build_scene();
    let (features, mut pairwise) = render(&scene);
    let keypoint_for_point = |image: usize, point: usize| {
        features[image]
            .descriptors
            .iter()
            .position(|descriptor| descriptor[0] == point as f32)
            .unwrap()
    };
    pairwise
        .iter_mut()
        .find(|pair| pair.image_i == 0 && pair.image_j == 1)
        .unwrap()
        .matches
        .push((keypoint_for_point(0, 0), keypoint_for_point(1, 1)));

    let base = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        ..IncrementalSfmConfig::default()
    };
    let control = incremental_sfm(&scene.camera, &features, &pairwise, &base).unwrap();
    let rejected = incremental_sfm(
        &scene.camera,
        &features,
        &pairwise,
        &IncrementalSfmConfig {
            geometry_guided_conflict_recovery: true,
            conflict_recovery_max_reprojection_error_px: 0.1,
            // Force the post-BA acceptance gate to reject every proposal.
            conflict_recovery_max_mean_reprojection_px: -1.0,
            ..base
        },
    )
    .unwrap();
    assert_eq!(rejected.geometry_recovered_tracks, 0);
    assert_eq!(rejected.geometry_recovered_observations, 0);
    assert_eq!(rejected.poses, control.poses);
    assert_eq!(rejected.tracks, control.tracks);
    assert_eq!(rejected.registered_images, control.registered_images);
    assert_eq!(rejected.mean_reprojection_px, control.mean_reprojection_px);
}

/// Minimal `FeatureSet`s with `kp_counts[i]` dummy keypoints per image —
/// enough for `build_tracks_via_graph` to declare each image's point2D
/// capacity; keypoint/descriptor content is irrelevant to track building.
fn dummy_features(kp_counts: &[usize]) -> Vec<FeatureSet> {
    kp_counts
        .iter()
        .map(|&n| {
            let kps = vec![Point2::new(0.0, 0.0); n];
            let descs = vec![vec![0.0f32; 4]; n];
            FeatureSet::new(kps, descs).unwrap()
        })
        .collect()
}

/// M2: the [`TrackSource::CorrespondenceGraph`] path reproduces
/// [`build_tracks_merges_shared_observations`] exactly.
#[test]
fn graph_tracks_merges_shared_observations() {
    let features = dummy_features(&[1, 1, 1]);
    let pairwise = vec![
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
        PairwiseMatches {
            image_i: 1,
            image_j: 2,
            matches: vec![(0, 0)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
    ];
    let tracks = build_tracks_via_graph(&features, &pairwise, 2);
    assert_eq!(tracks.len(), 1, "the chained matches form one track");
    assert_eq!(tracks[0].len(), 3, "track spans all three images");
}

/// M2: the [`TrackSource::CorrespondenceGraph`] path reproduces
/// [`build_tracks_drops_same_image_conflict`] exactly — including the
/// repeated-pair-entry input shape that exercises this function's
/// pre-merge step (see `build_tracks_via_graph`'s doc).
#[test]
fn graph_tracks_drops_same_image_conflict() {
    let features = dummy_features(&[2, 2]);
    let pairwise = vec![
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 1)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
    ];
    let tracks = build_tracks_via_graph(&features, &pairwise, 2);
    assert!(tracks.is_empty(), "same-image conflict track is dropped");
}

/// M2 acceptance bar: on a repeated-pair input in *swapped* direction
/// (`(1, 0)` instead of `(0, 1)`), the graph path's pre-merge
/// canonicalization must still see both entries as the same unordered
/// pair and produce the identical conflict-drop as
/// [`graph_tracks_drops_same_image_conflict`] — proving the merge step
/// doesn't silently drop the second entry via `DuplicatePair`.
#[test]
fn graph_tracks_drops_same_image_conflict_with_swapped_pair_direction() {
    let features = dummy_features(&[2, 2]);
    let pairwise = vec![
        PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0)],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
        PairwiseMatches {
            image_i: 1,
            image_j: 0,
            matches: vec![(1, 0)], // (kp1 in image1, kp0 in image0),
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        },
    ];
    let tracks = build_tracks_via_graph(&features, &pairwise, 2);
    assert!(tracks.is_empty(), "same-image conflict track is dropped");
}

#[test]
fn visibility_pyramid_prefers_distribution_over_count() {
    assert_eq!(
        NextImagePolicy::default(),
        NextImagePolicy::CorrespondenceCount
    );
    // A small cluster of MANY points in one corner versus FEWER points spread
    // across the frame: the COLMAP visibility score must rank the spread set
    // higher (better-conditioned PnP), even though it has fewer correspondences.
    let (w, h) = (640u32, 480u32);
    let clustered: Vec<Point2<f64>> = (0..50)
        .map(|i| Point2::new(2.0 + (i % 5) as f64, 2.0 + (i / 5) as f64))
        .collect();
    let spread: Vec<Point2<f64>> = (0..5)
        .flat_map(|gy| {
            (0..4).map(move |gx| {
                Point2::new(
                    (gx as f64 + 0.5) * w as f64 / 4.0,
                    (gy as f64 + 0.5) * h as f64 / 5.0,
                )
            })
        })
        .collect();
    let clustered_score = visibility_pyramid_score(w, h, clustered.iter().copied());
    let spread_score = visibility_pyramid_score(w, h, spread.iter().copied());
    assert!(
        spread_score > clustered_score,
        "spread ({spread_score}, {} pts) should beat clustered ({clustered_score}, {} pts)",
        spread.len(),
        clustered.len(),
    );
    // The 50 clustered points collapse onto a handful of cells (occupancy
    // saturates), unlike a raw count which would have ranked them first.
    assert!(
        clustered_score < clustered.len(),
        "clustered occupancy {clustered_score} must saturate below the point count"
    );

    let camera = Camera::pinhole(0, w, h, 500.0, 500.0, 320.0, 240.0);
    let to_corrs = |points: &[Point2<f64>]| {
        points
            .iter()
            .copied()
            .map(|point2d| Correspondence2D3D {
                point2d,
                point3d: Point3::new(0.0, 0.0, 5.0),
                confidence: None,
            })
            .collect::<Vec<_>>()
    };
    let clustered_corrs = to_corrs(&clustered);
    let spread_corrs = to_corrs(&spread);
    assert!(
        next_image_rank(&camera, NextImagePolicy::VisibilityPyramid, &spread_corrs,)
            > next_image_rank(
                &camera,
                NextImagePolicy::VisibilityPyramid,
                &clustered_corrs,
            ),
        "visibility policy must prefer coverage"
    );
    assert!(
        next_image_rank(
            &camera,
            NextImagePolicy::CorrespondenceCount,
            &clustered_corrs,
        ) > next_image_rank(&camera, NextImagePolicy::CorrespondenceCount, &spread_corrs,),
        "count policy must reproduce the legacy ordering"
    );
}

#[test]
fn auto_compares_count_for_every_incomplete_visibility_candidate() {
    assert!(next_image_auto_count_candidate_is_needed(89, 100));
    assert!(next_image_auto_count_candidate_is_needed(90, 100));
    assert!(next_image_auto_count_candidate_is_needed(9, 10));
    assert!(!next_image_auto_count_candidate_is_needed(10, 10));
    assert!(!next_image_auto_count_candidate_is_needed(0, 0));

    assert!(next_image_auto_post_candidate_is_needed(9, 10));
    assert!(!next_image_auto_post_candidate_is_needed(10, 10));

    let visibility = NextImageAutoMetrics {
        registered_images: 17,
        valid_observations: 100,
        tracks: 20,
        mean_reprojection_px: 2.0,
    };
    let count = NextImageAutoMetrics {
        registered_images: 18,
        valid_observations: 1,
        tracks: 1,
        mean_reprojection_px: 100.0,
    };
    assert!(next_image_auto_metrics_are_better(count, visibility));
}

#[test]
fn auto_post_completion_requires_strict_registration_gain() {
    let mut baseline = IncrementalSfmResult {
        poses: Vec::new(),
        tracks: Vec::new(),
        track_build_stats: TrackBuildStats::default(),
        registered_images: 9,
        post_refinement_registered_images: 0,
        structureless_registered_images: 0,
        geometry_recovered_tracks: 0,
        geometry_recovered_observations: 0,
        geometry_recovery_pose_ba_applied: false,
        mean_reprojection_px: 1.0,
        ba_result: None,
        refined_camera: None,
        seed_image_i: 0,
        seed_image_j: 1,
        seed_match_count: 0,
    };
    let mut candidate = baseline.clone();
    candidate.registered_images = 10;
    assert!(next_image_auto_post_candidate_is_better(
        &candidate, &baseline
    ));

    // A registration gain must not be allowed to replace a materially
    // cleaner incumbent with a finite but much worse post candidate.
    candidate.mean_reprojection_px = 100.0;
    assert!(!next_image_auto_post_candidate_is_better(
        &candidate, &baseline
    ));
    candidate.mean_reprojection_px = 1.0;
    assert!(next_image_auto_post_candidate_is_better(
        &candidate, &baseline
    ));

    // A post pass may improve reprojection or change tracks while leaving
    // registration unchanged. Auto must retain the untouched candidate in
    // that case, so the primary model's bytes/trajectory are stable.
    candidate.registered_images = baseline.registered_images;
    candidate.mean_reprojection_px = 0.1;
    assert!(!next_image_auto_post_candidate_is_better(
        &candidate, &baseline
    ));

    candidate.registered_images = baseline.registered_images.saturating_sub(1);
    assert!(!next_image_auto_post_candidate_is_better(
        &candidate, &baseline
    ));
    candidate.registered_images = baseline.registered_images + 1;
    candidate.mean_reprojection_px = f64::NAN;
    assert!(!next_image_auto_post_candidate_is_better(
        &candidate, &baseline
    ));

    // A finite post candidate can repair an incumbent whose aggregate
    // reprojection metric is unavailable, while a non-finite candidate
    // can never be adopted.
    baseline.mean_reprojection_px = f64::NAN;
    candidate.mean_reprojection_px = 0.5;
    assert!(next_image_auto_post_candidate_is_better(
        &candidate, &baseline
    ));
    baseline.registered_images = 0;
    candidate.registered_images = 1;
    assert!(next_image_auto_post_candidate_is_better(
        &candidate, &baseline
    ));
}

#[test]
fn auto_selection_is_lexicographic_and_ties_keep_visibility() {
    let visibility = NextImageAutoMetrics {
        registered_images: 17,
        valid_observations: 100,
        tracks: 20,
        mean_reprojection_px: 2.0,
    };
    let more_observations = NextImageAutoMetrics {
        valid_observations: 101,
        ..visibility
    };
    assert!(next_image_auto_metrics_are_better(
        more_observations,
        visibility
    ));

    let more_tracks = NextImageAutoMetrics {
        tracks: 21,
        ..visibility
    };
    assert!(next_image_auto_metrics_are_better(more_tracks, visibility));

    let lower_reprojection = NextImageAutoMetrics {
        mean_reprojection_px: 1.0,
        ..visibility
    };
    assert!(next_image_auto_metrics_are_better(
        lower_reprojection,
        visibility
    ));
    assert!(!next_image_auto_metrics_are_better(visibility, visibility));

    let nonfinite = NextImageAutoMetrics {
        mean_reprojection_px: f64::NAN,
        ..visibility
    };
    assert!(!next_image_auto_metrics_are_better(nonfinite, visibility));
}

#[test]
fn reconstructs_synthetic_ring_scene() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    assert!(pairwise.len() >= 5, "expected an overlapping view graph");

    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();

    // Most images register and most points triangulate.
    assert!(
        result.registered_images >= 5,
        "registered only {}",
        result.registered_images
    );
    assert!(
        result.tracks.len() >= 20,
        "triangulated only {} tracks",
        result.tracks.len()
    );
    // Reprojection is tight (synthetic, noise-free).
    assert!(
        result.mean_reprojection_px < 1.0,
        "mean reprojection {} px too high",
        result.mean_reprojection_px
    );

    // The reconstruction is correct up to a similarity transform. Check the
    // recovered camera-center geometry matches GT up to scale by comparing
    // pairwise center-distance ratios between two registered images.
    let registered: Vec<usize> = (0..scene.poses.len())
        .filter(|&i| result.poses[i].is_some())
        .collect();
    assert!(registered.len() >= 3);
    let center = |i: usize| {
        result.poses[i]
            .as_ref()
            .unwrap()
            .camera_to_world()
            .translation
    };
    let gt_center = |i: usize| scene.poses[i].camera_to_world().translation;
    let (a, b, c) = (registered[0], registered[1], registered[2]);
    let est_ratio = (center(a) - center(b)).norm() / (center(b) - center(c)).norm();
    let gt_ratio = (gt_center(a) - gt_center(b)).norm() / (gt_center(b) - gt_center(c)).norm();
    assert!(
        (est_ratio - gt_ratio).abs() / gt_ratio < 0.1,
        "camera-spacing ratio {est_ratio} != GT {gt_ratio} (similarity-invariant)"
    );
}

#[test]
fn auto_policy_runs_through_public_incremental_api() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        next_image_policy: NextImagePolicy::Auto,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
    assert!(
        result.registered_images >= 5,
        "Auto policy registered only {} synthetic images",
        result.registered_images
    );
}

/// Look-at world→camera poses on an arc of `n` cameras at `radius` from
/// `target`, spanning `span` radians (so neighbours keep a real baseline).
fn arc_cameras(n: usize, target: Point3<f64>, radius: f64, span: f64) -> Vec<Pose> {
    let mut poses = Vec::new();
    let denom = (n.max(2) - 1) as f64;
    for k in 0..n {
        let angle = -span / 2.0 + span * (k as f64) / denom;
        let cam_center = target + Vector3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
        let forward = (target - cam_center).normalize();
        let right = forward.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(&forward);
        let r_c2w = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
        let q_c2w = UnitQuaternion::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_unchecked(r_c2w),
        );
        let q_w2c = q_c2w.inverse();
        let t_w2c = -(q_w2c * cam_center.coords);
        poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
    }
    poses
}

/// A scene with two geometrically disjoint components: a small dense "trap"
/// cluster (3 cameras, ~100 co-visible points → the *strongest-match* pairs in
/// the whole graph) far to one side, and a larger "main" component (8 cameras
/// over a grid). The trap's frustums never see the main grid and vice versa,
/// so they form two connected components; the strongest seed reconstructs only
/// the 3-camera trap, and recovering the main component needs the multi-seed
/// search to look past it. Cameras: indices 0..3 trap, 3..11 main.
fn build_two_component_scene() -> Scene {
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let mut points = Vec::new();
    // Trap cluster: a dense cube at the origin (every trap camera sees all of
    // it, so each trap pair carries the most matches).
    for xi in -2..=2 {
        for yi in -2..=2 {
            for zi in -2..=1 {
                points.push(Point3::new(
                    xi as f64 * 0.2,
                    yi as f64 * 0.2,
                    zi as f64 * 0.2,
                ));
            }
        }
    }
    // Main grid: a separate, larger structure offset far along +x.
    for xi in -2..=2 {
        for yi in -2..=2 {
            for zi in 0..=2 {
                points.push(Point3::new(
                    20.0 + xi as f64 * 0.3,
                    yi as f64 * 0.3,
                    zi as f64 * 0.3,
                ));
            }
        }
    }
    let mut poses = arc_cameras(3, Point3::origin(), 3.0, 0.5);
    poses.extend(arc_cameras(8, Point3::new(20.0, 0.0, 0.0), 3.0, 1.2));
    Scene {
        camera,
        points,
        poses,
    }
}

#[test]
fn multi_seed_escapes_strongest_isolated_cluster() {
    let scene = build_two_component_scene();
    let (features, pairwise) = render(&scene);

    // The strongest-match pair is inside the 3-camera trap.
    let strongest = pairwise
        .iter()
        .max_by_key(|p| p.matches.len())
        .expect("a view graph");
    assert!(
        strongest.image_i < 3 && strongest.image_j < 3,
        "expected the densest pair to be inside the trap cluster, got ({},{})",
        strongest.image_i,
        strongest.image_j
    );

    // One trial commits to that strongest seed and is trapped in the cluster.
    let trapped = incremental_sfm(
        &scene.camera,
        &features,
        &pairwise,
        &IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            seed_trials: 1,
            ..IncrementalSfmConfig::default()
        },
    )
    .unwrap();
    assert!(
        trapped.registered_images <= 3,
        "single-seed should be stuck in the 3-camera trap, got {}",
        trapped.registered_images
    );

    // The multi-seed search looks past the trap and recovers the 8-camera
    // main component instead.
    let escaped = incremental_sfm(
        &scene.camera,
        &features,
        &pairwise,
        &IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            ..IncrementalSfmConfig::default() // seed_trials = 12
        },
    )
    .unwrap();
    assert!(
        escaped.registered_images >= 7,
        "multi-seed should recover the 8-camera main component, got {}",
        escaped.registered_images
    );
}

type OutlierTrackFixture = (
    Camera,
    Vec<FeatureSet>,
    Vec<Option<Pose>>,
    Vec<Vec<(usize, usize)>>,
    Vec<Option<Point3<f64>>>,
);

/// Build three views (identity rotation, small lateral offsets) of one world
/// point, with `outlier_views` images observing it at a planted off-by-50px
/// outlier keypoint instead of the true projection.
fn outlier_track_fixture(outlier_views: &[usize]) -> OutlierTrackFixture {
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let point = Point3::new(0.1, -0.2, 5.0);
    let mut features = Vec::new();
    let mut poses = Vec::new();
    for k in 0..3 {
        let pose = Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(k as f64 * 0.3 - 0.3, 0.0, 0.0),
        );
        let mut px = camera.project(&pose.transform_world_point(&point)).unwrap();
        if outlier_views.contains(&k) {
            px += Vector3::new(50.0, 50.0, 0.0).xy();
        }
        features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
        poses.push(Some(pose));
    }
    let tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
    let track_point = vec![Some(point)];
    (camera, features, poses, tracks, track_point)
}

#[test]
fn filter_strips_single_outlier_observation_keeps_track() {
    let (camera, features, poses, mut tracks, mut track_point) = outlier_track_fixture(&[2]);
    let config = IncrementalSfmConfig::default();
    let removed = filter_outlier_observations(
        &camera,
        &features,
        &mut tracks,
        &config,
        &poses,
        &mut track_point,
    );
    assert_eq!(removed, 1, "the planted outlier observation is removed");
    assert_eq!(
        tracks[0],
        vec![(0, 0), (1, 0)],
        "only the two inliers remain"
    );
    assert!(track_point[0].is_some(), "track survives with >= 2 inliers");
}

#[test]
fn filter_drops_low_parallax_far_point() {
    // A point 500 units away, seen by three cameras 0.6 units apart, projects
    // with ZERO reprojection error (perfect) yet has ~0.07 deg parallax — the
    // depth-ambiguous far-flung outlier the reprojection test cannot catch.
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let point = Point3::new(0.0, 0.0, 500.0);
    let mut features = Vec::new();
    let mut poses = Vec::new();
    for k in 0..3 {
        let pose = Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(k as f64 * 0.3 - 0.3, 0.0, 0.0),
        );
        let px = camera.project(&pose.transform_world_point(&point)).unwrap();
        features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
        poses.push(Some(pose));
    }
    let mut tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
    let mut track_point = vec![Some(point)];
    let config = IncrementalSfmConfig::default(); // min_triangulation_angle_deg = 2.0
    let changed = filter_outlier_observations(
        &camera,
        &features,
        &mut tracks,
        &config,
        &poses,
        &mut track_point,
    );
    assert_eq!(changed, 1, "the low-parallax track is dropped");
    assert!(
        track_point[0].is_none(),
        "depth-ambiguous far point dropped despite zero reprojection error"
    );
}

#[test]
fn filter_drops_track_below_min_observations() {
    // Two of three views are outliers -> a single inlier left -> drop track.
    let (camera, features, poses, mut tracks, mut track_point) = outlier_track_fixture(&[1, 2]);
    let config = IncrementalSfmConfig::default();
    let removed = filter_outlier_observations(
        &camera,
        &features,
        &mut tracks,
        &config,
        &poses,
        &mut track_point,
    );
    assert_eq!(removed, 3, "2 observations stripped + 1 track dropped");
    assert!(
        track_point[0].is_none(),
        "track with < 2 inlier observations is dropped"
    );
}

#[test]
fn filter_images_deregisters_unsupported_pose_and_protects_seed() {
    // Register all six ring cameras with their true poses, triangulate, then
    // corrupt one non-seed image's pose so none of its observations reproject.
    // FilterImages must de-register exactly that image, keep the well-supported
    // ones, and never touch the two seed (lowest-index) images.
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let tracks = build_tracks(features.len(), &pairwise, 2);
    let config = IncrementalSfmConfig {
        filter_images: true,
        filter_min_image_observations: 5,
        ..IncrementalSfmConfig::default()
    };
    let mut poses: Vec<Option<Pose>> = scene.poses.iter().map(|p| Some(p.clone())).collect();
    let mut track_point = vec![None; tracks.len()];
    triangulate_pending(
        &scene.camera,
        &features,
        &tracks,
        &poses,
        &config,
        &mut track_point,
    );

    // Corrupt image 3 (not in the protected seed pair): aim it away from the
    // cloud so every observation reprojects far off or behind the camera.
    let bad = Pose::from_world_to_camera(
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::PI),
        Vector3::new(0.0, 0.0, 0.0),
    );
    poses[3] = Some(bad);

    let removed = filter_images(
        &scene.camera,
        &features,
        &tracks,
        &config,
        &mut poses,
        &track_point,
    );
    assert_eq!(removed, 1, "only the unsupported image is de-registered");
    assert!(poses[3].is_none(), "the corrupted-pose image is filtered");
    assert!(
        poses[0].is_some() && poses[1].is_some(),
        "the seed pair is protected from filtering"
    );
    assert!(
        poses[2].is_some() && poses[4].is_some() && poses[5].is_some(),
        "well-supported images stay registered"
    );
}

#[test]
fn retriangulate_completes_untriangulated_track() {
    // Three identity-rotation views with a real lateral baseline see one
    // world point. The track exists in the union-find but was never given a
    // 3D point (it failed the parallax gate at growth time, say). With the
    // poses now fixed, re-triangulation must complete it.
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let point = Point3::new(0.1, -0.2, 5.0);
    let mut features = Vec::new();
    let mut poses = Vec::new();
    for k in 0..3 {
        let pose = Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(k as f64 * 0.5 - 0.5, 0.0, 0.0),
        );
        let px = camera.project(&pose.transform_world_point(&point)).unwrap();
        features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
        poses.push(Some(pose));
    }
    let tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
    let mut track_point: Vec<Option<Point3<f64>>> = vec![None]; // not yet triangulated
    let config = IncrementalSfmConfig::default();
    let changed = retriangulate_tracks(
        &camera,
        &features,
        &tracks,
        &config,
        &poses,
        &mut track_point,
    );
    assert_eq!(changed, 1, "the un-triangulated track is completed");
    let p = track_point[0].expect("track now has a 3D point");
    assert!(
        (p - point).norm() < 1e-6,
        "re-triangulated point {p:?} should recover the true point {point:?}"
    );
}

#[test]
fn retriangulate_guarded_swap_replaces_noisy_point_only_when_better() {
    // Same three-view geometry, but the track already carries a *noisy* point
    // displaced far along the depth ray. Re-triangulation from the true
    // observations fits them better, so the guarded swap must replace it; a
    // second pass (now exact) must be a no-op (never regress an exact point).
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let point = Point3::new(0.1, -0.2, 5.0);
    let mut features = Vec::new();
    let mut poses = Vec::new();
    for k in 0..3 {
        let pose = Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(k as f64 * 0.5 - 0.5, 0.0, 0.0),
        );
        let px = camera.project(&pose.transform_world_point(&point)).unwrap();
        features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
        poses.push(Some(pose));
    }
    let tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
    let noisy = Point3::new(0.3, -0.6, 8.0);
    let mut track_point = vec![Some(noisy)];
    let config = IncrementalSfmConfig::default();

    let changed = retriangulate_tracks(
        &camera,
        &features,
        &tracks,
        &config,
        &poses,
        &mut track_point,
    );
    assert_eq!(changed, 1, "the noisy point is replaced by a better fit");
    let p = track_point[0].unwrap();
    assert!(
        (p - point).norm() < 1e-6,
        "guarded swap should land on the true point, got {p:?}"
    );

    // Re-running on the now-exact point changes nothing.
    let again = retriangulate_tracks(
        &camera,
        &features,
        &tracks,
        &config,
        &poses,
        &mut track_point,
    );
    assert_eq!(again, 0, "an already-exact point must not be regressed");
}

#[test]
fn colmap_style_mapper_reconstructs_ring_scene() {
    // The COLMAP schedule (per-registration local BA + growth-triggered
    // iterative global refinement + registration retries) must reconstruct the
    // synthetic ring at least as completely as the simple schedule, with tight
    // reprojection and a similarity-correct camera geometry.
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        colmap_style_mapper: true,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
    assert!(
        result.registered_images >= 5,
        "registered only {}",
        result.registered_images
    );
    assert!(
        result.tracks.len() >= 20,
        "triangulated only {} tracks",
        result.tracks.len()
    );
    assert!(
        result.mean_reprojection_px < 1.0,
        "mean reprojection {} px too high",
        result.mean_reprojection_px
    );
    let registered: Vec<usize> = (0..scene.poses.len())
        .filter(|&i| result.poses[i].is_some())
        .collect();
    let center = |i: usize| {
        result.poses[i]
            .as_ref()
            .unwrap()
            .camera_to_world()
            .translation
    };
    let gt_center = |i: usize| scene.poses[i].camera_to_world().translation;
    let (a, b, c) = (registered[0], registered[1], registered[2]);
    let est_ratio = (center(a) - center(b)).norm() / (center(b) - center(c)).norm();
    let gt_ratio = (gt_center(a) - gt_center(b)).norm() / (gt_center(b) - gt_center(c)).norm();
    assert!(
        (est_ratio - gt_ratio).abs() / gt_ratio < 0.1,
        "camera-spacing ratio {est_ratio} != GT {gt_ratio} (similarity-invariant)"
    );
}

#[test]
fn colmap_style_co_evolves_intrinsics_toward_truth() {
    // The synthetic ring is observable geometry, so a focal error is
    // recoverable. Render with the TRUE camera (fx=fy=500) but reconstruct from
    // a WRONG horizontal focal (fx=530). The arc moves the cameras only in the
    // x-z plane, so the *horizontal* focal fx is well constrained by the
    // azimuthal parallax (fy would need elevation change — exercised instead by
    // the anisotropic South-Building benchmark). The joint solve must pull fx
    // substantially back toward 500 — the COLMAP self-calibration formulation
    // (intrinsics co-estimated inside the Schur camera system, using the coupled
    // landmark-eliminated gradient), which a final-only alternating refinement
    // against converged structure cannot do. The orthogonal, un-perturbed
    // vertical axis (fy, cy) must stay fixed. The horizontal principal point cx
    // is allowed to co-adjust: on a pure look-at arc fx and cx are only *jointly*
    // constrained (the focal/principal-point ambiguity), so the joint solve
    // legitimately distributes the correction across both — this confound is
    // absent on the richer South-Building viewpoints, where cx stays put.
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let wrong = Camera::pinhole(0, 640, 480, 530.0, 500.0, 320.0, 240.0);
    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        colmap_style_mapper: true,
        refine_intrinsics: true,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&wrong, &features, &pairwise, &config).unwrap();
    let cam = result
        .refined_camera
        .expect("refine_intrinsics returns the refined camera");
    let (fx, fy, cx, cy) = cam.intrinsics().unwrap();
    eprintln!("joint-refined intrinsics: fx {fx} fy {fy} cx {cx} cy {cy}");
    // fx recovers at least a third of its injected error (530 - 500 = 30).
    assert!(
        fx < 530.0 - 0.33 * 30.0,
        "fx {fx} should recover substantially toward 500 from 530"
    );
    // The orthogonal vertical axis was not perturbed and must not drift.
    assert!((fy - 500.0).abs() < 2.0, "fy {fy} drifted from 500");
    assert!((cy - 240.0).abs() < 2.0, "cy {cy} drifted from 240");
    // cx co-adjusts with fx within the look-at arc's focal/centre ambiguity, but
    // must stay sane (no blow-up).
    assert!((cx - 320.0).abs() < 20.0, "cx {cx} blew up from 320");
}

#[test]
fn colmap_style_mapper_retries_a_filtered_image_up_to_its_trial_budget_then_gives_up() {
    // M4 (`docs/colmap_port_plan.md`'s "M4 results"): the growth loop's
    // stall-triggered recovery must give a `filter_images`-demoted image
    // genuine retry attempts across multiple growth stalls — not filter it
    // once and abandon it, the pre-M4 behaviour, since pre-M4
    // `growth_global_refinement` (and the `filter_images` call inside it)
    // only ever ran on the growth-*ratio* trigger, never on a stall — while
    // still terminating cleanly once `max_registration_trials` is spent,
    // rather than cycling forever. `global_ba_images_ratio` is set absurdly
    // high so the *only* thing that can ever invoke `growth_global_refinement`
    // / `filter_images` in this test is the stall path, isolating exactly
    // the mechanism this milestone added.
    //
    // Scene: 4 cameras looking at the same 40-point cloud (two z-layers so
    // the essential-matrix seed estimator sees a non-degenerate, non-planar
    // point set). The seed pair (0, 1) and camera 3 all see all 40 points;
    // camera 2 is built (by construction, not by frustum geometry) to see
    // only the first 15 — enough to clear `min_pnp_inliers` and register,
    // but below `filter_min_image_observations` (16), so every time
    // `filter_images` runs it demotes camera 2 and nothing else (the seed
    // pair is exempt from filtering by construction, and camera 3 is
    // well-supported). A 4th, well-supported camera is needed because
    // `filter_images` refuses to drop *anyone* once the registered count
    // is already at its floor of 3 (`incremental_sfm.rs`'s
    // `filter_images`: `if remaining <= 3 { continue; }`) — with only 3
    // total cameras, camera 2 could never be filtered no matter how weak
    // its support, which would make this test vacuous. Since camera 2's
    // supporting-observation count can never improve (it structurally
    // only ever sees 15 points), this is a fixed point: register, demote,
    // retry, register, demote, … — bounded only by
    // `max_registration_trials`. Never resetting `trials` on the stall
    // (see `grow_from_seed`'s module-level doc on `stalled_once`) is what
    // makes this terminate at all instead of cycling indefinitely.
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let mut points = Vec::new();
    for xi in -2..=2 {
        for yi in -1..=2 {
            for zi in 0..=1 {
                points.push(Point3::new(
                    xi as f64 * 0.25,
                    yi as f64 * 0.25,
                    1.0 + zi as f64 * 0.3,
                ));
            }
        }
    }
    assert_eq!(points.len(), 40, "test fixture must have exactly 40 points");
    let poses = arc_cameras(4, Point3::origin(), 3.0, 0.6);

    // Camera 2 ("the weak straggler") only ever observes the first 15 of
    // the 40 points; cameras 0, 1 (the seed pair) and 3 observe all 40.
    let mut features = Vec::new();
    let mut visible: Vec<HashMap<usize, usize>> = Vec::new();
    for (cam_idx, pose) in poses.iter().enumerate() {
        let n_visible = if cam_idx == 2 { 15 } else { points.len() };
        let mut kps = Vec::new();
        let mut descs = Vec::new();
        let mut vis = HashMap::new();
        for (pidx, p) in points.iter().enumerate().take(n_visible) {
            let px = project(&camera, pose, p)
                .expect("fixture point must project in front of every camera");
            vis.insert(pidx, kps.len());
            kps.push(px);
            descs.push(vec![pidx as f32, 1.0, 0.0, 0.0]);
        }
        features.push(FeatureSet::new(kps, descs).unwrap());
        visible.push(vis);
    }

    let n_cams = poses.len();
    let mut pairwise = Vec::new();
    for i in 0..n_cams {
        for j in (i + 1)..n_cams {
            let mut matches = Vec::new();
            for (pidx, &ki) in &visible[i] {
                if let Some(&kj) = visible[j].get(pidx) {
                    matches.push((ki, kj));
                }
            }
            if matches.len() >= 8 {
                pairwise.push(PairwiseMatches {
                    image_i: i,
                    image_j: j,
                    matches,
                    two_view_config: None,
                    essential_matches: None,
                    essential_matrix: None,
                });
            }
        }
    }

    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 8,
        colmap_style_mapper: true,
        filter_images: true,
        filter_min_image_observations: 16,
        global_ba_images_ratio: 1000.0,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&camera, &features, &pairwise, &config).unwrap();

    assert_eq!(
        result.registered_images, 3,
        "camera 2 can never clear filter_min_image_observations=16 with its \
             fixed 15 supporting observations, so it must end up excluded, not \
             stuck mid-retry or wrongly kept, leaving the other 3 cameras registered"
    );
    assert!(
        result.poses[0].is_some() && result.poses[1].is_some(),
        "the seed pair stays registered (protected from filter_images)"
    );
    assert!(
        result.poses[2].is_none(),
        "the weakly-supported straggler ends up filtered, not registered"
    );
    assert!(
        result.poses[3].is_some(),
        "the well-supported 4th camera stays registered"
    );
}

#[test]
fn colmap_style_mapper_is_deterministic_across_repeated_runs() {
    // M4 regression pin: multi-seed search (`seed_trials`) and the new
    // stall-triggered recovery must stay fully deterministic (fixed PnP
    // RANSAC seed, no reset-driven or iteration-order-driven nondeterminism)
    // — running the identical config against the identical view graph twice
    // must produce byte-identical registered counts, track counts, and mean
    // reprojection error. Uses `build_two_component_scene` (multiple seed
    // candidates, one of them a trap) with `colmap_style_mapper` on so both
    // the multi-seed sweep and the stall-recovery path are exercised.
    let scene = build_two_component_scene();
    let (features, pairwise) = render(&scene);
    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        colmap_style_mapper: true,
        ..IncrementalSfmConfig::default()
    };
    let a = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
    let b = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
    assert_eq!(a.registered_images, b.registered_images);
    assert_eq!(a.tracks.len(), b.tracks.len());
    assert_eq!(
        a.mean_reprojection_px.to_bits(),
        b.mean_reprojection_px.to_bits(),
        "mean reprojection error must be bit-identical across repeated runs"
    );
    for i in 0..scene.poses.len() {
        assert_eq!(
            a.poses[i].is_some(),
            b.poses[i].is_some(),
            "image {i}'s registration outcome must be identical across runs"
        );
    }
}

#[test]
fn repeated_bundle_adjustment_does_not_collapse_scale() {
    // A monocular reconstruction has a free scale gauge; without anchoring it
    // a second BA from the converged state collapses the reconstruction.
    // run_bundle_adjustment fixes a second (farthest) pose to pin scale, so
    // re-optimising must be stable — track refinement relies on this.
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let config = IncrementalSfmConfig {
        min_seed_matches: 8,
        min_pnp_inliers: 6,
        track_filter_iterations: 4,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
    // A scale collapse manifests as nearly all tracks dropping out (the EuRoC
    // symptom was 630 -> 1) and the camera geometry degenerating; with the
    // gauge anchored, structure and registration survive four BA rounds.
    assert!(
        result.registered_images >= 5,
        "registration must survive repeated BA, got {}",
        result.registered_images
    );
    assert!(
        result.tracks.len() >= 20,
        "structure must survive repeated BA, got {} tracks",
        result.tracks.len()
    );
    assert!(
        result.mean_reprojection_px < 1.0,
        "reprojection {} px too high after repeated BA",
        result.mean_reprojection_px
    );
    // Camera-spacing ratio stays similarity-correct (a collapse would warp it).
    let registered: Vec<usize> = (0..scene.poses.len())
        .filter(|&i| result.poses[i].is_some())
        .collect();
    let center = |i: usize| {
        result.poses[i]
            .as_ref()
            .unwrap()
            .camera_to_world()
            .translation
    };
    let gt_center = |i: usize| scene.poses[i].camera_to_world().translation;
    let (a, b, c) = (registered[0], registered[1], registered[2]);
    let est_ratio = (center(a) - center(b)).norm() / (center(b) - center(c)).norm();
    let gt_ratio = (gt_center(a) - gt_center(b)).norm() / (gt_center(b) - gt_center(c)).norm();
    assert!(
        (est_ratio - gt_ratio).abs() / gt_ratio < 0.1,
        "camera geometry warped after repeated BA: {est_ratio} vs GT {gt_ratio}"
    );
}

#[test]
fn final_ba_polish_keeps_support_and_is_deterministic() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let tracks = build_tracks(features.len(), &pairwise, 2);
    let poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
    let mut track_point = vec![None; tracks.len()];
    let config = IncrementalSfmConfig {
        final_ba_polish_iterations: 5,
        ..IncrementalSfmConfig::default()
    };
    triangulate_pending(
        &scene.camera,
        &features,
        &tracks,
        &poses,
        &config,
        &mut track_point,
    );
    let poses_initial = poses;
    let points_initial = track_point.clone();
    let mut poses_a = poses_initial.clone();
    let mut points_a = points_initial.clone();
    let (stats_a, result_a) = final_fixed_support_ba_polish(
        &scene.camera,
        &features,
        &tracks,
        &config,
        &mut poses_a,
        &mut points_a,
    )
    .expect("fixed-support polish should solve the synthetic scene");
    assert!(stats_a.accepted);
    assert!(stats_a.initial_sse.is_finite());
    assert!(stats_a.final_sse <= stats_a.initial_sse);
    assert_eq!(stats_a.tracks_before, stats_a.tracks_after);
    assert_eq!(
        stats_a.observations_before, stats_a.observations_after,
        "polish must not change the supported observation set"
    );
    assert!(result_a.is_some());

    let mut poses_b = poses_initial;
    let mut points_b = points_initial;
    let (stats_b, result_b) = final_fixed_support_ba_polish(
        &scene.camera,
        &features,
        &tracks,
        &config,
        &mut poses_b,
        &mut points_b,
    )
    .expect("repeated fixed-support polish should solve identically");
    assert_eq!(stats_a, stats_b);
    assert_eq!(poses_a, poses_b);
    assert_eq!(points_a, points_b);
    assert_eq!(result_a, result_b);

    let mut poses_disabled = poses_a.clone();
    let mut points_disabled = points_a.clone();
    let disabled = final_fixed_support_ba_polish(
        &scene.camera,
        &features,
        &tracks,
        &IncrementalSfmConfig::default(),
        &mut poses_disabled,
        &mut points_disabled,
    )
    .expect("disabled polish is a no-op");
    assert_eq!(disabled.0.requested_iterations, 0);
    assert!(!disabled.0.accepted);
    assert!(disabled.1.is_none());
    assert_eq!(poses_disabled, poses_a);
    assert_eq!(points_disabled, points_a);
}

#[test]
fn structureless_local_bundle_keeps_registered_boundary_poses_exactly_fixed() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let tracks = build_tracks(features.len(), &pairwise, 2);
    let config = IncrementalSfmConfig::default();
    let mut poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
    let mut track_point = vec![None; tracks.len()];
    triangulate_pending(
        &scene.camera,
        &features,
        &tracks,
        &poses,
        &config,
        &mut track_point,
    );

    // Mimic a slightly inaccurate multi-neighbour structure-less proposal.
    // Only image 2 is allowed to move during the admission refinement.
    let truth = scene.poses[2].clone();
    poses[2].as_mut().unwrap().world_to_camera.translation += Vector3::new(0.03, -0.02, 0.01);
    let before = poses.clone();
    let mut variable = HashSet::new();
    variable.insert(2usize);
    bundle_adjust_local(
        &scene.camera,
        &features,
        &tracks,
        &config,
        &mut poses,
        &mut track_point,
        &variable,
    )
    .expect("fixed-boundary structure-less BA should converge");

    for image in [0usize, 1, 3, 4, 5] {
        assert_eq!(
            poses[image], before[image],
            "registered boundary pose {image} must remain byte-for-byte unchanged"
        );
    }
    let error_before = (before[2].as_ref().unwrap().matrix() - truth.matrix()).norm();
    let error_after = (poses[2].as_ref().unwrap().matrix() - truth.matrix()).norm();
    assert!(
        error_after < error_before,
        "recovered pose should improve while its registered boundary stays fixed: \
             {error_before} -> {error_after}"
    );
}

#[test]
fn structureless_fixed_pose_submap_refines_only_new_landmarks() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let tracks = build_tracks(features.len(), &pairwise, 2);
    let config = IncrementalSfmConfig::default();
    let poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
    let mut track_point = vec![None; tracks.len()];
    triangulate_pending(
        &scene.camera,
        &features,
        &tracks,
        &poses,
        &config,
        &mut track_point,
    );
    let new_track = tracks
        .iter()
        .position(|track| track.iter().any(|&(image, _)| image == 2))
        .unwrap();
    let truth = track_point[new_track].unwrap();
    track_point[new_track] = Some(truth + Vector3::new(0.08, -0.05, 0.12));
    let points_before = track_point.clone();
    let mut preexisting = vec![true; track_point.len()];
    preexisting[new_track] = false;

    refine_structureless_new_landmarks(
        &scene.camera,
        &features,
        &tracks,
        &config,
        &poses,
        &mut track_point,
        2,
        &preexisting,
    )
    .expect("fixed-pose local submap should converge");

    for track_id in 0..track_point.len() {
        if track_id != new_track {
            assert_eq!(track_point[track_id], points_before[track_id]);
        }
    }
    assert!(
        (track_point[new_track].unwrap() - truth).norm()
            < (points_before[new_track].unwrap() - truth).norm()
    );
}

#[test]
fn structureless_local_tracks_use_consensus_edges_and_unowned_observations() {
    let scene = build_scene();
    let (features, pairwise) = render(&scene);
    let poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
    let missing = 0usize;
    let missing_center = scene.poses[missing].camera_center_world();
    let missing_rotation = scene.poses[missing].world_to_camera.rotation;
    let constraints: Vec<_> = [1usize, 2, 3]
        .into_iter()
        .map(|neighbor| {
            structureless_constraint(
                neighbor,
                scene.poses[neighbor].camera_center_world(),
                missing_center,
                missing_rotation,
            )
        })
        .collect();
    let proposal = StructurelessPoseProposal {
        pose: scene.poses[missing].clone(),
        neighbor_spread: 1.0,
        line_error_ratio: 0.0,
        consensus_indices: vec![0, 1, 2],
    };
    let local = build_structureless_local_tracks(
        &scene.camera,
        &features,
        &pairwise,
        &[],
        &[],
        &poses,
        missing,
        &constraints,
        &proposal,
        &IncrementalSfmConfig::default(),
    );
    assert!(!local.is_empty());
    for (track, point) in local {
        assert!(track.len() >= 2);
        assert_eq!(
            track.iter().filter(|(image, _)| *image == missing).count(),
            1
        );
        let unique_images: HashSet<_> = track.iter().map(|(image, _)| *image).collect();
        assert_eq!(unique_images.len(), track.len());
        for &(image, keypoint) in &track {
            let error = reprojection_error_px(
                &scene.camera,
                poses[image].as_ref().unwrap(),
                &point,
                &features[image].keypoints[keypoint],
            )
            .unwrap();
            assert!(error <= 2.0);
        }
    }
}

fn structureless_constraint(
    neighbor: usize,
    neighbor_center: Point3<f64>,
    missing_center: Point3<f64>,
    missing_rotation: UnitQuaternion<f64>,
) -> StructurelessConstraint {
    StructurelessConstraint {
        neighbor,
        neighbor_center,
        missing_rotation,
        center_direction: (missing_center - neighbor_center).normalize(),
        weight: 100.0 - neighbor as f64,
    }
}

#[test]
fn structureless_multineighbor_lines_recover_scaled_camera_pose() {
    let missing_center = Point3::new(1.2, -0.4, 3.5);
    let rotation = UnitQuaternion::from_euler_angles(0.05, -0.12, 0.08);
    let constraints = vec![
        structureless_constraint(0, Point3::new(-1.0, 0.0, 0.0), missing_center, rotation),
        structureless_constraint(1, Point3::new(2.0, 0.5, 0.2), missing_center, rotation),
        structureless_constraint(2, Point3::new(0.0, -2.0, 0.4), missing_center, rotation),
    ];
    let config = IncrementalSfmConfig {
        structureless_min_intersection_angle_deg: 1.0,
        structureless_max_center_line_error_ratio: 1e-8,
        ..IncrementalSfmConfig::default()
    };
    let proposal = solve_structureless_pose(&constraints, &config).unwrap();
    assert!((proposal.pose.camera_center_world() - missing_center).norm() < 1e-9);
    assert!((proposal.pose.world_to_camera.rotation.inverse() * rotation).angle() < 1e-12);
    assert!(proposal.line_error_ratio < 1e-9);
}

#[test]
fn structureless_pose_interpolation_uses_camera_centers_and_slerp() {
    let from_center = Point3::new(-1.0, 0.5, 2.0);
    let to_center = Point3::new(3.0, -0.5, 4.0);
    let from_rotation = UnitQuaternion::identity();
    let to_rotation = UnitQuaternion::from_euler_angles(0.0, 0.4, 0.0);
    let from = Pose::from_world_to_camera(
        from_rotation,
        -from_rotation.transform_vector(&from_center.coords),
    );
    let to = Pose::from_world_to_camera(
        to_rotation,
        -to_rotation.transform_vector(&to_center.coords),
    );
    let midpoint = interpolate_structureless_pose(&from, &to, 0.5);
    let expected_center = Point3::from((from_center.coords + to_center.coords) * 0.5);
    assert!((midpoint.camera_center_world() - expected_center).norm() < 1e-12);
    let expected_rotation = from_rotation.slerp(&to_rotation, 0.5);
    assert!((midpoint.world_to_camera.rotation.inverse() * expected_rotation).angle() < 1e-12);
    assert_eq!(interpolate_structureless_pose(&from, &to, 0.0), from);
    assert_eq!(interpolate_structureless_pose(&from, &to, 1.0), to);
}

#[test]
fn structureless_pose_rejects_single_neighbor_arbitrary_scale() {
    let missing_center = Point3::new(0.0, 0.0, 3.0);
    let constraints = vec![structureless_constraint(
        0,
        Point3::origin(),
        missing_center,
        UnitQuaternion::identity(),
    )];
    assert!(solve_structureless_pose(&constraints, &IncrementalSfmConfig::default()).is_err());
}

#[test]
fn structureless_pose_rejects_rotation_disagreement() {
    let missing_center = Point3::new(0.5, 0.2, 3.0);
    let constraints = vec![
        structureless_constraint(
            0,
            Point3::new(-1.0, 0.0, 0.0),
            missing_center,
            UnitQuaternion::identity(),
        ),
        structureless_constraint(
            1,
            Point3::new(1.0, 0.0, 0.0),
            missing_center,
            UnitQuaternion::from_euler_angles(0.0, 0.2, 0.0),
        ),
    ];
    assert!(solve_structureless_pose(&constraints, &IncrementalSfmConfig::default()).is_err());
}

#[test]
fn structureless_pose_uses_largest_rotation_consensus() {
    let missing_center = Point3::new(0.5, 0.2, 3.0);
    let good_rotation = UnitQuaternion::from_euler_angles(0.01, -0.02, 0.03);
    let mut constraints = vec![
        structureless_constraint(
            0,
            Point3::new(-1.0, 0.0, 0.0),
            missing_center,
            good_rotation,
        ),
        structureless_constraint(1, Point3::new(1.0, 0.0, 0.0), missing_center, good_rotation),
        structureless_constraint(
            2,
            Point3::new(0.0, -1.0, 0.0),
            missing_center,
            good_rotation,
        ),
        structureless_constraint(
            3,
            Point3::new(0.0, 1.0, 0.0),
            missing_center,
            UnitQuaternion::from_euler_angles(0.0, 0.8, 0.0),
        ),
    ];
    constraints[3].weight = 1000.0;
    let proposal = solve_structureless_pose(&constraints, &IncrementalSfmConfig::default())
        .expect("three coherent rotations must outvote one high-support outlier");
    assert_eq!(proposal.consensus_indices.len(), 3);
    assert!((proposal.pose.camera_center_world() - missing_center).norm() < 1e-9);
    assert!((proposal.pose.world_to_camera.rotation.inverse() * good_rotation).angle() < 1e-12);
}

#[test]
fn structureless_pose_keeps_rotation_consensus_centre_not_strongest_edge() {
    let missing_center = Point3::new(0.5, 0.2, 3.0);
    let centre_rotation = UnitQuaternion::identity();
    let positive_edge = UnitQuaternion::from_euler_angles(0.0, 2.5f64.to_radians(), 0.0);
    let negative_edge = UnitQuaternion::from_euler_angles(0.0, -2.5f64.to_radians(), 0.0);
    let mut constraints = vec![
        structureless_constraint(
            0,
            Point3::new(-1.0, 0.0, 0.0),
            missing_center,
            centre_rotation,
        ),
        structureless_constraint(1, Point3::new(1.0, 0.0, 0.0), missing_center, positive_edge),
        structureless_constraint(
            2,
            Point3::new(0.0, -1.0, 0.0),
            missing_center,
            negative_edge,
        ),
    ];
    constraints[1].weight = 1000.0;
    let proposal = solve_structureless_pose(&constraints, &IncrementalSfmConfig::default())
        .expect("the centre edge supports a valid three-rotation consensus");
    assert!(
        (proposal.pose.world_to_camera.rotation.inverse() * centre_rotation).angle() < 1e-12,
        "the high-weight +2.5deg edge would be 5deg from the negative edge"
    );
    let consistency = structureless_pose_consistency(
        &proposal.pose,
        &constraints,
        &proposal,
        &IncrementalSfmConfig::default(),
    );
    assert!(consistency.accepted);
    assert!(consistency.max_rotation_deg <= 3.0);
}

#[test]
fn structureless_pose_uses_robust_translation_consensus() {
    let missing_center = Point3::new(0.5, 0.2, 3.0);
    let rotation = UnitQuaternion::identity();
    let mut constraints = vec![
        structureless_constraint(0, Point3::new(-1.0, 0.0, 0.0), missing_center, rotation),
        structureless_constraint(1, Point3::new(1.0, 0.0, 0.0), missing_center, rotation),
        structureless_constraint(2, Point3::new(0.0, -1.0, 0.0), missing_center, rotation),
        StructurelessConstraint {
            neighbor: 3,
            neighbor_center: Point3::new(0.0, 1.0, 0.0),
            missing_rotation: rotation,
            center_direction: Vector3::x(),
            weight: 1000.0,
        },
    ];
    constraints.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    let config = IncrementalSfmConfig {
        structureless_max_center_line_error_ratio: 0.01,
        ..IncrementalSfmConfig::default()
    };
    let proposal = solve_structureless_pose(&constraints, &config)
        .expect("three coherent directions must reject one high-support translation outlier");
    assert_eq!(proposal.consensus_indices.len(), 3);
    assert!((proposal.pose.camera_center_world() - missing_center).norm() < 1e-9);
}

#[test]
fn structureless_pose_reclassifies_lines_after_weighted_refit() {
    let rotation = UnitQuaternion::identity();
    let mut constraints = vec![
        StructurelessConstraint {
            neighbor: 0,
            neighbor_center: Point3::new(-1.0, 0.0, 0.0),
            missing_rotation: rotation,
            center_direction: Vector3::x(),
            weight: 100.0,
        },
        StructurelessConstraint {
            neighbor: 1,
            neighbor_center: Point3::new(0.0, -1.0, 0.0),
            missing_rotation: rotation,
            center_direction: Vector3::y(),
            weight: 100.0,
        },
        StructurelessConstraint {
            neighbor: 2,
            neighbor_center: Point3::new(0.0, 1.0, 0.0),
            missing_rotation: rotation,
            center_direction: Vector3::new(0.1, -1.0, 0.0).normalize(),
            weight: 1000.0,
        },
        // This short directed baseline agrees with the winning pairwise
        // hypothesis at the origin, but the high-weight tilted line moves
        // the least-squares refit behind it. It must be reclassified as an
        // outlier instead of vetoing the other three consistent lines.
        StructurelessConstraint {
            neighbor: 3,
            neighbor_center: Point3::new(0.01, 0.0, 0.0),
            missing_rotation: rotation,
            center_direction: -Vector3::x(),
            weight: 100.0,
        },
    ];
    constraints.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    let proposal = solve_structureless_pose(&constraints, &IncrementalSfmConfig::default())
        .expect("a marginal directed line must not veto a stable 3-line refit");
    assert_eq!(proposal.consensus_indices.len(), 3);
    assert!(proposal
        .consensus_indices
        .iter()
        .all(|&index| constraints[index].neighbor != 3));
    assert!(
        structureless_pose_consistency(
            &proposal.pose,
            &constraints,
            &proposal,
            &IncrementalSfmConfig::default(),
        )
        .accepted
    );
}

#[test]
fn structureless_pose_rejects_parallel_center_directions() {
    let constraints = vec![
        StructurelessConstraint {
            neighbor: 0,
            neighbor_center: Point3::new(0.0, 0.0, 0.0),
            missing_rotation: UnitQuaternion::identity(),
            center_direction: Vector3::z(),
            weight: 100.0,
        },
        StructurelessConstraint {
            neighbor: 1,
            neighbor_center: Point3::new(1.0, 0.0, 0.0),
            missing_rotation: UnitQuaternion::identity(),
            center_direction: Vector3::z(),
            weight: 90.0,
        },
    ];
    assert!(solve_structureless_pose(&constraints, &IncrementalSfmConfig::default()).is_err());
}

/// Island-chain fixture. Ten arc cameras all observing one point cloud,
/// with the verified pair graph pruned into a main component
/// `{0, 1, 2, 3, 6, 7, 8, 9}` and a two-image island `{4, 5}` where the
/// bridge image `5` has a *higher* index than its dependent `4`:
///
/// - `4` pairs only with `{2, 3, 5}` — two registered neighbours while
///   `5` is unregistered, below [`IncrementalSfmConfig::
///   structureless_min_neighbors`];
/// - `5` pairs with registered `{3, 6, 7}` plus the island partner `4`.
///
/// Every island pair is narrow-baseline (adjacent arc steps) because the
/// two-view essential estimate degrades on this synthetic cloud beyond
/// ~0.4 rad of arc separation. Disjoint keypoint bands keep every
/// island-touching union-find component at two images, below the
/// track-length floor: the clean global model triangulates from
/// main-component tracks only, leaving the island's observations free
/// for local-submap synthesis — exactly the thin-per-image-structure
/// regime the courtyard second component exposed.
struct IslandScene {
    camera: Camera,
    poses: Vec<Pose>,
    features: Vec<FeatureSet>,
    pairwise: Vec<PairwiseMatches>,
}

fn build_island_scene() -> IslandScene {
    let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let mut points = Vec::new();
    for xi in -3..=3 {
        for yi in -2..=2 {
            for zi in 0..=4 {
                points.push(Point3::new(
                    xi as f64 * 0.3,
                    yi as f64 * 0.3,
                    zi as f64 * 0.25,
                ));
            }
        }
    }
    let mut poses = Vec::new();
    for k in 0..10 {
        let angle = -0.585 + k as f64 * 0.13;
        let radius = 3.0;
        let cam_center = Point3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
        let forward = (Point3::origin() - cam_center).normalize();
        let world_up = Vector3::new(0.0, 1.0, 0.0);
        let right = forward.cross(&world_up).normalize();
        let up = right.cross(&forward);
        let r_cam_to_world = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
        let q_c2w = UnitQuaternion::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_unchecked(r_cam_to_world),
        );
        let q_w2c = q_c2w.inverse();
        let t_w2c = -(q_w2c * cam_center.coords);
        poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
    }

    // Every camera sees every point; keypoint index == point index.
    let features: Vec<FeatureSet> = poses
        .iter()
        .map(|pose| {
            let (kps, descs): (Vec<_>, Vec<_>) = points
                .iter()
                .enumerate()
                .filter_map(|(pidx, p)| {
                    project(&camera, pose, p).map(|px| (px, vec![pidx as f32, 1.0, 0.0, 0.0]))
                })
                .unzip();
            FeatureSet::new(kps, descs).unwrap()
        })
        .collect();

    // Strided keypoint bands. Every band must mix points across all
    // three grid axes: a band confined to one grid slice is exactly
    // planar, and a planar correspondence set makes the two-view
    // essential estimate chirality-degenerate (the failure that
    // motivated this design).
    let all: Vec<usize> = (0..points.len()).collect();
    let main_points: Vec<usize> = all.iter().step_by(4).copied().collect();
    let remainder: Vec<usize> = {
        let main_set: HashSet<usize> = main_points.iter().copied().collect();
        all.into_iter().filter(|p| !main_set.contains(p)).collect()
    };
    let island_band =
        |k: usize| -> Vec<usize> { remainder.iter().skip(k).step_by(6).copied().collect() };
    let band_a = island_band(0);
    let band_b = island_band(1);
    let band_c = island_band(2);
    let band_d = island_band(3);
    let band_e = island_band(4);
    let band_f = island_band(5);

    let pair = |image_i: usize, image_j: usize, band: &[usize]| PairwiseMatches {
        image_i,
        image_j,
        matches: band.iter().map(|&p| (p, p)).collect(),
        two_view_config: None,
        essential_matches: None,
        essential_matrix: None,
    };

    let mut pairwise = Vec::new();
    let main = [0usize, 1, 2, 3, 6, 7, 8, 9];
    for (a, &i) in main.iter().enumerate() {
        for &j in main.iter().skip(a + 1) {
            pairwise.push(pair(i, j, &main_points));
        }
    }
    pairwise.push(pair(4, 3, &band_a));
    pairwise.push(pair(4, 2, &band_b));
    pairwise.push(pair(4, 5, &band_c));
    pairwise.push(pair(5, 6, &band_d));
    pairwise.push(pair(5, 3, &band_e));
    pairwise.push(pair(5, 7, &band_f));

    IslandScene {
        camera,
        poses,
        features,
        pairwise,
    }
}

#[test]
fn structureless_rounds_chain_an_island_through_a_higher_indexed_bridge() {
    let scene = build_island_scene();
    let min_track_length = 5;
    let mut tracks = build_tracks(scene.features.len(), &scene.pairwise, min_track_length);
    assert!(!tracks.is_empty(), "fixture sanity: main tracks must form");
    for track in &tracks {
        let images: HashSet<usize> = track.iter().map(|&(image, _)| image).collect();
        assert!(
            !images.contains(&4) && !images.contains(&5),
            "fixture sanity: island observations must not join global tracks"
        );
    }

    // Register only the main component with ground-truth poses and
    // triangulate the clean model.
    let mut poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
    poses[4] = None;
    poses[5] = None;
    let mut track_point = vec![None; tracks.len()];
    let config = IncrementalSfmConfig {
        colmap_style_mapper: true,
        structureless_registration: true,
        structureless_min_pair_inliers: 5,
        structureless_min_support_tracks: 6,
        // The 20-point synthetic essentials carry ~1 deg of rotation
        // noise, which at fx=500 is ~9 px of reprojection — far beyond
        // the production-default 2 px admission gate that real
        // hundreds-of-inlier matches easily meet. This fixture exercises
        // the round-chaining mechanics, not the pixel gate (which has
        // its own dedicated tests), so the gate is widened accordingly.
        structureless_max_reprojection_error_px: 12.0,
        max_reprojection_error_px: 12.0,
        ..IncrementalSfmConfig::default()
    };
    triangulate_pending(
        &scene.camera,
        &scene.features,
        &tracks,
        &poses,
        &config,
        &mut track_point,
    );
    let clean_points = track_point.iter().filter(|p| p.is_some()).count();
    assert!(
        clean_points >= 10,
        "fixture sanity: clean model must triangulate ({clean_points} points)"
    );

    // A single ascending scan must register the bridge `6` but leave `3`
    // behind: when the scan reaches `3`, `6` is still unregistered and `3`
    // has only two admissible neighbours.
    let single_round_config = IncrementalSfmConfig {
        structureless_max_rounds: 1,
        ..config.clone()
    };
    let single_registered = structureless_registration_rounds(
        &scene.camera,
        &scene.features,
        &scene.pairwise,
        &mut tracks.clone(),
        &single_round_config,
        &mut poses.clone(),
        &mut track_point.clone(),
    );
    assert_eq!(
        single_registered, 1,
        "one ascending pass must recover exactly the bridge image"
    );

    // Multiple rounds feed `6` back in as a neighbour and chain `3`.
    let total_registered = structureless_registration_rounds(
        &scene.camera,
        &scene.features,
        &scene.pairwise,
        &mut tracks,
        &config,
        &mut poses,
        &mut track_point,
    );
    assert_eq!(
        total_registered, 2,
        "rounds must chain the dependent island image through the bridge"
    );
    assert!(poses.iter().all(Option::is_some));

    // The chained pose must sit at the true (metric) geometry: rotation
    // tight, centre within a fraction of the neighbour spread.
    for image in [4usize, 5] {
        let pose = poses[image].as_ref().unwrap();
        let rotation_error = (pose.world_to_camera.rotation.inverse()
            * scene.poses[image].world_to_camera.rotation)
            .angle();
        let center_error =
            (pose.camera_center_world() - scene.poses[image].camera_center_world()).norm();
        assert!(
            rotation_error < 0.01,
            "image {image} rotation error {rotation_error} rad too large"
        );
        assert!(
            center_error < 0.05,
            "image {image} centre error {center_error} m too large"
        );
    }
}
