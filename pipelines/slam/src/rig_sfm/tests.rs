use nalgebra::{UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;
use visloc_core::types::Camera;
use visloc_vision::pnp::RigSensor;

use super::*;

#[test]
fn covisibility_deduplicates_stereo_votes_and_selects_deterministically() {
    let assignments = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)];
    let track = |images: &[usize]| WorkingTrack {
        observations: images.iter().map(|&image| (image, 0)).collect(),
        position: Some(Point3::new(0.0, 0.0, 1.0)),
        metric_anchored: true,
    };
    let tracks = vec![track(&[0, 1, 2, 3, 4]), track(&[0, 4]), track(&[5])];
    let selected =
        covisible_local_frames(0, 2, [0, 0, 1, 2], &tracks, &assignments, |_, _, _| true);
    assert_eq!(selected, HashSet::from([0, 2]));
    assert_eq!(
        selected,
        covisible_local_frames(0, 2, [2, 1, 0], &tracks, &assignments, |_, _, _| true)
    );
    assert_eq!(
        covisible_local_frames(0, 2, [0], &tracks, &assignments, |_, _, _| true),
        HashSet::from([0, 1])
    );
    assert!(
        covisible_local_frames(0, 40, [0, 1], &tracks, &assignments, |image, _, _| image
            > 1)
        .is_empty()
    );
    for cap in [0, 1] {
        assert!(
            covisible_local_frames(0, cap, [0, 1], &tracks, &assignments, |_, _, _| true)
                .is_empty()
        );
    }
}

#[test]
fn covisibility_long_track_keeps_bounded_window() {
    let assignments = (0..100_000).map(|frame| (frame, 0)).collect::<Vec<_>>();
    let mut tracks = vec![WorkingTrack {
        observations: (0..100_000).map(|image| (image, 0)).collect(),
        position: Some(Point3::new(0.0, 0.0, 1.0)),
        metric_anchored: true,
    }];
    let selected = covisible_local_frames(99_999, 40, [0], &tracks, &assignments, |_, _, _| true);
    assert_eq!(selected.len(), 40);
    assert!(selected.contains(&99_999));
    assert!((0..39).all(|frame| selected.contains(&frame)));
    tracks[0].position = None;
    assert!(
        covisible_local_frames(99_999, 40, [0], &tracks, &assignments, |_, _, _| true).is_empty()
    );
}

#[test]
fn registration_trace_line_formats_camera_center_precisely() {
    let pose =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-1.25, 2.5, -3.75));
    assert_eq!(
        registration_trace_line(
            RegistrationTrace {
                mode: "dynamic",
                order: 4,
                frame: 7,
                support: 12,
                inliers: 10,
                sensors: 2,
            },
            &pose,
        ),
        "rig-registration-trace: mode=dynamic order=4 frame=7 support=12 inliers=10 sensors=2 cx=1.25000000000000000e0 cy=-2.50000000000000000e0 cz=3.75000000000000000e0"
    );
}

#[test]
fn metric_seed_support_counts_each_multisensor_track_once_per_frame() {
    let assignment = vec![(2, 1), (0, 0), (2, 0), (1, 0), (0, 1), (1, 0), (2, 2)];
    let track = |observations| WorkingTrack {
        observations,
        position: None,
        metric_anchored: false,
    };
    let tracks = vec![
        track(vec![(0, 0), (1, 0), (2, 0), (3, 0)]),
        track(vec![(4, 0), (1, 1)]),
        track(vec![(3, 1), (5, 0)]),
        track(vec![(6, 0), (2, 1), (0, 1)]),
    ];

    assert_eq!(metric_frame_supports(3, &tracks, &assignment), [1, 0, 2]);
    assert_eq!(metric_seed_candidates(&[6, 8, 8, 5]), [1, 2, 0]);
}

#[test]
fn temporal_track_support_bins_tracks_by_anchor_and_span_without_pair_state() {
    let assignment = vec![
        (0, 0),
        (0, 1),
        (7, 0),
        (8, 0),
        (15, 0),
        (16, 0),
        (127, 0),
        (128, 0),
    ];
    let counts = temporal_track_support(
        vec![
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![0, 5],
            vec![0, 6],
            vec![0, 7],
        ],
        &assignment,
        8,
    );

    assert_eq!(
        counts.get(&(0, "same-frame")),
        Some(&TemporalTrackSupportCount {
            tracks: 1,
            observations: 2,
            metric_tracks: 1,
        })
    );
    for (frame_bin, span_class) in [
        (0, "1-7"),
        (1, "8-15"),
        (2, "16-31"),
        (15, "32-127"),
        (16, "128+"),
    ] {
        assert_eq!(
            counts.get(&(frame_bin, span_class)),
            Some(&TemporalTrackSupportCount {
                tracks: 1,
                observations: 2,
                metric_tracks: 0,
            })
        );
    }
}

#[test]
fn robust_triangulation_prunes_one_registered_outlier_instead_of_dropping_track() {
    let camera = Camera::pinhole(1, 640, 480, 100.0, 100.0, 320.0, 240.0);
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: camera.clone(),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera,
            sensor_from_rig: SE3::identity(),
        },
    ])
    .unwrap();
    let features = vec![
        FeatureSet::new(vec![Point2::new(320.0, 240.0)], vec![Vec::new()]).unwrap(),
        FeatureSet::new(vec![Point2::new(300.0, 240.0)], vec![Vec::new()]).unwrap(),
        FeatureSet::new(vec![Point2::new(280.0, 240.0)], vec![Vec::new()]).unwrap(),
        FeatureSet::new(vec![Point2::new(300.0, 242.0)], vec![Vec::new()]).unwrap(),
    ];
    let assignment = vec![(0, 0), (1, 0), (2, 0), (1, 1)];
    let image_poses = [0.0, 1.0, 2.0, 1.0]
        .into_iter()
        .map(|centre_x| {
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-centre_x, 0.0, 0.0),
            ))
        })
        .collect::<Vec<_>>();
    let track = || WorkingTrack {
        observations: vec![(0, 0), (1, 0), (2, 0), (3, 0)],
        position: None,
        metric_anchored: true,
    };
    let config = RigSfmConfig {
        max_reprojection_error_px: 1.0,
        robust_triangulation_pruning: true,
        triangulation_min_inlier_fraction: 0.5,
        ..RigSfmConfig::default()
    };
    let mut robust_tracks = vec![track()];
    let robust = triangulate_frontier(
        &rig,
        &features,
        &assignment,
        &image_poses,
        &config,
        &mut robust_tracks,
        HashSet::from([0]),
    );
    assert_eq!(robust.landmarks, [0]);
    assert_eq!(robust.robust_tracks, 1);
    assert_eq!(robust.pruned_observations, 1);
    assert_eq!(robust_tracks[0].observations.len(), 3);

    let mut strict_tracks = vec![track()];
    let strict = triangulate_frontier(
        &rig,
        &features,
        &assignment,
        &image_poses,
        &RigSfmConfig {
            max_reprojection_error_px: 1.0,
            ..RigSfmConfig::default()
        },
        &mut strict_tracks,
        HashSet::from([0]),
    );
    assert!(strict.landmarks.is_empty());
}

#[test]
fn final_filter_prunes_bad_registered_observations_but_keeps_unregistered_support() {
    let camera = Camera::pinhole(1, 640, 480, 100.0, 100.0, 320.0, 240.0);
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: camera.clone(),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera,
            sensor_from_rig: SE3::identity(),
        },
    ])
    .unwrap();
    let features = vec![
        FeatureSet::new(vec![Point2::new(320.0, 240.0)], vec![Vec::new()]).unwrap(),
        FeatureSet::new(vec![Point2::new(300.0, 240.0)], vec![Vec::new()]).unwrap(),
        FeatureSet::new(vec![Point2::new(280.0, 240.0)], vec![Vec::new()]).unwrap(),
        FeatureSet::new(vec![Point2::new(300.0, 242.0)], vec![Vec::new()]).unwrap(),
        FeatureSet::new(vec![Point2::new(0.0, 0.0)], vec![Vec::new()]).unwrap(),
    ];
    let assignment = vec![(0, 0), (1, 0), (2, 0), (1, 1), (3, 0)];
    let mut image_poses = [0.0, 1.0, 2.0, 1.0]
        .into_iter()
        .map(|centre_x| {
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-centre_x, 0.0, 0.0),
            ))
        })
        .collect::<Vec<_>>();
    image_poses.push(None);
    let point = Point3::new(0.0, 0.0, 5.0);
    let mut tracks = vec![
        WorkingTrack {
            observations: vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            position: Some(point),
            metric_anchored: true,
        },
        WorkingTrack {
            observations: vec![(0, 0), (3, 0)],
            position: Some(point),
            metric_anchored: true,
        },
    ];

    let pruned = filter_positioned_track_observations(
        &rig,
        &features,
        &assignment,
        &image_poses,
        1.0,
        &mut tracks,
    );

    assert_eq!(pruned, 3);
    assert_eq!(tracks[0].observations, [(0, 0), (1, 0), (2, 0), (4, 0)]);
    assert!(!tracks[0].metric_anchored);
    assert_eq!(tracks[0].position, Some(point));
    assert_eq!(tracks[1].position, None);
}

#[test]
fn bounded_ray_sampling_is_stable_and_never_exceeds_the_cap() {
    assert!(bounded_evenly_spaced_indices(8, 0).is_empty());
    assert_eq!(bounded_evenly_spaced_indices(0, 8), Vec::<usize>::new());
    assert_eq!(bounded_evenly_spaced_indices(4, 8), [0, 1, 2, 3]);
    assert_eq!(bounded_evenly_spaced_indices(100, 1), [0]);

    let sampled = bounded_evenly_spaced_indices(10_000, 16);
    assert_eq!(sampled.len(), 16);
    assert_eq!(sampled.first(), Some(&0));
    assert_eq!(sampled.last(), Some(&9_999));
    assert!(sampled.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn deferred_retriangulation_prunes_only_failed_new_tracks() {
    let track = |observation, position| WorkingTrack {
        observations: vec![observation],
        position,
        metric_anchored: false,
    };
    let point = Point3::new(0.0, 0.0, 5.0);
    let mut tracks = vec![
        track((0, 0), None),
        track((1, 0), Some(point)),
        track((2, 0), None),
        track((3, 0), Some(point)),
    ];

    let update = retain_positioned_deferred_tracks(&mut tracks, 2);

    assert_eq!(update.tracks, 1);
    assert_eq!(update.observations, 1);
    assert_eq!(tracks.len(), 3);
    assert_eq!(tracks[0].observations, [(0, 0)]);
    assert_eq!(tracks[1].observations, [(1, 0)]);
    assert_eq!(tracks[2].observations, [(3, 0)]);
}

#[test]
fn metric_anchor_frame_count_requires_distinct_sensors_in_distinct_frames() {
    let assignment = vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 0)];
    assert_eq!(
        metric_anchor_frame_count(
            &[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
            &assignment,
        ),
        2
    );
    assert_eq!(
        metric_anchor_frame_count(&[(0, 0), (2, 0), (3, 0)], &assignment),
        1
    );
}

#[test]
fn bounded_track_completion_is_transitive_without_merging_positioned_tracks() {
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.2, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let features = (0..4)
        .map(|_| FeatureSet::new(vec![Point2::new(320.0, 240.0)], vec![vec![0.0]]).unwrap())
        .collect::<Vec<_>>();
    let assignment = vec![(0, 0), (1, 0), (2, 0), (3, 0)];
    let poses = vec![Some(Pose::identity()); 4];
    // The first edge is deliberately visited before image 1 acquires an
    // owner, forcing the continuation to the second bounded pass.
    let pairwise = vec![
        PairwiseMatches::new(1, 2, vec![(0, 0)]),
        PairwiseMatches::new(0, 1, vec![(0, 0)]),
        PairwiseMatches::new(0, 3, vec![(0, 0)]),
    ];
    let mut tracks = vec![
        WorkingTrack {
            observations: vec![(0, 0)],
            position: Some(Point3::new(0.0, 0.0, 5.0)),
            metric_anchored: true,
        },
        WorkingTrack {
            observations: vec![(3, 0)],
            position: Some(Point3::new(0.0, 0.0, 5.0)),
            metric_anchored: true,
        },
        WorkingTrack {
            observations: vec![(1, 0)],
            position: None,
            metric_anchored: false,
        },
    ];

    let update = complete_positioned_tracks(
        &rig,
        &features,
        &pairwise,
        &assignment,
        &poses,
        2,
        1.0,
        &mut tracks,
    );

    assert_eq!(update.passes, 2);
    assert_eq!(update.pair_visits, 6);
    assert_eq!(update.observations, 2);
    assert_eq!(tracks[0].observations, [(0, 0), (1, 0), (2, 0)]);
    assert_eq!(tracks[1].observations, [(3, 0)]);
    assert!(tracks[2].observations.is_empty());
}

#[test]
fn conflict_preserving_track_builder_rejects_only_the_bad_edge() {
    let features = (0..3)
        .map(|_| {
            FeatureSet::new(
                vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
                vec![vec![0.0], vec![1.0]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let pairwise = vec![
        PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1), (0, 1)]),
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
    ];
    let assignment = vec![(0, 0), (1, 0), (2, 0)];
    let legacy =
        build_rig_track_output(&features, &pairwise, &assignment, &RigSfmConfig::default());
    let preserved = build_rig_track_output(
        &features,
        &pairwise,
        &assignment,
        &RigSfmConfig {
            track_builder: RigTrackBuilder::ConflictPreserving,
            ..RigSfmConfig::default()
        },
    );

    assert_eq!(legacy.stats.retained_tracks, 0);
    assert_eq!(legacy.stats.conflicting_components, 1);
    assert_eq!(preserved.stats.retained_tracks, 2);
    assert_eq!(preserved.stats.retained_observations, 6);
    assert_eq!(preserved.stats.conflicting_components, 1);
}

#[test]
fn stream_order_builder_protects_a_trusted_prefix_from_a_late_bridge() {
    let features = (0..3)
        .map(|_| {
            FeatureSet::new(
                vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
                vec![vec![0.0], vec![1.0]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let pairwise = vec![
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(0, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(0, 1, vec![(0, 1), (1, 0)]),
    ];
    let assignment = vec![(0, 0), (1, 0), (2, 0)];
    let output = build_rig_track_output(
        &features,
        &pairwise,
        &assignment,
        &RigSfmConfig {
            track_builder: RigTrackBuilder::StreamOrderConflictPreserving,
            ..RigSfmConfig::default()
        },
    );

    assert_eq!(output.stats.conflicting_components, 2);
    assert_eq!(output.tracks.len(), 2);
    assert!(output.tracks.contains(&vec![(0, 0), (1, 0), (2, 0)]));
    assert!(output.tracks.contains(&vec![(0, 1), (1, 1), (2, 1)]));
}

#[test]
fn trusted_pair_prefix_is_admitted_before_a_stronger_late_bridge() {
    let features = (0..3)
        .map(|_| {
            FeatureSet::new(
                vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
                vec![vec![0.0], vec![1.0]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let pairwise = vec![
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(0, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(0, 1, vec![(0, 1), (1, 0), (0, 1)]),
    ];
    let assignment = vec![(0, 0), (1, 0), (2, 0)];
    let output = build_rig_track_output(
        &features,
        &pairwise,
        &assignment,
        &RigSfmConfig {
            track_builder: RigTrackBuilder::TrustedPrefixPairConfidence(2),
            ..RigSfmConfig::default()
        },
    );

    assert_eq!(output.tracks.len(), 2);
    assert!(output.tracks.contains(&vec![(0, 0), (1, 0), (2, 0)]));
    assert!(output.tracks.contains(&vec![(0, 1), (1, 1), (2, 1)]));
}

#[test]
fn sparse_cycle_orders_third_view_supported_edges_before_a_bad_edge() {
    let features = (0..3)
        .map(|_| {
            FeatureSet::new(
                vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
                vec![vec![0.0], vec![1.0]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let pairwise = vec![
        PairwiseMatches::new(0, 1, vec![(0, 0), (0, 1), (1, 1)]),
        PairwiseMatches::new(0, 2, vec![(0, 0), (1, 1)]),
        PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
    ];

    let output = build_tracks_sparse_cycle(&features, &pairwise, 2, None, false);

    assert_eq!(output.tracks.len(), 2);
    assert_eq!(output.stats.retained_observations, 6);
    assert_eq!(output.stats.conflicting_components, 1);
    assert!(output.tracks.contains(&vec![(0, 0), (1, 0), (2, 0)]));
    assert!(output.tracks.contains(&vec![(0, 1), (1, 1), (2, 1)]));
}

#[test]
fn metric_temporal_quadrilateral_requires_all_four_unique_edges() {
    let features = (0..4)
        .map(|_| {
            FeatureSet::new(
                vec![
                    Point2::new(0.0, 0.0),
                    Point2::new(1.0, 0.0),
                    Point2::new(2.0, 0.0),
                ],
                vec![vec![0.0], vec![1.0], vec![2.0]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    // Images 0/1 are the two sensors at frame 0; 2/3 are frame 1.
    let assignment = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
    let identity = vec![(0, 0), (1, 1)];
    let complete = vec![
        PairwiseMatches::new(0, 1, identity.clone()),
        PairwiseMatches::new(2, 3, identity.clone()),
        PairwiseMatches::new(0, 2, identity.clone()),
        PairwiseMatches::new(1, 3, identity),
    ];

    let output = build_metric_temporal_quadrilaterals(&features, &complete, &assignment, 2);
    assert_eq!(output.tracks.len(), 2);
    assert_eq!(output.stats.retained_observations, 8);
    assert!(output
        .tracks
        .contains(&vec![(0, 0), (1, 0), (2, 0), (3, 0)]));
    assert!(output
        .tracks
        .contains(&vec![(0, 1), (1, 1), (2, 1), (3, 1)]));

    let wide_assignment = vec![(0, 0), (0, 1), (32, 0), (32, 1)];
    assert!(
        build_metric_temporal_quadrilaterals(&features, &complete, &wide_assignment, 2,)
            .tracks
            .is_empty()
    );
    assert_eq!(
        metric_temporal_quadrilateral_tracks_in_frame_gap(
            &features,
            &complete,
            &wide_assignment,
            32,
            128,
        )
        .len(),
        2
    );

    let missing_temporal_sensor = &complete[..3];
    assert!(build_metric_temporal_quadrilaterals(
        &features,
        missing_temporal_sensor,
        &assignment,
        2,
    )
    .tracks
    .is_empty());

    let mut ambiguous = complete;
    ambiguous[0].matches.push((0, 2));
    let ambiguous_output =
        build_metric_temporal_quadrilaterals(&features, &ambiguous, &assignment, 2);
    assert_eq!(
        ambiguous_output.tracks,
        vec![vec![(0, 1), (1, 1), (2, 1), (3, 1)]]
    );
}

#[test]
fn metric_temporal_quadrilateral_assigns_each_observation_once() {
    let features = (0..6)
        .map(|_| FeatureSet::new(vec![Point2::origin()], vec![vec![0.0]]).unwrap())
        .collect::<Vec<_>>();
    let assignment = vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)];
    let pairwise = vec![
        PairwiseMatches::new(0, 1, vec![(0, 0)]),
        PairwiseMatches::new(3, 4, vec![(0, 0)]),
        PairwiseMatches::new(0, 3, vec![(0, 0)]),
        PairwiseMatches::new(1, 4, vec![(0, 0)]),
        PairwiseMatches::new(0, 2, vec![(0, 0)]),
        PairwiseMatches::new(3, 5, vec![(0, 0)]),
        PairwiseMatches::new(2, 5, vec![(0, 0)]),
    ];

    let output = build_metric_temporal_quadrilaterals(&features, &pairwise, &assignment, 2);
    assert_eq!(output.tracks, vec![vec![(0, 0), (1, 0), (3, 0), (4, 0)]]);
    assert_eq!(output.stats.connected_components, 2);
    assert_eq!(output.stats.conflicting_components, 1);
}

#[test]
fn deferred_interpolation_requires_verified_support_and_a_bounded_gap() {
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.20, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let frames = (0..4)
        .map(|frame| RigFrame {
            images: vec![
                RigFrameImage {
                    image_index: frame * 2,
                    sensor_index: 0,
                },
                RigFrameImage {
                    image_index: frame * 2 + 1,
                    sensor_index: 1,
                },
            ],
        })
        .collect::<Vec<_>>();
    let endpoints = || {
        vec![
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::zeros(),
            )),
            None,
            None,
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-3.0, 0.0, 0.0),
            )),
        ]
    };
    let verified = vec![vec![], vec![0], vec![1], vec![]];

    let mut too_short = endpoints();
    let mut image_poses = vec![None; 8];
    assert_eq!(
        interpolate_deferred_pose_gaps(
            &rig,
            &frames,
            &verified,
            1,
            &mut too_short,
            &mut image_poses,
        ),
        0
    );
    assert!(too_short[1].is_none() && too_short[2].is_none());

    let mut unsupported = endpoints();
    let mut image_poses = vec![None; 8];
    let missing_support = vec![vec![], vec![0], vec![], vec![]];
    assert_eq!(
        interpolate_deferred_pose_gaps(
            &rig,
            &frames,
            &missing_support,
            2,
            &mut unsupported,
            &mut image_poses,
        ),
        0
    );

    let mut interpolated = endpoints();
    let mut image_poses = vec![None; 8];
    assert_eq!(
        interpolate_deferred_pose_gaps(
            &rig,
            &frames,
            &verified,
            2,
            &mut interpolated,
            &mut image_poses,
        ),
        2
    );
    assert!((interpolated[1].as_ref().unwrap().camera_center_world().x - 1.0).abs() < 1.0e-12);
    assert!((interpolated[2].as_ref().unwrap().camera_center_world().x - 2.0).abs() < 1.0e-12);
    assert!(image_poses[2..6].iter().all(Option::is_some));
}

#[test]
fn isolated_pose_repair_replaces_only_a_large_one_frame_detour() {
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.20, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let frames = (0..5)
        .map(|frame| RigFrame {
            images: vec![
                RigFrameImage {
                    image_index: frame * 2,
                    sensor_index: 0,
                },
                RigFrameImage {
                    image_index: frame * 2 + 1,
                    sensor_index: 1,
                },
            ],
        })
        .collect::<Vec<_>>();
    let mut frame_poses = [0.0, 1.0, 11.0, 3.0, 4.0]
        .into_iter()
        .map(|centre| {
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-centre, 0.0, 0.0),
            ))
        })
        .collect::<Vec<_>>();
    let mut image_poses = vec![None; 10];

    let update = repair_isolated_pose_outliers(
        &rig,
        &frames,
        0,
        0.25,
        8.0,
        2,
        &mut frame_poses,
        &mut image_poses,
    );

    assert_eq!(
        update,
        IsolatedPoseRepairUpdate {
            passes: 1,
            repairs: 1,
        }
    );
    assert!((frame_poses[2].as_ref().unwrap().camera_center_world().x - 2.0).abs() < 1.0e-12);
    assert!((frame_poses[1].as_ref().unwrap().camera_center_world().x - 1.0).abs() < 1.0e-12);
    assert!((frame_poses[3].as_ref().unwrap().camera_center_world().x - 3.0).abs() < 1.0e-12);
    assert!(image_poses[4].is_some() && image_poses[5].is_some());
}

#[test]
fn paired_pose_jump_repair_shifts_a_contiguous_offset_and_respects_boundaries() {
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.20, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let frames = (0..7)
        .map(|frame| RigFrame {
            images: (0..2)
                .map(|sensor| RigFrameImage {
                    image_index: frame * 2 + sensor,
                    sensor_index: sensor,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let centres = [0.0, 0.1, 10.2, 10.3, 0.4, 0.5, 0.6];
    let make_poses = || {
        centres
            .into_iter()
            .map(|centre| {
                Some(Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(-centre, 0.0, 0.0),
                ))
            })
            .collect::<Vec<_>>()
    };
    let mut frame_poses = make_poses();
    let mut image_poses = vec![None; frames.len() * 2];
    let update = repair_paired_pose_jumps(
        &rig,
        &frames,
        0,
        0.25,
        8.0,
        4,
        0.1,
        &mut frame_poses,
        &mut image_poses,
    );
    assert_eq!(
        update,
        PairedPoseJumpRepairUpdate {
            repairs: 1,
            repaired_frames: 2,
        }
    );
    for (actual, expected) in frame_poses.iter().zip([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) {
        assert!((actual.as_ref().unwrap().camera_center_world().x - expected).abs() < 1.0e-12);
    }
    assert!(image_poses[4..8].iter().all(Option::is_some));

    // A normal fast trajectory has no jump above the robust global-step
    // floor, while a missing frame prevents a candidate from crossing the
    // model boundary. Both cases leave every pose untouched.
    let normal_centres = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
    let mut normal = normal_centres
        .into_iter()
        .map(|centre| {
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-centre, 0.0, 0.0),
            ))
        })
        .collect::<Vec<_>>();
    let normal_before = normal.clone();
    let mut normal_images = vec![None; normal.len() * 2];
    assert_eq!(
        repair_paired_pose_jumps(
            &rig,
            &frames[..normal.len()],
            0,
            0.25,
            8.0,
            4,
            0.1,
            &mut normal,
            &mut normal_images,
        ),
        PairedPoseJumpRepairUpdate::default()
    );
    assert_eq!(normal, normal_before);

    let mut missing = make_poses();
    missing[3] = None;
    let missing_before = missing.clone();
    let mut missing_images = vec![None; frames.len() * 2];
    assert_eq!(
        repair_paired_pose_jumps(
            &rig,
            &frames,
            0,
            0.25,
            8.0,
            4,
            0.1,
            &mut missing,
            &mut missing_images,
        ),
        PairedPoseJumpRepairUpdate::default()
    );
    assert_eq!(missing, missing_before);

    let mut seed_internal = make_poses();
    let seed_before = seed_internal.clone();
    let mut seed_images = vec![None; frames.len() * 2];
    assert_eq!(
        repair_paired_pose_jumps(
            &rig,
            &frames,
            2,
            0.25,
            8.0,
            4,
            0.1,
            &mut seed_internal,
            &mut seed_images,
        ),
        PairedPoseJumpRepairUpdate::default()
    );
    assert_eq!(seed_internal, seed_before);

    // Candidate selection and application are deterministic across runs.
    let mut repeat = make_poses();
    let mut repeat_images = vec![None; frames.len() * 2];
    let repeat_update = repair_paired_pose_jumps(
        &rig,
        &frames,
        0,
        0.25,
        8.0,
        4,
        0.1,
        &mut repeat,
        &mut repeat_images,
    );
    assert_eq!(repeat, frame_poses);
    assert_eq!(repeat_update, update);
}

#[test]
fn reconstructs_metric_synchronized_rig_frames() {
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera: Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5),
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.20, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let world_points = (0..24)
        .map(|index| {
            Point3::new(
                (index % 6) as f64 * 0.28 - 0.7,
                (index / 6) as f64 * 0.24 - 0.35,
                4.0 + (index % 5) as f64 * 0.17,
            )
        })
        .collect::<Vec<_>>();
    let truth = (0..8)
        .map(|frame| {
            Pose::from_world_to_camera(
                UnitQuaternion::from_euler_angles(
                    0.002 * frame as f64,
                    -0.01 * frame as f64,
                    0.003 * frame as f64,
                ),
                Vector3::new(-0.08 * frame as f64, 0.01 * frame as f64, 0.0),
            )
        })
        .collect::<Vec<_>>();
    let mut frames = Vec::new();
    let mut features = Vec::new();
    for (frame_index, frame_pose) in truth.iter().enumerate() {
        let mut frame_images = Vec::new();
        for sensor_index in 0..2 {
            let image_index = features.len();
            let image_pose = rig.sensors()[sensor_index]
                .sensor_from_rig
                .compose(&frame_pose.world_to_camera);
            let keypoints = world_points
                .iter()
                .map(|point| {
                    rig.sensors()[sensor_index]
                        .camera
                        .project(&image_pose.transform_point(point))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            features.push(FeatureSet::new(keypoints, vec![vec![0.0]; world_points.len()]).unwrap());
            frame_images.push(RigFrameImage {
                image_index,
                sensor_index,
            });
        }
        assert_eq!(frame_images[0].image_index, frame_index * 2);
        frames.push(RigFrame {
            images: frame_images,
        });
    }
    let identity_matches = (0..world_points.len())
        .map(|index| (index, index))
        .collect::<Vec<_>>();
    let mut pairwise = Vec::new();
    for frame in 0..8 {
        pairwise.push(PairwiseMatches::new(
            frame * 2,
            frame * 2 + 1,
            identity_matches.clone(),
        ));
        if frame + 1 < 8 {
            pairwise.push(PairwiseMatches::new(
                frame * 2,
                (frame + 1) * 2,
                identity_matches.clone(),
            ));
            pairwise.push(PairwiseMatches::new(
                frame * 2 + 1,
                (frame + 1) * 2 + 1,
                identity_matches.clone(),
            ));
        }
    }

    let result = incremental_rig_sfm(
        &rig,
        &frames,
        &features,
        &pairwise,
        &RigSfmConfig::default(),
    )
    .unwrap();
    let recovery_pairs = vec![
        PairwiseMatches::new(0, 1, vec![(0, 0)]),
        PairwiseMatches::new(1, 2, vec![(0, 0)]),
        PairwiseMatches::new(0, 2, vec![(0, 0), (1, 0)]),
    ];
    let recovered = recover_metric_conflict_tracks(
        &rig,
        &features,
        &recovery_pairs,
        &[vec![(0, 0), (0, 1), (1, 0), (2, 0)]],
        &image_assignment(&frames, features.len()),
        &result.image_poses,
        &RigSfmConfig {
            recover_metric_conflict_tracks: true,
            conflict_recovery_max_reprojection_error_px: 0.1,
            conflict_recovery_max_mean_reprojection_px: 0.05,
            ..RigSfmConfig::default()
        },
    );
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].observations, vec![(0, 0), (1, 0), (2, 0)]);
    assert!(recovered[0].metric_anchored);
    assert!((recovered[0].position.unwrap() - world_points[0]).norm() < 1.0e-4);
    assert_eq!(result.registered_frames, 8);
    assert_eq!(result.registered_images, 16);
    assert_eq!(result.tracks.len(), world_points.len());
    assert!(result.mean_reprojection_error_px < 1.0e-4);
    let total_observations = result
        .tracks
        .iter()
        .map(|track| track.observations.len())
        .sum::<usize>();
    assert_eq!(
        result.work.correspondence_cache_insertions,
        total_observations
    );
    assert!(result.work.triangulation_attempts <= total_observations);
    assert!(result.work.pnp_attempts < frames.len());
    assert_eq!(
        result.work.pnp_attempts,
        result.work.pnp_insufficient_sensor_attempts
            + result.work.pnp_estimation_failures
            + result.work.pnp_inlier_rejections
            + result.work.pnp_registrations
    );
    assert_eq!(
        frames.len() - result.registered_frames,
        result.work.unregistered_zero_support_frames
            + result.work.unregistered_below_pnp_support_frames
            + result.work.unregistered_eligible_pnp_frames
    );
    for (estimated, expected) in result.frame_poses.iter().zip(&truth) {
        let estimated = estimated.as_ref().unwrap();
        assert!((estimated.camera_center_world() - expected.camera_center_world()).norm() < 1.0e-4);
        assert!(
            estimated
                .world_to_camera
                .rotation
                .angle_to(&expected.world_to_camera.rotation)
                < 1.0e-4
        );
    }
    let baseline = (result.image_poses[0]
        .as_ref()
        .unwrap()
        .camera_center_world()
        - result.image_poses[1]
            .as_ref()
            .unwrap()
            .camera_center_world())
    .norm();
    assert!((baseline - 0.20).abs() < 1.0e-9);

    // Keep the last frame's stereo pair in the base graph, but defer its
    // two temporal links. The base mapper must remain byte-for-byte
    // equivalent for already registered poses; the suffix may only fill
    // the isolated final frame from established landmarks.
    let mut base_pairs = pairwise
        .iter()
        .filter(|pair| !matches!((pair.image_i, pair.image_j), (12, 14) | (13, 15)))
        .cloned()
        .collect::<Vec<_>>();
    let deferred_prefix = base_pairs.len();
    base_pairs.extend(
        pairwise
            .iter()
            .filter(|pair| matches!((pair.image_i, pair.image_j), (12, 14) | (13, 15)))
            .cloned(),
    );
    let no_final_ba = RigSfmConfig {
        final_bundle_adjustment: false,
        structure_refinement_iterations: 0,
        ..RigSfmConfig::default()
    };
    let base = incremental_rig_sfm(
        &rig,
        &frames,
        &features,
        &base_pairs[..deferred_prefix],
        &no_final_ba,
    )
    .unwrap();
    assert_eq!(base.registered_frames, 7);
    let completed = incremental_rig_sfm(
        &rig,
        &frames,
        &features,
        &base_pairs,
        &RigSfmConfig {
            deferred_registration_pair_prefix: Some(deferred_prefix),
            ..no_final_ba
        },
    )
    .unwrap();
    assert_eq!(completed.registered_frames, 8);
    assert_eq!(completed.work.deferred_pair_visits, 2);
    assert_eq!(completed.work.deferred_pnp_attempts, 1);
    assert_eq!(completed.work.deferred_registrations, 1);
    assert_eq!(
        completed.work.deferred_observations_attached,
        world_points.len() * 2
    );
    for frame in 0..7 {
        assert_eq!(completed.frame_poses[frame], base.frame_poses[frame]);
    }

    // A suffix source feature need not belong to a retained base track:
    // synchronized source stereo can supply a short-lived 3D point for
    // PnP without adding that point to the accepted reconstruction.
    let direct_suffix = vec![
        PairwiseMatches::new(12, 14, identity_matches.clone()),
        PairwiseMatches::new(13, 15, identity_matches.clone()),
    ];
    let source_stereo = vec![PairwiseMatches::new(12, 13, identity_matches)];
    let assignment = image_assignment(&frames, features.len());
    let direct_links =
        build_relevant_verified_stereo_links(&source_stereo, &direct_suffix, &assignment);
    let mut direct_frame_poses = truth
        .iter()
        .enumerate()
        .map(|(frame, pose)| (frame < 7).then_some(pose.clone()))
        .collect::<Vec<_>>();
    let mut direct_image_poses = vec![None; features.len()];
    for frame in 0..7 {
        install_image_poses(&rig, &frames[frame], &truth[frame], &mut direct_image_poses);
    }
    let mut no_tracks = Vec::new();
    let mut no_image_tracks = vec![Vec::new(); features.len()];
    let direct = register_deferred_frames(
        &rig,
        &frames,
        &features,
        &direct_suffix,
        &assignment,
        &no_final_ba,
        &mut direct_frame_poses,
        &mut direct_image_poses,
        &mut no_tracks,
        &mut no_image_tracks,
        &direct_links,
    );
    assert_eq!(direct.registrations, 1);
    assert_eq!(direct.observations_attached, 0);
    assert!(
        (direct_frame_poses[7]
            .as_ref()
            .unwrap()
            .camera_center_world()
            - truth[7].camera_center_world())
        .norm()
            < 1.0e-4
    );

    let links = build_verified_stereo_links(&pairwise, &assignment);
    let adjacency = build_frame_pair_adjacency(&pairwise, &assignment, frames.len());
    let mut image_poses = vec![None; features.len()];
    install_image_poses(&rig, &frames[0], &Pose::identity(), &mut image_poses);
    let mut source_queue = vec![0];
    let mut pair_visited = vec![false; pairwise.len()];
    let mut target_observations = vec![HashSet::new(); frames.len()];
    let mut cached = vec![Vec::new(); frames.len()];
    let mut versions = vec![0; frames.len()];
    let mut heap = BinaryHeap::new();
    let direct_config = RigSfmConfig {
        direct_stereo_pnp_max_frame_gap: 1,
        ..RigSfmConfig::default()
    };
    let direct = append_direct_stereo_pnp_correspondences(
        &rig,
        &frames,
        &features,
        &pairwise,
        &assignment,
        &image_poses,
        &direct_config,
        &links,
        &adjacency,
        &mut source_queue,
        &mut pair_visited,
        &mut target_observations,
        &mut cached,
        &mut versions,
        &mut heap,
    );
    assert_eq!(direct.pair_visits, 2);
    assert_eq!(direct.insertions, world_points.len() * 2);
    let correspondences = cached[1]
        .iter()
        .map(|cached| GeneralizedCorrespondence2D3D {
            sensor_index: cached.sensor_index,
            point2d: cached.point2d,
            point3d: cached.direct_point3d.unwrap(),
            confidence: None,
        })
        .collect::<Vec<_>>();
    let bridged = GeneralizedPnPRansac {
        iterations: direct_config.pnp_max_iterations,
        reprojection_threshold: direct_config.max_reprojection_error_px,
        seed: direct_config.ransac_seed,
        ..GeneralizedPnPRansac::default()
    }
    .estimate(&rig, &correspondences)
    .unwrap();
    assert!(bridged.inliers.len() >= world_points.len());
    assert!((bridged.pose.camera_center_world() - truth[1].camera_center_world()).norm() < 1.0e-4);

    let mut motion_frame_poses = vec![None; frames.len()];
    motion_frame_poses[0] = Some(truth[0].clone());
    motion_frame_poses[1] = Some(truth[1].clone());
    let motion_config = RigSfmConfig {
        motion_bridge_max_frame_gap: 1,
        motion_bridge_min_inliers: 12,
        motion_bridge_max_rotation_deviation_deg: 5.0,
        ..RigSfmConfig::default()
    };
    let mut motion_queue = vec![1];
    let mut motion_visited = vec![false; pairwise.len()];
    let motion = collect_motion_bridge_candidates(
        &rig,
        &frames,
        &features,
        &pairwise,
        &assignment,
        &motion_frame_poses,
        &motion_config,
        &adjacency,
        &mut motion_queue,
        &mut motion_visited,
    );
    let predicted = motion
        .candidates
        .iter()
        .find(|candidate| candidate.frame == 2)
        .unwrap();
    assert!(predicted.inliers >= world_points.len());
    assert!((predicted.pose.camera_center_world() - truth[2].camera_center_world()).norm() < 0.01);
    assert!(
        predicted
            .pose
            .world_to_camera
            .rotation
            .angle_to(&truth[2].world_to_camera.rotation)
            .to_degrees()
            < 0.1
    );
}

type DynamicTwoImageFixture = (
    GeneralizedCameraRig,
    Vec<RigFrame>,
    Vec<FeatureSet>,
    Vec<(usize, usize)>,
    Vec<Option<Pose>>,
    RigSfmConfig,
);

fn dynamic_two_image_fixture() -> DynamicTwoImageFixture {
    let camera = Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0);
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: camera.clone(),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera,
            sensor_from_rig: SE3::identity(),
        },
    ])
    .unwrap();
    let frames = vec![RigFrame {
        images: vec![
            RigFrameImage {
                image_index: 0,
                sensor_index: 0,
            },
            RigFrameImage {
                image_index: 1,
                sensor_index: 1,
            },
        ],
    }];
    let features = vec![
        FeatureSet::new(
            vec![Point2::new(320.0, 240.0), Point2::new(350.0, 240.0)],
            vec![Vec::new(), Vec::new()],
        )
        .unwrap(),
        FeatureSet::new(
            vec![Point2::new(320.0, 240.0), Point2::new(350.0, 240.0)],
            vec![Vec::new(), Vec::new()],
        )
        .unwrap(),
    ];
    let assignment = image_assignment(&frames, features.len());
    let image_poses = vec![Some(Pose::identity()); features.len()];
    let config = RigSfmConfig {
        dynamic_correspondence_tracking: true,
        final_bundle_adjustment: false,
        local_ba_every: 0,
        structure_refinement_iterations: 0,
        ..RigSfmConfig::default()
    };
    (rig, frames, features, assignment, image_poses, config)
}

#[test]
fn dynamic_activation_is_idempotent_for_a_registered_frame() {
    let (rig, frames, features, assignment, image_poses, config) = dynamic_two_image_fixture();
    let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0)])];
    let csr = build_rig_correspondence_csr_from_features(&features, &pairwise).unwrap();
    let observations = dynamic_observation_lookup(&csr, &features).unwrap();
    let mut owner = vec![DYNAMIC_EMPTY_SLOT; csr.total_observations()];
    let mut image_tracks = vec![Vec::new(); features.len()];
    let mut tracks = Vec::new();
    let mut activated_rows = vec![false; csr.total_observations()];
    let mut first = DynamicActivationUpdate::default();
    dynamic_activate_frame(
        0,
        &frames,
        &csr,
        &observations,
        &assignment,
        &[Some(Pose::identity())],
        &image_poses,
        &rig,
        &features,
        &config,
        &mut owner,
        &mut image_tracks,
        &mut tracks,
        &mut activated_rows,
        &mut first,
    )
    .unwrap();
    assert_eq!(first.creates, 1);
    assert_eq!(first.continues, 0);
    assert_eq!(first.activated_edges, csr.directed_unique_edges());
    let owner_after_first = owner.clone();
    let tracks_after_first = tracks.clone();
    let mut second = DynamicActivationUpdate::default();
    dynamic_activate_frame(
        0,
        &frames,
        &csr,
        &observations,
        &assignment,
        &[Some(Pose::identity())],
        &image_poses,
        &rig,
        &features,
        &config,
        &mut owner,
        &mut image_tracks,
        &mut tracks,
        &mut activated_rows,
        &mut second,
    )
    .unwrap();
    assert_eq!(second.activated_rows, 0);
    assert_eq!(second.activated_edges, 0);
    assert_eq!(second.creates, 0);
    assert_eq!(second.continues, 0);
    assert_eq!(owner, owner_after_first);
    assert_eq!(tracks, tracks_after_first);
}

#[test]
fn dynamic_activation_rejects_owner_and_same_image_conflicts() {
    let (rig, frames, features, assignment, image_poses, config) = dynamic_two_image_fixture();
    let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0)])];
    let csr = build_rig_correspondence_csr_from_features(&features, &pairwise).unwrap();
    let observations = dynamic_observation_lookup(&csr, &features).unwrap();
    let mut activated_rows = vec![false; csr.total_observations()];

    let mut owner = vec![DYNAMIC_EMPTY_SLOT; csr.total_observations()];
    owner[0] = dynamic_track_id(0).unwrap();
    owner[2] = dynamic_track_id(1).unwrap();
    let mut image_tracks = vec![vec![(0, 0)], vec![(0, 1)]];
    let mut tracks = vec![
        WorkingTrack {
            observations: vec![(0, 0)],
            position: None,
            metric_anchored: false,
        },
        WorkingTrack {
            observations: vec![(1, 0)],
            position: None,
            metric_anchored: false,
        },
    ];
    let mut conflicts = DynamicActivationUpdate::default();
    dynamic_activate_frame(
        0,
        &frames,
        &csr,
        &observations,
        &assignment,
        &[Some(Pose::identity())],
        &image_poses,
        &rig,
        &features,
        &config,
        &mut owner,
        &mut image_tracks,
        &mut tracks,
        &mut activated_rows,
        &mut conflicts,
    )
    .unwrap();
    assert_eq!(conflicts.owner_conflicts, 2);
    assert_eq!(conflicts.continues, 0);
    assert_eq!(tracks.len(), 2);
    assert_eq!(tracks[0].observations, vec![(0, 0)]);
    assert_eq!(tracks[1].observations, vec![(1, 0)]);

    let pairwise = vec![PairwiseMatches::new(1, 0, vec![(0, 1)])];
    let csr = build_rig_correspondence_csr_from_features(&features, &pairwise).unwrap();
    let observations = dynamic_observation_lookup(&csr, &features).unwrap();
    let mut owner = vec![DYNAMIC_EMPTY_SLOT; csr.total_observations()];
    owner[0] = dynamic_track_id(0).unwrap();
    owner[2] = dynamic_track_id(0).unwrap();
    let mut image_tracks = vec![vec![(0, 0)], vec![(0, 0)]];
    let mut tracks = vec![WorkingTrack {
        observations: vec![(0, 0), (1, 0)],
        position: None,
        metric_anchored: false,
    }];
    let mut activated_rows = vec![false; csr.total_observations()];
    let mut conflicts = DynamicActivationUpdate::default();
    dynamic_activate_frame(
        0,
        &frames,
        &csr,
        &observations,
        &assignment,
        &[Some(Pose::identity())],
        &image_poses,
        &rig,
        &features,
        &config,
        &mut owner,
        &mut image_tracks,
        &mut tracks,
        &mut activated_rows,
        &mut conflicts,
    )
    .unwrap();
    assert!(conflicts.same_image_conflicts > 0);
    assert_eq!(conflicts.continues, 0);
    assert_eq!(owner[1], DYNAMIC_EMPTY_SLOT);
}

#[test]
fn dynamic_activation_rejects_positioned_geometry_outliers() {
    let (rig, frames, mut features, assignment, image_poses, config) = dynamic_two_image_fixture();
    features[1].keypoints[0] = Point2::new(0.0, 0.0);
    let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0)])];
    let csr = build_rig_correspondence_csr_from_features(&features, &pairwise).unwrap();
    let observations = dynamic_observation_lookup(&csr, &features).unwrap();
    let mut owner = vec![DYNAMIC_EMPTY_SLOT; csr.total_observations()];
    owner[0] = dynamic_track_id(0).unwrap();
    let mut image_tracks = vec![vec![(0, 0)], Vec::new()];
    let mut tracks = vec![WorkingTrack {
        observations: vec![(0, 0)],
        position: Some(Point3::new(0.0, 0.0, 5.0)),
        metric_anchored: true,
    }];
    let mut activated_rows = vec![false; csr.total_observations()];
    let mut update = DynamicActivationUpdate::default();
    dynamic_activate_frame(
        0,
        &frames,
        &csr,
        &observations,
        &assignment,
        &[Some(Pose::identity())],
        &image_poses,
        &rig,
        &features,
        &config,
        &mut owner,
        &mut image_tracks,
        &mut tracks,
        &mut activated_rows,
        &mut update,
    )
    .unwrap();
    assert!(update.geometry_rejections > 0);
    assert_eq!(update.continues, 0);
    assert_eq!(owner[2], DYNAMIC_EMPTY_SLOT);
}

#[test]
fn dynamic_seed_creates_metric_two_observation_tracks() {
    let camera = Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0);
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: camera.clone(),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera,
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.2, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let point = Point3::new(0.1, 0.0, 4.0);
    let features = vec![
        FeatureSet::new(
            vec![rig.sensors()[0].camera.project(&point).unwrap()],
            vec![Vec::new()],
        )
        .unwrap(),
        FeatureSet::new(
            vec![rig.sensors()[1]
                .camera
                .project(&rig.sensors()[1].sensor_from_rig.transform_point(&point))
                .unwrap()],
            vec![Vec::new()],
        )
        .unwrap(),
    ];
    let frames = vec![RigFrame {
        images: vec![
            RigFrameImage {
                image_index: 0,
                sensor_index: 0,
            },
            RigFrameImage {
                image_index: 1,
                sensor_index: 1,
            },
        ],
    }];
    let assignment = image_assignment(&frames, features.len());
    let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0)])];
    let csr = build_rig_correspondence_csr_from_features(&features, &pairwise).unwrap();
    let observations = dynamic_observation_lookup(&csr, &features).unwrap();
    let mut image_poses = vec![None; features.len()];
    install_image_poses(&rig, &frames[0], &Pose::identity(), &mut image_poses);
    let mut owner = vec![DYNAMIC_EMPTY_SLOT; csr.total_observations()];
    let mut image_tracks = vec![Vec::new(); features.len()];
    let mut tracks = Vec::new();
    let update = dynamic_create_seed_tracks(
        &rig,
        &features,
        &observations,
        &assignment,
        &image_poses,
        &RigSfmConfig {
            final_bundle_adjustment: false,
            ..RigSfmConfig::default()
        },
        0,
        &[(0, 1)],
        &mut owner,
        &mut image_tracks,
        &mut tracks,
    )
    .unwrap();
    assert_eq!(tracks.len(), 1);
    assert_eq!(tracks[0].observations.len(), 2);
    assert!(tracks[0].metric_anchored);
    assert!(tracks[0].position.is_some());
    assert_eq!(update.landmarks, [0]);
    assert_eq!(owner, [0, 0]);
}

#[test]
fn dynamic_bootstrap_uses_transitive_legacy_membership_before_direct_fallback() {
    let camera = Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0);
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: camera.clone(),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera,
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.2, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let points = (0..6)
        .map(|index| {
            Point3::new(
                (index % 3) as f64 * 0.2 - 0.2,
                (index / 3) as f64 * 0.2 - 0.1,
                4.0,
            )
        })
        .collect::<Vec<_>>();
    let truth = [
        Pose::identity(),
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.35, 0.0, 0.0)),
    ];
    let frames = vec![
        RigFrame {
            images: vec![
                RigFrameImage {
                    image_index: 0,
                    sensor_index: 0,
                },
                RigFrameImage {
                    image_index: 1,
                    sensor_index: 1,
                },
            ],
        },
        RigFrame {
            images: vec![RigFrameImage {
                image_index: 2,
                sensor_index: 0,
            }],
        },
    ];
    let mut features = Vec::new();
    for (image, (frame, sensor)) in [(0, 0), (0, 1), (1, 0)].into_iter().enumerate() {
        let image_pose = rig.sensors()[sensor]
            .sensor_from_rig
            .compose(&truth[frame].world_to_camera);
        let keypoints = points
            .iter()
            .map(|point| {
                rig.sensors()[sensor]
                    .camera
                    .project(&image_pose.transform_point(point))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        features.push(FeatureSet::new(keypoints, vec![vec![0.0]; points.len()]).unwrap());
        assert_eq!(image, features.len() - 1);
    }
    let identity = (0..points.len())
        .map(|index| (index, index))
        .collect::<Vec<_>>();
    // There is no direct frame-0 stereo pair.  The legacy component still
    // contains both frame-0 sensors through image 2, so membership is the
    // only source of this metric seed.
    let pairwise = vec![
        PairwiseMatches::new(0, 2, identity.clone()),
        PairwiseMatches::new(2, 1, identity),
    ];
    let assignment = image_assignment(&frames, features.len());
    let csr = build_rig_correspondence_csr_from_features(&features, &pairwise).unwrap();
    let observations = dynamic_observation_lookup(&csr, &features).unwrap();
    let legacy =
        build_rig_track_output(&features, &pairwise, &assignment, &RigSfmConfig::default());
    let membership_edges =
        dynamic_membership_seed_edges(&legacy.tracks, &csr, &assignment, frames.len());
    let direct_edges = dynamic_direct_seed_edges(&csr, &observations, &assignment, frames.len());
    assert!(direct_edges[0].is_empty());
    assert_eq!(membership_edges[0].len(), points.len());
    let candidates = dynamic_seed_candidates(membership_edges, false);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].frame, 0);
    assert_eq!(candidates[0].support, points.len());
    assert!(candidates[0].pairs.iter().all(|&(left, right)| {
        let left = observations[usize::try_from(left).unwrap()].image();
        let right = observations[usize::try_from(right).unwrap()].image();
        [left, right] == [0, 1]
    }));

    let config = RigSfmConfig {
        dynamic_correspondence_tracking: true,
        min_pnp_sensors: 1,
        min_pnp_inliers: points.len(),
        final_bundle_adjustment: false,
        local_ba_every: 0,
        structure_refinement_iterations: 0,
        ..RigSfmConfig::default()
    };
    let result = incremental_rig_sfm(&rig, &frames, &features, &pairwise, &config).unwrap();
    assert_eq!(result.seed_frame_index, 0);
    assert_eq!(result.registered_frames, 2);
    assert_eq!(result.tracks.len(), points.len());
    assert!(result
        .tracks
        .iter()
        .all(|track| track.observations.len() == 3));
    assert_eq!(result.work.dynamic_bootstrap_legacy_tracks, points.len());
    assert_eq!(result.work.dynamic_bootstrap_candidates, 1);
    assert_eq!(result.work.dynamic_bootstrap_seed_support, points.len());
    assert_eq!(result.work.dynamic_bootstrap_seed_pairs, points.len());
    assert_eq!(result.work.dynamic_bootstrap_seed_landmarks, points.len());
    assert_eq!(result.work.dynamic_bootstrap_direct_fallbacks, 0);

    let permuted_pairwise = pairwise
        .iter()
        .rev()
        .map(|pair| {
            PairwiseMatches::new(
                pair.image_j,
                pair.image_i,
                pair.matches
                    .iter()
                    .map(|&(left, right)| (right, left))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let permuted =
        incremental_rig_sfm(&rig, &frames, &features, &permuted_pairwise, &config).unwrap();
    assert_eq!(result, permuted);
}

#[test]
fn dynamic_correspondence_registration_creates_and_continues_metric_tracks() {
    let camera = Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0);
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: camera.clone(),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera,
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.2, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let points = (0..8)
        .map(|index| Point3::new(index as f64 * 0.15 - 0.5, 0.1 * (index % 3) as f64, 4.0))
        .collect::<Vec<_>>();
    let frame_truth = [
        Pose::identity(),
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.4, 0.0, 0.0)),
    ];
    let mut frames = Vec::new();
    let mut features = Vec::new();
    for (frame, frame_pose) in frame_truth.iter().enumerate() {
        let mut images = Vec::new();
        for sensor in 0..2 {
            let image = features.len();
            let pose = rig.sensors()[sensor]
                .sensor_from_rig
                .compose(&frame_pose.world_to_camera);
            let keypoints = points
                .iter()
                .map(|point| {
                    rig.sensors()[sensor]
                        .camera
                        .project(&pose.transform_point(point))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            features.push(FeatureSet::new(keypoints, vec![vec![0.0]; points.len()]).unwrap());
            images.push(RigFrameImage {
                image_index: image,
                sensor_index: sensor,
            });
        }
        assert_eq!(images[0].image_index, frame * 2);
        frames.push(RigFrame { images });
    }
    let identity = (0..points.len())
        .map(|index| (index, index))
        .collect::<Vec<_>>();
    let pairwise = vec![
        PairwiseMatches::new(0, 1, identity.clone()),
        PairwiseMatches::new(2, 3, identity.clone()),
        PairwiseMatches::new(0, 2, identity.clone()),
        PairwiseMatches::new(1, 3, identity),
    ];
    let config = RigSfmConfig {
        dynamic_correspondence_tracking: true,
        final_bundle_adjustment: false,
        local_ba_every: 0,
        structure_refinement_iterations: 0,
        ..RigSfmConfig::default()
    };
    let result = incremental_rig_sfm(&rig, &frames, &features, &pairwise, &config).unwrap();
    assert_eq!(result.registered_frames, 2);
    assert_eq!(result.tracks.len(), points.len());
    assert!(result
        .tracks
        .iter()
        .all(|track| track.observations.len() == 4));
    assert_eq!(result.work.dynamic_track_creates, points.len());
    assert_eq!(result.work.dynamic_track_continues, points.len() * 2);
    assert!(result.work.dynamic_activated_rows > 0);
    assert!(result.work.dynamic_activated_edges > 0);
    assert!(result.work.dynamic_pnp_graph_insertions >= points.len() * 2);
    let csr = build_rig_correspondence_csr_from_features(&features, &pairwise).unwrap();
    assert_eq!(
        result.work.dynamic_observation_lookup_entries,
        csr.total_observations()
    );
    assert!(result.work.dynamic_activated_rows <= csr.total_observations());
    assert!(result.work.dynamic_activated_edges <= csr.directed_unique_edges());
    assert!(result.mean_reprojection_error_px < 1.0e-6);

    // CSR canonicalization makes the registration schedule independent of
    // pair stream order and endpoint orientation.
    let permuted_pairwise = pairwise
        .iter()
        .rev()
        .map(|pair| {
            PairwiseMatches::new(
                pair.image_j,
                pair.image_i,
                pair.matches
                    .iter()
                    .map(|&(left, right)| (right, left))
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let permuted =
        incremental_rig_sfm(&rig, &frames, &features, &permuted_pairwise, &config).unwrap();
    assert_eq!(result, permuted);

    // Explicitly spelling out the default keeps the legacy branch's
    // result byte-for-byte identical to the default configuration.
    let legacy_default = incremental_rig_sfm(
        &rig,
        &frames,
        &features,
        &pairwise,
        &RigSfmConfig::default(),
    )
    .unwrap();
    let legacy_explicit = incremental_rig_sfm(
        &rig,
        &frames,
        &features,
        &pairwise,
        &RigSfmConfig {
            dynamic_correspondence_tracking: false,
            ..RigSfmConfig::default()
        },
    )
    .unwrap();
    assert_eq!(legacy_default, legacy_explicit);
}

#[test]
fn fixed_active_rotations_refine_rig_translation_and_landmarks() {
    check_fixed_rotation_backend(RigBaBackend::MatrixFreeStrict);
}

#[test]
fn cluster8_native_preserves_fixed_rotations_observations_and_rollback() {
    check_fixed_rotation_backend(RigBaBackend::MatrixFreeCluster8);
}

#[test]
fn cluster8_restart_native_preserves_fixed_state_and_rollback() {
    check_fixed_rotation_backend(RigBaBackend::MatrixFreeCluster8Restart1);
}

#[test]
fn qr_native_preserves_fixed_state_and_rollback() {
    check_fixed_rotation_backend(RigBaBackend::MatrixFreeQr);
}

#[test]
fn ba_anchor_connectivity_uses_landmark_paths_not_stereo_support_alone() {
    let observations = [
        (0, 10),
        (1, 10),
        (1, 11),
        (2, 11),
        (3, 12),
        (3, 12),
        (4, 13),
        (5, 13),
    ];
    assert_eq!(
        rig_ba_anchor_reachable(observations, [0]),
        BTreeSet::from([0, 1, 2])
    );
    assert_eq!(
        rig_ba_anchor_reachable(observations, [0, 4]),
        BTreeSet::from([0, 1, 2, 4, 5])
    );
    assert!(rig_ba_anchor_reachable(observations, []).is_empty());
    assert_eq!(rig_ba_anchor_reachable([], [7]), BTreeSet::from([7]));
}

#[test]
fn component_anchors_preserve_fixed_components_and_are_order_independent() {
    let observations = [(0, 10), (1, 10), (2, 11), (3, 11), (4, 12), (4, 12)];
    let fixed = BTreeSet::from([1]);
    assert_eq!(
        rig_ba_component_anchors(observations, 0..6, &fixed),
        vec![2, 4, 5]
    );
    assert_eq!(
        rig_ba_component_anchors(observations.into_iter().rev(), (0..6).rev(), &fixed),
        vec![2, 4, 5]
    );
    let all_fixed = BTreeSet::from([1, 3, 4, 5]);
    assert!(rig_ba_component_anchors(observations, 0..6, &all_fixed).is_empty());
    assert!(!RigSfmConfig::default().ba_anchor_disconnected_components);
}

#[test]
fn bounded_policy_selects_before_solve_and_preserves_fixed_state() {
    for n in [0, 1, 63, 64] {
        assert_eq!(
            select_bounded_rig_backend(RigBaBackend::BoundedDirect64Qr, n),
            RigBaBackend::Legacy
        );
    }
    for n in [65, 10000, usize::MAX] {
        assert_eq!(
            select_bounded_rig_backend(RigBaBackend::BoundedDirect64Qr, n),
            RigBaBackend::MatrixFreeQr
        );
    }
    assert_eq!(
        select_bounded_rig_backend(RigBaBackend::Legacy, 10000),
        RigBaBackend::Legacy
    );
    check_fixed_rotation_backend(RigBaBackend::BoundedDirect64Qr);
}

fn check_fixed_rotation_backend(backend: RigBaBackend) {
    check_fixed_rotation_backend_with_component_anchors(backend, false, false);
}

#[test]
fn component_anchoring_native_preserves_fixed_state_and_rollback() {
    check_fixed_rotation_backend_with_component_anchors(
        RigBaBackend::BoundedDirect64Qr,
        true,
        false,
    );
}

#[test]
fn fixed_boundary_keeps_single_active_observations_and_external_poses_fixed() {
    check_fixed_rotation_backend_with_component_anchors(RigBaBackend::Legacy, false, true);
}

fn check_fixed_rotation_backend_with_component_anchors(
    backend: RigBaBackend,
    component_anchors: bool,
    boundary_test: bool,
) {
    check_fixed_rotation_backend_with_retry_reuse(backend, component_anchors, boundary_test, false);
}

#[test]
fn retry_reuse_native_preserves_fixed_state() {
    check_fixed_rotation_backend_with_retry_reuse(RigBaBackend::Legacy, false, false, true);
    check_fixed_rotation_backend_with_retry_reuse(RigBaBackend::Legacy, false, true, true);
}

fn check_fixed_rotation_backend_with_retry_reuse(
    backend: RigBaBackend,
    component_anchors: bool,
    boundary_test: bool,
    retry_reuse: bool,
) {
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera: Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5),
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.20, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let world_points = (0..24)
        .map(|index| {
            Point3::new(
                (index % 6) as f64 * 0.28 - 0.7,
                (index / 6) as f64 * 0.24 - 0.35,
                4.0 + (index % 5) as f64 * 0.17,
            )
        })
        .collect::<Vec<_>>();
    let truth = [
        Pose::identity(),
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.02, -0.03, 0.04),
            Vector3::new(-0.45, 0.03, 0.02),
        ),
    ];
    let frames = (0..2)
        .map(|frame| RigFrame {
            images: (0..2)
                .map(|sensor| RigFrameImage {
                    image_index: frame * 2 + sensor,
                    sensor_index: sensor,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let mut features = Vec::new();
    for (frame, frame_pose) in truth.iter().enumerate() {
        for sensor in 0..2 {
            let image_pose = rig.sensors()[sensor]
                .sensor_from_rig
                .compose(&frame_pose.world_to_camera);
            let keypoints = world_points
                .iter()
                .map(|point| {
                    rig.sensors()[sensor]
                        .camera
                        .project(&image_pose.transform_point(point))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            features.push(FeatureSet::new(keypoints, vec![vec![0.0]; world_points.len()]).unwrap());
            assert_eq!(features.len(), frame * 2 + sensor + 1);
        }
    }
    let image_assignment = image_assignment(&frames, features.len());
    let mut frame_poses = vec![
        Some(truth[0].clone()),
        Some(Pose::from_world_to_camera(
            truth[1].world_to_camera.rotation,
            truth[1].world_to_camera.translation + Vector3::new(0.04, -0.025, 0.02),
        )),
    ];
    let rotations_before = frame_poses
        .iter()
        .map(|pose| pose.as_ref().unwrap().world_to_camera.rotation)
        .collect::<Vec<_>>();
    let mut image_poses = vec![None; features.len()];
    for (frame, pose) in frame_poses.iter().enumerate() {
        install_image_poses(
            &rig,
            &frames[frame],
            pose.as_ref().unwrap(),
            &mut image_poses,
        );
    }
    let mut tracks = world_points
        .iter()
        .enumerate()
        .map(|(track, point)| WorkingTrack {
            observations: (0..features.len()).map(|image| (image, track)).collect(),
            position: Some(*point + Vector3::new(0.015, -0.01, 0.025)),
            metric_anchored: true,
        })
        .collect::<Vec<_>>();
    let public_tracks = |tracks: &[WorkingTrack]| {
        tracks
            .iter()
            .map(|track| SfmTrack {
                position: track.position.unwrap(),
                observations: track
                    .observations
                    .iter()
                    .map(|&(image, keypoint)| {
                        (image, keypoint, features[image].keypoints[keypoint])
                    })
                    .collect(),
            })
            .collect::<Vec<_>>()
    };
    let before_tracks = public_tracks(&tracks);
    let (before_sum, before_count) =
        reprojection_error(&rig, &image_assignment, &image_poses, &before_tracks);
    let before_reprojection = before_sum / before_count as f64;
    let before_center_error = (frame_poses[1].as_ref().unwrap().camera_center_world()
        - truth[1].camera_center_world())
    .norm();
    let before_landmark_error = tracks
        .iter()
        .zip(&world_points)
        .map(|(track, point)| (track.position.unwrap() - point).norm())
        .sum::<f64>()
        / world_points.len() as f64;
    let config = RigSfmConfig {
        ba_anchor_disconnected_components: component_anchors,
        final_bundle_adjustment: false,
        structure_refinement_iterations: 0,
        ..RigSfmConfig::default()
    };
    let selection_config = RigSfmConfig {
        local_ba_covisibility: true,
        ..config
    };
    let index = image_track_index(features.len(), &tracks);
    let select =
        |order: &[usize], poses: &[Option<Pose>], tracks: &[WorkingTrack], cfg: &RigSfmConfig| {
            select_local_ba_frames(
                &rig,
                &frames,
                &features,
                &image_assignment,
                &index,
                poses,
                tracks,
                order,
                cfg,
            )
        };
    let (active_frames, anchor) = select(&[0, 1], &image_poses, &tracks, &selection_config);
    assert_eq!(active_frames, HashSet::from([0, 1]));
    assert_eq!(anchor, 0);
    assert_eq!(
        select(&[1, 0], &image_poses, &tracks, &selection_config).1,
        1
    );
    let mut unsupported = image_poses.clone();
    unsupported[0] = None;
    unsupported[1] = None;
    assert!(select(&[1], &unsupported, &tracks, &selection_config)
        .0
        .is_empty());
    let mut nonmetric = tracks.clone();
    for track in &mut nonmetric {
        track.metric_anchored = false;
    }
    let metric_config = RigSfmConfig {
        ba_metric_tracks_only: true,
        ..selection_config
    };
    assert!(select(&[0, 1], &image_poses, &nonmetric, &metric_config)
        .0
        .is_empty());
    assert_eq!(
        select(&[0, 1], &image_poses, &tracks, &config).0,
        active_frames
    );
    let ba_config = BaConfig {
        max_iterations: 40,
        reuse_rejected_pose_diagonal: retry_reuse,
        linear_solver: LinearSolver::Sparse,
        robust_kernel: RobustKernel::None,
        parallel: false,
        ..BaConfig::default()
    };
    if boundary_test {
        let mut outside_only = tracks[0].clone();
        outside_only.observations.retain(|&(image, _)| image < 2);
        let outside_position = outside_only.position;
        // One sensor observation per frame: without the boundary each
        // track has only one active row and must be excluded.
        for track in &mut tracks {
            track.observations.retain(|&(image, _)| image % 2 == 0);
        }
        tracks.push(outside_only);
        let before_poses = frame_poses.clone();
        let before_images = image_poses.clone();
        let before_observations: Vec<_> = tracks.iter().map(|t| t.observations.clone()).collect();
        let active = HashSet::from([1]);
        assert!(run_rig_bundle_adjustment(
            &rig,
            &features,
            &image_assignment,
            &config,
            &active,
            1,
            &ba_config,
            0,
            &[],
            false,
            &mut frame_poses,
            &mut image_poses,
            &mut tracks,
        )
        .unwrap()
        .is_none());
        let boundary_config = RigSfmConfig {
            ba_fixed_boundary_observations: true,
            ..config
        };
        let stats = run_rig_bundle_adjustment(
            &rig,
            &features,
            &image_assignment,
            &boundary_config,
            &active,
            1,
            &ba_config,
            0,
            &[],
            false,
            &mut frame_poses,
            &mut image_poses,
            &mut tracks,
        )
        .unwrap()
        .unwrap();
        assert_eq!(stats.observations, world_points.len() * 2);
        assert_eq!(tracks.last().unwrap().position, outside_position);
        assert!(stats.final_cost <= stats.initial_cost);
        assert_eq!(frame_poses, before_poses);
        assert_eq!(image_poses, before_images);
        assert_eq!(
            tracks
                .iter()
                .map(|t| t.observations.clone())
                .collect::<Vec<_>>(),
            before_observations
        );
        return;
    }
    let direct_frame_poses_before = frame_poses.clone();
    let direct_observations_before = tracks
        .iter()
        .map(|track| track.observations.clone())
        .collect::<Vec<_>>();
    let direct_track_positions_before = tracks
        .iter()
        .map(|track| track.position)
        .collect::<Vec<_>>();
    let mut matrix_free_frame_poses = frame_poses.clone();
    let mut matrix_free_image_poses = image_poses.clone();
    let mut matrix_free_tracks = tracks.clone();
    let matrix_free_image_poses_before = matrix_free_image_poses.clone();
    let matrix_free_observations_before = matrix_free_tracks
        .iter()
        .map(|track| track.observations.clone())
        .collect::<Vec<_>>();
    let matrix_free_track_positions_before = matrix_free_tracks
        .iter()
        .map(|track| track.position)
        .collect::<Vec<_>>();
    let matrix_free_stats = run_rig_bundle_adjustment(
        &rig,
        &features,
        &image_assignment,
        &RigSfmConfig {
            ba_backend: backend,
            ..config
        },
        &active_frames,
        0,
        &ba_config,
        0,
        &[],
        true,
        &mut matrix_free_frame_poses,
        &mut matrix_free_image_poses,
        &mut matrix_free_tracks,
    )
    .unwrap()
    .expect("matrix-free native rig fixture should have observations");
    assert!(matrix_free_stats.final_cost <= matrix_free_stats.initial_cost);
    assert!(matrix_free_frame_poses
        .iter()
        .zip(&direct_frame_poses_before)
        .any(|(after, before)| after != before));
    assert!(matrix_free_tracks
        .iter()
        .map(|track| track.position)
        .zip(&matrix_free_track_positions_before)
        .any(|(after, before)| after != *before));
    assert_eq!(
        matrix_free_tracks
            .iter()
            .map(|track| track.observations.clone())
            .collect::<Vec<_>>(),
        matrix_free_observations_before
    );
    assert!(matrix_free_image_poses
        .iter()
        .zip(&matrix_free_image_poses_before)
        .any(|(after, before)| after != before));
    let matrix_free_after_tracks = public_tracks(&matrix_free_tracks);
    let (matrix_free_after_sum, matrix_free_after_count) = reprojection_error(
        &rig,
        &image_assignment,
        &matrix_free_image_poses,
        &matrix_free_after_tracks,
    );
    let matrix_free_after_reprojection = matrix_free_after_sum / matrix_free_after_count as f64;
    let matrix_free_after_center_error = (matrix_free_frame_poses[1]
        .as_ref()
        .unwrap()
        .camera_center_world()
        - truth[1].camera_center_world())
    .norm();
    let matrix_free_after_landmark_error = matrix_free_tracks
        .iter()
        .zip(&world_points)
        .map(|(track, point)| (track.position.unwrap() - point).norm())
        .sum::<f64>()
        / world_points.len() as f64;
    assert!(matrix_free_after_reprojection < before_reprojection);
    assert!(matrix_free_after_center_error < before_center_error);
    assert!(matrix_free_after_landmark_error < before_landmark_error);
    for (before, after) in rotations_before.iter().zip(&matrix_free_frame_poses) {
        assert!(before.angle_to(&after.as_ref().unwrap().world_to_camera.rotation) < 1.0e-10);
    }
    let stats = run_rig_bundle_adjustment(
        &rig,
        &features,
        &image_assignment,
        &config,
        &active_frames,
        0,
        &ba_config,
        0,
        &[],
        true,
        &mut frame_poses,
        &mut image_poses,
        &mut tracks,
    )
    .unwrap()
    .expect("direct native rig fixture should have observations");
    assert_eq!(matrix_free_stats.observations, stats.observations);
    assert_eq!(matrix_free_stats.initial_cost, stats.initial_cost);
    for (before, after) in rotations_before.iter().zip(&frame_poses) {
        assert!(before.angle_to(&after.as_ref().unwrap().world_to_camera.rotation) < 1.0e-10);
    }
    let after_tracks = public_tracks(&tracks);
    let (after_sum, after_count) =
        reprojection_error(&rig, &image_assignment, &image_poses, &after_tracks);
    let after_reprojection = after_sum / after_count as f64;
    let after_center_error = (frame_poses[1].as_ref().unwrap().camera_center_world()
        - truth[1].camera_center_world())
    .norm();
    let after_landmark_error = tracks
        .iter()
        .zip(&world_points)
        .map(|(track, point)| (track.position.unwrap() - point).norm())
        .sum::<f64>()
        / world_points.len() as f64;
    assert!(after_reprojection < before_reprojection);
    assert!(after_center_error < before_center_error);
    assert!(after_landmark_error < before_landmark_error);
    assert!(frame_poses
        .iter()
        .zip(&direct_frame_poses_before)
        .any(|(after, before)| after != before));
    assert!(tracks
        .iter()
        .map(|track| track.position)
        .zip(&direct_track_positions_before)
        .any(|(after, before)| after != *before));
    assert_eq!(
        tracks
            .iter()
            .map(|track| track.observations.clone())
            .collect::<Vec<_>>(),
        direct_observations_before
    );

    let landmark_only_frame_poses = matrix_free_frame_poses.clone();
    let mut landmark_only_image_poses = matrix_free_image_poses.clone();
    let mut landmark_only_tracks = matrix_free_tracks.clone();
    let landmark_only_stats = run_rig_bundle_adjustment(
        &rig,
        &features,
        &image_assignment,
        &RigSfmConfig {
            ba_backend: backend,
            ..config
        },
        &active_frames,
        0,
        &ba_config,
        0,
        &[1],
        false,
        &mut matrix_free_frame_poses,
        &mut landmark_only_image_poses,
        &mut landmark_only_tracks,
    )
    .unwrap()
    .expect("landmark-only native rig fixture should have observations");
    assert!(landmark_only_stats.final_cost <= landmark_only_stats.initial_cost);
    assert_eq!(matrix_free_frame_poses, landmark_only_frame_poses);

    // Calibration refinement is supported by Legacy; the unsupported-
    // configuration rollback check below is specific to matrix-free.
    if backend == RigBaBackend::Legacy {
        return;
    }

    let rejected_frame_poses = matrix_free_frame_poses.clone();
    let mut rejected_image_poses = landmark_only_image_poses.clone();
    let mut rejected_tracks = landmark_only_tracks.clone();
    let rejected_image_poses_before = rejected_image_poses.clone();
    let rejected_tracks_before = rejected_tracks.clone();
    let error = run_rig_bundle_adjustment(
        &rig,
        &features,
        &image_assignment,
        &RigSfmConfig {
            ba_backend: backend,
            ..config
        },
        &active_frames,
        0,
        &BaConfig {
            refine_intrinsics: true,
            ..ba_config
        },
        0,
        &[1],
        false,
        &mut matrix_free_frame_poses,
        &mut rejected_image_poses,
        &mut rejected_tracks,
    )
    .unwrap_err();
    assert!(
        matches!(error, RigSfmError::BundleAdjustment(message) if message.contains("matrix-free"))
    );
    assert_eq!(matrix_free_frame_poses, rejected_frame_poses);
    assert_eq!(rejected_image_poses, rejected_image_poses_before);
    assert_eq!(rejected_tracks, rejected_tracks_before);
}

#[test]
fn public_fixed_frame_rotation_refinement_updates_result_transactionally() {
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera: Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5),
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.20, 0.0, 0.0)),
        },
    ])
    .unwrap();
    let world_points = (0..24)
        .map(|index| {
            Point3::new(
                (index % 6) as f64 * 0.28 - 0.7,
                (index / 6) as f64 * 0.24 - 0.35,
                4.0 + (index % 5) as f64 * 0.17,
            )
        })
        .collect::<Vec<_>>();
    let truth = [
        Pose::identity(),
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.02, -0.03, 0.04),
            Vector3::new(-0.45, 0.03, 0.02),
        ),
    ];
    let frames = (0..2)
        .map(|frame| RigFrame {
            images: (0..2)
                .map(|sensor| RigFrameImage {
                    image_index: frame * 2 + sensor,
                    sensor_index: sensor,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let mut features = Vec::new();
    for frame_pose in &truth {
        for sensor in 0..2 {
            let image_pose = rig.sensors()[sensor]
                .sensor_from_rig
                .compose(&frame_pose.world_to_camera);
            let keypoints = world_points
                .iter()
                .map(|point| {
                    rig.sensors()[sensor]
                        .camera
                        .project(&image_pose.transform_point(point))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            features.push(FeatureSet::new(keypoints, vec![vec![0.0]; world_points.len()]).unwrap());
        }
    }
    let image_assignment = image_assignment(&frames, features.len());
    let mut frame_poses = vec![Some(truth[0].clone()), Some(truth[1].clone())];
    frame_poses[1] = Some(Pose::from_world_to_camera(
        UnitQuaternion::from_euler_angles(0.02, -0.03, 0.04),
        truth[1].world_to_camera.translation + Vector3::new(0.04, -0.025, 0.02),
    ));
    let mut image_poses = vec![None; features.len()];
    for (frame, pose) in frame_poses.iter().enumerate() {
        install_image_poses(
            &rig,
            &frames[frame],
            pose.as_ref().unwrap(),
            &mut image_poses,
        );
    }
    let tracks = world_points
        .iter()
        .enumerate()
        .map(|(track, point)| SfmTrack {
            position: *point + Vector3::new(0.015, -0.01, 0.025),
            observations: (0..features.len())
                .map(|image| (image, track, features[image].keypoints[track]))
                .collect(),
        })
        .collect::<Vec<_>>();
    let (before_sum, before_count) =
        reprojection_error(&rig, &image_assignment, &image_poses, &tracks);
    let before_mean = before_sum / before_count as f64;
    let before_center = frame_poses[1].as_ref().unwrap().camera_center_world();
    let before_center_error = (before_center - truth[1].camera_center_world()).norm();
    let mut result = RigSfmResult {
        frame_poses,
        image_poses,
        tracks,
        registered_frames: 2,
        registered_images: 4,
        mean_reprojection_error_px: before_mean,
        seed_frame_index: 0,
        track_build_stats: TrackBuildStats::default(),
        work: RigSfmWorkStats::default(),
        bundle_adjustment: None,
    };
    let config = RigSfmConfig {
        final_bundle_adjustment: false,
        structure_refinement_iterations: 5,
        deferred_registration_pair_prefix: Some(1),
        ba_config: BaConfig {
            max_iterations: 40,
            linear_solver: LinearSolver::Sparse,
            robust_kernel: RobustKernel::None,
            parallel: false,
            ..BaConfig::default()
        },
        ..RigSfmConfig::default()
    };
    let target_rotations = vec![None, Some(truth[1].world_to_camera.rotation)];
    refine_rig_sfm_with_fixed_frame_rotations(
        &rig,
        &frames,
        &features,
        &target_rotations,
        &config,
        &mut result,
    )
    .unwrap();
    assert!(
        result.frame_poses[1]
            .as_ref()
            .unwrap()
            .world_to_camera
            .rotation
            .angle_to(&truth[1].world_to_camera.rotation)
            < 1.0e-10
    );
    let after_center_error = (result.frame_poses[1]
        .as_ref()
        .unwrap()
        .camera_center_world()
        - truth[1].camera_center_world())
    .norm();
    assert!(after_center_error < before_center_error);
    assert!(result.mean_reprojection_error_px.is_finite());
    assert!(result.bundle_adjustment.is_some());
    assert!(!result.tracks.is_empty());
}

#[test]
fn rejects_pnp_sensor_count_outside_the_calibrated_rig() {
    let rig = GeneralizedCameraRig::new(vec![
        RigSensor {
            camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::identity(),
        },
        RigSensor {
            camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.20, 0.0, 0.0)),
        },
    ])
    .unwrap();

    let error = validate_inputs(
        &rig,
        &[],
        &[],
        &[],
        &RigSfmConfig {
            seed_frame: Some(0),
            ..RigSfmConfig::default()
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        RigSfmError::InvalidSeedFrame {
            frame: 0,
            frame_count: 0,
        }
    );

    for requested in [0, 3] {
        let error = validate_inputs(
            &rig,
            &[],
            &[],
            &[],
            &RigSfmConfig {
                min_pnp_sensors: requested,
                ..RigSfmConfig::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            RigSfmError::InvalidMinPnpSensors {
                requested,
                sensor_count: 2,
            }
        );
    }
    for requested in [0, 3] {
        let error = validate_inputs(
            &rig,
            &[],
            &[],
            &[],
            &RigSfmConfig {
                direct_stereo_min_pnp_sensors: Some(requested),
                ..RigSfmConfig::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            RigSfmError::InvalidDirectStereoMinPnpSensors {
                requested,
                sensor_count: 2,
            }
        );
    }
    let error = validate_inputs(
        &rig,
        &[],
        &[],
        &[],
        &RigSfmConfig {
            direct_stereo_min_triangulation_angle_deg: 0.0,
            ..RigSfmConfig::default()
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        RigSfmError::InvalidDirectStereoTriangulationAngle(0.0)
    );

    let error = validate_inputs(
        &rig,
        &[],
        &[],
        &[],
        &RigSfmConfig {
            complete_tracks_after_registration: true,
            track_completion_max_passes: 0,
            ..RigSfmConfig::default()
        },
    )
    .unwrap_err();
    assert_eq!(error, RigSfmError::InvalidTrackCompletionPasses);
    let error = validate_inputs(
        &rig,
        &[],
        &[],
        &[],
        &RigSfmConfig {
            complete_tracks_after_registration: true,
            track_completion_max_reprojection_error_px: f64::NAN,
            ..RigSfmConfig::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        RigSfmError::InvalidTrackCompletionReprojectionError(value) if value.is_nan()
    ));

    for requested in [0.0, f64::NAN] {
        let error = validate_inputs(
            &rig,
            &[],
            &[],
            &[],
            &RigSfmConfig {
                repair_paired_pose_jumps: true,
                paired_pose_jump_absolute_step_m: requested,
                ..RigSfmConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RigSfmError::InvalidPairedPoseJumpStep(value)
                if value == requested || (value.is_nan() && requested.is_nan())
        ));
    }
    for requested in [1.0, f64::NAN] {
        let error = validate_inputs(
            &rig,
            &[],
            &[],
            &[],
            &RigSfmConfig {
                repair_paired_pose_jumps: true,
                paired_pose_jump_min_step_ratio: requested,
                ..RigSfmConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RigSfmError::InvalidPairedPoseJumpStepRatio(value)
                if value == requested || (value.is_nan() && requested.is_nan())
        ));
    }
    let error = validate_inputs(
        &rig,
        &[],
        &[],
        &[],
        &RigSfmConfig {
            repair_paired_pose_jumps: true,
            paired_pose_jump_max_frame_span: 0,
            ..RigSfmConfig::default()
        },
    )
    .unwrap_err();
    assert_eq!(error, RigSfmError::InvalidPairedPoseJumpFrameSpan);
    for requested in [0.0, 1.0, f64::NAN] {
        let error = validate_inputs(
            &rig,
            &[],
            &[],
            &[],
            &RigSfmConfig {
                repair_paired_pose_jumps: true,
                paired_pose_jump_max_closure_ratio: requested,
                ..RigSfmConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RigSfmError::InvalidPairedPoseJumpClosureRatio(value)
                if value == requested || (value.is_nan() && requested.is_nan())
        ));
    }
}

#[test]
fn native_ba_backend_defaults_to_legacy_and_is_explicit() {
    let config = RigSfmConfig::default();
    assert_eq!(config.ba_backend, RigBaBackend::Legacy);
    assert_ne!(config.ba_backend, RigBaBackend::MatrixFreeStrict);
}
