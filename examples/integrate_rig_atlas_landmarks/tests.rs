use super::*;

fn minimal_parse_args(extra: &[&str]) -> Result<Args, String> {
    let mut args = vec![
        "example",
        "--rig-manifest",
        "r",
        "--nodes-tsv",
        "n",
        "--atlas-dir",
        "a",
        "--out-dir",
        "o",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    args.extend(extra.iter().map(|value| (*value).to_owned()));
    parse_args(args)
}

fn test_image(id: u64, name: &str, keypoints: Vec<Point2<f64>>) -> GlobalImage {
    GlobalImage {
        atlas: AtlasImage {
            global_image_id: id,
            frame_id: id,
            sensor_index: 0,
            name: name.to_owned(),
            camera_id: 1,
            pose: Pose::identity(),
        },
        keypoints,
    }
}

fn test_camera() -> Camera {
    Camera::pinhole(1, 640, 480, 100.0, 100.0, 320.0, 240.0)
}

fn recovery_fixture(
    add_bad_second_sensor_observation: bool,
) -> (
    RigManifest,
    TrackStore,
    BTreeMap<u64, GlobalImage>,
    BTreeMap<u64, Camera>,
    Vec<LandmarkOutput>,
    Pose,
) {
    let camera0 = Camera::pinhole(1, 640, 480, 100.0, 101.0, 320.0, 240.0);
    let camera1 = Camera::pinhole(2, 640, 480, 102.0, 99.0, 321.0, 239.0);
    let sensor0 = SensorCalibration {
        camera_id: 1,
        width: 640,
        height: 480,
        fx: 100.0,
        fy: 101.0,
        cx: 320.0,
        cy: 240.0,
        sensor_from_rig: SE3::identity(),
    };
    let sensor1 = SensorCalibration {
        camera_id: 2,
        width: 640,
        height: 480,
        fx: 102.0,
        fy: 99.0,
        cx: 321.0,
        cy: 239.0,
        sensor_from_rig: SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.005, -0.01, 0.008)),
            Vector3::new(-0.35, 0.01, -0.02),
        ),
    };
    let manifest = RigManifest {
        sensors: BTreeMap::from([(0, sensor0.clone()), (1, sensor1.clone())]),
        assignments: BTreeMap::new(),
    };
    let truth_rig_pose = Pose::from_world_to_camera(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.04, -0.03, 0.02)),
        Vector3::new(0.25, -0.15, 0.35),
    );
    let bad_rig_pose = Pose::from_world_to_camera(
        UnitQuaternion::from_scaled_axis(Vector3::new(-0.1, 0.08, -0.04)),
        Vector3::new(-0.8, 0.4, 0.2),
    );
    let sensor_pose = |sensor: &SensorCalibration, rig_pose: &Pose| Pose {
        world_to_camera: sensor.sensor_from_rig.compose(&rig_pose.world_to_camera),
    };
    let mut images = BTreeMap::new();
    let mut insert_image = |global_image_id, frame_id, sensor_index, camera_id, pose| {
        images.insert(
            global_image_id,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id,
                    frame_id,
                    sensor_index,
                    name: format!("{global_image_id}.png"),
                    camera_id,
                    pose,
                },
                keypoints: vec![Point2::new(0.0, 0.0); 7],
            },
        );
    };
    insert_image(1, 10, 0, 1, sensor_pose(&sensor0, &Pose::identity()));
    insert_image(2, 10, 1, 2, sensor_pose(&sensor1, &Pose::identity()));
    insert_image(3, 20, 0, 1, sensor_pose(&sensor0, &bad_rig_pose));
    insert_image(4, 20, 1, 2, sensor_pose(&sensor1, &bad_rig_pose));

    let mut states = BTreeMap::new();
    let mut tracks = Vec::new();
    let base_point = Point3::new(0.1, -0.2, 4.6);
    let base_xy0 = camera0
        .project(&images[&1].atlas.pose.transform_world_point(&base_point))
        .unwrap();
    let base_xy1 = camera1
        .project(&images[&2].atlas.pose.transform_world_point(&base_point))
        .unwrap();
    images.get_mut(&1).unwrap().keypoints[6] = base_xy0;
    images.get_mut(&2).unwrap().keypoints[6] = base_xy1;
    let base_keys = vec![
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 6,
        },
        ObservationKey {
            global_image_id: 2,
            keypoint_index: 6,
        },
    ];
    for (key, xy) in [(base_keys[0], base_xy0), (base_keys[1], base_xy1)] {
        states.insert(key, ObservationState { xy, owner_track: 0 });
    }
    tracks.push(GlobalTrack {
        observations: base_keys.clone(),
    });
    let baseline_landmark = LandmarkOutput {
        track_id: 0,
        position: base_point,
        observations: base_keys,
        errors: vec![0.0, 0.0],
        rms_error: 0.0,
        mean_error: 0.0,
        max_error: 0.0,
        dlt_sample_count: 2,
    };

    let points = [
        Point3::new(-1.2, -0.6, 4.5),
        Point3::new(-0.7, 0.4, 5.1),
        Point3::new(-0.1, -0.3, 5.8),
        Point3::new(0.4, 0.5, 4.8),
        Point3::new(0.9, -0.4, 5.5),
        Point3::new(1.3, 0.2, 6.2),
    ];
    let target_pose0 = sensor_pose(&sensor0, &truth_rig_pose);
    let target_pose1 = sensor_pose(&sensor1, &truth_rig_pose);
    let point_count = points.len();
    for (index, point) in points.into_iter().enumerate() {
        let xy0 = camera0
            .project(&target_pose0.transform_world_point(&point))
            .unwrap();
        let xy1 = camera1
            .project(&target_pose1.transform_world_point(&point))
            .unwrap();
        let anchor_xy0 = camera0
            .project(&images[&1].atlas.pose.transform_world_point(&point))
            .unwrap();
        let anchor_xy1 = camera1
            .project(&images[&2].atlas.pose.transform_world_point(&point))
            .unwrap();
        images.get_mut(&1).unwrap().keypoints[index] = anchor_xy0;
        images.get_mut(&2).unwrap().keypoints[index] = anchor_xy1;
        images.get_mut(&4).unwrap().keypoints[index] = xy1;
        if add_bad_second_sensor_observation && index == point_count - 1 {
            images.get_mut(&3).unwrap().keypoints[index] = Point2::new(xy0.x + 120.0, xy0.y);
        }
        let key1 = ObservationKey {
            global_image_id: 1,
            keypoint_index: index,
        };
        let key2 = ObservationKey {
            global_image_id: 2,
            keypoint_index: index,
        };
        let key4 = ObservationKey {
            global_image_id: 4,
            keypoint_index: index,
        };
        let mut keys = vec![key1, key2, key4];
        states.insert(
            key1,
            ObservationState {
                xy: images[&1].keypoints[index],
                owner_track: index + 1,
            },
        );
        states.insert(
            key2,
            ObservationState {
                xy: images[&2].keypoints[index],
                owner_track: index + 1,
            },
        );
        states.insert(
            key4,
            ObservationState {
                xy: xy1,
                owner_track: index + 1,
            },
        );
        if add_bad_second_sensor_observation && index == points.len() - 1 {
            let key3 = ObservationKey {
                global_image_id: 3,
                keypoint_index: index,
            };
            keys.insert(2, key3);
            states.insert(
                key3,
                ObservationState {
                    xy: images[&3].keypoints[index],
                    owner_track: index + 1,
                },
            );
        }
        keys.sort_unstable();
        tracks.push(GlobalTrack { observations: keys });
    }
    (
        manifest,
        TrackStore {
            observations: states,
            tracks,
        },
        images,
        BTreeMap::from([(1, camera0), (2, camera1)]),
        vec![baseline_landmark],
        truth_rig_pose,
    )
}

fn boundary_repair_fixture() -> (
    RigManifest,
    TrackStore,
    BTreeMap<u64, GlobalImage>,
    BTreeMap<u64, Camera>,
    Vec<LandmarkOutput>,
) {
    let camera0 = Camera::pinhole(1, 640, 480, 100.0, 101.0, 320.0, 240.0);
    let camera1 = Camera::pinhole(2, 640, 480, 102.0, 99.0, 321.0, 239.0);
    let sensor0 = SensorCalibration {
        camera_id: 1,
        width: 640,
        height: 480,
        fx: 100.0,
        fy: 101.0,
        cx: 320.0,
        cy: 240.0,
        sensor_from_rig: SE3::identity(),
    };
    let sensor1 = SensorCalibration {
        camera_id: 2,
        width: 640,
        height: 480,
        fx: 102.0,
        fy: 99.0,
        cx: 321.0,
        cy: 239.0,
        sensor_from_rig: SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.005, -0.01, 0.008)),
            Vector3::new(-0.35, 0.01, -0.02),
        ),
    };
    let manifest = RigManifest {
        sensors: BTreeMap::from([(0, sensor0.clone()), (1, sensor1.clone())]),
        assignments: BTreeMap::new(),
    };
    let rig_pose = Pose::identity();
    let mut images = BTreeMap::new();
    for (image_id, frame_id, sensor_index, sensor) in [
        (1, 10, 0, &sensor0),
        (2, 10, 1, &sensor1),
        (3, 30, 0, &sensor0),
        (4, 30, 1, &sensor1),
    ] {
        images.insert(
            image_id,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: image_id,
                    frame_id,
                    sensor_index,
                    name: format!("{frame_id}-{sensor_index}.png"),
                    camera_id: sensor.camera_id,
                    pose: Pose {
                        world_to_camera: sensor.sensor_from_rig.compose(&rig_pose.world_to_camera),
                    },
                },
                keypoints: vec![Point2::new(0.0, 0.0); 7],
            },
        );
    }
    let cameras = BTreeMap::from([(1, camera0.clone()), (2, camera1.clone())]);
    let mut observations = BTreeMap::new();
    let mut tracks = Vec::new();
    let mut landmarks = Vec::new();
    let mut add_track = |track_id: usize, point: Point3<f64>, keys: Vec<ObservationKey>| {
        for key in &keys {
            let image = images.get_mut(&key.global_image_id).unwrap();
            let camera = cameras.get(&image.atlas.camera_id).unwrap();
            let xy = camera
                .project(&image.atlas.pose.transform_world_point(&point))
                .unwrap();
            image.keypoints[key.keypoint_index] = xy;
            observations.insert(
                *key,
                ObservationState {
                    xy,
                    owner_track: track_id,
                },
            );
        }
        tracks.push(GlobalTrack {
            observations: keys.clone(),
        });
        (point, keys)
    };
    let base10 = add_track(
        0,
        Point3::new(0.1, -0.2, 4.6),
        vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 6,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 6,
            },
        ],
    );
    let base30 = add_track(
        1,
        Point3::new(-0.2, 0.3, 5.2),
        vec![
            ObservationKey {
                global_image_id: 3,
                keypoint_index: 6,
            },
            ObservationKey {
                global_image_id: 4,
                keypoint_index: 6,
            },
        ],
    );
    landmarks.push(LandmarkOutput {
        track_id: 0,
        position: base10.0,
        observations: base10.1,
        errors: vec![0.0, 0.0],
        rms_error: 0.0,
        mean_error: 0.0,
        max_error: 0.0,
        dlt_sample_count: 2,
    });
    landmarks.push(LandmarkOutput {
        track_id: 1,
        position: base30.0,
        observations: base30.1,
        errors: vec![0.0, 0.0],
        rms_error: 0.0,
        mean_error: 0.0,
        max_error: 0.0,
        dlt_sample_count: 2,
    });
    let points = [
        Point3::new(-1.2, -0.6, 4.5),
        Point3::new(-0.7, 0.4, 5.1),
        Point3::new(-0.1, -0.3, 5.8),
        Point3::new(0.4, 0.5, 4.8),
        Point3::new(0.9, -0.4, 5.5),
        Point3::new(1.3, 0.2, 6.2),
    ];
    for (index, point) in points.into_iter().enumerate() {
        let keys = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: index,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: index,
            },
            ObservationKey {
                global_image_id: 3,
                keypoint_index: index,
            },
        ];
        add_track(index + 2, point, keys);
    }
    (
        manifest,
        TrackStore {
            observations,
            tracks,
        },
        images,
        cameras,
        landmarks,
    )
}

fn joint_ba_fixture() -> (
    RigManifest,
    TrackStore,
    BTreeMap<u64, GlobalImage>,
    BTreeMap<u64, Camera>,
    Vec<LandmarkOutput>,
    BTreeSet<u64>,
) {
    let camera0 = Camera::pinhole(1, 640, 480, 180.0, 181.0, 320.0, 240.0);
    let camera1 = Camera::pinhole(2, 640, 480, 179.0, 180.0, 321.0, 239.0);
    let sensor0 = SensorCalibration {
        camera_id: 1,
        width: 640,
        height: 480,
        fx: 180.0,
        fy: 181.0,
        cx: 320.0,
        cy: 240.0,
        sensor_from_rig: SE3::identity(),
    };
    let sensor1 = SensorCalibration {
        camera_id: 2,
        width: 640,
        height: 480,
        fx: 179.0,
        fy: 180.0,
        cx: 321.0,
        cy: 239.0,
        sensor_from_rig: SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.015)),
            Vector3::new(-0.35, 0.02, -0.01),
        ),
    };
    let manifest = RigManifest {
        sensors: BTreeMap::from([(0, sensor0.clone()), (1, sensor1.clone())]),
        assignments: BTreeMap::new(),
    };
    let truth_poses = [
        (
            100,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros()),
        ),
        (
            140,
            Pose::from_world_to_camera(
                UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.015, 0.02)),
                Vector3::new(-0.35, 0.02, 0.03),
            ),
        ),
        (
            200,
            Pose::from_world_to_camera(
                UnitQuaternion::from_scaled_axis(Vector3::new(-0.02, 0.01, -0.01)),
                Vector3::new(-0.75, -0.03, 0.02),
            ),
        ),
    ];
    let mut images = BTreeMap::new();
    let mut image_for = BTreeMap::new();
    let mut global_image_id = 1;
    for (frame_id, truth_pose) in &truth_poses {
        let rig_pose = if *frame_id == 140 {
            Pose::from_world_to_camera(
                truth_pose.world_to_camera.rotation
                    * UnitQuaternion::from_scaled_axis(Vector3::new(0.006, -0.004, 0.003)),
                truth_pose.world_to_camera.translation + Vector3::new(0.04, -0.02, 0.015),
            )
        } else {
            truth_pose.clone()
        };
        for (sensor_index, sensor) in [(0usize, &sensor0), (1usize, &sensor1)] {
            let image_id = global_image_id;
            global_image_id += 1;
            let pose = Pose {
                world_to_camera: sensor.sensor_from_rig.compose(&rig_pose.world_to_camera),
            };
            images.insert(
                image_id,
                GlobalImage {
                    atlas: AtlasImage {
                        global_image_id: image_id,
                        frame_id: *frame_id,
                        sensor_index,
                        name: format!("{frame_id}-{sensor_index}.png"),
                        camera_id: sensor.camera_id,
                        pose,
                    },
                    keypoints: vec![Point2::new(0.0, 0.0); 12],
                },
            );
            image_for.insert((*frame_id, sensor_index), image_id);
        }
    }
    let points = (0..12)
        .map(|index| {
            Point3::new(
                -0.9 + index as f64 * 0.16,
                -0.45 + (index % 4) as f64 * 0.22,
                4.0 + (index % 5) as f64 * 0.25,
            )
        })
        .collect::<Vec<_>>();
    let mut observations = BTreeMap::new();
    let mut tracks = Vec::new();
    let mut landmarks = Vec::new();
    for (point_index, point) in points.iter().enumerate() {
        let mut keys = Vec::new();
        for (frame_id, truth_pose) in &truth_poses {
            for (sensor_index, sensor, camera) in
                [(0usize, &sensor0, &camera0), (1usize, &sensor1, &camera1)]
            {
                let image_id = image_for[&(*frame_id, sensor_index)];
                let xy = camera
                    .project(
                        &sensor
                            .sensor_from_rig
                            .transform_point(&truth_pose.transform_world_point(point)),
                    )
                    .unwrap();
                images.get_mut(&image_id).unwrap().keypoints[point_index] = xy;
                let key = ObservationKey {
                    global_image_id: image_id,
                    keypoint_index: point_index,
                };
                observations.insert(
                    key,
                    ObservationState {
                        xy,
                        owner_track: point_index,
                    },
                );
                keys.push(key);
            }
        }
        keys.sort_unstable();
        tracks.push(GlobalTrack {
            observations: keys.clone(),
        });
        landmarks.push(LandmarkOutput {
            track_id: point_index,
            position: Point3::from(point.coords + Vector3::new(0.015, -0.012, 0.025)),
            observations: keys,
            errors: vec![0.0; 6],
            rms_error: 0.0,
            mean_error: 0.0,
            max_error: 0.0,
            dlt_sample_count: 6,
        });
    }
    (
        manifest,
        TrackStore {
            observations,
            tracks,
        },
        images,
        BTreeMap::from([(1, camera0), (2, camera1)]),
        landmarks,
        BTreeSet::from([100, 140, 200]),
    )
}

#[test]
fn args_require_all_paths_and_reject_unknown() {
    let error = parse_args(["example".to_owned(), "--rig-manifest".to_owned()]).unwrap_err();
    assert!(error.contains("requires PATH"));
    let error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--bad".to_owned(),
    ])
    .unwrap_err();
    assert!(error.contains("unknown argument"));

    let args = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
    ])
    .unwrap();
    assert!(!args.recover_zero_support_frames);
    assert!(!args.joint_rig_ba);
    assert!(!args.joint_rig_ba_filter_observations);
    assert!(!args.joint_rig_ba_preserve_optimized_points);
    assert_eq!(args.pre_ba_out_dir, None);
    assert_eq!(args.post_pass1_out_dir, None);
    assert_eq!(args.joint_rig_ba_filter_sweeps, 1);
    assert!(!args.diagnose_cross_boundary_pnp);
    assert_eq!(args.diagnostic_left_max_frame, None);
    assert!(!args.repair_cross_boundary);
    assert_eq!(args.repair_left_max_frame, None);
    let args = parse_args([
        "example".to_owned(),
        "--recover-zero-support-frames".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
    ])
    .unwrap();
    assert!(args.recover_zero_support_frames);
    assert!(!args.joint_rig_ba);
    let duplicate_error = parse_args([
        "example".to_owned(),
        "--recover-zero-support-frames".to_owned(),
        "--recover-zero-support-frames".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
    ])
    .unwrap_err();
    assert!(duplicate_error.contains("duplicate argument"));
    let args = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--joint-rig-ba".to_owned(),
    ])
    .unwrap();
    assert!(args.joint_rig_ba);
    let args = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--joint-rig-ba".to_owned(),
        "--joint-rig-ba-filter-observations".to_owned(),
    ])
    .unwrap();
    assert!(args.joint_rig_ba_filter_observations);
    let args = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--joint-rig-ba".to_owned(),
        "--joint-rig-ba-filter-observations".to_owned(),
        "--joint-rig-ba-preserve-optimized-points".to_owned(),
    ])
    .unwrap();
    assert!(args.joint_rig_ba_preserve_optimized_points);
    assert_eq!(args.joint_rig_ba_filter_sweeps, 1);
    let error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--joint-rig-ba-filter-observations".to_owned(),
    ])
    .unwrap_err();
    assert!(error.contains("requires --joint-rig-ba"));
    let error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--joint-rig-ba-preserve-optimized-points".to_owned(),
    ])
    .unwrap_err();
    assert!(error.contains("requires --joint-rig-ba and --joint-rig-ba-filter-observations"));

    let args = minimal_parse_args(&[
        "--joint-rig-ba",
        "--joint-rig-ba-filter-observations",
        "--joint-rig-ba-filter-sweeps",
        "2",
        "--post-pass1-out-dir",
        "pass1",
    ])
    .unwrap();
    assert_eq!(args.joint_rig_ba_filter_sweeps, 2);
    assert_eq!(args.post_pass1_out_dir, Some(PathBuf::from("pass1")));
    for invalid in ["0", "3", "255"] {
        let error = minimal_parse_args(&[
            "--joint-rig-ba",
            "--joint-rig-ba-filter-observations",
            "--joint-rig-ba-filter-sweeps",
            invalid,
            "--post-pass1-out-dir",
            "pass1",
        ])
        .unwrap_err();
        assert!(error.contains("accepts only 1 or 2"), "{invalid}: {error}");
    }
    let error = minimal_parse_args(&[
        "--joint-rig-ba",
        "--joint-rig-ba-filter-observations",
        "--joint-rig-ba-filter-sweeps",
        "2",
    ])
    .unwrap_err();
    assert!(error.contains("post-pass1-out-dir"));
    let error = minimal_parse_args(&[
        "--joint-rig-ba",
        "--joint-rig-ba-filter-sweeps",
        "2",
        "--post-pass1-out-dir",
        "pass1",
    ])
    .unwrap_err();
    assert!(error.contains("filter-observations"));
    let error = minimal_parse_args(&[
        "--joint-rig-ba",
        "--joint-rig-ba-filter-observations",
        "--joint-rig-ba-filter-sweeps",
        "2",
        "--post-pass1-out-dir",
        "pass1",
        "--joint-rig-ba-filter-sweeps",
        "1",
    ])
    .unwrap_err();
    assert!(error.contains("duplicate argument"));
    let error = minimal_parse_args(&[
        "--joint-rig-ba",
        "--joint-rig-ba-filter-observations",
        "--joint-rig-ba-filter-sweeps",
        "2",
        "--post-pass1-out-dir",
        "pass1",
        "--joint-rig-ba-preserve-optimized-points",
    ])
    .unwrap_err();
    assert!(error.contains("cannot be combined"));
    let error = minimal_parse_args(&["--post-pass1-out-dir", "pass1"]).unwrap_err();
    assert!(error.contains("filter-sweeps 2"));
    let args = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--joint-rig-ba".to_owned(),
        "--pre-ba-out-dir".to_owned(),
        "checkpoint".to_owned(),
    ])
    .unwrap();
    assert_eq!(args.pre_ba_out_dir, Some(PathBuf::from("checkpoint")));
    let error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--pre-ba-out-dir".to_owned(),
        "checkpoint".to_owned(),
    ])
    .unwrap_err();
    assert!(error.contains("requires --joint-rig-ba"));
    let duplicate_error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--joint-rig-ba".to_owned(),
        "--joint-rig-ba".to_owned(),
    ])
    .unwrap_err();
    assert!(duplicate_error.contains("duplicate argument"));

    let args = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--diagnose-cross-boundary-pnp".to_owned(),
        "--diagnostic-left-max-frame".to_owned(),
        "1999".to_owned(),
    ])
    .unwrap();
    assert!(args.diagnose_cross_boundary_pnp);
    assert_eq!(args.diagnostic_left_max_frame, Some(1999));
    let error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--diagnose-cross-boundary-pnp".to_owned(),
    ])
    .unwrap_err();
    assert!(error.contains("diagnostic-left-max-frame"));
    let error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--diagnostic-left-max-frame".to_owned(),
        "1999".to_owned(),
    ])
    .unwrap_err();
    assert!(error.contains("requires --diagnose-cross-boundary-pnp"));

    let args = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--repair-cross-boundary".to_owned(),
        "--repair-left-max-frame".to_owned(),
        "1999".to_owned(),
        "--recover-zero-support-frames".to_owned(),
    ])
    .unwrap();
    assert!(args.repair_cross_boundary);
    assert_eq!(args.repair_left_max_frame, Some(1999));
    assert!(args.recover_zero_support_frames);
    let error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--repair-cross-boundary".to_owned(),
    ])
    .unwrap_err();
    assert!(error.contains("repair-left-max-frame"));
    let error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--repair-left-max-frame".to_owned(),
        "1999".to_owned(),
    ])
    .unwrap_err();
    assert!(error.contains("requires --repair-cross-boundary"));
    let args = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--repair-cross-boundary".to_owned(),
        "--repair-left-max-frame".to_owned(),
        "1999".to_owned(),
        "--joint-rig-ba".to_owned(),
    ])
    .unwrap();
    assert!(args.repair_cross_boundary);
    assert!(args.joint_rig_ba);
    let error = parse_args([
        "example".to_owned(),
        "--rig-manifest".to_owned(),
        "r".to_owned(),
        "--nodes-tsv".to_owned(),
        "n".to_owned(),
        "--atlas-dir".to_owned(),
        "a".to_owned(),
        "--out-dir".to_owned(),
        "o".to_owned(),
        "--repair-cross-boundary".to_owned(),
        "--repair-left-max-frame".to_owned(),
        "1999".to_owned(),
        "--diagnose-cross-boundary-pnp".to_owned(),
        "--diagnostic-left-max-frame".to_owned(),
        "1999".to_owned(),
    ])
    .unwrap_err();
    assert!(error.contains("mutually exclusive"));
}

#[test]
fn pre_ba_order_view_is_canonical_without_mutating_ba_order() {
    let (_manifest, _store, _images, _cameras, mut landmarks, _active) = joint_ba_fixture();
    landmarks.reverse();
    let before = landmarks.clone();
    let indices = canonical_landmark_indices(&landmarks);
    assert_eq!(indices.len(), landmarks.len());
    assert_eq!(indices, (0..landmarks.len()).rev().collect::<Vec<_>>());
    assert_eq!(landmarks, before);
}

#[test]
fn canonical_pass1_checkpoint_writer_does_not_mutate_filter_state() {
    let (_manifest, _store, images, cameras, mut landmarks, _active) = joint_ba_fixture();
    landmarks.reverse();
    let before_images = images.clone();
    let before_landmarks = landmarks.clone();
    let root = std::env::temp_dir().join(format!(
        "visloc_integrate_canonical_checkpoint_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = fs::remove_dir_all(&root);
    let stats = IngestStats {
        output_landmarks: landmarks.len(),
        output_observations: landmark_observation_count(&landmarks),
        ..IngestStats::default()
    };
    write_model_canonical(&root, &images, &cameras, &landmarks, &stats).unwrap();
    assert_eq!(images, before_images);
    assert_eq!(landmarks, before_landmarks);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn output_and_checkpoint_ancestor_collision_is_rejected() {
    let root = std::env::temp_dir().join(format!(
        "visloc_integrate_checkpoint_paths_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let args = Args {
        rig_manifest: root.join("manifest.tsv"),
        nodes_tsv: root.join("nodes.tsv"),
        atlas_dir: root.join("atlas"),
        out_dir: root.join("out"),
        pre_ba_out_dir: Some(root.join("out/checkpoint")),
        post_pass1_out_dir: None,
        recover_zero_support_frames: false,
        joint_rig_ba: true,
        joint_rig_ba_filter_observations: false,
        joint_rig_ba_preserve_optimized_points: false,
        joint_rig_ba_filter_sweeps: 1,
        diagnose_cross_boundary_pnp: false,
        diagnostic_left_max_frame: None,
        repair_cross_boundary: false,
        repair_left_max_frame: None,
    };
    let nodes = vec![NodeSpec {
        node_id: 1,
        window_start: 0,
        images_txt: root.join("source/images.txt"),
    }];
    let error = validate_write_destinations(&args, &nodes).unwrap_err();
    assert!(error.contains("ambiguous output destinations"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn post_pass1_checkpoint_requires_empty_nonoverlapping_destination() {
    let root = std::env::temp_dir().join(format!(
        "visloc_integrate_post_pass1_paths_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("source")).unwrap();
    fs::write(root.join("source/images.txt"), "").unwrap();
    let nodes = vec![NodeSpec {
        node_id: 1,
        window_start: 0,
        images_txt: root.join("source/images.txt"),
    }];
    let mut args = Args {
        rig_manifest: root.join("manifest.tsv"),
        nodes_tsv: root.join("nodes.tsv"),
        atlas_dir: root.join("atlas"),
        out_dir: root.join("out"),
        pre_ba_out_dir: None,
        post_pass1_out_dir: Some(root.join("post")),
        recover_zero_support_frames: false,
        joint_rig_ba: true,
        joint_rig_ba_filter_observations: true,
        joint_rig_ba_preserve_optimized_points: false,
        joint_rig_ba_filter_sweeps: 2,
        diagnose_cross_boundary_pnp: false,
        diagnostic_left_max_frame: None,
        repair_cross_boundary: false,
        repair_left_max_frame: None,
    };
    validate_write_destinations(&args, &nodes).unwrap();
    fs::create_dir_all(args.post_pass1_out_dir.as_ref().unwrap()).unwrap();
    fs::write(
        args.post_pass1_out_dir.as_ref().unwrap().join("stale"),
        "stale",
    )
    .unwrap();
    let error = validate_write_destinations(&args, &nodes).unwrap_err();
    assert!(error.contains("post-pass1-out-dir") && error.contains("new or empty"));
    args.post_pass1_out_dir = Some(root.join("out/post"));
    let error = validate_write_destinations(&args, &nodes).unwrap_err();
    assert!(error.contains("ambiguous output destinations"));
    let _ = fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn path_comparison_resolves_symlink_before_parent_dir() {
    use std::os::unix::fs::symlink;

    let root = std::env::temp_dir().join(format!(
        "visloc_integrate_checkpoint_symlink_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("real")).unwrap();
    symlink(root.join("real"), root.join("link")).unwrap();
    assert!(paths_overlap(&root.join("link/../real"), &root.join("real")).unwrap());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn filtered_joint_ba_candidate_updates_transactionally_and_preserves_rig() {
    let (manifest, mut store, mut images, cameras, mut landmarks, active) = joint_ba_fixture();
    let baseline =
        support_connectivity_for_landmarks(&landmarks, &BTreeMap::new(), &BTreeSet::new(), &images)
            .unwrap();
    let candidate = build_joint_rig_ba_filter_candidate(
        &manifest, &store, &images, &cameras, &landmarks, &active, &baseline,
    )
    .unwrap();
    assert!(cost_non_increasing(
        candidate.full_pre_ba_cost,
        candidate.full_post_ba_cost
    ));
    assert!(cost_non_increasing(
        candidate.retained_pre_ba_cost,
        candidate.retained_post_filter_cost
    ));
    apply_joint_rig_ba_filter_candidate(&candidate, &mut store, &mut images, &mut landmarks)
        .unwrap();
    validate_filtered_model(&baseline, &store, &images, &cameras, &landmarks).unwrap();
    for image in images.values() {
        let sensor = &manifest.sensors[&image.atlas.sensor_index];
        let rig_pose = derive_rig_pose_for_frame(&manifest, image.atlas.frame_id, &images).unwrap();
        let expected = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
        assert!(
            (expected.translation - image.atlas.pose.world_to_camera.translation).norm() < 1.0e-8
        );
    }
}

#[test]
fn filtered_apply_removes_observations_and_track_without_resurrection() {
    let (_manifest, mut store, mut images, cameras, mut landmarks, _active) = joint_ba_fixture();
    let baseline =
        support_connectivity_for_landmarks(&landmarks, &BTreeMap::new(), &BTreeSet::new(), &images)
            .unwrap();
    let removed = landmarks[0].clone();
    let removed_observations = removed
        .observations
        .iter()
        .copied()
        .map(|key| (key, FilterObservationReason::TrackBelowMinimum))
        .collect();
    let candidate = JointRigBaFilteringCandidate {
        pose_overrides: BTreeMap::new(),
        landmark_updates: BTreeMap::new(),
        removed_observations,
        removed_track_ids: BTreeSet::from([removed.track_id]),
        selected_landmarks: 1,
        selected_observations: removed.observations.len(),
        referenced_frames: 3,
        free_frames: 1,
        full_pre_ba_cost: 0.0,
        full_post_ba_cost: 0.0,
        retained_pre_ba_cost: 0.0,
        retained_post_filter_cost: 0.0,
        retriangulated_tracks: 0,
        raw_preserved_tracks: 0,
        dlt_attempted_tracks: 0,
        raw_fallback_reason_counts: BTreeMap::new(),
        iterations: 0,
        converged: true,
    };
    apply_joint_rig_ba_filter_candidate(&candidate, &mut store, &mut images, &mut landmarks)
        .unwrap();
    assert!(store.tracks[removed.track_id].observations.is_empty());
    for key in removed.observations {
        assert!(!store.observations.contains_key(&key));
    }
    assert!(!landmarks
        .iter()
        .any(|landmark| landmark.track_id == removed.track_id));
    validate_filtered_model(&baseline, &store, &images, &cameras, &landmarks).unwrap();
}

#[test]
fn filtered_apply_removes_a_partial_observation_and_updates_source_track() {
    let (_manifest, mut store, mut images, cameras, mut landmarks, _active) = joint_ba_fixture();
    let baseline =
        support_connectivity_for_landmarks(&landmarks, &BTreeMap::new(), &BTreeSet::new(), &images)
            .unwrap();
    let old = landmarks[0].clone();
    let removed_key = old.observations[0];
    let replacement = triangulate_track(
        old.track_id,
        &GlobalTrack {
            observations: old.observations[1..].to_vec(),
        },
        &store.observations,
        &images,
        &cameras,
    )
    .unwrap();
    let candidate = JointRigBaFilteringCandidate {
        pose_overrides: BTreeMap::new(),
        landmark_updates: BTreeMap::from([(0, replacement.clone())]),
        removed_observations: BTreeMap::from([(
            removed_key,
            FilterObservationReason::ReprojectionOverMax,
        )]),
        removed_track_ids: BTreeSet::new(),
        selected_landmarks: 1,
        selected_observations: old.observations.len(),
        referenced_frames: 3,
        free_frames: 1,
        full_pre_ba_cost: 0.0,
        full_post_ba_cost: 0.0,
        retained_pre_ba_cost: 0.0,
        retained_post_filter_cost: 0.0,
        retriangulated_tracks: 1,
        raw_preserved_tracks: 0,
        dlt_attempted_tracks: 1,
        raw_fallback_reason_counts: BTreeMap::new(),
        iterations: 0,
        converged: true,
    };
    apply_joint_rig_ba_filter_candidate(&candidate, &mut store, &mut images, &mut landmarks)
        .unwrap();
    assert!(!store.observations.contains_key(&removed_key));
    assert_eq!(landmarks[0].observations, replacement.observations);
    assert_eq!(store.tracks[0].observations, replacement.observations);
    validate_filtered_model(&baseline, &store, &images, &cameras, &landmarks).unwrap();
}

#[test]
fn filtered_apply_rejects_invalid_candidate_without_partial_mutation() {
    let (_manifest, mut store, mut images, _cameras, mut landmarks, _active) = joint_ba_fixture();
    let replacement = landmarks[0].clone();
    let candidate = JointRigBaFilteringCandidate {
        pose_overrides: BTreeMap::new(),
        landmark_updates: BTreeMap::from([
            (0, replacement.clone()),
            (landmarks.len() + 1, replacement),
        ]),
        removed_observations: BTreeMap::new(),
        removed_track_ids: BTreeSet::new(),
        selected_landmarks: 2,
        selected_observations: 0,
        referenced_frames: 0,
        free_frames: 0,
        full_pre_ba_cost: 0.0,
        full_post_ba_cost: 0.0,
        retained_pre_ba_cost: 0.0,
        retained_post_filter_cost: 0.0,
        retriangulated_tracks: 0,
        raw_preserved_tracks: 0,
        dlt_attempted_tracks: 0,
        raw_fallback_reason_counts: BTreeMap::new(),
        iterations: 0,
        converged: false,
    };
    let before_store = store.clone();
    let before_images = images.clone();
    let before_landmarks = landmarks.clone();
    let error =
        apply_joint_rig_ba_filter_candidate(&candidate, &mut store, &mut images, &mut landmarks)
            .unwrap_err();
    assert!(error.contains("landmark index"));
    assert_eq!(store, before_store);
    assert_eq!(images, before_images);
    assert_eq!(landmarks, before_landmarks);
}

#[test]
fn filtered_joint_ba_run_validates_final_model_and_reports_counts() {
    let (manifest, mut store, mut images, cameras, mut landmarks, _active) = joint_ba_fixture();
    let summary =
        run_joint_rig_ba_filtering(&manifest, &mut store, &mut images, &cameras, &mut landmarks)
            .unwrap();
    assert_eq!(summary.windows_considered, 1);
    assert_eq!(summary.windows_accepted, 1);
    assert_eq!(summary.windows_skipped, 0);
    assert!(summary.retriangulated_tracks > 0);
}

#[test]
fn two_filter_sweeps_are_deterministic_and_preserve_support() {
    let (manifest, mut store, mut images, cameras, mut landmarks, _active) = joint_ba_fixture();
    let first = run_joint_rig_ba_filtering_with_policy_and_pass(
        &manifest,
        &mut store,
        &mut images,
        &cameras,
        &mut landmarks,
        false,
        Some(1),
    )
    .unwrap();
    let second = run_joint_rig_ba_filtering_with_policy_and_pass(
        &manifest,
        &mut store,
        &mut images,
        &cameras,
        &mut landmarks,
        false,
        Some(2),
    )
    .unwrap();
    assert!(first.windows_accepted > 0);
    assert!(second.windows_accepted > 0);
    for image in images.values() {
        let sensor = &manifest.sensors[&image.atlas.sensor_index];
        let rig_pose = derive_rig_pose_for_frame(&manifest, image.atlas.frame_id, &images).unwrap();
        let expected = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
        assert!(
            (expected.translation - image.atlas.pose.world_to_camera.translation).norm() < 1.0e-8
        );
        let rotation_error =
            expected.rotation.inverse() * image.atlas.pose.world_to_camera.rotation;
        assert!(rotation_error.angle() < 1.0e-8);
    }
    let support =
        support_connectivity_for_landmarks(&landmarks, &BTreeMap::new(), &BTreeSet::new(), &images)
            .unwrap();
    assert_eq!(support.supported_frames.len(), 3);

    let (
        manifest_again,
        mut store_again,
        mut images_again,
        cameras_again,
        mut landmarks_again,
        _active,
    ) = joint_ba_fixture();
    let first_again = run_joint_rig_ba_filtering_with_policy_and_pass(
        &manifest_again,
        &mut store_again,
        &mut images_again,
        &cameras_again,
        &mut landmarks_again,
        false,
        Some(1),
    )
    .unwrap();
    let second_again = run_joint_rig_ba_filtering_with_policy_and_pass(
        &manifest_again,
        &mut store_again,
        &mut images_again,
        &cameras_again,
        &mut landmarks_again,
        false,
        Some(2),
    )
    .unwrap();
    assert_eq!(first, first_again);
    assert_eq!(second, second_again);
    assert_eq!(store, store_again);
    assert_eq!(images, images_again);
    assert_eq!(landmarks, landmarks_again);
}

#[test]
fn filtered_joint_ba_candidate_is_deterministic_and_rollback_is_sparse() {
    let (manifest, store, images, cameras, landmarks, active) = joint_ba_fixture();
    let baseline =
        support_connectivity_for_landmarks(&landmarks, &BTreeMap::new(), &BTreeSet::new(), &images)
            .unwrap();
    let first = build_joint_rig_ba_filter_candidate(
        &manifest, &store, &images, &cameras, &landmarks, &active, &baseline,
    )
    .unwrap();
    let second = build_joint_rig_ba_filter_candidate(
        &manifest, &store, &images, &cameras, &landmarks, &active, &baseline,
    )
    .unwrap();
    assert_eq!(first, second);

    let mut impossible_baseline = baseline.clone();
    impossible_baseline.component_count = 0;
    let error = build_joint_rig_ba_filter_candidate(
        &manifest,
        &store,
        &images,
        &cameras,
        &landmarks,
        &active,
        &impossible_baseline,
    )
    .unwrap_err();
    assert!(error.to_string().contains("connectivity"));
}

#[test]
fn optimized_point_policy_is_flag_dependent_and_deterministic() {
    let (manifest, store, images, cameras, landmarks, active) = joint_ba_fixture();
    let baseline =
        support_connectivity_for_landmarks(&landmarks, &BTreeMap::new(), &BTreeSet::new(), &images)
            .unwrap();
    let disabled = build_joint_rig_ba_filter_candidate(
        &manifest, &store, &images, &cameras, &landmarks, &active, &baseline,
    )
    .unwrap();
    assert_eq!(disabled.raw_preserved_tracks, 0);
    assert_eq!(disabled.raw_fallback_reason_counts, BTreeMap::new());

    let enabled = build_joint_rig_ba_filter_candidate_with_policy(
        &manifest, &store, &images, &cameras, &landmarks, &active, &baseline, true,
    )
    .unwrap();
    let enabled_again = build_joint_rig_ba_filter_candidate_with_policy(
        &manifest, &store, &images, &cameras, &landmarks, &active, &baseline, true,
    )
    .unwrap();
    let selected = selected_landmarks_for_frames(&landmarks, &active, &images).unwrap();
    let enabled_shared = build_joint_rig_ba_filter_candidate_with_selected(
        &manifest, &store, &images, &cameras, &landmarks, &active, &selected, &baseline, true,
    )
    .unwrap();
    assert_eq!(enabled, enabled_again);
    assert_eq!(enabled, enabled_shared);
    assert!(enabled.raw_preserved_tracks > 0);
    assert!(enabled.dlt_attempted_tracks < enabled.selected_landmarks);
}

#[test]
fn filtered_observation_depth_and_cost_gates_are_explicit() {
    let (_manifest, store, images, cameras, landmarks, _active) = joint_ba_fixture();
    let landmark = &landmarks[0];
    let key = landmark.observations[0];
    let image = &images[&key.global_image_id];
    let behind = image
        .atlas
        .pose
        .camera_to_world()
        .transform_point(&Point3::new(0.0, 0.0, -1.0));
    let error = classify_filter_observation(
        key,
        &behind,
        &store.observations,
        &images,
        &cameras,
        &BTreeMap::new(),
    )
    .unwrap_err();
    assert_eq!(error, FilterObservationReason::BehindCamera);
    assert!(cost_non_increasing(10.0, 10.0 + 1.0e-9));
    assert!(!cost_non_increasing(10.0, 10.1));
}

#[test]
fn optimized_raw_point_is_retained_only_with_complete_valid_keys_and_metrics() {
    let (_manifest, store, images, cameras, landmarks, _active) = joint_ba_fixture();
    let track = &store.tracks[landmarks[0].track_id];
    let dlt = triangulate_track(
        landmarks[0].track_id,
        track,
        &store.observations,
        &images,
        &cameras,
    )
    .unwrap();
    let mut raw = dlt.clone();
    raw.position += Vector3::new(1.0e-4, -1.0e-4, 2.0e-4);
    let retained = retain_raw_ba_point_if_valid(
        &raw,
        &raw.observations,
        &store.observations,
        &images,
        &cameras,
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(retained.position, raw.position);
    assert_ne!(retained.position, dlt.position);
    assert_eq!(retained.observations, raw.observations);
    assert!(retained.mean_error <= MAX_MEAN_REPROJECTION_PX);
    assert!(retained.max_error <= MAX_REPROJECTION_PX);
    assert_eq!(retained.errors.len(), retained.observations.len());

    let changed_keys = raw.observations[..raw.observations.len() - 1].to_vec();
    assert_eq!(
        retain_raw_ba_point_if_valid(
            &raw,
            &changed_keys,
            &store.observations,
            &images,
            &cameras,
            &BTreeMap::new(),
        )
        .unwrap_err(),
        RawPointFallbackReason::ObservationKeysChanged
    );
}

#[test]
fn optimized_raw_point_rejects_parallax_depth_and_mean_gates() {
    let (_manifest, store, images, cameras, landmarks, _active) = joint_ba_fixture();
    let raw = triangulate_track(
        landmarks[0].track_id,
        &store.tracks[landmarks[0].track_id],
        &store.observations,
        &images,
        &cameras,
    )
    .unwrap();

    let duplicate_key = raw.observations[0];
    let mut no_parallax = raw.clone();
    no_parallax.observations = vec![duplicate_key, duplicate_key];
    assert_eq!(
        retain_raw_ba_point_if_valid(
            &no_parallax,
            &no_parallax.observations,
            &store.observations,
            &images,
            &cameras,
            &BTreeMap::new(),
        )
        .unwrap_err(),
        RawPointFallbackReason::NoObservableParallax
    );

    let mut behind = raw.clone();
    let first_image = &images[&raw.observations[0].global_image_id];
    behind.position = first_image
        .atlas
        .pose
        .camera_to_world()
        .transform_point(&Point3::new(0.0, 0.0, -1.0));
    assert_eq!(
        retain_raw_ba_point_if_valid(
            &behind,
            &behind.observations,
            &store.observations,
            &images,
            &cameras,
            &BTreeMap::new(),
        )
        .unwrap_err(),
        RawPointFallbackReason::NonPositiveDepth
    );

    let mut shifted_observations = store.observations.clone();
    for key in &raw.observations {
        let image = &images[&key.global_image_id];
        let camera = &cameras[&image.atlas.camera_id];
        let projected = camera
            .project(&image.atlas.pose.transform_world_point(&raw.position))
            .unwrap();
        shifted_observations.get_mut(key).unwrap().xy = Point2::new(projected.x + 2.5, projected.y);
    }
    assert_eq!(
        retain_raw_ba_point_if_valid(
            &raw,
            &raw.observations,
            &shifted_observations,
            &images,
            &cameras,
            &BTreeMap::new(),
        )
        .unwrap_err(),
        RawPointFallbackReason::MeanOverMax
    );
}

#[test]
fn filtered_full_cost_gate_rejects_when_only_retained_cost_improves() {
    let error = validate_filter_costs(10.0, 11.0, 5.0, 4.0).unwrap_err();
    assert!(error.contains("full selected cost increased"));
    let error = validate_filter_costs(10.0, 9.0, 5.0, 6.0).unwrap_err();
    assert!(error.contains("retained cost increased"));
}

#[test]
fn filtered_partial_observation_removal_rejects_support_loss() {
    let (_manifest, _store, images, _cameras, landmarks, _active) = joint_ba_fixture();
    let single_track = vec![landmarks[0].clone()];
    let baseline = support_connectivity_for_landmarks(
        &single_track,
        &BTreeMap::new(),
        &BTreeSet::new(),
        &images,
    )
    .unwrap();
    let replacement = LandmarkOutput {
        observations: single_track[0].observations[..2].to_vec(),
        ..single_track[0].clone()
    };
    let candidate = support_connectivity_for_landmarks(
        &single_track,
        &BTreeMap::from([(0, replacement)]),
        &BTreeSet::new(),
        &images,
    )
    .unwrap();
    assert!(!connectivity_not_split(&baseline, &candidate));
}

#[test]
fn filtered_connectivity_rejects_a_split_and_accepts_a_merge() {
    let baseline = SupportConnectivity {
        supported_images: BTreeSet::new(),
        supported_frames: BTreeSet::from([1, 2, 3]),
        frame_components: BTreeMap::from([(1, 0), (2, 0), (3, 0)]),
        component_count: 1,
    };
    let split = SupportConnectivity {
        supported_images: BTreeSet::new(),
        supported_frames: BTreeSet::from([1, 2, 3]),
        frame_components: BTreeMap::from([(1, 0), (2, 1), (3, 1)]),
        component_count: 2,
    };
    assert!(!connectivity_not_split(&baseline, &split));
    let merged = SupportConnectivity {
        supported_images: BTreeSet::new(),
        supported_frames: BTreeSet::from([1, 2, 3]),
        frame_components: BTreeMap::from([(1, 0), (2, 0), (3, 0)]),
        component_count: 1,
    };
    assert!(connectivity_not_split(&baseline, &merged));
}

#[test]
fn joint_ba_selection_is_deterministic_and_respects_window_caps() {
    let (_manifest, store, images, _cameras, landmarks, active) = joint_ba_fixture();
    let first = joint_ba_window_counts(&landmarks, &images, &active).unwrap();
    let second = joint_ba_window_counts(&landmarks, &images, &active).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.0, landmarks.len());
    assert_eq!(first.1, landmarks.len() * 6);
    assert_eq!(first.2, 3);
    assert_eq!(first.3, 3);
    assert!(store
        .tracks
        .iter()
        .all(|track| track.observations.len() == 6));
    assert!(first.2 <= JOINT_BA_MAX_REFERENCED_FRAMES);
    assert!(first.3 <= JOINT_BA_MAX_FREE_FRAMES);
}

#[test]
fn compact_connectivity_matches_tree_reference_under_deletions_and_permutations() {
    let (_, _, images, _, landmarks, _) = joint_ba_fixture();
    for mask in 0..64usize {
        let mut varied = landmarks.clone();
        for (index, landmark) in varied.iter_mut().enumerate() {
            landmark
                .observations
                .retain(|key| ((key.global_image_id as usize + index) & mask) == 0);
            if mask % 2 == 0 {
                landmark.observations.reverse();
            }
            if let Some(key) = landmark.observations.first().copied() {
                landmark.observations.push(key);
            }
        }
        let removed = varied
            .iter()
            .enumerate()
            .filter(|(i, _)| (i + mask) % 5 == 0)
            .map(|(_, l)| l.track_id)
            .collect::<BTreeSet<_>>();
        let replacements = varied
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == 0)
            .map(|(i, l)| {
                let mut l = l.clone();
                l.observations.truncate(1);
                (i, l)
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            support_connectivity_for_landmarks(&varied, &replacements, &removed, &images),
            support_connectivity_reference(&varied, &replacements, &removed, &images)
        );
    }
    let mut malformed = landmarks.clone();
    malformed[0].observations[0].global_image_id = u64::MAX;
    assert_eq!(
        support_connectivity_for_landmarks(&malformed, &BTreeMap::new(), &BTreeSet::new(), &images),
        support_connectivity_reference(&malformed, &BTreeMap::new(), &BTreeSet::new(), &images)
    );
}

#[test]
fn filter_window_reuses_selection_and_reselects_after_compaction() {
    let (manifest, store, images, cameras, mut landmarks, active) = joint_ba_fixture();
    let selected = selected_landmarks_for_frames(&landmarks, &active, &images).unwrap();
    let counts_from_wrapper = joint_ba_window_counts(&landmarks, &images, &active).unwrap();
    let counts_from_shared =
        joint_ba_window_counts_with_selected(&landmarks, &images, &active, &selected).unwrap();
    assert_eq!(counts_from_wrapper, counts_from_shared);

    let baseline =
        support_connectivity_for_landmarks(&landmarks, &BTreeMap::new(), &BTreeSet::new(), &images)
            .unwrap();
    let wrapper_candidate = build_joint_rig_ba_filter_candidate_with_policy(
        &manifest, &store, &images, &cameras, &landmarks, &active, &baseline, false,
    )
    .unwrap();
    let shared_candidate = build_joint_rig_ba_filter_candidate_with_selected(
        &manifest, &store, &images, &cameras, &landmarks, &active, &selected, &baseline, false,
    )
    .unwrap();
    assert_eq!(wrapper_candidate, shared_candidate);

    let old_len = landmarks.len();
    landmarks.remove(0);
    let selected_after_compaction =
        selected_landmarks_for_frames(&landmarks, &active, &images).unwrap();
    assert_eq!(selected_after_compaction.len(), old_len - 1);
    assert_eq!(
        selected_after_compaction,
        (0..landmarks.len()).collect::<Vec<_>>()
    );
    let compacted_counts = joint_ba_window_counts_with_selected(
        &landmarks,
        &images,
        &active,
        &selected_after_compaction,
    )
    .unwrap();
    assert_eq!(compacted_counts.0, old_len - 1);
}

#[test]
fn joint_ba_synthetic_update_is_transactional_and_preserves_rig_extrinsics() {
    let (manifest, store, mut images, cameras, mut landmarks, active) = joint_ba_fixture();
    let before_images = images.clone();
    let before_landmarks = landmarks.clone();
    let update =
        build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
            .unwrap();
    assert!(update.final_cost <= update.initial_cost + 1.0e-8);
    assert!(update.solver_final_cost <= update.solver_initial_cost + 1.0e-8);
    assert!(update.iterations <= JOINT_BA_MAX_ITERATIONS);
    assert_eq!(update.landmark_updates.len(), landmarks.len());
    apply_pose_overrides(&mut images, &update.pose_overrides).unwrap();
    for (index, candidate) in update.landmark_updates {
        landmarks[index] = candidate;
    }
    assert_ne!(images, before_images);
    assert_ne!(landmarks, before_landmarks);
    for image in images.values() {
        let sensor = &manifest.sensors[&image.atlas.sensor_index];
        let rig_pose = derive_rig_pose_for_frame(&manifest, image.atlas.frame_id, &images).unwrap();
        let expected = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
        assert!(
            (expected.translation - image.atlas.pose.world_to_camera.translation).norm() < 1.0e-8
        );
        assert!(
            (expected.rotation * image.atlas.pose.world_to_camera.rotation.inverse()).angle()
                < 1.0e-8
        );
    }
}

#[test]
fn joint_ba_external_anchors_are_fixed_and_all_track_observations_are_used() {
    let (manifest, store, mut images, cameras, landmarks, _active) = joint_ba_fixture();
    let active = BTreeSet::from([140]);
    let outside_before = images
        .iter()
        .filter(|(_, image)| image.atlas.frame_id != 140)
        .map(|(id, image)| (*id, image.atlas.pose.clone()))
        .collect::<BTreeMap<_, _>>();
    let update =
        build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
            .unwrap();
    assert_eq!(update.selected_observations, landmarks.len() * 6);
    assert_eq!(update.referenced_frames, 3);
    assert_eq!(update.free_frames, 1);
    assert!(update
        .pose_overrides
        .keys()
        .all(|image_id| { images[image_id].atlas.frame_id == 140 }));
    apply_pose_overrides(&mut images, &update.pose_overrides).unwrap();
    for (image_id, pose) in outside_before {
        assert_eq!(images[&image_id].atlas.pose, pose);
    }
}

#[test]
fn joint_ba_repeated_input_is_bitwise_deterministic_and_validator_rejects_mutation() {
    let (manifest, store, images, cameras, landmarks, active) = joint_ba_fixture();
    let first =
        build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
            .unwrap();
    let second =
        build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
            .unwrap();
    assert_eq!(first, second);
    let mut malformed = first.clone();
    let index = *malformed.landmark_updates.keys().next().unwrap();
    malformed
        .landmark_updates
        .get_mut(&index)
        .unwrap()
        .observations
        .pop();
    let error = validate_joint_ba_update(
        &malformed,
        &store,
        &images,
        &cameras,
        &landmarks,
        &selected_landmarks_for_frames(&landmarks, &active, &images).unwrap(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("identity or observations"));
    let mut nan_cost = first.clone();
    nan_cost.initial_cost = f64::NAN;
    let error = validate_joint_ba_update(
        &nan_cost,
        &store,
        &images,
        &cameras,
        &landmarks,
        &selected_landmarks_for_frames(&landmarks, &active, &images).unwrap(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("validated cost"));
}

#[test]
fn joint_ba_rejects_cap_without_truncating_observations() {
    let (manifest, store, images, cameras, mut landmarks, active) = joint_ba_fixture();
    landmarks.extend(
        (12..=JOINT_BA_MAX_LANDMARKS).map(|track_id| LandmarkOutput {
            track_id,
            position: Point3::new(0.0, 0.0, 4.0),
            observations: vec![ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            }],
            errors: vec![0.0],
            rms_error: 0.0,
            mean_error: 0.0,
            max_error: 0.0,
            dlt_sample_count: 1,
        }),
    );
    let error =
        build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
            .unwrap_err();
    assert!(matches!(error, JointRigBaSkip::LandmarkCap { .. }));
    assert_eq!(landmarks.len(), JOINT_BA_MAX_LANDMARKS + 1);
    let selected = selected_landmarks_for_frames(&landmarks, &active, &images).unwrap();
    let baseline =
        support_connectivity_for_landmarks(&landmarks, &BTreeMap::new(), &BTreeSet::new(), &images)
            .unwrap();
    let wrapper_error = build_joint_rig_ba_filter_candidate_with_policy(
        &manifest, &store, &images, &cameras, &landmarks, &active, &baseline, true,
    )
    .unwrap_err();
    let shared_error = build_joint_rig_ba_filter_candidate_with_selected(
        &manifest, &store, &images, &cameras, &landmarks, &active, &selected, &baseline, true,
    )
    .unwrap_err();
    assert_eq!(wrapper_error, shared_error);
}

#[test]
fn joint_ba_rejects_malformed_candidate_without_mutating_inputs() {
    let (manifest, mut store, images, cameras, landmarks, active) = joint_ba_fixture();
    let before_store = store.clone();
    let before_images = images.clone();
    let before_landmarks = landmarks.clone();
    let key = *store.observations.keys().next().unwrap();
    store.observations.get_mut(&key).unwrap().xy.x = f64::NAN;
    let error =
        build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
            .unwrap_err();
    assert!(matches!(error, JointRigBaSkip::Invalid(_)));
    assert_eq!(images, before_images);
    assert_eq!(landmarks, before_landmarks);
    assert_ne!(store, before_store);
}

#[test]
fn merge_ignores_local_point_ids_and_remaps_by_global_image_key() {
    let mut images = BTreeMap::new();
    images.insert(1, test_image(1, "a", vec![Point2::new(1.0, 2.0)]));
    images.insert(2, test_image(2, "b", vec![Point2::new(3.0, 4.0)]));
    let mut store = TrackStore::default();
    let first = vec![
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 0,
        },
        ObservationKey {
            global_image_id: 2,
            keypoint_index: 0,
        },
    ];
    assert_eq!(
        merge_candidate(&mut store, &first, &images),
        Ok(MergeOutcome::New)
    );
    let second = first.clone();
    assert_eq!(
        merge_candidate(&mut store, &second, &images),
        Ok(MergeOutcome::Extended { duplicate_count: 2 })
    );
    assert_eq!(store.tracks.len(), 1);
    assert_eq!(store.observations.len(), 2);
}

#[test]
fn repeated_image_with_changed_xy_is_rejected_before_track_merge() {
    let mut images = BTreeMap::new();
    images.insert(1, test_image(1, "shared", vec![Point2::new(10.0, 20.0)]));
    let mut names = BTreeMap::new();
    names.insert("shared".to_owned(), 1);
    let first = SourceWindow {
        images: vec![SourceImage {
            local_id: 7,
            camera_id: 1,
            name: "shared".to_owned(),
            keypoints: vec![SourceKeypoint {
                xy: Point2::new(10.0, 20.0),
                point3d_id: None,
            }],
        }],
        points: Vec::new(),
    };
    let second = SourceWindow {
        images: vec![SourceImage {
            local_id: 99,
            camera_id: 1,
            name: "shared".to_owned(),
            keypoints: vec![SourceKeypoint {
                xy: Point2::new(10.25, 20.0),
                point3d_id: None,
            }],
        }],
        points: Vec::new(),
    };
    let mut stats = IngestStats::default();
    update_global_images(&mut images, &first, &names, &mut stats).unwrap();
    let error = update_global_images(&mut images, &second, &names, &mut stats).unwrap_err();
    assert!(error.contains("inconsistent ordered keypoint coordinates"));
}

#[test]
fn same_image_different_keypoint_is_rejected_transactionally() {
    let mut images = BTreeMap::new();
    images.insert(
        1,
        test_image(1, "a", vec![Point2::new(1.0, 2.0), Point2::new(3.0, 4.0)]),
    );
    let mut store = TrackStore::default();
    let candidate = vec![
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 0,
        },
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 1,
        },
    ];
    assert_eq!(
        merge_candidate(&mut store, &candidate, &images),
        Err(MergeRejection::SameImageDifferentKeypoint)
    );
    assert!(store.tracks.is_empty());
    assert!(store.observations.is_empty());
}

#[test]
fn conflict_free_crossing_two_existing_tracks_is_unioned_transactionally() {
    let mut images = BTreeMap::new();
    for id in 1..=4 {
        images.insert(
            id,
            test_image(id, &id.to_string(), vec![Point2::new(id as f64, 0.0)]),
        );
    }
    let mut store = TrackStore::default();
    let a = vec![
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 0,
        },
        ObservationKey {
            global_image_id: 2,
            keypoint_index: 0,
        },
    ];
    let b = vec![
        ObservationKey {
            global_image_id: 3,
            keypoint_index: 0,
        },
        ObservationKey {
            global_image_id: 4,
            keypoint_index: 0,
        },
    ];
    assert_eq!(
        merge_candidate(&mut store, &a, &images),
        Ok(MergeOutcome::New)
    );
    assert_eq!(
        merge_candidate(&mut store, &b, &images),
        Ok(MergeOutcome::New)
    );
    let crossing = vec![a[0], b[0]];
    assert_eq!(
        merge_candidate(&mut store, &crossing, &images),
        Ok(MergeOutcome::Extended { duplicate_count: 2 })
    );
    assert_eq!(store.tracks.len(), 2);
    assert_eq!(store.observations.len(), 4);
    assert!(store.tracks[1].observations.is_empty());
    assert_eq!(store.tracks[0].observations.len(), 4);
}

#[test]
fn conflicting_existing_tracks_are_rejected_without_partial_union() {
    let mut images = BTreeMap::new();
    for id in 1..=4 {
        images.insert(
            id,
            test_image(
                id,
                &id.to_string(),
                vec![
                    Point2::new(id as f64, 0.0),
                    Point2::new(id as f64 + 0.5, 0.0),
                ],
            ),
        );
    }
    let mut store = TrackStore::default();
    let a = vec![
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 0,
        },
        ObservationKey {
            global_image_id: 2,
            keypoint_index: 0,
        },
    ];
    let b = vec![
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 1,
        },
        ObservationKey {
            global_image_id: 3,
            keypoint_index: 0,
        },
    ];
    assert_eq!(
        merge_candidate(&mut store, &a, &images),
        Ok(MergeOutcome::New)
    );
    assert_eq!(
        merge_candidate(&mut store, &b, &images),
        Ok(MergeOutcome::New)
    );
    let bridge = vec![a[1], b[1]];
    assert_eq!(
        merge_candidate(&mut store, &bridge, &images),
        Err(MergeRejection::ExistingTrackSameImageConflict)
    );
    assert_eq!(store.tracks.len(), 2);
    assert_eq!(store.tracks[0].observations.len(), 2);
    assert_eq!(store.tracks[1].observations.len(), 2);
}

#[test]
fn dlt_retriangulates_final_pose_and_reports_actual_error() {
    let camera = test_camera();
    let point = Point3::new(0.2, -0.1, 4.0);
    let pose_a = Pose::identity();
    let pose_b =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-1.0, 0.0, 0.0));
    let mut images = BTreeMap::new();
    images.insert(
        1,
        GlobalImage {
            atlas: AtlasImage {
                global_image_id: 1,
                frame_id: 1,
                sensor_index: 0,
                name: "a".to_owned(),
                camera_id: 1,
                pose: pose_a.clone(),
            },
            keypoints: vec![],
        },
    );
    images.insert(
        2,
        GlobalImage {
            atlas: AtlasImage {
                global_image_id: 2,
                frame_id: 2,
                sensor_index: 0,
                name: "b".to_owned(),
                camera_id: 1,
                pose: pose_b.clone(),
            },
            keypoints: vec![],
        },
    );
    let pose_c =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, -0.75, 0.0));
    images.insert(
        3,
        GlobalImage {
            atlas: AtlasImage {
                global_image_id: 3,
                frame_id: 3,
                sensor_index: 0,
                name: "c".to_owned(),
                camera_id: 1,
                pose: pose_c.clone(),
            },
            keypoints: vec![],
        },
    );
    let xy_a = camera
        .project(&pose_a.transform_world_point(&point))
        .unwrap();
    let xy_b = camera
        .project(&pose_b.transform_world_point(&point))
        .unwrap();
    let xy_c = camera
        .project(&pose_c.transform_world_point(&point))
        .unwrap();
    let keys = vec![
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 0,
        },
        ObservationKey {
            global_image_id: 2,
            keypoint_index: 0,
        },
        ObservationKey {
            global_image_id: 3,
            keypoint_index: 0,
        },
    ];
    let mut observations = BTreeMap::new();
    observations.insert(
        keys[0],
        ObservationState {
            xy: xy_a,
            owner_track: 0,
        },
    );
    observations.insert(
        keys[1],
        ObservationState {
            xy: Point2::new(xy_b.x + 0.25, xy_b.y),
            owner_track: 0,
        },
    );
    observations.insert(
        keys[2],
        ObservationState {
            xy: xy_c,
            owner_track: 0,
        },
    );
    let mut cameras = BTreeMap::new();
    cameras.insert(1, camera);
    let track = GlobalTrack { observations: keys };
    let landmark = triangulate_track(0, &track, &observations, &images, &cameras).unwrap();
    assert!(landmark.rms_error > 0.0);
    assert!(landmark.max_error < MAX_REPROJECTION_PX);
}

#[test]
fn coincident_camera_centres_are_rejected_even_with_finite_zero_error() {
    let camera = test_camera();
    let point = Point3::new(0.0, 0.0, 4.0);
    let pose = Pose::identity();
    let mut images = BTreeMap::new();
    for id in 1..=2 {
        images.insert(
            id,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: id,
                    frame_id: id,
                    sensor_index: 0,
                    name: id.to_string(),
                    camera_id: 1,
                    pose: pose.clone(),
                },
                keypoints: vec![],
            },
        );
    }
    let xy = camera.project(&pose.transform_world_point(&point)).unwrap();
    let keys = vec![
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 0,
        },
        ObservationKey {
            global_image_id: 2,
            keypoint_index: 0,
        },
    ];
    let observations = keys
        .iter()
        .map(|key| (*key, ObservationState { xy, owner_track: 0 }))
        .collect::<BTreeMap<_, _>>();
    let mut cameras = BTreeMap::new();
    cameras.insert(1, camera);
    let error = triangulate_track(
        0,
        &GlobalTrack { observations: keys },
        &observations,
        &images,
        &cameras,
    )
    .unwrap_err();
    assert!(error.contains("no observable camera baseline"));
}

#[test]
fn mismatched_sensor_poses_for_one_rig_frame_are_rejected() {
    let sensor = |camera_id| SensorCalibration {
        camera_id,
        width: 640,
        height: 480,
        fx: 100.0,
        fy: 100.0,
        cx: 320.0,
        cy: 240.0,
        sensor_from_rig: SE3::identity(),
    };
    let mut sensors = BTreeMap::new();
    sensors.insert(0, sensor(1));
    sensors.insert(1, sensor(1));
    let mut assignments = BTreeMap::new();
    assignments.insert(
        "a".to_owned(),
        ImageAssignment {
            frame_id: 7,
            sensor_index: 0,
            global_image_id: 1,
        },
    );
    assignments.insert(
        "b".to_owned(),
        ImageAssignment {
            frame_id: 7,
            sensor_index: 1,
            global_image_id: 2,
        },
    );
    let manifest = RigManifest {
        sensors,
        assignments,
    };
    let mut images = BTreeMap::new();
    let mut first = test_image(1, "a", vec![Point2::new(320.0, 240.0)]);
    first.atlas.frame_id = 7;
    images.insert(1, first);
    images.insert(
        2,
        GlobalImage {
            atlas: AtlasImage {
                global_image_id: 2,
                frame_id: 7,
                sensor_index: 1,
                name: "b".to_owned(),
                camera_id: 1,
                pose: Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(0.01, 0.0, 0.0),
                ),
            },
            keypoints: vec![Point2::new(320.0, 240.0)],
        },
    );
    let mut cameras = BTreeMap::new();
    cameras.insert(1, test_camera());
    let error = validate_output_cameras(&manifest, &images, &cameras).unwrap_err();
    assert!(error.contains("sensor poses disagree"));
}

#[test]
fn bounded_sample_is_sorted_endpoint_inclusive_and_deterministic() {
    let keys = (0..100)
        .map(|index| ObservationKey {
            global_image_id: index,
            keypoint_index: 0,
        })
        .collect::<Vec<_>>();
    let sample = bounded_sample(&keys, 7);
    assert_eq!(sample.len(), 7);
    assert_eq!(sample.first(), keys.first());
    assert_eq!(sample.last(), keys.last());
    assert_eq!(sample, bounded_sample(&keys, 7));
}

#[test]
fn recovery_anchor_filter_never_uses_another_unsupported_frame() {
    let mut images = BTreeMap::new();
    let mut supported = test_image(1, "supported", vec![Point2::new(1.0, 1.0)]);
    supported.atlas.frame_id = 10;
    images.insert(1, supported);
    let mut target = test_image(2, "target", vec![Point2::new(2.0, 2.0)]);
    target.atlas.frame_id = 20;
    images.insert(2, target);
    let mut other_unsupported = test_image(3, "other", vec![Point2::new(3.0, 3.0)]);
    other_unsupported.atlas.frame_id = 30;
    images.insert(3, other_unsupported);
    let track = GlobalTrack {
        observations: vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 3,
                keypoint_index: 0,
            },
        ],
    };
    let supported_frames = BTreeSet::from([10]);
    let keys = recovery_track_keys(&track, 20, &supported_frames, &images);
    assert_eq!(
        keys,
        vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ]
    );
}

#[test]
fn recovery_synthetic_end_to_end_accepts_six_target_landmarks() {
    let (manifest, store, images, cameras, baseline, truth_rig_pose) = recovery_fixture(false);
    let result =
        recover_zero_support_frames(&manifest, &store, &images, &cameras, &baseline).unwrap();
    assert_eq!(result.summary.frames_recovered, 1);
    assert_eq!(result.summary.pnp_inliers, 6);
    assert_eq!(result.summary.candidate_target_landmarks, 6);
    assert_eq!(result.summary.accepted_target_landmarks, 6);
    assert_eq!(result.landmarks.len(), 6);
    assert_eq!(result.pose_overrides.len(), 2);
    for (global_image_id, sensor) in &manifest.sensors {
        let image_id = *global_image_id as u64 + 3;
        let expected = sensor
            .sensor_from_rig
            .compose(&truth_rig_pose.world_to_camera);
        let actual = &result.pose_overrides[&image_id].world_to_camera;
        assert!((actual.translation - expected.translation).norm() < 1.0e-8);
        assert!((actual.rotation * expected.rotation.inverse()).angle() < 1.0e-8);
    }
}

#[test]
fn recovery_transactionally_rejects_pose_when_only_five_target_tracks_survive() {
    let (manifest, store, images, cameras, baseline, _) = recovery_fixture(true);
    let result =
        recover_zero_support_frames(&manifest, &store, &images, &cameras, &baseline).unwrap();
    assert!(result.summary.pnp_inliers >= RECOVERY_MIN_PNP_INLIERS);
    assert_eq!(result.summary.candidate_target_landmarks, 5);
    assert_eq!(result.summary.accepted_target_landmarks, 0);
    assert!(result.pose_overrides.is_empty());
    assert!(result.landmarks.is_empty());
}

#[test]
fn recovered_frame_pose_recomposes_both_fixed_sensor_extrinsics() {
    let sensor0 = SensorCalibration {
        camera_id: 1,
        width: 640,
        height: 480,
        fx: 100.0,
        fy: 101.0,
        cx: 320.0,
        cy: 240.0,
        sensor_from_rig: SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.03)),
            Vector3::new(0.1, -0.02, 0.04),
        ),
    };
    let sensor1 = SensorCalibration {
        camera_id: 2,
        width: 640,
        height: 480,
        fx: 102.0,
        fy: 99.0,
        cx: 321.0,
        cy: 239.0,
        sensor_from_rig: SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.03, 0.01, 0.02)),
            Vector3::new(-0.2, 0.03, 0.01),
        ),
    };
    let manifest = RigManifest {
        sensors: BTreeMap::from([(0, sensor0.clone()), (1, sensor1.clone())]),
        assignments: BTreeMap::new(),
    };
    let mut images = BTreeMap::new();
    for (global_image_id, (sensor_index, camera_id)) in [(1, (0, 1)), (2, (1, 2))] {
        images.insert(
            global_image_id,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id,
                    frame_id: 7,
                    sensor_index,
                    name: global_image_id.to_string(),
                    camera_id,
                    pose: Pose::identity(),
                },
                keypoints: vec![],
            },
        );
    }
    let rig_pose = Pose::from_world_to_camera(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.1, 0.05)),
        Vector3::new(1.0, -0.4, 0.7),
    );
    let mut overrides = BTreeMap::new();
    compose_recovered_frame_poses(&manifest, 7, &rig_pose, &images, &mut overrides).unwrap();
    assert_eq!(overrides.len(), 2);
    for (sensor_index, calibration) in [(0, sensor0), (1, sensor1)] {
        let expected = calibration
            .sensor_from_rig
            .compose(&rig_pose.world_to_camera);
        let actual = &overrides[&(sensor_index as u64 + 1)].world_to_camera;
        assert!((actual.translation - expected.translation).norm() < 1.0e-12);
        assert!((actual.rotation * expected.rotation.inverse()).angle() < 1.0e-12);
    }
    assert!(images
        .values()
        .all(|image| image.atlas.pose == Pose::identity()));
}

#[test]
fn multiple_atlas_components_are_rejected_as_independent_gauges() {
    let root = std::env::temp_dir().join(format!(
        "visloc_integrate_landmark_components_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("component-000")).unwrap();
    fs::create_dir_all(root.join("component-001")).unwrap();
    fs::write(root.join("component-000/images.txt"), "").unwrap();
    fs::write(root.join("component-001/images.txt"), "").unwrap();
    let manifest = RigManifest {
        sensors: BTreeMap::new(),
        assignments: BTreeMap::new(),
    };
    let error = parse_atlas_images(&root, &manifest).unwrap_err();
    assert!(error.contains("independent component images.txt"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn empty_landmark_set_cannot_be_published_as_pose_only_model() {
    let error = require_nonempty_landmarks(&[], 0).unwrap_err();
    assert!(error.contains("refusing to publish a pose-only model"));
}

#[test]
fn actual_error_and_bidirectional_track_are_written() {
    let root = std::env::temp_dir().join(format!(
        "visloc_integrate_landmark_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let camera = test_camera();
    let pose = Pose::identity();
    let xy = camera.project(&Point3::new(0.0, 0.0, 4.0)).unwrap();
    let mut images = BTreeMap::new();
    images.insert(
        1,
        GlobalImage {
            atlas: AtlasImage {
                global_image_id: 1,
                frame_id: 1,
                sensor_index: 0,
                name: "a.png".to_owned(),
                camera_id: 1,
                pose,
            },
            keypoints: vec![xy],
        },
    );
    images.insert(
        2,
        GlobalImage {
            atlas: AtlasImage {
                global_image_id: 2,
                frame_id: 2,
                sensor_index: 0,
                name: "b.png".to_owned(),
                camera_id: 1,
                pose: Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(-0.5, 0.0, 0.0),
                ),
            },
            keypoints: vec![Point2::new(320.0, 240.0)],
        },
    );
    let mut cameras = BTreeMap::new();
    cameras.insert(1, camera);
    let landmark = LandmarkOutput {
        track_id: 0,
        position: Point3::new(0.0, 0.0, 4.0),
        observations: vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ],
        errors: vec![0.1, 0.3],
        rms_error: (0.05_f64).sqrt(),
        mean_error: 0.2,
        max_error: 0.3,
        dlt_sample_count: 2,
    };
    let mut stats = IngestStats::default();
    stats.output_landmarks = 1;
    stats.output_observations = 2;
    write_model(&root, &images, &cameras, &[landmark], &stats).unwrap();
    let points = fs::read_to_string(root.join("points3D.txt")).unwrap();
    let image = fs::read_to_string(root.join("images.txt")).unwrap();
    assert!(points.contains(" 0.2 1 0 2 0"));
    assert!(image.lines().any(|line| line.contains("320 240 1")));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn diagnostic_cross_boundary_partition_is_deterministic() {
    let mut images = BTreeMap::new();
    images.insert(1, test_image(1, "a", vec![Point2::new(1.0, 1.0)]));
    images.insert(2, test_image(2, "b", vec![Point2::new(2.0, 2.0)]));
    images.insert(3, test_image(3, "c", vec![Point2::new(3.0, 3.0)]));
    let store = TrackStore {
        observations: BTreeMap::new(),
        tracks: vec![GlobalTrack {
            observations: vec![
                ObservationKey {
                    global_image_id: 3,
                    keypoint_index: 0,
                },
                ObservationKey {
                    global_image_id: 1,
                    keypoint_index: 0,
                },
                ObservationKey {
                    global_image_id: 2,
                    keypoint_index: 0,
                },
            ],
        }],
    };
    let first = collect_diagnostic_cross_tracks(&store, &images, 1).unwrap();
    let second = collect_diagnostic_cross_tracks(&store, &images, 1).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].left_keys.len(), 1);
    assert_eq!(first[0].right_keys.len(), 2);
    assert_eq!(first[0].left_keys[0].global_image_id, 1);
    assert_eq!(
        first[0]
            .right_keys
            .iter()
            .map(|key| key.global_image_id)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn diagnostic_inliers_count_distinct_tracks_not_observations() {
    let make = |track_id, image_id| DiagnosticCandidateObservation {
        track_id,
        key: ObservationKey {
            global_image_id: image_id,
            keypoint_index: 0,
        },
        correspondence: GeneralizedCorrespondence2D3D {
            sensor_index: 0,
            point2d: Point2::new(0.0, 0.0),
            point3d: Point3::new(0.0, 0.0, 1.0),
            confidence: None,
        },
    };
    let candidates = vec![make(4, 1), make(4, 2), make(8, 3), make(9, 4)];
    assert_eq!(diagnostic_distinct_track_ids(&candidates).len(), 3);
    assert_eq!(
        diagnostic_inlier_track_ids(&[0, 1, 2], &candidates).len(),
        2
    );
    assert_eq!(
        diagnostic_inlier_track_ids(&[0, 2, 3], &candidates).len(),
        3
    );
}

#[test]
fn diagnostic_pose_round_trip_preserves_fixed_sensor_extrinsics() {
    let (manifest, _store, images, cameras, _baseline, _) = recovery_fixture(false);
    let rig_pose = Pose::from_world_to_camera(
        UnitQuaternion::from_scaled_axis(Vector3::new(0.03, -0.02, 0.01)),
        Vector3::new(0.2, -0.1, 0.4),
    );
    assert_eq!(
        validate_diagnostic_rig_pose(&manifest, 10, &images, &rig_pose).unwrap(),
        2
    );
    let rig = build_generalized_rig(&manifest, &cameras).unwrap();
    assert_eq!(rig.sensors().len(), 2);
}

#[test]
fn diagnostic_target_track_deduplicates_shared_observations() {
    let left = vec![ObservationKey {
        global_image_id: 1,
        keypoint_index: 3,
    }];
    let right = vec![
        ObservationKey {
            global_image_id: 2,
            keypoint_index: 3,
        },
        ObservationKey {
            global_image_id: 1,
            keypoint_index: 3,
        },
    ];
    let track = diagnostic_track_with_keys(&left, &right);
    assert_eq!(track.observations.len(), 2);
    assert_eq!(track.observations[0], left[0]);
    assert_eq!(track.observations[1], right[0]);
}

#[test]
fn diagnostic_all_supported_frames_can_remain_disconnected() {
    let (sizes, supported) =
        diagnostic_component_sizes_from_frame_tracks(&[0, 1, 2, 3], &[vec![0, 1], vec![2, 3]]);
    assert_eq!(supported, BTreeSet::from([0, 1, 2, 3]));
    assert_eq!(sizes, vec![2, 2]);
}

#[test]
fn boundary_repair_acceptance_requires_strict_connectivity_and_support_subset() {
    let baseline = DiagnosticConnectivityReport {
        supported_images: BTreeSet::from([1, 2]),
        supported_frames: BTreeSet::from([10, 30]),
        supported_component_sizes: vec![1, 1],
        removed_tracks: 0,
        removed_observations: 0,
        added_tracks: 0,
        added_observations: 0,
    };
    let mut candidate = baseline.clone();
    candidate.supported_images.insert(3);
    candidate.supported_frames.insert(40);
    candidate.supported_component_sizes = vec![3];
    assert!(boundary_repair_candidate_is_accepted(
        &baseline, &candidate, 1
    ));
    candidate.supported_frames.remove(&10);
    assert!(!boundary_repair_candidate_is_accepted(
        &baseline, &candidate, 1
    ));
    candidate.supported_frames.insert(10);
    candidate.supported_component_sizes = vec![1, 2];
    assert!(!boundary_repair_candidate_is_accepted(
        &baseline, &candidate, 1
    ));
    assert!(!boundary_repair_candidate_is_accepted(
        &baseline, &candidate, 0
    ));
}

#[test]
fn boundary_repair_accepts_first_candidate_transactionally_and_preserves_extrinsics() {
    let (manifest, store, mut images, cameras, mut landmarks) = boundary_repair_fixture();
    let before_images = images.clone();
    let before_landmarks = landmarks.clone();
    let first = repair_cross_boundary(&manifest, 10, &store, &mut images, &cameras, &mut landmarks)
        .unwrap();
    assert_eq!(first.status, "accepted");
    assert_eq!(first.accepted_frame, Some(30));
    assert_eq!(first.added_tracks, 6);
    assert_eq!(first.removed_tracks, 0);
    assert_eq!(first.baseline_component_count, 2);
    assert_eq!(first.candidate_component_count, 1);
    assert_ne!(images, before_images);
    assert_ne!(landmarks, before_landmarks);
    assert_eq!(landmarks.len(), 8);
    assert_eq!(
        validate_diagnostic_rig_pose(&manifest, 30, &images, &Pose::identity()),
        Ok(2)
    );
    for image in images.values() {
        let sensor = &manifest.sensors[&image.atlas.sensor_index];
        let rig_pose = derive_rig_pose_for_frame(&manifest, image.atlas.frame_id, &images).unwrap();
        let expected = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
        assert!(
            (expected.translation - image.atlas.pose.world_to_camera.translation).norm() < 1.0e-8
        );
        assert!(
            (expected.rotation * image.atlas.pose.world_to_camera.rotation.inverse()).angle()
                < 1.0e-8
        );
    }

    let (manifest_again, store_again, mut images_again, cameras_again, mut landmarks_again) =
        boundary_repair_fixture();
    let second = repair_cross_boundary(
        &manifest_again,
        10,
        &store_again,
        &mut images_again,
        &cameras_again,
        &mut landmarks_again,
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(images, images_again);
    assert_eq!(landmarks, landmarks_again);
}

#[test]
fn boundary_repair_rejection_is_transactional_when_candidate_is_underconstrained() {
    let (manifest, mut store, mut images, cameras, mut landmarks) = boundary_repair_fixture();
    store.tracks.truncate(7);
    let before_images = images.clone();
    let before_landmarks = landmarks.clone();
    let summary =
        repair_cross_boundary(&manifest, 10, &store, &mut images, &cameras, &mut landmarks)
            .unwrap();
    assert_eq!(summary.status, "no-accepted-candidate");
    assert_eq!(summary.accepted_frame, None);
    assert_eq!(images, before_images);
    assert_eq!(landmarks, before_landmarks);
}
