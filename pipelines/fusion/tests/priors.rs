use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::Frame;
use visloc_fusion::{
    FramePriorSource, FrameTimestampIndex, GnssMeasurement, ImuMeasurement,
    LocalizationPriorProvider, MeasurementBuffer, PoseCovariance, PosePriorMeasurement,
    PositionCovariance, PriorConfig, TimeDelta, Timed, TimedFrame, TimedMeasurement, TimedPose,
    Timestamp,
};

#[test]
fn timestamp_tracks_nanoseconds_and_duration() {
    let first = Timestamp::from_seconds_nanoseconds(10, 250);
    let second = Timestamp::from_nanoseconds(12_000_000_500);

    let delta = second.duration_since(first).unwrap();

    assert_eq!(first.as_nanoseconds(), 10_000_000_250);
    assert_eq!(delta.as_nanoseconds(), 2_000_000_250);
    assert!((delta.as_seconds_f64() - 2.00000025).abs() < 1.0e-12);
    assert!(first.duration_since(second).is_none());
}

#[test]
fn timed_wraps_measurements() {
    let timestamp = Timestamp::from_nanoseconds(42);
    let timed = Timed::new(timestamp, "frame_42");

    assert_eq!(timed.timestamp, timestamp);
    assert_eq!(timed.timestamp(), timestamp);
    assert_eq!(timed.value, "frame_42");
}

#[test]
fn timed_frame_and_pose_aliases_keep_existing_core_types_timestamped() {
    let frame = Frame::new(7, 3);
    let timed_frame: TimedFrame = Timed::new(Timestamp::from_nanoseconds(700), frame);
    let pose = Pose::identity();
    let timed_pose: TimedPose = Timed::new(Timestamp::from_nanoseconds(800), pose);

    assert_eq!(timed_frame.timestamp(), Timestamp::from_nanoseconds(700));
    assert_eq!(timed_frame.value.id, 7);
    assert_eq!(timed_pose.timestamp(), Timestamp::from_nanoseconds(800));
    assert_eq!(timed_pose.value, Pose::identity());
}

#[test]
fn frame_timestamp_index_maps_core_frames_to_sensor_time() {
    let frame_a = Frame::new(10, 1);
    let frame_b = Frame::new(11, 1);
    let mut index = FrameTimestampIndex::new();

    assert!(index.is_empty());
    index.insert_frame(&frame_a, Timestamp::from_nanoseconds(1_000));
    index.insert_frame_id(frame_b.id, Timestamp::from_nanoseconds(2_000));

    assert_eq!(index.len(), 2);
    assert_eq!(
        index.timestamp_for_frame(&frame_a),
        Some(Timestamp::from_nanoseconds(1_000))
    );
    assert_eq!(
        index.timestamp_for_frame_id(frame_b.id),
        Some(Timestamp::from_nanoseconds(2_000))
    );
    assert_eq!(index.timestamp_for_frame_id(99), None);
}

#[test]
fn frame_timestamp_index_can_be_built_from_timed_frames() {
    let index = FrameTimestampIndex::from_timed_frames([
        Timed::new(Timestamp::from_nanoseconds(20), Frame::new(2, 1)),
        Timed::new(Timestamp::from_nanoseconds(10), Frame::new(1, 1)),
    ]);

    assert_eq!(
        index.timestamp_for_frame_id(1),
        Some(Timestamp::from_nanoseconds(10))
    );
    assert_eq!(
        index.timestamp_for_frame_id(2),
        Some(Timestamp::from_nanoseconds(20))
    );
}

#[test]
fn gnss_measurement_builds_position_localization_prior() {
    let config = PriorConfig {
        default_radius: 100.0,
        min_radius: 5.0,
        confidence_multiplier: 2.0,
    };
    let gnss = GnssMeasurement::new(
        Timestamp::from_nanoseconds(1),
        Point3::new(10.0, 20.0, 30.0),
    )
    .with_accuracy(Some(4.0), Some(10.0));

    let prior = gnss.localization_prior(&config).unwrap();

    assert_eq!(gnss.timestamp(), Timestamp::from_nanoseconds(1));
    assert_eq!(prior.position_world, Some(Point3::new(10.0, 20.0, 30.0)));
    assert_eq!(prior.radius, Some(20.0));
    assert_eq!(prior.center_world(), Some(Point3::new(10.0, 20.0, 30.0)));
}

#[test]
fn gnss_measurement_uses_default_and_min_radius() {
    let config = PriorConfig {
        default_radius: 3.0,
        min_radius: 5.0,
        confidence_multiplier: 2.0,
    };
    let gnss = GnssMeasurement::new(Timestamp::from_nanoseconds(1), Point3::origin());

    assert_eq!(gnss.search_radius(&config), 5.0);
}

#[test]
fn position_covariance_reports_axis_standard_deviations() {
    let covariance = PositionCovariance::from_standard_deviations(Vector3::new(2.0, 3.0, 4.0));

    assert_eq!(covariance.horizontal_standard_deviation(), Some(3.0));
    assert_eq!(covariance.vertical_standard_deviation(), Some(4.0));
    assert_eq!(covariance.max_standard_deviation(), Some(4.0));
}

#[test]
fn gnss_measurement_can_derive_radius_from_position_covariance() {
    let config = PriorConfig {
        default_radius: 100.0,
        min_radius: 1.0,
        confidence_multiplier: 2.0,
    };
    let gnss = GnssMeasurement::new(Timestamp::from_nanoseconds(1), Point3::origin())
        .with_position_covariance(PositionCovariance::from_standard_deviations(Vector3::new(
            1.0, 4.0, 2.0,
        )));

    assert_eq!(gnss.search_radius(&config), 8.0);
}

#[test]
fn pose_prior_measurement_builds_pose_localization_prior() {
    let config = PriorConfig::default();
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-1.0, 0.0, 0.0));
    let measurement =
        PosePriorMeasurement::new(Timestamp::from_nanoseconds(7), pose).with_translation_sigma(2.0);

    let prior = measurement.localization_prior(&config).unwrap();

    assert_eq!(measurement.timestamp(), Timestamp::from_nanoseconds(7));
    assert!(prior.pose.is_some());
    assert_eq!(prior.radius, Some(6.0));
    assert_eq!(prior.center_world(), Some(Point3::new(1.0, 0.0, 0.0)));
}

#[test]
fn pose_covariance_reports_translation_and_rotation_uncertainty() {
    let covariance = PoseCovariance::from_translation_rotation_standard_deviations(
        Vector3::new(1.0, 2.0, 3.0),
        Vector3::new(0.01, 0.02, 0.03),
    );

    assert_eq!(covariance.max_translation_standard_deviation(), Some(3.0));
    assert_eq!(covariance.max_rotation_standard_deviation(), Some(0.03));
}

#[test]
fn pose_prior_measurement_can_derive_radius_from_pose_covariance() {
    let config = PriorConfig {
        default_radius: 50.0,
        min_radius: 1.0,
        confidence_multiplier: 3.0,
    };
    let measurement = PosePriorMeasurement::new(Timestamp::from_nanoseconds(2), Pose::identity())
        .with_pose_covariance(
            PoseCovariance::from_translation_rotation_standard_deviations(
                Vector3::new(1.0, 2.0, 4.0),
                Vector3::zeros(),
            ),
        );

    assert_eq!(measurement.search_radius(&config), 12.0);
}

#[test]
fn imu_measurement_keeps_optional_orientation() {
    let imu = ImuMeasurement::new(
        Timestamp::from_nanoseconds(3),
        Vector3::new(0.1, 0.2, 0.3),
        Vector3::new(0.0, 0.0, 9.81),
    )
    .with_orientation(UnitQuaternion::identity());

    assert_eq!(imu.timestamp(), Timestamp::from_nanoseconds(3));
    assert!(imu.orientation.is_some());
}

#[test]
fn measurement_buffer_orders_measurements_and_finds_latest_before_timestamp() {
    let mut buffer = MeasurementBuffer::new();
    buffer.push(GnssMeasurement::new(
        Timestamp::from_nanoseconds(30),
        Point3::new(30.0, 0.0, 0.0),
    ));
    buffer.push(GnssMeasurement::new(
        Timestamp::from_nanoseconds(10),
        Point3::new(10.0, 0.0, 0.0),
    ));
    buffer.push(GnssMeasurement::new(
        Timestamp::from_nanoseconds(20),
        Point3::new(20.0, 0.0, 0.0),
    ));

    let ordered_timestamps = buffer
        .iter()
        .map(TimedMeasurement::timestamp)
        .collect::<Vec<_>>();
    let latest = buffer
        .latest_before_or_at(Timestamp::from_nanoseconds(25))
        .unwrap();

    assert_eq!(
        ordered_timestamps,
        vec![
            Timestamp::from_nanoseconds(10),
            Timestamp::from_nanoseconds(20),
            Timestamp::from_nanoseconds(30)
        ]
    );
    assert_eq!(buffer.len(), 3);
    assert_eq!(latest.position_world, Point3::new(20.0, 0.0, 0.0));
}

#[test]
fn measurement_buffer_finds_nearest_measurement_with_tolerance() {
    let buffer = MeasurementBuffer::from_measurements([
        ImuMeasurement::new(
            Timestamp::from_nanoseconds(100),
            Vector3::zeros(),
            Vector3::zeros(),
        ),
        ImuMeasurement::new(
            Timestamp::from_nanoseconds(130),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::zeros(),
        ),
    ]);

    let nearest = buffer
        .nearest(
            Timestamp::from_nanoseconds(121),
            TimeDelta::from_nanoseconds(10),
        )
        .unwrap();

    assert_eq!(nearest.timestamp(), Timestamp::from_nanoseconds(130));
    assert!(buffer
        .nearest(
            Timestamp::from_nanoseconds(121),
            TimeDelta::from_nanoseconds(8)
        )
        .is_none());
}

#[test]
fn measurement_buffer_builds_nearest_localization_prior() {
    let config = PriorConfig {
        default_radius: 50.0,
        min_radius: 1.0,
        confidence_multiplier: 2.0,
    };
    let buffer = MeasurementBuffer::from_measurements([
        GnssMeasurement::new(Timestamp::from_nanoseconds(100), Point3::new(1.0, 0.0, 0.0))
            .with_accuracy(Some(3.0), None),
        GnssMeasurement::new(Timestamp::from_nanoseconds(150), Point3::new(2.0, 0.0, 0.0))
            .with_accuracy(Some(5.0), None),
    ]);

    let prior = buffer
        .nearest_localization_prior(
            Timestamp::from_nanoseconds(148),
            TimeDelta::from_nanoseconds(5),
            &config,
        )
        .unwrap();

    assert_eq!(prior.position_world, Some(Point3::new(2.0, 0.0, 0.0)));
    assert_eq!(prior.radius, Some(10.0));
}

#[test]
fn measurement_buffer_finds_nearest_measurement_for_frame_timestamp() {
    let frame = Frame::new(42, 1);
    let mut frame_timestamps = FrameTimestampIndex::new();
    frame_timestamps.insert_frame(&frame, Timestamp::from_nanoseconds(1_020));
    let buffer = MeasurementBuffer::from_measurements([
        ImuMeasurement::new(
            Timestamp::from_nanoseconds(1_000),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::zeros(),
        ),
        ImuMeasurement::new(
            Timestamp::from_nanoseconds(1_025),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::zeros(),
        ),
    ]);

    let nearest = buffer
        .nearest_for_frame(&frame, &frame_timestamps, TimeDelta::from_nanoseconds(10))
        .unwrap();

    assert_eq!(nearest.timestamp(), Timestamp::from_nanoseconds(1_025));
    assert!(buffer
        .nearest_for_frame_id(99, &frame_timestamps, TimeDelta::from_nanoseconds(10))
        .is_none());
}

#[test]
fn measurement_buffer_builds_localization_prior_for_frame_timestamp() {
    let frame = Frame::new(5, 1);
    let frame_timestamps = FrameTimestampIndex::from_timed_frames([Timed::new(
        Timestamp::from_nanoseconds(2_005),
        frame.clone(),
    )]);
    let config = PriorConfig {
        default_radius: 100.0,
        min_radius: 1.0,
        confidence_multiplier: 3.0,
    };
    let gnss_buffer = MeasurementBuffer::from_measurements([
        GnssMeasurement::new(
            Timestamp::from_nanoseconds(1_000),
            Point3::new(1.0, 0.0, 0.0),
        )
        .with_accuracy(Some(2.0), None),
        GnssMeasurement::new(
            Timestamp::from_nanoseconds(2_000),
            Point3::new(2.0, 0.0, 0.0),
        )
        .with_accuracy(Some(4.0), None),
    ]);

    let prior = gnss_buffer
        .nearest_localization_prior_for_frame(
            &frame,
            &frame_timestamps,
            TimeDelta::from_nanoseconds(10),
            &config,
        )
        .unwrap();
    let prior_by_id = gnss_buffer
        .nearest_localization_prior_for_frame_id(
            frame.id,
            &frame_timestamps,
            TimeDelta::from_nanoseconds(10),
            &config,
        )
        .unwrap();

    assert_eq!(prior.position_world, Some(Point3::new(2.0, 0.0, 0.0)));
    assert_eq!(prior.radius, Some(12.0));
    assert_eq!(prior_by_id, prior);
}

#[test]
fn frame_prior_source_packages_frame_timestamps_measurements_and_prior_config() {
    let frame = Frame::new(7, 1);
    let frame_timestamps = FrameTimestampIndex::from_timed_frames([Timed::new(
        Timestamp::from_nanoseconds(5_005),
        frame.clone(),
    )]);
    let measurements = MeasurementBuffer::from_measurements([
        GnssMeasurement::new(
            Timestamp::from_nanoseconds(4_000),
            Point3::new(1.0, 0.0, 0.0),
        )
        .with_accuracy(Some(2.0), None),
        GnssMeasurement::new(
            Timestamp::from_nanoseconds(5_000),
            Point3::new(5.0, 0.0, 0.0),
        )
        .with_accuracy(Some(3.0), None),
    ]);
    let source = FramePriorSource::new(
        frame_timestamps,
        measurements,
        TimeDelta::from_nanoseconds(10),
    )
    .with_prior_config(PriorConfig {
        default_radius: 100.0,
        min_radius: 1.0,
        confidence_multiplier: 2.0,
    });

    let nearest = source.nearest_measurement_for_frame(&frame).unwrap();
    let prior = source.localization_prior_for_frame(&frame).unwrap();
    let prior_by_id = source.localization_prior_for_frame_id(frame.id).unwrap();

    assert_eq!(source.frame_count(), 1);
    assert_eq!(source.measurement_count(), 2);
    assert_eq!(
        source.timestamp_for_frame(&frame),
        Some(Timestamp::from_nanoseconds(5_005))
    );
    assert_eq!(nearest.timestamp(), Timestamp::from_nanoseconds(5_000));
    assert_eq!(prior.position_world, Some(Point3::new(5.0, 0.0, 0.0)));
    assert_eq!(prior.radius, Some(6.0));
    assert_eq!(prior_by_id, prior);
}
