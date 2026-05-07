use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_fusion::{
    GnssMeasurement, ImuMeasurement, LocalizationPriorProvider, PosePriorMeasurement, PriorConfig,
    Timed, TimedMeasurement, Timestamp,
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
    assert_eq!(timed.value, "frame_42");
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
