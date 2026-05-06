use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    FramePriorSource, FrameTimestampIndex, GnssMeasurement, InMemoryMapProvider,
    LocalizationPipeline, MeasurementBuffer, PriorConfig, TimeDelta, Timed, Timestamp, Tracker,
    TrackingConfig,
};

fn main() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let near_points = [
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());

    let mut frame_a = Frame::new(100, camera.id);
    let mut frame_b = Frame::new(101, camera.id);
    for (index, point) in near_points.iter().enumerate() {
        let descriptor = vec![index as f32, 9.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);

        let keypoint = camera
            .project(&pose.transform_world_point(point))
            .expect("dummy point must be in front of the camera");
        frame_a.keypoints.push(keypoint);
        frame_a.descriptors.push(descriptor.clone());
        frame_b.keypoints.push(keypoint);
        frame_b.descriptors.push(descriptor);
    }

    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }

    let provider = InMemoryMapProvider::new(map);
    let frame_timestamps = FrameTimestampIndex::from_timed_frames([
        Timed::new(Timestamp::from_nanoseconds(0), frame_a.clone()),
        Timed::new(Timestamp::from_nanoseconds(100_000_000), frame_b.clone()),
    ]);
    let gnss_measurements = MeasurementBuffer::from_measurements([
        GnssMeasurement::new(Timestamp::from_nanoseconds(5_000_000), Point3::origin())
            .with_accuracy(Some(4.0), None),
        GnssMeasurement::new(
            Timestamp::from_nanoseconds(105_000_000),
            Point3::new(0.1, 0.0, 0.0),
        )
        .with_accuracy(Some(4.0), None),
    ]);
    let prior_source = FramePriorSource::new(
        frame_timestamps,
        gnss_measurements,
        TimeDelta::from_nanoseconds(20_000_000),
    )
    .with_prior_config(PriorConfig {
        default_radius: 50.0,
        min_radius: 2.0,
        confidence_multiplier: 2.0,
    });

    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());
    for frame in [&frame_a, &frame_b] {
        let prior = prior_source
            .localization_prior_for_frame(frame)
            .expect("dummy GNSS prior must be available");
        let result =
            tracker.track_frame_with_localization_prior_submap_provider(frame, &provider, &prior);

        println!(
            "frame={} gnss_radius={:?} map_landmarks={} candidates={} success={} inliers={} event={:?}",
            result.frame_id,
            prior.radius,
            result.map_landmark_count,
            result.localization.candidate_landmark_count,
            result.localization.success,
            result.localization.inlier_count,
            result.event,
        );
    }
}
