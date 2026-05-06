use std::convert::Infallible;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Landmark, VisualMap};
use visloc_rs::{
    FeatureSet, FnFeatureExtractor, ImageTracker, LocalizationPipeline, Tracker, TrackingConfig,
};

#[derive(Debug, Clone)]
struct DummyImage;

fn main() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = [
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 31.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        keypoints.push(
            camera
                .project(&pose.transform_world_point(point))
                .expect("dummy point must be in front of the camera"),
        );
        descriptors.push(descriptor);
    }

    let features = FeatureSet::new(keypoints, descriptors).expect("dummy features must be valid");
    let extractor =
        FnFeatureExtractor::new(move |_image: &DummyImage| Ok::<_, Infallible>(features.clone()));
    let tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            last_pose_candidate_radius: Some(8.0),
            ..TrackingConfig::default()
        },
    );
    let mut image_tracker = ImageTracker::with_tracker(extractor, tracker);

    let images = [DummyImage, DummyImage, DummyImage, DummyImage];
    let frames = [
        (100, camera.id, &images[0]),
        (101, camera.id, &images[1]),
        (102, 999, &images[2]),
        (103, 999, &images[3]),
    ];
    for tracking in image_tracker
        .track_frame_images(frames, &map)
        .expect("dummy extractor is infallible")
    {
        println!(
            "frame={} state={:?} event={:?} success={} prior={} reason={:?} map_landmarks={} descriptors={} inliers={}",
            tracking.frame_id,
            tracking.state,
            tracking.event,
            tracking.localization.success,
            tracking.used_pose_prior,
            tracking.tracking_failure_reason,
            tracking.map_landmark_count,
            tracking.map_stats.descriptor_count,
            tracking.localization.inlier_count,
        );
    }
}
