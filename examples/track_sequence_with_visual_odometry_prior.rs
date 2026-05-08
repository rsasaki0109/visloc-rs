use std::collections::HashMap;
use std::convert::Infallible;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    InMemoryMapProvider, LocalizationPipeline, LocalizationPrior, Tracker, TrackingConfig,
    VisualOdometryEstimate, VisualOdometryFrontend, VisualOdometryPriorProvider,
};

#[derive(Debug, Clone)]
struct FixedVisualOdometryFrontend {
    motions: HashMap<(u64, u64), SE3>,
}

impl VisualOdometryFrontend for FixedVisualOdometryFrontend {
    type Error = Infallible;

    fn estimate_relative_pose(
        &self,
        previous_frame: &Frame,
        current_frame: &Frame,
    ) -> Result<Option<VisualOdometryEstimate>, Self::Error> {
        let Some(previous_to_current) = self
            .motions
            .get(&(previous_frame.id, current_frame.id))
            .cloned()
        else {
            return Ok(None);
        };

        let mut estimate =
            VisualOdometryEstimate::new(previous_frame.id, current_frame.id, previous_to_current);
        estimate.match_count = 128;
        estimate.inlier_count = 92;
        estimate.mean_reprojection_error = Some(0.8);
        Ok(Some(estimate))
    }
}

fn main() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let map_points = [
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];
    let poses = [
        (
            100,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0)),
        ),
        (
            101,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.45, 0.0, 0.0)),
        ),
        (
            102,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.9, 0.0, -0.1)),
        ),
    ];

    let (map, descriptors) = build_map(&camera, &map_points);
    let frames = poses
        .iter()
        .map(|(frame_id, pose)| {
            frame_from_projected_landmarks(*frame_id, &camera, pose, &map_points, &descriptors)
        })
        .collect::<Vec<_>>();
    let provider = InMemoryMapProvider::new(map);
    let vo_prior_provider = VisualOdometryPriorProvider::new(FixedVisualOdometryFrontend {
        motions: HashMap::from([
            (
                (100, 101),
                SE3::new(UnitQuaternion::identity(), Vector3::new(-0.45, 0.0, 0.0)),
            ),
            (
                (101, 102),
                SE3::new(UnitQuaternion::identity(), Vector3::new(-0.45, 0.0, -0.1)),
            ),
        ]),
    });
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());
    let mut previous_frame = None;
    let mut previous_pose = None;

    for frame in &frames {
        let vo_prior = previous_frame.zip(previous_pose.as_ref()).and_then(
            |(previous_frame, previous_pose)| {
                vo_prior_provider
                    .predict_pose_prior(previous_frame, previous_pose, frame)
                    .expect("fixed VO frontend cannot fail")
            },
        );
        let result = if let Some(vo_prior) = vo_prior {
            let prior = LocalizationPrior::from_pose(vo_prior.pose, 8.0);
            let result = tracker
                .track_frame_with_localization_prior_submap_provider(frame, &provider, &prior);
            println!(
                "frame={} vo_prior=true matches={} vo_inliers={} candidates={} success={} tracking_external_prior={} center={}",
                result.frame_id,
                vo_prior.estimate.match_count,
                vo_prior.estimate.inlier_count,
                result.localization.candidate_landmark_count,
                result.localization.success,
                result.used_external_localization_prior,
                format_estimated_center(&result.localization.pose),
            );
            result
        } else {
            let result = tracker.track_frame_with_provider(frame, &provider);
            println!(
                "frame={} vo_prior=false candidates={} success={} tracking_external_prior={} center={}",
                result.frame_id,
                result.localization.candidate_landmark_count,
                result.localization.success,
                result.used_external_localization_prior,
                format_estimated_center(&result.localization.pose),
            );
            result
        };

        previous_frame = Some(frame);
        previous_pose = result.localization.pose.clone();
    }

    let stats = tracker.stats();
    println!(
        "stats frames={} success_rate={:.3} external_prior_rate={:.3} external_prior_count={}",
        stats.frame_count,
        stats.success_rate(),
        stats.external_localization_prior_usage_rate(),
        stats.external_localization_prior_used_count
    );
}

fn build_map(camera: &Camera, points: &[Point3<f64>]) -> (VisualMap, Vec<Vec<f32>>) {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 9.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        descriptors.push(descriptor);
    }

    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }

    (map, descriptors)
}

fn frame_from_projected_landmarks(
    frame_id: u64,
    camera: &Camera,
    pose: &Pose,
    points: &[Point3<f64>],
    descriptors: &[Vec<f32>],
) -> Frame {
    let mut frame = Frame::new(frame_id, camera.id);
    for (point, descriptor) in points.iter().zip(descriptors.iter()) {
        let keypoint = camera
            .project(&pose.transform_world_point(point))
            .expect("dummy point must be in front of the camera");
        frame.keypoints.push(keypoint);
        frame.descriptors.push(descriptor.clone());
    }
    frame
}

fn format_estimated_center(pose: &Option<Pose>) -> String {
    pose.as_ref()
        .map(|pose| {
            let center = pose.camera_center_world();
            format!("[{:.3}, {:.3}, {:.3}]", center.x, center.y, center.z)
        })
        .unwrap_or_else(|| "n/a".to_string())
}
