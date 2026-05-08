use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    LocalMappingPipeline, LocalizationPipeline, LoopClosureConfig, OnlineSlamConfig,
    OnlineSlamPipeline, Tracker, TrackingConfig,
};

fn main() {
    let (map, first_frame) = map_and_frame(10, 1);
    let (_, return_frame) = map_and_frame(30, 1);
    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 5,
                min_shared_landmarks: 4,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
        },
    );

    let first = slam.process_frame(&first_frame, []);
    println!(
        "frame={} tracking={} keyframes={} loop_candidates={}",
        first.tracking.frame_id,
        first.tracking_succeeded(),
        first.map_keyframe_count,
        first.loop_closure_candidates.len()
    );

    let returned = slam.process_frame(&return_frame, []);
    println!(
        "frame={} tracking={} keyframes={} loop_candidates={}",
        returned.tracking.frame_id,
        returned.tracking_succeeded(),
        returned.map_keyframe_count,
        returned.loop_closure_candidates.len()
    );

    for candidate in &returned.loop_closure_candidates {
        println!(
            "loop_candidate query={} matched_keyframe={} shared_landmarks={} ratio={:.2} score={:.2} verified={}",
            candidate.query_frame_id,
            candidate.matched_keyframe_id,
            candidate.shared_landmark_count,
            candidate.shared_landmark_ratio,
            candidate.score,
            candidate.geometrically_verified
        );
    }
}

fn map_and_frame(frame_id: u64, camera_id: u64) -> (VisualMap, Frame) {
    let camera = Camera::pinhole(camera_id, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = [
        Point3::new(-1.0, -1.0, 5.0),
        Point3::new(1.0, -1.0, 5.0),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.0),
        Point3::new(0.0, -0.5, 6.0),
        Point3::new(0.5, 0.75, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut frame = Frame::new(frame_id, camera_id);

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 1.0];
        let mut landmark = Landmark::new(landmark_id, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark_id, landmark);
        frame
            .keypoints
            .push(camera.project(&pose.transform_world_point(point)).unwrap());
        frame.descriptors.push(descriptor);
    }

    (map, frame)
}
