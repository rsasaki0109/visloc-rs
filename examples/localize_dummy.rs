use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Landmark, QueryImage, VisualMap};
use visloc_rs::localize;

fn main() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let ground_truth_pose =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());

    let points = [
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
        Point3::new(-0.5, 0.4, 6.5),
        Point3::new(0.25, 0.75, 8.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());

    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();
    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 1.0, 0.5];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);

        keypoints.push(
            camera
                .project(&ground_truth_pose.transform_world_point(point))
                .expect("dummy point must be in front of the camera"),
        );
        descriptors.push(descriptor);
    }

    let query = QueryImage {
        camera,
        keypoints,
        descriptors,
    };

    let result = localize(query, map);
    println!("success: {}", result.success);
    println!("inliers: {}", result.inlier_count);
    println!("mean reprojection error: {:?}", result.reprojection_error);
    println!("estimator diagnostics: {:?}", result.estimator_diagnostics);
    println!("pose: {:#?}", result.pose);
}
