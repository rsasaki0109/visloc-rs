use std::convert::Infallible;
use std::path::PathBuf;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Landmark, VisualMap};
use visloc_rs::{FeatureSet, FnFeatureExtractor, ImageLocalizer};

#[derive(Debug)]
struct DemoImage {
    path: PathBuf,
    bytes: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("data")
        .join("query_frame.svg");
    let image = DemoImage {
        path: image_path.clone(),
        bytes: std::fs::read(&image_path)?,
    };

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
        let descriptor = vec![index as f32, 13.0, 0.75];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        keypoints.push(
            camera
                .project(&ground_truth_pose.transform_world_point(point))
                .expect("demo point must be in front of the camera"),
        );
        descriptors.push(descriptor);
    }

    let marker_count = count_demo_markers(&image.bytes);
    let features = FeatureSet::new(keypoints, descriptors).expect("demo features must be valid");
    let localizer = ImageLocalizer::new(FnFeatureExtractor::new(move |image: &DemoImage| {
        assert!(
            image.bytes.starts_with(b"<svg") && contains_bytes(&image.bytes, b"data-visloc-demo"),
            "demo image must be the checked-in SVG query frame"
        );
        assert_eq!(
            count_demo_markers(&image.bytes),
            features.keypoints.len(),
            "demo image markers and feature set must stay in sync"
        );
        Ok::<_, Infallible>(features.clone())
    }));

    let result = localizer.localize_image(&image, camera, &map)?;

    println!("image: {}", image.path.display());
    println!("image bytes: {}", image.bytes.len());
    println!("detected demo markers: {marker_count}");
    println!("success: {}", result.success);
    println!("matches: {}", result.match_count);
    println!("inliers: {}", result.inlier_count);
    println!("mean reprojection error: {:?}", result.reprojection_error);
    println!("pose: {:#?}", result.pose);

    Ok(())
}

fn count_demo_markers(bytes: &[u8]) -> usize {
    let text = std::str::from_utf8(bytes).unwrap_or_default();
    text.matches("data-landmark-id=").count()
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}
