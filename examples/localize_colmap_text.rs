use std::path::PathBuf;

use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::QueryImage;
use visloc_rs::io::colmap::read_colmap_text_model;
use visloc_rs::io::descriptors::read_landmark_descriptors_txt;
use visloc_rs::localize_with_descriptor_store;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let example_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let map = read_colmap_text_model(example_dir.join("data").join("colmap_text"))?;
    let descriptor_store =
        read_landmark_descriptors_txt(example_dir.join("data").join("landmark_descriptors.txt"))?;

    let camera = map
        .cameras
        .get(&1)
        .expect("example fixture must contain camera 1")
        .clone();
    let ground_truth_pose =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());

    let mut landmark_ids = map.landmarks.keys().copied().collect::<Vec<_>>();
    landmark_ids.sort_unstable();

    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();
    for landmark_id in landmark_ids {
        let landmark = map
            .landmarks
            .get(&landmark_id)
            .expect("landmark id was collected from the map");
        keypoints.push(
            camera
                .project(&ground_truth_pose.transform_world_point(&landmark.position))
                .expect("example landmark must be in front of the camera"),
        );
        descriptors.push(
            descriptor_store
                .get(landmark_id)
                .expect("example fixture must contain descriptor for every landmark")
                .to_vec(),
        );
    }

    let query = QueryImage {
        camera,
        keypoints,
        descriptors,
    };

    let result = localize_with_descriptor_store(query, map, descriptor_store);
    println!("success: {}", result.success);
    println!("candidate landmarks: {}", result.candidate_landmark_count);
    println!("matches: {}", result.match_count);
    println!("correspondences: {}", result.correspondence_count);
    println!("inliers: {}", result.inlier_count);
    println!("outliers: {}", result.outlier_count);
    println!("inlier ratio: {:.3}", result.inlier_ratio);
    println!("inlier landmark ids: {:?}", result.inlier_landmark_ids);
    println!("mean reprojection error: {:?}", result.reprojection_error);
    println!(
        "median reprojection error: {:?}",
        result.median_reprojection_error
    );
    println!(
        "max reprojection error: {:?}",
        result.max_reprojection_error
    );
    println!("pose: {:#?}", result.pose);

    Ok(())
}
