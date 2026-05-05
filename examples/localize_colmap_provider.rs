use std::path::PathBuf;

use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{QueryImage, VisualMapValidationReport};
use visloc_rs::io::colmap::ColmapMapProvider;
use visloc_rs::{map_provider_stats, DescriptorProvider, LocalizationPipeline, MapProvider};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let example_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let provider = ColmapMapProvider::from_text_model_dir_with_descriptors_validated(
        example_dir.join("data").join("colmap_text"),
        example_dir.join("data").join("landmark_descriptors.txt"),
    )?;

    print_validation_report("map", &provider.validate_map());
    print_validation_report("localization", &provider.validate_for_localization());
    let stats = map_provider_stats(&provider);
    println!(
        "map stats: cameras={} keyframes={} landmarks={} descriptors={}",
        stats.camera_count, stats.keyframe_count, stats.landmark_count, stats.descriptor_count
    );

    let map = provider.visual_map();
    let descriptor_store = provider
        .landmark_descriptor_store()
        .expect("example fixture must contain descriptors");
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

    let result = LocalizationPipeline::default().localize_with_provider(
        &QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        &provider,
    );

    println!("success: {}", result.success);
    println!("candidate landmarks: {}", result.candidate_landmark_count);
    println!("matches: {}", result.match_count);
    println!("inliers: {}", result.inlier_count);
    println!("inlier ratio: {:.3}", result.inlier_ratio);
    println!("mean reprojection error: {:?}", result.reprojection_error);
    println!("estimator diagnostics: {:?}", result.estimator_diagnostics);

    Ok(())
}

fn print_validation_report(name: &str, report: &VisualMapValidationReport) {
    if report.is_valid() {
        println!("{name} validation: ok");
    } else {
        println!("{name} validation: {} issue(s)", report.issue_count());
        for issue in &report.issues {
            println!("  - {issue:?}");
        }
    }
}
