use std::env;
use std::path::PathBuf;

use visloc_rs::core::types::QueryImage;
use visloc_rs::io::colmap::ColmapMapProvider;
use visloc_rs::io::query_features::read_query_features_txt;
use visloc_rs::{DescriptorProvider, LocalizationPipeline, MapProvider};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let example_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let (map_dir, descriptor_path, query_feature_path, camera_id) = match args.as_slice() {
        [] => (
            example_dir.join("data").join("colmap_text"),
            example_dir.join("data").join("landmark_descriptors.txt"),
            example_dir.join("data").join("query_features.txt"),
            1_u64,
        ),
        [map_dir, descriptor_path, query_feature_path, camera_id] => (
            PathBuf::from(map_dir),
            PathBuf::from(descriptor_path),
            PathBuf::from(query_feature_path),
            camera_id.parse()?,
        ),
        _ => {
            eprintln!(
                "usage: cargo run --example localize_from_files -- <colmap_text_dir> <landmark_descriptors.txt> <query_features.txt> <camera_id>"
            );
            std::process::exit(2);
        }
    };

    let provider = ColmapMapProvider::from_text_model_dir_with_descriptors_validated(
        map_dir,
        descriptor_path,
    )?;
    let map = provider.visual_map();
    let camera = map
        .cameras
        .get(&camera_id)
        .cloned()
        .ok_or_else(|| format!("camera id {camera_id} not found in map"))?;
    let features = read_query_features_txt(query_feature_path)?;
    let query = QueryImage {
        camera,
        keypoints: features.keypoints,
        descriptors: features.descriptors,
    };

    let descriptor_count = provider
        .landmark_descriptor_store()
        .map(|store| store.len())
        .unwrap_or(0);
    println!(
        "loaded map: cameras={} keyframes={} landmarks={} descriptors={}",
        map.cameras.len(),
        map.keyframes.len(),
        map.landmarks.len(),
        descriptor_count
    );

    let result = LocalizationPipeline::default().localize_with_provider(&query, &provider);
    println!("success: {}", result.success);
    println!("failure: {:?}", result.failure_reason);
    println!("candidate landmarks: {}", result.candidate_landmark_count);
    println!("matches: {}", result.match_count);
    println!("correspondences: {}", result.correspondence_count);
    println!("inliers: {}", result.inlier_count);
    println!("inlier ratio: {:.3}", result.inlier_ratio);
    println!("mean reprojection error: {:?}", result.reprojection_error);
    println!(
        "pose failure diagnostics: {:?}",
        result.pose_failure_diagnostics
    );
    println!("estimator diagnostics: {:?}", result.estimator_diagnostics);
    println!("pose: {:#?}", result.pose);

    Ok(())
}
