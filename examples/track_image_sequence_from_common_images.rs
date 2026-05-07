use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::Point3;
use visloc_rs::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sequence_dir = demo_sequence_dir();
    fs::create_dir_all(&sequence_dir)?;
    write_demo_sequence(&sequence_dir)?;

    let frames = read_common_image_sequence_dir(&sequence_dir)?;
    let first = frames
        .first()
        .ok_or("demo sequence should contain at least one frame")?;

    let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
        max_features: 16,
        min_score: 0.25,
        descriptor_radius: 2,
    });
    let camera = Camera::pinhole(1, 64, 64, 50.0, 50.0, 32.0, 32.0);
    let map = build_map_from_image(&first.image, &extractor, &camera)?;

    let mut tracker = ImageTracker::new(extractor, TrackingConfig::default());
    let results = frames
        .iter()
        .map(|frame| tracker.track_frame_image(frame.frame_id, camera.id, &frame.image, &map))
        .collect::<Result<Vec<_>, _>>()?;

    println!("sequence dir: {}", sequence_dir.display());
    println!("loaded image frames: {}", frames.len());
    println!("map landmarks: {}", map.landmarks.len());
    for (frame, result) in frames.iter().zip(results.iter()) {
        println!(
            "frame={} path={} event={:?} success={} inliers={} mean_reprojection_error={:?}",
            result.frame_id,
            frame.path.display(),
            result.event,
            result.localization.success,
            result.localization.inlier_count,
            result.localization.reprojection_error
        );
    }
    println!(
        "tracking stats: frames={} success_rate={:.3} trajectory_poses={}",
        tracker.tracker().stats().frame_count,
        tracker.tracker().stats().success_rate(),
        PoseTrajectory::from_tracking_results(&results).len()
    );

    Ok(())
}

fn demo_sequence_dir() -> PathBuf {
    PathBuf::from("target").join("visloc_common_image_sequence_demo")
}

fn write_demo_sequence(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let image = synthetic_marker_image()?;
    for frame_id in 0..3 {
        write_png_gray(dir.join(format!("{frame_id:04}.png")), &image)?;
    }
    Ok(())
}

fn build_map_from_image(
    image: &GrayscaleImage,
    extractor: &CornerFeatureExtractor,
    camera: &Camera,
) -> Result<VisualMap, Box<dyn std::error::Error>> {
    let features = extractor.extract(image)?;
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());

    for (index, (keypoint, descriptor)) in features
        .keypoints
        .iter()
        .zip(features.descriptors.iter())
        .take(12)
        .enumerate()
    {
        let z = 4.0 + index as f64 * 0.15;
        let normalized = camera
            .normalize_pixel(keypoint)
            .expect("pinhole camera has intrinsics");
        let mut landmark = Landmark::new(
            index as u64 + 1,
            Point3::new(normalized.x * z, normalized.y * z, z),
        );
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
    }

    Ok(map)
}

fn synthetic_marker_image() -> Result<GrayscaleImage, GrayscaleImageError> {
    let width = 64;
    let height = 64;
    let mut pixels = vec![0_u8; width * height];
    let blocks = [
        (8, 8, 180),
        (24, 10, 220),
        (42, 12, 150),
        (12, 34, 240),
        (34, 38, 200),
    ];

    for (left, top, value) in blocks {
        for y in top..(top + 8) {
            for x in left..(left + 8) {
                pixels[y * width + x] = value;
            }
        }
    }

    GrayscaleImage::from_luma_u8(width, height, pixels)
}
