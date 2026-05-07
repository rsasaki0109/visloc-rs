use std::fs;
use std::path::PathBuf;

use nalgebra::Point3;
use visloc_rs::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = demo_image_path();
    if let Some(parent) = image_path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_pgm_ascii(&image_path, &synthetic_marker_image()?)?;

    let image = read_pgm(&image_path)?;
    let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
        max_features: 16,
        min_score: 0.25,
        descriptor_radius: 2,
    });
    let features = extractor.extract(&image)?;

    let camera = Camera::pinhole(1, 64, 64, 50.0, 50.0, 32.0, 32.0);
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

    let localizer = ImageLocalizer::new(extractor);
    let result = localizer.localize_image(&image, camera, &map)?;

    println!("image: {}", image_path.display());
    println!("loaded image: {}x{}", image.width(), image.height());
    println!("extracted features: {}", features.len());
    println!("map landmarks: {}", map.landmarks.len());
    println!("success: {}", result.success);
    println!("matches: {}", result.match_count);
    println!("inliers: {}", result.inlier_count);
    println!("mean reprojection error: {:?}", result.reprojection_error);

    Ok(())
}

fn demo_image_path() -> PathBuf {
    PathBuf::from("target")
        .join("visloc_pgm_demo")
        .join("synthetic_marker.pgm")
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
