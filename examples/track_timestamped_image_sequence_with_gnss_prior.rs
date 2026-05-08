use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::Point3;
use visloc_rs::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sequence_dir = demo_sequence_dir();
    fs::create_dir_all(&sequence_dir)?;
    let image_paths = write_demo_sequence(&sequence_dir)?;
    let timestamps = [0, 100_000_000, 200_000_000];

    let frames = read_common_image_sequence_with_timestamps(&image_paths, &timestamps)?;
    let timestamp_issues = validate_common_image_sequence_timestamps(&frames);
    let sequence_summary = common_image_sequence_summary(&frames);
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
    let provider = InMemoryMapProvider::new(map);
    let prior_source = gnss_prior_source_from_loaded_frames(&frames);

    let mut tracker = ImageTracker::new(extractor, TrackingConfig::default());
    let mut results = Vec::new();
    for frame in &frames {
        let prior = prior_source
            .localization_prior_for_frame_id(frame.frame_id)
            .expect("timestamped image frame should have a GNSS prior");
        let result = tracker.track_frame_image_with_localization_prior_submap_provider(
            frame.frame_id,
            camera.id,
            &frame.image,
            &provider,
            &prior,
        )?;
        println!(
            "frame={} timestamp_ns={:?} gnss_radius={:?} event={:?} success={} inliers={}",
            result.frame_id,
            frame.timestamp_nanoseconds,
            prior.radius,
            result.event,
            result.localization.success,
            result.localization.inlier_count,
        );
        results.push(result);
    }

    println!("sequence dir: {}", sequence_dir.display());
    println!(
        "frames={} timestamps={} timestamp_valid={} timestamp_issues={}",
        sequence_summary.frame_count,
        sequence_summary.timestamp_count,
        sequence_summary.timestamps_valid,
        timestamp_issues.len()
    );
    println!(
        "tracking stats: frames={} success_rate={:.3} external_prior_rate={:.3} trajectory_poses={}",
        tracker.tracker().stats().frame_count,
        tracker.tracker().stats().success_rate(),
        tracker
            .tracker()
            .stats()
            .external_localization_prior_usage_rate(),
        PoseTrajectory::from_tracking_results(&results).len()
    );

    Ok(())
}

fn demo_sequence_dir() -> PathBuf {
    PathBuf::from("target").join("visloc_timestamped_image_sequence_demo")
}

fn write_demo_sequence(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let image = synthetic_marker_image()?;
    let mut paths = Vec::new();
    for frame_id in 0..3 {
        let path = dir.join(format!("{frame_id:04}.png"));
        write_png_gray(&path, &image)?;
        paths.push(path);
    }
    Ok(paths)
}

fn gnss_prior_source_from_loaded_frames(
    frames: &[LoadedImageFrame],
) -> FramePriorSource<GnssMeasurement> {
    let mut frame_timestamps = FrameTimestampIndex::new();
    let mut measurements = Vec::new();
    for frame in frames {
        let timestamp = Timestamp::from_nanoseconds(
            frame
                .timestamp_nanoseconds
                .expect("demo frames are timestamped"),
        );
        frame_timestamps.insert_frame_id(frame.frame_id, timestamp);
        measurements.push(GnssMeasurement::new(timestamp, Point3::origin()));
    }

    FramePriorSource::new(
        frame_timestamps,
        MeasurementBuffer::from_measurements(measurements),
        TimeDelta::from_nanoseconds(1),
    )
    .with_prior_config(PriorConfig {
        default_radius: 20.0,
        min_radius: 1.0,
        confidence_multiplier: 2.0,
    })
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
