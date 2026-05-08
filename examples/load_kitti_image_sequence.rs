use std::fs;
use std::path::{Path, PathBuf};

use visloc_rs::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = demo_root();
    let image_dir = root.join("image_2");
    let timestamp_path = root.join("times_ns.txt");
    let calibration_path = root.join("calib.txt");
    fs::create_dir_all(&image_dir)?;
    write_demo_kitti_sequence(&image_dir, &timestamp_path, &calibration_path)?;

    let sequence = read_kitti_image_sequence_dir_with_timestamp_file(
        &image_dir,
        &timestamp_path,
        &calibration_path,
        "P2",
        2,
    )?;

    println!("kitti root: {}", root.display());
    println!("image dir: {}", image_dir.display());
    println!("calibration: {}", calibration_path.display());
    println!("timestamps: {}", timestamp_path.display());
    println!(
        "camera id={} size={}x{} intrinsics={:?}",
        sequence.camera.id,
        sequence.camera.width,
        sequence.camera.height,
        sequence.camera.intrinsics()
    );
    println!(
        "frames={} timestamps={} timestamp_valid={} dimension_issues={} timestamp_issues={}",
        sequence.summary.frame_count,
        sequence.summary.timestamp_count,
        sequence.summary.timestamps_valid,
        sequence.dimension_issues.len(),
        sequence.timestamp_issues.len()
    );
    for frame in &sequence.frames {
        println!(
            "frame={} timestamp_ns={:?} path={}",
            frame.frame_id,
            frame.timestamp_nanoseconds,
            frame.path.display()
        );
    }

    Ok(())
}

fn demo_root() -> PathBuf {
    PathBuf::from("target").join("visloc_kitti_image_sequence_demo")
}

fn write_demo_kitti_sequence(
    image_dir: &Path,
    timestamp_path: &Path,
    calibration_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let frames = [
        synthetic_road_marker_image(0)?,
        synthetic_road_marker_image(8)?,
        synthetic_road_marker_image(16)?,
    ];
    for (index, image) in frames.iter().enumerate() {
        write_png_gray(image_dir.join(format!("{index:06}.png")), image)?;
    }

    fs::write(timestamp_path, "0\n100000000\n200000000\n")?;
    fs::write(
        calibration_path,
        "P0: 700 0 32 0 0 700 24 0 0 0 1 0\n\
         P2: 710 0 32 0 0 705 24 0 0 0 1 0\n",
    )?;

    Ok(())
}

fn synthetic_road_marker_image(offset: usize) -> Result<GrayscaleImage, GrayscaleImageError> {
    let width = 64;
    let height = 48;
    let mut pixels = vec![20_u8; width * height];

    for y in 0..height {
        let center = width / 2 + offset / 4;
        let left_lane = center.saturating_sub(14 + y / 10);
        let right_lane = (center + 14 + y / 10).min(width - 1);
        pixels[y * width + left_lane] = 210;
        pixels[y * width + right_lane] = 210;
    }

    for block in 0..3 {
        let top = 8 + block * 12;
        let left = 10 + offset + block * 8;
        for y in top..(top + 5).min(height) {
            for x in left..(left + 8).min(width) {
                pixels[y * width + x] = 180;
            }
        }
    }

    GrayscaleImage::from_luma_u8(width, height, pixels)
}
