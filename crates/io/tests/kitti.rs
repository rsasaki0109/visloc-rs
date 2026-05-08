#![cfg(feature = "image-io")]

use visloc_io::images::write_png_gray;
use visloc_io::kitti::{
    read_kitti_image_sequence_dir, read_kitti_image_sequence_dir_with_timestamp_file,
    KittiDatasetError,
};
use visloc_vision::features::GrayscaleImage;

#[test]
fn reads_kitti_image_sequence_with_calibration() {
    let dir = tempfile_dir();
    let image_dir = dir.join("image_2");
    std::fs::create_dir_all(&image_dir).unwrap();
    let calib_path = dir.join("calib.txt");
    write_png_gray(
        image_dir.join("000001.png"),
        &GrayscaleImage::from_luma_u8(3, 2, vec![255, 0, 0, 0, 255, 0]).unwrap(),
    )
    .unwrap();
    write_png_gray(
        image_dir.join("000000.png"),
        &GrayscaleImage::from_luma_u8(3, 2, vec![0, 255, 0, 255, 0, 0]).unwrap(),
    )
    .unwrap();
    std::fs::write(&calib_path, "P2: 700 0 600 0 0 710 180 0 0 0 1 0\n").unwrap();

    let sequence = read_kitti_image_sequence_dir(&image_dir, &calib_path, "P2", 2).unwrap();

    assert_eq!(sequence.camera.id, 2);
    assert_eq!(sequence.camera.width, 3);
    assert_eq!(sequence.camera.height, 2);
    assert_eq!(
        sequence.camera.intrinsics(),
        Some((700.0, 710.0, 600.0, 180.0))
    );
    assert_eq!(sequence.frames.len(), 2);
    assert!(sequence.frames[0].path.ends_with("000000.png"));
    assert!(sequence.frames[1].path.ends_with("000001.png"));
    assert_eq!(sequence.summary.frame_count, 2);
    assert_eq!(sequence.summary.timestamp_count, 0);
    assert!(sequence.dimension_issues.is_empty());
    assert!(sequence.timestamp_issues.is_empty());
}

#[test]
fn reads_timestamped_kitti_image_sequence_with_calibration() {
    let dir = tempfile_dir();
    let image_dir = dir.join("image_2");
    std::fs::create_dir_all(&image_dir).unwrap();
    let calib_path = dir.join("calib.txt");
    let timestamp_path = dir.join("times_ns.txt");
    write_png_gray(
        image_dir.join("000000.png"),
        &GrayscaleImage::from_luma_u8(2, 1, vec![0, 255]).unwrap(),
    )
    .unwrap();
    write_png_gray(
        image_dir.join("000001.png"),
        &GrayscaleImage::from_luma_u8(2, 1, vec![255, 0]).unwrap(),
    )
    .unwrap();
    std::fs::write(&calib_path, "P0: 500 0 1 0 0 510 2 0 0 0 1 0\n").unwrap();
    std::fs::write(&timestamp_path, "1000\n2000\n").unwrap();

    let sequence = read_kitti_image_sequence_dir_with_timestamp_file(
        &image_dir,
        &timestamp_path,
        &calib_path,
        "P0",
        1,
    )
    .unwrap();

    assert_eq!(sequence.frames[0].timestamp_nanoseconds, Some(1000));
    assert_eq!(sequence.frames[1].timestamp_nanoseconds, Some(2000));
    assert_eq!(sequence.summary.timestamp_count, 2);
    assert!(sequence.summary.timestamps_valid);
}

#[test]
fn rejects_empty_kitti_image_sequence() {
    let dir = tempfile_dir();
    let image_dir = dir.join("image_2");
    std::fs::create_dir_all(&image_dir).unwrap();
    let calib_path = dir.join("calib.txt");
    std::fs::write(&calib_path, "P2: 700 0 600 0 0 700 180 0 0 0 1 0\n").unwrap();

    let error = read_kitti_image_sequence_dir(&image_dir, &calib_path, "P2", 2)
        .expect_err("empty image sequence should fail");

    assert!(matches!(error, KittiDatasetError::EmptyImageSequence));
}

fn tempfile_dir() -> std::path::PathBuf {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "visloc_kitti_dataset_test_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
