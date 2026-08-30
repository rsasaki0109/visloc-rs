#![cfg(feature = "image-io")]

use std::io::Cursor;

use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, Rgb, Rgba};
use visloc_io::images::{
    common_image_sequence_summary, decode_common_image, decode_common_image_colmap_grayscale,
    parse_timestamp_nanoseconds_txt, read_common_image, read_common_image_sequence,
    read_common_image_sequence_dir, read_common_image_sequence_dir_with_timestamp_file,
    read_common_image_sequence_dir_with_timestamps, read_common_image_sequence_with_timestamps,
    read_timestamp_nanoseconds_txt, validate_common_image_sequence_dimensions,
    validate_common_image_sequence_timestamps, write_png_gray, ImageSequenceError,
    ImageSequenceValidationIssue,
};
use visloc_vision::features::GrayscaleImage;

#[test]
fn decodes_png_as_grayscale_image() {
    let buffer = ImageBuffer::from_fn(2, 2, |x, y| if x == y { Luma([255]) } else { Luma([0]) });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(buffer)
        .write_to(&mut bytes, ImageFormat::Png)
        .unwrap();

    let image = decode_common_image(bytes.get_ref()).unwrap();

    assert_eq!(image.width(), 2);
    assert_eq!(image.height(), 2);
    assert_eq!(image.pixels()[0], 1.0);
    assert_eq!(image.pixels()[1], 0.0);
}

#[test]
fn decodes_jpeg_as_grayscale_image() {
    let buffer = ImageBuffer::from_fn(3, 2, |x, _| {
        if x == 0 {
            Rgb([255, 255, 255])
        } else {
            Rgb([0, 0, 0])
        }
    });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(buffer)
        .write_to(&mut bytes, ImageFormat::Jpeg)
        .unwrap();

    let image = decode_common_image(bytes.get_ref()).unwrap();

    assert_eq!(image.width(), 3);
    assert_eq!(image.height(), 2);
    assert!(image.pixels()[0] > image.pixels()[1]);
}

#[test]
fn colmap_grayscale_uses_float_rounding_and_ignores_alpha() {
    // The blue-only pixel exercises the observable difference from the
    // image crate's integer/floor conversion: 0.0722*7 is below one, but
    // COLMAP's +0.5f rounds it to one.
    let buffer = ImageBuffer::from_fn(3, 1, |x, _| match x {
        0 => Rgba([0, 0, 7, 0]),
        1 => Rgba([10, 20, 30, 255]),
        _ => Rgba([255, 255, 255, 17]),
    });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(buffer)
        .write_to(&mut bytes, ImageFormat::Png)
        .unwrap();

    let image = decode_common_image_colmap_grayscale(bytes.get_ref()).unwrap();
    let legacy = decode_common_image(bytes.get_ref()).unwrap();

    assert_eq!(image.pixels(), &[1.0 / 255.0, 19.0 / 255.0, 1.0]);
    assert_eq!(legacy.pixels()[0], 0.0);
}

#[test]
fn writes_and_reads_png_grayscale_image() {
    let dir = tempfile_dir();
    let path = dir.join("fixture.png");
    let image = GrayscaleImage::from_luma_u8(3, 1, vec![0, 128, 255]).unwrap();

    write_png_gray(&path, &image).unwrap();
    let loaded = read_common_image(&path).unwrap();

    assert_eq!(loaded.width(), 3);
    assert_eq!(loaded.height(), 1);
    assert_eq!(loaded.pixels()[0], 0.0);
    assert!(loaded.pixels()[1] > 0.49 && loaded.pixels()[1] < 0.51);
    assert_eq!(loaded.pixels()[2], 1.0);
}

#[test]
fn reads_common_image_sequence_in_given_order() {
    let dir = tempfile_dir();
    let first = dir.join("first.png");
    let second = dir.join("second.png");
    write_png_gray(
        &first,
        &GrayscaleImage::from_luma_u8(1, 1, vec![0]).unwrap(),
    )
    .unwrap();
    write_png_gray(
        &second,
        &GrayscaleImage::from_luma_u8(1, 1, vec![255]).unwrap(),
    )
    .unwrap();

    let frames = read_common_image_sequence(&[second.clone(), first.clone()]).unwrap();

    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].frame_id, 0);
    assert_eq!(frames[0].timestamp_nanoseconds, None);
    assert_eq!(frames[0].path, second);
    assert_eq!(frames[0].image.pixels()[0], 1.0);
    assert_eq!(frames[1].frame_id, 1);
    assert_eq!(frames[1].timestamp_nanoseconds, None);
    assert_eq!(frames[1].path, first);
    assert_eq!(frames[1].image.pixels()[0], 0.0);
}

#[test]
fn reads_common_image_sequence_with_timestamps() {
    let dir = tempfile_dir();
    let first = dir.join("0000.png");
    let second = dir.join("0001.png");
    write_png_gray(
        &first,
        &GrayscaleImage::from_luma_u8(1, 1, vec![0]).unwrap(),
    )
    .unwrap();
    write_png_gray(
        &second,
        &GrayscaleImage::from_luma_u8(1, 1, vec![255]).unwrap(),
    )
    .unwrap();

    let frames = read_common_image_sequence_with_timestamps(&[first, second], &[100, 200]).unwrap();

    assert_eq!(frames[0].timestamp_nanoseconds, Some(100));
    assert_eq!(frames[1].timestamp_nanoseconds, Some(200));
}

#[test]
fn rejects_mismatched_image_sequence_timestamp_count() {
    let dir = tempfile_dir();
    let first = dir.join("0000.png");
    write_png_gray(
        &first,
        &GrayscaleImage::from_luma_u8(1, 1, vec![0]).unwrap(),
    )
    .unwrap();

    let error = read_common_image_sequence_with_timestamps(&[first], &[100, 200])
        .expect_err("timestamp count mismatch should fail");

    assert!(matches!(
        error,
        ImageSequenceError::TimestampCountMismatch {
            path_count: 1,
            timestamp_count: 2,
        }
    ));
}

#[test]
fn parses_timestamp_nanoseconds_text() {
    let timestamps = parse_timestamp_nanoseconds_txt(
        "\n# timestamp_ns\n100\n200 frame_0001.png\n 300   # inline note treated as extra columns\n",
    )
    .unwrap();

    assert_eq!(timestamps, vec![100, 200, 300]);
}

#[test]
fn rejects_invalid_timestamp_text_line() {
    let error = parse_timestamp_nanoseconds_txt("100\nnot_a_timestamp\n")
        .expect_err("invalid timestamp should fail");

    assert!(matches!(
        error,
        ImageSequenceError::InvalidTimestampLine { line_number: 2, .. }
    ));
}

#[test]
fn reads_common_image_sequence_dir_sorted_by_path() {
    let dir = tempfile_dir();
    let a = dir.join("0000.png");
    let b = dir.join("0001.png");
    let ignored = dir.join("notes.txt");
    write_png_gray(&b, &GrayscaleImage::from_luma_u8(1, 1, vec![255]).unwrap()).unwrap();
    write_png_gray(&a, &GrayscaleImage::from_luma_u8(1, 1, vec![0]).unwrap()).unwrap();
    std::fs::write(ignored, "not an image").unwrap();

    let frames = read_common_image_sequence_dir(&dir).unwrap();

    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].frame_id, 0);
    assert_eq!(frames[0].path, a);
    assert_eq!(frames[1].frame_id, 1);
    assert_eq!(frames[1].path, b);
}

#[test]
fn reads_common_image_sequence_dir_with_timestamps() {
    let dir = tempfile_dir();
    let a = dir.join("0000.png");
    let b = dir.join("0001.png");
    write_png_gray(&b, &GrayscaleImage::from_luma_u8(1, 1, vec![255]).unwrap()).unwrap();
    write_png_gray(&a, &GrayscaleImage::from_luma_u8(1, 1, vec![0]).unwrap()).unwrap();

    let frames = read_common_image_sequence_dir_with_timestamps(&dir, &[10, 20]).unwrap();

    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].path, a);
    assert_eq!(frames[0].timestamp_nanoseconds, Some(10));
    assert_eq!(frames[1].path, b);
    assert_eq!(frames[1].timestamp_nanoseconds, Some(20));
}

#[test]
fn reads_common_image_sequence_dir_with_timestamp_file() {
    let dir = tempfile_dir();
    let a = dir.join("0000.png");
    let b = dir.join("0001.png");
    let timestamp_path = dir.join("timestamps_ns.txt");
    write_png_gray(&b, &GrayscaleImage::from_luma_u8(1, 1, vec![255]).unwrap()).unwrap();
    write_png_gray(&a, &GrayscaleImage::from_luma_u8(1, 1, vec![0]).unwrap()).unwrap();
    std::fs::write(&timestamp_path, "# timestamp_ns\n100\n200\n").unwrap();

    let timestamps = read_timestamp_nanoseconds_txt(&timestamp_path).unwrap();
    let frames = read_common_image_sequence_dir_with_timestamp_file(&dir, &timestamp_path).unwrap();

    assert_eq!(timestamps, vec![100, 200]);
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].path, a);
    assert_eq!(frames[0].timestamp_nanoseconds, Some(100));
    assert_eq!(frames[1].path, b);
    assert_eq!(frames[1].timestamp_nanoseconds, Some(200));
}

#[test]
fn summarizes_common_image_sequence() {
    let dir = tempfile_dir();
    let first = dir.join("0000.png");
    let second = dir.join("0001.png");
    write_png_gray(
        &first,
        &GrayscaleImage::from_luma_u8(2, 1, vec![0, 255]).unwrap(),
    )
    .unwrap();
    write_png_gray(
        &second,
        &GrayscaleImage::from_luma_u8(2, 1, vec![255, 0]).unwrap(),
    )
    .unwrap();
    let frames = read_common_image_sequence(&[first, second]).unwrap();

    let summary = common_image_sequence_summary(&frames);

    assert_eq!(summary.frame_count, 2);
    assert_eq!(summary.first_frame_id, Some(0));
    assert_eq!(summary.last_frame_id, Some(1));
    assert_eq!(summary.width, Some(2));
    assert_eq!(summary.height, Some(1));
    assert!(!summary.varying_dimensions);
    assert_eq!(summary.timestamp_count, 0);
    assert_eq!(summary.first_timestamp_nanoseconds, None);
    assert_eq!(summary.last_timestamp_nanoseconds, None);
    assert!(summary.timestamps_valid);
}

#[test]
fn validates_common_image_sequence_dimensions() {
    let dir = tempfile_dir();
    let first = dir.join("0000.png");
    let second = dir.join("0001.png");
    write_png_gray(
        &first,
        &GrayscaleImage::from_luma_u8(2, 1, vec![0, 255]).unwrap(),
    )
    .unwrap();
    write_png_gray(
        &second,
        &GrayscaleImage::from_luma_u8(1, 2, vec![255, 0]).unwrap(),
    )
    .unwrap();
    let frames = read_common_image_sequence(&[first, second.clone()]).unwrap();

    let issues = validate_common_image_sequence_dimensions(&frames);
    let summary = common_image_sequence_summary(&frames);

    assert_eq!(
        issues,
        vec![ImageSequenceValidationIssue::InconsistentDimensions {
            frame_id: 1,
            path: second,
            width: 1,
            height: 2,
            expected_width: 2,
            expected_height: 1,
        }]
    );
    assert!(summary.varying_dimensions);
}

#[test]
fn summarizes_timestamped_common_image_sequence() {
    let dir = tempfile_dir();
    let first = dir.join("0000.png");
    let second = dir.join("0001.png");
    write_png_gray(
        &first,
        &GrayscaleImage::from_luma_u8(1, 1, vec![0]).unwrap(),
    )
    .unwrap();
    write_png_gray(
        &second,
        &GrayscaleImage::from_luma_u8(1, 1, vec![255]).unwrap(),
    )
    .unwrap();
    let frames =
        read_common_image_sequence_with_timestamps(&[first, second], &[1_000, 2_000]).unwrap();

    let summary = common_image_sequence_summary(&frames);

    assert_eq!(summary.timestamp_count, 2);
    assert_eq!(summary.first_timestamp_nanoseconds, Some(1_000));
    assert_eq!(summary.last_timestamp_nanoseconds, Some(2_000));
    assert!(summary.timestamps_valid);
}

#[test]
fn validates_common_image_sequence_timestamps() {
    let dir = tempfile_dir();
    let first = dir.join("0000.png");
    let second = dir.join("0001.png");
    let third = dir.join("0002.png");
    write_png_gray(
        &first,
        &GrayscaleImage::from_luma_u8(1, 1, vec![0]).unwrap(),
    )
    .unwrap();
    write_png_gray(
        &second,
        &GrayscaleImage::from_luma_u8(1, 1, vec![128]).unwrap(),
    )
    .unwrap();
    write_png_gray(
        &third,
        &GrayscaleImage::from_luma_u8(1, 1, vec![255]).unwrap(),
    )
    .unwrap();
    let mut frames = read_common_image_sequence_with_timestamps(
        &[first, second.clone(), third.clone()],
        &[100, 200, 50],
    )
    .unwrap();
    frames[1].timestamp_nanoseconds = None;

    let issues = validate_common_image_sequence_timestamps(&frames);
    let summary = common_image_sequence_summary(&frames);

    assert_eq!(
        issues,
        vec![
            ImageSequenceValidationIssue::MissingTimestamp {
                frame_id: 1,
                path: second,
            },
            ImageSequenceValidationIssue::NonMonotonicTimestamp {
                frame_id: 2,
                path: third,
                timestamp_nanoseconds: 50,
                previous_frame_id: 0,
                previous_timestamp_nanoseconds: 100,
            },
        ]
    );
    assert!(!summary.timestamps_valid);
}

fn tempfile_dir() -> std::path::PathBuf {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "visloc_common_image_test_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
