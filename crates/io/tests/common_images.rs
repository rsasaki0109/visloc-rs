#![cfg(feature = "image-io")]

use std::io::Cursor;

use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, Rgb};
use visloc_io::images::{
    common_image_sequence_summary, decode_common_image, read_common_image,
    read_common_image_sequence, read_common_image_sequence_dir,
    validate_common_image_sequence_dimensions, write_png_gray, ImageSequenceValidationIssue,
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
    assert_eq!(frames[0].path, second);
    assert_eq!(frames[0].image.pixels()[0], 1.0);
    assert_eq!(frames[1].frame_id, 1);
    assert_eq!(frames[1].path, first);
    assert_eq!(frames[1].image.pixels()[0], 0.0);
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
