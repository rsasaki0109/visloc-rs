#![cfg(feature = "image-io")]

use std::io::Cursor;

use image::{DynamicImage, ImageBuffer, ImageFormat, Luma, Rgb};
use visloc_io::images::{decode_common_image, read_common_image, write_png_gray};
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

fn tempfile_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("visloc_common_image_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
