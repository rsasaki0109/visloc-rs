use std::fs;

use visloc_io::images::{parse_pgm, read_pgm, to_pgm_ascii, write_pgm_ascii, PgmImageError};
use visloc_vision::features::GrayscaleImage;

#[test]
fn parses_ascii_pgm_with_comments() {
    let image = parse_pgm(
        b"P2
# small image
3 2
255
0 127 255
255 127 0
",
    )
    .unwrap();

    assert_eq!(image.width(), 3);
    assert_eq!(image.height(), 2);
    assert_eq!(image.get(0, 0), Some(0.0));
    assert!((image.get(1, 0).unwrap() - 127.0 / 255.0).abs() < 1.0e-6);
    assert_eq!(image.get(2, 0), Some(1.0));
}

#[test]
fn parses_binary_pgm() {
    let image = parse_pgm(b"P5\n2 2\n255\n\x00\x7f\xff\x40").unwrap();

    assert_eq!(image.width(), 2);
    assert_eq!(image.height(), 2);
    assert_eq!(image.get(0, 0), Some(0.0));
    assert_eq!(image.get(0, 1), Some(1.0));
}

#[test]
fn rejects_pixel_count_mismatch() {
    let error = parse_pgm(b"P2\n2 2\n255\n0 1 2\n").unwrap_err();

    assert!(matches!(
        error,
        PgmImageError::PixelCountMismatch {
            expected_len: 4,
            actual_len: 3,
        }
    ));
}

#[test]
fn writes_and_reads_ascii_pgm() {
    let path = std::env::temp_dir().join(format!("visloc-rs-image-{}.pgm", std::process::id()));
    let image = GrayscaleImage::from_luma_u8(2, 2, vec![0, 64, 128, 255]).unwrap();

    write_pgm_ascii(&path, &image).unwrap();
    let round_trip = read_pgm(&path).unwrap();
    fs::remove_file(path).unwrap();

    assert_eq!(round_trip.width(), 2);
    assert_eq!(round_trip.height(), 2);
    assert_eq!(
        to_pgm_ascii(&round_trip).unwrap().lines().next(),
        Some("P2")
    );
    assert_eq!(round_trip.get(1, 1), Some(1.0));
}
