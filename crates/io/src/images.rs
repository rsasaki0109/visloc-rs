use std::fs;
use std::path::Path;
use thiserror::Error;
use visloc_vision::features::{GrayscaleImage, GrayscaleImageError};

#[derive(Debug, Error)]
pub enum PgmImageError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid PGM header: {0}")]
    InvalidHeader(String),
    #[error("unsupported PGM magic {0}; expected P2 or P5")]
    UnsupportedMagic(String),
    #[error("invalid PGM integer: {0}")]
    InvalidInteger(String),
    #[error("unsupported PGM max value {0}; expected 1..=255")]
    UnsupportedMaxValue(u32),
    #[error("PGM pixel count mismatch: expected {expected_len}, got {actual_len}")]
    PixelCountMismatch {
        expected_len: usize,
        actual_len: usize,
    },
    #[error("grayscale image error: {0}")]
    Grayscale(#[from] GrayscaleImageError),
}

pub fn read_pgm(path: impl AsRef<Path>) -> Result<GrayscaleImage, PgmImageError> {
    parse_pgm(&fs::read(path)?)
}

pub fn write_pgm_ascii(
    path: impl AsRef<Path>,
    image: &GrayscaleImage,
) -> Result<(), PgmImageError> {
    fs::write(path, to_pgm_ascii(image)?)?;
    Ok(())
}

pub fn parse_pgm(bytes: &[u8]) -> Result<GrayscaleImage, PgmImageError> {
    let (magic, offset) = next_token(bytes, 0)
        .ok_or_else(|| PgmImageError::InvalidHeader("missing magic".to_owned()))?;
    let (width, offset) = parse_usize_token(bytes, offset, "width")?;
    let (height, offset) = parse_usize_token(bytes, offset, "height")?;
    let (max_value, offset) = parse_u32_token(bytes, offset, "max value")?;
    if max_value == 0 || max_value > 255 {
        return Err(PgmImageError::UnsupportedMaxValue(max_value));
    }

    let expected_len = width * height;
    match magic.as_str() {
        "P2" => parse_pgm_ascii_pixels(bytes, offset, width, height, expected_len, max_value),
        "P5" => parse_pgm_binary_pixels(bytes, offset, width, height, expected_len, max_value),
        other => Err(PgmImageError::UnsupportedMagic(other.to_owned())),
    }
}

pub fn to_pgm_ascii(image: &GrayscaleImage) -> Result<String, PgmImageError> {
    let mut output = format!("P2\n{} {}\n255\n", image.width(), image.height());
    for (index, pixel) in image.pixels().iter().enumerate() {
        let value = (pixel.clamp(0.0, 1.0) * 255.0).round() as u8;
        output.push_str(&value.to_string());
        if (index + 1) % image.width() == 0 {
            output.push('\n');
        } else {
            output.push(' ');
        }
    }
    Ok(output)
}

fn parse_pgm_ascii_pixels(
    bytes: &[u8],
    mut offset: usize,
    width: usize,
    height: usize,
    expected_len: usize,
    max_value: u32,
) -> Result<GrayscaleImage, PgmImageError> {
    let mut pixels = Vec::with_capacity(expected_len);
    while let Some((token, next_offset)) = next_token(bytes, offset) {
        let value = parse_u32(&token)?;
        if value > max_value {
            return Err(PgmImageError::InvalidInteger(format!(
                "pixel value {value} exceeds max value {max_value}"
            )));
        }
        pixels.push(value as f32 / max_value as f32);
        offset = next_offset;
    }

    if pixels.len() != expected_len {
        return Err(PgmImageError::PixelCountMismatch {
            expected_len,
            actual_len: pixels.len(),
        });
    }
    GrayscaleImage::new(width, height, pixels).map_err(PgmImageError::from)
}

fn parse_pgm_binary_pixels(
    bytes: &[u8],
    offset: usize,
    width: usize,
    height: usize,
    expected_len: usize,
    max_value: u32,
) -> Result<GrayscaleImage, PgmImageError> {
    let pixel_offset = skip_single_whitespace(bytes, offset);
    let data = bytes.get(pixel_offset..).unwrap_or_default();
    if data.len() != expected_len {
        return Err(PgmImageError::PixelCountMismatch {
            expected_len,
            actual_len: data.len(),
        });
    }

    let pixels = data
        .iter()
        .map(|value| *value as f32 / max_value as f32)
        .collect();
    GrayscaleImage::new(width, height, pixels).map_err(PgmImageError::from)
}

fn parse_usize_token(
    bytes: &[u8],
    offset: usize,
    label: &str,
) -> Result<(usize, usize), PgmImageError> {
    let (token, offset) = next_token(bytes, offset)
        .ok_or_else(|| PgmImageError::InvalidHeader(format!("missing {label}")))?;
    Ok((parse_usize(&token)?, offset))
}

fn parse_u32_token(
    bytes: &[u8],
    offset: usize,
    label: &str,
) -> Result<(u32, usize), PgmImageError> {
    let (token, offset) = next_token(bytes, offset)
        .ok_or_else(|| PgmImageError::InvalidHeader(format!("missing {label}")))?;
    Ok((parse_u32(&token)?, offset))
}

fn parse_usize(token: &str) -> Result<usize, PgmImageError> {
    token
        .parse::<usize>()
        .map_err(|error| PgmImageError::InvalidInteger(error.to_string()))
}

fn parse_u32(token: &str) -> Result<u32, PgmImageError> {
    token
        .parse::<u32>()
        .map_err(|error| PgmImageError::InvalidInteger(error.to_string()))
}

fn next_token(bytes: &[u8], mut offset: usize) -> Option<(String, usize)> {
    loop {
        while offset < bytes.len() && bytes[offset].is_ascii_whitespace() {
            offset += 1;
        }
        if offset >= bytes.len() {
            return None;
        }
        if bytes[offset] != b'#' {
            break;
        }
        while offset < bytes.len() && bytes[offset] != b'\n' {
            offset += 1;
        }
    }

    let start = offset;
    while offset < bytes.len() && !bytes[offset].is_ascii_whitespace() {
        offset += 1;
    }
    Some((
        String::from_utf8_lossy(&bytes[start..offset]).to_string(),
        offset,
    ))
}

fn skip_single_whitespace(bytes: &[u8], offset: usize) -> usize {
    if offset < bytes.len() && bytes[offset].is_ascii_whitespace() {
        offset + 1
    } else {
        offset
    }
}
