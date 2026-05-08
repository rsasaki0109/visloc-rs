use std::fs;
use std::path::Path;
#[cfg(feature = "image-io")]
use std::path::PathBuf;
use thiserror::Error;
#[cfg(feature = "image-io")]
use visloc_core::types::FrameId;
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

#[cfg(feature = "image-io")]
#[derive(Debug, Error)]
pub enum CommonImageError {
    #[error("image decode/encode error: {0}")]
    Image(#[from] image::ImageError),
    #[error("grayscale image error: {0}")]
    Grayscale(#[from] GrayscaleImageError),
    #[error("image dimensions are too large for common image encoders: {width}x{height}")]
    DimensionTooLarge { width: usize, height: usize },
    #[error("failed to build grayscale image buffer")]
    InvalidImageBuffer,
}

#[cfg(feature = "image-io")]
#[derive(Debug, Error)]
pub enum ImageSequenceError {
    #[error("image sequence I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image frame {path} failed to load: {source}")]
    FrameLoad {
        path: PathBuf,
        source: CommonImageError,
    },
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedImageFrame {
    pub frame_id: FrameId,
    pub path: PathBuf,
    pub image: GrayscaleImage,
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSequenceSummary {
    pub frame_count: usize,
    pub first_frame_id: Option<FrameId>,
    pub last_frame_id: Option<FrameId>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub varying_dimensions: bool,
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageSequenceValidationIssue {
    InconsistentDimensions {
        frame_id: FrameId,
        path: PathBuf,
        width: usize,
        height: usize,
        expected_width: usize,
        expected_height: usize,
    },
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

#[cfg(feature = "image-io")]
pub fn read_common_image(path: impl AsRef<Path>) -> Result<GrayscaleImage, CommonImageError> {
    dynamic_image_to_grayscale(image::open(path)?)
}

#[cfg(feature = "image-io")]
pub fn decode_common_image(bytes: &[u8]) -> Result<GrayscaleImage, CommonImageError> {
    dynamic_image_to_grayscale(image::load_from_memory(bytes)?)
}

#[cfg(feature = "image-io")]
pub fn write_png_gray(
    path: impl AsRef<Path>,
    grayscale: &GrayscaleImage,
) -> Result<(), CommonImageError> {
    let width =
        u32::try_from(grayscale.width()).map_err(|_| CommonImageError::DimensionTooLarge {
            width: grayscale.width(),
            height: grayscale.height(),
        })?;
    let height =
        u32::try_from(grayscale.height()).map_err(|_| CommonImageError::DimensionTooLarge {
            width: grayscale.width(),
            height: grayscale.height(),
        })?;
    let pixels = grayscale
        .pixels()
        .iter()
        .map(|pixel| (pixel.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let buffer = image::GrayImage::from_raw(width, height, pixels)
        .ok_or(CommonImageError::InvalidImageBuffer)?;
    buffer.save_with_format(path, image::ImageFormat::Png)?;
    Ok(())
}

#[cfg(feature = "image-io")]
pub fn read_common_image_sequence<P>(
    paths: &[P],
) -> Result<Vec<LoadedImageFrame>, ImageSequenceError>
where
    P: AsRef<Path>,
{
    let mut frames = Vec::with_capacity(paths.len());
    for (index, path) in paths.iter().enumerate() {
        let path = path.as_ref();
        let image = read_common_image(path).map_err(|source| ImageSequenceError::FrameLoad {
            path: path.to_path_buf(),
            source,
        })?;
        frames.push(LoadedImageFrame {
            frame_id: index as FrameId,
            path: path.to_path_buf(),
            image,
        });
    }
    Ok(frames)
}

#[cfg(feature = "image-io")]
pub fn read_common_image_sequence_dir(
    dir: impl AsRef<Path>,
) -> Result<Vec<LoadedImageFrame>, ImageSequenceError> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if is_supported_common_image_path(&path) {
            paths.push(path);
        }
    }
    paths.sort();
    read_common_image_sequence(&paths)
}

#[cfg(feature = "image-io")]
pub fn common_image_sequence_summary(frames: &[LoadedImageFrame]) -> ImageSequenceSummary {
    let first = frames.first();
    ImageSequenceSummary {
        frame_count: frames.len(),
        first_frame_id: first.map(|frame| frame.frame_id),
        last_frame_id: frames.last().map(|frame| frame.frame_id),
        width: first.map(|frame| frame.image.width()),
        height: first.map(|frame| frame.image.height()),
        varying_dimensions: !validate_common_image_sequence_dimensions(frames).is_empty(),
    }
}

#[cfg(feature = "image-io")]
pub fn validate_common_image_sequence_dimensions(
    frames: &[LoadedImageFrame],
) -> Vec<ImageSequenceValidationIssue> {
    let Some(first) = frames.first() else {
        return Vec::new();
    };
    let expected_width = first.image.width();
    let expected_height = first.image.height();

    frames
        .iter()
        .skip(1)
        .filter_map(|frame| {
            let width = frame.image.width();
            let height = frame.image.height();
            (width != expected_width || height != expected_height).then(|| {
                ImageSequenceValidationIssue::InconsistentDimensions {
                    frame_id: frame.frame_id,
                    path: frame.path.clone(),
                    width,
                    height,
                    expected_width,
                    expected_height,
                }
            })
        })
        .collect()
}

#[cfg(feature = "image-io")]
fn is_supported_common_image_path(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "png"
    )
}

#[cfg(feature = "image-io")]
fn dynamic_image_to_grayscale(
    dynamic: image::DynamicImage,
) -> Result<GrayscaleImage, CommonImageError> {
    let luma = dynamic.to_luma8();
    let (width, height) = luma.dimensions();
    GrayscaleImage::from_luma_u8(width as usize, height as usize, luma.into_raw())
        .map_err(CommonImageError::from)
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
