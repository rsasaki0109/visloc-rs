//! `sensor_msgs/Image` decoding into the luma buffers the pipelines consume.
//!
//! Supported encodings (the `sensor_msgs/image_encodings.hpp` names):
//!
//! | encoding                 | handling                                  |
//! |--------------------------|-------------------------------------------|
//! | `mono8`, `8UC1`          | copied as 8-bit luma                      |
//! | `mono16`, `16UC1`        | 16-bit luma (honours `is_bigendian`)      |
//! | `rgb8`, `bgr8`           | BT.601 luma `(299 R + 587 G + 114 B)/1000` |
//! | `rgba8`, `bgra8`         | as above, alpha ignored                   |
//!
//! Row padding (`step > width * bytes_per_pixel`) is honoured. Anything else
//! (Bayer, YUV, float depth images, ...) is rejected with a descriptive error
//! rather than guessed at.
//!
//! Basalt's raw image container is `u16`; 8-bit luma is widened with the
//! same `u16 = u8 << 8` convention the EuRoC reader uses, so a `mono8` ROS
//! stream feeds the estimator bit-identically to the PNG dataset path.

use std::fmt;

use crate::msgs::Image;

/// Pixel storage of a decoded luma image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LumaPixels {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

/// A decoded single-channel image, row-major and tightly packed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LumaImage {
    pub width: usize,
    pub height: usize,
    pub pixels: LumaPixels,
}

/// Why an incoming `sensor_msgs/Image` could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageConversionError {
    UnsupportedEncoding(String),
    EmptyImage {
        width: u32,
        height: u32,
    },
    StepTooSmall {
        step: u32,
        min_step: usize,
    },
    DataTooShort {
        len: usize,
        expected: usize,
    },
    SizeMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },
    Pipeline(String),
}

impl fmt::Display for ImageConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedEncoding(encoding) => write!(
                f,
                "unsupported image encoding `{encoding}` (expected mono8, mono16, rgb8, bgr8, rgba8 or bgra8)"
            ),
            Self::EmptyImage { width, height } => write!(f, "empty image {width}x{height}"),
            Self::StepTooSmall { step, min_step } => {
                write!(f, "row step {step} is smaller than the packed row size {min_step}")
            }
            Self::DataTooShort { len, expected } => {
                write!(f, "image data has {len} bytes, expected at least {expected}")
            }
            Self::SizeMismatch { expected, actual } => write!(
                f,
                "image is {}x{}, calibration expects {}x{}",
                actual.0, actual.1, expected.0, expected.1
            ),
            Self::Pipeline(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ImageConversionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    Mono8,
    Mono16,
    /// Three or four 8-bit channels; indices of R, G, B within a pixel.
    Color {
        channels: usize,
        r: usize,
        g: usize,
        b: usize,
    },
}

fn layout_for(encoding: &str) -> Result<Layout, ImageConversionError> {
    Ok(match encoding {
        "mono8" | "8UC1" => Layout::Mono8,
        "mono16" | "16UC1" => Layout::Mono16,
        "rgb8" => Layout::Color {
            channels: 3,
            r: 0,
            g: 1,
            b: 2,
        },
        "bgr8" => Layout::Color {
            channels: 3,
            r: 2,
            g: 1,
            b: 0,
        },
        "rgba8" => Layout::Color {
            channels: 4,
            r: 0,
            g: 1,
            b: 2,
        },
        "bgra8" => Layout::Color {
            channels: 4,
            r: 2,
            g: 1,
            b: 0,
        },
        other => return Err(ImageConversionError::UnsupportedEncoding(other.to_owned())),
    })
}

/// ITU-R BT.601 luma with round-half-up, in integer arithmetic.
#[inline]
pub fn bt601_luma(r: u8, g: u8, b: u8) -> u8 {
    ((299 * u32::from(r) + 587 * u32::from(g) + 114 * u32::from(b) + 500) / 1000) as u8
}

/// Decodes a `sensor_msgs/Image` into a tightly packed luma image.
pub fn decode_luma(msg: &Image) -> Result<LumaImage, ImageConversionError> {
    let layout = layout_for(&msg.encoding)?;
    if msg.width == 0 || msg.height == 0 {
        return Err(ImageConversionError::EmptyImage {
            width: msg.width,
            height: msg.height,
        });
    }
    let width = msg.width as usize;
    let height = msg.height as usize;
    let bytes_per_pixel = match layout {
        Layout::Mono8 => 1,
        Layout::Mono16 => 2,
        Layout::Color { channels, .. } => channels,
    };
    let min_step = width * bytes_per_pixel;
    let step = msg.step as usize;
    if step < min_step {
        return Err(ImageConversionError::StepTooSmall {
            step: msg.step,
            min_step,
        });
    }
    // The last row only needs its packed pixels, not the trailing padding.
    let expected = step * (height - 1) + min_step;
    if msg.data.len() < expected {
        return Err(ImageConversionError::DataTooShort {
            len: msg.data.len(),
            expected,
        });
    }
    let rows = (0..height).map(|y| &msg.data[y * step..y * step + min_step]);
    let pixels = match layout {
        Layout::Mono8 => {
            let mut out = Vec::with_capacity(width * height);
            for row in rows {
                out.extend_from_slice(row);
            }
            LumaPixels::U8(out)
        }
        Layout::Mono16 => {
            let big_endian = msg.is_bigendian != 0;
            let mut out = Vec::with_capacity(width * height);
            for row in rows {
                out.extend(row.chunks_exact(2).map(|pair| {
                    let bytes = [pair[0], pair[1]];
                    if big_endian {
                        u16::from_be_bytes(bytes)
                    } else {
                        u16::from_le_bytes(bytes)
                    }
                }));
            }
            LumaPixels::U16(out)
        }
        Layout::Color { channels, r, g, b } => {
            let mut out = Vec::with_capacity(width * height);
            for row in rows {
                out.extend(
                    row.chunks_exact(channels)
                        .map(|pixel| bt601_luma(pixel[r], pixel[g], pixel[b])),
                );
            }
            LumaPixels::U8(out)
        }
    };
    Ok(LumaImage {
        width,
        height,
        pixels,
    })
}

impl LumaImage {
    /// Widens to Basalt's raw `u16` convention (`u8 << 8`; 16-bit as is).
    pub fn to_u16(&self) -> Vec<u16> {
        match &self.pixels {
            LumaPixels::U8(pixels) => pixels.iter().map(|&v| u16::from(v) << 8).collect(),
            LumaPixels::U16(pixels) => pixels.clone(),
        }
    }

    /// Normalizes to `[0, 1]` floats (the SIFT / feature-extractor input).
    pub fn to_unit_f32(&self) -> Vec<f32> {
        match &self.pixels {
            LumaPixels::U8(pixels) => pixels.iter().map(|&v| f32::from(v) / 255.0).collect(),
            LumaPixels::U16(pixels) => pixels.iter().map(|&v| f32::from(v) / 65535.0).collect(),
        }
    }

    /// Converts into Basalt's raw image container.
    pub fn to_basalt(&self) -> Result<visloc_basalt::RawU16Image, ImageConversionError> {
        visloc_basalt::RawU16Image::new(self.width, self.height, self.to_u16())
            .map_err(|error| ImageConversionError::Pipeline(format!("{error:?}")))
    }
}

/// Builds a `sensor_msgs/Image` from packed 8-bit luma (test/tool helper).
pub fn mono8_image(header: crate::msgs::Header, width: u32, height: u32, data: Vec<u8>) -> Image {
    Image {
        header,
        height,
        width,
        encoding: "mono8".into(),
        is_bigendian: 0,
        step: width,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msgs::Header;

    fn image(encoding: &str, width: u32, height: u32, step: u32, data: Vec<u8>) -> Image {
        Image {
            header: Header::default(),
            height,
            width,
            encoding: encoding.into(),
            is_bigendian: 0,
            step,
            data,
        }
    }

    #[test]
    fn mono8_with_row_padding() {
        let msg = image("mono8", 2, 2, 4, vec![1, 2, 99, 99, 3, 4, 99, 99]);
        let luma = decode_luma(&msg).unwrap();
        assert_eq!(luma.pixels, LumaPixels::U8(vec![1, 2, 3, 4]));
        assert_eq!(luma.to_u16(), vec![256, 512, 768, 1024]);
        // The last row's padding may be omitted.
        let msg = image("mono8", 2, 2, 4, vec![1, 2, 99, 99, 3, 4]);
        assert_eq!(
            decode_luma(&msg).unwrap().pixels,
            LumaPixels::U8(vec![1, 2, 3, 4])
        );
    }

    #[test]
    fn bgr8_and_rgb8_agree() {
        // Pure red, green, blue, white, black.
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0];
        let bgr: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect();
        let from_rgb = decode_luma(&image("rgb8", 5, 1, 15, rgb)).unwrap();
        let from_bgr = decode_luma(&image("bgr8", 5, 1, 15, bgr)).unwrap();
        assert_eq!(from_rgb, from_bgr);
        assert_eq!(from_rgb.pixels, LumaPixels::U8(vec![76, 150, 29, 255, 0]));
    }

    #[test]
    fn bgra8_ignores_alpha() {
        let msg = image("bgra8", 1, 1, 4, vec![0, 0, 255, 7]);
        assert_eq!(decode_luma(&msg).unwrap().pixels, LumaPixels::U8(vec![76]));
    }

    #[test]
    fn mono16_endianness() {
        let mut msg = image("mono16", 2, 1, 4, vec![0x34, 0x12, 0xcd, 0xab]);
        assert_eq!(
            decode_luma(&msg).unwrap().pixels,
            LumaPixels::U16(vec![0x1234, 0xabcd])
        );
        msg.is_bigendian = 1;
        assert_eq!(
            decode_luma(&msg).unwrap().pixels,
            LumaPixels::U16(vec![0x3412, 0xcdab])
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(matches!(
            decode_luma(&image("bayer_rggb8", 1, 1, 1, vec![0])),
            Err(ImageConversionError::UnsupportedEncoding(_))
        ));
        assert!(matches!(
            decode_luma(&image("mono8", 4, 1, 3, vec![0; 4])),
            Err(ImageConversionError::StepTooSmall { .. })
        ));
        assert!(matches!(
            decode_luma(&image("rgb8", 2, 2, 6, vec![0; 11])),
            Err(ImageConversionError::DataTooShort { .. })
        ));
        assert!(matches!(
            decode_luma(&image("mono8", 0, 2, 0, vec![])),
            Err(ImageConversionError::EmptyImage { .. })
        ));
    }

    #[test]
    fn unit_float_range() {
        let luma = decode_luma(&image("mono8", 2, 1, 2, vec![0, 255])).unwrap();
        assert_eq!(luma.to_unit_f32(), vec![0.0, 1.0]);
        let raw = luma.to_basalt().unwrap();
        assert_eq!(raw.width(), 2);
    }
}
