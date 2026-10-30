//! 8-bit RGB images, the input of the object detectors.

use std::fmt;

/// An interleaved 8-bit RGB image, row-major, `width * height * 3` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

/// The pixel buffer does not have the size the dimensions call for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSizeError {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub len: usize,
}

impl fmt::Display for ImageSizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}×{} image with {} channel(s) needs {} bytes, got {}",
            self.width,
            self.height,
            self.channels,
            self.width * self.height * self.channels,
            self.len
        )
    }
}

impl std::error::Error for ImageSizeError {}

impl RgbImage {
    /// An image from interleaved RGB bytes.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self, ImageSizeError> {
        if data.len() != width * height * 3 {
            return Err(ImageSizeError {
                width,
                height,
                channels: 3,
                len: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// An image from one 8-bit grey channel, copied into R, G and B (for the
    /// grayscale cameras of EuRoC and TUM-VI).
    pub fn from_gray(width: usize, height: usize, gray: &[u8]) -> Result<Self, ImageSizeError> {
        if gray.len() != width * height {
            return Err(ImageSizeError {
                width,
                height,
                channels: 1,
                len: gray.len(),
            });
        }
        let data = gray.iter().flat_map(|&v| [v, v, v]).collect();
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// The interleaved RGB bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The `[r, g, b]` value at column `x`, row `y`.
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let i = 3 * (y * self.width + x);
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_the_buffer_size_and_replicates_gray() {
        assert!(RgbImage::new(2, 2, vec![0; 11]).is_err());
        let image = RgbImage::from_gray(2, 1, &[10, 200]).unwrap();
        assert_eq!(image.pixel(1, 0), [200, 200, 200]);
        assert_eq!(image.data().len(), 6);
        let error = RgbImage::from_gray(3, 3, &[0; 8]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "3×3 image with 1 channel(s) needs 9 bytes, got 8"
        );
    }
}
