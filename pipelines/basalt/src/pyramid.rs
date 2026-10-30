//! Raw `u16` image storage and the Basalt five-tap binomial pyramid.
//!
//! The downsample operation follows `ManagedImagePyr::subsample` from the
//! pinned `basalt-headers` revision: a separable `[1, 4, 6, 4, 1]` kernel,
//! reflection at the image boundary, decimation by two, and integer rounding
//! `(accumulator + 128) >> 8`. `max_level` is an index, so a pyramid built
//! with `max_level = N` contains levels `0..=N`.

use crate::fma::FusedMulAdd;
use nalgebra::Vector2;
use thiserror::Error;

const BINOMIAL5: [i32; 5] = [1, 4, 6, 4, 1];
const BINOMIAL5_MIN_INPUT_DIMENSION: usize = 3;

/// A contiguous row-major raw image with unsigned 16-bit pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawU16Image {
    width: usize,
    height: usize,
    pixels: Vec<u16>,
}

impl RawU16Image {
    pub fn new(width: usize, height: usize, pixels: Vec<u16>) -> Result<Self, ImageError> {
        let expected = width
            .checked_mul(height)
            .ok_or(ImageError::DimensionOverflow)?;
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions { width, height });
        }
        if pixels.len() != expected {
            return Err(ImageError::PixelCountMismatch {
                width,
                height,
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn from_fn<F>(width: usize, height: usize, mut f: F) -> Result<Self, ImageError>
    where
        F: FnMut(usize, usize) -> u16,
    {
        let count = width
            .checked_mul(height)
            .ok_or(ImageError::DimensionOverflow)?;
        let mut pixels = Vec::with_capacity(count);
        for y in 0..height {
            for x in 0..width {
                pixels.push(f(x, y));
            }
        }
        Self::new(width, height, pixels)
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[u16] {
        &self.pixels
    }

    pub fn pixel(&self, x: usize, y: usize) -> Option<u16> {
        if x < self.width && y < self.height {
            Some(self.pixels[y * self.width + x])
        } else {
            None
        }
    }

    /// Basalt's floating-point bounds convention for interpolation.
    ///
    /// A floating coordinate requires four neighbouring pixels, and `border`
    /// is an additional margin. Thus `border = 2` matches the optical-flow
    /// patch's `InBounds(p, 2)` call before central-difference gradients.
    pub fn in_bounds(&self, point: Vector2<f32>, border: f32) -> bool {
        let offset = 1.0_f32;
        border <= point.x
            && point.x < self.width as f32 - border - offset
            && border <= point.y
            && point.y < self.height as f32 - border - offset
    }

    /// Bilinear interpolation matching `Image::interp<float>`.
    pub fn interp(&self, point: Vector2<f32>) -> Option<f32> {
        if !self.in_bounds(point, 0.0) {
            return None;
        }
        Some(self.interp_in_bounds(point))
    }

    /// [`Self::interp`] for a point the caller has already checked with
    /// [`Self::in_bounds`] (any border `>= 0`).
    #[inline]
    pub(crate) fn interp_in_bounds(&self, point: Vector2<f32>) -> f32 {
        let ix = point.x as usize;
        let iy = point.y as usize;
        let dx = point.x - ix as f32;
        let dy = point.y - iy as f32;
        let [p00, p10] = self.pixel_pair(ix, iy);
        let [p01, p11] = self.pixel_pair(ix, iy + 1);
        bilinear_fma(p00, p01, p10, p11, dx, dy)
    }

    /// Returns `[value, dvalue/dx, dvalue/dy]` using Basalt's central-
    /// difference image followed by bilinear interpolation.
    pub fn interp_grad(&self, point: Vector2<f32>) -> Option<[f32; 3]> {
        if !self.in_bounds(point, 1.0) {
            return None;
        }
        Some(self.interp_grad_in_bounds(point))
    }

    /// [`Self::interp_grad`] for a point the caller has already checked with
    /// [`Self::in_bounds`] (any border `>= 1`). The five bilinear samples
    /// share one 4x4 neighbourhood, which is read once.
    #[inline]
    pub(crate) fn interp_grad_in_bounds(&self, point: Vector2<f32>) -> [f32; 3] {
        let ix = point.x as usize;
        let iy = point.y as usize;
        let dx = point.x - ix as f32;
        let dy = point.y - iy as f32;
        // Rows iy-1 and iy+2 need columns ix..=ix+1; rows iy and iy+1 need
        // ix-1..=ix+2.
        let [above0, above1] = self.pixel_pair(ix, iy - 1);
        let [row_m, row0, row1, row2] = self.pixel_quad(ix - 1, iy);
        let [next_m, next0, next1, next2] = self.pixel_quad(ix - 1, iy + 1);
        let [below0, below1] = self.pixel_pair(ix, iy + 2);

        let value = bilinear_fma(row0, next0, row1, next1, dx, dy);
        let value_x_minus = bilinear_fma(row_m, next_m, row0, next0, dx, dy);
        let value_x_plus = bilinear_fma(row1, next1, row2, next2, dx, dy);
        let value_y_minus = bilinear_fma(above0, row0, above1, row1, dx, dy);
        let value_y_plus = bilinear_fma(next0, below0, next1, below1, dx, dy);
        [
            value,
            0.5 * (value_x_plus - value_x_minus),
            0.5 * (value_y_plus - value_y_minus),
        ]
    }

    /// Pixels `(x, y)` and `(x + 1, y)`.
    #[inline(always)]
    fn pixel_pair(&self, x: usize, y: usize) -> [f32; 2] {
        let start = y * self.width + x;
        let row: &[u16; 2] = self.pixels[start..start + 2]
            .try_into()
            .expect("two pixels");
        row.map(f32::from)
    }

    /// Pixels `(x, y)` to `(x + 3, y)`.
    #[inline(always)]
    fn pixel_quad(&self, x: usize, y: usize) -> [f32; 4] {
        let start = y * self.width + x;
        let row: &[u16; 4] = self.pixels[start..start + 4]
            .try_into()
            .expect("four pixels");
        row.map(f32::from)
    }

    /// Applies the pinned Basalt 5x5 binomial filter and decimates by two.
    pub fn subsample_binomial5(&self) -> Result<Self, ImageError> {
        let mut scratch = Vec::new();
        self.subsample_binomial5_with_scratch(&mut scratch)
    }

    /// Applies the pinned filter while reusing the caller's vertical-pass
    /// storage.  The scratch contents are intentionally not cleared: every
    /// element read by the horizontal pass is assigned exactly once by the
    /// vertical pass first, so retaining the allocation cannot affect the
    /// arithmetic or traversal order.
    fn subsample_binomial5_with_scratch(&self, scratch: &mut Vec<i32>) -> Result<Self, ImageError> {
        let output_width = self.width / 2;
        let output_height = self.height / 2;
        // The reflected five-tap stencil below reaches source index 2 for
        // the first output sample.  A dimension of two would therefore
        // produce an in-bounds-looking output dimension of one but still
        // index past the source row/column.  Treat it as the same terminal
        // level as a zero-output dimension instead of relying on a panic.
        if output_width == 0
            || output_height == 0
            || self.width < BINOMIAL5_MIN_INPUT_DIMENSION
            || self.height < BINOMIAL5_MIN_INPUT_DIMENSION
        {
            return Err(ImageError::LevelTooDeep {
                width: self.width,
                height: self.height,
            });
        }

        let vertical_len = output_height
            .checked_mul(self.width)
            .ok_or(ImageError::DimensionOverflow)?;
        if scratch.len() < vertical_len {
            scratch.resize(vertical_len, 0_i32);
        }
        for row in 0..output_height {
            let source_rows = [
                (2 * row).abs_diff(2),
                (2 * row).abs_diff(1),
                2 * row,
                border101(2 * row + 1, self.height),
                border101(2 * row + 2, self.height),
            ];
            for col in 0..self.width {
                let mut accumulator = 0_i32;
                for (weight, source_row) in BINOMIAL5.iter().zip(source_rows) {
                    accumulator += *weight * i32::from(self.pixels[source_row * self.width + col]);
                }
                scratch[row * self.width + col] = accumulator;
            }
        }

        let mut output = vec![0_u16; output_width * output_height];
        for col in 0..output_width {
            let source_cols = [
                (2 * col).abs_diff(2),
                (2 * col).abs_diff(1),
                2 * col,
                border101(2 * col + 1, self.width),
                border101(2 * col + 2, self.width),
            ];
            for row in 0..output_height {
                let mut accumulator = 0_i32;
                for (weight, source_col) in BINOMIAL5.iter().zip(source_cols) {
                    accumulator += *weight * scratch[row * self.width + source_col];
                }
                output[row * output_width + col] = ((accumulator + (1 << 7)) >> 8) as u16;
            }
        }

        Self::new(output_width, output_height, output)
    }
}

/// Evaluate Basalt's bilinear interpolation with the scalar contraction used
/// by the pinned native build.  The upstream expression is compiled to one
/// rounded product for the second term followed by three fused multiply-adds;
/// spelling that contract out keeps the result independent of Rust's target
/// FMA/codegen choices.
#[inline]
fn bilinear_fma(p00: f32, p01: f32, p10: f32, p11: f32, dx: f32, dy: f32) -> f32 {
    let ddx = 1.0 - dx;
    let ddy = 1.0 - dy;
    let term1 = ddx * dy * p01;
    let acc = (ddx * ddy).fused_mul_add(p00, term1);
    let acc = (dx * ddy).fused_mul_add(p10, acc);
    (dx * dy).fused_mul_add(p11, acc)
}

/// A pyramid whose level indices are exactly `0..=max_level`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawU16Pyramid {
    levels: Vec<RawU16Image>,
}

impl RawU16Pyramid {
    pub fn from_image(image: RawU16Image, max_level: usize) -> Result<Self, ImageError> {
        let mut scratch = Vec::new();
        Self::from_image_with_scratch(image, max_level, &mut scratch)
    }

    pub(crate) fn from_image_with_scratch(
        image: RawU16Image,
        max_level: usize,
        scratch: &mut Vec<i32>,
    ) -> Result<Self, ImageError> {
        let capacity = max_level
            .checked_add(1)
            .ok_or(ImageError::DimensionOverflow)?;
        let mut levels = Vec::with_capacity(capacity);
        levels.push(image);
        for _ in 0..max_level {
            let next = levels
                .last()
                .expect("level zero is always present")
                .subsample_binomial5_with_scratch(scratch)?;
            levels.push(next);
        }
        Ok(Self { levels })
    }

    pub fn levels(&self) -> &[RawU16Image] {
        &self.levels
    }

    pub fn level(&self, level: usize) -> Option<&RawU16Image> {
        self.levels.get(level)
    }

    pub fn max_level(&self) -> usize {
        self.levels.len() - 1
    }
}

const fn border101(x: usize, height: usize) -> usize {
    height - 1 - (height - 1).abs_diff(x)
}

/// Errors raised by raw image and pyramid construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ImageError {
    #[error("image dimensions must be non-zero, got {width}x{height}")]
    InvalidDimensions { width: usize, height: usize },
    #[error("image dimensions overflow a pixel count")]
    DimensionOverflow,
    #[error("expected {expected} pixels for {width}x{height}, got {actual}")]
    PixelCountMismatch {
        width: usize,
        height: usize,
        expected: usize,
        actual: usize,
    },
    #[error("cannot build another pyramid level from {width}x{height}")]
    LevelTooDeep { width: usize, height: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-optimization `interp` / `interp_grad`: one bounds-checked
    /// read per pixel, five independent bilinear samples.
    fn reference_bilinear(image: &RawU16Image, point: Vector2<f32>, x0: usize, y0: usize) -> f32 {
        let dx = point.x - (point.x as usize) as f32;
        let dy = point.y - (point.y as usize) as f32;
        let p00 = image.pixels[y0 * image.width + x0] as f32;
        let p01 = image.pixels[(y0 + 1) * image.width + x0] as f32;
        let p10 = image.pixels[y0 * image.width + x0 + 1] as f32;
        let p11 = image.pixels[(y0 + 1) * image.width + x0 + 1] as f32;
        bilinear_fma(p00, p01, p10, p11, dx, dy)
    }

    fn reference_interp_grad(image: &RawU16Image, point: Vector2<f32>) -> [f32; 3] {
        let (ix, iy) = (point.x as usize, point.y as usize);
        let bilinear = |x0, y0| reference_bilinear(image, point, x0, y0);
        [
            bilinear(ix, iy),
            0.5 * (bilinear(ix + 1, iy) - bilinear(ix - 1, iy)),
            0.5 * (bilinear(ix, iy + 1) - bilinear(ix, iy - 1)),
        ]
    }

    #[test]
    fn interpolation_matches_the_reference_bit_for_bit() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let image = RawU16Image::from_fn(37, 29, |_, _| next() as u16).unwrap();
        let mut checked = 0;
        for _ in 0..20_000 {
            let point = Vector2::new(
                (next() % 40_000) as f32 / 1000.0 - 1.0,
                (next() % 32_000) as f32 / 1000.0 - 1.0,
            );
            if image.in_bounds(point, 0.0) {
                let (ix, iy) = (point.x as usize, point.y as usize);
                assert_eq!(
                    image.interp(point).map(f32::to_bits),
                    Some(reference_bilinear(&image, point, ix, iy).to_bits())
                );
            } else {
                assert_eq!(image.interp(point), None);
            }
            if image.in_bounds(point, 1.0) {
                assert_eq!(
                    image.interp_grad(point).map(|v| v.map(f32::to_bits)),
                    Some(reference_interp_grad(&image, point).map(f32::to_bits)),
                    "{point:?}"
                );
                checked += 1;
            } else {
                assert_eq!(image.interp_grad(point), None);
            }
        }
        assert!(checked > 10_000);
    }

    #[test]
    fn border_reflection_matches_basalt101() {
        assert_eq!(border101(8, 8), 6);
        assert_eq!(border101(9, 8), 5);
    }

    #[test]
    fn levels_are_indexed_inclusive_of_max_level() {
        let image = RawU16Image::new(8, 8, (1..=64).collect()).unwrap();
        let pyramid = RawU16Pyramid::from_image(image, 2).unwrap();
        assert_eq!(pyramid.levels().len(), 3);
        assert_eq!(pyramid.level(0).unwrap().width(), 8);
        assert_eq!(pyramid.level(1).unwrap().width(), 4);
        assert_eq!(pyramid.level(2).unwrap().width(), 2);
    }

    #[test]
    fn pyramid_scratch_capacity_reuses_allocation_without_changing_levels() {
        let image = RawU16Image::from_fn(32, 24, |x, y| {
            ((x * 977 + y * 613 + x * y * 17) & 0xffff) as u16
        })
        .unwrap();
        let expected = RawU16Pyramid::from_image(image.clone(), 3).unwrap();
        let mut scratch = Vec::new();
        let actual = RawU16Pyramid::from_image_with_scratch(image, 3, &mut scratch).unwrap();
        assert_eq!(actual, expected);

        let capacity = scratch.capacity();
        let pointer = scratch.as_ptr();
        let second_image = RawU16Image::from_fn(32, 24, |x, y| {
            ((x * 1499 + y * 431 + x * y * 29 + 7) & 0xffff) as u16
        })
        .unwrap();
        let expected_second = RawU16Pyramid::from_image(second_image.clone(), 3).unwrap();
        let actual_second =
            RawU16Pyramid::from_image_with_scratch(second_image, 3, &mut scratch).unwrap();
        assert_eq!(actual_second, expected_second);
        assert_eq!(scratch.capacity(), capacity);
        assert_eq!(scratch.as_ptr(), pointer);
        assert!(capacity >= (24 / 2) * 32);
    }

    #[test]
    fn pyramid_rejects_two_pixel_tap_dimension_without_indexing_past_input() {
        for (width, height, max_level) in [(4, 4, 2), (2, 8, 1), (8, 2, 1)] {
            let image = RawU16Image::from_fn(width, height, |x, y| (x + y) as u16).unwrap();
            let error = RawU16Pyramid::from_image(image, max_level).unwrap_err();
            assert!(matches!(error, ImageError::LevelTooDeep { .. }));
        }

        // Three pixels are sufficient for the reflected index-2 tap and
        // remain a valid one-level input.
        let image = RawU16Image::from_fn(3, 3, |x, y| (x + y) as u16).unwrap();
        assert!(RawU16Pyramid::from_image(image, 1).is_ok());
    }
}
