//! Local contrast equalization for the ORB frontend of map building and map
//! localization.
//!
//! ORB compares the brightness of nearby pixels, so it is unaffected when a
//! whole image gets brighter or darker. A change of lighting across a scene
//! (ceiling lights at night, daylight from the windows by day) is different:
//! it moves the shading and flattens the contrast of the darker parts, which
//! ORB's corners and descriptors depend on. Contrast-limited adaptive
//! histogram equalization (CLAHE) evens the local contrast out:
//!
//! - the image is split into a grid of tiles, and each tile's brightness
//!   histogram is equalized (stretched over the whole range);
//! - each histogram is first clipped at `clip_limit` times the count of a
//!   flat histogram, with the excess spread over all bins, so a nearly
//!   uniform tile does not turn its sensor noise into texture;
//! - each pixel blends the mappings of the four tiles around it, so the
//!   tile borders leave no seams.
//!
//! Pixels are Basalt's raw `u16` (8-bit values shifted left by 8); the
//! equalization works on the high byte and writes the result back the same
//! way. Pixels at 0 stay 0, so the black area around a fisheye lens is not
//! brightened.

use crate::pyramid::RawU16Image;

/// Settings of [`LocalContrast::apply`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalContrast {
    /// Tiles along each image side.
    pub tiles: usize,
    /// Histogram clip limit, in multiples of a flat histogram's bin count.
    pub clip_limit: f64,
}

impl Default for LocalContrast {
    fn default() -> Self {
        Self {
            tiles: 8,
            clip_limit: 2.0,
        }
    }
}

/// Bilinear interpolation weights along one image axis: for each pixel, the
/// lower and upper tile and the upper tile's weight, between tile centres.
fn axis_weights(len: usize, tiles: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<usize>) {
    let bounds = (0..=tiles).map(|i| i * len / tiles).collect::<Vec<_>>();
    let centres = bounds
        .windows(2)
        .map(|w| (w[0] + w[1]) as f64 / 2.0)
        .collect::<Vec<_>>();
    let mut lower = Vec::with_capacity(len);
    let mut upper = Vec::with_capacity(len);
    let mut weight = Vec::with_capacity(len);
    for position in 0..len {
        let p = position as f64;
        let tile = centres.iter().rposition(|&c| c <= p).unwrap_or(0);
        if p <= centres[0] {
            lower.push(0);
            upper.push(0);
            weight.push(0.0);
        } else if tile + 1 >= tiles {
            lower.push(tiles - 1);
            upper.push(tiles - 1);
            weight.push(0.0);
        } else {
            lower.push(tile);
            upper.push(tile + 1);
            weight.push((p - centres[tile]) / (centres[tile + 1] - centres[tile]));
        }
    }
    (lower, upper, weight, bounds)
}

impl LocalContrast {
    /// The image with its local contrast equalized.
    pub fn apply(&self, image: &RawU16Image) -> RawU16Image {
        let (width, height) = (image.width(), image.height());
        let tiles = self.tiles.clamp(1, width.min(height));
        let pixels = image.pixels();
        let value = |x: usize, y: usize| usize::from(pixels[y * width + x] >> 8);
        let (row_lo, row_hi, row_w, row_bounds) = axis_weights(height, tiles);
        let (col_lo, col_hi, col_w, col_bounds) = axis_weights(width, tiles);

        // Per tile, the clipped-histogram equalization of its 256 levels.
        let mut maps = vec![[0.0_f64; 256]; tiles * tiles];
        for ty in 0..tiles {
            for tx in 0..tiles {
                let mut histogram = [0.0_f64; 256];
                for y in row_bounds[ty]..row_bounds[ty + 1] {
                    for x in col_bounds[tx]..col_bounds[tx + 1] {
                        histogram[value(x, y)] += 1.0;
                    }
                }
                let count = ((row_bounds[ty + 1] - row_bounds[ty])
                    * (col_bounds[tx + 1] - col_bounds[tx])) as f64;
                let limit = (self.clip_limit * count / 256.0).max(1.0);
                let excess = histogram
                    .iter()
                    .map(|&bin| (bin - limit).max(0.0))
                    .sum::<f64>();
                let map = &mut maps[ty * tiles + tx];
                let mut total = 0.0;
                for (level, &bin) in histogram.iter().enumerate() {
                    total += bin.min(limit) + excess / 256.0;
                    map[level] = total;
                }
                for level in map.iter_mut() {
                    *level *= 255.0 / total;
                }
            }
        }

        let mut out = Vec::with_capacity(width * height);
        for y in 0..height {
            let (top, bottom, wy) = (row_lo[y], row_hi[y], row_w[y]);
            for x in 0..width {
                let level = value(x, y);
                if level == 0 {
                    out.push(0);
                    continue;
                }
                let (left, right, wx) = (col_lo[x], col_hi[x], col_w[x]);
                let at = |ty: usize, tx: usize| maps[ty * tiles + tx][level];
                let equalized = (1.0 - wy) * ((1.0 - wx) * at(top, left) + wx * at(top, right))
                    + wy * ((1.0 - wx) * at(bottom, left) + wx * at(bottom, right));
                out.push((equalized.round().clamp(0.0, 255.0) as u16) << 8);
            }
        }
        RawU16Image::new(width, height, out).expect("same dimensions as the input")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: usize, height: usize, f: impl Fn(usize, usize) -> u8) -> RawU16Image {
        RawU16Image::from_fn(width, height, |x, y| u16::from(f(x, y)) << 8).expect("image")
    }

    fn level(image: &RawU16Image, x: usize, y: usize) -> u8 {
        (image.pixel(x, y).expect("in bounds") >> 8) as u8
    }

    #[test]
    fn a_dim_low_contrast_image_is_spread_over_the_range() {
        // Levels 20..=51 in a fine repeating pattern: dim and flat. The
        // clip limit keeps the stretch moderate (more than three times the
        // range here); without it every tile would span 0..=255.
        let dim = image(96, 64, |x, y| 20 + ((x * 7 + y * 3) % 32) as u8);
        let equalized = LocalContrast::default().apply(&dim);
        let levels = equalized
            .pixels()
            .iter()
            .map(|&p| p >> 8)
            .collect::<Vec<_>>();
        let (min, max) = (levels.iter().min().unwrap(), levels.iter().max().unwrap());
        assert!(max - min > 3 * 31, "{min}..{max}");
    }

    #[test]
    fn black_stays_black_and_dimensions_are_kept() {
        let ring = image(80, 60, |x, y| {
            let (dx, dy) = (x as f64 - 40.0, y as f64 - 30.0);
            if dx * dx + dy * dy > 25.0 * 25.0 {
                0
            } else {
                60 + ((x + y) % 40) as u8
            }
        });
        let equalized = LocalContrast::default().apply(&ring);
        assert_eq!((equalized.width(), equalized.height()), (80, 60));
        assert_eq!(level(&equalized, 0, 0), 0);
        assert_eq!(level(&equalized, 79, 59), 0);
        assert!(level(&equalized, 40, 30) > 0);
        assert!(equalized.pixels().iter().all(|p| p & 0xff == 0));
    }

    #[test]
    fn levels_keep_their_order_at_a_fixed_place() {
        // One tile per side: a single monotonic mapping for every pixel.
        let ramp = image(64, 4, |x, _| 30 + x as u8);
        for clip_limit in [2.0, 1000.0] {
            let equalized = LocalContrast {
                tiles: 1,
                clip_limit,
            }
            .apply(&ramp);
            for x in 1..64 {
                assert!(level(&equalized, x, 0) > level(&equalized, x - 1, 0));
            }
            // Without clipping the brightest level maps to the top.
            assert_eq!(level(&equalized, 63, 0) == 255, clip_limit > 2.0);
        }
        // A tiny image cannot have more tiles than pixels per side.
        let tiny = image(3, 2, |x, y| (x * 40 + y * 80 + 10) as u8);
        let equalized = LocalContrast::default().apply(&tiny);
        assert_eq!(equalized.pixels().len(), 6);
    }
}
