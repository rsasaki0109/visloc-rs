//! Gaussian smoothing for the ORB frontend of map building and map
//! localization.
//!
//! ORB's descriptor compares single pixels pairwise. On a dim, grainy image
//! (OpenLORIS-Scene's corridor, under fluorescent light) sensor noise flips
//! many of those comparisons, so the same corner seen in two sessions gets
//! descriptors too far apart to match, and the detector picks noise as
//! corners. OpenCV's ORB blurs before its descriptor for this reason.
//! Blurring the image before the whole frontend, keypoints and descriptors
//! alike, raised the geometrically consistent matches between
//! near-identical views of two corridor sessions from 22 to 38 per image
//! pair (sigma 1.5).
//!
//! Pixels are Basalt's raw `u16`; the blur keeps the black area around a
//! fisheye lens black (pixels at 0 stay 0).

use crate::pyramid::RawU16Image;

/// Settings of [`GaussianSmoothing::apply`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianSmoothing {
    /// Standard deviation of the Gaussian, in pixels.
    pub sigma: f64,
}

impl Default for GaussianSmoothing {
    fn default() -> Self {
        Self { sigma: 1.5 }
    }
}

impl GaussianSmoothing {
    /// The image blurred by a separable Gaussian out to three sigma, its
    /// borders reflected.
    pub fn apply(&self, image: &RawU16Image) -> RawU16Image {
        let (width, height) = (image.width(), image.height());
        if self.sigma.is_nan() || self.sigma <= 0.0 || width == 0 || height == 0 {
            return image.clone();
        }
        let radius = (3.0 * self.sigma).ceil() as isize;
        let mut kernel = (-radius..=radius)
            .map(|i| (-((i * i) as f64) / (2.0 * self.sigma * self.sigma)).exp() as f32)
            .collect::<Vec<_>>();
        let total: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|weight| *weight /= total);
        let reflect = |i: isize, len: usize| -> usize {
            let len = len as isize;
            if len == 1 {
                return 0;
            }
            let period = 2 * (len - 1);
            let i = i.rem_euclid(period);
            (if i < len { i } else { period - i }) as usize
        };
        let pixels = image.pixels();
        let mut rows = vec![0.0_f32; width * height];
        for y in 0..height {
            let row = &pixels[y * width..(y + 1) * width];
            for x in 0..width {
                rows[y * width + x] = kernel
                    .iter()
                    .zip(-radius..=radius)
                    .map(|(weight, i)| weight * f32::from(row[reflect(x as isize + i, width)]))
                    .sum();
            }
        }
        let mut out = vec![0_u16; width * height];
        for y in 0..height {
            for x in 0..width {
                if pixels[y * width + x] == 0 {
                    continue;
                }
                let value: f32 = kernel
                    .iter()
                    .zip(-radius..=radius)
                    .map(|(weight, i)| weight * rows[reflect(y as isize + i, height) * width + x])
                    .sum();
                out[y * width + x] = value.round().clamp(0.0, f32::from(u16::MAX)) as u16;
            }
        }
        RawU16Image::new(width, height, out).expect("same dimensions as the input")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_image_stays_flat_and_zero_stays_zero() {
        let image =
            RawU16Image::from_fn(20, 12, |x, _| if x < 3 { 0 } else { 100 << 8 }).expect("image");
        let smoothed = GaussianSmoothing::default().apply(&image);
        for y in 0..12 {
            assert_eq!(smoothed.pixel(0, y), Some(0));
            assert_eq!(smoothed.pixel(15, y), Some(100 << 8));
        }
        // Next to the black border the blur reaches into it.
        assert!(smoothed.pixel(3, 5).expect("pixel") < 100 << 8);
    }

    #[test]
    fn noise_is_damped_and_the_mean_kept() {
        // A checkerboard of +-20 grey levels around 128.
        let image =
            RawU16Image::from_fn(
                40,
                40,
                |x, y| {
                    if (x + y) % 2 == 0 {
                        148 << 8
                    } else {
                        108 << 8
                    }
                },
            )
            .expect("image");
        let smoothed = GaussianSmoothing::default().apply(&image);
        for y in 10..30 {
            for x in 10..30 {
                let value = f64::from(smoothed.pixel(x, y).expect("pixel")) / 256.0;
                assert!((value - 128.0).abs() < 1.0, "{x} {y}: {value}");
            }
        }
        // Without a sigma, the image is as given.
        assert_eq!(GaussianSmoothing { sigma: 0.0 }.apply(&image), image);
    }
}
