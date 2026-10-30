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
        let r = radius as usize;
        let pixels = image.pixels();
        // Along each row: inside it, one tap at a time across the row (each
        // output still sums its taps in order); within `radius` of its ends,
        // the taps reflected.
        let mut rows = vec![0.0_f32; width * height];
        let mut line = vec![0.0_f32; width];
        let inside = r..width.saturating_sub(r).max(r);
        for y in 0..height {
            for (value, &pixel) in line.iter_mut().zip(&pixels[y * width..(y + 1) * width]) {
                *value = f32::from(pixel);
            }
            let out = &mut rows[y * width..(y + 1) * width];
            for (tap, weight) in kernel.iter().enumerate().filter(|_| !inside.is_empty()) {
                let source = &line[inside.start + tap - r..inside.end + tap - r];
                for (out, value) in out[inside.clone()].iter_mut().zip(source) {
                    *out += weight * value;
                }
            }
            for x in (0..width).filter(|x| !inside.contains(x)) {
                out[x] = kernel
                    .iter()
                    .zip(-radius..=radius)
                    .map(|(weight, i)| weight * line[reflect(x as isize + i, width)])
                    .sum();
            }
        }
        // Down each column, a whole row at a time.
        let mut out = vec![0_u16; width * height];
        let mut sum = vec![0.0_f32; width];
        for y in 0..height {
            sum.iter_mut().for_each(|value| *value = 0.0);
            for (weight, i) in kernel.iter().zip(-radius..=radius) {
                let source = reflect(y as isize + i, height);
                for (value, &row) in sum
                    .iter_mut()
                    .zip(&rows[source * width..(source + 1) * width])
                {
                    *value += weight * row;
                }
            }
            let target = &mut out[y * width..(y + 1) * width];
            for ((out, &value), &pixel) in target
                .iter_mut()
                .zip(&sum)
                .zip(&pixels[y * width..(y + 1) * width])
            {
                if pixel != 0 {
                    *out = value.round().clamp(0.0, f32::from(u16::MAX)) as u16;
                }
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
    fn matches_a_direct_two_dimensional_blur() {
        let image = RawU16Image::from_fn(23, 17, |x, y| ((x * 7 + y * 13) % 31) as u16 * 1000 + 1)
            .expect("image");
        let smoothing = GaussianSmoothing { sigma: 1.2 };
        let smoothed = smoothing.apply(&image);
        let radius = (3.0 * smoothing.sigma).ceil() as isize;
        let weight = |i: isize| (-((i * i) as f64) / (2.0 * 1.2 * 1.2)).exp();
        let total: f64 = (-radius..=radius).map(weight).sum();
        let reflect = |i: isize, len: isize| {
            let period = 2 * (len - 1);
            let i = i.rem_euclid(period);
            if i < len {
                i
            } else {
                period - i
            }
        };
        for y in 0..17_isize {
            for x in 0..23_isize {
                let mut value = 0.0;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let pixel = image
                            .pixel(reflect(x + dx, 23) as usize, reflect(y + dy, 17) as usize)
                            .expect("pixel");
                        value += weight(dx) * weight(dy) * f64::from(pixel);
                    }
                }
                let expected = value / (total * total);
                let got = f64::from(smoothed.pixel(x as usize, y as usize).expect("pixel"));
                assert!(
                    (got - expected).abs() <= 1.0,
                    "{x} {y}: {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn rows_match_the_reflected_sums_to_the_bit() {
        // Each output summed tap by tap, every read reflected, then down
        // each column the same way.
        let image =
            RawU16Image::from_fn(31, 9, |x, y| ((x * 7919 + y * 104_729) % 65_521) as u16 + 1)
                .expect("image");
        let smoothing = GaussianSmoothing { sigma: 1.5 };
        let radius = (3.0 * smoothing.sigma).ceil() as isize;
        let mut kernel = (-radius..=radius)
            .map(|i| (-((i * i) as f64) / (2.0 * 1.5 * 1.5)).exp() as f32)
            .collect::<Vec<_>>();
        let total: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|weight| *weight /= total);
        let reflect = |i: isize, len: isize| {
            let i = i.rem_euclid(2 * (len - 1));
            if i < len {
                i
            } else {
                2 * (len - 1) - i
            }
        };
        let sum = |at: &dyn Fn(isize) -> f32| -> f32 {
            kernel
                .iter()
                .zip(-radius..=radius)
                .map(|(weight, i)| weight * at(i))
                .sum()
        };
        let pixel =
            |x: isize, y: isize| f32::from(image.pixel(x as usize, y as usize).expect("pixel"));
        let row = |x: isize, y: isize| sum(&|i| pixel(reflect(x + i, 31), y));
        let smoothed = smoothing.apply(&image);
        for y in 0..9 {
            for x in 0..31 {
                let column = sum(&|i| row(x, reflect(y + i, 9)));
                let expected = column.round().clamp(0.0, f32::from(u16::MAX)) as u16;
                assert_eq!(
                    smoothed.pixel(x as usize, y as usize),
                    Some(expected),
                    "{x} {y}"
                );
            }
        }
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
