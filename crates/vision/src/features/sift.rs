//! Classical SIFT feature extraction — a from-scratch pure-Rust port.
//!
//! Algorithm per Lowe, "Distinctive Image Features from Scale-Invariant
//! Keypoints" (IJCV 2004); VLFeat's BSD-licensed `sift.c` serves as the
//! legally-clean behavioural reference for the scale-space construction
//! (`docs/colmap_port_plan.md` §License: COLMAP's CPU path itself uses
//! VLFeat, while `SiftGPU` and `LSD` are license-encumbered and must not be
//! ported or depended on).
//!
//! Pipeline:
//! 1. **Scale space** — Gaussian pyramid octaves; each holds
//!    `intervals + 3` blurred levels so DoG has `intervals + 2` layers. The
//!    input is doubled per Lowe §3.3.
//! 2. **DoG extrema** — 3×3×3 neighbourhood maxima/minima with the contrast
//!    gate and Lowe's edge test (principal-curvature ratio r).
//! 3. **Orientation** — 36-bin gradient histogram over a Gaussian window of
//!    1.5σ; every peak within 80 % of the maximum spawns an oriented copy.
//! 4. **Descriptor** — 4×4 cells × 8 orientation bins over a 4σ window
//!    rotated to the keypoint orientation, trilinearly weighted,
//!    L2-normalized to 128 floats.
//!
//! Determinism: no randomness; identical inputs give identical outputs.

use nalgebra::Point2;

use super::FeatureSet;

/// Tunables mirroring Lowe's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct SiftConfig {
    /// Number of octaves explored. `0` derives the count from image size.
    pub octaves: usize,
    /// Difference-of-Gaussian intervals per octave.
    pub intervals: usize,
    /// Assumed input blur σ (0.5 for camera images).
    pub sigma_input: f64,
    /// Base σ at the doubled first octave (1.6).
    pub sigma_base: f64,
    /// Minimum |DoG| for an extremum to survive.
    pub contrast_threshold: f64,
    /// Edge-test curvature ratio threshold (Lowe r = 10).
    pub edge_threshold: f64,
    /// Cap on returned keypoints (strongest contrast first when exceeded).
    pub max_keypoints: usize,
}

impl Default for SiftConfig {
    fn default() -> Self {
        Self {
            octaves: 0,
            intervals: 3,
            sigma_input: 0.5,
            sigma_base: 1.6,
            contrast_threshold: 0.02,
            edge_threshold: 10.0,
            max_keypoints: usize::MAX,
        }
    }
}

/// Errors surfaced by [`extract_sift`].
#[derive(Debug, Clone, PartialEq)]
pub enum SiftError {
    /// The image is smaller than the minimum workable size.
    ImageTooSmall { width: usize, height: usize },
}

impl std::fmt::Display for SiftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SiftError::ImageTooSmall { width, height } => {
                write!(f, "image {width}x{height} too small for SIFT")
            }
        }
    }
}
impl std::error::Error for SiftError {}

/// Row-major grayscale image with values in `[0, 1]`.
pub struct GrayImage<'a> {
    pub width: usize,
    pub height: usize,
    pub pixels: &'a [f32],
}

impl<'a> GrayImage<'a> {
    pub fn new(width: usize, height: usize, pixels: &'a [f32]) -> Result<Self, SiftError> {
        if width < 16 || height < 16 || pixels.len() < width * height {
            return Err(SiftError::ImageTooSmall { width, height });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    fn get(&self, x: i64, y: i64) -> f64 {
        let xi = x.clamp(0, self.width as i64 - 1) as usize;
        let yi = y.clamp(0, self.height as i64 - 1) as usize;
        self.pixels[yi * self.width + xi] as f64
    }
}

#[derive(Clone)]
struct Layer {
    width: usize,
    height: usize,
    data: Vec<f64>,
}

impl Layer {
    fn get(&self, x: i64, y: i64) -> f64 {
        let xi = x.clamp(0, self.width as i64 - 1) as usize;
        let yi = y.clamp(0, self.height as i64 - 1) as usize;
        self.data[yi * self.width + xi]
    }
}

fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (sigma * 3.0).ceil().max(1.0) as i64;
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    let denom = 2.0 * sigma * sigma;
    for k in -radius..=radius {
        kernel.push((-(k * k) as f64 / denom).exp());
    }
    let sum: f64 = kernel.iter().sum();
    kernel.iter().map(|v| v / sum).collect()
}

fn blur(input: &Layer, sigma: f64) -> Layer {
    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() as i64 / 2;
    let mut tmp = vec![0.0f64; input.data.len()];
    for y in 0..input.height as i64 {
        for x in 0..input.width as i64 {
            let mut acc = 0.0;
            for (k, &w) in kernel.iter().enumerate() {
                acc += w * input.get(x + k as i64 - radius, y);
            }
            tmp[y as usize * input.width + x as usize] = acc;
        }
    }
    let horizontal = Layer {
        width: input.width,
        height: input.height,
        data: tmp,
    };
    let get_tmp = |x: i64, y: i64| -> f64 {
        let xi = x.clamp(0, horizontal.width as i64 - 1) as usize;
        let yi = y.clamp(0, horizontal.height as i64 - 1) as usize;
        horizontal.data[yi * horizontal.width + xi]
    };
    let mut out = vec![0.0f64; input.data.len()];
    for y in 0..input.height as i64 {
        for x in 0..input.width as i64 {
            let mut acc = 0.0;
            for (k, &w) in kernel.iter().enumerate() {
                acc += w * get_tmp(x, y + k as i64 - radius);
            }
            out[y as usize * input.width + x as usize] = acc;
        }
    }
    Layer {
        width: input.width,
        height: input.height,
        data: out,
    }
}

fn halve(layer: &Layer) -> Layer {
    let w = layer.width / 2;
    let h = layer.height / 2;
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = layer.get(2 * x as i64, 2 * y as i64);
        }
    }
    Layer {
        width: w,
        height: h,
        data: out,
    }
}

fn double_up(layer: &Layer) -> Layer {
    let w = layer.width * 2;
    let h = layer.height * 2;
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            // Nearest-neighbour at half-sample offsets keeps the doubling
            // simple; the σ_base pre-blur smooths the stair artefacts away.
            let sx = (x + 1).div_ceil(2).min(layer.width - 1);
            let sy = (y + 1).div_ceil(2).min(layer.height - 1);
            out[y * w + x] = layer.data[sy * layer.width + sx];
        }
    }
    Layer {
        width: w,
        height: h,
        data: out,
    }
}

/// A detected keypoint in the original image frame.
#[derive(Debug, Clone, PartialEq)]
pub struct SiftKeypoint {
    pub x: f64,
    pub y: f64,
    /// Absolute σ of the detection, mapped back to original-image pixels.
    pub sigma: f64,
    /// Dominant gradient orientation, radians in [0, 2π).
    pub orientation: f64,
    contrast: f64,
}

/// Extract SIFT keypoints and 128-d descriptors from a grayscale image.
///
/// The returned [`FeatureSet`] stores `(x, y)` keypoint locations and one
/// L2-normalized 128-dim descriptor per keypoint; descriptor order matches
/// keypoint order (strongest contrast first when `max_keypoints` truncates).
pub fn extract_sift(
    image: &GrayImage<'_>,
    config: &SiftConfig,
) -> Result<(Vec<SiftKeypoint>, Vec<Vec<f32>>), SiftError> {
    let intervals = config.intervals.max(1);
    let auto_octaves = ((image.width.min(image.height) as f64).log2() - 4.0).max(1.0) as usize;
    let num_octaves = if config.octaves == 0 {
        auto_octaves
    } else {
        config.octaves.min(auto_octaves)
    };

    // ---- Scale space over the doubled input --------------------------------
    let base = Layer {
        width: image.width,
        height: image.height,
        data: image.pixels.iter().map(|&p| p as f64).collect(),
    };
    let mut octave = double_up(&base);
    octave = blur(
        &octave,
        (config.sigma_base * config.sigma_base - (2.0 * config.sigma_input).powi(2))
            .max(1e-4)
            .sqrt(),
    );

    let k = (2.0f64).powf(1.0 / intervals as f64);
    let mut keypoints: Vec<SiftKeypoint> = Vec::new();

    for oct in 0..num_octaves {
        // `intervals + 3` Gaussian levels at σ_base·k^j, j = 0..=intervals+2
        // (Lowe §3.2); `gaussians[0]` is the octave base at σ_base.
        let mut gaussians = vec![octave.clone()];
        for j in 1..=(intervals + 2) {
            let target = config.sigma_base * k.powi(j as i32);
            let prev_target = config.sigma_base * k.powi(j as i32 - 1);
            let step = ((target * target - prev_target * prev_target).max(1e-6)).sqrt();
            gaussians.push(blur(gaussians.last().unwrap(), step));
        }
        let dogs: Vec<Layer> = (0..gaussians.len() - 1)
            .map(|i| {
                let (a, b) = (&gaussians[i], &gaussians[i + 1]);
                Layer {
                    width: a.width,
                    height: a.height,
                    data: (0..a.data.len()).map(|p| b.data[p] - a.data[p]).collect(),
                }
            })
            .collect();

        for di in 1..dogs.len().saturating_sub(1) {
            let (below, mid, above) = (&dogs[di - 1], &dogs[di], &dogs[di + 1]);
            for y in 1..mid.height as i64 - 1 {
                for x in 1..mid.width as i64 - 1 {
                    detect_extremum(
                        below,
                        mid,
                        above,
                        x,
                        y,
                        config,
                        oct,
                        di as i32,
                        k,
                        &mut keypoints,
                    );
                }
            }
        }

        if keypoints.len() >= config.max_keypoints {
            break;
        }
        if oct + 1 < num_octaves {
            octave = halve(&gaussians[intervals]);
        }
    }

    if keypoints.len() > config.max_keypoints {
        keypoints.sort_by(|a, b| b.contrast.total_cmp(&a.contrast));
        keypoints.truncate(config.max_keypoints);
    }
    let descriptors = keypoints
        .iter()
        .map(|kp| describe(image, kp))
        .collect::<Vec<_>>();
    Ok((keypoints, descriptors))
}

#[allow(clippy::too_many_arguments)]
fn detect_extremum(
    below: &Layer,
    mid: &Layer,
    above: &Layer,
    x: i64,
    y: i64,
    config: &SiftConfig,
    octave: usize,
    interval: i32,
    k: f64,
    out: &mut Vec<SiftKeypoint>,
) {
    let value = mid.get(x, y);
    if value.abs() < config.contrast_threshold {
        return;
    }
    let mut is_min = true;
    let mut is_max = true;
    for &(layer, dz) in &[(below, -1i64), (mid, 0), (above, 1)] {
        let _ = dz;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if std::ptr::eq(layer, mid) && dx == 0 && dy == 0 {
                    continue;
                }
                let neighbour = layer.get(x + dx, y + dy);
                if neighbour >= value {
                    is_max = false;
                }
                if neighbour <= value {
                    is_min = false;
                }
            }
        }
    }
    if !(is_min || is_max) {
        return;
    }
    // Lowe edge test.
    let dxx = mid.get(x + 1, y) + mid.get(x - 1, y) - 2.0 * value;
    let dyy = mid.get(x, y + 1) + mid.get(x, y - 1) - 2.0 * value;
    let dxy = (mid.get(x + 1, y + 1) - mid.get(x + 1, y - 1) - mid.get(x - 1, y + 1)
        + mid.get(x - 1, y - 1))
        / 4.0;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    if det <= 0.0 || tr * tr * config.edge_threshold > det * (config.edge_threshold + 1.0).powi(2) {
        return;
    }
    assign_orientations(mid, x, y, interval, k, value, octave, config, out);
}

#[allow(clippy::too_many_arguments)]
fn assign_orientations(
    mid: &Layer,
    x: i64,
    y: i64,
    interval: i32,
    k: f64,
    value: f64,
    octave: usize,
    config: &SiftConfig,
    out: &mut Vec<SiftKeypoint>,
) {
    // Orientation histogram on the DoG magnitude (a common simplification of
    // Lowe's Gaussian-blurred-image gradients).
    let hist_sigma = 1.5 * config.sigma_base * k.powi(interval) / 2.0f64.powi(octave as i32);
    let radius = (hist_sigma * 3.0).ceil() as i64;
    let mut hist = [0.0f64; 36];
    for wy in -radius..=radius {
        for wx in -radius..=radius {
            let gx = 0.5 * (mid.get(x + wx + 1, y) - mid.get(x + wx - 1, y));
            let gy = 0.5 * (mid.get(x + wx, y + 1) - mid.get(x + wx, y - 1));
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude <= f64::EPSILON {
                continue;
            }
            let weight = (-(wx * wx + wy * wy) as f64 / (2.0 * hist_sigma * hist_sigma)).exp();
            let angle = gy.atan2(gx).to_degrees().rem_euclid(360.0);
            let bin = ((angle / 10.0) as usize).min(35);
            hist[bin] += magnitude * weight;
        }
    }
    let max_hist = hist.iter().cloned().fold(f64::MIN, f64::max);
    if max_hist <= 0.0 {
        return;
    }
    // Layer → original-frame mapping: original = layer · 2^(octave−1)
    // (the first octave lives on the doubled grid).
    let upsample = (1usize << octave) as f64 / 2.0;
    for bin in 0..36 {
        if hist[bin] < 0.8 * max_hist {
            continue;
        }
        let prev = hist[(bin + 35) % 36];
        let next = hist[(bin + 1) % 36];
        let denom = 2.0 * (hist[bin] * 2.0 - prev - next);
        let delta = if denom.abs() > f64::EPSILON {
            ((prev - next) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        out.push(SiftKeypoint {
            x: x as f64 * upsample,
            y: y as f64 * upsample,
            sigma: config.sigma_base * k.powi(interval) * upsample,
            orientation: ((bin as f64 + delta) * 10.0).rem_euclid(360.0).to_radians(),
            contrast: value.abs(),
        });
    }
}

/// 128-dim descriptor: 4×4 cells × 8 orientation bins around the keypoint.
fn describe(image: &GrayImage<'_>, kp: &SiftKeypoint) -> Vec<f32> {
    const DIM: usize = 128;
    let mut desc = vec![0.0f64; DIM];
    let cell_size = (8.0 * kp.sigma).max(3.0); // window half-width ≈ 4σ·√2/√2
    let (cos, sin) = (kp.orientation.cos(), kp.orientation.sin());
    let half = (cell_size * 4.0f64.sqrt()) as i64;
    for dy in -half..=half {
        for dx in -half..=half {
            // Rotate sample offset into the keypoint frame.
            let rx = dx as f64 * cos - dy as f64 * sin;
            let ry = dx as f64 * sin + dy as f64 * cos;
            let px = (kp.x + dx as f64).round() as i64;
            let py = (kp.y + dy as f64).round() as i64;
            let gx = 0.5 * (image.get(px + 1, py) - image.get(px - 1, py));
            let gy = 0.5 * (image.get(px, py + 1) - image.get(px, py - 1));
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude <= f64::EPSILON {
                continue;
            }
            let theta = gy.atan2(gx) - kp.orientation;
            let weight = (-(rx * rx + ry * ry) / (2.0 * (cell_size * 2.0).powi(2))).exp();
            // Trilinear bin assignment across cell and orientation bins.
            let cx = (rx / cell_size) + 2.0 - 0.5;
            let cy = (ry / cell_size) + 2.0 - 0.5;
            let obin =
                ((theta.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU) * 8.0) % 8.0;
            let o0 = obin.floor() as i64;
            let dobin = obin - o0 as f64;
            for dc in 0..2 {
                for dr in 0..2 {
                    for dth in 0..2 {
                        let ci = cx.floor() as i64 + dc;
                        let ri = cy.floor() as i64 + dr;
                        let oi = ((o0 + dth as i64) % 8 + 8) % 8;
                        if !(0..4).contains(&ci) || !(0..4).contains(&ri) {
                            continue;
                        }
                        let w_x = (1.0 - (cx - ci as f64).abs()).clamp(0.0, 1.0);
                        let w_y = (1.0 - (cy - ri as f64).abs()).clamp(0.0, 1.0);
                        let w_o = if dth == 0 { 1.0 - dobin } else { dobin };
                        desc[((ri * 4 + ci) * 8 + oi) as usize] +=
                            magnitude * weight * w_x * w_y * w_o;
                    }
                }
            }
        }
    }
    // L2 normalization with the clamp-and-renormalize step (Lowe §6.1).
    let norm: f64 = desc.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > f64::EPSILON {
        for v in &mut desc {
            *v /= norm;
        }
    }
    for v in &mut desc {
        *v = v.min(0.2);
    }
    let clipped: f64 = desc.iter().map(|v| v * v).sum::<f64>().sqrt();
    if clipped > f64::EPSILON {
        for v in &mut desc {
            *v /= clipped;
        }
    }
    desc.iter().map(|&v| v as f32).collect()
}

/// Convenience wrapper producing a [`FeatureSet`] directly.
pub fn extract_sift_features(
    image: &GrayImage<'_>,
    config: &SiftConfig,
) -> Result<FeatureSet, SiftError> {
    let (keypoints, descriptors) = extract_sift(image, config)?;
    FeatureSet::new(
        keypoints.iter().map(|k| Point2::new(k.x, k.y)).collect(),
        descriptors,
    )
    .map_err(|_| SiftError::ImageTooSmall {
        width: image.width,
        height: image.height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dark image with one bright Gaussian blob at `center`.
    fn blob_image(width: usize, height: usize, center: (f64, f64), sigma_px: f64) -> Vec<f32> {
        let mut pixels = vec![0.05f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - center.0;
                let dy = y as f64 - center.1;
                let g = (-((dx * dx + dy * dy) / (2.0 * sigma_px * sigma_px))).exp();
                pixels[y * width + x] = (0.05 + 0.85 * g) as f32;
            }
        }
        pixels
    }

    #[test]
    fn sift_finds_the_blob_center_and_128d_descriptors() {
        let (w, h) = (64usize, 64usize);
        let pixels = blob_image(w, h, (32.0, 30.0), 4.0);
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let config = SiftConfig {
            octaves: 2,
            ..SiftConfig::default()
        };
        let (keypoints, descriptors) = extract_sift(&image, &config).unwrap();
        assert!(
            !keypoints.is_empty(),
            "a strong isolated blob must produce keypoints"
        );
        for d in &descriptors {
            assert_eq!(d.len(), 128, "descriptor dimension");
        }
        // Some keypoint should sit within a few pixels of the blob centre.
        let nearest = keypoints
            .iter()
            .map(|k| ((k.x - 32.0).powi(2) + (k.y - 30.0).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest < 6.0,
            "nearest keypoint to the blob centre is {nearest:.2} px away"
        );
    }

    #[test]
    fn sift_is_deterministic() {
        let (w, h) = (48usize, 48usize);
        let pixels = blob_image(w, h, (20.0, 24.0), 3.5);
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let config = SiftConfig {
            octaves: 1,
            intervals: 2,
            ..SiftConfig::default()
        };
        let (kp_a, d_a) = extract_sift(&image, &config).unwrap();
        let (kp_b, d_b) = extract_sift(&image, &config).unwrap();
        assert_eq!(kp_a, kp_b);
        assert_eq!(d_a, d_b);
    }

    #[test]
    fn sift_rejects_tiny_images() {
        let pixels = vec![0.5f32; 8 * 8];
        assert!(GrayImage::new(8, 8, &pixels).is_err());
    }

    #[test]
    fn feature_set_wrapper_matches_keypoints() {
        let (w, h) = (64usize, 64usize);
        let mut pixels = blob_image(w, h, (30.0, 32.0), 4.0);
        // Second blob so two independent detections exist.
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - 50.0;
                let dy = y as f64 - 14.0;
                let g = (-((dx * dx + dy * dy) / 18.0)).exp();
                pixels[y * w + x] += (0.7 * g) as f32;
            }
        }
        let image = GrayImage::new(w, h, &pixels).unwrap();
        let config = SiftConfig {
            octaves: 2,
            ..SiftConfig::default()
        };
        let features = extract_sift_features(&image, &config).unwrap();
        assert!(!features.keypoints.is_empty());
        assert_eq!(features.keypoints.len(), features.descriptors.len());
    }
}
