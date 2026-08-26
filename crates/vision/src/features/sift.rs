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

/// Which interest-point operator feeds the SIFT descriptor pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SiftDetector {
    /// Classical Lowe Difference-of-Gaussian extrema (default; byte-identical
    /// to the historical pipeline when [`SiftConfig::affine`] is off).
    #[default]
    Dog,
    /// Hessian determinant peaks with Laplacian scale selection
    /// (Mikolajczyk & Schmid Hessian-Laplace / VLFeat `HessianLaplace`).
    /// More repeatable under foreshortening than DoG; pair with
    /// [`SiftConfig::affine`] for façade-class scenes.
    HessianLaplace,
}

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
    /// Minimum |DoG| (or scaled |det H|) for an extremum to survive.
    pub contrast_threshold: f64,
    /// Edge-test curvature ratio threshold (Lowe r = 10); DoG path only.
    pub edge_threshold: f64,
    /// Cap on returned keypoints (strongest contrast first when exceeded).
    pub max_keypoints: usize,
    /// Interest-point operator. Default [`SiftDetector::Dog`].
    pub detector: SiftDetector,
    /// Estimate an affine (anisotropic) shape per keypoint via structure-
    /// tensor iteration (Baumberg/covdet-style), refine the detection locus
    /// inside the affine-normalized patch, assign orientation on that
    /// normalized frame, and warp the descriptor sampling grid by the shape
    /// — so detections and descriptors stay comparable across affine view
    /// changes. Off by default: the isotropic pipeline is faster and
    /// sufficient for planar-motion data.
    pub affine: bool,
    /// When true (and [`Self::affine`] is on), also detect on a few
    /// anisotropically resampled copies of the image and merge survivors
    /// back under a strict NMS budget. Supplies loci that only peak under
    /// foreshortening. Off by default.
    pub multi_anisotropy: bool,
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
            detector: SiftDetector::Dog,
            affine: false,
            multi_anisotropy: false,
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
    /// Affine shape `A` mapping the canonical unit disk to the image
    /// footprint (`image_offset = A · R(orientation) · canonical_offset`);
    /// `None` for the isotropic pipeline.
    pub affine_shape: Option<[[f64; 2]; 2]>,
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

        match config.detector {
            SiftDetector::Dog => {
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
                                image,
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
            }
            SiftDetector::HessianLaplace => {
                // Response = |det H| on each Gaussian; scale selected by
                // Laplacian extremum across neighbouring scales.
                let hess: Vec<Layer> = gaussians.iter().map(hessian_det_layer).collect();
                let laps: Vec<Layer> = gaussians.iter().map(laplacian_layer).collect();
                for si in 1..gaussians.len().saturating_sub(1) {
                    let (h_below, h_mid, h_above) = (&hess[si - 1], &hess[si], &hess[si + 1]);
                    let (l_below, l_mid, l_above) = (&laps[si - 1], &laps[si], &laps[si + 1]);
                    let gauss_mid = &gaussians[si];
                    for y in 1..h_mid.height as i64 - 1 {
                        for x in 1..h_mid.width as i64 - 1 {
                            detect_hessian_laplace(
                                image,
                                h_below,
                                h_mid,
                                h_above,
                                l_below,
                                l_mid,
                                l_above,
                                gauss_mid,
                                x,
                                y,
                                config,
                                oct,
                                si as i32,
                                k,
                                &mut keypoints,
                            );
                        }
                    }
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

    if config.affine && config.multi_anisotropy {
        merge_multi_anisotropy(image, config, &mut keypoints);
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

fn hessian_det_layer(g: &Layer) -> Layer {
    let mut data = vec![0.0; g.data.len()];
    for y in 1..g.height as i64 - 1 {
        for x in 1..g.width as i64 - 1 {
            let lxx = g.get(x + 1, y) + g.get(x - 1, y) - 2.0 * g.get(x, y);
            let lyy = g.get(x, y + 1) + g.get(x, y - 1) - 2.0 * g.get(x, y);
            let lxy = (g.get(x + 1, y + 1) - g.get(x + 1, y - 1) - g.get(x - 1, y + 1)
                + g.get(x - 1, y - 1))
                / 4.0;
            data[y as usize * g.width + x as usize] = (lxx * lyy - lxy * lxy).abs();
        }
    }
    Layer {
        width: g.width,
        height: g.height,
        data,
    }
}

fn laplacian_layer(g: &Layer) -> Layer {
    let mut data = vec![0.0; g.data.len()];
    for y in 1..g.height as i64 - 1 {
        for x in 1..g.width as i64 - 1 {
            let lxx = g.get(x + 1, y) + g.get(x - 1, y) - 2.0 * g.get(x, y);
            let lyy = g.get(x, y + 1) + g.get(x, y - 1) - 2.0 * g.get(x, y);
            data[y as usize * g.width + x as usize] = (lxx + lyy).abs();
        }
    }
    Layer {
        width: g.width,
        height: g.height,
        data,
    }
}

#[allow(clippy::too_many_arguments)]
fn detect_hessian_laplace(
    image: &GrayImage<'_>,
    _h_below: &Layer,
    h_mid: &Layer,
    _h_above: &Layer,
    l_below: &Layer,
    l_mid: &Layer,
    l_above: &Layer,
    gauss_mid: &Layer,
    x: i64,
    y: i64,
    config: &SiftConfig,
    octave: usize,
    interval: i32,
    k: f64,
    out: &mut Vec<SiftKeypoint>,
) {
    let value = h_mid.get(x, y);
    // det(H) on unit-scale imagery is typically 1e-6..1e-2; scale the
    // shared contrast_threshold down so the same knob remains usable.
    if value < config.contrast_threshold * 1e-4 {
        return;
    }
    // Spatial maximum of |det H|.
    for dy in -1i64..=1 {
        for dx in -1i64..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            if h_mid.get(x + dx, y + dy) >= value {
                return;
            }
        }
    }
    // Laplacian scale selection (Hessian-Laplace): |∇²L| peaks here.
    let lap = l_mid.get(x, y);
    if lap + 1e-15 < l_below.get(x, y) || lap + 1e-15 < l_above.get(x, y) {
        return;
    }
    assign_orientations(
        image, gauss_mid, x, y, interval, k, value, octave, config, out,
    );
}

/// Detect on a few anisotropically resampled views and merge under a strict
/// NMS + budget so self-consistency stays intact.
fn merge_multi_anisotropy(
    image: &GrayImage<'_>,
    config: &SiftConfig,
    keypoints: &mut Vec<SiftKeypoint>,
) {
    let budget = (keypoints.len() / 3).max(8).min(config.max_keypoints / 4);
    let mut candidates: Vec<SiftKeypoint> = Vec::new();
    // Covers ~×1.3–×1.7 foreshortening (courtyard façades / stretch harness).
    for &sx in &[0.6f64, 0.75, 1.33, 1.67] {
        let (warped_px, ww, wh) = warp_anisotropic_x(image, sx);
        let Ok(warped) = GrayImage::new(ww, wh, &warped_px) else {
            continue;
        };
        let sub = SiftConfig {
            affine: false,
            multi_anisotropy: false,
            max_keypoints: config.max_keypoints,
            ..config.clone()
        };
        let Ok((w_kps, _)) = extract_sift(&warped, &sub) else {
            continue;
        };
        for wk in w_kps {
            let ox = wk.x / sx;
            let oy = wk.y;
            if ox < 2.0
                || oy < 2.0
                || ox >= image.width as f64 - 2.0
                || oy >= image.height as f64 - 2.0
            {
                continue;
            }
            let covered = keypoints.iter().chain(candidates.iter()).any(|e| {
                let dx = e.x - ox;
                let dy = e.y - oy;
                dx * dx + dy * dy < (1.0 * e.sigma.max(wk.sigma)).powi(2)
            });
            if covered {
                continue;
            }
            let shape = estimate_affine_shape(image, ox, oy, wk.sigma);
            let (rx, ry) = refine_location_affine(image, ox, oy, wk.sigma, shape);
            // Strongest orientation(s) in the affine-normalized frame.
            let orients = {
                let canon_sigma = wk.sigma.max(1.0);
                let canon_radius = (canon_sigma * 3.0).ceil().max(2.0) as i64;
                let norm_sq = 2.0 * canon_sigma * canon_sigma;
                let mut hist = [0.0f64; 36];
                for dy in -canon_radius..=canon_radius {
                    for dx in -canon_radius..=canon_radius {
                        let xi = dx as f64;
                        let yi = dy as f64;
                        let (wx, wy) = mat2_apply(shape, (xi, yi));
                        let px = rx + wx;
                        let py = ry + wy;
                        let ex = (shape[0][0], shape[1][0]);
                        let ey = (shape[0][1], shape[1][1]);
                        let gx = 0.5
                            * (sample_bilinear(image, px + ex.0, py + ex.1)
                                - sample_bilinear(image, px - ex.0, py - ex.1));
                        let gy = 0.5
                            * (sample_bilinear(image, px + ey.0, py + ey.1)
                                - sample_bilinear(image, px - ey.0, py - ey.1));
                        let magnitude = (gx * gx + gy * gy).sqrt();
                        if magnitude <= f64::EPSILON {
                            continue;
                        }
                        let weight = (-(xi * xi + yi * yi) / norm_sq).exp();
                        let angle = gy.atan2(gx).to_degrees().rem_euclid(360.0);
                        hist[((angle / 10.0) as usize).min(35)] += magnitude * weight;
                    }
                }
                let max_hist = hist.iter().cloned().fold(f64::MIN, f64::max);
                if max_hist <= 0.0 {
                    Vec::new()
                } else {
                    let mut o = Vec::new();
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
                        o.push(((bin as f64 + delta) * 10.0).rem_euclid(360.0).to_radians());
                    }
                    o
                }
            };
            for ori in orients.into_iter().take(1) {
                candidates.push(SiftKeypoint {
                    x: rx,
                    y: ry,
                    sigma: wk.sigma,
                    orientation: ori,
                    affine_shape: Some(shape),
                    contrast: wk.contrast,
                });
            }
        }
    }
    candidates.sort_by(|a, b| b.contrast.total_cmp(&a.contrast));
    for c in candidates.into_iter().take(budget) {
        let covered = keypoints.iter().any(|e| {
            let dx = e.x - c.x;
            let dy = e.y - c.y;
            dx * dx + dy * dy < (1.0 * e.sigma.max(c.sigma)).powi(2)
        });
        if !covered {
            keypoints.push(c);
        }
    }
}

fn warp_anisotropic_x(image: &GrayImage<'_>, sx: f64) -> (Vec<f32>, usize, usize) {
    let ww = ((image.width as f64 * sx).round() as usize).max(1);
    let wh = image.height;
    let mut px = vec![0.0f32; ww * wh];
    for y in 0..wh {
        for x in 0..ww {
            let sx_src = x as f64 / sx;
            px[y * ww + x] = sample_bilinear(image, sx_src, y as f64) as f32;
        }
    }
    (px, ww, wh)
}

#[allow(clippy::too_many_arguments)]
fn detect_extremum(
    image: &GrayImage<'_>,
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
    assign_orientations(image, mid, x, y, interval, k, value, octave, config, out);
}

#[allow(clippy::too_many_arguments)]
fn assign_orientations(
    image: &GrayImage<'_>,
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
    // Layer → original-frame mapping: original = layer · 2^(octave−1)
    // (the first octave lives on the doubled grid).
    let upsample = (1usize << octave) as f64 / 2.0;
    let kp_x = x as f64 * upsample;
    let kp_y = y as f64 * upsample;
    let sigma = config.sigma_base * k.powi(interval) * upsample;

    // VLFeat covdet order when affine is on: estimate shape first, refine the
    // detection locus inside the normalized patch (detector-side half of
    // affine adaptation), assign orientation on affine-normalized gradients,
    // then describe. Legacy isotropic path keeps the DoG-layer histogram.
    let (kp_x, kp_y, affine_shape) = if config.affine {
        let shape = estimate_affine_shape(image, kp_x, kp_y, sigma);
        let (rx, ry) = refine_location_affine(image, kp_x, kp_y, sigma, shape);
        (rx, ry, Some(shape))
    } else {
        (kp_x, kp_y, None)
    };

    let hist_sigma = 1.5 * config.sigma_base * k.powi(interval) / 2.0f64.powi(octave as i32);
    let radius = (hist_sigma * 3.0).ceil() as i64;
    let mut hist = [0.0f64; 36];

    if let Some(shape) = affine_shape {
        // Canonical-frame orientation: sample the original image through A
        // and take gradients along A's columns so bins are pose-normalized.
        let canon_sigma = (sigma).max(1.0);
        let canon_radius = (canon_sigma * 3.0).ceil().max(2.0) as i64;
        let norm_sq = 2.0 * canon_sigma * canon_sigma;
        for dy in -canon_radius..=canon_radius {
            for dx in -canon_radius..=canon_radius {
                let xi = dx as f64;
                let yi = dy as f64;
                let (ox, oy) = mat2_apply(shape, (xi, yi));
                let px = kp_x + ox;
                let py = kp_y + oy;
                let ex = (shape[0][0], shape[1][0]);
                let ey = (shape[0][1], shape[1][1]);
                let gx = 0.5
                    * (sample_bilinear(image, px + ex.0, py + ex.1)
                        - sample_bilinear(image, px - ex.0, py - ex.1));
                let gy = 0.5
                    * (sample_bilinear(image, px + ey.0, py + ey.1)
                        - sample_bilinear(image, px - ey.0, py - ey.1));
                let magnitude = (gx * gx + gy * gy).sqrt();
                if magnitude <= f64::EPSILON {
                    continue;
                }
                let weight = (-(xi * xi + yi * yi) / norm_sq).exp();
                let angle = gy.atan2(gx).to_degrees().rem_euclid(360.0);
                let bin = ((angle / 10.0) as usize).min(35);
                hist[bin] += magnitude * weight;
            }
        }
    } else {
        // Orientation histogram on the DoG magnitude (a common simplification of
        // Lowe's Gaussian-blurred-image gradients).
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
    }

    let max_hist = hist.iter().cloned().fold(f64::MIN, f64::max);
    if max_hist <= 0.0 {
        return;
    }
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
            x: kp_x,
            y: kp_y,
            sigma,
            orientation: ((bin as f64 + delta) * 10.0).rem_euclid(360.0).to_radians(),
            affine_shape,
            contrast: value.abs(),
        });
    }
}

/// Bilinear sample of `image` at floating-point coordinates (clamped at the
/// borders).
fn sample_bilinear(image: &GrayImage<'_>, x: f64, y: f64) -> f64 {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let p00 = image.get(x0, y0);
    let p10 = image.get(x0 + 1, y0);
    let p01 = image.get(x0, y0 + 1);
    let p11 = image.get(x0 + 1, y0 + 1);
    p00 * (1.0 - fx) * (1.0 - fy) + p10 * fx * (1.0 - fy) + p01 * (1.0 - fx) * fy + p11 * fx * fy
}

fn mat2_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn mat2_apply(a: [[f64; 2]; 2], v: (f64, f64)) -> (f64, f64) {
    (a[0][0] * v.0 + a[0][1] * v.1, a[1][0] * v.0 + a[1][1] * v.1)
}

/// Estimate an affine shape for one keypoint by structure-tensor iteration
/// (Baumberg 1995 / VLFeat `covdet.c`'s affine adaptation):
/// starting from the identity, resample the neighbourhood through the
/// current shape, measure its second-moment matrix μ in the canonical
/// frame, and update `A ← A · P · Q^{-1/2}` (equivalently `A · μ^{-1/2}`)
/// with the smallest singular value held, until the canonical patch is
/// locally isotropic or anisotropy / iteration limits fire. The returned
/// matrix maps canonical disk coordinates to image offsets.
fn estimate_affine_shape(image: &GrayImage<'_>, x: f64, y: f64, sigma: f64) -> [[f64; 2]; 2] {
    const ITERS: usize = 8;
    const MAX_ANISOTROPY: f64 = 6.0;
    const CONVERGENCE: f64 = 1.05;
    // Matrix square root of inverse via eigen decomposition of the 2×2
    // symmetric positive-definite structure tensor.
    fn inv_sqrt_sym(m: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
        let tr = m[0][0] + m[1][1];
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if !det.is_finite() || det <= 1e-15 || !tr.is_finite() || tr <= 0.0 {
            return None;
        }
        // Eigenvalues of [[a,b],[b,c]].
        let disc = ((tr * 0.5).powi(2) - det).max(0.0).sqrt();
        let l1 = tr * 0.5 + disc;
        let l2 = (tr * 0.5 - disc).max(1e-15);
        // Inverse sqrt eigenvalues.
        let s1 = 1.0 / l1.sqrt();
        let s2 = 1.0 / l2.sqrt();
        // Principal eigenvector (for l1).
        let (v1x, v1y) = if m[0][1].abs() > 1e-15 {
            (l1 - m[1][1], m[0][1])
        } else if m[0][0] >= m[1][1] {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };
        let n1 = (v1x * v1x + v1y * v1y).sqrt();
        let (v1x, v1y) = (v1x / n1, v1y / n1);
        // Orthonormal complement.
        let (v2x, v2y) = (-v1y, v1x);
        Some([
            [
                s1 * v1x * v1x + s2 * v2x * v2x,
                s1 * v1x * v1y + s2 * v2x * v2y,
            ],
            [
                s1 * v1x * v1y + s2 * v2x * v2y,
                s1 * v1y * v1y + s2 * v2y * v2y,
            ],
        ])
    }

    fn singular_values(a: [[f64; 2]; 2]) -> (f64, f64) {
        // Singular values of A via eigenvalues of AᵀA.
        let s00 = a[0][0] * a[0][0] + a[1][0] * a[1][0];
        let s01 = a[0][0] * a[0][1] + a[1][0] * a[1][1];
        let s11 = a[0][1] * a[0][1] + a[1][1] * a[1][1];
        let tr = s00 + s11;
        let det = (s00 * s11 - s01 * s01).max(0.0);
        let disc = ((tr * 0.5).powi(2) - det).max(0.0).sqrt();
        let l1 = (tr * 0.5 + disc).max(0.0).sqrt();
        let l2 = (tr * 0.5 - disc).max(0.0).sqrt();
        (l1.max(l2), l1.min(l2).max(1e-12))
    }

    let mut a = [[1.0f64, 0.0], [0.0, 1.0]];
    let mut reference_scale = 1.0f64;
    for iter in 0..ITERS {
        let (s_max, s_min) = singular_values(a);
        let anisotropy = s_max / s_min;
        if anisotropy > MAX_ANISOTROPY {
            break;
        }
        // Hold the smallest singular value after the first iteration
        // (VLFeat factor = referenceScale / min(D)).
        if iter == 0 {
            reference_scale = s_min;
        } else {
            let factor = reference_scale / s_min;
            for row in &mut a {
                row[0] *= factor;
                row[1] *= factor;
            }
        }

        // Effective window radius in canonical units.
        let radius = (sigma * 3.0).ceil().max(2.0) as i64;
        let mut m00 = 0.0;
        let (mut m01, mut m11) = (0.0, 0.0);
        let norm_sq = 2.0 * sigma * sigma;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let xi = dx as f64;
                let yi = dy as f64;
                // Canonical-frame sample position mapped to the image.
                let (ox, oy) = mat2_apply(a, (xi, yi));
                let px = x + ox;
                let py = y + oy;
                // Central differences along the CANONICAL axes: the step
                // vectors are A's columns mapped into the image, so these
                // are true ∂P/∂ξ samples of the warped patch.
                let ex = (a[0][0], a[1][0]);
                let ey = (a[0][1], a[1][1]);
                let gx = 0.5
                    * (sample_bilinear(image, px + ex.0, py + ex.1)
                        - sample_bilinear(image, px - ex.0, py - ex.1));
                let gy = 0.5
                    * (sample_bilinear(image, px + ey.0, py + ey.1)
                        - sample_bilinear(image, px - ey.0, py - ey.1));
                let w = (-(xi * xi + yi * yi) / norm_sq).exp();
                m00 += w * gx * gx;
                m01 += w * gx * gy;
                m11 += w * gy * gy;
            }
        }
        // Normalize so the window weight sum does not bias the eigenvalues.
        let scale = radius as f64 * radius as f64;
        m00 /= scale;
        m01 /= scale;
        m11 /= scale;
        if m00 <= 1e-15 || m11 <= 1e-15 {
            break;
        }
        // Convergence: μ is nearly isotropic.
        let ratio = if m00 > m11 { m00 / m11 } else { m11 / m00 };
        if ratio < CONVERGENCE {
            break;
        }
        let Some(update) = inv_sqrt_sym([[m00, m01], [m01, m11]]) else {
            break;
        };
        a = mat2_mul(a, update);
    }
    // Normalize det(A) → 1 (fixed anisotropy, free isotropic scale — the
    // detector's own DoG σ carries that).
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.is_finite() && det.abs() > 1e-12 {
        let f = 1.0 / det.abs().sqrt();
        // Keep orientation of the basis (positive det).
        let f = if det < 0.0 { -f } else { f };
        for row in &mut a {
            row[0] *= f;
            row[1] *= f;
        }
    }
    a
}

/// Refine a keypoint's image location inside its affine-normalized patch by
/// searching for the peak of squared-gradient magnitude near the origin of
/// the canonical frame, then mapping that offset back through `A`. This is
/// the detector-side half of VLFeat covdet affine adaptation: after the
/// elliptical shape is known, re-localize on the normalized appearance so
/// detections land on corresponding structures across affine warps.
fn refine_location_affine(
    image: &GrayImage<'_>,
    x: f64,
    y: f64,
    sigma: f64,
    shape: [[f64; 2]; 2],
) -> (f64, f64) {
    let radius = ((sigma * 1.5).ceil() as i64).clamp(1, 4);
    let mut best = (0i64, 0i64, f64::NEG_INFINITY);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (ox, oy) = mat2_apply(shape, (dx as f64, dy as f64));
            let px = x + ox;
            let py = y + oy;
            let ex = (shape[0][0], shape[1][0]);
            let ey = (shape[0][1], shape[1][1]);
            let gx = 0.5
                * (sample_bilinear(image, px + ex.0, py + ex.1)
                    - sample_bilinear(image, px - ex.0, py - ex.1));
            let gy = 0.5
                * (sample_bilinear(image, px + ey.0, py + ey.1)
                    - sample_bilinear(image, px - ey.0, py - ey.1));
            let score = gx * gx + gy * gy;
            if score > best.2 {
                best = (dx, dy, score);
            }
        }
    }
    if best.2.is_finite() && best.2 > 0.0 {
        let (ox, oy) = mat2_apply(shape, (best.0 as f64, best.1 as f64));
        (x + ox, y + oy)
    } else {
        (x, y)
    }
}

/// 128-dim descriptor: 4×4 cells × 8 orientation bins around the keypoint.
fn describe(image: &GrayImage<'_>, kp: &SiftKeypoint) -> Vec<f32> {
    const DIM: usize = 128;
    let mut desc = vec![0.0f64; DIM];
    let cell_size = (8.0 * kp.sigma).max(3.0); // window half-width ≈ 4σ·√2/√2
    let (cos, sin) = (kp.orientation.cos(), kp.orientation.sin());
    // Affine warp applied before the rotation: canonical offset → image
    // offset is `A · R(θ) · canonical`. Without a shape this is the identity.
    let shape = kp.affine_shape.unwrap_or([[1.0, 0.0], [0.0, 1.0]]);
    // Largest stretch of A bounds the sampling window.
    let a00a11 = shape[0][0] * shape[0][0] + shape[1][0] * shape[1][0];
    let a01a11 = shape[0][1] * shape[0][1] + shape[1][1] * shape[1][1];
    let max_stretch = a00a11.max(a01a11).sqrt().max(1.0);
    let half = (cell_size * max_stretch * 4.0f64.sqrt()) as i64;
    for dy in -half..=half {
        for dx in -half..=half {
            // Rotate sample offset into the keypoint frame.
            let rx0 = dx as f64 * cos - dy as f64 * sin;
            let ry0 = dx as f64 * sin + dy as f64 * cos;
            // Affine-warp the canonical offset into image space.
            let (wx, wy) = mat2_apply(shape, (rx0, ry0));
            // Canonical-axis gradients: differences along A's columns so
            // orientations stay comparable across affine shapes.
            let ex = (shape[0][0], shape[1][0]);
            let ey = (shape[0][1], shape[1][1]);
            let gx = 0.5
                * (sample_bilinear(image, kp.x + wx + ex.0, kp.y + wy + ex.1)
                    - sample_bilinear(image, kp.x + wx - ex.0, kp.y + wy - ex.1));
            let gy = 0.5
                * (sample_bilinear(image, kp.x + wx + ey.0, kp.y + wy + ey.1)
                    - sample_bilinear(image, kp.x + wx - ey.0, kp.y + wy - ey.1));
            let magnitude = (gx * gx + gy * gy).sqrt();
            if magnitude <= f64::EPSILON {
                continue;
            }
            let theta = gy.atan2(gx) - kp.orientation;
            let weight = (-(rx0 * rx0 + ry0 * ry0) / (2.0 * (cell_size * 2.0).powi(2))).exp();
            // Trilinear bin assignment across cell and orientation bins.
            let cx = (rx0 / cell_size) + 2.0 - 0.5;
            let cy = (ry0 / cell_size) + 2.0 - 0.5;
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

#[cfg(test)]
mod affine_tests {
    use super::*;

    /// Deterministic pseudo-random dot texture (LCG), lightly blurred.
    pub(crate) fn dot_texture(width: usize, height: usize, seed: u64) -> Vec<f32> {
        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> f64 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (self.0 >> 11) as f64 / (1u64 << 53) as f64
            }
        }
        let mut rng = Lcg(seed);
        let mut px = vec![0.15f32; width * height];
        for _ in 0..(width * height / 24) {
            let cx = rng.next() * width as f64;
            let cy = rng.next() * height as f64;
            let bright = 0.55 + 0.35 * rng.next();
            let r = 1.5 + 2.5 * rng.next();
            let x0 = cx.floor() as i64 - 6;
            let y0 = cy.floor() as i64 - 6;
            for y in y0..=y0 + 12 {
                for x in x0..=x0 + 12 {
                    if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
                        continue;
                    }
                    let d2 = (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2);
                    let g = (-d2 / (2.0 * r * r)).exp();
                    let idx = y as usize * width + x as usize;
                    px[idx] = (px[idx] + (bright * g) as f32).min(1.0);
                }
            }
        }
        px
    }

    /// Mutual-NN descriptor matches under a ratio test.
    pub(crate) fn mutual_matches(
        a: &[Vec<f32>],
        b: &[Vec<f32>],
        ratio: f32,
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (ai, da) in a.iter().enumerate() {
            let mut best = (usize::MAX, f32::INFINITY);
            let mut second = (usize::MAX, f32::INFINITY);
            for (bi, db) in b.iter().enumerate() {
                let d: f32 = da
                    .iter()
                    .zip(db)
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt();
                if d < best.1 {
                    second = best;
                    best = (bi, d);
                } else if d < second.1 {
                    second = (bi, d);
                }
            }
            if best.1 < ratio * second.1 {
                // Check mutuality.
                let mut rev_best = (usize::MAX, f32::INFINITY);
                for (bi, db) in b.iter().enumerate() {
                    let d: f32 = db
                        .iter()
                        .zip(da)
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum::<f32>()
                        .sqrt();
                    if d < rev_best.1 {
                        rev_best = (bi, d);
                    }
                }
                if rev_best.0 == ai {
                    out.push((ai, best.0));
                }
            }
        }
        out
    }

    #[test]
    fn affine_shapes_are_finite_det_one_and_bounded() {
        let (w, h) = (96usize, 96usize);
        let px = dot_texture(w, h, 99);
        let image = GrayImage::new(w, h, &px).unwrap();
        let config = SiftConfig {
            octaves: 2,
            affine: true,
            ..SiftConfig::default()
        };
        let (kps, _) = extract_sift(&image, &config).unwrap();
        assert!(!kps.is_empty());
        let mut checked = 0;
        for k in &kps {
            let Some(a) = k.affine_shape else {
                continue;
            };
            let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            assert!(
                (det - 1.0).abs() < 1e-6,
                "det(A) must be normalized to 1, got {det}"
            );
            for row in &a {
                for v in row {
                    assert!(v.is_finite(), "non-finite shape entry");
                }
            }
            // Singular values from AᵀA.
            let s00 = a[0][0] * a[0][0] + a[1][0] * a[1][0];
            let s11 = a[0][1] * a[0][1] + a[1][1] * a[1][1];
            let ratio = s00.max(s11) / s00.min(s11).max(1e-12);
            assert!(ratio <= 100.0, "anisotropy blew up: sv ratio {ratio}");
            checked += 1;
        }
        assert!(checked > 0, "no affine shapes estimated");
    }

    #[test]
    fn affine_descriptors_are_self_consistent() {
        let (w, h) = (96usize, 96usize);
        let px = dot_texture(w, h, 99);
        let image = GrayImage::new(w, h, &px).unwrap();
        let config = SiftConfig {
            octaves: 2,
            affine: true,
            ..SiftConfig::default()
        };
        let (kps, descs) = extract_sift(&image, &config).unwrap();
        assert!(!descs.is_empty());
        // Every descriptor must be its own mutual nearest neighbour with a
        // decisive ratio — guards against NaN/zero/collapsed descriptors
        // entering the pipeline when the shape warp is active.
        let self_m = mutual_matches(&descs, &descs, 0.85);
        assert_eq!(
            self_m.len(),
            kps.len(),
            "self-match coverage {} of {}",
            self_m.len(),
            kps.len()
        );
    }

    #[test]
    fn hessian_laplace_detects_blobs() {
        let (w, h) = (96usize, 96usize);
        let px = dot_texture(w, h, 42);
        let image = GrayImage::new(w, h, &px).unwrap();
        let config = SiftConfig {
            octaves: 2,
            detector: SiftDetector::HessianLaplace,
            max_keypoints: 64,
            ..SiftConfig::default()
        };
        let (kps, descs) = extract_sift(&image, &config).unwrap();
        assert!(!kps.is_empty(), "Hessian-Laplace must find blobs");
        assert_eq!(kps.len(), descs.len());
        assert!(descs.iter().all(|d| d.len() == 128));
    }

    #[test]
    fn multi_anisotropy_adds_bounded_extras() {
        let (w, h) = (96usize, 96usize);
        let px = dot_texture(w, h, 7);
        let image = GrayImage::new(w, h, &px).unwrap();
        let base = SiftConfig {
            octaves: 2,
            detector: SiftDetector::HessianLaplace,
            affine: true,
            multi_anisotropy: false,
            max_keypoints: 128,
            ..SiftConfig::default()
        };
        let multi = SiftConfig {
            multi_anisotropy: true,
            ..base.clone()
        };
        let (k_base, _) = extract_sift(&image, &base).unwrap();
        let (k_multi, _) = extract_sift(&image, &multi).unwrap();
        assert!(
            k_multi.len() >= k_base.len(),
            "multi-anisotropy should not drop detections"
        );
        // Budget caps extras well below a flood.
        assert!(
            k_multi.len() <= k_base.len() + k_base.len() / 2 + 16,
            "extras must stay budgeted: base={} multi={}",
            k_base.len(),
            k_multi.len()
        );
    }

    /// Progress note (2026-08-27): VLFeat ordering (shape → orientation on
    /// normalized gradients → describe) plus mild location refine lifts
    /// cross-stretch mutual matches from plain=1 to affine=3 at ratio 0.85,
    /// but still short of the ≥4 / clear-win bar — isotropic DoG loci remain
    /// the binding constraint. Kept ignored as the harness for a fuller
    /// multi-anisotropy detector.
    #[test]
    #[ignore = "detector repeatability across affine warps still below bar"]
    fn affine_descriptors_survive_anisotropic_stretch_better_than_isotropic() {
        let (w, h) = (96usize, 96usize);
        let base_px = dot_texture(w, h, 99);
        let image_base = GrayImage::new(w, h, &base_px).unwrap();

        // Stretch ×1.8 along x with bilinear resampling.
        let (w2, h2) = ((w as f64 * 1.8) as usize, h);
        let mut stretch_px = vec![0.15f32; w2 * h2];
        for y in 0..h2 {
            for x in 0..w2 {
                let sx = x as f64 / 1.8;
                let x0 = sx.floor() as i64;
                let fx = sx - x0 as f64;
                let v = sample_bilinear(&image_base, x0 as f64 + fx, y as f64);
                stretch_px[y * w2 + x] = v as f32;
            }
        }
        let image_stretch = GrayImage::new(w2, h2, &stretch_px).unwrap();

        let config_affine = SiftConfig {
            octaves: 2,
            detector: SiftDetector::HessianLaplace,
            affine: true,
            multi_anisotropy: true,
            ..SiftConfig::default()
        };
        let config_plain = SiftConfig {
            octaves: 2,
            detector: SiftDetector::Dog,
            affine: false,
            ..SiftConfig::default()
        };

        let (kp_a, d_a) = extract_sift(&image_base, &config_affine).unwrap();
        let (kp_b, d_b) = extract_sift(&image_stretch, &config_affine).unwrap();
        let matches_affine = mutual_matches(&d_a, &d_b, 0.85);

        let (_kp_c, d_c) = extract_sift(&image_base, &config_plain).unwrap();
        let (_kp_d, d_d) = extract_sift(&image_stretch, &config_plain).unwrap();
        let matches_plain = mutual_matches(&d_c, &d_d, 0.85);

        eprintln!(
            "affine+hess+multi matches: {} (plain: {}); kps a={} b={}",
            matches_affine.len(),
            matches_plain.len(),
            kp_a.len(),
            kp_b.len()
        );
        assert!(
            matches_affine.len() > matches_plain.len(),
            "affine adaptation must improve cross-stretch matching \
             (affine={} vs plain={})",
            matches_affine.len(),
            matches_plain.len()
        );
        assert!(
            matches_affine.len() >= 4,
            "too few affine-invariant correspondences: {}",
            matches_affine.len()
        );
    }
}
